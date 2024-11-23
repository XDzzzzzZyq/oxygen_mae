# This file is used to run the MAE model
# for the oxygen pretraining.
# It is based on MAE code.
#
# Authors: Zuchuan Li
# Date 10/09/2024
#
import numpy as np
import time
import pickle

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
import timm.optim.optim_factory as optim_factory
import mae_main.util.misc as misc
from mae_main.util import lr_sched
from dataclasses import dataclass

from functools import partial
import oxygen_pretraining
import utils
from utils import SatDataFolder, sat_loader_pickle, init_distributed_mode
import oxygen_mae_constants as o2_con_vars


# ------------------------ #
# Configurate model training
# ------------------------ #
def main(args):
    # Initiate the distribution
    init_distributed_mode(args)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # training data
    dataset_train = SatDataFolder(args.data_path, sat_loader_pickle, extensions=('.pickle',))

    world_size = misc.get_world_size()
    rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # validation data
    dataset_val = SatDataFolder(args.data_path_val, sat_loader_pickle, extensions=('.pickle',))
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # define ddp model
    model = DistributedDataParallel(args.model.to(args.gpu), device_ids=[args.gpu], find_unused_parameters=True)

    # only regularize the slope
    param_groups = optim_factory.param_groups_weight_decay(model.module, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    # loss scaler for mixed precision
    loss_scaler = torch.amp.GradScaler()

    # resume from previous training
    if args.resume:
        print('resume previous training!!!')
        misc.load_model(args=args, model_without_ddp=model.module, optimizer=optimizer, loss_scaler=loss_scaler)

    start_time = time.time()
    accs = []
    for epoch in range(args.start_epoch, args.epochs):
        data_loader_train.sampler.set_epoch(epoch)
        loss = simple_train_one_epoch(model, data_loader_train,
                                      optimizer, args, loss_scaler, epoch)
        misc.save_model(args=args, model=model, model_without_ddp=model.module, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch)
        loss_val = validate_model(model, data_loader_val, args)

        print('Epoch {}: tr loss {}, val loss {}'.format(epoch + 1, loss, loss_val))
        accs.append([epoch + 1, loss, loss_val])
        if misc.get_rank() == 0:
            with open(args.output_dir + "/accuracy{}.log".format(epoch), 'wb') as h:
                pickle.dump(accs, h)
    print('Training time {}'.format(time.time() - start_time))


# ------------------- #
# Training for one epoch
# ------------------- #
def simple_train_one_epoch(model, data_loader, optimizer, args, loss_scaler, epoch):
    metric_logger = misc.MetricLogger()
    model.train()
    for ep, dt in enumerate(data_loader):
        # adjust learning rate
        lr_sched.adjust_learning_rate(optimizer, ep / len(data_loader) + epoch, args)

        # get data
        dt = utils.sat_preprocess_per_batch(dt[0][0], dt[0][1])
        img = dt[0]
        loc = dt[1].to(torch.float)
        dys = dt[2].to(args.gpu, non_blocking=True).to(torch.float)
        n = img.shape[0]
        idx = torch.randperm(n)
        img = img[idx, :, :, :]
        loc = loc[idx, :]

        # loop the data
        for ii in range(0, args.batch_per_epoch * args.batch_size_in, args.batch_size_in):
            img_t = img[ii: (ii+args.batch_size_in), :, :, :].to(args.gpu, non_blocking=True)
            loc_t = loc[ii: (ii+args.batch_size_in), :].to(args.gpu, non_blocking=True)
            with torch.autocast(enabled=False, device_type='cuda'):
                loss, _, _ = model((img_t, loc_t, dys), args.mask_ratio)
            loss_scaler.scale(loss).backward()
            loss_scaler.unscale_(optimizer)
            loss_scaler.step(optimizer)
            loss_scaler.update()
            metric_logger.meters['loss'].update(loss.item(), n=args.batch_size_in)
        if ep % 20 == 0:
            print('Epoch {}: '.format(ep + 1), metric_logger.meters['loss'].avg)
    metric_logger.synchronize_between_processes()
    return metric_logger.meters['loss'].global_avg


# ------------------- #
# Validate model
# ------------------- #
def validate_model(model, data_loader, arg):
    metric_logger = misc.MetricLogger()
    model.eval()
    for ep, dt in enumerate(data_loader):
        dt = utils.sat_preprocess_per_batch(dt[0][0], dt[0][1])
        img = dt[0]
        loc = dt[1].to(torch.float)
        dys = dt[2].to(arg.gpu, non_blocking=True).to(torch.float)
        n = img.shape[0]
        idx = torch.randperm(n)
        img = img[idx, :, :, :]
        loc = loc[idx, :]
        for ii in range(0, args.batch_per_epoch * args.batch_size_in, args.batch_size_in):
            img_t = img[ii: (ii + args.batch_size_in), :, :, :].to(arg.gpu, non_blocking=True)
            loc_t = loc[ii: (ii + args.batch_size_in), :].to(arg.gpu, non_blocking=True)
            with torch.no_grad():
                with torch.autocast(enabled=False, device_type='cuda'):
                    loss, _, _ = model((img_t, loc_t, dys), arg.mask_ratio)
            metric_logger.meters['loss'].update(loss.item(), n=args.batch_size_in)
        if ep % 20 == 0:
            print('Epoch {}: '.format(ep + 1), metric_logger.meters['loss'].avg)
    metric_logger.synchronize_between_processes()
    return metric_logger.meters['loss'].global_avg


@dataclass
class Parameters:
    batch_size = 1
    epochs = 100
    batch_size_in = 512
    batch_per_epoch = 100

    # Model parameters
    mask_ratio = 0.75

    # Optimizer parameters
    lr = 1e-4
    min_lr = 0
    weight_decay = 0.01
    warmup_epochs = 10

    # Dataset parameters
    data_path = '/data0/zuchuan/processed_data/'
    data_path_val = '/data0/zuchuan/processed_data_val/'
    output_dir = '/data0/zuchuan/mae_output_l32d128'
    log_dir = '/data0/zuchuan/mae_output_l32d128'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 0

    # resume previous training
    resume = "/data0/zuchuan/mae_output_l32d128/checkpoint-50.pth"
    start_epoch = 50

    # distributive
    num_workers = 2
    pin_mem = True

    # model dimension
    enc_depth = 32
    enc_dim = 128
    dec_depth = 6
    dec_dim = 64

    def __init__(self, dt_meta):
        # Configurate model
        emb_num = np.sum([dt_meta['LEVEL_NUM'][ii]
                          for ii in o2_con_vars.cat_cols])

        # NOTE: only SST and CHL are masked.
        # They are placed at the beginning columns
        in_chans = 20
        mask_chn_idx_sst = torch.zeros((in_chans,), dtype=torch.bool)
        mask_chn_idx_sst[:3] = True
        mask_chn_idx_chl = torch.zeros((in_chans,), dtype=torch.bool)
        mask_chn_idx_chl[3:6] = True
        mask_chn_idx = {'SST': mask_chn_idx_sst, 'CHL': mask_chn_idx_chl}

        self.model = oxygen_pretraining.MaskedAutoencoderViT(
            img_size=3, in_chans=in_chans, patch_size=1,
            embed_dim=self.enc_dim, depth=self.enc_depth, num_heads=8,
            decoder_embed_dim=self.dec_dim, decoder_depth=self.dec_depth, decoder_num_heads=8,
            mlp_ratio=4, num_embeddings=emb_num,
            mask_chn_idx=mask_chn_idx, mask_chn_name=['SST', 'CHL'], cls=False,
            mean=dt_meta['NAN_MEAN'], std=dt_meta['NAN_STD'],
            norm_layer=partial(nn.LayerNorm, eps=1e-6))


if __name__ == '__main__':
    args = Parameters(o2_con_vars.dt_meta)
    main(args)
    torch.distributed.destroy_process_group()

