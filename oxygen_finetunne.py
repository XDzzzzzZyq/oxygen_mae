# This file is for fine tune to MAE model for O2 flux.
#
# Authors: Zuchuan Li
# Date 10/09/2024
#
import pickle
import numpy as np
import time
import os
from sklearn.metrics import r2_score, root_mean_squared_error

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
import mae_main.util.misc as misc
import mae_main.util.lr_decay as lr_decay
import mae_main.util.lr_sched as lr_sched
from dataclasses import dataclass

from functools import partial
import oxygen_pretraining
import utils

import oxygen_mae_constants as o2_con_vars


# --------------------- #
# Fine tune model
# --------------------- #
class O2Finetune(oxygen_pretraining.MaskedAutoencoderViT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # add new components
        reg_dim = self.embed_dim * self.pre_num
        self.head = nn.Linear(reg_dim, 1, bias=True)

        # remove unused components
        del self.decoder_embed
        del self.decoder_pos_embed
        del self.decoder_blocks
        del self.decoder_norm
        del self.decoder_pred

        # initialize new components
        torch.nn.init.trunc_normal_(self.head.weight, std=2e-5)
        torch.nn.init.constant_(self.head.bias, 0)

    def forward(self, dt, mask_ratio=0.0):
        # load to device
        if self.pos_embed.device != dt[0].device:
            self.patch_embed.to(dt[0].device)
            self.pos_embed = self.pos_embed.to(dt[0].device)
            self.cls_token.to(dt[0].device)
            self.blocks.to(dt[0].device)

        # encode data with normalized output
        xx, _, _ = self.forward_encoder(dt, mask_ratio=mask_ratio)

        # regression
        ft = xx[:, :self.pre_num, :]
        ft = torch.reshape(ft, (xx.shape[0], -1))
        yy = self.head(ft)
        return yy


# ------------------------ #
# Configurate model training
# ------------------------ #
def load_pretrain_mod(model, pretrain_checkpoint, loss_scaler=None):
    print('Load pretrained model: {}'.format(pretrain_checkpoint))
    checkpoint = torch.load(pretrain_checkpoint, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    if 'scaler' in checkpoint and loss_scaler is not None:
        loss_scaler.load_state_dict(checkpoint['scaler'])


def create_data_loader(file_name, args, distributed=True):
    with open(file_name, 'rb') as h:
        dt = pickle.load(h)
        dataset = utils.MyDataset(dt['x'], dt['y'], dt['locs'], dt['dys'])

    if distributed:
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            num_replicas=misc.get_world_size(),
            rank=misc.get_rank(),
            shuffle=True,
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_in,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return data_loader


# ------------------------ #
# Configurate model training
# ------------------------ #
def main(args):
    # initiate the distribution
    utils.init_distributed_mode(args)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    if (args.output_dir is not None) and (not os.path.exists(args.output_dir)):
        os.makedirs(args.output_dir, exist_ok=True)

    # build training data
    data_loader_train = create_data_loader(args.data_path + args.train_fn, args)
    print('training data: ')
    print(args.data_path + args.train_fn)

    # build validation data
    data_loader_val = create_data_loader(args.data_path + args.val_fn, args, distributed=False)
    print('validation data: ')
    print(args.data_path + args.val_fn)

    # load pretrain model
    if args.pretrain:
        load_pretrain_mod(args.model, args.pretrain_checkpoint)

    # build ddp model
    model = args.model.to(args.gpu)
    model = DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    n_parameters = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    print("Model = %s" % str(model.module))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    # build optimizer with layer-wise lr decay
    param_groups = lr_decay.param_groups_lrd(
        model=model.module,
        weight_decay=args.weight_decay,
        no_weight_decay_list=['patch_embed', 'pos_embed', 'cls_token'],
        layer_decay=args.layer_decay,
    )

    # optimizer
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = torch.amp.GradScaler()

    # begin training
    start_time = time.time()
    acc = []
    acc_max = -np.inf
    rmse_min = np.inf
    fmf = '{}: train loss {}, val acc: r2={}, rmse={}, r2_max={}, rmse_min={}'
    for epoch in range(0, args.epochs):
        data_loader_train.sampler.set_epoch(epoch)
        loss = simple_train_one_epoch(model, data_loader_train,
                                      optimizer, args, loss_scaler, epoch)

        if args.gpu == 0:
            r2, rmse, _, _ = validate_model(model, data_loader_val, args)
            acc.append([epoch, loss, r2, rmse])
            acc_max = max(acc_max, r2)
            rmse_min = min(rmse_min, rmse)
            print(fmf.format(epoch, loss, r2, rmse, acc_max, rmse_min))

            # save checkpoint
            to_save = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            name = args.output_dir + 'fine_tune_2024-10-16.checkpoint.{}.pth'.format(epoch)
            torch.save(to_save, name)

    print('Training time {}'.format(time.time() - start_time))
    if args.gpu == 0:
        with open(args.output_dir + 'fine_tune_accuracy.pickle', 'wb') as h:
            pickle.dump(acc, h)


# ------------------- #
# Training for one epoch
# ------------------- #
def simple_train_one_epoch(model, data_loader, optimizer, arg, loss_scaler, epoch):
    metric_logger = misc.MetricLogger()
    model.train()
    for ep, (xx, yy, locs, dys) in enumerate(data_loader):
        # adjust learning rate
        lr_sched.adjust_learning_rate(optimizer, ep / len(data_loader) + epoch, arg)
        optimizer.zero_grad()

        # prepare data
        xx = xx.to(arg.gpu, non_blocking=True).to(torch.int)
        yy = yy.to(arg.gpu, non_blocking=True)
        locs = locs.to(arg.gpu, non_blocking=True)
        dys = dys.to(arg.gpu, non_blocking=True)

        # calculate loss
        with torch.autocast(enabled=False, device_type='cuda'):
            y_pred = model([xx, locs, dys], mask_ratio=arg.mask_ratio)
            loss = torch.mean((y_pred[:, 0] - yy)**2)

        # update parameters
        loss_scaler.scale(loss).backward()
        loss_scaler.unscale_(optimizer)
        loss_scaler.step(optimizer)
        loss_scaler.update()
        metric_logger.meters['loss'].update(loss.item(), n=arg.batch_size_in)
    metric_logger.synchronize_between_processes()
    return metric_logger.meters['loss'].global_avg


# ------------------- #
# Validate model
# ------------------- #
def validate_model(model, data_loader, arg):
    # prepare
    model.eval()
    yys = []
    pred = []

    # loop data for prediction
    for xx, yy, locs, dys in data_loader:
        xx = xx.to(arg.gpu, non_blocking=True).to(torch.int)
        locs = locs.to(arg.gpu, non_blocking=True)
        dys = dys.to(arg.gpu, non_blocking=True)
        with torch.autocast(enabled=False, device_type='cuda'):
            with torch.no_grad():
                y_pred = model([xx, locs, dys], mask_ratio=arg.mask_ratio)
        pred.append(y_pred.detach().cpu().numpy())
        yys.append(yy)

    # calculate validation accuracy
    yy_true = np.concatenate(yys, axis=0)
    yy_pred = np.concatenate(pred, axis=0)[:, 0]
    r2 = r2_score(yy_true, yy_pred)
    rmse = root_mean_squared_error(yy_true, yy_pred)
    return r2, rmse, yy_true, yy_pred


@dataclass
class Parameters:
    epochs = 300
    batch_size_in = 128

    # Optimizer parameters
    lr = 1e-4
    min_lr = 1e-7
    weight_decay = 0.01
    layer_decay = 0.75
    warmup_epochs = 5

    # Dataset parameters
    data_path = '/data0/zuchuan/mae_output/'
    output_dir = '/data0/zuchuan/mae_output/l32d128.cls.loc.dy.drop0.2.yr/'
    pretrain_checkpoint = '/data0/zuchuan/mae_output_l32d128/checkpoint-99.pth'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 0

    train_fn = 'o2_finetune_data_train_yr.pickle'
    val_fn = 'o2_finetune_data_val_yr.pickle'
    test_fn = 'o2_finetune_data_test_yr.pickle'

    # distributive
    num_workers = 2
    pin_mem = True

    # model dimension
    enc_depth = 32
    enc_dim = 128
    dec_depth = 6
    dec_dim = 64

    pretrain = True
    mask_ratio = 0.0
    drop_path_rate = 0.2

    # image
    in_chans = 20
    img_size = 3
    patch_size = 1

    def __init__(self, dt_meta):
        # Configurate model
        emb_num = np.sum([dt_meta['LEVEL_NUM'][ii]
                          for ii in o2_con_vars.cat_cols])

        # NOTE: only SST and CHL are masked.
        # They are placed at the beginning columns.
        mask_chn_idx_sst = torch.zeros((self.in_chans,), dtype=torch.bool)
        mask_chn_idx_sst[:3] = True
        mask_chn_idx_chl = torch.zeros((self.in_chans,), dtype=torch.bool)
        mask_chn_idx_chl[3:6] = True
        mask_chn_idx = {'SST': mask_chn_idx_sst, 'CHL': mask_chn_idx_chl}

        self.model = O2Finetune(
            img_size=self.img_size, in_chans=self.in_chans, patch_size=self.patch_size,
            embed_dim=self.enc_dim, depth=self.enc_depth, num_heads=8,
            decoder_embed_dim=self.dec_dim, decoder_depth=self.dec_depth, decoder_num_heads=8,
            mlp_ratio=4, num_embeddings=emb_num,
            mask_chn_idx=mask_chn_idx, mask_chn_name=['SST', 'CHL'], cls=False,
            mean=dt_meta['NAN_MEAN'], std=dt_meta['NAN_STD'],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_path_rate=self.drop_path_rate,
        )


if __name__ == '__main__':
    args = Parameters(o2_con_vars.dt_meta)
    main(args)
    torch.distributed.destroy_process_group()

