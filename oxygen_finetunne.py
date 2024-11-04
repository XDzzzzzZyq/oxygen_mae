# This file is for fine tune to MAE model for O2 flux.
#
# Authors: Zuchuan Li
# Date 10/09/2024
#
import pickle
import numpy as np
import time
from sklearn.metrics import r2_score, root_mean_squared_error

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
import mae_main.util.misc as misc
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
        norm_layer = kwargs.get('norm_layer', nn.LayerNorm)
        self.fc_norm = norm_layer(self.embed_dim * (self.in_chans * self.h**2 + 1))
        self.reg = nn.Linear(self.embed_dim * (self.in_chans * self.h**2 + 1), 1, bias=True)

        # remove some unused components
        del self.norm
        del self.decoder_embed
        del self.decoder_pos_embed
        del self.decoder_blocks
        del self.decoder_norm
        del self.decoder_pred

        # initialize new component
        torch.nn.init.xavier_uniform_(self.reg.weight)
        torch.nn.init.constant_(self.fc_norm.weight, 1)
        torch.nn.init.constant_(self.reg.bias, 0)
        torch.nn.init.constant_(self.fc_norm.bias, 0)

    def forward(self, xx, mask_ratio=0.0):
        # load to device
        self.patch_embed.to(xx.device)
        self.pos_embed = self.pos_embed.to(xx.device)
        self.cls_token.to(xx.device)
        self.blocks.to(xx.device)

        # embed patches
        xx = self.patchify(xx)
        xx = self.patch_embed(xx)

        # add pos embed w/o cls token
        pos = self.pos_embed[:, 1:, :].reshape((-1, self.h ** 2, self.in_chans, self.embed_dim))
        xx = xx + pos
        xx = xx.reshape((xx.shape[0], -1, xx.shape[3]))

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = cls_token.expand(xx.shape[0], -1, -1)
        xx = torch.cat((cls_token, xx), dim=1)

        # apply encoder blocks
        for blk in self.blocks:
            xx = blk(xx)

        # regression using cls
        # ft = xx[:, 0, :]  # cls_token
        # ft = torch.mean(xx[:, 1:, :], dim=1)  # average
        # ft = torch.reshape(xx[:, (1+4*self.in_chans):(1+5*self.in_chans), :],
        #                   (xx.shape[0], -1))  # pixel located with oxygen
        ft = torch.reshape(xx, (xx.shape[0], -1))
        yy = self.reg(self.fc_norm(ft))
        return yy


# ------------------------ #
# Configurate model training
# ------------------------ #
def load_pretrain_mod(model, pretrain_checkpoint, loss_scaler=None):
    print('Load pretrained model: {}'.format(pretrain_checkpoint))
    checkpoint = torch.load(pretrain_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    if 'scaler' in checkpoint and loss_scaler is not None:
        loss_scaler.load_state_dict(checkpoint['scaler'])


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

    # build training data
    with open(args.data_path + '/o2_finetune_data_train.pickle', 'rb') as h:
        dt = pickle.load(h)
        dataset_train = utils.MyDataset(dt['x'], dt['y'])

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=misc.get_world_size(),
        rank=misc.get_rank(),
        shuffle=True
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size_in,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # build validation data
    with open(args.data_path + '/o2_finetune_data_val.pickle', 'rb') as h:
        dt = pickle.load(h)
        dataset_val = utils.MyDataset(dt['x'], dt['y'])

    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val,
        num_replicas=misc.get_world_size(),
        rank=misc.get_rank(),
        shuffle=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size_in,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # load pretrain model
    loss_scaler = torch.amp.GradScaler()
    load_pretrain_mod(args.model, args.pretrain_checkpoint, loss_scaler)

    # build ddp model
    model = args.model.to(args.gpu)
    model = DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    # optimizer
    optimizer = torch.optim.AdamW(
        [
            {'params': model.module.parameters(), 'lr': args.lr},
        ])

    # begin training
    start_time = time.time()
    acc = []
    for epoch in range(0, args.epochs):
        data_loader_train.sampler.set_epoch(epoch)
        loss = simple_train_one_epoch(model, data_loader_train,
                                      optimizer, args, loss_scaler)

        # data_loader_val.sampler.set_epoch(epoch)
        r2, rmse, _, _ = validate_model(model, data_loader_val, args)
        acc.append([epoch, loss, r2, rmse])
        print('{}: train loss {}, val acc: r2={}, rmse={}'.format(epoch, loss, r2, rmse))

        # save checkpoint
        if args.gpu == 0:
            to_save = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            name = args.output_dir + '/fine_tune_2024-10-16.checkpoint.{}.pth'.format(epoch)
            torch.save(to_save, name)
    print('Training time {}'.format(time.time() - start_time))
    if args.gpu == 0:
        with open(args.output_dir + '/accuracy.pickle', 'wb') as h:
            pickle.dump(acc, h)
    torch.distributed.destroy_process_group()


# ------------------- #
# Training for one epoch
# ------------------- #
def simple_train_one_epoch(model, data_loader, optimizer, arg, loss_scaler):
    model.train()
    loss_avg = []
    for xx, yy in data_loader:
        optimizer.zero_grad()
        xx = xx.to(arg.gpu, non_blocking=True)
        yy = yy.to(arg.gpu, non_blocking=True)
        with torch.autocast(enabled=False, device_type='cuda'):
            y_pred = model(xx)[:, 0]
            loss = torch.mean((y_pred - yy)**2)
        loss_scaler.scale(loss).backward()
        loss_scaler.unscale_(optimizer)
        loss_scaler.step(optimizer)
        loss_scaler.update()
        loss_avg.append(loss.item())
    return np.mean(loss_avg)


# ------------------- #
# Validate model
# ------------------- #
def validate_model(model, data_loader, arg):
    model.eval()
    yys = []
    pred = []
    for xx, yy in data_loader:
        xx = xx.to(arg.gpu, non_blocking=True)
        with torch.no_grad():
            y_pred = model(xx)[:, 0]
        yys.append(yy)
        pred.append(y_pred.detach().cpu().numpy())
    yy_true = np.concatenate(yys, axis=0)
    yy_pred = np.concatenate(pred, axis=0)
    r2 = r2_score(yy_true, yy_pred)
    rmse = root_mean_squared_error(yy_true, yy_pred)
    return r2, rmse, yy_true, yy_pred


@dataclass
class Parameters:
    epochs = 300
    batch_size_in = 512

    # Optimizer parameters
    lr = 1e-5

    # Dataset parameters
    data_path = '/data0/zuchuan/mae_output'
    output_dir = '/data0/zuchuan/mae_output/finetune'
    pretrain_checkpoint = '/data0/zuchuan/mae_output/checkpoint-3.pth'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 0

    # distributive
    num_workers = 6
    pin_mem = True

    # model dimension
    enc_depth = 16
    enc_dim = 128
    dec_depth = 4
    dec_dim = 64
    drop_path_rate = 0.1

    # image
    in_chans = 20
    img_size = 3
    patch_size = 1

    def __init__(self, dt_meta):
        # Configurate model
        emb_num = np.sum([dt_meta['LEVEL_NUM'][ii]
                          for ii in o2_con_vars.cat_cols])

        self.model = O2Finetune(
            img_size=self.img_size, in_chans=self.in_chans, patch_size=self.patch_size,
            embed_dim=self.enc_dim, depth=self.enc_depth, num_heads=8,
            decoder_embed_dim=self.dec_dim, decoder_depth=self.dec_depth, decoder_num_heads=8,
            mlp_ratio=4, num_embeddings=emb_num,
            mean=dt_meta['MEAN'], std=dt_meta['STD'],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_path_rate=self.drop_path_rate,
        )


if __name__ == '__main__':
    args = Parameters(o2_con_vars.dt_meta)
    main(args)

