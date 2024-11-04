import pandas as pd
import numpy as np
import pickle
import os
import datetime
import torch
import torchvision.datasets as datasets
from mae_main.util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
import mae_main.util.misc as misc
from oxygen_mae_constants import cat_cols


# the input data is numpy ndarray
def binning_img(dt, bins_num=100, bins_level=None):
    dt_shape = dt.shape
    if bins_level is None:
        dt_bin, bins = pd.qcut(np.ravel(dt),
                               q=bins_num,
                               retbins=True,
                               labels=False,
                               )
    else:
        dt_bin, bins = pd.cut(np.ravel(dt),
                              bins=bins_level,
                              labels=False,
                              retbins=True,
                              )
    idx = np.isnan(dt_bin)
    dt_bin[idx] = len(bins) - 1
    dt_bin = np.reshape(dt_bin, dt_shape).astype(int)
    levels = len(bins) - 1 if idx.sum() == 0 else len(bins)
    return dt_bin, bins, levels


# -----------------------#
# Encode space
# -----------------------#
def encode_space_2d(embed_dim, h, cls_token=True):
    pos = get_2d_sincos_pos_embed(embed_dim, h, cls_token=cls_token)
    pos_embed = torch.from_numpy(pos).float().unsqueeze(0)
    return pos_embed


# -----------------------#
# Encode channel
# -----------------------#
def encode_space_1d(embed_dim, h, cls_token=True):
    chn = get_1d_sincos_pos_embed_from_grid(embed_dim, np.arange(h))
    if cls_token:
        chn = np.concatenate((np.zeros([1, embed_dim]), chn), axis=0)
    chn_embed = torch.from_numpy(chn).float().unsqueeze(0)
    return chn_embed


# -----------------------#
# Encode space and channel
# -----------------------#
def encode_pos2d_chn1d(embed_dim, pos_size, chn_size, cls_token=True):
    pos = encode_space_2d(embed_dim // 2, pos_size, cls_token=False)
    chn = encode_space_1d(embed_dim // 2, chn_size, cls_token=False)
    pos = pos[:, :, None, :].repeat(1, 1, chn_size, 1)
    chn = chn[:, None, :, :].repeat(1, pos_size**2, 1, 1)
    pos_chn = torch.cat((pos, chn), dim=3).reshape((1, -1, embed_dim))
    if cls_token:
        pos_chn = torch.concat((torch.zeros((1, 1, embed_dim)), pos_chn), dim=1)
    return pos_chn


# --------------------------- #
# Encode position
# --------------------------- #
def _encode_spacetime2(pos_idx, dim_num=2, large_num=10000):
    pos = np.arange(pos_idx)[:, None] if type(pos_idx) == int \
        else np.ravel(pos_idx)[:, None]
    ii = np.arange(np.ceil(dim_num / 2)).astype(int)
    deno = np.power(large_num, 2 * ii[None, :] / dim_num)
    rs = np.ones((pos.shape[0], int(np.ceil(dim_num / 2) * 2)))
    rs[:, ii * 2] = np.sin(pos / deno)
    rs[:, ii * 2 + 1] = np.cos(pos / deno)
    return rs[:, :dim_num]


# -----------------------#
# Randomly mask
# xx: [N, L, D]; batch, length, dim
# -----------------------#
def random_masking(xx, mask_ratio):
    # generate random values
    n, l, d = xx.shape
    noise = torch.rand(n, l, device=xx.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # extract masked values
    len_keep = int(l * (1 - mask_ratio))
    ids_keep = ids_shuffle[:, :len_keep].unsqueeze(-1).repeat(1, 1, d)
    x_masked = torch.gather(xx, dim=1, index=ids_keep)

    # document the binary mask: 0 kept, 1 removed
    mask = torch.ones([n, l], device=xx.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


# -----------------------#
# Randomly mask length and channel
# xx: [N, L, C, D]; batch, length, channel, dim
# -----------------------#
def random_masking_lc(self, xx, mask_ratio):
    n, l, c, _ = xx.shape
    xx = xx.reshape((n, l * c, -1))
    x_masked, mask, ids_restore = self.random_masking(xx, mask_ratio)
    return x_masked, mask, ids_restore


# -----------------------#
# Randomly mask length and channel
# xx: [N, L, C, D]; batch, length, channel, dim
# mask_cond: [C]; 0 unmasked, 1 masked
# -----------------------#
def random_masking_lc_cond(xx, mask_ratio, mask_cond):
    n, l, c, d = xx.shape
    xx_masked = xx[:, :, mask_cond, :].reshape((n, -1, d))
    xx_unm = xx[:, :, ~mask_cond, :].reshape((n, -1, d))
    x_masked, mask, ids_restore = random_masking(xx_masked, mask_ratio)
    x_masked = torch.concat((x_masked, xx_unm), dim=1)
    return x_masked, mask, ids_restore


# -----------------------#
# Randomly mask length and channel
# xx: [N, L, C, D]; batch, length, channel, dim
# mask_cond: [C]; 0 unmasked, 1 masked
# -----------------------#
class SatDataFolder(datasets.DatasetFolder):
    def __init__(self, root, loader, extensions=None, is_valid_file=None):
        super().__init__(root, loader,
                         extensions=extensions,
                         is_valid_file=is_valid_file)

    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


# -----------------------#
# Read pickle file from given name
# -----------------------#
# sin and cos days of year
def sin_days(dy):
    dy1 = torch.sin(dy / 365 * 2 * np.pi)
    dy2 = torch.cos(dy / 365 * 2 * np.pi)
    return torch.cat((dy1, dy2), dim=0)


# sin and cos of location from indices
def sin_loc_idx(row_idx, col_idx, row=2160, col=4320):
    itv = 180 / row
    lat_deg = torch.linspace(90 - itv / 2, -90 + itv / 2, row)[row_idx]
    lon_deg = torch.linspace(-180 + itv / 2, 180 - itv / 2, col)[col_idx]
    loc1 = torch.sin(lat_deg * np.pi / 180)
    loc2 = torch.sin(lon_deg * np.pi / 180) * torch.cos(lat_deg * np.pi / 180)
    loc3 = -torch.cos(lon_deg * np.pi / 180) * torch.cos(lat_deg * np.pi / 180)
    loc = torch.cat((loc1[:, None], loc2[:, None], loc3[:, None]), dim=1)
    return loc


# sin and cos of location from mask
def sin_loc_mask(mask):
    row, col = mask.shape
    row_idx, col_idx = torch.where(mask)
    return sin_loc_idx(row_idx, col_idx, row, col)


def sat_loader_pickle(name):
    with open(name, 'rb') as h:
        dt = pickle.load(h)
        return dt, name


def sat_preprocess_per_batch(dt, name):
    img = torch.cat([dt[ii][0] for ii in cat_cols], dim=2)
    img = torch.permute(img, (3, 2, 0, 1))

    # latitude and longitude
    loc = sin_loc_mask(dt['MASK'][0])

    # year and day
    dy = float(name[0].split('/')[-1].split('.')[1])
    dys = sin_days(torch.tensor([dy], dtype=torch.float))

    return img, loc, dys


# -----------------------#
# My dataset
# -----------------------#
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx], self.y[idx]


# ------------------------ #
# Configurate distributed
# ------------------------ #
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): gpu {}: world_size {}'.format(
        args.rank, args.gpu, args.world_size), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend,
                                         world_size=args.world_size,
                                         rank=args.rank,
                                         timeout=datetime.timedelta(seconds=60*60),
                                         )
    torch.distributed.barrier()
    misc.setup_for_distributed(args.rank == 0)
