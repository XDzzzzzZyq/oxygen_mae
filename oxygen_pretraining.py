# This file follows MAE to customize for oxygen.
#
import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from utils import encode_pos2d_chn1d, random_masking_lc_cond
from oxygen_mae_constants import get_column33344111_min_idx
from oxygen_mae_constants import dt_meta


# -----------------------#
# MAE encoder customized for
# spatiotemporal satellite data.
# -----------------------#
class MaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=3, patch_size=1, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 num_embeddings=180,
                 mask_chn_idx=None, mask_chn_name=None,
                 mean=None, std=None, cls=False,
                 **kwargs
                 ):
        super().__init__()

        # -----------------------#
        # Some parameters
        # -----------------------#
        self.h = int(img_size / patch_size)
        self.num_patches = self.h ** 2
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)
        self.cls = cls

        self.mask_chn_name = mask_chn_name
        self.mask_chn_idx = mask_chn_idx

        # -----------------------#
        # MAE Encoder
        # -----------------------#
        # Embedding
        self.patch_embed = nn.Embedding(num_embeddings=num_embeddings,
                                        embedding_dim=embed_dim)

        # Embedding for location and days of year
        self.patch_embed_loc = nn.Linear(3, embed_dim, bias=True)
        self.patch_embed_day = nn.Linear(2, embed_dim, bias=True)

        # cls_token following BERT
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)

        # encode position + channel
        self.pos_embed = encode_pos2d_chn1d(embed_dim, self.h, in_chans, cls_token=True)

        # Stacked encoder
        drop_path_rate = kwargs.get('drop_path_rate', 0)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio,
                   qkv_bias=True,
                   qk_norm=False,
                   drop_path=dpr[ii],
                   norm_layer=norm_layer,
                   )
             for ii in range(depth)]
        )
        self.norm = norm_layer(embed_dim)

        # -----------------------#
        # MAE decoder
        # -----------------------#
        # decoder embed
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim), requires_grad=True)

        # encode position + channel
        self.decoder_pos_embed = encode_pos2d_chn1d(decoder_embed_dim, self.h, in_chans, cls_token=True)

        # Stacked decoder
        self.decoder_blocks = nn.ModuleList(
            [Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                   qkv_bias=True,
                   qk_norm=False,
                   norm_layer=norm_layer,
                   )
             for _ in range(decoder_depth)]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.decoder_pred = nn.ModuleDict(
            {ii: nn.Linear(decoder_embed_dim,
                           dt_meta['LEVEL_NUM'][ii] if self.cls else self.patch_size,
                           bias=True)
             for ii in self.mask_chn_name
             }
        )

        # loss function
        self.loss_fn = nn.CrossEntropyLoss(reduction='none') if self.cls \
            else nn.MSELoss(reduction='none')

        # auxiliary variables
        # assume the same tiers for all variables
        self.min = self.get_channel_min()
        self.pre_num = 3
        self.missing_idx = dt_meta['LEVEL_NUM']['CHL'] - 1

        # initialize the network
        self.initialize_weights()

    # -----------------------#
    # Initialization
    # -----------------------#
    def initialize_weights(self):
        # initialize nn.Embedding
        torch.nn.init.xavier_uniform_(self.patch_embed.weight)

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    # -----------------------#
    # Initialize linear components
    # -----------------------#
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # -----------------------#
    # Get patches of image
    # img: (N, channel, H, W)
    # output: (N, L, patch_size**2 *channel)
    # -----------------------#
    def patchify(self, img):
        p = self.patch_size
        n = img.shape[0]
        c = self.in_chans
        h = w = self.h
        assert img.shape[2] == img.shape[3] and img.shape[2] % p == 0

        xx = img.reshape(shape=(n, c, h, p, w, p))
        xx = torch.einsum('nchpwq->nhwpqc', xx)
        xx = xx.reshape(shape=(n, h * w, p ** 2 * c))
        return xx

    def get_mask_chn_idx_all(self):
        return torch.cat([ii[:, None] for _, ii in self.mask_chn_idx.items()],
                         dim=1).any(dim=1)

    # -----------------------#
    # Encoder
    # dt: 0->image, 1->location, 2->time
    # xx: [N, Channel, H, W]
    # -----------------------#
    def forward_encoder(self, dt, mask_ratio):
        xx = dt[0]
        loc = dt[1]
        dys = dt[2]

        # embed patches
        xx = self.patchify(xx)
        xx = self.patch_embed(xx)
        loc = self.patch_embed_loc(loc)
        dys = self.patch_embed_day(dys)[None, :].repeat(xx.shape[0], 1)

        # add pos embed w/o cls token
        if self.pos_embed.device != xx.device:
            self.pos_embed = self.pos_embed.to(xx.device)
        pos = self.pos_embed[:, 1:, :].reshape((-1, self.h**2, self.in_chans, self.embed_dim))
        xx = xx + pos

        # mask L and C randomly
        xx, mask, ids_restore = random_masking_lc_cond(xx, mask_ratio, self.get_mask_chn_idx_all())

        # append cls token, loc, time
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = cls_token.expand(xx.shape[0], -1, -1)
        xx = torch.cat((cls_token, loc[:, None, :], dys[:, None, :], xx), dim=1)

        # apply encoder blocks
        for blk in self.blocks:
            xx = blk(xx)
        xx = self.norm(xx)

        return xx, mask, ids_restore

    # -----------------------#
    # Decoder
    # xx: [N, L, D]; batch, length, dim
    # -----------------------#
    def forward_decoder(self, xx, ids_restore):
        # embed tokens
        xx = self.decoder_embed(xx)
        n, l, d = xx.shape

        # append mask tokens back to sequence
        # xx includes cls_token, loc, and time, so plus 3
        um_count = (~self.get_mask_chn_idx_all()).sum() * (self.h**2)
        total = self.in_chans * (self.h**2) + self.pre_num

        mask_tokens = self.mask_token.repeat(n, total - l, 1)
        x_ = torch.cat((xx[:, self.pre_num:(-um_count), :], mask_tokens), dim=1)

        # restore masked subset
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, d))

        # add position
        if self.decoder_pos_embed.device != xx.device:
            self.decoder_pos_embed = self.decoder_pos_embed.to(xx.device)

        pos = self.decoder_pos_embed[:, 1:, :]\
                  .reshape((-1, self.h**2, self.in_chans, self.decoder_embed_dim))
        pos_ = pos[:, :, self.get_mask_chn_idx_all(), :].reshape((pos.shape[0], -1, self.decoder_embed_dim))
        x_ = x_ + pos_

        # add back unmasked subset
        pos_xx = pos[:, :, ~self.get_mask_chn_idx_all(), :].reshape((pos.shape[0], -1, self.decoder_embed_dim))
        x_um = xx[:, (-um_count):, :] + pos_xx
        x_ = torch.cat((x_, x_um), dim=1)

        # add back cls token, loc, and time
        xx = torch.cat((xx[:, :self.pre_num, :], x_), dim=1)

        # add pos to cls_token
        xx[:, :1, :] = xx[:, :1, :] + self.decoder_pos_embed[:, :1, :]

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            xx = blk(xx)
        xx = self.decoder_norm(xx)

        # prediction
        xx = xx[:, self.pre_num:(-um_count), :].reshape((n, self.h**2, -1, self.decoder_embed_dim))
        xx = [self.decoder_pred[col](xx[:, :, self.mask_chn_idx[col], :])
              for col in self.mask_chn_name]
        return torch.cat(xx, dim=2)

    # -----------------------#
    # Normalize data
    # img: [N, L, channel]
    # -----------------------#
    def normalize(self, img):
        self.mean = self.mean.to(img.device)
        self.std = self.std.to(img.device)
        return (img - self.mean[None, None, :]) / (self.std[None, None, :] + 1e-6)

    def get_cls_idx(self, img):
        return img - self.min.to(img.device)[None, None, :]

    def get_missing_mask(self, img):
        return img == self.missing_idx

    def get_channel_min(self):
        col_min = np.array([get_column33344111_min_idx(ii)
                            for ii in range(self.in_chans)])
        idx = torch.tensor(col_min, dtype=torch.int)
        return idx

    # -----------------------#
    # Calculate loss
    # img: [N, channel, H, W]
    # pred: [N, H*W, channel, D]
    # mask: [N, L], 0 keep, 1 removed
    # -----------------------#
    def forward_loss(self, img, pred, mask):
        # patchify and exclude missing
        img_p = self.patchify(img[0])
        img_p_cls_idx = self.get_cls_idx(img_p)
        img_p_miss_mask = self.get_missing_mask(img_p_cls_idx)[:, :, self.get_mask_chn_idx_all()]
        mask = torch.logical_and(mask, ~img_p_miss_mask.reshape(-1, mask.shape[1]))

        # extract masked subset of data
        img_norm = img_p_cls_idx if self.cls else self.normalize(img_p)
        target = img_norm[:, :, self.get_mask_chn_idx_all()]

        # calculate loss
        loss = self.loss_fn(pred.reshape(-1, pred.shape[3]), target.flatten()) if self.cls else \
            self.loss_fn(pred, target.unsqueeze(-1))  # (pred[:, :, :, 0] - target) ** 2

        # reduction ('mean')
        loss = (loss.reshape((-1, mask.shape[1])) * mask).sum() / (mask.sum() + 1e-7)
        return loss

    # -----------------------#
    # img: [N, channel, H, W]
    # -----------------------#
    def forward(self, img, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(img, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(img, pred, mask)
        return loss, pred, mask

