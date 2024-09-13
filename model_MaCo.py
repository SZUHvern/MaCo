# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MRM: https://github.com/RL4M/MRM-pytorch
# CheXzero: https://github.com/rajpurkarlab/CheXzero
# --------------------------------------------------------

from functools import partial
import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision.transforms.functional import InterpolationMode
from timm.models.vision_transformer import PatchEmbed, Block
from transformers import AutoModel

from util.pos_embed import get_2d_sincos_pos_embed
from bert.bert_encoder import BertEncoder
import torch.distributed as dist
from einops import rearrange
import torch.distributed.nn.functional as distnn
from transformers import BertTokenizer, BertModel

def gather(embeddings, gradient=True):
    world_size = dist.get_world_size()
    embeddings = embeddings.contiguous()
    embeddings_list = [torch.zeros_like(embeddings) for _ in range(world_size)]
    # distnn.all_gather(embeddings_list, embeddings)
    dist.all_gather(embeddings_list, embeddings)
    if gradient == True:
        embeddings_list[dist.get_rank()] = embeddings
    embeddings = torch.cat(embeddings_list)
    return embeddings

def create_logits(x1, x2, logit_scale=1):
    # logit_scale=1
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()
    return logits_per_x1, logits_per_x2

class GatherLayer(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if dist.is_available() and dist.is_initialized():
            all_gradients = torch.stack(grads)
            dist.all_reduce(all_gradients)
            grad_out = all_gradients[dist.get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather_gradient(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)

class MaCo(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=True, lam=0.9, T=0.07, SR=1.0, warmE=0):
        super().__init__()

        # --------------------------------------------------------------------------
        # image encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # image decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.pool = nn.MaxPool1d(169, stride=1)
        if SR==1.0:
            self.decoder_pred = nn.Linear(decoder_embed_dim, (patch_size * 2)**2 * in_chans, bias=True)
        elif SR == 0.0:
            self.decoder_pred = nn.Linear(decoder_embed_dim, (patch_size)**2 * in_chans, bias=True)
        # --------------------------------------------------------------------------
        # Bert encoder
        self.bert_encoder = BertEncoder()
        self.img_mlp = nn.Linear(embed_dim, 768, bias=True)
        self.pos_weight_img = nn.Linear(num_patches, 1)
        self.pos_weight_text = nn.Linear(768, 1)
        self.pos_cat = nn.Linear(2, 1)

        self.norm_pix_loss = norm_pix_loss
        self.T = 1 / T
        self.Tpar = nn.Parameter(torch.ones([]) * self.T)
        self.lam = lam
        self.SR = SR
        self.warm = warmE
        
        self.initialize_weights()
        self.init_pos_weight()

    def init_pos_weight(self):
        init_weight1 = torch.linspace(0.5,1,7)
        init_weight2 = torch.linspace(1,0.5,7)
        init_weight = torch.cat([init_weight1, init_weight2], dim=-1).unsqueeze(-1)
        init_weight = init_weight @ init_weight.t()
        init_weight = init_weight.view(1, -1).softmax(dim=-1)
        
        init_weight = init_weight * 0 ## only for softplus
        self.pos_weight_img.weight = nn.Parameter(init_weight)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
       

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        if self.SR == 1.0:
            p = self.patch_embed.patch_size[0]*2
        elif self.SR == 0:
            p = self.patch_embed.patch_size[0]
        
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        
        x = rearrange(x, 'n c h p w q -> n (h w) (p q c)')
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        # p = self.patch_embed.patch_size[0] * 2
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_img_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore
    
    def forward_img_encoder_nomask(self, x):
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x
    
    def finetune(self, x, global_pool):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        outcome = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        return outcome

    def forward_img_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
    def forward_img_decoder_only(self, x, ids_restore):


        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_report_decoder(self, latent, caption_ids, labels, attention_mask, token_type_ids):
        latent = self.bert_mlp(latent)
        latent = latent[:, 1:, :].mean(dim=1)
        outputs = self.bert_encoder(latent, caption_ids, labels, attention_mask, token_type_ids)
        return outputs.loss

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, batch, mask_ratio=0.50, epoch=0):

        big_imgs, ids, labels, attention_mask, type_ids = batch["image"], batch["ids"], batch["labels"], batch["attention_mask"], batch["type_ids"]
        big_imgs = big_imgs.cuda()
        ids = ids.cuda()
        labels = labels.cuda()
        attention_mask = attention_mask.cuda()
        type_ids = type_ids.cuda()
        
        if self.SR == 1.0:
            imgs = torchvision.transforms.Resize([224, 224], interpolation=InterpolationMode.BICUBIC)(big_imgs)
        elif self.SR == 0:
            imgs = big_imgs

        latent, mask, ids_restore = self.forward_img_encoder(imgs, mask_ratio)

        ones_sample = torch.ones_like(latent, requires_grad=False)
        mask_token_fixed = torch.zeros_like(self.mask_token, requires_grad=False)
        mask_token_fixed = mask_token_fixed.repeat(ones_sample.shape[0], ids_restore.shape[1] + 1 - ones_sample.shape[1], 1)
        x_pos = torch.cat([ones_sample[:, 1:, :], mask_token_fixed], dim=1)  # no cls token
        x_pos = torch.gather(x_pos, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, ones_sample.shape[2])).mean(dim=-1) # unshuffle        

        pred = self.forward_img_decoder(latent, ids_restore)  # [N, L, p*p*3]
        target = self.patchify(big_imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss_img = (pred - target) ** 2
        loss_img = loss_img.mean(dim=-1)  # [N, L], mean loss per patch
        loss_img = (loss_img * mask).sum() / mask.sum()  # mean loss on removed patches 
        
        # MaCo
        latent_img = self.img_mlp(latent)
        latent_img = latent_img[:, 1:, :]
        latent_img_global = latent_img.mean(dim=1) 
        labels = None
        latent_report = self.bert_encoder(latent_img_global, ids, labels, attention_mask, type_ids)
        logits_report = latent_report.logits
        x_ = self.pos_weight_img(x_pos).type(torch.float32)

        latent_img_global = gather_gradient(latent_img_global)
        logits_report = gather_gradient(logits_report)
        x_ = gather_gradient(x_)

        logits1, logits2 = create_logits(latent_img_global, logits_report, self.Tpar)
        gt = torch.arange(logits1.shape[0], dtype=torch.long).cuda()

        self.CE = torch.nn.CrossEntropyLoss()
        scale = torch.log(1 + torch.exp(x_))
        logits1_scale = logits1 * scale
        logits2_scale = logits2 * scale
        loss_c = self.CE(logits1_scale, gt)
        loss_c += self.CE(logits2_scale, gt)
        self.WCE = torch.nn.CrossEntropyLoss(weight=scale.detach().squeeze(-1))
        loss_c += self.WCE(logits1, gt)
        loss_c += self.WCE(logits2, gt)
        loss_c /= 4

        return (loss_img, loss_c), pred, mask

def maco(**kwargs):
    model = MaCo(
        patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
