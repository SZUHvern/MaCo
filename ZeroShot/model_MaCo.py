
from functools import partial
from typing import Any, Dict, Union, List


import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn
from torchvision.transforms.functional import InterpolationMode
from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.layers import Mlp, DropPath

from ZeroShot.utils.pos_embed import get_2d_sincos_pos_embed
# from bert.bert_encoder import BertEncoder
# from bert.custom_bert_encoder import BertEncoder
from ZeroShot.bert.builder import build_bert
import torch.distributed as dist
from einops import rearrange
from ZeroShot.utils.misc import gather, get_rank
from scipy import ndimage



def create_logits(x1, x2):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    # cosine similarity as logits
    logits_per_x1 = (x1 @ x2.t()) 
    logits_per_x2 = (x2 @ x1.t()) 

    return logits_per_x1, logits_per_x2


def similarity(x, y, norm=True):
    if norm:
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
    return x @ y.t()


class Attention(nn.Module):
    """
    Code was copied from timm.models.vision_transformer.Attention
    We add hooks to get the attention map.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self, 
            x, 
            attention_probs_forward_hook=None, 
            attention_probs_backwards_hook=None,
        ):

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # use hooks for the attention weights if necessary
        if attention_probs_forward_hook is not None and attention_probs_backwards_hook is not None:
            attention_probs_forward_hook(attn)
            attn.register_hook(attention_probs_backwards_hook)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """
    Code was copied from timm.models.vision_transformer.Block
    We add hooks to get the attention map.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.attn_hook = False
        self.attn_probs = None
        self.attn_grad = None

    def set_attn_probs(self, attn_probs):
        self.attn_probs = attn_probs

    def set_attn_grad(self, attn_grad):
        self.attn_grad = attn_grad

    def set_attn_hook(self):
        self.attn_hook = True

    def unset_attn_hook(self):
        self.attn_hook = False
        # delete previous saved attentions
        self.attn_probs = None
        self.attn_grad = None

    def attention(self, x):
        # only record the attention when self.attn_hook=True
        if self.attn_hook:
            return self.attn(
                    x, 
                    attention_probs_forward_hook=self.set_attn_probs,
                    attention_probs_backwards_hook=self.set_attn_grad,
                )
        else:
            return self.attn(x)

    def forward(self, x):
        x = x + self.drop_path(self.attention(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MaCo(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(
            self, 
            img_size=224, 
            patch_size=16, 
            in_chans=3,
            embed_dim=1024,
            depth=24, 
            num_heads=16,
            decoder_embed_dim=512, 
            decoder_depth=8, 
            decoder_num_heads=16,
            mlp_ratio=4.,
            norm_layer=nn.LayerNorm,
            use_decoder=False,
            bert_type="MaCo",
            force_download_bert=False,
            normalize_feature=False,
            temperature=0.4,
            num_classes=5,
            multi_class=True,
        ):
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

        self.decoder_blocks = nn.ModuleList([
        Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        for i in range(decoder_depth)])

        # --------------------------------------------------------------------------
        # Bert encoder
        self.bert_type = bert_type
        self.bert_encoder = build_bert(bert_type=bert_type, force_download=force_download_bert)
        # if self.bert_type.lower() == "MaCo": self.bert_mlp = nn.Linear(embed_dim, 768, bias=True)
        
        self.normalize_feature = normalize_feature
        self.temperature = temperature
        
        # --------------------------------------------------------------------------
        # classification head
        self.cls_head = ZeroShotClassificationHead(
            input_dim=768,
            num_classes=num_classes,
            multi_class=multi_class,
        )
        self.img_mlp = nn.Linear(embed_dim, 768, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        # torch.nn.init.normal_(self.mask_token, std=.02)

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

        # p = self.patch_embed.patch_size[0]*2
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        # x = torch.einsum('nchpwq->nhwpqc', x)
        # x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        x = rearrange(x, 'n c h p w q -> n (h w) (p q c)')
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0] * 2
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        # x = torch.einsum('nhwpqc->nchpwq', x)
        # imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        imgs = rearrange(x, 'n c h p w q -> n (h w) (p q c)')
        return imgs

    def patchify_heatmap(self, imgs: torch.Tensor):
        """
        Inputs:
            imgs (torch.Tensor): (N, H, W)
        Returns:
            x (torch.Tensor): (N, L, patch_size**2)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[1] == imgs.shape[2] and imgs.shape[1] % p == 0

        h = w = imgs.shape[1] // p
        x = imgs.reshape(shape=(imgs.shape[0], h, p, w, p))
        x = rearrange(x, "n h p w q -> n (h w) (p q)")
        
        return x

    def set_attn_hook(self):
        for blk in self.blocks:
            blk.set_attn_hook()

    def unset_attn_hook(self):
        for blk in self.blocks:
            blk.unset_attn_hook()

    # TODO: write conditional masking
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

    def condition_masking(self, x, mask_ratio, prob):
        """
        Input:
            x (torch.Tensor): [batch, length, dim]
            mask_ratio (float): masking ratio
            prob (torch.Tensor): [batch, length]
        """        

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        if type(prob) is torch.Tensor or type(prob) is torch.cuda.FloatTensor:
            prob = prob.cpu().numpy()

        # select patches based on the prob one by one
        indexs = np.arange(L)
        ids_keep = np.zeros((N, len_keep), dtype=np.int64) - 1
        for i in range(N):
            ids_keep[i] = np.random.choice(indexs, len_keep, replace=False, p=prob[i])

        # keep unmasked patches
        ids_keep = torch.from_numpy(ids_keep).to(x.device)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1,1,D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.zeros(N, L, device=x.device).scatter_(1, ids_keep, 1)

        return x_masked, mask, None

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
    
    def forward_img_encoder(self, x, mask_ratio, prob=None):
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if prob is None:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            x, mask, ids_restore = self.condition_masking(x, mask_ratio, prob)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_report_decoder(self, caption_ids, attention_mask, token_type_ids):
        # word_embed, sent_embed, sents = self.bert_encoder(caption_ids, attention_mask, token_type_ids)
        word_embed, sent_embed, sents, seg_embed, seg_mask = self.bert_encoder(
            caption_ids, attention_mask, token_type_ids)
        return sent_embed

    def forward(
            self, 
            imgs: torch.Tensor,
        ):
        
        assert self.cls_head is not None, "Please set the classification head first."

        # image encoder
        latent_img = self.forward_img_encoder_nomask(imgs)
        # latent_img = latent_img[:, 1:, :].mean(dim=1)  # use global image features
        # latent_img = self.img_mlp(latent_img)
        
        latent_img = latent_img[:, 1:, :].mean(dim=1)  # use global image features
        latent_img = self.img_mlp(latent_img)
        
        pred, logits_pos, logits_neg = self.cls_head(latent_img)

        return pred, latent_img, logits_pos, logits_neg

    def set_cls_feature(
            self,
            prompts: Dict[str, Dict[str, List[str]]],
    ):
        pos_cls_embed = torch.zeros(len(prompts), 768)
        neg_cls_embed = torch.zeros(len(prompts), 768)
        
        for i, (key, value) in enumerate(prompts.items()):
            pos_embed = []
            for prompt in value["pos"]:
                # get positive cls feature
                ids, attention_mask, type_ids = self.bert_encoder.encode(prompt)
                ids, attention_mask, type_ids = ids.cuda(), attention_mask.cuda(), type_ids.cuda()
                # word_embed, sent_embed, sents  = self.bert_encoder(ids, attention_mask, type_ids)
                sent_embed = self.bert_encoder(ids, attention_mask, type_ids).logits
                pos_embed.append(sent_embed)

            neg_embed = []
            for prompt in value["neg"]:
                # get negative cls feature
                if prompt is not None:
                    ids, attention_mask, type_ids = self.bert_encoder.encode(prompt)
                    ids, attention_mask, type_ids = ids.cuda(), attention_mask.cuda(), type_ids.cuda()
                    # word_embed, sent_embed, sents  = self.bert_encoder(ids, attention_mask, type_ids)
                    sent_embed = self.bert_encoder(ids, attention_mask, type_ids).logits
                    neg_embed.append(sent_embed)

            pos_cls_embed[i] = torch.stack(pos_embed).mean(dim=0)
            neg_cls_embed[i] = torch.stack(neg_embed).mean(dim=0)

        # set cls feature
        self.cls_head.set_cls_feature(pos_cls_embed, neg_cls_embed)

    def get_heatmaps2(
        self, 
        imgs,
        prompt,
        resize: bool = True, 
        temperature: float = 0.2, 
        mode: str = "bilinear",  # [bilinear, area]
        normalize: bool = False,
        scale_value: bool = False,
        gaussian_filter: bool = False,
    ):
        imgs = imgs.cuda()

        # get text embeddings
        ids, attention_mask, type_ids = self.bert_encoder.encode(prompt)
        ids, attention_mask, type_ids = ids.cuda(), attention_mask.cuda(), type_ids.cuda()

        # batch_size, num_patch, num_dim
        latent_img = self.forward_img_encoder_nomask(imgs)
        # # reshuffle to original order
        # latent_img = torch.gather(
        #     latent_img[:, 1:, :], 
        #     dim=1, 
        #     index=ids_restore.unsqueeze(-1).repeat(1, 1, latent_img.shape[2])
        #     )
        # batch_size, num_dim
        latent_report = self.forward_report_decoder(ids, attention_mask, type_ids)

        # TODO: consider optimal transport distance
        # compute similarity
        if normalize:
            latent_img = F.normalize(latent_img, dim=-1)
            latent_report = F.normalize(latent_report, dim=-1)
        sim = torch.bmm(latent_img, latent_report.unsqueeze(-1))  # batch_size, num_path, 1
        sim = F.softmax(sim * temperature, dim=1)
        num_patch = int(np.sqrt(sim.shape[1]))
        sim = sim.reshape(-1, 1, num_patch, num_patch)

        if gaussian_filter:
            sim = torch.tensor(
                ndimage.gaussian_filter(sim.squeeze().cpu().numpy(), sigma=(1.5, 1.5), order=0)
            ).reshape(-1, 1, num_patch, num_patch)

        if scale_value:
            sim = (sim - sim.min()) / (sim.max() - sim.min())

        if resize:
            sim = F.interpolate(sim, size=224, mode=mode)
            # sim = F.interpolate(sim, size=224, mode="area")
        return sim




class ZeroShotClassificationHead(nn.Module):
    """
    Zero-shot classification head.
    """
    def __init__(self, input_dim, num_classes=5, multi_class=True) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.multi_class = multi_class

        self.fc = nn.Linear(input_dim, num_classes, bias=False)
        if self.multi_class:
            self.neg_fc = nn.Linear(input_dim, num_classes, bias=False)

    def set_cls_feature(
            self, 
            cls_feature: torch.Tensor,
            neg_cls_feature: torch.Tensor = None,
        ):

        print("Setting positive classification head weights...")
        assert cls_feature.shape == self.fc.weight.shape
        self.fc.weight.data.copy_(cls_feature)
        self.fc.weight.requires_grad = False
        
        if neg_cls_feature is not None:
            print("Setting negative classification head weights...")
            assert neg_cls_feature.shape == self.neg_fc.weight.shape
            self.neg_fc.weight.data.copy_(neg_cls_feature)
            self.neg_fc.weight.requires_grad = False
        
        print("Weights have been set.")

    def forward(self, x):
        logits = self.fc(x)  # [N, num_classes]
        
        
        if self.multi_class:
            neg_logits = self.neg_fc(x)  # [N, num_classes]
            logits = torch.cat([logits.unsqueeze(-1), neg_logits.unsqueeze(-1)], dim=2) # [N, num_classes, 2]
            logits = F.softmax(logits, dim=2)  # [N, num_classes, 2]
            
            return logits, self.fc.weight.data, self.neg_fc.weight.data
        else:
            logits = F.softmax(logits, dim=1)  # [N, num_classes]
            return logits, self.fc.weight.data, None
        


def maco(**kwargs):
    model = MaCo(
        patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


