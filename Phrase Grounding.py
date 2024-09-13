
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MRM: https://github.com/RL4M/MRM-pytorch
# CheXzero: https://github.com/rajpurkarlab/CheXzero
# --------------------------------------------------------

import os
import sys

import torch
import matplotlib
import argparse
import torch.nn.functional as F
from math import ceil, floor
from pathlib import Path

sys.path.append(os.getcwd())
from eval.common import Pipeline, ImageTextInferenceEngine
from eval.utils import sort_result
FONT_MAX = 50
matplotlib.use('Agg')
import torch.nn.functional as F
import torch.nn as nn
from math import ceil, floor
from pathlib import Path
from functools import partial
from einops import rearrange
import sys
sys.path.append(os.getcwd())
from model_MaCo import MaCo

from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import random
import tokenizers


def trans():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4978], std=[0.2449])
        ])
    
class Engine(ImageTextInferenceEngine):

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = tokenizers.Tokenizer.from_file("/path/to/mimic_wordpiece.json")
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.tokenizer.enable_truncation(max_length=100)
        self.tokenizer.enable_padding(length=100)

    def load_model(self, ckpt, **kwargs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MaCo(img_size=224,
            patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=6,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), T= 0.07, SR=sr).cuda()

        ckpt = torch.load(ckpt, map_location=device)
        ckpt = ckpt["model"]
        try:
            del ckpt['WCE.weight']
        except:
            a=1
        self.model.load_state_dict(ckpt)

        
    def pil_loader(self, path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            img.convert('RGB')
            img = img.resize((224, 224), resample=Image.Resampling.BICUBIC)
            return img
    
    def _text_process(self, text):
        tem = text.split('.')
        tem = [i.strip() + '. ' for i in tem]
        tem_tem = [t.lower() for t in tem if len(t) > 5 and '_' not in t and t[0] != ',']
        tem = tem_tem
        random.shuffle(tem)

        choice = len(tem)
        # choice = random.randint(1, len(tem))
        report = ''
        for i in range(choice):
            report += tem[i]
        return report
    
    def get_emb(self, image_path: Path, query_text: str, device):
        '''
        return  iel: [h, w, feature_size]
                teg: [1, feature_size]
        '''

        with torch.no_grad():
            self.model.eval()

            imgs = self.pil_loader(str(image_path))
            imgs = trans()(imgs)
            
            sent = self._text_process(query_text)
            sent = '[CLS] ' + sent
            encoded = self.tokenizer.encode(sent)
            ids = torch.tensor(encoded.ids).unsqueeze(0)
            attention_mask = torch.tensor(encoded.attention_mask).unsqueeze(0)
            type_ids = torch.tensor(encoded.type_ids).unsqueeze(0)

            imgs = imgs.cuda()
            ids = ids.cuda()
            attention_mask = attention_mask.cuda()
            type_ids = type_ids.cuda()

            latent_img = self.model.forward_img_encoder_nomask(imgs.unsqueeze(0))
            latent_img = latent_img[0, 1:, :]
            latent_img = self.model.img_mlp(latent_img)
            
            labels = None
            latent_report = self.model.bert_encoder(ids, ids, labels, attention_mask, type_ids).logits
            
            tau = 0.02
            w = (self.model.pos_weight_img.weight/tau).softmax(dim=-1).detach().squeeze(0).unsqueeze(-1)
            latent_img = latent_img * w

            latent_img = rearrange(latent_img, '(h w) f -> h w f', h=14, w=14).detach()
            latent_report = latent_report.detach()
            
        return latent_img, latent_report
    
    def get_similarity_map_from_raw_data(
        self, image_path: Path, query_text: str, device, interpolation: str = "nearest",
        ):
        
        iel, teg = self.get_emb(image_path, query_text, device)
        sim = self._get_similarity_map_from_embeddings(iel, teg).view(-1, 1) 
        sim = sim.view(14, 14)
        resized_sim_map = self.convert_similarity_to_image_size(
            sim,
            width=224,
            height=224,
            resize_size=224,
            crop_size=224,
            interpolation=interpolation,
        )

        return resized_sim_map

def main(**kwargs):

    ckpt_dir = os.path.abspath(kwargs["ckpt_dir"])
    if not os.path.exists(ckpt_dir):
        return False
    ckpt_list = sorted([os.path.join(ckpt_dir, i) for i in os.listdir(ckpt_dir) if i.endswith(".pth")])
    engine = Engine()
    for merge in [True, ]:
        for margin in [False, ]:
            pipeline = Pipeline(engine, merge=merge, margin=margin, **kwargs)
            for ckpt in ckpt_list:
                if checkpoint not in ckpt:
                    continue
                print(ckpt)
                result = pipeline.run(ckpt=ckpt, **kwargs)
    return result



    
if __name__ == "__main__":
    global sr
    global checkpoint 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-ds", type=str, default="MS_CXR")
    parser.add_argument("--redo", "-r", type=bool, default=True)
    parser.add_argument("--save_fig", "-s", type=bool, default=True)
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--opt_th', type=bool, default=False)
    parser.add_argument('--ckpt_dir', type=str, default="/path/to/maco.pth")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    result = main(**vars(args))
    iou = result['iou'].values[-1]
    cnr = result['cnr'].values[-1]

    with open('Result-grounding.txt', "a") as file:
        file.write('%s  iou:%.4f  cnr:%.4f' %  (args.ckpt_dir, iou, cnr) + "\n")