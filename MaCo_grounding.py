
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
# from model_MaCo import MaCo
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
        self.tokenizer = tokenizers.Tokenizer.from_file("./MaCo/mimic_wordpiece.json")
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.tokenizer.enable_truncation(max_length=100)
        self.tokenizer.enable_padding(length=100)

    def load_model(self, ckpt, **kwargs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MaCo(img_size=224,
            patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=6,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            SR=0.0, T= 0.07, lam=0.95
            ).cuda()

        ckpt = torch.load(ckpt, map_location=device)
        ckpt = ckpt["model"]
        try:
            del ckpt['CE.weight']
        except:
            a =1
        self.model.load_state_dict(ckpt)
        self.image_inference_engine = self.model.img_mlp
        self.text_inference_engine = self.model.bert_encoder
        
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
        # random.shuffle(tem)

        choice = len(tem)
        # choice = random.randint(1, len(tem))
        
        report = ''
        for i in range(choice):
            if tem[i] != '.' and tem[i] != '. ' and tem[i] != 'None. ' and tem[i] != 'none. ' and tem[i] != '':
                report += tem[i].lower()
        if report == '' and tem != []:
            report = tem[0]

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

            latent_img, _, ids_restore = self.model.forward_img_encoder(imgs.unsqueeze(0), mask_ratio=0)
            latent_img_restore = torch.gather(latent_img[:, 1:, :], dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, latent_img.shape[2]))
            latent_img[:,1:,:] = latent_img_restore
            latent_img = self.image_inference_engine(latent_img)
            
            labels = None
            latent_img_global = latent_img[:, 1:, :].mean(dim=1)
            latent_report = self.text_inference_engine(latent_img_global, ids, labels, attention_mask, type_ids).logits
            x1 = latent_img[0, 1:, :]

            teg = latent_report[0, :].unsqueeze(0).detach()
            x1 = x1 / x1.norm(dim=-1, keepdim=True)
            teg = teg / teg.norm(dim=-1, keepdim=True)
            w = (self.model.pos_weight.weight/0.05).softmax(dim=-1).detach().squeeze(0).unsqueeze(-1)
            x1 = x1 * w
            iel = rearrange(x1, '(h w) f -> h w f', h=14, w=14).detach()

        return iel, teg
    
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
                if "-20.pth" not in ckpt:
                    continue
                print(ckpt)
                pipeline.run(ckpt=ckpt, **kwargs)
    return True

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", "-d", type=str, default="")
    parser.add_argument("--dataset", "-ds", type=str, default="MS_CXR",
                        choices=["MS_CXR", "RSNA", "RSNA_MEDKLIP", "SIIM_ACR",
                                 "COVID_RURAL"])
    parser.add_argument("--redo", "-r", type=bool, default=True)
    parser.add_argument("--save_fig", "-s", type=bool, default=True)
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--opt_th', type=bool, default=False)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    ckpt_dir_list = []

    args.redo = True
    ckpt_dir_list.append("./output_dir/model/trained_model")

    if args.ckpt_dir != "":
        ckpt_dir = args.ckpt_dir
        res = main(**vars(args))
        if res:
            sort_result(ckpt_dir=ckpt_dir, file_name="iou_merge.csv")
    else:
        for ckpt_dir in ckpt_dir_list:
            args.ckpt_dir = ckpt_dir
            res = main(**vars(args))
            if res:
                sort_result(ckpt_dir=ckpt_dir, file_name="iou_merge.csv")