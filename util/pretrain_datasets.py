from copy import deepcopy
import os
from typing import List, Tuple
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import tokenizers
from .tokenization_bert import BertTokenizer
import numpy as np
from transformers import AutoTokenizer, AutoModel
import json


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class MultimodalBertDataset(Dataset):
    def __init__(
        self,
        data_root,
        transform,
        SR,
        max_caption_length: int = 100,
        out_path = False
    ):
        self.max_caption_length = max_caption_length
        self.data_root = data_root
        self.transform = transform
        self.images_list, self.report_list = self.read_csv()
        self.out_path = out_path
        
        self.tokenizer = tokenizers.Tokenizer.from_file("mimic_wordpiece.json")
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.tokenizer.enable_truncation(max_length=self.max_caption_length)
        self.tokenizer.enable_padding(length=self.max_caption_length)

        # self.tokenizer = BertTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        
        self.SR = SR

    def __len__(self):
        return len(self.images_list)
    
    def _text_process(self, text):
        tem = text.split('.')
        tem = [i.strip() + '. ' for i in tem]
        random.shuffle(tem)

        # choice = len(tem)
        choice = random.randint(1, len(tem))
        # choice = 1
        report = ''
        for i in range(choice):
            if tem[i] != '.' and tem[i] != '. ' and tem[i] != 'None. ' and tem[i] != 'none. ' and tem[i] != '':
                report += tem[i].lower()
        if report == '' and tem != []:
            report = tem[0]

        return report
    
    def _random_mask(self,tokens):
        masked_tokens = deepcopy(tokens)  # [1, 100]
        for i in range(1, masked_tokens.shape[1]-1):
            if masked_tokens[0][i] == 0:
                break
            
            if masked_tokens[0][i-1] == 3 and self.idxtoword[masked_tokens[0][i].item()][0:2] == '##':
                masked_tokens[0][i] = 3
                continue
            
            if masked_tokens[0][i-1] != 3 and self.idxtoword[masked_tokens[0][i].item()][0:2] == '##':
                continue

            prob = random.random()
            if prob < 0:   # 0.5
                masked_tokens[0][i] = 3

        return masked_tokens

    def __getitem__(self, index):
        if self.SR == 1:
            data_root = '/path/to/MIMIC-448/files/'
            # data_root = '/mnt/disk1/MIMIC-DATA-Final/MIMIC-CXR/MIMIC/files/'
        elif self.SR == 0:
            data_root = '/path/to/MIMIC-224/files/'
        image = pil_loader(data_root + self.images_list[index])
        image = self.transform(image)
        sent = self.report_list[index]
        sent = self._text_process(sent)
        
        ## MaCo
        sent = '[CLS] ' + sent
        encoded = self.tokenizer.encode(sent)
        ids = torch.tensor(encoded.ids).unsqueeze(0)
        attention_mask = torch.tensor(encoded.attention_mask).unsqueeze(0)
        type_ids = torch.tensor(encoded.type_ids).unsqueeze(0)
        masked_ids = self._random_mask(ids)

        
        if self.out_path:
            return image, ids, attention_mask, type_ids, masked_ids, self.images_list[index]
        else:
            return image, ids, attention_mask, type_ids, masked_ids


    def read_csv(self):
        csv_path = os.path.join(self.data_root,'BASE-MIMIC.csv')
        df = pd.read_csv(csv_path, sep=',')
        df.fillna('', inplace=True)
        return df["path"], df["report"]


    def collate_fn(self, instances: List[Tuple]):
        image_list, ids_list, attention_mask_list, type_ids_list, masked_ids_list = [], [], [], [], []
        # flattern
        for b in instances:
            image, ids, attention_mask, type_ids, masked_ids = b
            image_list.append(image)
            ids_list.append(ids)
            attention_mask_list.append(attention_mask)
            type_ids_list.append(type_ids)
            masked_ids_list.append(masked_ids)

        # stack
        image_stack = torch.stack(image_list)
        ids_stack = torch.stack(ids_list).squeeze()
        attention_mask_stack = torch.stack(attention_mask_list).squeeze()
        type_ids_stack = torch.stack(type_ids_list).squeeze()
        masked_ids_stack = torch.stack(masked_ids_list).squeeze()

        # sort and add to dictionary
        return_dict = {
            "image": image_stack,
            "labels": ids_stack,
            "attention_mask": attention_mask_stack,
            "type_ids": type_ids_stack,
            "ids": masked_ids_stack
        }

        return return_dict
    
    
