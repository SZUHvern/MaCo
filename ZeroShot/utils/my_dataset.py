import os
import cv2
import json
import random
import numpy as np 
import pandas as pd 
import pydicom as dicom

import torch
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
from pycocotools.coco import COCO
from os.path import join
from typing import List, Tuple, Optional


class Base(data.Dataset):

    categories = np.array([
        "cardiomegaly",
        "edema",
        "consolidation",
        "atelectasis",
        "pleural effusion"
    ])
    
    def __init__(self, root, root_split, data_volume, split="train", transform=None):
        super(Base, self)
        if data_volume == '1':
            train_label_data = "train_1.txt"
        if data_volume == '10':
            train_label_data = "train_10.txt"
        if data_volume == '100':
            train_label_data = "train_list.txt"
        test_label_data = "test_list.txt"
        val_label_data = "val_list.txt"
        
        self.split = split
        self.root = root
        self.transform = transform
        self.listImagePaths = []
        self.listImageLabels = []
        
        if self.split == "train":
            downloaded_data_label_txt = train_label_data
        
        elif self.split == "val":
            downloaded_data_label_txt = val_label_data
                 
        elif self.split == "test":
            downloaded_data_label_txt = test_label_data
           
        #---- Open file, get image paths and labels
        fileDescriptor = open(os.path.join(root_split, downloaded_data_label_txt), "r")
        
        #---- get into the loop
        line = True
        
        # root_tmp = os.path.join(root,self.split)
        while line:
                
            line = fileDescriptor.readline()
            
            #--- if not empty
            if line:
          
                lineItems = line.split()
                imagePath = os.path.join(root, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)   
            
        fileDescriptor.close()

    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.listImageLabels[index])
        categorie = self.categories[imageLabel == 1].tolist()
        # try:
        #     assert imageLabel.sum() > 0, "imageLabel.sum() > 0"
        # except:
        #     print(imagePath)
        
        if self.transform != None: imageData = self.transform(imageData)
        
        # return imageData, imageLabel
        return dict(image=imageData, label=imageLabel, categories=categorie)

    def __len__(self):
        return len(self.listImagePaths)

    def collate_fn(self, instances: List[Tuple]):
        image_list, categories_list, label_list = [], [], []

        for b in instances:
            image_list.append(b["image"])
            categories_list.append(b["categories"])
            label_list.append(torch.tensor(b["label"]))

        image_stack = torch.stack(image_list)
        label_stack = torch.stack(label_list)

        return dict(
            image=image_stack,
            categories=categories_list,
            label=label_stack,
        )


class NIHChestXray(Base):
    categories = np.array([
        "atelectasis",
        "cardiomegaly",
        "effusion",
        "infiltration",
        "mass",
        "nodule",
        "pneumonia",
        "pneumothorax",
        "consolidation",
        "edema",
        "emphysema",
        "fibrosis",
        "pleural thickening",
        "hernia",
    ])
    
    def __init__(self, root, root_split, data_volume, split="train", transform=None):
        super(Base, self)
        if data_volume == '1':
            train_label_data = "train_1.txt"
        if data_volume == '10':
            train_label_data = "train_10.txt"
        if data_volume == '100':
            train_label_data = "train_list.txt"
        test_label_data = "test_list.txt"
        val_label_data = "val_list.txt"
        
        self.split = split
        self.root = root
        self.transform = transform
        self.listImagePaths = []
        self.listImageLabels = []
        
        if self.split == "train":
            downloaded_data_label_txt = train_label_data
        
        elif self.split == "val":
            downloaded_data_label_txt = val_label_data
                 
        elif self.split == "test":
            downloaded_data_label_txt = test_label_data
           
        #---- Open file, get image paths and labels
        fileDescriptor = open(os.path.join(root_split, downloaded_data_label_txt), "r")
        
        #---- get into the loop
        line = True
        
        # root_tmp = os.path.join(root,self.split)
        while line:
                
            line = fileDescriptor.readline()
            
            #--- if not empty
            if line:
          
                lineItems = line.split()
                imagePath = os.path.join(root, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)   
            
        fileDescriptor.close()

    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.listImageLabels[index])
        categorie = self.categories[imageLabel == 1].tolist()
        
        if self.transform != None: imageData = self.transform(imageData)
        
        # return imageData, imageLabel
        return dict(image=imageData, label=imageLabel, categories=categorie)

    def __len__(self):
        return len(self.listImagePaths)

    def collate_fn(self, instances: List[Tuple]):
        image_list, categories_list, label_list = [], [], []

        for b in instances:
            image_list.append(b["image"])
            categories_list.append(b["categories"])
            label_list.append(b["label"])

        image_stack = torch.stack(image_list)
        label_stack = torch.stack(label_list)

        return dict(
            image=image_stack,
            categories=categories_list,
            label=label_stack,
        )


class SIIM(data.Dataset):

    categories = ["Pneumothorax",]
    
    def __init__(self, root, root_split, data_volume, split="train", transform=None):
        super(SIIM, self)
        if data_volume == '1':
            raise ValueError('1% data not available for SIIM')
        if data_volume == '10':
            train_label_data = "train_10.txt"
        if data_volume == '100':
            train_label_data = "train_list.txt"
        test_label_data = "test_list.txt"
        val_label_data = "val_list.txt"
        
        self.split = split
        self.root = root
        self.transform = transform
        self.listImagePaths = []
        self.listImageLabels = []
        
        if self.split == "train":
            downloaded_data_label_txt = train_label_data
        elif self.split == "val":
            downloaded_data_label_txt = val_label_data
        elif self.split == "test":
            downloaded_data_label_txt = test_label_data

        df = pd.read_csv(os.path.join(root_split, "siim.csv")).groupby("ImageId")
           
        #---- Open file, get image paths and labels
        fileDescriptor = open(os.path.join(root_split, downloaded_data_label_txt), "r")
        
        #---- get into the loop
        line = True
        to_rlp = "/path/to/dataset/siim/dicom-images-train"
        rlp = "SIIM_ACR_Pneumothorax_and_RSNA_Pneumonia/SIIM ACR Pneumothorax Segmentation Data/jpg-images-train"
        while line:
            line = fileDescriptor.readline()
            #--- if not empty
            if line:
                line = line.strip()
                imgid_df = df.get_group(line)
                imagePath = imgid_df["Path"].tolist()[0].replace(".dcm", ".jpg")
                imagePath = os.path.join(root, imagePath.replace(to_rlp, rlp))
                imageLabel = [0,] if imgid_df[" EncodedPixels"].tolist()[0] == "-1" else [1,]
                assert os.path.isfile(imagePath)
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)   

        fileDescriptor.close()

    def __getitem__(self, index):
        imagePath = self.listImagePaths[index]
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel= torch.FloatTensor(self.listImageLabels[index])
        
        if self.transform != None: imageData = self.transform(imageData)
        
        return dict(image=imageData, label=imageLabel)

    def __len__(self):
        return len(self.listImagePaths)

    def collate_fn(self, instances: List[Tuple]):
        image_list, label_list = [], []

        for b in instances:
            image_list.append(b["image"])
            label_list.append(b["label"])

        image_stack = torch.stack(image_list)
        label_stack = torch.stack(label_list)

        return dict(image=image_stack, label=label_stack)

class RSNAPneumonia(torch.utils.data.Dataset):

    categories = np.array([
        "Lung Opacity"
    ])
    
    def __init__(self, root, root_split, data_volume, split="train", transform=None):
        # super(RSNAPneumonia, self)
        if data_volume == '1':
            train_label_data = "train_1.txt"
        if data_volume == '10':
            train_label_data = "train_10.txt"
        if data_volume == '100':
            train_label_data = "train_list.txt"
        test_label_data = "test_list.txt"
        val_label_data = "val_list.txt"
        
        self.split = split
        self.root = root
        self.transform = transform
        self.listImagePaths = []
        self.listImageLabels = []
        
        if self.split == "train":
            downloaded_data_label_txt = train_label_data
        
        elif self.split == "val":
            downloaded_data_label_txt = val_label_data
                 
        elif self.split == "test":
            downloaded_data_label_txt = test_label_data
           
        #---- Open file, get image paths and labels
        fileDescriptor = open(os.path.join(root_split, downloaded_data_label_txt), "r")
        #---- get into the loop
        line = True
        while line:
            line = fileDescriptor.readline()
            #--- if not empty
            if line:
                lineItems = line.split()
                imagePath = os.path.join(root, lineItems[0])
                imageLabel = lineItems[1:]
                assert len(imageLabel) == 1
                imageLabel = int(imageLabel[0])
                # imageLabel = [int(i) for i in imageLabel]
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)   
            
        fileDescriptor.close()

    def load_image(self, path) -> Image.Image:
        """Load an image from disk.

        The image values are remapped to :math:`[0, 255]` and cast to 8-bit unsigned integers.

        :param path: Path to image.
        :returns: Image as ``Pillow`` ``Image``.
        """
        image = dicom.dcmread(path).pixel_array
        image = remap_to_uint8(image)
        return Image.fromarray(image).convert("RGB")

    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = self.load_image(imagePath)
        imageLabel = torch.tensor(self.listImageLabels[index])
        # imageLabel = F.one_hot(imageLabel, num_classes=2)
        imageLabel = imageLabel.unsqueeze(0)
        categorie = self.categories[0]
        
        if self.transform is not None: 
            imageData = self.transform(imageData)
        
        return dict(image=imageData, label=imageLabel, categories=categorie)

    def __len__(self):
        return len(self.listImagePaths)

    def collate_fn(self, instances: List[Tuple]):
        image_list, categories_list, label_list = [], [], []

        for b in instances:
            image_list.append(b["image"])
            categories_list.append(b["categories"])
            label_list.append(b["label"])

        image_stack = torch.stack(image_list)
        label_stack = torch.stack(label_list)

        return dict(
            image=image_stack,
            categories=categories_list,
            label=label_stack,
        )


def remap_to_uint8(array: np.ndarray, percentiles: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Remap values in input so the output range is :math:`[0, 255]`.

    Percentiles can be used to specify the range of values to remap.
    This is useful to discard outliers in the input data.

    :param array: Input array.
    :param percentiles: Percentiles of the input values that will be mapped to ``0`` and ``255``.
        Passing ``None`` is equivalent to using percentiles ``(0, 100)`` (but faster).
    :returns: Array with ``0`` and ``255`` as minimum and maximum values.
    """
    array = array.astype(float)
    if percentiles is not None:
        len_percentiles = len(percentiles)
        if len_percentiles != 2:
            message = 'The value for percentiles should be a sequence of length 2,' f' but has length {len_percentiles}'
            raise ValueError(message)
        a, b = percentiles
        if a >= b:
            raise ValueError(f'Percentiles must be in ascending order, but a sequence "{percentiles}" was passed')
        if a < 0 or b > 100:
            raise ValueError(f'Percentiles must be in the range [0, 100], but a sequence "{percentiles}" was passed')
        cutoff: np.ndarray = np.percentile(array, percentiles)
        array = np.clip(array, *cutoff)
    array -= array.min()
    array /= array.max()
    array *= 255
    return array.astype(np.uint8)
