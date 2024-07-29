import math
from typing import Any
import numpy as np
import random
from PIL import Image
 
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
 
 
class Resize(object):
    def __init__(self, size):
        self.size = size
 
    def __call__(self, image, target=None):
        image = F.resize(image, self.size)
        if target is not None:
            if type(target) is list:
                target = [F.resize(t, self.size, interpolation=F.InterpolationMode.NEAREST) for t in target]
            else:
                target = F.resize(target, self.size, interpolation=F.InterpolationMode.NEAREST)
        return image, target
 
 
class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob
 
    def __call__(self, image, target=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if target is not None:
                if type(target) is list:
                    target = [F.hflip(t) for t in target]
                else:
                    target = F.hflip(target)
        return image, target
 

class RandomCrop(object):
    def __init__(self, size):
        self.size = size
 
    def __call__(self, image, target):
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        if target is not None:
            if type(target) is list:
                target = [F.crop(t, *crop_params) for t in target]
            else:
                target = F.crop(target, *crop_params)
        return image, target
    
 
class CenterCrop(object):
    def __init__(self, size):
        self.size = size
 
    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        if target is not None:
            if type(target) is list:
                target = [F.center_crop(t, self.size) for t in target]
            else:
                target = F.center_crop(target, self.size)
        return image, target


class Grayscale(object):
    def __init__(self, num_output_channels=1):
        super().__init__()
        self.num_output_channels = num_output_channels

    def __call__(self, image, target):
        image = F.rgb_to_grayscale(image, num_output_channels=self.num_output_channels)
        return image, target

 
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
 
    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
 

class Pad(object):
    def __init__(self, padding_n, padding_fill_value=0, padding_fill_target_value=0):
        self.padding_n = padding_n
        self.padding_fill_value = padding_fill_value
        self.padding_fill_target_value = padding_fill_target_value
 
    def __call__(self, image, target):
        image = F.pad(image, self.padding_n, self.padding_fill_value)
        if target is not None:
            if type(target) is list:
                target = [F.pad(t, self.padding_n, self.padding_fill_target_value) for t in target]
            else:
                target = F.pad(target, self.padding_n, self.padding_fill_target_value)
        return image, target
 

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        if target is not None:
            if type(target) is list:
                target = [torch.as_tensor(np.array(t), dtype=torch.int64) for t in target]
            else:
                target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
 
    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask
 
