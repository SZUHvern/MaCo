import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.config import LazyCall as L

# Data using LSJ
image_size = 1024
dataloader = model_zoo.get_config("common/data/rsna.py").dataloader
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),  # flip first
    L(T.ResizeScale)(
        min_scale=0.8, max_scale=1.2, target_height=image_size, target_width=image_size
    ),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
]
dataloader.train.mapper.image_format = "RGB"
dataloader.train.total_batch_size = 32
# recompute boxes due to cropping
dataloader.train.mapper.recompute_boxes = True
# dataloader.train.filter_empty_annotations = False
dataloader.filter_empty_annotations = False

dataloader.test.mapper.augmentations = [
    L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
]
