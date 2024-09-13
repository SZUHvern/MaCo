from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler, CosineParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

from ..common.rsna_loader import dataloader
    
model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model
model.pixel_mean = [125.281, 125.281, 125.281]
model.pixel_std = [63.929,63.929,63.929] #list(std)
model.roi_heads.num_classes = 2
dataloader.filter_empty_annotations = False

'''
# Options are: "smooth_l1", "giou", "diou", "ciou"
_C.MODEL.RPN.BBOX_REG_LOSS_TYPE = "smooth_l1"
# Options are: "smooth_l1", "giou", "diou", "ciou"
_C.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "smooth_l1"
# Whether to use class agnostic for bbox regression
_C.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = False
# If true, RoI heads use bounding boxes predicted by the box head rather than proposal boxes.
_C.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES = False
'''

# model.rpn.bbox_reg_loss_type = "ciou"
# model.roi_box_head.bbox_reg_loss_type = "ciou"
# model.roi_box_head.cls_agnostic_bbox_reg = False
# model.roi_box_head.train_on_pred_boxes = False


# Initialization and trainer settings
train = model_zoo.get_config("common/train_rsna.py").train

# 3584 / 32 * 50 = 5600
instance_size = 3584

batch_size = 32
epoch = 50
dataloader.train.total_batch_size = batch_size
train.max_iter = int(instance_size / batch_size) * epoch

train.amp.enabled = True
train.ddp.fp16_compression = True
train.eval_period = (instance_size / batch_size) * 5
train.amp = dict(enabled = True)



lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(CosineParamScheduler)(
        start_value=1, 
        end_value=0.01,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.01,
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.lr = 5e-4  #100per-3e4 10per5e4
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
