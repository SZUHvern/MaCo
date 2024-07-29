#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging

from detectron2.checkpoint import DetectionCheckpointer
import os
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm

logger = logging.getLogger("detectron2")

from detectron2.data import DatasetCatalog
import numpy as np
import pandas as pd

def RSNA_dataset_train():
    path = '/path/to/detectron2/projects/ViTDet/datasets/rsna-pneumonia-detection-challenge/maco_splits/train_maco_1per.npy'
    list_dict = np.load(path, allow_pickle=True)
    return list_dict
def RSNA_dataset_test():
    path = '/path/to/detectron2/projects/ViTDet/datasets/rsna-pneumonia-detection-challenge/maco_splits/test_maco.npy'
    list_dict = np.load(path, allow_pickle=True)
    return list_dict
def RSNA_dataset_valid():
    path = '/path/to/detectron2/projects/ViTDet/datasets/rsna-pneumonia-detection-challenge/maco_splits/val_maco.npy'
    list_dict = np.load(path, allow_pickle=True)
    return list_dict

DatasetCatalog.register("rsna_train", RSNA_dataset_train)
DatasetCatalog.register("rsna_test", RSNA_dataset_test)
DatasetCatalog.register("rsna_valid", RSNA_dataset_valid)
from detectron2.data import MetadataCatalog
MetadataCatalog.get("rsna_train").thing_classes = ["none_pneu", "pneu"]
MetadataCatalog.get("rsna_test").thing_classes = ["none_pneu", "pneu"]
MetadataCatalog.get("rsna_valid").thing_classes = ["none_pneu", "pneu"]

def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        cfg.dataloader.evaluator.output_dir = './out_valid'
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret

def do_valid(cfg, model):
    if "evaluator" in cfg.dataloader:
        cfg.dataloader.evaluator.output_dir = './out_valid'
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.valid), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret

def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    # cfg.dataloader.train.filter_empty_annotations = False
    train_loader = instantiate(cfg.dataloader.train)
    # DATALOADER.FILTER_EMPTY_ANNOTATIONS

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    # cfg.dataloader.FILTER_EMPTY_ANNOTATIONS = False
    # cfg.dataloader.filter_empty_annotations = False
    # cfg.MODEL.PIXEL_MEAN = [89.37338774, 85.83162284, 85.34460669]
    # cfg.MODEL.PIXEL_STD = [1,1,1] #list(std)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        if args.test:
            print(do_test(cfg, model))
        else:
            print(do_valid(cfg, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    # ../../tools/lazyconfig_train_net_rsna.py
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
    args = default_argument_parser().parse_args()
    # 
    ''''''
    args.config_file = '/path/to/detectron2/projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_50ep_maco.py'
    args.num_gpus=4
    args.num_machines=1
    args.test = True
    args.opts=['train.init_checkpoint=/path/to/maco.pth']
    ''''''
    
    print(args.machine_rank)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
