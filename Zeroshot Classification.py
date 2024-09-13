"""
This is a code templet for evaluation. It provided some basic evaluation 
functions. You should add your custom part at each `TODO` comments. 

Make sure that all `TODO` comments were checked and removed before running.
"""

# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MRM: https://github.com/RL4M/MRM-pytorch
# CheXzero: https://github.com/rajpurkarlab/CheXzero
# --------------------------------------------------------

import os
import argparse
import numpy as np
from os.path import join
from tqdm import tqdm
from typing import Dict, List, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from ZeroShot.utils.utils import AverageMeter, MultiAverageMeter
from ZeroShot.utils import metrics as Metrics

import ZeroShot.model_MaCo as model_MaCo
from ZeroShot.utils.my_dataset import RSNAPneumonia, NIHChestXray, SIIM
from ZeroShot.utils.prompts import get_all_text_prompts

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from einops import rearrange

class BaseEvaluateEngine:
    """
    This is a universal code structure for downstream task evaluation.
    """

    support_tasks = [
        "single_classification",
        "multi_classification",
    ]

    available_dataloaders = {
        "RSNA": RSNAPneumonia,
        "NIH": NIHChestXray,
        "SIIM": SIIM,
    }

    def __init__(self, args) -> None:
        self.args = args
        # loading data
        self.dataset, self.dataloader = self.load_dataset(args)
        # create model
        self.device = args.device
        self.model = self.load_model(args)
        self.model.to(self.device)
        # self.task = "multi_classification"
        self.loggers = self.create_metrics_logger(task=self.task) 
        # create other necessary components (e.g. text prompts in zero-shot classification)
        self.create_necessary_components()
        self.output_file = self.args.output_file

    @staticmethod
    def get_arguments() -> argparse:
        parser = argparse.ArgumentParser(description='args')
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--device", type=str, default="cuda:0")

        parser.add_argument("--output_file", type=str, 
                            default="results/test_evaluation_result.json") 
        
        parser.add_argument("--bert_type", default="MaCo",
                            choices=["MaCo"]
                            )
        parser.add_argument("--force_download_bert", default=False, action="store_true")
        parser.add_argument("--normalize_feature", default=False, action="store_true")
        parser.add_argument("--text_prompt", default="default", choices=["default", "siim", "covid"])
        parser.add_argument("--multi_class", default=True)
        parser.add_argument("--pretrained_path", type=str,
                            default="/path/to/pretrained_model.pth")
        parser.add_argument("--dataset", default="NIH", 
                            choices=["RSNA", "NIH", "SIIM"])
        parser.add_argument("--dataset_path", default='/path/to/MIMIC-DATA-Final/', type=str)
        parser.add_argument("--dataset_list", default=["NIH", "RSNA", "SIIM"], type=str)

        args = parser.parse_args()

        return args

    def create_necessary_components(self):
        # set text prompts and cls feature
        print("Setting text prompts and cls feature...")
        all_prompts = get_all_text_prompts(self.dataset.categories, self.args)
        self.model.set_cls_feature(all_prompts)

    def load_model(self, args) -> torch.nn.Module:
        """ Load the model. 
        Just load the model and leave the post-processing in self.post_process()
        Input:
            - Necessary arguments to load the pretrained model. (e.g. ckpt_dir) 
        Return:
            - model(torch.nn.Module): Any torch model with loaded parameters
        """
        # step-1: init model
        model = model_MaCo.maco(
            bert_type=args.bert_type,
            normalize_feature=args.normalize_feature,
            multi_class=args.multi_class,
            num_classes=len(self.dataset.categories),
        )

        # step-2: load pretrain parameters
        checkpoint = torch.load(args.pretrained_path, map_location=torch.device('cpu'))
        try:
            model_dict = checkpoint['state_dict']
        except:
            model_dict = checkpoint['model']
        msg = model.load_state_dict(model_dict, strict=False)
        print(msg)

        model.train(False)

        return model

    def load_dataset(self, args) -> torch.utils.data.Dataset:
        """ Load the dataset
        Input:
            - Necessary arguments to load the dataset. (e.g. dataset_dir) 
        Return:
            - dataset(torch.utils.data.Dataset): A torch dataset.
        """
        #### transforms.Normalize(mean=[0.4978], std=[0.2449]) #####
        if args.dataset == "NIH":
            self.task = "multi_classification"
            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4978],std=[0.2449])
                # transforms.Normalize(mean=[0.4977],std=[0.2276])
            ])
            dataset = self.available_dataloaders["NIH"](
                root=join(args.dataset_path, "/path/to/COVID-19_and_ChestX-ray14/CXR8/images/images_all"),
                root_split="/path/to/DatasetsSplits/NIH_ChestX-ray/",
                data_volume="100",
                split="test",
                transform=transform_test,
            )
            dataloader = DataLoader(
                dataset, 
                batch_size=args.batch_size,
                num_workers=8,
                shuffle=False,
                collate_fn=dataset.collate_fn,
            )
        elif args.dataset == "SIIM":
            self.task = "multi_classification"
            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4978],std=[0.2449])
                # transforms.Normalize(mean=[0.4998],std=[0.2348])
            ])
            dataset = self.available_dataloaders["SIIM"](
                root=args.dataset_path,
                root_split="/path/to/DatasetsSplits/SIIM-ACR_Pneumothorax/",
                data_volume="100",
                split="test",
                transform=transform_test,
            )
            dataloader = DataLoader(
                dataset, 
                batch_size=args.batch_size,
                num_workers=0,
                shuffle=False,
                collate_fn=dataset.collate_fn,
            )
        elif args.dataset == "RSNA":
            # self.task = "single_classification"
            self.task = "multi_classification"
            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4978],std=[0.2449])
                # transforms.Normalize(mean=[0.5034],std=[0.2343])
            ])
            dataset = self.available_dataloaders["RSNA"](
                root=join(args.dataset_path, "rsna-pneumonia-detection-challenge"),
                root_split="/path/to/DatasetsSplits/RSNA_Pneumonia/",
                data_volume="100",
                split="test",
                transform=transform_test,
            )
            dataloader = DataLoader(
                dataset, 
                batch_size=args.batch_size,
                num_workers=8,
                shuffle=False,
                collate_fn=dataset.collate_fn,
            )
        return dataset, dataloader

    def compute_metrics(self, data) -> Dict[str,float]:
        """ Compute metrics
        Input:
            - data (Dict[str,np.ndarray]): A dict contain prediction & ground-truth
                For more details about the input Dict, please refer to the instruction
                of function compute_xxx_metrics() in utils.metrics
        Return:
            - metrics (Dict): A dict with multiple metrics
        """
        if self.task == "single_classification":
            # auc, acc, f1
            return Metrics.compute_single_classification_metrics(**data)
        elif self.task == "multi_classification":
            # auc, acc, f1
            return Metrics.compute_multi_classification_metrics(**data)
        elif self.task == "grounding":
            # point_game, recall, precision, iou, dice
            return Metrics.compute_grounding_metrics(**data)
        elif self.task == "segmentation":
            # precision, iou, dice
            return Metrics.compute_segmentation_metrics(**data)
        else:
            raise ValueError("[ERROR] Do not support metric computation for "
                             f"task: {self.task} Currently supported tasks are:"
                             f" {self.support_tasks}")

    def create_metrics_logger(
            self, 
            task: str
        ) -> Dict[str,Union[AverageMeter,MultiAverageMeter]]:
        """
        Input: 
            - task (str): task name in self.support_tasks
        Return:
            - loggers (Dict[Union[AverageMeter,MultiAverageMeter]])
        """
        if task not in self.support_tasks:
            raise ValueError(f"[ERROR]: Task {task} was not supportef. "
                             f"Here are the supported tasks {self.support_tasks}")

        if task == "single_classification":
            loggers = {
                "auc": AverageMeter(),
                "acc": AverageMeter(),
                "f1": AverageMeter(),
            }
        elif task == "multi_classification":
            loggers = {
                # "auc": AverageMeter(),
                "auc": MultiAverageMeter(
                    len(self.dataset.categories), 
                    self.dataset.categories
                ),
                "acc": MultiAverageMeter(
                    len(self.dataset.categories), 
                    self.dataset.categories
                ),
                "f1": MultiAverageMeter(
                    len(self.dataset.categories), 
                    self.dataset.categories
                ),
                "mcc": MultiAverageMeter(
                    len(self.dataset.categories),
                    self.dataset.categories
                ),
                "prec": MultiAverageMeter(
                    len(self.dataset.categories),
                    self.dataset.categories
                ),
                "recall": MultiAverageMeter(
                    len(self.dataset.categories),
                    self.dataset.categories
                ),
                "tnr": MultiAverageMeter(
                    len(self.dataset.categories),
                    self.dataset.categories
                ),
                "jac": MultiAverageMeter(
                    len(self.dataset.categories),
                    self.dataset.categories
                ),
            }
        elif task == "segmentation":
            loggers = {
                "point_game": AverageMeter(),
                "recall": AverageMeter(),
                "precision": AverageMeter(),
                "iou": AverageMeter(),
                "dice": AverageMeter(),
            }
        elif task == "grounding":
            loggers = {
                "precision": AverageMeter(),
                "iou": AverageMeter(),
                "dice": AverageMeter(),
            }

        return loggers
    
    def log_metrics(self, metrics: Dict, num_sample: int) -> None:
        for k in self.loggers.keys():
            self.loggers[k].update(metrics[k], n=num_sample)

    def post_process(self, output: torch.Tensor, **kwargs) -> np.ndarray:
        """ Re-format your model output to match the unified metrics.
        Input: 
            - output: any data type & shapes from your model.
            - Other necessary arguments.
        Return:
            - output (np.ndarray): re-formatted data to fit metrics. 
                                   It should be: (batch_size, class_number)
        """

        """ Rules of output. The rules is diff for vaious tasks:        
        - Classification (single/multi-class): 
            np.ndarry(float32) with shape (batch_size, class_number)
        - Grounding: 
            np.ndarry(float32) with shape (batch_size, height, width)
        - Segmentation: 
            np.ndarry(float32) with shape (batch_size, height, width)
        """
        # output = None  # np.ndarray follow the format above
        if self.task == "single_classification":
            # reshape to [batch_size, num_classes]
            pos, neg = output[:,0,:].detach().chunk(2, dim=-1)
            # swap the order of pos and neg
            output = torch.cat([neg, pos], dim=-1).cpu().numpy()
        else:
            output = output[:,:,0].detach().cpu().numpy()
        
        return output

    def prepare_eval_data(
            self, 
            output: torch.Tensor, 
            batch: Dict[str,Union[np.ndarray,torch.Tensor,float]],
        ) -> Dict[str,np.ndarray]:
        """Prepare the `data` required in self.compute_metrics()
        Input:
            - output: model output or prediction
            - batch: batch data from dataloader
        Return:
            - Dict[str,np.ndarray]
        """
        if self.task == "single_classification":
            data = dict(
                pred=output,
                label=batch["label"], #.cpu().numpy(),
            )
        elif self.task == "multi_classification":
            data = dict(
                pred=output,
                label=batch["label"], #.cpu().numpy(),
                categories=self.dataset.categories,
            )
        elif self.task == "grounding":
            data = dict(
                pred=output,
                cls_label=batch["label"].cpu().numpy(),
                seg_mask=batch["mask"].cpu().numpy(),
                bbox_mask=batch['bbox'].cpu().numpy(),
            )
        elif self.task == "segmentation":
            data = dict(
                pred=output,
                seg_mask=batch["mask"].cpu().numpy(),
            )

        return data


    @torch.no_grad()
    def evaluate(self) -> None:

        all_outputs, all_labels = [], []
        img_output, logits_pos_output, logits_neg_output = [], [], []
        for batch in tqdm(self.dataloader, ncols=100, total=len(self.dataloader)):
            input_img = batch["image"].to(self.device)
            output, latent_img, logits_pos, logits_neg = self.model(input_img)
            # post-process
            output = self.post_process(output)

            all_outputs.append(output)
            all_labels.append(batch["label"].cpu().numpy())
            img_output.append(latent_img)
        
        all_labels = np.concatenate(all_labels)
        img_output = torch.cat(img_output, dim=0)
        
        img_output = img_output.unsqueeze(1)
        logits_pos = logits_pos.unsqueeze(0).repeat(img_output.shape[0], 1, 1)
        logits_neg = logits_neg.unsqueeze(0).repeat(img_output.shape[0], 1, 1)
        tsne_fea = img_output * logits_pos
        tsne_fea = rearrange(tsne_fea, 'b c d -> b (c d)')
        img_output = tsne_fea
        

        img_output_singleclass = []
        label_singleclass = []
        if all_labels.shape[1] != 1:
            for i in range(len(all_labels)):
                if sum(all_labels[i]) == 1:
                    label_value = np.where(all_labels[i] == 1)[0][0]
                    label_singleclass.append(label_value)
                    img_output_singleclass.append(img_output[i])
            img_output_singleclass = torch.stack(img_output_singleclass).detach().cpu().numpy()
            img_output_singleclass = np.array(img_output_singleclass)
        else:
            img_output_singleclass = img_output.detach().cpu().numpy()
            img_output_singleclass = np.array(img_output_singleclass)
            label_singleclass = all_labels

        label_singleclass = np.array(label_singleclass)
        
        metric_data = self.prepare_eval_data(
            np.concatenate(all_outputs), 
            dict(label=all_labels)
        )
        metrics = self.compute_metrics(metric_data)
        print('%s  auc:%.4f' %  (metrics['auc']) + "\n")

    def run(self):
        self.evaluate()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    args = BaseEvaluateEngine.get_arguments()

    for j in args.dataset_list:
        if j == 'SIIM':
            args.text_prompt = 'siim'
        else:
            args.text_prompt = 'default'
        args.dataset=j
        engine = BaseEvaluateEngine(args)
        engine.run()
