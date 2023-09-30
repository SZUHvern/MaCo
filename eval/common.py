
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd 
import matplotlib
import torch.nn.functional as F
from math import ceil, floor
from pathlib import Path
from typing import Callable, Optional
from scipy import ndimage
sys.path.append(os.getcwd())
from eval.utils import load_data, norm_heatmap, biovil_show, biovil_show2, draw_n_images, load_ms_cxr, load_rsna_medklip, load_rsna
from eval.metric import compute_iou, compute_cnr, compute_pg, compute_dice, compute_recall, compute_precision

FONT_MAX = 50
matplotlib.use('Agg')
import time

class ImageTextInferenceEngine:

    def __init__(self) -> None:
        pass

    def load_model(self, **kwargs):
        '''
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = gloria.load_gloria(name=kwargs["ckpt"], device=device)
        self.image_inference_engine = self.model.image_encoder_forward
        self.text_inference_engine = self.model.text_encoder_forward
        '''
        raise NotImplementedError
        
    def get_emb(self, image_path: Path, query_text: str, device):
        '''
        return  iel: [h, w, feature_size]
                teg: [1, feature_size]

        
        pi = self.model.process_img([image_path], device)
        pt = self.model.process_text([query_text], device)
        with torch.no_grad():
            iel, ieg = self.image_inference_engine(pi.to(device))
            res = self.text_inference_engine(
                    pt["caption_ids"].to(device),
                    pt["attention_mask"].to(device),
                    pt["token_type_ids"].to(device))
        tel, teg, sts = res["word_embeddings"], res["sent_embeddings"], res["sents"]

        iel = iel.view(-1, *iel.shape[2:]).permute(1, 2, 0)
        return iel, teg
        '''
        raise NotImplementedError
        

    def get_similarity_map_from_raw_data(
        self, image_path: Path, query_text: str, device, interpolation: str = "nearest",
        ) -> np.ndarray:
        iel, teg = self.get_emb(image_path, query_text, device)
        sim = self._get_similarity_map_from_embeddings(iel, teg)
        resized_sim_map = self.convert_similarity_to_image_size(
            sim,
            width=224,
            height=224,
            resize_size=224,
            crop_size=224,
            interpolation=interpolation,
        )

        return resized_sim_map
    
    @staticmethod
    def set_margin(similarity_map, width=224, height=224, resize_size=512, crop_size=448):
        smallest_dimension = min(height, width)
        cropped_size_orig_space = int(crop_size * smallest_dimension / resize_size)
        target_size = cropped_size_orig_space, cropped_size_orig_space
        margin_w, margin_h = (width - target_size[0]), (height - target_size[1])
        margins_for_pad = (floor(margin_w / 2), ceil(margin_w / 2), floor(margin_h / 2), ceil(margin_h / 2))
        mask = torch.zeros(target_size)
        mask = F.pad(mask, margins_for_pad, value=float("NaN"))
        nan = torch.isnan(mask)
        similarity_map[nan] = float("NaN")
        return similarity_map

    @staticmethod
    def _get_similarity_map_from_embeddings(
        projected_patch_embeddings: torch.Tensor, projected_text_embeddings: torch.Tensor, sigma: float = 1.5
    ) -> torch.Tensor:
        """Get smoothed similarity map for a given image patch embeddings and text embeddings.

        :param projected_patch_embeddings: [n_patches_h, n_patches_w, feature_size]
        :param projected_text_embeddings: [1, feature_size]
        :return: similarity_map: similarity map of shape [n_patches_h, n_patches_w]
        """
        n_patches_h, n_patches_w, feature_size = projected_patch_embeddings.shape
        assert feature_size == projected_text_embeddings.shape[1]
        assert projected_text_embeddings.shape[0] == 1
        assert projected_text_embeddings.dim() == 2
        patch_wise_similarity = projected_patch_embeddings.view(-1, feature_size) @ projected_text_embeddings.t()
        patch_wise_similarity = patch_wise_similarity.reshape(n_patches_h, n_patches_w).cpu().numpy()
        smoothed_similarity_map = torch.tensor(
            ndimage.gaussian_filter(patch_wise_similarity, sigma=(sigma, sigma), order=0)
        )
        return smoothed_similarity_map
    
    @staticmethod
    def convert_similarity_to_image_size(
        similarity_map: torch.Tensor,
        width: int,
        height: int,
        resize_size: Optional[int],
        crop_size: Optional[int],
        interpolation: str = "nearest",
    ) -> np.ndarray:
        """
        Convert similarity map from raw patch grid to original image size,
        taking into account whether the image has been resized and/or cropped prior to entering the network.
        """
        n_patches_h, n_patches_w = similarity_map.shape[0], similarity_map.shape[1]
        target_shape = 1, 1, n_patches_h, n_patches_w
        smallest_dimension = min(height, width)

        # TODO:
        # verify_resize_params(val_img_transforms, resize_size, crop_size)

        reshaped_similarity = similarity_map.reshape(target_shape)
        align_corners_modes = "linear", "bilinear", "bicubic", "trilinear"
        align_corners = False if interpolation in align_corners_modes else None

        if crop_size is not None:
            if resize_size is not None:
                cropped_size_orig_space = int(crop_size * smallest_dimension / resize_size)
                target_size = cropped_size_orig_space, cropped_size_orig_space
            else:
                target_size = crop_size, crop_size
            similarity_map = F.interpolate(
                reshaped_similarity,
                size=target_size,
                mode=interpolation,
                align_corners=align_corners,
            )
            margin_w, margin_h = (width - target_size[0]), (height - target_size[1])
            margins_for_pad = (floor(margin_w / 2), ceil(margin_w / 2), floor(margin_h / 2), ceil(margin_h / 2))
            similarity_map = F.pad(similarity_map[0, 0], margins_for_pad, value=float("NaN"))
        else:
            similarity_map = F.interpolate(
                reshaped_similarity,
                size=(height, width),
                mode=interpolation,
                align_corners=align_corners,
            )[0, 0]
        return similarity_map.numpy()



class Pipeline:

    def __init__(self, inference: Callable, save_fig=False, **kwargs):
        self.image_text_inference = inference
        self.save_dir = None
        self.kwargs = kwargs
        self.save_fig = save_fig
    
    def run(self, **kwargs):
        self.kwargs.update(kwargs)
        self.createdir(kwargs["ckpt"], kwargs["dataset"])
        if kwargs["opt_th"]:
            save_path = f"{self.save_dir}/opt_th.csv"
            redo = kwargs["redo"]
            if not(os.path.exists(save_path) and redo == False):
                data = load_data(split="val", **kwargs)
                hmaps = self.get_hmaps(data["path"], data["label_text"], suffix="_val", **kwargs)
                self.get_opt_th(path_list=data["path"],
                    label_text=data["label_text"],
                    gtmasks=data["gtmasks"],
                    boxes=data["boxes"],
                    category=data["category"],
                    hmaps=hmaps,
                    **kwargs)
            data = load_data(split="test", **kwargs)
            hmaps = self.get_hmaps(data["path"], data["label_text"], suffix="_test", **kwargs)
            self.test_use_opt_th(path_list=data["path"],
                label_text=data["label_text"],
                gtmasks=data["gtmasks"],
                boxes=data["boxes"],
                category=data["category"],
                hmaps=hmaps,
                **kwargs) 
        else:
            data = load_data(**kwargs)
            hmaps = self.get_hmaps(data["path"], data["label_text"], **kwargs)
            self.test(path_list=data["path"],
                    label_text=data["label_text"],
                    gtmasks=data["gtmasks"],
                    boxes=data["boxes"],
                    category=data["category"],
                    hmaps=hmaps,
                    **kwargs) 

    def createdir(self, ckpt: str, dataset: str):
        if os.path.exists(ckpt):
            dn = os.path.join(os.path.dirname(ckpt), dataset)
            bn = os.path.splitext(os.path.basename(ckpt))[0]
            self.save_dir = os.path.join(dn, bn)
        else:
            self.save_dir = os.path.join(os.getcwd(), "result", dataset)
        os.makedirs(self.save_dir, exist_ok=True)

    def get_hmaps(self, path_list: list, label_text: list, redo=False, **kwargs):
        dataset = kwargs["dataset"]
        suffix = kwargs["suffix"] if "suffix" in kwargs else ""
        save_path = os.path.join(self.save_dir, f"hmaps{suffix}.npy")
        if os.path.exists(save_path) and redo == False:
            hmaps = np.load(save_path, allow_pickle=True).item()
        else:
            hmaps = {}
            self.image_text_inference.load_model(**kwargs)
            for i in tqdm(range(len(path_list[:]))):
                hmap = self.image_text_inference.get_similarity_map_from_raw_data(
                    image_path=path_list[i],
                    query_text=label_text[i],
                    device="cuda",
                    interpolation="bilinear",
                )
                key = str(path_list[i]) + label_text[i]
                hmaps[key] = hmap
            np.save(save_path, hmaps)
        return hmaps
    
    @staticmethod
    def set_margin(similarity_map, width=224, height=224, resize_size=512, crop_size=448):
        smallest_dimension = min(height, width)
        cropped_size_orig_space = int(crop_size * smallest_dimension / resize_size)
        target_size = cropped_size_orig_space, cropped_size_orig_space
        margin_w, margin_h = (width - target_size[0]), (height - target_size[1])
        margins_for_pad = (floor(margin_w / 2), ceil(margin_w / 2), floor(margin_h / 2), ceil(margin_h / 2))
        mask = torch.zeros(target_size)
        mask = F.pad(mask, margins_for_pad, value=float("NaN"))
        nan = torch.isnan(mask)
        similarity_map[nan] = float("NaN")
        return similarity_map

    def test(self, path_list: list, label_text: list, hmaps: list, gtmasks: list, boxes: list, category:list, **kwargs):
        dataset = kwargs["dataset"]
        res = pd.DataFrame()
        iou_thre = []
        cnr_thre = []
        pg_thre = []
        dice_thre = []
        recall_thre = []
        precision_thre = []
        iou_thre_cat = []
        cnr_thre_cat = []
        pg_thre_cat = []
        dice_thre_cat = []
        recall_thre_cat = []
        precision_thre_cat = []
        sep_cat = {}

        threshold_list = list(np.arange(0.1, 0.6, 0.1))
        for threshold in threshold_list:
            ious = []
            cnrs = []
            pgs = []
            dices = []
            recalls = []
            precisions = []
            cat_ious = {}
            cat_cnrs = {}
            cat_pgs = {}
            cat_dices = {}
            cat_recalls = {}
            cat_precisions = {}

            for i in tqdm(range(len(path_list[:]))):
                key = str(path_list[i]) + label_text[i]
                hmap = hmaps[key]
                if self.kwargs["margin"]:
                    hmap = self.set_margin(hmap)
                    
                nan = np.isnan(hmap)
                heatmap = norm_heatmap(hmap, nan, mode=0) # [-1, 1]
                gtmask = gtmasks[i]

                mask = np.where(heatmap > threshold, 1, 0)
                iou = compute_iou(gtmask, mask, nan)
                ious.append(iou)
                
                cnr = compute_cnr(gtmask, heatmap, nan)
                cnrs.append(cnr)
                
                pg = compute_pg(gtmask, heatmap, nan)
                pgs.append(pg)

                dice = compute_dice(gtmask, mask, nan)
                dices.append(dice)

                recall = compute_recall(gtmask, mask, nan)
                recalls.append(recall)

                precision = compute_precision(gtmask, mask, nan)
                precisions.append(precision)

                if kwargs["save_fig"] and threshold == threshold_list[0]:
                    save_path = f"{self.save_dir}/image/heatmap/{dataset}/{i}_{round(iou, 2)}_{round(cnr, 2)}_{label_text[i]}.png"
                    if boxes[i] is None:
                        nanmask = gtmask.astype(np.float32)
                        nanmask[nan] = float("NaN")
                    else:
                        nanmask = None
                    if iou >0.3:
                        biovil_show(path_list[i], heatmap, boxes[i], save_path, resize=224, gtmask=nanmask)

                cat = category[i]
                if cat not in cat_ious:
                    cat_ious[cat] = []
                    cat_cnrs[cat] = []
                    cat_pgs[cat] = []
                    cat_dices[cat] = []
                    cat_recalls[cat] = []
                    cat_precisions[cat] = []
                    
                cat_ious[cat].append(iou)
                cat_cnrs[cat].append(cnr)
                cat_pgs[cat].append(pg)
                cat_dices[cat].append(dice)
                cat_recalls[cat].append(recall)
                cat_precisions[cat].append(precision)

            iou_thre_cat.append(dict_mean(cat_ious))
            cnr_thre_cat.append(dict_mean(cat_cnrs))
            pg_thre_cat.append(dict_mean(cat_pgs))
            dice_thre_cat.append(dict_mean(cat_dices))
            recall_thre_cat.append(dict_mean(cat_recalls))
            precision_thre_cat.append(dict_mean(cat_precisions))

            iou_thre.append(np.mean(ious))
            cnr_thre.append(np.mean(cnrs))
            pg_thre.append(np.mean(pgs))
            dice_thre.append(np.mean(dices))
            recall_thre.append(np.mean(recalls))
            precision_thre.append(np.mean(precisions))

            if "iou" not in sep_cat:
                sep_cat["iou"] = pd.DataFrame(dict_mean(cat_ious, sep=True))
                sep_cat["cnr"] = pd.DataFrame(dict_mean(cat_cnrs, sep=True))
                sep_cat["pg"] = pd.DataFrame(dict_mean(cat_pgs, sep=True))
            else:
                sep_cat["iou"] = pd.concat([sep_cat["iou"], pd.DataFrame(dict_mean(cat_ious, sep=True))], axis=0, ignore_index=True)
                sep_cat["cnr"] = pd.concat([sep_cat["cnr"], pd.DataFrame(dict_mean(cat_cnrs, sep=True))], axis=0, ignore_index=True)
                sep_cat["pg"] = pd.concat([sep_cat["pg"], pd.DataFrame(dict_mean(cat_pgs, sep=True))], axis=0, ignore_index=True)

        res["threshold"] = threshold_list + ["mean"]
        res["iou"] = iou_thre + [np.mean(iou_thre)]
        res["cnr"] = cnr_thre + [np.mean(cnr_thre)]
        res["pg"] = pg_thre + [np.mean(pg_thre)]
        res["dice"] = dice_thre + [np.mean(dice_thre)]
        res["recall"] = recall_thre + [np.mean(recall_thre)]
        res["precision"] = precision_thre + [np.mean(precision_thre)]
        res["iou_cat"] = iou_thre_cat + [np.mean(iou_thre_cat)]
        res["cnr_cat"] = cnr_thre_cat + [np.mean(cnr_thre_cat)]
        res["pg_cat"] = pg_thre_cat + [np.mean(pg_thre_cat)]
        res["dice_cat"] = dice_thre_cat + [np.mean(dice_thre_cat)]
        res["recall_cat"] = recall_thre_cat + [np.mean(recall_thre_cat)]
        res["precision_cat"] = precision_thre_cat + [np.mean(precision_thre_cat)]

        sep_cat["iou"].rename(columns=prefix("iou_", sep_cat["iou"].columns), inplace=True)
        sep_cat["cnr"].rename(columns=prefix("cnr_", sep_cat["cnr"].columns), inplace=True)
        sep_cat["pg"].rename(columns=prefix("pg_", sep_cat["pg"].columns), inplace=True)

        # sep_cat add a row: mean
        sep_cat["iou"] = pd.concat([sep_cat["iou"], pd.DataFrame(sep_cat["iou"].mean(axis=0)).T], axis=0, ignore_index=True)
        sep_cat["cnr"] = pd.concat([sep_cat["cnr"], pd.DataFrame(sep_cat["cnr"].mean(axis=0)).T], axis=0, ignore_index=True)
        sep_cat["pg"] = pd.concat([sep_cat["pg"], pd.DataFrame(sep_cat["pg"].mean(axis=0)).T], axis=0, ignore_index=True)

        # concat
        for k, v in sep_cat.items():
            res = pd.concat([res, v], axis=1)

        res = res.round(3)
        res.to_csv(f"{self.save_dir}/metric.csv", index=False)
        print(res.loc[:, ["threshold", "iou", "cnr", "dice", "iou_cat", "cnr_cat", "dice_cat"]])

    def test_use_opt_th(self, path_list: list, label_text: list, hmaps: list, gtmasks: list, boxes: list, category:list, **kwargs):
        dataset = kwargs["dataset"]
        res = pd.DataFrame()
        iou_thre = []
        cnr_thre = []
        pg_thre = []
        dice_thre = []
        recall_thre = []
        precision_thre = []
        iou_thre_cat = []
        cnr_thre_cat = []
        pg_thre_cat = []
        dice_thre_cat = []
        recall_thre_cat = []
        precision_thre_cat = []
        sep_cat = {}

        opt_th = pd.read_csv(f"{self.save_dir}/opt_th.csv")
        
        ious = []
        cnrs = []
        pgs = []
        dices = []
        recalls = []
        precisions = []
        cat_ious = {}
        cat_cnrs = {}
        cat_pgs = {}
        cat_dices = {}
        cat_recalls = {}
        cat_precisions = {}

        for i in tqdm(range(len(path_list[:]))):
            cat = category[i]
            key = str(path_list[i]) + label_text[i]
            hmap = hmaps[key]
            if self.kwargs["margin"]:
                hmap = self.set_margin(hmap)
                
            nan = np.isnan(hmap)
            heatmap = norm_heatmap(hmap, nan, mode=1) # [0, 1]
            gtmask = gtmasks[i]
            threshold = opt_th[cat].values[0]
            mask = np.where(heatmap > threshold, 1, 0)
            iou = compute_iou(gtmask, mask, nan)
            ious.append(iou)
            
            cnr = compute_cnr(gtmask, heatmap, nan)
            cnrs.append(cnr)
            
            pg = compute_pg(gtmask, heatmap, nan)
            pgs.append(pg)

            dice = compute_dice(gtmask, mask, nan)
            dices.append(dice)

            recall = compute_recall(gtmask, mask, nan)
            recalls.append(recall)

            precision = compute_precision(gtmask, mask, nan)
            precisions.append(precision)

            if cat not in cat_ious:
                cat_ious[cat] = []
                cat_cnrs[cat] = []
                cat_pgs[cat] = []
                cat_dices[cat] = []
                cat_recalls[cat] = []
                cat_precisions[cat] = []
                
            cat_ious[cat].append(iou)
            cat_cnrs[cat].append(cnr)
            cat_pgs[cat].append(pg)
            cat_dices[cat].append(dice)
            cat_recalls[cat].append(recall)
            cat_precisions[cat].append(precision)

        iou_thre_cat.append(dict_mean(cat_ious))
        cnr_thre_cat.append(dict_mean(cat_cnrs))
        pg_thre_cat.append(dict_mean(cat_pgs))
        dice_thre_cat.append(dict_mean(cat_dices))
        recall_thre_cat.append(dict_mean(cat_recalls))
        precision_thre_cat.append(dict_mean(cat_precisions))

        iou_thre.append(np.mean(ious))
        cnr_thre.append(np.mean(cnrs))
        pg_thre.append(np.mean(pgs))
        dice_thre.append(np.mean(dices))
        recall_thre.append(np.mean(recalls))
        precision_thre.append(np.mean(precisions))

        if "iou" not in sep_cat:
            sep_cat["iou"] = pd.DataFrame(dict_mean(cat_ious, sep=True))
            sep_cat["cnr"] = pd.DataFrame(dict_mean(cat_cnrs, sep=True))
            sep_cat["pg"] = pd.DataFrame(dict_mean(cat_pgs, sep=True))
        else:
            sep_cat["iou"] = pd.concat([sep_cat["iou"], pd.DataFrame(dict_mean(cat_ious, sep=True))], axis=0, ignore_index=True)
            sep_cat["cnr"] = pd.concat([sep_cat["cnr"], pd.DataFrame(dict_mean(cat_cnrs, sep=True))], axis=0, ignore_index=True)
            sep_cat["pg"] = pd.concat([sep_cat["pg"], pd.DataFrame(dict_mean(cat_pgs, sep=True))], axis=0, ignore_index=True)

        res["iou"] = iou_thre
        res["cnr"] = cnr_thre
        res["pg"] = pg_thre
        res["dice"] = dice_thre
        res["recall"] = recall_thre
        res["precision"] = precision_thre
        res["iou_cat"] = iou_thre_cat
        res["cnr_cat"] = cnr_thre_cat
        res["pg_cat"] = pg_thre_cat
        res["dice_cat"] = dice_thre_cat
        res["recall_cat"] = recall_thre_cat
        res["precision_cat"] = precision_thre_cat

        sep_cat["iou"].rename(columns=prefix("iou_", sep_cat["iou"].columns), inplace=True)
        sep_cat["cnr"].rename(columns=prefix("cnr_", sep_cat["cnr"].columns), inplace=True)
        sep_cat["pg"].rename(columns=prefix("pg_", sep_cat["pg"].columns), inplace=True)

        # concat
        for k, v in sep_cat.items():
            res = pd.concat([res, v], axis=1)

        res = res.round(3)
        res.to_csv(f"{self.save_dir}/metric_opt_th.csv", index=False)
        print(res)

    def get_opt_th(self, path_list: list, label_text: list, hmaps: list, gtmasks: list, boxes: list, category:list, **kwargs):
        dataset = kwargs["dataset"]
        res = pd.DataFrame()
        iou_thre = []
        cnr_thre = []
        pg_thre = []
        dice_thre = []
        recall_thre = []
        precision_thre = []
        iou_thre_cat = []
        cnr_thre_cat = []
        pg_thre_cat = []
        dice_thre_cat = []
        recall_thre_cat = []
        precision_thre_cat = []
        sep_cat = {}

        threshold_list = np.arange(0.2, 0.8, 0.1)
        for threshold in threshold_list:
            ious = []
            cnrs = []
            pgs = []
            dices = []
            recalls = []
            precisions = []
            cat_ious = {}
            cat_cnrs = {}
            cat_pgs = {}
            cat_dices = {}
            cat_recalls = {}
            cat_precisions = {}

            for i in tqdm(range(len(path_list[:]))):
                key = str(path_list[i]) + label_text[i]
                hmap = hmaps[key]
                if self.kwargs["margin"]:
                    hmap = self.set_margin(hmap)
                    
                nan = np.isnan(hmap)
                heatmap = norm_heatmap(hmap, nan, mode=1) # [0, 1]
                gtmask = gtmasks[i]

                mask = np.where(heatmap > threshold, 1, 0)
                iou = compute_iou(gtmask, mask, nan)
                ious.append(iou)
                
                cnr = compute_cnr(gtmask, heatmap, nan)
                cnrs.append(cnr)
                
                pg = compute_pg(gtmask, heatmap, nan)
                pgs.append(pg)

                dice = compute_dice(gtmask, mask, nan)
                dices.append(dice)

                recall = compute_recall(gtmask, mask, nan)
                recalls.append(recall)

                precision = compute_precision(gtmask, mask, nan)
                precisions.append(precision)

                if kwargs["save_fig"] and threshold == threshold_list[0]:
                    save_path = f"{self.save_dir}/image/heatmap/{dataset}/{i}_{round(iou, 2)}_{round(cnr, 2)}.png"
                    if boxes[i] is None:
                        nanmask = gtmask.astype(np.float32)
                        nanmask[nan] = float("NaN")
                    else:
                        nanmask = None
                    biovil_show(path_list[i], heatmap, boxes[i], save_path, resize=224, gtmask=nanmask)

                cat = category[i]
                if cat not in cat_ious:
                    cat_ious[cat] = []
                    cat_cnrs[cat] = []
                    cat_pgs[cat] = []
                    cat_dices[cat] = []
                    cat_recalls[cat] = []
                    cat_precisions[cat] = []
                    
                cat_ious[cat].append(iou)
                cat_cnrs[cat].append(cnr)
                cat_pgs[cat].append(pg)
                cat_dices[cat].append(dice)
                cat_recalls[cat].append(recall)
                cat_precisions[cat].append(precision)

            iou_thre_cat.append(dict_mean(cat_ious))
            cnr_thre_cat.append(dict_mean(cat_cnrs))
            pg_thre_cat.append(dict_mean(cat_pgs))
            dice_thre_cat.append(dict_mean(cat_dices))
            recall_thre_cat.append(dict_mean(cat_recalls))
            precision_thre_cat.append(dict_mean(cat_precisions))

            iou_thre.append(np.mean(ious))
            cnr_thre.append(np.mean(cnrs))
            pg_thre.append(np.mean(pgs))
            dice_thre.append(np.mean(dices))
            recall_thre.append(np.mean(recalls))
            precision_thre.append(np.mean(precisions))

            if "iou" not in sep_cat:
                sep_cat["iou"] = pd.DataFrame(dict_mean(cat_ious, sep=True))
                sep_cat["cnr"] = pd.DataFrame(dict_mean(cat_cnrs, sep=True))
                sep_cat["pg"] = pd.DataFrame(dict_mean(cat_pgs, sep=True))
            else:
                sep_cat["iou"] = pd.concat([sep_cat["iou"], pd.DataFrame(dict_mean(cat_ious, sep=True))], axis=0, ignore_index=True)
                sep_cat["cnr"] = pd.concat([sep_cat["cnr"], pd.DataFrame(dict_mean(cat_cnrs, sep=True))], axis=0, ignore_index=True)
                sep_cat["pg"] = pd.concat([sep_cat["pg"], pd.DataFrame(dict_mean(cat_pgs, sep=True))], axis=0, ignore_index=True)

        argmax = sep_cat["iou"].values.argmax(axis=0)
        res = pd.DataFrame([threshold_list[argmax]], columns=sep_cat["iou"].columns)
        res = res.round(2)
        res.to_csv(f"{self.save_dir}/opt_th.csv", index=False)
        print(res)



def dict_mean(d: dict, sep=False):
    if sep:
        res = {}
        for k, v in d.items():
            res[k] = np.mean(v, keepdims=True)
        return res
    res = []
    for k, v in d.items():
        res.append(np.mean(v))
        # print(k, len(v))
    return np.mean(res)


def prefix(prefix, l):
    return {i: prefix + i for i in l}