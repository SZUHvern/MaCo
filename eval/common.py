
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
from eval.utils import load_data, norm_heatmap, biovil_show, biovil_show2, draw_n_images, load_ms_cxr, load_rsna_medklip, load_rsna, biovil_show_crop_heat, biovil_show_crop_img
from eval.metric import compute_iou, compute_cnr, compute_pg, compute_dice, compute_recall, compute_precision, compute_iou1, ACC, AUC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score
FONT_MAX = 50
matplotlib.use('Agg')
import time
from functools import reduce

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

    def get_img_emb(self, image_path: Path, device):
        raise NotImplementedError

    def get_text_emb(self, query_text: str, device):
        raise NotImplementedError
    
    def head_pred(self, img_emb_g, label, device):
        MIMIC_TASKS = [
            "No Finding",
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Lesion",
            "Lung Opacity",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
        ]
        with torch.no_grad():
            pred = self.model.decoder(img_emb_g)
        if label == 'Airspace Opacity':
            index = MIMIC_TASKS.index("Lung Opacity")
        elif label == "COVID-19":
            index = MIMIC_TASKS.index("Pneumonia")
        else:
            index = MIMIC_TASKS.index(label)
        prob = torch.sigmoid(pred[0, index])
        out = {"head_prob": prob.cpu().numpy()}
        return out

    def zero_shot_cls(self, ieg, label, device):
        # query_texts = [i.format(label) for i in ("{}", "no {}")]
        query_texts = [i.format(label) for i in ("Findings suggesting {}", "no {}")]
        text_emb = self.get_text_emb(query_texts, device)
        teg = text_emb["teg"]
        pred = self._get_similarity_from_embeddings(ieg, teg)
        pos_pred = pred[0, 0]
        neg_pred = pred[0, 1]
        sum_pred = torch.exp(pos_pred) + torch.exp(neg_pred)
        y_pred_sfmax = torch.exp(pos_pred) / sum_pred
        out = {"prob_sfmax": y_pred_sfmax.cpu().numpy(),
               "pos_prob": pos_pred.cpu().numpy(), 
               "neg_prob": neg_pred.cpu().numpy()}
        return out

    def get_similarity_map_from_raw_data(
        self, image_path: Path, query_text: str, device, interpolation: str = "nearest",
        ) -> np.ndarray:
        img_emb = self.get_img_emb(image_path, device)
        text_emb = self.get_text_emb(query_text, device)
        iel, teg = img_emb["iel"], text_emb["teg"]
        sim = self._get_similarity_map_from_embeddings(iel, teg)
        resized_sim_map = self.convert_similarity_to_image_size(
            sim,
            width=224,
            height=224,
            resize_size=224,
            crop_size=224,
            interpolation=interpolation,
        )

        return resized_sim_map, img_emb, text_emb
    
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
    def _get_similarity_from_embeddings(
        projected_image_embeddings: torch.Tensor, projected_text_embeddings: torch.Tensor, sigma: float = 1.5
    ) -> torch.Tensor:
        """
        :param projected_image_embeddings: [1, feature_size]
        :param projected_text_embeddings: [cls_num, feature_size]
        :return: similarity: similarity of shape [1, cls_num]
        """
        img_norm = projected_image_embeddings / projected_image_embeddings.norm(dim=-1, keepdim=True)
        text_norm = projected_text_embeddings / projected_text_embeddings.norm(dim=-1, keepdim=True)
        similarity = img_norm @ text_norm.t()

        # similarity = projected_image_embeddings @ projected_text_embeddings.t()

        return similarity

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
            # if "only_pos" in kwargs and kwargs["only_pos"]:
            if False:
                save_path = f"{self.save_dir}/opt_th_only_pos.csv"
            else:
                suffix = "_use_prob" if "use_prob" in kwargs and kwargs["use_prob"] else ""
                save_path = f"{self.save_dir}/opt_th{suffix}.csv" 
            redo = kwargs["redo"]
            if not(os.path.exists(save_path) and redo == False):
                data = load_data(split="val", **kwargs)
                hmaps = self.get_hmaps(data["path"], data["label_text"], data["category"], suffix="_val", **kwargs)
                self.get_opt_th(path_list=data["path"],
                    label_text=data["label_text"],
                    gtmasks=data["gtmasks"],
                    boxes=data["boxes"],
                    category=data["category"],
                    label_list=data["label"],
                    hmaps=hmaps,
                    **kwargs)
            for split in ["val", "test"]:
                kwargs["eval_val_or_test"] = split
                data = load_data(split=split, **kwargs)
                hmaps = self.get_hmaps(data["path"], data["label_text"], data["category"], suffix=f"_{split}", **kwargs)
                self.test_use_opt_th(path_list=data["path"],
                    label_text=data["label_text"],
                    gtmasks=data["gtmasks"],
                    boxes=data["boxes"],
                    category=data["category"],
                    label_list=data["label"],
                    hmaps=hmaps,
                    **kwargs)

        else:
            data = load_data(**kwargs)
            hmaps = self.get_hmaps(data["path"], data["label_text"], data["category"], **kwargs)
            result = self.test(path_list=data["path"],
                    label_text=data["label_text"],
                    gtmasks=data["gtmasks"],
                    boxes=data["boxes"],
                    category=data["category"],
                    hmaps=hmaps,
                    **kwargs) 
            return result

    def createdir(self, ckpt: str, dataset: str):
        if os.path.exists(ckpt):
            dn = os.path.join(os.path.dirname(ckpt), dataset)
            bn = os.path.splitext(os.path.basename(ckpt))[0]
            self.save_dir = os.path.join(dn, bn)
        else:
            self.save_dir = os.path.join(os.getcwd(), "result", dataset)
        os.makedirs(self.save_dir, exist_ok=True)

    def get_hmaps(self, path_list: list, label_text: list, category, redo=False, **kwargs):
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
                hmaps[key] = {"hmap": hmap}

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
        metric_df_thre_iou = []
        metric_df_thre_cnr = []

        threshold_list = list(np.arange(0.1, 0.6, 0.1))
        path_dict = {}
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
                hmap = hmaps[key]["hmap"]
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
                    name = label_text[i].replace("/", "_")
                    if path_list[i] not in path_dict:
                        path_dict[path_list[i]] = 0
                    else:
                        path_dict[path_list[i]] += 1
                    cxr_idx = len(path_dict) - 1
                    dis_idx = path_dict[path_list[i]]
                    save_path = f"{self.save_dir}/image/heatmap/{dataset}/{cxr_idx}_{dis_idx}_{round(iou, 2)}_{round(cnr, 2)}_{name}.png"
                    # print(save_path)
                    if boxes[i] is None:
                        nanmask = gtmask.astype(np.float32)
                        nanmask[nan] = float("NaN")
                    else:
                        nanmask = None

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

            metric_df_iou = to_metric_df(cat_ious)
            metric_df_thre_iou.append(metric_df_iou)
            metric_df_cnr = to_metric_df(cat_cnrs)
            metric_df_thre_cnr.append(metric_df_cnr)

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
                sep_cat["dice"] = pd.DataFrame(dict_mean(cat_dices, sep=True))
            else:
                sep_cat["iou"] = pd.concat([sep_cat["iou"], pd.DataFrame(dict_mean(cat_ious, sep=True))], axis=0, ignore_index=True)
                sep_cat["cnr"] = pd.concat([sep_cat["cnr"], pd.DataFrame(dict_mean(cat_cnrs, sep=True))], axis=0, ignore_index=True)
                sep_cat["pg"] = pd.concat([sep_cat["pg"], pd.DataFrame(dict_mean(cat_pgs, sep=True))], axis=0, ignore_index=True)
                sep_cat["dice"] = pd.concat([sep_cat["dice"], pd.DataFrame(dict_mean(cat_dices, sep=True))], axis=0, ignore_index=True)

        total_df_iou = reduce(lambda x, y: x.add(y, fill_value=0), metric_df_thre_iou)
        total_df_cnr = reduce(lambda x, y: x.add(y, fill_value=0), metric_df_thre_cnr)
        average_df_iou = total_df_iou / len(metric_df_thre_iou)
        average_df_cnr = total_df_cnr / len(metric_df_thre_cnr)
        # ci_df_iou = bootci(average_df_iou, self.save_dir, "iou")
        # ci_df_cnr = bootci(average_df_cnr, self.save_dir, "cnr")

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
        sep_cat["dice"].rename(columns=prefix("dice_", sep_cat["dice"].columns), inplace=True)

        # sep_cat add a row: mean
        sep_cat["iou"] = pd.concat([sep_cat["iou"], pd.DataFrame(sep_cat["iou"].mean(axis=0)).T], axis=0, ignore_index=True)
        sep_cat["cnr"] = pd.concat([sep_cat["cnr"], pd.DataFrame(sep_cat["cnr"].mean(axis=0)).T], axis=0, ignore_index=True)
        sep_cat["pg"] = pd.concat([sep_cat["pg"], pd.DataFrame(sep_cat["pg"].mean(axis=0)).T], axis=0, ignore_index=True)
        sep_cat["dice"] = pd.concat([sep_cat["dice"], pd.DataFrame(sep_cat["dice"].mean(axis=0)).T], axis=0, ignore_index=True)

        # concat
        for k, v in sep_cat.items():
            res = pd.concat([res, v], axis=1)

        res = res.round(3)
        res.to_csv(f"{self.save_dir}/metric.csv", index=False)
        print(res.loc[:, ["threshold", "iou", "cnr", "dice", "pg", "iou_cat", "cnr_cat", "dice_cat", "pg_cat"]])
        return res

    def test_use_opt_th(self, path_list: list, label_text: list, hmaps: list, gtmasks: list, boxes: list, category:list, label_list:list, **kwargs):
        dataset = kwargs["dataset"]
        res = pd.DataFrame()
        iou_thre = []
        iou_thre_cat = []
        split = kwargs["eval_val_or_test"] if "eval_val_or_test" in kwargs else "test"
        only_pos = kwargs["only_pos"] if "only_pos" in kwargs else False
        sep_cat = {}

        suffix = "_use_prob" if "use_prob" in kwargs and kwargs["use_prob"] else ""
        opt_th = pd.read_csv(f"{self.save_dir}/opt_th{suffix}.csv")
        opt_th_only_pos = pd.read_csv(f"{self.save_dir}/opt_th_only_pos.csv")
        
        ious = []
        cat_ious = {}
        tp_cat_ious = {}
        neg = 0
        pos = 0
        FN = 0
        tp = 0
        iou_0 = 0
        iou_0_p = 0
        iou_0_t = 0
        iou_0_tp = 0
        iou_nan = 0
        iou_nan_t = 0
        iou_nan_p = 0
        iou_nan_tp = 0

        select = {}
        for i in tqdm(range(len(path_list[:]))):
            cat = category[i]
            key = str(path_list[i]) + label_text[i]
            hmap = hmaps[key]["hmap"]

            if self.kwargs["margin"]:
                hmap = self.set_margin(hmap)

            nan = np.isnan(hmap)
            gtmask = gtmasks[i]
            heatmap = norm_heatmap(hmap, nan, mode=1) # [0, 1]
            # if only_pos:
            if "use_prob" in kwargs and kwargs["use_prob"]:
                prob = hmaps[key]["prob_sfmax"]
                # prob = hmaps[key]["neg_prob"]
                # prob = hmaps[key]["head_prob"]
                confidence = prob
            else:
                min_v, max_v = 0, 2
                clip = np.clip(hmap[~nan], min_v, max_v)
                confidence = np.max((clip - min_v) / (max_v - min_v))
            confidence_th = opt_th[cat].values[0]
            if confidence < confidence_th:
                mask = np.zeros_like(heatmap)
            else:
                mask_th = opt_th_only_pos[cat].values[0]
                mask = np.where(heatmap > mask_th, 1, 0)
            iou = compute_iou(gtmask, mask, nan, only_pos=only_pos)

            ious.append(iou)
            if iou == 0:
                iou_0 += 1
                if np.sum(gtmask) == 0 and np.sum(mask) != 0:
                    iou_0_p += 1
                if np.sum(mask) == 0 and np.sum(gtmask) != 0:
                    iou_0_t += 1
                if np.sum(gtmask) != 0 and np.sum(mask) != 0:
                    iou_0_tp += 1
                if np.sum(gtmask) == 0 and np.sum(mask) == 0:
                    print(key)
            if np.isnan(iou):
                iou_nan += 1
                if np.sum(gtmask) == 0 and np.sum(mask) != 0:
                    iou_nan_p += 1
                if np.sum(mask) == 0 and np.sum(gtmask) != 0:
                    iou_nan_t += 1
                if np.sum(gtmask) != 0 and np.sum(mask) != 0:
                    iou_nan_tp += 1
            if iou > 0:
                tp += 1            

            if cat not in cat_ious:
                cat_ious[cat] = []

            cat_ious[cat].append(iou)
            if cat == "Pneumothorax" and not np.isnan(iou):
                select[key] = iou

            if cat not in tp_cat_ious:
                tp_cat_ious[cat] = []
            if gtmask.sum() > 0 and mask.sum() > 0:
                tp_cat_ious[cat].append(iou)

        print(f"tp: {tp}, iou_0: {iou_0}, iou_0_p: {iou_0_p}, iou_0_t: {iou_0_t}, iou_0_tp: {iou_0_tp}, iou_nan: {iou_nan}, iou_nan_t: {iou_nan_t}, iou_nan_p: {iou_nan_p}, iou_nan_tp: {iou_nan_tp}, total:{iou_0+iou_nan+tp}")
        iou_thre_cat.append(dict_mean(cat_ious))
        iou_thre.append(np.nanmean(ious))
        print(f"pos: {pos}, neg: {neg}, FN: {FN}")
        if "iou" not in sep_cat:
            sep_cat["iou"] = pd.DataFrame(dict_mean(cat_ious, sep=True))
        else:
            sep_cat["iou"] = pd.concat([sep_cat["iou"], pd.DataFrame(dict_mean(cat_ious, sep=True))], axis=0, ignore_index=True)

        res["iou"] = iou_thre
        res["iou_cat"] = iou_thre_cat

        sep_cat["iou"].rename(columns=prefix("iou_", sep_cat["iou"].columns), inplace=True)

        # concat
        for k, v in sep_cat.items():
            res = pd.concat([res, v], axis=1)

        res = res.round(3)
        if only_pos:
            suffix = "_only_pos"
        else:
            suffix = "_use_prob" if "use_prob" in kwargs and kwargs["use_prob"] else ""
        metric_df = to_metric_df(cat_ious)    
        ci_df = bootci(metric_df, self.save_dir, f"iou{suffix}")
        res.to_csv(f"{self.save_dir}/metric_opt_th_{split}{suffix}.csv", index=False)
        print(res)
        print(f"{self.save_dir}/metric_opt_th_{split}{suffix}.csv")

    def get_opt_th(self, path_list: list, label_text: list, hmaps: list, gtmasks: list, boxes: list, category:list, label_list:list, **kwargs):
        dataset = kwargs["dataset"]
        suffix = "_use_prob" if "use_prob" in kwargs and kwargs["use_prob"] else ""
        only_pos = kwargs["only_pos"] if "only_pos" in kwargs else False
        opt_th_only_pos_path = f"{self.save_dir}/opt_th_only_pos.csv"
        opt_th_path = f"{self.save_dir}/opt_th{suffix}.csv"
        auc_path = f"{self.save_dir}/auc{suffix}.csv"

        if not os.path.exists(opt_th_only_pos_path):
            iou_thre = []
            iou_thre_cat = []
            sep_cat = {}
            threshold_list = np.arange(0.2, 0.8, 0.1)
            for threshold in threshold_list:
                ious = []
                cat_ious = {}

                for i in tqdm(range(len(path_list[:]))):
                    
                    key = str(path_list[i]) + label_text[i]
                    hmap = hmaps[key]["hmap"]
                    if self.kwargs["margin"]:
                        hmap = self.set_margin(hmap)

                    nan = np.isnan(hmap)
                    heatmap = norm_heatmap(hmap, nan, mode=1) # [0, 1]

                    mask = np.where(heatmap > threshold, 1, 0)
                    gtmask = gtmasks[i]

                    iou = compute_iou(gtmask, mask, nan, only_pos=True)
                    ious.append(iou)

                    cat = category[i]
                    if cat not in cat_ious:
                        cat_ious[cat] = []
                    cat_ious[cat].append(iou)

                iou_thre_cat.append(dict_mean(cat_ious))
                iou_thre.append(np.mean(ious))

                if "iou" not in sep_cat:
                    sep_cat["iou"] = pd.DataFrame(dict_mean(cat_ious, sep=True))
                else:
                    sep_cat["iou"] = pd.concat([sep_cat["iou"], pd.DataFrame(dict_mean(cat_ious, sep=True))], axis=0, ignore_index=True)

            argmax = sep_cat["iou"].values.argmax(axis=0)
            res = pd.DataFrame([threshold_list[argmax], sep_cat["iou"].max().values], columns=sep_cat["iou"].columns)
            res = pd.concat([res, res.mean(axis=1).rename("mean")], axis=1)
            res = res.round(3)
            res.to_csv(opt_th_only_pos_path, index=False)
            print(res)

        # if not only_pos:
        if True:
            iou_thre = []
            iou_thre_cat = []
            sep_cat = {}
            threshold_list = np.arange(0.1, 0.9, 0.1)
            opt_th_only_pos = pd.read_csv(opt_th_only_pos_path)
            confidence_list = []
            for threshold in threshold_list:
                ious = []
                cat_ious = {}

                for i in tqdm(range(len(path_list[:]))):
                    cat = category[i]
                    key = str(path_list[i]) + label_text[i]
                    hmap = hmaps[key]["hmap"]
                    
                    if self.kwargs["margin"]:
                        hmap = self.set_margin(hmap)

                    nan = np.isnan(hmap)
                    gtmask = gtmasks[i]
                    heatmap = norm_heatmap(hmap, nan, mode=1) # [0, 1]
                    if "use_prob" in kwargs and kwargs["use_prob"]:
                        prob = hmaps[key]["prob_sfmax"]
                        confidence = prob
                    else:
                        min_v, max_v = 0, 2
                        clip = np.clip(hmap[~nan], min_v, max_v)
                        confidence = np.max((clip - min_v) / (max_v - min_v))
                    if threshold == threshold_list[0]:
                        confidence_list.append(confidence)
                    if confidence < threshold:
                        mask = np.zeros_like(heatmap)
                    else:
                        mask_th = opt_th_only_pos[cat].values[0]
                        mask = np.where(heatmap > mask_th, 1, 0)
                    iou = compute_iou(gtmask, mask, nan, only_pos=False)

                    ious.append(iou)
                    
                    if cat not in cat_ious:
                        cat_ious[cat] = []

                    cat_ious[cat].append(iou)

                iou_thre_cat.append(dict_mean(cat_ious))

                iou_thre.append(np.nanmean(ious))

                if "iou" not in sep_cat:
                    sep_cat["iou"] = pd.DataFrame(dict_mean(cat_ious, sep=True))
                else:
                    sep_cat["iou"] = pd.concat([sep_cat["iou"], pd.DataFrame(dict_mean(cat_ious, sep=True))], axis=0, ignore_index=True)

            argmax = sep_cat["iou"].values.argmax(axis=0)
            res = pd.DataFrame([threshold_list[argmax], sep_cat["iou"].max().values], columns=sep_cat["iou"].columns)
            res = pd.concat([res, res.mean(axis=1).rename("mean")], axis=1)
            res = res.round(3)
            res.to_csv(opt_th_path, index=False)
            print(res)

            probs_task = {cat: [] for cat in set(category)}
            label_task = {cat: [] for cat in set(category)}
            for i in tqdm(range(len(path_list[:]))):
                key = str(path_list[i]) + label_text[i]
                prob = confidence_list[i]
                probs_task[category[i]].append(prob)
                label_task[category[i]].append(label_list[i])

            cat_auc = {}
            for cat in sorted(set(category)):
                # print("cat:", cat)
                probs = probs_task[cat]
                label = label_task[cat]
                probs = np.array(probs)
                label = np.array(label)
                acc = ACC(np.expand_dims(label, axis=1), np.expand_dims(probs, axis=1))
                auc = AUC(np.expand_dims(label, axis=1), np.expand_dims(probs, axis=1))
                # print(f"acc: {acc}, auc: {auc}")
                # print(f"T: {label.sum()}")
                threshold = res[cat].values[0]
                acc = round(accuracy_score(label, np.where(probs > threshold, 1, 0)), 3)
                prec = round(precision_score(label, np.where(probs > threshold, 1, 0)), 3)
                recall = round(recall_score(label, np.where(probs > threshold, 1, 0)), 3)
                tn, fp, fn, tp = confusion_matrix(label, np.where(probs > threshold, 1, 0)).ravel()
                f1 = round(f1_score(label, np.where(probs > threshold, 1, 0)), 3)
                tnr = round(tn / (tn + fp), 3)
                tpr = round(tp / (tp + fn), 3)
                # print(f"threshold: {threshold}, acc: {acc}, prec: {prec}, recall: {recall}, tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}, f1: {f1}, tnr: {tnr}, tpr: {tpr}")
                cat_auc[cat] = [auc]
            cat_auc["mean"] = [np.mean(list(cat_auc.values()))]
            df_auc = pd.DataFrame(cat_auc)
            df_auc = df_auc.round(3)
            df_auc.to_csv(auc_path, index=False)

def dict_mean(d: dict, sep=False):
    if sep:
        res = {}
        for k, v in d.items():
            res[k] = np.nanmean(v, keepdims=True)
        return res
    res = []
    for k, v in d.items():
        res.append(np.nanmean(v))
        # print(k, len(v))
    return np.nanmean(res)


def prefix(prefix, l):
    return {i: prefix + i for i in l}


def to_metric_df(cat_matric):
    res = pd.DataFrame()
    columns = cat_matric.keys()
    row_n = sum([len(list(i)) for i in cat_matric.values()])
    data = np.zeros((row_n, len(columns)))
    data[data==0] = np.nan
    res = pd.DataFrame(data=data, columns=columns)
    i = 0
    for k, v in cat_matric.items():
        for j in range(len(v)):
            res[k][i] = v[j]
            i += 1
    return res


def bootstrap_metric(df, num_replicates):
    """Create dataframe of bootstrap samples."""
    def single_replicate_performances():
        sample_ids = np.random.choice(len(df), size=len(df), replace=True)
        replicate_performances = {}
        df_replicate = df.iloc[sample_ids]

        for task in df.columns:
            performance = df_replicate[task].mean()
            replicate_performances[task] = performance
        return replicate_performances

    all_performances = []
    for _ in range(num_replicates):
        replicate_performances = single_replicate_performances()
        all_performances.append(replicate_performances)

    df_performances = pd.DataFrame.from_records(all_performances)
    return df_performances


def compute_cis(series, confidence_level):
    sorted_perfs = series.sort_values()
    lower_index = int(confidence_level/2 * len(sorted_perfs)) - 1
    upper_index = int((1 - confidence_level/2) * len(sorted_perfs)) - 1
    lower = sorted_perfs.iloc[lower_index].round(3)
    upper = sorted_perfs.iloc[upper_index].round(3)
    mean = round(sorted_perfs.mean(),3)
    return lower, mean, upper


def create_ci_record(perfs, task):
    lower, mean, upper = compute_cis(perfs, confidence_level = 0.05)
    record = {"name": task,
              "lower": lower,
              "mean": mean,
              "upper": upper}
    return record

def bootci(df, save_dir, metric="iou"):
    bs_df = bootstrap_metric(df, 1000)
    bs_df.to_csv(f'{save_dir}/{metric}_bootstrap_results.csv', index=False)
    records = []
    for task in bs_df.columns:
        records.append(create_ci_record(bs_df[task], task))
    ci_df = pd.DataFrame.from_records(records).sort_values(by='name')
    mean = ci_df.mean(axis=0)
    ci_df = ci_df.append(mean, ignore_index=True)
    ci_df = ci_df.round(3)
    ci_df.to_csv(f'{save_dir}/test_{metric}_summary_results.csv', index=False)
    return ci_df