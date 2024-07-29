import os
import io
import cv2
import copy
import json
import skimage
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
import pycocotools
import pydicom
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Optional, Union
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import reduce
FONT_MAX = 50
matplotlib.use('Agg')
from eval.box_transfer import box_transfer, box2mask
from eval.io import load_image
from eval.constants import (MIMIC_IMG_DIR, MS_CXR_JSON, RSNA_CSV, 
    RSNA_IMG_DIR, RSNA_MEDKLIP_CSV, PNEUMOTHORAX_ORIGINAL_CSV, 
    PNEUMOTHORAX_IMG_DIR, PNEUMOTHORAX_MAP_CSV, COVID_RURAL_IMG_DIR,
    COVID_RURAL_MASK_DIR, CHEXLOCALIZE_VAL_IMG_DIR, CHEXLOCALIZE_VAL_JSON,
    CHEXLOCALIZE_TEST_IMG_DIR, CHEXLOCALIZE_TEST_JSON, CHESTX_DET10_JSON,
    CHESTX_DET10_IMG_DIR)
TypeArrayImage = Union[np.ndarray, Image.Image]


def norm_heatmap(heatmap_, nan, mode=0):
    # mode: 0 -> "[-1,1]"
    #       1 -> "[0, 1]"
    heatmap = copy.deepcopy(heatmap_)
    heatmap_wo_nan = heatmap[~nan]

    if heatmap_wo_nan.max() - heatmap_wo_nan.min() == 0:
        print(f"heatmap max == min == {heatmap_wo_nan.max()}")
        return heatmap_wo_nan
    
    heatmap_wo_nan = (heatmap_wo_nan - heatmap_wo_nan.min()) / (heatmap_wo_nan.max() - heatmap_wo_nan.min())

    if mode == 0:
        heatmap_wo_nan  = heatmap_wo_nan * 2 - 1 
    heatmap[~nan] = heatmap_wo_nan
    return heatmap


def build_heatmap(real_imgs, attn):
    vis_size = real_imgs.size(2)
    att_sze = attn.size(2)
    attn_max = attn.max(dim=1, keepdim=True)
    attn = torch.cat([attn_max[0], attn], 1)

    attn = attn.view(-1, att_sze, att_sze)
    res = []
    for one_map in attn:
        one_map = skimage.transform.pyramid_expand(
            one_map, sigma=20, upscale=vis_size / att_sze
            )
        if (one_map.max() - one_map.min()) == 0:
            one_map = one_map * 0
            print(f"one_map max == min == {one_map.max()}")
        else:
            one_map = (one_map - one_map.min()) / (one_map.max() - one_map.min())
            one_map *= 255
        res.append(one_map)
    return res



def get_heatmap(
        latent_img,
        latent_report,
        softmax: bool = True,
        temperature: float = 0.2,
        resize: bool = True,
        mode: str = "area",  # [bilinear, area]):  
        ):
        # latent_img: b, m, w, h -> b, w*h, m
        # latent_report: b, m -> b, m, 1
        latent_img = latent_img.permute(0, 2, 3, 1)
        latent_img = latent_img.reshape(latent_img.shape[0], -1, latent_img.shape[-1])
        latent_report = latent_report.unsqueeze(2)

        # latent_img = F.normalize(latent_img, dim=-1)
        # latent_report = F.normalize(latent_report, dim=-2)
        sim = torch.bmm(latent_img, latent_report)  
        if softmax:
            sim = F.softmax(sim * temperature, dim=1)
        num_patch = int(np.sqrt(sim.shape[1]))
        sim = sim.reshape(-1, 1, num_patch, num_patch)

        if resize:
            sim = F.interpolate(sim, size=224, mode=mode)
            # sim = F.interpolate(sim, size=224, mode="area")

        return sim


def load_data(dataset, **kwargs):
    if dataset == "MS_CXR":
        return load_ms_cxr(**kwargs)
    if dataset == "MS_CXR_CLS":
        return load_ms_cxr(use_cxr_text=False, **kwargs)
    if dataset == "MS_CXR_ERR":
        return load_ms_cxr(use_cxr_text=True, use_error_description=True, **kwargs)
    elif dataset == "RSNA":
        return load_rsna(**kwargs)
    elif dataset == "RSNA_MEDKLIP":
        return load_rsna_medklip(**kwargs)
    elif dataset == "SIIM_ACR":
        return load_siim_acr(**kwargs)
    elif dataset == "COVID_RURAL_0":
        return load_covid_rural(mode=0, **kwargs)
    elif dataset == "COVID_RURAL_1":
        return load_covid_rural(mode=1, **kwargs)
    elif dataset == "COVID_RURAL_2":
        return load_covid_rural(mode=2, **kwargs)
    elif dataset == "CHEXLOCALIZE":
        return load_chexlocalize(**kwargs)
    elif dataset == "CHESTX_DET10":
        return load_chestx_det10(**kwargs)
    else:
        raise NotImplementedError


def load_chestx_det10(**kwargs):
    path_list = []
    label_text_list = []
    gtmasks_list = []
    boxes_list = []
    category_list = []
    annos = json.load(open(CHESTX_DET10_JSON))
    size = 224

    for entry in annos:
        syms, boxes = entry['syms'], entry['boxes']
        file_name = entry['file_name']
        for idx, (sym, box) in enumerate(zip(syms, boxes)):
            path_list.append(CHESTX_DET10_IMG_DIR / file_name)
            x, y, w, h = box[0], box[1], box[2]-box[0], box[3]-box[1]
            bbox = (x, y, w, h)
            tbox = box_transfer(bbox, 1024, 1024, size)
            mask = box2mask(tbox, size, size)
            label_text = f"Findings suggesting {sym}."
            gtmasks_list.append(mask)
            boxes_list.append([tbox])
            category_list.append(sym)
            label_text_list.append(label_text)

    data = pd.DataFrame()
    data["path"] = path_list
    data["label_text"] = label_text_list
    data["gtmasks"] = gtmasks_list
    data["boxes"] = boxes_list
    data["category"] = category_list

    return data

def load_ms_cxr(merge=True, use_cxr_text=True, use_error_description=False, **kwargs):
    # loading data
    print("loading data...")
    if merge == True:
        data = merge_annotation(MS_CXR_JSON, use_cxr_text=use_cxr_text, use_error_description=use_error_description)
    else:
        data = get_annotation(MS_CXR_JSON)

    data["path"] = list(map(lambda x: MIMIC_IMG_DIR/x.replace("files/", ""), data["path"]))

    return data


def load_rsna(**kwargs):
    #     # loading data
    print("loading data...")
    size = 224  # image size, setting by default
    path_to_file = RSNA_CSV
    path_images = RSNA_IMG_DIR
    df = pd.read_csv(path_to_file)
    patientId = sorted(list(set(df["patientId"].values.tolist())))

    path_list = []
    label_text_list = []
    gtmasks_list = []
    boxes_list = []
    category_list = []

    for pid in patientId:
        pat_info = df[(df["patientId"] == pid)&(df["Target"] == 1)]
        mask_dct = {}
        bbox_dct = {}
        cats_dct = {}
        path = path_images / (pid + '.png')
        if pat_info.size == 0:
            continue
        for idx, row in pat_info.iterrows():
            label_text = 'Findings suggesting pneumonia.'
            category_name = 'Pneumonia'
            bbox = (row["x"], row["y"], row["width"], row["height"])
            tbox = box_transfer(bbox, 1024, 1024, size)
            mask = box2mask(tbox, size, size)
            if label_text not in mask_dct:
                mask_dct[label_text] = mask
                bbox_dct[label_text] = [tbox]
                cats_dct[label_text] = category_name
            else:
                mask_dct[label_text] += mask
                bbox_dct[label_text].append(tbox)
                cats_dct[label_text] = category_name

        for k, v in mask_dct.items():
            path_list.append(path)
            gtmasks_list.append(v)
            boxes_list.append(bbox_dct[k])
            category_list.append(cats_dct[k])
            label_text_list.append(k)

    data = pd.DataFrame()
    data["path"] = path_list
    data["label_text"] = label_text_list
    data["gtmasks"] = gtmasks_list
    data["boxes"] = boxes_list
    data["category"] = category_list
    return data


def load_rsna_medklip(**kwargs):
    #     # loading data
    print("loading data...")
    size = 224  # image size, setting by default
    path_to_file = RSNA_MEDKLIP_CSV
    path_images = RSNA_IMG_DIR
    df = pd.read_csv(path_to_file)
    df = df[df["classes"] == 1]

    label_text_list = ['Findings suggesting pneumonia.'] * len(df)
    category_list = ['Pneumonia'] * len(df)
    path_list = []
    gtmasks_list = []
    boxes_list = []
    save_path = os.path.join("./dataset", f"rsna_medklip.npy")
    if os.path.exists(save_path):
        data = pd.DataFrame(np.load(save_path, allow_pickle=True).item())
        return data

    for idx, row in df.iterrows():
        pid = row["ID"]
        path = path_images / (pid + '.png')
        path_list.append(path)
        boxes = []
        mask = None
        for box_str in row["boxes"].split("|"):
            bbox = list(map(int, map(float, box_str.split(";"))))
            tbox = box_transfer(bbox, 1024, 1024, size)
            boxes.append(tbox)
            if mask is None:
                mask = box2mask(tbox, size, size)
            else:
                mask += box2mask(tbox, size, size)
        gtmasks_list.append(mask)
        boxes_list.append(boxes)

    data = pd.DataFrame()
    data["path"] = path_list
    data["label_text"] = label_text_list
    data["gtmasks"] = gtmasks_list
    data["boxes"] = boxes_list
    data["category"] = category_list
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, data.to_dict())
    return data


def load_siim_acr(**kwargs):
    size = 224
    df = pd.read_csv(PNEUMOTHORAX_ORIGINAL_CSV)
    df = df[df[" EncodedPixels"] != " -1"]
    # get image paths

    img_paths = {}
    for subdir, dirs, files in os.walk(PNEUMOTHORAX_IMG_DIR):
        for f in files:
            if "dcm" in f:
                # remove dcm
                file_id = f[:-4]
                img_paths[file_id] = Path(os.path.join(subdir, f))

    patientId = sorted(list(set(df["ImageId"].values.tolist())))[:3]
    # patientId = [patientId[526], patientId[1192], patientId[1543], patientId[1772]]
    label_text_list = []
    category_list = []
    path_list = []
    gtmasks_list = []
    boxes_list = []

    for pid in tqdm(patientId):
        pat_info = df.groupby("ImageId").get_group(pid)
        
        rle_list = pat_info[" EncodedPixels"].tolist()
        mask = np.zeros([1024, 1024])
        if rle_list[0] != " -1":
            for rle in rle_list:
                mask += rle2mask(
                    rle, 1024, 1024
                )
        mask = (mask >= 1).astype("float32")
        mask = _resize_img(mask, 224)
        mask = (mask >= 1e-5).astype("uint8")

        path = img_paths[pid]
        path_list.append(path)
        gtmasks_list.append(mask)
        boxes_list.append(None)
        category_list.append("Pneumothorax")
        label_text_list.append("Findings suggesting pneumothorax.")
            
    data = pd.DataFrame()
    data["path"] = path_list
    data["label_text"] = label_text_list
    data["gtmasks"] = gtmasks_list
    data["boxes"] = boxes_list
    data["category"] = category_list
    return data


def load_covid_rural(mode=1, **kwargs):
    label_text_list = []
    category_list = []
    path_list = []
    gtmasks_list = []
    boxes_list = []
    save_path = os.path.join("./dataset", f"covid_rural_{mode}.npy")
    if os.path.exists(save_path):
        data = pd.DataFrame(np.load(save_path, allow_pickle=True).item())
        return data

    img_list = sorted([i.rstrip(".jpg") for i in os.listdir(COVID_RURAL_IMG_DIR)
                if i.endswith(".jpg")])
    mask_list = [i.rstrip(".png") for i in os.listdir(COVID_RURAL_MASK_DIR)
                 if i.endswith(".png")]
    
    for img in tqdm(img_list):
        if img not in mask_list:
            continue
        path = os.path.join(COVID_RURAL_IMG_DIR, img + ".jpg")
        mask_path = os.path.join(COVID_RURAL_MASK_DIR, img + ".png")
        mask = Image.open(mask_path)
        mask = np.array(mask).astype("float32")
        if mask.sum() == 0:
            # print(f"{img} mask is empty")
            continue
        mask = _resize_img(mask, 224)
        mask = (mask >= 1e-5).astype("uint8")

        path_list.append(Path(path))
        gtmasks_list.append(mask)
        boxes_list.append(None)
        category_list.append("COVID-19")
        if mode == 0:
            label_text_list.append("Findings suggesting COVID-19.")
        elif mode == 1:
            label_text_list.append("Findings suggesting pneumonia.")
        elif mode == 2:
            label_text_list.append("Similar to pneumonia.")
        

    data = pd.DataFrame()
    data["path"] = path_list
    data["label_text"] = label_text_list
    data["gtmasks"] = gtmasks_list
    data["boxes"] = boxes_list
    data["category"] = category_list
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, data.to_dict())
    return data


def load_chexlocalize(**kwargs):
    LOCALIZATION_TASKS =  ["Enlarged Cardiomediastinum",
                       "Cardiomegaly",
                       "Lung Lesion",
                       "Airspace Opacity",
                       "Edema",
                       "Consolidation",
                       "Atelectasis",
                       "Pneumothorax",
                       "Pleural Effusion",
                       "Support Devices"]
    suffix = "_" + kwargs["split"] if "split" in kwargs else "_test"
    save_path = os.path.join("./dataset", f"chexlocalize{suffix}.npy")
    if os.path.exists(save_path):
        data = pd.DataFrame(np.load(save_path, allow_pickle=True).item())
        return data
    
    label_text_list = []
    category_list = []
    path_list = []
    gtmasks_list = []
    boxes_list = []
    label_list = []
    if "split" in kwargs and kwargs["split"] == "val":
        jsonfile = CHEXLOCALIZE_VAL_JSON
        img_dir = CHEXLOCALIZE_VAL_IMG_DIR
    else:
        jsonfile = CHEXLOCALIZE_TEST_JSON
        img_dir = CHEXLOCALIZE_TEST_IMG_DIR
        
    with open(jsonfile) as f:
        gt_dict = json.load(f)
    cxr_ids = gt_dict.keys()
    cxr_ids = sorted(list(cxr_ids))
    neg = 0
    pos = 0
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith(".jpg"):
                path = os.path.join(root, file)
                
                cxr_id = "_".join(path.split("/")[-3:]).rstrip(".jpg")
                if cxr_id not in cxr_ids:
                    for category in LOCALIZATION_TASKS:
                        label_text_list.append(f"Findings suggesting {category}.")
                        category_list.append(category)
                        gtmasks_list.append(np.zeros((224, 224)))
                        boxes_list.append(None)
                        path_list.append(Path(path))
                        neg += 1
                        label_list.append(0)
                else:
                    for category, elem in gt_dict[cxr_id].items():
                        gt_mask = pycocotools.mask.decode(elem)
                        if gt_mask.sum() == 0:
                            gt_mask_ = np.zeros((224, 224))
                            neg += 1
                            label_list.append(0)
                        else:
                            gt_mask_ = _resize_img(gt_mask, 224)
                            gt_mask_ = (gt_mask_ >= 1e-5).astype("uint8")
                            pos += 1
                            label_list.append(1)
                        gtmasks_list.append(gt_mask_)
                        boxes_list.append(None)
                        category_list.append(category)
                        label_text_list.append(f"Findings suggesting {category}.")
                        path_list.append(Path(path))
                        
                        
    print(f"neg: {neg}, pos: {pos}", "total: ", neg + pos)
    data = pd.DataFrame()
    data["path"] = path_list
    data["label_text"] = label_text_list
    data["gtmasks"] = gtmasks_list
    data["boxes"] = boxes_list
    data["category"] = category_list
    data["label"] = label_list
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, data.to_dict())
    return data

def rle2mask(rle, width, height):
        """Run length encoding to segmentation mask"""

        mask = np.zeros(width * height)
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]
        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position:current_position + lengths[index]] = 1
            current_position += lengths[index]

        return mask.reshape(width, height).T


def _resize_img(img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img


def merge_annotation(path_to_json, scale=224, use_cxr_text=True, use_error_description=False):
    coco = COCO(annotation_file=path_to_json)
    cats = coco.cats
    merged = {}
    merged["path"] = []
    merged["gtmasks"] = []
    merged["label_text"] = []
    merged["boxes"] = []
    merged["category"] = []

    for img_id, anns in coco.imgToAnns.items():
        img = coco.loadImgs(img_id)[0]
        path = img["path"]
        mask_dct = {}
        bbox_dct = {}
        cats_dct = {}
        for ann in anns:
            bbox = ann["bbox"]
            w = ann["width"]
            h = ann["height"]
            category_id = ann["category_id"]
            tbox = box_transfer(bbox, w, h, scale)
            mask = box2mask(tbox, scale, scale)
            if use_cxr_text:
                category = cats[category_id]["name"]
                label_text = ann["label_text"].lower()
                if use_error_description:
                    if "right" in label_text:
                        label_text = label_text.replace("right", "left")
                    elif "left" in label_text:
                        label_text = label_text.replace("left", "right")
                    # if "small" in label_text:
                    #     label_text = label_text.replace("small", "large")
                    # elif "large" in label_text:
                    #     label_text = label_text.replace("large", "small")
                if label_text not in mask_dct:
                    mask_dct[label_text] = mask
                    bbox_dct[label_text] = [tbox]
                    cats_dct[label_text] = category
                else:
                    mask_dct[label_text] += mask
                    bbox_dct[label_text].append(tbox)
                    cats_dct[label_text] = category
            else:
                category = cats[category_id]["name"]
                label_text = f"Findings suggesting {category}."
                # label_text = f"{category}"
                if label_text not in mask_dct:
                    mask_dct[label_text] = mask
                    bbox_dct[label_text] = [tbox]
                    cats_dct[label_text] = category
                else:
                    mask_dct[label_text] += mask
                    bbox_dct[label_text].append(tbox)
                    cats_dct[label_text] = category

        for k, v in mask_dct.items():
            merged["path"].append(path)
            merged["gtmasks"].append(v)
            merged["label_text"].append(k)
            merged["boxes"].append(bbox_dct[k])
            merged["category"].append(cats_dct[k])

    return merged


def get_annotation(path_to_json, scale=224):
    coco = COCO(annotation_file=path_to_json)
    res = {}
    res["path"] = []
    res["gtmasks"] = []
    res["label_text"] = []
    res["boxes"] = []
    res["category"] = []

    for img_id, anns in coco.imgToAnns.items():
        img = coco.loadImgs(img_id)[0]
        path = img["path"]
        for ann in anns:
            bbox = ann["bbox"]
            w = ann["width"]
            h = ann["height"]
            label_text = ann["label_text"]
            category_id = ann["category_id"]
            tbox = box_transfer(bbox, w, h, scale)
            mask = box2mask(tbox, scale, scale)
            category = coco.cats[category_id]["name"]
            res["path"].append(path)
            res["gtmasks"].append(mask)
            res["label_text"].append(label_text)
            res["boxes"].append([tbox])
            res["category"].append(category)
            
    return res


def toImage(img):
    if isinstance(img, Image.Image):
        return img
    elif isinstance(img, np.ndarray):
        return Image.fromarray(img).convert("RGB")
    else:
        return Image.open(io.BytesIO(img)).convert("RGB")


def blend_images(ori, heatmap, opacity=0.5):
    original_image = toImage(ori)
    heatmap_image = toImage(heatmap)
    blended_image = Image.blend(original_image, heatmap_image, opacity)
    return blended_image



# draw n images use plt
def draw_n_images(images, sub_titles, title, save_path):
    n = len(images)
    fig = plt.figure(figsize=(n*5, 5))
    plt.axis('off')
    plt.title(title)
    for i in range(n):
        ax = fig.add_subplot(1, n, i+1)
        ax.imshow(images[i])
        ax.set_title(sub_titles[i])
        ax.axis('off')
    save_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)


# draw n*m images use plt
def draw_n_m_images(images, sub_titles, title, save_path):
    n = len(images)
    m = len(images[0])
    fig = plt.figure(figsize=(m*5, n*5))
    plt.title(title)
    for i in range(n):
        for j in range(m):
            ax = fig.add_subplot(n, m, i*m+j+1)
            ax.imshow(images[i][j])
            ax.set_title(sub_titles[i][j])
            ax.axis("off")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)


def _plot_bounding_boxes(
    ax: plt.Axes, bboxes: List[Tuple[float, float, float, float]], linewidth: float = 1.5, alpha: float = 0.45
) -> None:
    """
    Plot bounding boxes on an existing axes object.

    :param ax: The axes object to plot the bounding boxes on.
    :param bboxes: A list of bounding box coordinates as (x, y, width, height) tuples.
    :param linewidth: Optional line width for the bounding box edges (default is 2).
    :param alpha: Optional opacity for the bounding box edges (default is 1.0).
    """
    for bbox in bboxes:
        x, y, width, height = bbox
        rect = patches.Rectangle(
            (x, y), width, height, linewidth=linewidth, edgecolor='k', facecolor='none', linestyle='--', alpha=alpha
        )
        ax.add_patch(rect)


def _plot_isolines(
    heatmap: np.ndarray,
    axis: plt.Axes,
    title: Optional[str] = None,
    colormap: str = "RdBu_r",
    step: float = 1,
) -> None:
    """Plot an image and overlay heatmap isolines on it.

    :param image: Input image.
    :param heatmap: Heatmap of the same size as the image.
    :param axis: Axis to plot the image on.
    :param title: Title used for the axis.
    :param colormap: Name of the Matplotlib colormap used for the isolines.
    :param step: Step size between the isolines levels. The levels are in :math:`(0, 1]`.
        For example, a step size of 0.25 will result in isolines levels of 0.25, 0.5, 0.75 and 1.
    """

    levels = _get_isolines_levels(step)
    contours = axis.contour(
        heatmap,
        cmap=None,
        vmin=-1,
        vmax=1,
        levels=levels,
        colors=[(50/255,50/255,50/255)],
        linestyles=['--']
    )
    # axis.clabel(contours, inline=False, fontsize=10)
    axis.axis("off")
    if title is not None:
        axis.set_title(title)

def _get_isolines_levels(step_size: float) -> np.ndarray:
    num_steps = np.floor(round(1 / step_size)).astype(int)
    levels = np.linspace(step_size, 1, num_steps)
    return levels

def _plot_heatmap(
    image: TypeArrayImage,
    heatmap: np.ndarray,
    figure: plt.Figure,
    axis: plt.Axes,
    colormap: str = "RdBu_r",
    title: Optional[str] = None,
    alpha: float = 0.5,
    fontsize: int = 15,
    **kwargs,
) -> None:
    """Plot a heatmap overlaid on an image.

    :param image: Input image.
    :param heatmap: Input heatmap of the same size as the image.
    :param figure: Figure to plot the images on.
    :param axis: Axis to plot the images on.
    :param colormap: Name of the Matplotlib colormap for the heatmap.
    :param title: Title used for the axis.
    :param alpha: Heatmap opacity. Must be in :math:`[0, 1]`.
    """
    axis.imshow(image)
    axes_image = axis.matshow(heatmap, alpha=alpha, cmap=colormap, vmin=-1, vmax=1)
    # https://www.geeksforgeeks.org/how-to-change-matplotlib-color-bar-size-in-python/
    # divider = make_axes_locatable(axis)
    # colorbar_axes = divider.append_axes("right", size="10%", pad=0.1)
    # colorbar = figure.colorbar(axes_image, cax=colorbar_axes)
    # # https://stackoverflow.com/a/50671487/3956024
    # colorbar.ax.tick_params(pad=35)
    # plt.setp(colorbar.ax.get_yticklabels(), ha="right")
    axis.axis("off")
    if title is not None:
        axis.set_title(title, fontsize=fontsize)


def _plot_mask(
    mask: np.ndarray,
    axis: plt.Axes,
    **kwargs,
) -> None:
    axis.matshow(mask, cmap="RdBu_r", alpha=0.5, vmin=-1, vmax=1)
    axis.axis("off")

def _plot_image(
    image: TypeArrayImage,
    axis: plt.Axes,
    title: Optional[str] = None,
) -> None:
    """Plot an image on a given axis, deleting the axis ticks and axis labels.

    :param image: Input image.
    :param axis: Axis to plot the image on.
    :param title: Title used for the axis.
    """
    axis.imshow(image)
    axis.axis("off")
    if title is not None:
        axis.set_title(title)


def biovil_show(image_path, similarity_map, bboxes, save_path="", title="Similarity heatmap", gtmask=None, **kwargs):

    figsize = kwargs["figsize"] if "figsize" in kwargs else (6, 6)
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    if image_path.suffix == ".dcm":
        image = read_from_dicom(image_path)
    elif image_path.suffix == ".png" or image_path.suffix == ".jpg":
        image = load_image(image_path)
    else:
        raise ValueError("Unsupported image format")
    if "resize" in kwargs:
        # image = image.resize((kwargs["resize"], kwargs["resize"]))
        image = _resize_img(np.array(image), kwargs["resize"])
        image = Image.fromarray(image)
    
    image = image.convert("RGB")
    _plot_heatmap(image, similarity_map, figure=fig, axis=axes, title=title, **kwargs)
    # _plot_isolines(similarity_map, axis=axes)

    if gtmask is not None:
        _plot_isolines(gtmask, axis=axes)
        # _plot_mask(gtmask, axis=axes)
    if bboxes is not None:
        _plot_bounding_boxes(ax=axes, bboxes=bboxes)

    if save_path:
        save_path = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, transparent=True)
        plt.close()


def biovil_show_crop_img(image_path, similarity_map, bboxes, save_path="", title=None, gtmask=None, crop_size=38, **kwargs):

    figsize = kwargs["figsize"] if "figsize" in kwargs else (6, 6)
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    if image_path.suffix == ".dcm":
        image = read_from_dicom(image_path)
    elif image_path.suffix == ".png" or image_path.suffix == ".jpg":
        image = load_image(image_path)
    else:
        raise ValueError("Unsupported image format")
    if "resize" in kwargs:
        # image = image.resize((kwargs["resize"], kwargs["resize"]))
        image = _resize_img(np.array(image), kwargs["resize"])
        # image = image[crop_size:-crop_size, crop_size:-crop_size]
        image = Image.fromarray(image)
    
    image = image.convert("RGB")
    _plot_image(image, axis=axes, title=None)

    if save_path:
        save_path = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img = plt_to_img(fig)
        img = img.crop((crop_size, crop_size, img.size[0]-crop_size, img.size[1]-crop_size))
        img.save(save_path)

def biovil_show_crop_heat(image_path, similarity_map, bboxes, save_path="", title=None, gtmask=None, crop_size=38, **kwargs):

    figsize = kwargs["figsize"] if "figsize" in kwargs else (6, 6)
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    if image_path.suffix == ".dcm":
        image = read_from_dicom(image_path)
    elif image_path.suffix == ".png" or image_path.suffix == ".jpg":
        image = load_image(image_path)
    else:
        raise ValueError("Unsupported image format")
    if "resize" in kwargs:
        # image = image.resize((kwargs["resize"], kwargs["resize"]))
        image = _resize_img(np.array(image), kwargs["resize"])
        # image = image[crop_size:-crop_size, crop_size:-crop_size]
        image = Image.fromarray(image)
    
    image = image.convert("RGB")
    # similarity_map = similarity_map[crop_size:-crop_size, crop_size:-crop_size]
    _plot_heatmap(image, similarity_map, figure=fig, axis=axes, title=title, **kwargs)
    # _plot_isolines(similarity_map, axis=axes)

    if gtmask is not None:
        _plot_isolines(gtmask, axis=axes)
        # _plot_mask(gtmask, axis=axes)
    if bboxes is not None:
        _plot_bounding_boxes(ax=axes, bboxes=bboxes)
    if save_path:
        save_path = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img = plt_to_img(fig)
        img = img.crop((crop_size, crop_size, img.size[0]-crop_size, img.size[1]-crop_size))
        img.save(save_path)

def plt_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    return img


def biovil_show2(image_path, similarity_map, bboxes, save_path="", title="Similarity heatmap", gtmask=None, **kwargs):

    figsize = kwargs["figsize"] if "figsize" in kwargs else (12, 6)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    if image_path.suffix == ".dcm":
        image = read_from_dicom(image_path)
    elif image_path.suffix == ".png" or image_path.suffix == ".jpg":
        image = load_image(image_path)
    else:
        raise ValueError("Unsupported image format")
    if "resize" in kwargs:
        image = _resize_img(np.array(image), kwargs["resize"])
        image = Image.fromarray(image)
    
    image = image.convert("RGB")
    _plot_image(image, axis=axes[0], title="Input image")
    _plot_heatmap(image, similarity_map, figure=fig, axis=axes[1], title=title, **kwargs)
    # _plot_isolines(similarity_map, axis=axes)

    if gtmask is not None:
        _plot_isolines(gtmask, axis=axes[1])
        # _plot_mask(gtmask, axis=axes)
    if bboxes is not None:
        _plot_bounding_boxes(ax=axes[1], bboxes=bboxes)

    if save_path:
        save_path = os.path.abspath(save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, transparent=True)
        plt.close()

def read_from_dicom(img_path):

    dcm = pydicom.read_file(img_path, force=True)
    x = dcm.pixel_array
    x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))

    if dcm.PhotometricInterpretation == "MONOCHROME1":
        x = cv2.bitwise_not(x)

    img = Image.fromarray(x)
    return img


def sort_result(ckpt_dir, dataset, file_name="metric.csv"):
    ckpt_dir = os.path.join(ckpt_dir, dataset)
    if not os.path.exists(ckpt_dir):
        return False
    
    path_list = sorted([os.path.join(ckpt_dir, i) for i in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, i))])
    ckpt_list = [i.split("/")[-1] for i in path_list if i.split("/")[-1]]
    res = pd.DataFrame()
    res["ckpt"] = ckpt_list
    mean_data = []
    for i in path_list:
        path = os.path.join(i, file_name)
        df = pd.read_csv(path)
        mean_data.append(df[df["threshold"]=="mean"])
    res = pd.concat([res, pd.concat(mean_data, axis=0, ignore_index=True)], axis=1)
    
    sort_by_iou = res.sort_values(by="iou_cat", ascending=False)
    sort_by_cnr = res.sort_values(by="cnr_cat", ascending=False)

    top_k = 3 if len(sort_by_iou) >= 3 else len(sort_by_iou)
    mean_iou = sort_by_iou.iloc[:top_k].mean()
    mean_cnr = sort_by_cnr.iloc[:top_k].mean()

    sort_by_iou.loc[len(sort_by_iou)] = mean_iou
    sort_by_cnr.loc[len(sort_by_cnr)] = mean_cnr
    sort_by_iou = sort_by_iou.round(3)
    sort_by_cnr = sort_by_cnr.round(3)
    suffix = file_name.replace(".csv", "")
    sort_by_iou.to_csv(os.path.join(ckpt_dir, f"sort_by_iou_from_{suffix}.csv"), index=False)
    sort_by_cnr.to_csv(os.path.join(ckpt_dir, f"sort_by_cnr_from_{suffix}.csv"), index=False)
    print(sort_by_iou.loc[:, ["threshold", "iou", "cnr", "dice", "iou_cat", "cnr_cat", "dice_cat"]])
    print(sort_by_cnr.loc[:, ["threshold", "iou", "cnr", "dice", "iou_cat", "cnr_cat", "dice_cat"]])


def sort_result_chex(ckpt_dir, dataset):
    ckpt_dir = os.path.join(ckpt_dir, dataset)
    if not os.path.exists(ckpt_dir):
        return False
    path_list = sorted([os.path.join(ckpt_dir, i) for i in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, i))])
    ckpt_list = [i.split("/")[-1] for i in path_list if i.split("/")[-1]]
    csv_list = [i for i in os.listdir(path_list[0]) if i.endswith(".csv")]
    res_list = []
    merge_cols = ["ckpt"]
    for csv_name in csv_list:
        res = pd.DataFrame()
        res["ckpt"] = ckpt_list
        total_data = []
        for i in path_list:
            path = os.path.join(i, csv_name)
            df = pd.read_csv(path)
            total_data.append(pd.DataFrame(df.iloc[-1]).T)
        res = pd.concat([res, pd.concat(total_data, axis=0, ignore_index=True)], axis=1)
        if "iou_cat" in res.columns:
            res = res.sort_values(by="iou_cat", ascending=False)
            res_m = res.rename(columns={"iou_cat": csv_name.replace(".csv", "")})
            merge_cols.append(csv_name.replace(".csv", ""))
        elif "mean" in res.columns:
            res = res.sort_values(by="mean", ascending=False)
            res_m = res.rename(columns={"mean": csv_name.replace(".csv", "")})
            merge_cols.append(csv_name.replace(".csv", ""))
        res = res.round(3)
        res.to_csv(os.path.join(ckpt_dir, csv_name), index=False)
        res_list.append(res_m)

    merged_df = reduce(lambda left, right: pd.merge(left, right, on='ckpt'), res_list)
    merged_df = merged_df[merge_cols]
    merged_df.to_csv(os.path.join(ckpt_dir, "merged.csv"), index=False)
