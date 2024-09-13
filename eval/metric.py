import numpy as np
from eval.utils import norm_heatmap
from sklearn.metrics import f1_score, roc_auc_score


# compute IOU
def compute_iou(gtmask_, premask_, nan, only_pos=True):
    gtmask = gtmask_[~nan]
    premask = premask_[~nan]
    intersection = np.logical_and(gtmask, premask)
    union = np.logical_or(gtmask, premask)
    if only_pos:
        if np.sum(premask) == 0 or np.sum(gtmask) == 0:
            iou_score = np.nan
        else:
            iou_score = np.sum(intersection) / (np.sum(union))
    else:
        if np.sum(union) == 0:
            iou_score = np.nan
        else:
            iou_score = np.sum(intersection) / (np.sum(union))
    return iou_score

def compute_iou1(gtmask_, premask_, nan, only_pos=True):
    gtmask = gtmask_[~nan]
    premask = premask_[~nan]
    intersection = np.logical_and(gtmask, premask)
    union = np.logical_or(gtmask, premask)
    if np.sum(gtmask) == 0:
        iou_score = np.nan
    else:
        iou_score = np.sum(intersection) / (np.sum(union))
    return iou_score

# compute dice
def compute_dice(gtmask_, premask_, nan):
    gtmask = gtmask_[~nan]
    premask = premask_[~nan]
    intersection = np.logical_and(gtmask, premask)
    dice_score = 2 * np.sum(intersection) / (np.sum(gtmask) + np.sum(premask))
    return dice_score


def compute_recall(gtmask_, premask_, nan):
    gtmask = gtmask_[~nan]
    premask = premask_[~nan]
    intersection = np.logical_and(gtmask, premask)
    recall_score = np.sum(intersection) / np.sum(gtmask)
    return recall_score


def compute_precision(gtmask_, premask_, nan):
    gtmask = gtmask_[~nan]
    premask = premask_[~nan]
    intersection = np.logical_and(gtmask, premask)
    precision_score = np.sum(intersection) / np.sum(premask)
    return precision_score


# For CNR, let A and A_ denote the interior and exterior of the bounding box, respectively.
# CNR = |meanA - meanA_| / pow((varA_ + varA), 0.5)
def compute_cnr(gtmask_, heatmap_, nan):
    heatmap = norm_heatmap(heatmap_, nan)
    heatmap_wo_nan = heatmap[~nan]
    gtmask_wo_nan = gtmask_[~nan]
    # assert (gtmask_wo_nan == 1).sum() > 0, 'gtmask_wo_nan == 1 is empty'
    A = heatmap_wo_nan[gtmask_wo_nan == 1]
    A_ = heatmap_wo_nan[gtmask_wo_nan == 0]
    meanA = A.mean()
    meanA_ = A_.mean()
    varA = A.var()
    varA_ = A_.var()
    if varA + varA_ == 0:
        CNR = 0
    else:
        CNR = (meanA - meanA_) / pow((varA + varA_), 0.5)
    return CNR


# compute pointing game
def compute_pg(gtmask_, heatmap_, nan):
    heatmap_wo_nan = heatmap_[~nan]
    gtmask_wo_nan = gtmask_[~nan]

    mask_PG = np.where(heatmap_wo_nan==heatmap_wo_nan.max(), 1, 0)
    pg = 1 if (mask_PG * gtmask_wo_nan).sum() >= 1 else 0
    return pg


def ACC(label, pred):
    pred = np.round(pred)
    return list(np.round(np.sum(label == pred, axis=0) / len(label), 4))


def F1(label, pred):
    pred = np.round(pred)
    return [f1_score(label[:, i], pred[:, i], average='weighted') for i in range(pred.shape[-1])]


def AUC(label, pred):
    rlt = roc_auc_score(label, pred)
    return rlt