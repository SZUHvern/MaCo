
import numpy as np
from typing import Dict, List
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import matthews_corrcoef, jaccard_score
from sklearn.metrics import confusion_matrix



def compute_acc(pred: np.ndarray, label: np.ndarray) -> float:
    """ Accuracy for single label classification.
    Input:
        - pred (np.ndarray): prediction of shape [batch_size, num_class]
        - label (np.ndarray): onehot label of shape [batch_size, num_class]
    Return:
        - (float) average accuracy 
    
    Code was given by Weijian Huang.
    """
    if pred.shape[-1] == 1:
        return np.mean((pred > 0.5).astype(np.long) == label.astype(np.long))
    return np.mean(pred.argmax(1) == label.argmax(1))


def compute_acc_multi(
        pred: np.ndarray,
        label: np.ndarray,
        threshold: np.ndarray = None
    ) -> List[float]:
    """ Accuracy for multi label classification.
    Input:
        - pred (np.ndarray): prediction of shape [batch_size, num_class]
        - label (np.ndarray): onehot label of shape [batch_size, num_class]
    Return:
        - (List[float]) average accuracy of each class

    Code was given by Weijian Huang.
    """
    if threshold is None:
        pred = np.round(pred)
    else:
        pred = (pred > threshold).astype(np.int64)
    # return list(np.round(np.sum(label == pred, axis=0) / len(label), 4))
    return list(np.sum(label == pred, axis=0) / len(label))


def compute_f1(
        pred: np.ndarray, 
        label: np.ndarray, 
        threshold: np.ndarray = None
    ) -> float:
    """ F1 score for classification task.
    Input:
        - pred (np.ndarray): prediction of shape [batch_size, num_class]
        - label (np.ndarray): onehot label of shape [batch_size, num_class]
    Return:
        - (float) f1 score

    Code was given by Weijian Huang.
    """
    if threshold is None:
        pred = np.round(pred)
    else:
        pred = (pred > threshold).astype(np.int64)
    return [f1_score(label[:, i], pred[:, i], average='weighted') for i in range(pred.shape[-1])]


def compute_auc(pred: np.ndarray, label: np.ndarray) -> float:
    """ ROCAUC score for classification task.
    Input:
        - pred (np.ndarray): prediction of shape [batch_size, num_class]
        - label (np.ndarray): onehot label of shape [batch_size, num_class]
    Return:
        - (float) averaged roc auc score

    Code was given by Weijian Huang.
    """
    score = roc_auc_score(label, pred, average=None)
    score = roc_auc_score(label, pred, average='macro')
    return score


def compute_point_game(
        pred_map: np.ndarray, 
        seg_map: np.ndarray,
        label: np.ndarray,
    ) -> float:
    """ point game score for segmentation task.
    Input:
        - pred (np.ndarray): prediction of shape [batch_size, height, width]
        - seg_map (np.ndarray): binary mask which consist of zero or one or multiple 
                                RoIs [batch_size, height, width]
        - label (np.ndarray): binary classification labels of shape [batch_size], 
                              which is the corresponding classification label
    Return:
        - (float) averaged recall score

    Point game is to evaluate weather the maximum prob is in the RoI or Not
    Reference: score_all() function in MedKLIP/Sample_Zero-Shot_Grounding_RSNA/test.py
    """

    total_num = label.sum()
    # only compute on positive samples, because some samples don't have corresponding targets
    mask = (label == 1)
    # extract positive samples from batch and reshape to (total_num, height*width)
    seg_map = seg_map[mask,:,:].reshape(total_num,-1)
    pred_map = pred_map[mask,:,:].reshape(total_num,-1)

    # get max values of each sample
    max_value = pred_map.max(1)

    # we use a for loop here to ensure the stability when there are two or more 
    # max_points with same value in a sample. you can re-write it in a parallel manner
    point_score = 0
    for i in range(total_num):
        # find out points with maximum probability
        max_points = (pred_map[i] == max_value[i])
        # if max_points in RoIs, point_score += 1 
        point_score += int(((max_points * seg_map[i]).sum()) > 0)

    return point_score / total_num


def compute_recall(
        pred: np.ndarray, 
        label: np.ndarray, 
        threshold: float = 0.5,
    ) -> float:
    """ recall score for segmentation task.
    Input:
        - pred (np.ndarray): prediction of shape [batch_size, height, width]
        - label (np.ndarray): binary masks of shape [batch_size, height, width]
        - threshold (float): a threshold to get binary prediction, default is 0.5
    Return:
        - (float) averaged recall score

    Reference: sklearn.metrics.recall_score
    """
    assert label.max() <= 1
    pred = (pred > threshold).flatten()
    label = label.flatten()
    return recall_score(label, pred)


def compute_precision(
        pred: np.ndarray, 
        label: np.ndarray, 
        threshold: float = 0.5,
    ) -> float:
    """ precision score for segmentation task.
    Input:
        - pred (np.ndarray): prediction of shape [batch_size, height, width] or [batch_size, 1]
        - label (np.ndarray): binary masks of shape [batch_size, height, width] or [batch_size, 1]
        - threshold (float): a threshold to get binary prediction, default is 0.5
    Return:
        - (float) averaged precision score

    Reference: sklearn.metrics.precision_score
    """
    assert label.max() <= 1
    pred = (pred > threshold).flatten()
    label = label.flatten()
    return precision_score(label, pred)


def compute_tnr(pred: np.ndarray, label: np.ndarray, threshold: float = 0.5) -> float:
    """ tnr score for classification task.
    Input:
        - label (np.ndarray): labels of shape [batch_size, 1]
        - pred (np.ndarray): predictions of shape [batch_size, 1]
    Return:
        - (float) averaged tnr score
    """
    assert label.max() <= 1
    pred = (pred > threshold).flatten()
    label = label.flatten()

    cm = confusion_matrix(label, pred)
    tn = cm[0, 0]
    fp = cm[0, 1]
    return tn / (tn + fp)


def compute_iou(
        pred: np.ndarray, 
        label: np.ndarray, 
        threshold: float = 0.5,
        eps: int = 1,
    ) -> float:
    """ IoU score for segmentation tasks. 
    Input:
        - pred (np.ndarray): prediction of shape [batch_size, height, width]
        - label (np.ndarray): binary masks of shape [batch_size, height, width]
        - threshold (float): a threshold to get binary prediction, default is 0.5
    Return:
        - (float) dice score
    
    Reference: mIoU() function in MedKLIP/Sample_Finetuning_SIIMACR/I2_segmentation/metric.py
    """
    assert label.max() <= 1

    batch_size = len(pred)
    # pred = (pred > threshold).astype(np.float64)
    # label = (label > 0.5).astype(np.float64)
    pred = pred > threshold
    label = label > 0.5

    pred = pred.reshape(batch_size, -1)
    label = label.reshape(batch_size, -1)
    intersection = (label * pred).sum(1)
    union = pred.sum(1) + label.sum(1) - intersection

    intersection = np.clip(intersection, 0, 1e8)
    union = np.clip(union, 0, 1e8)

    return ((intersection + eps) / (union + eps)).mean()


def compute_dice(
        pred: np.ndarray, 
        label: np.ndarray, 
        threshold: float = 0.5,
    ) -> float:
    """ Dice score for segmentation tasks. 
    Input:
        - pred (np.ndarray): prediction of shape [batch_size, height, width]
        - label (np.ndarray): binary masks of shape [batch_size, height, width]
        - threshold (float): a threshold to get binary prediction, default is 0.5
    Return:
        - (float) dice score
    
    Reference: dice() function in MedKLIP/Sample_Finetuning_SIIMACR/I2_segmentation/metric.py
    """
    assert label.max() <= 1
    
    batch_size = len(label)

    pred = pred.reshape(batch_size, -1)
    label = label.reshape(batch_size, -1)
    assert (pred.shape == label.shape)

    p = (pred > threshold).astype(np.float64)
    t = (label > 0.5).astype(np.float64)

    t_sum = t.sum(-1)
    p_sum = p.sum(-1)
    
    neg_index = np.expand_dims(np.nonzero(t_sum == 0)[0], 1)
    pos_index = np.expand_dims(np.nonzero(t_sum >= 1)[0], 1)

    dice_neg = (p_sum == 0).astype(np.float64)
    dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

    dice_neg = dice_neg[neg_index]
    dice_pos = dice_pos[pos_index]
    dice = np.concatenate([dice_pos, dice_neg])

    dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
    dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
    dice = dice.mean().item()

    return dice


def compute_mccs_threshold(pred, gt, threshold, n_class=14):
    gt_np = gt 
    pred_np = pred 
    mccs = []
    mccs.append('mccs')
    for i in range(n_class):
        pred_np[:,i][pred_np[:,i]>=threshold[i]]=1
        pred_np[:,i][pred_np[:,i]<threshold[i]]=0
        mccs.append(matthews_corrcoef(gt_np[:, i], pred_np[:, i]))
    mean_mccs = np.mean(np.array(mccs[1:]))
    mccs.append(mean_mccs)
    return mccs


def compute_mccs(pred, gt, n_class=14):
    # get a best threshold for all classes
    gt_np = gt 
    pred_np = pred 
    select_best_thresholds =[]
    best_mcc = 0.0

    for i in range(n_class):
        select_best_threshold_i = 0.0
        best_mcc_i = 0.0
        for threshold_idx in range(len(pred)):
            pred_np_ = pred_np.copy()
            thresholds = pred[threshold_idx]
            pred_np_[:,i][pred_np_[:,i]>=thresholds[i]]=1
            pred_np_[:,i][pred_np_[:,i]<thresholds[i]]=0
            mcc = matthews_corrcoef(gt_np[:, i], pred_np_[:, i])
            if mcc > best_mcc_i:
                best_mcc_i = mcc
                select_best_threshold_i = thresholds[i]
        select_best_thresholds.append(select_best_threshold_i)

    for i in range(n_class):
        pred_np[:,i][pred_np[:,i]>= select_best_thresholds[i]]=1
        pred_np[:,i][pred_np[:,i]< select_best_thresholds[i]]=0
    mccs = []
    mccs.append('mccs')
    for i in range(n_class):
        mccs.append(matthews_corrcoef(gt_np[:, i], pred_np[:, i]))
    mean_mcc = np.mean(np.array(mccs[1:]))
    mccs.append(mean_mcc)
    return mccs, select_best_thresholds


def compute_jaccard(
        pred: np.ndarray, 
        label: np.ndarray, 
        threshold: np.ndarray = None
    ) -> float:
    if threshold is None:
        pred = np.round(pred)
    else:
        pred = (pred > threshold).astype(np.int64)
    return jaccard_score(label, pred)


def compute_single_classification_metrics(
        pred: np.ndarray, 
        label: np.ndarray,
    ) -> Dict[str,float]:
    """ Evaluation metrics for single classification task.
    Input:
        - pred (np.ndarray): prediction (after softmax) of shape [batch_size, num_class]
        - label (np.ndarray): onehot label of shape [batch_size, num_class]
    Return:
        - (Dict[str,float]): auc, acc, and f1 score
    """
    acc_score = compute_acc(pred, label)
    f1_score = np.mean(compute_f1(pred, label))
    auc_score = np.mean(compute_auc(pred, label))

    return dict(auc=auc_score, acc=acc_score, f1=f1_score)


def compute_multi_classification_metrics(
        pred: np.ndarray, 
        label: np.ndarray,
        categories: List[str],
    ) -> Dict[str,Dict[str,float]]:
    """ Evaluation metrics for single classification task.
    Input:
        - pred (np.ndarray): prediction (after sigmoid) of shape [batch_size, num_class]
        - label (np.ndarray): onehot label of shape [batch_size, num_class]
        - categories (List[str]): name of each class, to mark 
    Return:
        - (Dict[str,Dict[str,float]]): auc, acc, and f1 score of each class
    """
    # compute & record acc for each class
    acc_score = {cat: acc for cat, acc in zip(categories, compute_acc_multi(pred, label))}
    f1_score = {cat: f1 for cat, f1 in zip(categories, compute_f1(pred, label))}
    # auc_score = compute_auc(pred, label)
    auc_score = compute_auc(pred, label)
    # auc_score = {cat: auc for cat, auc in zip(categories, compute_auc(pred, label))}
    # auc_score = {}
    # for i in range(len(categories)):
    #     auc_score[categories[i]] = compute_auc(pred[:,i], label[:,i])

    # # compute mcc 
    # mccs, best_thresholds = compute_mccs(pred, label, n_class=len(categories))
    # best_thresholds = np.array(best_thresholds)
    # mcc_score = {cat: mcc for cat, mcc in zip(categories, mccs[1:-1])}
    # acc_score = {cat: acc for cat, acc in zip(categories, compute_acc_multi(pred, label, threshold=best_thresholds))}
    # f1_score = {cat: f1 for cat, f1 in zip(categories, compute_f1(pred, label, threshold=best_thresholds))}
    # prec_score = {}
    # rec_score = {}
    # tnr_score = {}
    # jac_score = {}
    # for i, cat in enumerate(categories):
    #     prec_score[cat] = compute_precision(pred[:,i], label[:,i], best_thresholds[i])
    #     rec_score[cat] = compute_recall(pred[:,i], label[:,i], best_thresholds[i])
    #     tnr_score[cat] = compute_tnr(pred[:,i], label[:,i], best_thresholds[i])
    #     jac_score[cat] = compute_jaccard(pred[:,i], label[:,i], best_thresholds[i])

    # return dict(
    #     auc=auc_score, 
    #     acc=acc_score,
    #     f1=f1_score,
    #     mcc=mcc_score,
    #     prec=prec_score,
    #     recall=rec_score,
    #     tnr=tnr_score,
    #     jac=jac_score,
    # )
    return dict(
        auc=auc_score, 
        acc=acc_score,
        f1=f1_score,
    )


def compute_grounding_metrics(
        pred: np.ndarray,
        cls_label: np.ndarray,
        seg_mask: np.ndarray,
        bbox_mask: np.ndarray,
    ) -> Dict[str,float]:
    """ Evaluation metrics for grounding task.
    Input:
        - pred (np.ndarray): prediction of shape [batch_size, height, width]
        - cls_label (np.ndarray): classification labels of shape [batch_size],
                                  which is the corresponding classification label
        - seg_mask (np.ndarray): binary ground-truth segmentation mask of 
                                 shape [batch_size, height, width]
        - bbox_mask (np.ndarray): binary mask which consist of zero or one or multiple 
                                  rectangle RoIs [batch_size, height, width]. 
                                  You can use `seg_mask` to instead `bbox_mask` 
                                  if bbox was not provided by dataset.
    Return:
        - (Dict[str,float]) point_game_score, recall, precision, iou, dice
    """

    point_score = compute_point_game(pred, bbox_mask, cls_label)
    recall_score = compute_recall(pred, seg_mask)
    precision_score = compute_precision(pred, seg_mask)
    iou_score = 0
    for th in [0.1, 0.2, 0.3 , 0.4, 0.5]:
        iou_score += compute_iou(pred, seg_mask, threshold=th)
    iou_score /= 5
    # iou_score = compute_iou(pred, seg_mask, threshold=0.001)
    dice_score = compute_dice(pred, seg_mask)

    return dict(
        point_game=point_score, 
        recall=recall_score, 
        precision=precision_score, 
        iou=iou_score, 
        dice=dice_score,
    )


def compute_segmentation_metrics(
        pred: np.ndarray,
        seg_mask: np.ndarray,
    ) -> Dict[str,float]:
    """ Evaluation metrics for grounding task.
    Input:
        - pred (np.ndarray): prediction of shape [batch_size, height, width]
        - seg_mask (np.ndarray): binary ground-truth segmentation mask of 
                                 shape [batch_size, height, width]
    Return:
        - (Dict[str,float]) precision, iou, dice
    """
    precision_score = compute_precision(pred, seg_mask)
    iou_score = compute_iou(pred, seg_mask)
    dice_score = compute_dice(pred, seg_mask)

    return dict(
        precision=precision_score, iou=iou_score, dice=dice_score,
    )





if __name__ == "__main__":

    # ###  Here for classification metric  ###
    # '''
    # label [B,C]  numpy.float32
    # pred  [B,C]  numpy.float32
    # '''
    # acc = np.mean(ACC(label, pred))
    # f1 = np.mean(F1(label, pred))
    # auc = np.mean(AUC(label, pred))
    # print('ACC: %.2f, F1: %.2f, AUC: %.2f' % (acc, f1, auc))
    import torch
    import torch.nn.functional as F

    pred = torch.sigmoid(torch.randn(3, 10, 10))
    label = (torch.randn(3, 10, 10) > 0.5).float()

    # miou_score = mIoU(pred, label)
    # miou_score = mIoU(pred.numpy(), label.numpy())
    # miou_score_np = compute_iou(pred.numpy(), label.numpy())

    pred_map = torch.randn(30, 12, 12)
    seg_map = torch.zeros(30, 12, 12)
    seg_map[:,3:7,3:7] = 1
    label = torch.ones(30).long()

    print(seg_map[0])

    max_ind = np.random.choice(pred_map.shape[0], 3)
    pred_map[max_ind, 4, 4] = 9999

    point_score = compute_point_game(pred_map.numpy(), seg_map.numpy(), label.numpy())

    print(point_score)
