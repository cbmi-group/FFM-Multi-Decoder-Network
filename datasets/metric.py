from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score,roc_auc_score
from scipy.spatial.distance import directed_hausdorff
from skimage.morphology import skeletonize, skeletonize_3d
from datasets.BettiMatching import *
import cv2
import math

def extract_mask(pred_arr, gt_arr, mask_arr=None):
    # we want to make them into vectors
    pred_vec = pred_arr.flatten()
    gt_vec = gt_arr.flatten()

    if mask_arr is not None:
        mask_vec = mask_arr.flatten()
        idx = list(np.where(mask_vec == 0)[0])

        pred_vec = np.delete(pred_vec, idx)
        gt_vec = np.delete(gt_vec, idx)

    return pred_vec, gt_vec


def calc_auc(pred_arr, gt_arr, mask_arr=None):
    pred_vec, gt_vec = extract_mask(pred_arr, gt_arr, mask_arr=mask_arr)
    roc_auc = roc_auc_score(gt_vec, pred_vec)

    return roc_auc


def numeric_score(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    """Computation of statistical numerical scores:

    * FP = False Positives
    * FN = False Negatives
    * TP = True Positives
    * TN = True Negatives

    return: tuple (FP, FN, TP, TN)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    dilated_gt_arr = cv2.dilate(gt_arr, kernel, iterations=1)

    FP = np.float(np.sum(np.logical_and(pred_arr == 1, dilated_gt_arr == 0)))
    FN = np.float(np.sum(np.logical_and(pred_arr == 0, gt_arr == 1)))
    TP = np.float(np.sum(np.logical_and(pred_arr == 1, dilated_gt_arr == 1)))
    TN = np.float(np.sum(np.logical_and(pred_arr == 0, gt_arr == 0)))

    return FP, FN, TP, TN


def calc_acc(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    acc = (TP + TN) / (FP + FN + TP + TN)

    return acc


def calc_sen(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    sen = TP / (FN + TP + 1e-12)

    return sen


def calc_fdr(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    fdr = FP / (FP + TP + 1e-12)

    return fdr


def calc_spe(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    spe = TN / (FP + TN + 1e-12)

    return spe


def calc_gmean(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    sen = calc_sen(pred_arr, gt_arr, kernel_size=kernel_size)
    spe = calc_spe(pred_arr, gt_arr, kernel_size=kernel_size)

    return math.sqrt(sen * spe)


def calc_kappa(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size=kernel_size)
    matrix = np.array([[TP, FP],
                       [FN, TN]])
    n = np.sum(matrix)

    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col

    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)

    return (po - pe) / (1 - pe)


def calc_iou(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    iou = TP / (FP + FN + TP + 1e-12)

    return iou


def calc_dice(pred_arr, gt_arr, kernel_size=(1, 1)):  # DCC & ROSE-2: kernel_size=(3, 3)
    FP, FN, TP, TN = numeric_score(pred_arr, gt_arr, kernel_size)
    dice = 2.0 * TP / (FP + FN + 2.0 * TP + 1e-12)

    return dice


def hausdorff_distance_single(seg, label):
    # segmentation = seg.squeeze(1)
    # mask = label.squeeze(1)
    segmentation = seg
    mask = label

    non_zero_seg = np.transpose(np.nonzero(segmentation))
    non_zero_mask = np.transpose(np.nonzero(mask))
    h_dist = max(directed_hausdorff(non_zero_seg, non_zero_mask)[0],
                 directed_hausdorff(non_zero_mask, non_zero_seg)[0])

    return h_dist


def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v * s) / np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape) == 2:
        tprec = cl_score(v_p, skeletonize(v_l))
        tsens = cl_score(v_l, skeletonize(v_p))
    elif len(v_p.shape) == 3:
        tprec = cl_score(v_p, skeletonize_3d(v_l))
        tsens = cl_score(v_l, skeletonize_3d(v_p))
    return 2 * tprec * tsens / (tprec + tsens)


def compute_metrics(t, relative=False, comparison='union', filtration='superlevel', construction='V'):
    BM = BettiMatching(t[0], t[1], relative=relative, comparison=comparison, filtration=filtration,
                       construction=construction)
    return BM.loss(dimensions=[0, 1]), BM.loss(dimensions=[0]), BM.loss(dimensions=[1]), BM.Betti_number_error(
        threshold=0.5, dimensions=[0, 1]), BM.Betti_number_error(threshold=0.5, dimensions=[0]), BM.Betti_number_error(
        threshold=0.5, dimensions=[1])


def acc(seg, label):
    now_num = seg.shape[0]
    # seg, label = np.array(seg), np.array(label)
    seg_one = seg.reshape(-1)
    label_one = label.reshape(-1)

    label_T = label_one > 0
    corrects = torch.eq(seg_one, label_T).sum()
    all_num = seg_one.numel()

    # corrects = (seg.int() == label.int())
    acc = corrects / all_num
    return acc * now_num


def roc(pred, label):
    pred, label = np.array(pred), np.array(label)
    preds_roc = np.reshape(pred, -1)
    labels_roc = np.reshape(label, -1)
    fpr, tpr, thresholds = roc_curve(labels_roc, preds_roc)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def dice_cof(pred, label, reduce=False):
    matrix_sum = pred.int() + label.int()
    i = torch.sum(matrix_sum == 2, dim=(1, 2, 3))
    x1 = torch.sum(pred == 1, dim=(1, 2, 3))
    x2 = torch.sum(label == 1, dim=(1, 2, 3))
    dice_score = 2. * i.float() / (x1.float() + x2.float())
    if reduce:
        return torch.mean(dice_score)
    else:
        return torch.sum(dice_score)


def IoU(preds, labels, reduce=False):
    matrix_sum = preds.int() + labels.int()
    i = torch.sum(matrix_sum == 2, dim=(1, 2, 3))
    u = torch.sum(matrix_sum == 1, dim=(1, 2, 3))
    iou = i.float() / (i.float() + u.float() + 1e-9)
    if reduce:
        iou = torch.mean(iou)
    else:
        iou = torch.sum(iou)
    return iou


def IoU_r(preds, labels, reduce=False):
    matrix_sum = preds.int() + labels.int()
    i = torch.sum(matrix_sum == 2, dim=(1, 2))
    u = torch.sum(matrix_sum == 1, dim=(1, 2))
    iou = i.float() / (i.float() + u.float() + 1e-9)
    if reduce:
        iou = torch.mean(iou)
    else:
        iou = torch.sum(iou)
    return iou


def acc_list(seg, label):
    total_acc = 0.0
    img_num = len(seg)
    for auc_index in range(img_num):
        now_pred = seg[auc_index]
        now_labels = label[auc_index]
        val_acc = acc(now_pred[0], now_labels[0])
        total_acc += val_acc

    return total_acc


def dIoU(preds, labels, reduce=False):
    matrix_sum = preds.int() + labels.int()
    i = torch.sum(matrix_sum == 2)
    u = torch.sum(matrix_sum == 1)
    iou = i.float() / (i.float() + u.float() + 1e-9)
    if reduce:
        iou = torch.mean(iou)
    else:
        iou = iou
    return iou


def mIoU(preds, labels, reduce=False):
    matrix_sum = preds.int() + labels.int()
    f_i = torch.sum(matrix_sum == 2, dim=(1, 2, 3))
    u = torch.sum(matrix_sum == 1, dim=(1, 2, 3))
    b_i = torch.sum(matrix_sum == 0, dim=(1, 2, 3))
    f_iou = f_i.float() / (f_i.float() + u.float() + 1e-9)
    b_iou = b_i.float() / (b_i.float() + u.float() + 1e-9)
    miou = 0.5 * (f_iou + b_iou)
    if reduce:
        miou = torch.mean(miou)
    else:
        miou = torch.sum(miou)
    return miou


def dmIoU(preds, labels, reduce=False):
    matrix_sum = preds.int() + labels.int()
    f_i = torch.sum(matrix_sum == 2)
    u = torch.sum(matrix_sum == 1)
    b_i = torch.sum(matrix_sum == 0)
    f_iou = f_i.float() / (f_i.float() + u.float() + 1e-9)
    b_iou = b_i.float() / (b_i.float() + u.float() + 1e-9)
    miou = 0.5 * (f_iou + b_iou)
    if reduce:
        miou = torch.mean(miou)
    else:
        miou = miou
    return miou


def F1_score(pred, label, reduce=False):
    pred, label = pred.int(), label.int()
    p = torch.sum((label == 1).int(), dim=(1, 2, 3))
    tp = torch.sum((pred == 1).int() & (label == 1).int(), dim=(1, 2, 3))
    fp = torch.sum((pred == 1).int() & (label == 0).int(), dim=(1, 2, 3))
    recall = tp.float() / (p.float() + 1e-9)
    precision = tp.float() / (tp.float() + fp.float() + 1e-9)
    f1 = (2 * recall * precision) / (recall + precision + 1e-9)
    if reduce:
        f1 = torch.mean(f1)
    else:
        f1 = torch.sum(f1)
    return f1


def dF1_score(pred, label, reduce=False):
    pred, label = pred.int(), label.int()
    p = torch.sum((label == 1).int())
    tp = torch.sum((pred == 1).int() & (label == 1).int())
    fp = torch.sum((pred == 1).int() & (label == 0).int())
    recall = tp.float() / (p.float() + 1e-9)
    precision = tp.float() / (tp.float() + fp.float() + 1e-9)
    f1 = (2 * recall * precision) / (recall + precision + 1e-9)
    if reduce:
        f1 = torch.mean(f1)
    else:
        f1 = f1
    return f1
