from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from datasets.metric import *
import cv2
from sklearn.metrics import roc_auc_score, confusion_matrix
from hausdorff import hausdorff_distance
from multiprocessing import Pool

print("PyTorch Version: ", torch.__version__)

'''
evaluation
'''
def compute_metrics(y_scores, y_true, relative=True, comparison='union', filtration='superlevel', construction='V'):
    BM = BettiMatching(y_scores, y_true, relative=relative, comparison=comparison, filtration=filtration,
                       construction=construction)

    return [BM.loss(dimensions=[0, 1]), BM.loss(dimensions=[0]), BM.loss(dimensions=[1]), BM.Betti_number_error(
        threshold=0.5, dimensions=[0, 1]), BM.Betti_number_error(threshold=0.5, dimensions=[0]), BM.Betti_number_error(
        threshold=0.5, dimensions=[1])]


def infer_metric_threshold(infer_path, mask_path, low, end, size, no_betti):
    filenames = os.listdir(mask_path)
    img_num = len(filenames)

    thresholds = np.arange(low, end, size)
    for threshold in thresholds:
        total_img = 0
        total_iou = 0.0
        total_f1 = 0.0
        total_acc = 0.0
        total_sen = 0.0
        total_auc = 0.0
        total_spec = 0.0
        cldices = []
        hds = []
        betti_losses = []
        pool = Pool(8)
        for i in range(img_num):
            now_img = cv2.imread(os.path.join(infer_path, filenames[i][:-3] + 'tif'), -1)
            if now_img is None:
                # print('not exist')
                continue

            # now_img = now_img/255.0
            now_mask = cv2.imread(os.path.join(mask_path, filenames[i]), 0)
            gt_arr = now_mask // 255

            best_iou = 0.00
            # enable evaluation mode
            y_scores = np.zeros_like(now_img)
            y_true = np.zeros_like(now_mask)
            y_true[now_mask > 0.01] = 1
            y_scores[now_img > threshold] = 1
            hd = hausdorff_distance(y_scores, y_true)
            if 'nucleus' in mask_path:
                cldice = 0
            else:
                cldice = clDice(y_scores, y_true)
            if no_betti:
                loss = loss_0 = loss_1 = betti_err = betti_0_err = betti_1_err = 0
            else:
                betti_losses.append(pool.apply_async(compute_metrics, args=(y_scores, y_true,)))
            cldices.append(cldice)
            y_scores1 = y_scores.flatten()
            # y_pred = y_scores > threshold
            y_true1 = y_true.flatten()

            hds.append(hd)

            confusion = confusion_matrix(y_true1, y_scores1)
            tp = float(confusion[1, 1])
            fn = float(confusion[1, 0])
            fp = float(confusion[0, 1])
            tn = float(confusion[0, 0])

            val_acc = (tp + tn) / (tp + fn + fp + tn)
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            precision = tp / (tp + fp)
            f1 = 2 * sensitivity * precision / (sensitivity + precision + 1e-9)
            iou = tp / (tp + fn + fp + 1e-9)
            auc = calc_auc(now_img, gt_arr)
            total_iou += iou
            total_acc += val_acc
            total_f1 += f1
            total_auc += auc
            total_sen += sensitivity
            total_spec += specificity
            total_img += 1

        epoch_iou = (total_iou) / total_img
        if epoch_iou > best_iou:
            best_iou = epoch_iou
        epoch_f1 = total_f1 / total_img
        epoch_acc = total_acc / total_img
        epoch_auc = total_auc / total_img
        epoch_sen = total_sen / total_img
        epoch_spec = total_spec / total_img
        epoch_clDice = np.mean(cldices)
        epoch_hd = np.mean(hds)
        message = "inference  =====>threshold: {:.4f}: Evaluation  ACC: {:.4f}; IOU: {:.4f}; F1_score: {:.4f}; Auc: {:.4f} ;Sen: {:.4f}; Spec: {:.4f}; clDice: {:.4f}; hausdorff_distance: {:.4f};".format(
            threshold,
            epoch_acc,
            epoch_iou,
            epoch_f1, epoch_auc, epoch_sen, epoch_spec, epoch_clDice, epoch_hd)

        print("==> %s" % (message))

        pool.close()
        pool.join()
        if no_betti:
            Betti_error = Betti_error_std = Betti_0_error = Betti_0_error_std = Betti_1_error = Betti_1_error_std = 0
        else:
            betti_results = []
            for if_index in range(total_img):
                betti_result_now = betti_losses[if_index].get()
                betti_results.append(betti_result_now)

            betti_losses_array2 = np.array(betti_results)
            betti_mean = np.mean(betti_losses_array2, axis=0)
            Betti_error = betti_mean[3]
            Betti_error_std = betti_mean[3]
            Betti_0_error = betti_mean[4]
            Betti_0_error_std = betti_mean[4]
            Betti_1_error = betti_mean[5]
            Betti_1_error_std = betti_mean[5]

        print("Betti number error", Betti_error)
        # print("Betti number error std", Betti_error_std)
        print("Betti number error dim 0", Betti_0_error)
        # print("Betti number error dim 0 std", Betti_0_error_std)
        print("Betti number error dim 1", Betti_1_error)
        # print("Betti number error dim 1 std", Betti_1_error_std)


if __name__ == "__main__":
    er_end_path = '/predict_score/ER_best/'
    er_mask_dir = '/mnt/data1/ER/test/masks/'


    model_dir = 'er_fractal_HRNet_iou_32_0.05_50_0.3_1000_20240312_warmup'
    infer_path = './train_logs/' + model_dir + er_end_path

    infer_metric_threshold(infer_path, er_mask_dir, 0.3, 0.31, 0.01, no_betti=False)
