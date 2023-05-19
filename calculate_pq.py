# Base of this code is referenced from panopticapi : https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import argparse
from tqdm import tqdm
import PIL.Image as Image


class PQStatCat():
    def __init__(self):
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def __iadd__(self, pq_stat_cat):
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        self.tn += pq_stat_cat.tn
        return self

    def __str__(self):
        return f"iou: {self.iou}, tp: {self.tp}, fp: {self.fp}, fn: {self.fn}"

class PQStat():
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, categories):
        pq, sq, rq, n, precision_avg, recall_avg, f1_avg, miou = 0, 0, 0, 0, 0, 0, 0, 0
        per_class_results = {}
        for label in categories:
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn

            if tp + fp + fn == 0:
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = tp /( tp + fp + fn) if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            f1 = 2*(recall*precision)/(recall+precision)
            per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class, "precision": precision, "recall": recall, 'f': f1, 'iou': iou}
            pq += pq_class
            sq += sq_class
            rq += rq_class
            recall_avg += recall
            precision_avg += precision
            f1_avg += f1
            miou += iou
        return {'pq': pq /float(n), 'sq': sq /float(n), 'rq': rq / float(n), 'recall': recall_avg/n, 'precision': precision_avg/n, 'f':f1_avg/n, 'miou':miou/n, 'n': n}, per_class_results


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0]
    return int(color[0])


def pq_compute_single_core(gt_folder, pred_folder):
    pq_stat = PQStat()
    gt_images = os.listdir(gt_folder)
    pred_images = os.listdir(pred_folder)
    for filename in tqdm(pred_images):
        if filename not in gt_images:
            print(f"{filename} has no ground truth!")
            continue

        gt = np.asarray(Image.open(gt_folder + filename))
        pred = np.asarray(Image.open(pred_folder + filename).resize((len(gt[0]), len(gt))))
        pred = rgb2id(pred)

        # confusion matrix
        conf_mat = confusion_matrix(
            gt.flatten(), pred.flatten(), labels=[range(151)]
        )

        for class_label in range(151):
            pq_stat_class = PQStatCat()
            pq_stat_class.tp = conf_mat[class_label, class_label]
            pq_stat_class.fp = sum(conf_mat[class_label]) - pq_stat_class.tp
            transposed_cfm = conf_mat.transpose()
            pq_stat_class.fn = sum(transposed_cfm[class_label]) - pq_stat_class.tp

            # calculate intersection and union for class
            intersection = pq_stat_class.tp
            gt_freq = np.sum(gt == class_label)
            pred_freq = np.sum(pred == class_label)
            union = gt_freq + pred_freq - intersection

            # calculate iou
            if union == 0:
                iou = 1.0
            else:
                iou = intersection / union
            pq_stat_class.iou = iou


            pq_stat.pq_per_cat[class_label] += pq_stat_class
            #print(pq_stat_class, pq_stat[class_label])
            # iou/ (TP + 0,5FP + 0,5 FN )
            #pq = iou / (conf_mat[0, 0] + 0.5 * conf_mat[1, 0] + 0.5 * conf_mat[0, 1])
            #print(f"{pq} for class {class_label}")
    return pq_stat


def pq_compute(gt_folder, pred_folder):
    start_time = time.time()

    print("Evaluation panoptic segmentation metrics:")
    print("Ground truth:")
    print("\tSegmentation folder: {}".format(gt_folder))
    print("Prediction:")
    print("\tSegmentation folder: {}".format(pred_folder))

    if not os.path.isdir(gt_folder):
        raise Exception("Folder {} with ground truth segmentations doesn't exist".format(gt_folder))
    if not os.path.isdir(pred_folder):
        raise Exception("Folder {} with predicted segmentations doesn't exist".format(pred_folder))

    pq_stat = pq_compute_single_core(gt_folder, pred_folder)
    categories = range(151)
    metrics = [""]
    results = {}
    for name in metrics:
        results[name], per_class_results = pq_stat.pq_average(categories)
        results[f'per_class_{name}'] = per_class_results
        print(results[f'per_class_{name}'])
    print("{:10s}| {:>6s}  {:>6s}  {:>6s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s}".format("", "PQ", "SQ", "RQ", "RECALL", "PRECISION", "F1", "mIoU", "N"))
    print("-" * (10 + 7 * 12))

    for name in metrics:
        print("{:10s}| {:6.5f}  {:6.5f}  {:6.5f}  {:6.5f}  {:6.5f}  {:6.5f}  {:6.5f}  {:6d}".format(
            name,
            results[name]['pq'],
            results[name]['sq'],
            results[name]['rq'],
            results[name]["recall"],
            results[name]['precision'],
            results[name]['f'],
            results[name]['miou'],
            results[name]['n'])
        )
    j = 0
    for cl in range(1, 151):
       print(per_class_results[cl]['iou']/1000)
       j += per_class_results[cl]['iou']/1000
    print(j/151)
    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_folder', type=str, default=None,
                        help="Folder with ground truth")
    parser.add_argument('--pred_folder', type=str, default=None,
                        help="Folder with predictions")
    args = parser.parse_args()
    pq_compute(args.gt_folder, args.pred_folder)

