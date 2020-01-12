from sklearn.metrics import confusion_matrix
import numpy as np
import json
import os

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


from config_gqa import *


def print_cm(pred_labels, gt_labels, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    cm = confusion_matrix(gt_labels, pred_labels)
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def plot_cm(pred_labels, gt_labels, labels, output_path):
    cm = confusion_matrix(gt_labels, pred_labels, labels=list(range(len(labels))))
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(40, 40))
    sn.set(font_scale=2.5), sn.heatmap(df_cm, annot=True, fmt='d')
    plt.savefig(output_path)


def labels_to_one_hot(labels, output_length):
    output = np.zeros((labels.shape[0], output_length))
    for i in range(labels.shape[0]):
        output[i, labels[i]] = 1
    return output


def get_all_labels_dict(dset):
    if os.path.isfile(imgs_and_objs_dict_file.format(dset)):
        imgs_and_objs_dict = json.load(open(imgs_and_objs_dict_file.format(dset)))
    else:
        with open(relevant_imgs_file.format(dset), 'r') as f:
            imgs_and_objs = json.load(f)

        imgs_and_objs_dict = {}
        for value in imgs_and_objs:
            img, obj_label = value[0], value[1]
            if img in imgs_and_objs_dict:
                imgs_and_objs_dict[img].append(obj_label)
            else:
                imgs_and_objs_dict[img] = [obj_label]

        with open(imgs_and_objs_dict_file.format(dset), 'w') as f:
            json.dump(imgs_and_objs_dict, f)

    return imgs_and_objs_dict


def eval_batch_prediction_gqa(all_gt_labels, preds, img_ids):
    gt_labels = []
    relevant_pred_labels = []
    for img_idx in range(len(img_ids)):
        cur_gt_labels = all_gt_labels[img_ids[img_idx]]
        if len(cur_gt_labels) == 0:
            continue
        cur_preds = preds[img_idx]
        gt_labels += list(cur_gt_labels.values())
        relevant_pred_labels += [pred_label for i, pred_label in enumerate(cur_preds) if str(i) in cur_gt_labels.keys()]

    return gt_labels, relevant_pred_labels


def eval_batch_prediction_cc(all_gt_labels, preds, img_ids):
    gt_labels = []
    relevant_pred_labels = []
    for img_idx in range(len(img_ids)):
        cur_gt_labels = all_gt_labels[img_ids[img_idx]]
        if len(cur_gt_labels) == 0:
            continue
        cur_preds = preds[img_idx]
        gt_labels += list(cur_gt_labels.values())
        relevant_pred_labels += [pred_label for i, pred_label in enumerate(cur_preds) if str(i) in cur_gt_labels.keys()]

    return gt_labels, relevant_pred_labels



def get_cm_path(output_path, epoch, acc):
    return os.path.join(output_path, cm_filename.format(epoch, acc))
