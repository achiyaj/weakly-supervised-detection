from model import InferenceMLPModel
from bounding_box import bounding_box as bb
from random import sample
import json
from config_cc import mlp_params, WITH_ATTS
import h5py
import cv2
from collections import OrderedDict
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob

gqa_val_sgs = '/specific/netapp5_2/gamir/datasets/gqa/raw_data/val_sceneGraphs.json'
gqa_data_file = '/specific/netapp5_2/gamir/datasets/gqa/orig_features_our_format_all.h5'
ckpt_path = '/specific/netapp5_2/gamir/achiya/vqa/gqa_max_loss/exps/cc/{}/*.pt'
imgs_path = '/specific/netapp5_2/gamir/datasets/gqa/images/{}.jpg'
labels_path = '/specific/netapp5_2/gamir/achiya/vqa/gqa_max_loss/data/cc/top_gqa_50_objs_50_atts_data_val.json'
ref_objs_dict = '/specific/netapp5_2/gamir/datasets/gqa/objects_dict.json'
NUM_IMGS_TO_TEST = 20
EXP_NAME = 'cc_objs_and_atts'
OBJ_CONF_THRESH = 0.8
ATT_CONF_THRESH = 0.8
REF_CONF_THRESH = 0.2

ckpt_path = glob(ckpt_path.format(EXP_NAME))[-1]
output_path = os.path.join(os.path.dirname(ckpt_path), 'imgs')
# ref_output_path = os.path.join(os.path.dirname(ckpt_path), 'imgs_ref')
os.makedirs(output_path, exist_ok=True)
# os.makedirs(ref_output_path, exist_ok=True)
ref_objects_detector_ckpt = '/specific/netapp5_2/gamir/achiya/vqa/misc/offline_classification/ckpts/objs_cc.h5'


class MLPModel(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output = F.relu(self.fc1(x))
        output = F.relu(self.fc2(output))
        output = F.softmax(self.fc3(output))
        return output


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_ids = None
    # img_ids = ['61584', '2346590', '2357280', '2368559', '2400139', '2410441', '2410790', '2413051']
    if not img_ids:
        img_ids = sample(list(json.load(open(gqa_val_sgs)).keys()), NUM_IMGS_TO_TEST)
    ref_objects_dict = {value[0]: key for key, value in json.load(open(ref_objs_dict, 'r')).items()}
    model = InferenceMLPModel(mlp_params['hidden_dim'], mlp_params['input_dim'], mlp_params['objs_output_dim'],
                              mlp_params['atts_output_dim']).to(device)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    ref_model = MLPModel(128, 2048, mlp_params['objs_output_dim']).to(device)
    ref_model.load_state_dict(torch.load(ref_objects_detector_ckpt))
    obj_labels = json.load(open(labels_path), object_pairs_hook=OrderedDict)['relevant_objs']
    obj_labels_dict = {i: obj_labels[i] for i in range(len(obj_labels))}
    att_labels = json.load(open(labels_path), object_pairs_hook=OrderedDict)['relevant_atts']
    att_labels_dict = {i: att_labels[i] for i in range(len(att_labels))}

    with h5py.File(gqa_data_file, 'r') as data_f, torch.set_grad_enabled(False):
        for img_id in img_ids:
            features = data_f['features_' + img_id][()]
            tensor_features = torch.Tensor(features).unsqueeze(0).to(device)
            bboxes = data_f['bboxes_' + img_id][()]
            img = cv2.imread(imgs_path.format(img_id))
            img_copy = img.copy()
            pred_obj_labels, pred_obj_probs, pred_att_labels, pred_att_probs = \
                model(tensor_features, np.array([features.shape[0]]))
            pred_obj_labels = pred_obj_labels[0]
            pred_obj_probs = pred_obj_probs[0]
            pred_att_labels = pred_att_labels[0]
            pred_att_probs = pred_att_probs[0]

            # relevant_bboxes_data = [(i, pred_labels[i]) for i in range(len(pred_probs)) if pred_probs[i] > CONF_THRESH]
            relevant_bboxes_data = []
            for i in range(len(pred_obj_labels)):
                if pred_obj_probs[i] > OBJ_CONF_THRESH:
                    if pred_att_probs[i] > ATT_CONF_THRESH:
                        relevant_bboxes_data.append(
                            (i, att_labels_dict[pred_att_labels[i]] + ' ' + obj_labels_dict[pred_obj_labels[i]]))
                    else:
                        relevant_bboxes_data.append((i, obj_labels_dict[pred_obj_labels[i]]))

            for box_data in relevant_bboxes_data:
                box_id, box_label = box_data
                cur_bbox = bboxes[box_id, :]
                bb.add(img, cur_bbox[0], cur_bbox[1], cur_bbox[2], cur_bbox[3], box_label)

            # visualise reference model
            ref_preds = ref_model(tensor_features.squeeze(0)).cpu().detach().numpy()
            ref_pred_labels = np.argmax(ref_preds, axis=1)
            ref_pred_probs = np.max(ref_preds, axis=1)
            relevant_bboxes_data_ref = [(i, ref_pred_labels[i]) for i in range(len(ref_pred_probs))
                                        if ref_pred_probs[i] > REF_CONF_THRESH]
            for box_data in relevant_bboxes_data_ref:
                box_id, box_label = box_data
                cur_bbox = bboxes[box_id, :]
                bb.add(img_copy, cur_bbox[0], cur_bbox[1], cur_bbox[2], cur_bbox[3], ref_objects_dict[box_label])
            imgs_concat = np.concatenate((img, img_copy), axis=1)
            cv2.imwrite(os.path.join(output_path, img_id + '.jpg'), imgs_concat)


if __name__ == '__main__':
    main()
