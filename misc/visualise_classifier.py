from model import InferenceMLPModel
from bounding_box import bounding_box as bb
from random import sample
import json
from config_cc import mlp_params
import torch
import h5py
import cv2
from collections import OrderedDict
import numpy as np
import os


gqa_val_sgs = '/specific/netapp5_2/gamir/datasets/gqa/raw_data/val_sceneGraphs.json'
gqa_data_file = '/specific/netapp5_2/gamir/datasets/gqa/orig_features_our_format_all.h5'
ckpt_path = '/specific/netapp5_2/gamir/achiya/vqa/gqa_max_loss/exps/cc/{}/objs_ckpts_copy.pt'
imgs_path = '/specific/netapp5_2/gamir/datasets/gqa/images/{}.jpg'
labels_path = '/specific/netapp5_2/gamir/achiya/vqa/gqa_max_loss/data/cc/top_gqa_50_objs_50_atts_data_val.json'
NUM_IMGS_TO_TEST = 20
EXP_NAME = 'cc_top_50_objs_only_initial'
CLASSIFY_ATTS = False
CONF_THRESH = 0.95

ckpt_path = ckpt_path.format(EXP_NAME)
output_path = os.path.join(os.path.dirname(ckpt_path), 'imgs')
os.makedirs(output_path, exist_ok=True)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_ids = sample(list(json.load(open(gqa_val_sgs)).keys()), NUM_IMGS_TO_TEST)
    model = InferenceMLPModel(mlp_params['hidden_dim'], mlp_params['input_dim'], mlp_params['output_dim']).to(device)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    labels = json.load(open(labels_path), object_pairs_hook=OrderedDict)['relevant_objs']
    labels_dict = {i: labels[i] for i in range(len(labels))}

    with h5py.File(gqa_data_file, 'r') as data_f, torch.set_grad_enabled(False):
        for img_id in img_ids:
            features = data_f['features_' + img_id][()]
            tensor_features = torch.Tensor(features).unsqueeze(0).to(device)
            bboxes = data_f['bboxes_' + img_id][()]
            img = cv2.imread(imgs_path.format(img_id))
            abc = 123
            pred_labels, pred_probs = model(tensor_features, np.array([features.shape[0]]))
            pred_probs = pred_probs[0]
            pred_labels = pred_labels[0]
            relevant_bboxes_data = [(i, pred_labels[i]) for i in range(len(pred_probs)) if pred_probs[i] > CONF_THRESH]
            for box_data in relevant_bboxes_data:
                box_id, box_label = box_data
                cur_bbox = bboxes[box_id, :]
                bb.add(img, cur_bbox[0], cur_bbox[1], cur_bbox[2], cur_bbox[3], labels_dict[box_label], 'blue')
            cv2.imwrite(os.path.join(output_path, img_id + '.jpg'), img)


if __name__ == '__main__':
    main()
