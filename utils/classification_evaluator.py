import sys
sys.path.insert(0, "/specific/netapp5_2/gamir/achiya/vqa/gqa_max_loss/weakly-supervised-detection/")

from model.mlp_model import InferenceMLPModel
import json
from model.config_cc import mlp_params
import h5py
from collections import OrderedDict
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from tqdm import tqdm
from torch.utils import data
from sklearn.metrics import accuracy_score

ckpt_path = '/specific/netapp5_2/gamir/achiya/vqa/gqa_max_loss/exps/cc/{}/*_epoch_{}.pt'
labels_path = '/specific/netapp5_2/gamir/achiya/vqa/gqa_max_loss/data/cc/top_gqa_10000_objs_10000_atts_data_val_add_gqa.json'
ref_objs_dict = '/specific/netapp5_2/gamir/datasets/gqa/objects_dict.json'
ATT_CATEGORIES_FILE = '/specific/netapp5_2/gamir/datasets/gqa/raw_data/att_categories.json'
detectors_ckpt_template = '/specific/netapp5_2/gamir/achiya/vqa/misc/offline_classification/ckpts/{}.h5'
dset_path_template = '/specific/netapp5_2/gamir/achiya/vqa/misc/offline_classification/{}_val.h5'

categorize_atts = True
CATEGORIES_TO_DROP = ['hposition', 'place', 'realism', 'room', 'texture', 'vposition', 'company', 'depth', 'flavor',
                      'race', 'location', 'hardness', 'gender', 'brightness']

EXP_NAME = 'train_1cc_4gqa_cat_atts'
BEST_VAL_EPOCH = 5

ckpt_path = glob(ckpt_path.format(EXP_NAME, BEST_VAL_EPOCH))[-1]

loader_params = {'batch_size': 256,
                 'shuffle': True,
                 'num_workers': 0,
                 'drop_last': False}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


class HDF5Dataset(data.Dataset):
    def __init__(self, data_path):
        super(data.Dataset, self).__init__()
        with h5py.File(data_path, 'r') as data_f:
            my_data = data_f['data'][()]
        self.data = my_data[:, :-1]
        self.labels = my_data[:, -1]

    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index, :])
        y = torch.from_numpy(np.array([self.labels[index]]))
        return x, y

    def num_labels(self):
        return int(np.max(self.labels)) + 1

    def __len__(self):
        return self.labels.shape[0]


def compare_models(my_model, ref_model, dset_path, classifier_class, my_labels_to_ref_labels=None):
    dset = HDF5Dataset(dset_path)
    data_loader = data.DataLoader(dset, **loader_params)
    all_gt_labels = []
    all_ref_preds = []
    all_my_preds = []
    for batch in tqdm(data_loader):
        descs = batch[0].float().to(device)
        labels = batch[1].numpy()[:, 0]
        ref_model_output = ref_model(descs).argmax(axis=1).detach().cpu().numpy()
        my_model_output = my_model(descs.unsqueeze(0)).squeeze(0).argmax(axis=1).detach().cpu().numpy()
        if my_labels_to_ref_labels:
            my_model_output = [my_labels_to_ref_labels[x] for x in my_model_output]
        all_gt_labels += [x for x in labels]
        all_ref_preds += [x for x in ref_model_output]
        all_my_preds += [x for x in my_model_output]

    print(f'for ref model {classifier_class} accuracy is: {accuracy_score(all_gt_labels, all_ref_preds)}')
    print(f'for my model {classifier_class} accuracy is: {accuracy_score(all_gt_labels, all_my_preds)}')


def main():
    ref_objects_dict = {value[0]: key for key, value in json.load(open(ref_objs_dict, 'r')).items()}
    data_dict = json.load(open(labels_path), object_pairs_hook=OrderedDict)
    obj_labels = data_dict['relevant_objs']
    obj_labels_dict = {i: obj_labels[i] for i in range(len(obj_labels))}
    if 'BACKGROUND' not in obj_labels_dict:
        obj_labels_dict['BACKGROUND'] = len(obj_labels_dict)
    att_labels = data_dict['relevant_atts']
    att_labels_dict = {i: att_labels[i] for i in range(len(att_labels))}

    att_categories = None
    if categorize_atts:
        att_categories = json.load(open(ATT_CATEGORIES_FILE, 'r'))
        att_categories = \
            {key: list(value.keys()) for key, value in att_categories.items() if key not in CATEGORIES_TO_DROP}

    model = InferenceMLPModel(mlp_params['hidden_dim'], mlp_params['input_dim'], len(obj_labels_dict),
                              len(att_labels_dict), att_categories).to(device)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    ref_objs_to_ids = {value: key for key, value in ref_objects_dict.items()}
    my_objs_to_ids = {key: ref_objs_to_ids[value] for key, value in obj_labels_dict.items() if key != 'BACKGROUND'}

    def compare_category(category_name, categorize_vals):
        dset_path = dset_path_template.format(category_name)
        ref_model_path = detectors_ckpt_template.format(category_name)
        ref_model = MLPModel(128, 2048, len(categorize_vals)).to(device)
        ref_model.load_state_dict(torch.load(ref_model_path))
        ref_model.eval()
        my_model = getattr(model, f'{category_name}_mlp')
        if category_name == 'objs':
            compare_models(my_model, ref_model, dset_path, category_name, my_objs_to_ids)
        else:
            compare_models(my_model, ref_model, dset_path, category_name)

    compare_category('objs', obj_labels)

    for cur_category_name, cur_categorize_vals in att_categories.items():
        compare_category(cur_category_name, cur_categorize_vals)


if __name__ == '__main__':
    main()
