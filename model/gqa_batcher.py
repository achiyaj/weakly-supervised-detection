import os
import json
from random import choice
from model.config_cc import *
from misc.prepare_gt_obj_labels import get_objs_and_atts_datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import lmdb
import six
import torch


class GQAWithAttsDataset(Dataset):
    def __init__(self, objs, atts, dset, gqa_oversampling_rate):
        self.objs = objs
        self.atts = atts
        self.gqa_oversampling_rate = gqa_oversampling_rate
        self.gqa_env = lmdb.open(gqa_descriptors_file, subdir=False, readonly=True, lock=False, readahead=True,
                                 meminit=False)
        self.gqa_txn = self.gqa_env.begin(write=False)
        self.gqa_curs = self.gqa_txn.cursor()
        gqa_data_path = get_gqa_data_file(len(self.objs), len(self.atts), dset)
        if os.path.isfile(gqa_data_path):
            self.gqa_data = list(json.load(open(gqa_data_path)).items())
        else:
            gqa_data_dict = get_objs_and_atts_datasets(self.objs, self.atts, dset)
            with open(gqa_data_path, 'w') as out_f:
                json.dump(gqa_data_dict, out_f, indent=2)
            self.gqa_data = list(gqa_data_dict.items())

    def __len__(self):
        return len(self.gqa_data) * self.gqa_oversampling_rate

    def get_num_objs(self):
        return len(self.objs)  # add background label

    def get_num_atts(self):
        return len(self.atts)

    def get_label_vectors(self, raw_labels, num_descs):
        # first initialize all of the labels with "BACKGROUND"
        obj_labels, att_labels = np.ones((num_descs)) * self.objs['BACKGROUND'], np.ones((num_descs)) * (-1)
        for obj_id, labels in raw_labels.items():
            obj_labels[int(obj_id)] = self.objs[labels[0]]
            if len(labels[1]) > 0:
                att_labels[int(obj_id)] = self.atts[choice(labels[1])]

        return obj_labels, att_labels

    def __getitem__(self, idx):
        img_id, raw_labels = self.gqa_data[idx]
        raw_data = self.gqa_curs.get(f'features_{img_id}'.encode())
        data = np.load(six.BytesIO(raw_data))
        num_descs = data.shape[0]
        obj_labels, att_labels = self.get_label_vectors(raw_labels, num_descs)
        return {
            'descs': data,
            'num_descs': num_descs,
            'obj_labels': obj_labels,
            'att_labels': att_labels,
            'img_id': img_id,
            'num_labels_per_image': None
        }

    @staticmethod
    def pad_collate(batch):
        output = {}
        # pad object descriptors
        descs = [x['descs'] for x in batch]
        padded_descs = [np.concatenate((x, np.zeros((MAX_NUM_OBJS - x.shape[0], DESCRIPTORS_DIM)))) for x in descs]
        output['descs'] = torch.Tensor(np.stack(padded_descs))

        # pad object labels
        obj_labels = [x['obj_labels'] for x in batch]
        padded_obj_labels = [np.concatenate((x, np.array([-1] * (MAX_NUM_OBJS - x.shape[0])))) for x in obj_labels]
        output['obj_labels'] = torch.Tensor(np.stack(padded_obj_labels))

        if 'att_labels' in batch[0]:
            att_labels = [x['att_labels'] for x in batch]
            padded_att_labels = [np.concatenate((x, np.array([-1] * (MAX_NUM_OBJS - x.shape[0])))) for x in att_labels]
            output['att_labels'] = torch.Tensor(np.stack(padded_att_labels))

        # other data fields
        output['img_id'] = [x['img_id'] for x in batch]
        output['num_labels_per_image'] = [x['num_labels_per_image'] for x in batch]
        output['num_descs'] = torch.Tensor([x['num_descs'] for x in batch])
        return output

    def __del__(self):
        self.gqa_env.close()


def get_gqa_dataloader(objs, atts, dset):
    dataset = GQAWithAttsDataset(objs, atts, dset, GQA_OVEERSAMPLING_RATE)
    if dset == 'val':
        return DataLoader(dataset, **val_loader_params, collate_fn=dataset.pad_collate)
    return DataLoader(dataset, **gqa_train_loader_params, collate_fn=dataset.pad_collate)
