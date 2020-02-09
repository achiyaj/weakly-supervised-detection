import os
import json
from random import choice, choices, sample
from model.config_cc import *
from misc.prepare_gt_obj_labels import get_objs_and_atts_datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import lmdb
import six
import torch


class GQAWeakSupervisionDataset(Dataset):
    def __init__(self, objs, atts, gqa_data_path, dset, with_atts, att_categories):
        Dataset.__init__(self)
        self.objs = objs
        self.atts = atts
        self.with_atts = with_atts
        self.gqa_env = lmdb.open(gqa_descriptors_file, subdir=False, readonly=True, lock=False, readahead=False,
                                 meminit=False)
        self.gqa_txn = self.gqa_env.begin(write=False)
        self.gqa_curs = self.gqa_txn.cursor()
        if os.path.isfile(gqa_data_path):
            self.gqa_data = list(json.load(open(gqa_data_path)).items())
        else:
            gqa_data_dict = get_objs_and_atts_datasets(self.objs, self.atts, dset)
            with open(gqa_data_path, 'w') as out_f:
                json.dump(gqa_data_dict, out_f, indent=2)
            self.gqa_data = list(gqa_data_dict.items())

        self.categorize_atts = False
        if att_categories is not None:
            self.categorize_atts = True
            self.att_categories = att_categories
            self.att_to_category = \
                {x: (key, idx) for key, value in att_categories.items() for idx, x in enumerate(value)}

        self.num_labels_distribution = [x / sum(CC_NUM_LABELS_ORDERED) for x in CC_NUM_LABELS_ORDERED]

    def __len__(self):
        return len(self.gqa_data)

    def get_num_objs(self):
        return len(self.objs)  # add background label

    def get_num_atts(self):
        return len(self.atts)

    def get_categorized_atts(self, att_labels, num_labels):
        categorized_atts = {x: [-1] * num_labels for x in self.att_categories}
        for label_idx, att_label in enumerate(att_labels):
            if att_label == -1:
                continue
            cur_category, cur_label = self.att_to_category[att_label]
            categorized_atts[cur_category][label_idx] = cur_label
        return categorized_atts

    def __getitem__(self, idx):
        img_id, raw_labels = self.gqa_data[idx]
        cc_format_labels = [[x[0]] + x[1] for x in raw_labels.values()]
        # num_labels_to_sample = choices(range(1, len(self.num_labels_distribution) + 1), self.num_labels_distribution)[0]
        # if len(cc_format_labels) > num_labels_to_sample:
        #     cc_format_labels = sample(cc_format_labels, num_labels_to_sample)
        raw_data = self.gqa_curs.get(f'features_{img_id}'.encode())
        data = np.load(six.BytesIO(raw_data))

        obj_labels = [x[0] for x in cc_format_labels]
        num_labels = len(cc_format_labels)
        att_labels = None
        if self.with_atts:
            att_labels = [x[1] if len(x) > 1 else -1 for x in cc_format_labels]
            if self.categorize_atts:
                att_labels = self.get_categorized_atts(att_labels, num_labels)
            else:
                att_labels = [self.atts[x] for x in att_labels]

        num_descs = data.shape[0]
        output = {
            'descs': data,
            'num_descs': num_descs,
            'obj_labels': [self.objs[x] for x in obj_labels],
            'img_id': img_id,
            'num_labels_per_image': num_labels,
            'att_labels': att_labels
        }
        return output


    @staticmethod
    def pad_collate(batch):
        output = {}
        # pad object descriptors
        descs = [x['descs'] for x in batch]
        padded_descs = [np.concatenate((x, np.zeros((MAX_NUM_OBJS - x.shape[0], DESCRIPTORS_DIM)))) for x in descs]
        output['descs'] = torch.Tensor(np.stack(padded_descs))

        # pad object labels
        obj_labels = [x['obj_labels'] for x in batch]
        output['obj_labels'] = torch.Tensor([x for img_labels in obj_labels for x in img_labels])
        if batch[0]['att_labels'] is not None:
            att_labels = [x['att_labels'] for x in batch]
            if type(att_labels[0]) == list:  # attributes are uncategorized
                output['att_labels'] = torch.Tensor([x for img_labels in att_labels for x in img_labels])
            else:
                att_labels_dict = \
                    {key: [x for img_atts in att_labels for x in img_atts[key]] for key in att_labels[0].keys()}
                output['att_labels'] = {key: torch.Tensor(value) for key, value in att_labels_dict.items()}
        else:
            output['att_labels'] = None

        # other data fields
        output['img_id'] = [x['img_id'] for x in batch]
        output['num_labels_per_image'] = [x['num_labels_per_image'] for x in batch]
        output['num_descs'] = torch.Tensor([x['num_descs'] for x in batch])
        return output

    def __del__(self):
        self.gqa_env.close()


def get_gqa_dataloader(objs, atts, data_path, dset, att_categories=None):
    dataset = GQAWeakSupervisionDataset(objs, atts, data_path, dset, WITH_ATTS, att_categories)
    if dset == 'val':
        return DataLoader(dataset, **val_loader_params, collate_fn=dataset.pad_collate)
    return DataLoader(dataset, **gqa_train_loader_params, collate_fn=dataset.pad_collate)
