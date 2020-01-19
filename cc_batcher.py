import os
import json
from random import sample, shuffle
from config_cc import *
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import OrderedDict
import lmdb
import six
import torch


def build_cc_relevant_data(gqa_only, num_objs, num_atts, dset):
    cc_objs = list(json.load(open(cc_freqs_path.format('objs'), 'r'), object_pairs_hook=OrderedDict).keys())
    cc_atts = list(json.load(open(cc_freqs_path.format('atts'), 'r'), object_pairs_hook=OrderedDict).keys())
    gqa_objs = list(json.load(open(gqa_objs_file, 'r')).keys())
    gqa_atts = list(json.load(open(gqa_atts_file, 'r')).keys())
    cc_relevant_objs, cc_relevant_atts = [], []

    if gqa_only:
        for obj_label in cc_objs:
            if obj_label in gqa_objs:
                cc_relevant_objs.append(obj_label)
                if len(cc_relevant_objs) >= num_objs:
                    break

        for att_label in cc_atts:
            if att_label in gqa_atts:
                cc_relevant_atts.append(att_label)
                if len(cc_relevant_atts) >= num_atts:
                    break

    else:
        cc_relevant_objs = cc_objs[:num_objs]
        cc_relevant_atts = cc_atts[:num_atts]

    relevant_dset_objs = []
    relevant_dset_objs_and_atts = []
    # iterate over scene graphs and save relevant data
    dset_sgs = json.load(open(cc_sgs_path.format(dset)))
    cc_imgs = \
        set([x['id'].split('_')[0] for x in json.load(open(cc_metadata_path, 'r'))['images'] if x['split'] == dset])
    dset_sgs = {key: value for key, value in dset_sgs.items() if key in cc_imgs}

    for sg_id, sg in tqdm(dset_sgs.items(), desc=f'Building {dset} relevant labels dataset'):
        relevant_objs = [x for x in sg['objects'].values() if x['label'] in cc_relevant_objs]
        relevant_objs_and_atts = \
            [(x['label'], att) for x in relevant_objs for att in x['attributes'] if att in cc_relevant_atts]
        if len(relevant_objs) > 0:
            relevant_dset_objs.append((sg_id, [x['label'] for x in relevant_objs]))
        if len(relevant_objs_and_atts) > 0:
            relevant_dset_objs_and_atts.append((sg_id, [(x[0], x[1]) for x in relevant_objs_and_atts]))

        dset_data_file = get_relevant_data_file(gqa_only, num_objs, num_atts, dset)
    max_num_objs_per_image = max([len(x[1]) for x in relevant_dset_objs + relevant_dset_objs_and_atts])

    final_data = {
        'relevant_objs': cc_relevant_objs,
        'relevant_atts': cc_relevant_atts,
        'objs_labels': relevant_dset_objs,
        'objs_and_atts_labels': relevant_dset_objs_and_atts,
        'max_num_objs_per_image': max_num_objs_per_image
    }

    with open(dset_data_file, 'w') as out_f:
        json.dump(final_data, out_f, indent=2)

    return cc_relevant_objs, cc_relevant_atts, relevant_objs, relevant_objs_and_atts


class MaxLossCCDataset(Dataset):
    def __init__(self, gqa_only, num_objs, num_atts, dset, with_atts, img_ids=None):
        self.with_atts = with_atts
        relevant_data_file = get_relevant_data_file(gqa_only, num_objs, num_atts, dset)
        if os.path.isfile(relevant_data_file):
            relevant_data = json.load(open(relevant_data_file, 'r'))
            self.objs = {relevant_data['relevant_objs'][i]: i for i in range(num_objs)}
            self.atts = {relevant_data['relevant_atts'][i]: i for i in range(num_atts)}
            self.data = relevant_data['objs_labels']
            if self.with_atts:
                self.data += relevant_data['objs_and_atts_labels']
            self.max_num_objs_per_img = relevant_data['max_num_objs_per_image']

        else:  # build the relevant dataset
            self.objs, self.atts, objs_data, objs_and_atts_data = \
                build_cc_relevant_data(gqa_only, num_objs, num_atts, dset)
            self.data = objs_data
            if self.with_atts:
                self.data += objs_and_atts_data

        shuffle(self.data)

        self.descriptors_env = lmdb.open(descriptors_file, subdir=False, readonly=True, lock=False, readahead=True,
                                         meminit=False)
        self.txn = self.descriptors_env.begin(write=False)
        self.descriptors_curs = self.txn.cursor()
        self.line_to_imgs_id = json.load(open(line_to_img_id_file.format(dset)))

        if img_ids:  # if the image IDs to load are specified
            self.objs_data = [x for x in self.objs_data if x[0] in img_ids]
            self.objs_and_atts_data = [x for x in self.objs_and_atts_data if x[0] in img_ids]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line_id, labels = self.data[idx]
        att_label_present = (type(labels[0]) == list)
        if att_label_present:  # this is an object only label
            line_id, objs_and_atts_labels = self.data[idx]
            obj_labels = [x[0] for x in labels]
            att_labels = [x[1] for x in labels]
            num_labels = len(labels)
        else:  # this is object + attribute labels
            obj_labels = labels
            num_labels = len(obj_labels)
        img_id = self.line_to_imgs_id[line_id].encode()
        raw_data = np.load(six.BytesIO(self.descriptors_curs.get(img_id)))
        data = raw_data.f.feat
        num_descs = data.shape[0]
        output = {
            'descs': data,
            'num_descs': num_descs,
            'obj_labels': [self.objs[x] for x in obj_labels],
            'img_id': line_id,
            'num_labels_per_image': num_labels
        }
        if att_label_present:
            output['att_labels'] = [self.atts[x] for x in att_labels]
        else:
            output['att_labels'] = [-1] * len(obj_labels)

        return output

    def get_class_labels(self):
        return self.objs

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
        if 'att_labels' in batch[0]:
            att_labels = [x['att_labels'] for x in batch]
            output['att_labels'] = torch.Tensor([x for img_labels in att_labels for x in img_labels])

        # other data fields
        output['img_id'] = [x['img_id'] for x in batch]
        output['num_labels_per_image'] = [x['num_labels_per_image'] for x in batch]
        output['num_descs'] = torch.Tensor([x['num_descs'] for x in batch])
        return output

    def __del__(self):
        self.descriptors_env.close()


def get_dataloader(dset, img_ids=None):
    dataset = MaxLossCCDataset(GQA_LABELS_ONLY, NUM_TOP_OBJS, NUM_TOP_ATTS, dset, WITH_ATTS, img_ids)
    if dset == 'val':
        return DataLoader(dataset, **val_loader_params, collate_fn=dataset.pad_collate)
    return DataLoader(dataset, **train_loader_params, collate_fn=dataset.pad_collate)
