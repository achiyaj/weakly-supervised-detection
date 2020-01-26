import os
import json
from random import shuffle
from model.config_cc import *
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import OrderedDict
import lmdb
import six
import torch


def build_cc_relevant_data(gqa_only, num_objs, num_atts, dset, add_gqa):
    cc_obj_freqs = json.load(open(cc_freqs_path.format('objs'), 'r'), object_pairs_hook=OrderedDict)
    cc_att_freqs = json.load(open(cc_freqs_path.format('atts'), 'r'), object_pairs_hook=OrderedDict)
    cc_objs = list(cc_obj_freqs.keys())
    cc_atts = list(cc_att_freqs.keys())
    gqa_objs_dict = json.load(open(gqa_objs_file, 'r'))
    gqa_objs = list(OrderedDict(sorted(gqa_objs_dict.items(), key=lambda x: x[1][1], reverse=True)).keys())
    gqa_atts_dict = json.load(open(gqa_atts_file, 'r'))
    gqa_atts = list(OrderedDict(sorted(gqa_atts_dict.items(), key=lambda x: x[1][1], reverse=True)).keys())

    cc_relevant_objs, cc_relevant_atts = [], []

    if add_gqa:
        cc_relevant_objs = gqa_objs[:NUM_TOP_OBJS]
        cc_relevant_atts = gqa_atts[:NUM_TOP_ATTS]

    elif gqa_only:
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

        dset_data_file = get_relevant_data_file(gqa_only, num_objs, num_atts, dset, add_gqa)
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
    def __init__(self, gqa_only, num_objs, num_atts, dset, with_atts, add_gqa, img_ids=None):
        self.with_atts = with_atts
        self.add_gqa = add_gqa
        relevant_data_file = get_relevant_data_file(gqa_only, num_objs, num_atts, dset, add_gqa)
        if os.path.isfile(relevant_data_file):
            relevant_data = json.load(open(relevant_data_file, 'r'))
            relevant_objs = relevant_data['relevant_objs']
            relevant_atts = relevant_data['relevant_atts']
            self.objs = {relevant_objs[i]: i for i in range(min(num_objs, len(relevant_objs)))}
            if add_gqa:
                self.objs['BACKGROUND'] = len(self.objs)  # if gqa is present, add background label
            self.atts = {relevant_atts[i]: i for i in range(min(num_atts, len(relevant_atts)))}
            self.cc_data = relevant_data['objs_labels']
            if self.with_atts:
                self.cc_data += relevant_data['objs_and_atts_labels']
            self.max_num_objs_per_img = relevant_data['max_num_objs_per_image']

        else:  # build the relevant dataset
            self.objs, self.atts, objs_data, objs_and_atts_data = \
                build_cc_relevant_data(gqa_only, num_objs, num_atts, dset, add_gqa)
            self.cc_data = objs_data
            if self.with_atts:
                self.cc_data += objs_and_atts_data

        shuffle(self.cc_data)

        self.cc_env = lmdb.open(cc_descriptors_file, subdir=False, readonly=True, lock=False, readahead=True,
                                meminit=False)
        self.cc_txn = self.cc_env.begin(write=False)
        self.cc_curs = self.cc_txn.cursor()
        self.line_to_imgs_id = json.load(open(line_to_img_id_file.format(dset)))

        if img_ids:  # if the image IDs to load are specified
            self.objs_data = [x for x in self.objs_data if x[0] in img_ids]
            self.objs_and_atts_data = [x for x in self.objs_and_atts_data if x[0] in img_ids]

    def __len__(self):
        return len(self.cc_data)

    def get_num_objs(self):
        return len(self.objs)

    def get_num_atts(self):
        return len(self.atts)

    def get_cc_item(self, idx):
        line_id, labels = self.cc_data[idx]
        att_label_present = (type(labels[0]) == list)
        if att_label_present:  # this is an object only label
            line_id, objs_and_atts_labels = self.cc_data[idx]
            obj_labels = [x[0] for x in labels]
            att_labels = [x[1] for x in labels]
            num_labels = len(labels)
        else:  # this is object + attribute labels
            obj_labels = labels
            num_labels = len(obj_labels)
        img_id = self.line_to_imgs_id[line_id].encode()
        raw_data = np.load(six.BytesIO(self.cc_curs.get(img_id)))
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

    def __getitem__(self, idx):
        return self.get_cc_item(idx)

    def get_obj_labels(self):
        return self.objs

    def get_att_labels(self):
        return self.atts

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
        self.cc_env.close()


def get_cc_dataloader(dset, img_ids=None):
    dataset = MaxLossCCDataset(GQA_LABELS_ONLY, NUM_TOP_OBJS, NUM_TOP_ATTS, dset, WITH_ATTS, GQA_OVERSAMPLING_RATE > 0,
                               img_ids)
    if dset == 'val':
        return DataLoader(dataset, **val_loader_params, collate_fn=dataset.pad_collate)
    return DataLoader(dataset, **cc_train_loader_params, collate_fn=dataset.pad_collate)
