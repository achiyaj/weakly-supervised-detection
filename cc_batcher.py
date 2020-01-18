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


def pad_descriptors(data):
    output = np.zeros((MAX_NUM_OBJS, DESCRIPTORS_DIM))
    output[:data.shape[0], :] = data
    return output


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
            relevant_dset_objs += [(sg_id, x['label']) for x in relevant_objs]
        if len(relevant_objs_and_atts) > 0:
            relevant_dset_objs_and_atts += [(sg_id, x[0], x[1]) for x in relevant_objs_and_atts]

        dset_data_file = get_relevant_data_file(gqa_only, num_objs, num_atts, dset)

    final_data = {
        'relevant_objs': cc_relevant_objs,
        'relevant_atts': cc_relevant_atts,
        'objs_labels': relevant_dset_objs,
        'objs_and_atts_labels': relevant_dset_objs_and_atts
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
            if self.with_atts:
                self.data = relevant_data['objs_and_atts_labels']
            else:
                self.data = relevant_data['objs_labels']

        else:  # build the relevant dataset
            self.objs, self.atts, objs_data, objs_and_atts_data = \
                build_cc_relevant_data(gqa_only, num_objs, num_atts, dset)
            if self.with_atts:
                self.data = objs_and_atts_data
            else:
                self.data = objs_data

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
        if self.with_atts:
            line_id, obj_label, att_label = self.data[idx]
        else:
            line_id, obj_label = self.data[idx]
        img_id = self.line_to_imgs_id[line_id].encode()
        raw_data = np.load(six.BytesIO(self.descriptors_curs.get(img_id)))
        data = raw_data.f.feat
        num_descs = data.shape[0]
        data = pad_descriptors(data)
        output = {
            'descs': data,
            'num_descs': num_descs,
            'obj_label': self.objs[obj_label],
            'img_id': line_id
        }
        if self.with_atts:
            output['att_label'] = self.atts[att_label]
        return output

    def get_class_labels(self):
        return self.objs

    def __del__(self):
        self.descriptors_env.close()


def get_dataloader(dset, img_ids=None):
    dataset = MaxLossCCDataset(GQA_LABELS_ONLY, NUM_TOP_OBJS, NUM_TOP_ATTS, dset, WITH_ATTS, img_ids)
    if dset == 'val':
        return DataLoader(dataset, **val_loader_params)
    return DataLoader(dataset, **train_loader_params)
