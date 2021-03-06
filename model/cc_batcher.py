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


def build_cc_relevant_data(out_path, gqa_only, num_objs, num_atts, atts_to_use, dset, add_gqa, use_textual_sgs):
    cc_obj_freqs = json.load(open(cc_freqs_path.format('objs'), 'r'), object_pairs_hook=OrderedDict)
    cc_att_freqs = json.load(open(cc_freqs_path.format('atts'), 'r'), object_pairs_hook=OrderedDict)
    cc_objs = list(cc_obj_freqs.keys())
    cc_atts = list(cc_att_freqs.keys())
    gqa_objs_dict = json.load(open(gqa_objs_file, 'r'))
    gqa_objs = list(OrderedDict(sorted(gqa_objs_dict.items(), key=lambda x: x[1][1], reverse=True)).keys())
    gqa_atts_dict = json.load(open(gqa_atts_file, 'r'))
    gqa_atts = list(OrderedDict(sorted(gqa_atts_dict.items(), key=lambda x: x[1][1], reverse=True)).keys())

    if atts_to_use:
        cc_atts = [x for x in cc_atts if x in atts_to_use]
        gqa_atts = [x for x in gqa_atts if x in atts_to_use]

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
    dset_sgs = json.load(open(cc_data_path.format(dset)))
    cc_imgs = \
        set([x['id'].split('_')[0] for x in json.load(open(cc_metadata_path, 'r'))['images'] if x['split'] == dset])
    dset_sgs = {key: value for key, value in dset_sgs.items() if key in cc_imgs}

    for img_id, img_data in tqdm(dset_sgs.items(), desc=f'Building {dset} relevant labels dataset'):
        if use_textual_sgs:
            relevant_objs = [x for x in img_data['objects'].values() if x['label'] in cc_relevant_objs]
            relevant_objs_and_atts = \
                [(x['label'], att) for x in relevant_objs for att in x['attributes'] if att in cc_relevant_atts]

            if len(relevant_objs) > 0:
                relevant_dset_objs.append((img_id, [x['label'] for x in relevant_objs]))
            if len(relevant_objs_and_atts) > 0:
                relevant_dset_objs_and_atts.append((img_id, [(x[0], x[1]) for x in relevant_objs_and_atts]))

        else:
            img_relevant_data = {
                'objects': [x for x in img_data['objects'] if x in cc_relevant_objs],
                'attributes': [x for x in img_data['attributes'] if x in cc_relevant_atts]
            }
            if len(img_relevant_data['objects']) + len(img_relevant_data['attributes']) > 0:
                relevant_dset_objs_and_atts.append((img_id, img_relevant_data))

    if use_textual_sgs:
        max_num_objs_per_image = max([len(x[1]) for x in relevant_dset_objs + relevant_dset_objs_and_atts])
    else:
        max_num_objs_per_image = max([len(x[1]['objects']) + len(x[1]['attributes']) for x in relevant_dset_objs_and_atts])

    final_data = {
        'relevant_objs': cc_relevant_objs,
        'relevant_atts': cc_relevant_atts,
        'objs_labels': relevant_dset_objs,
        'objs_and_atts_labels': relevant_dset_objs_and_atts,
        'max_num_objs_per_image': max_num_objs_per_image
    }

    with open(out_path, 'w') as out_f:
        json.dump(final_data, out_f, indent=2)

    return cc_relevant_objs, cc_relevant_atts, relevant_dset_objs, relevant_dset_objs_and_atts


class MaxLossCCDataset(Dataset):
    def __init__(self, gqa_only, num_objs, num_atts, dset, with_atts, add_gqa, att_categories, use_textual_sgs,
                 img_ids):
        Dataset.__init__(self)
        self.with_atts = with_atts
        self.add_gqa = add_gqa
        self.use_textual_sgs = use_textual_sgs
        atts_to_use = None
        self.categorize_atts = not (att_categories is None)
        if self.categorize_atts:
            self.att_categories = list(att_categories.keys())
            self.att_to_category = \
                {x: (key, idx) for key, value in att_categories.items() for idx, x in enumerate(value)}
            atts_to_use = [x for category in att_categories.values() for x in category]
        self.cc_data_file = \
            get_relevant_data_file(gqa_only, num_objs, num_atts, self.categorize_atts, dset, add_gqa)
        if os.path.isfile(self.cc_data_file):
            relevant_data = json.load(open(self.cc_data_file, 'r'))
            relevant_objs = relevant_data['relevant_objs']
            relevant_atts = relevant_data['relevant_atts']
            self.objs = {relevant_objs[i]: i for i in range(min(num_objs, len(relevant_objs)))}
            if add_gqa:
                self.objs['BACKGROUND'] = len(self.objs)  # if gqa is present, add background label
            self.atts = {relevant_atts[i]: i for i in range(min(num_atts, len(relevant_atts)))}
            cc_data = relevant_data['objs_labels']
            if self.with_atts:
                cc_data += relevant_data['objs_and_atts_labels']
            self.max_num_objs_per_img = relevant_data['max_num_objs_per_image']

        else:  # build the relevant dataset
            self.objs, self.atts, objs_data, objs_and_atts_data = \
                build_cc_relevant_data(self.cc_data_file, gqa_only, num_objs, num_atts, atts_to_use, dset, add_gqa,
                                       use_textual_sgs=use_textual_sgs)
            cc_data = objs_data
            if self.with_atts:
                cc_data += objs_and_atts_data

        if self.use_textual_sgs:
            cc_data_dict = {}
            for data_entry in cc_data:
                key, data = data_entry
                if type(data[0]) == str:  # this is an objects list
                    data = [[x] for x in data]  # wrap each element in a list
                if key in cc_data_dict:
                    cc_data_dict[key] += data
                else:
                    cc_data_dict[key] = data
        else:
            cc_data_dict = {x[0]: x[1] for x in cc_data}

        self.cc_data = cc_data_dict

        self.cc_env = lmdb.open(cc_descriptors_file, subdir=False, readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.cc_txn = self.cc_env.begin(write=False)
        self.cc_curs = self.cc_txn.cursor()
        self.line_to_imgs_id = json.load(open(line_to_img_id_file.format(dset)))
        self.imgs_id_to_line = {value: key for key, value in self.line_to_imgs_id.items()}

        if img_ids:  # if the image IDs to load are specified
            self.objs_data = [x for x in self.objs_data if x[0] in img_ids]
            self.objs_and_atts_data = [x for x in self.objs_and_atts_data if x[0] in img_ids]

        self.data_generator = self.iterate_lmdb_cursor()

    def __len__(self):
        return len(self.cc_data)

    def get_num_objs(self):
        return len(self.objs)

    def get_num_atts(self):
        if not self.with_atts:
            return 0
        return len(self.atts)

    def get_datafile(self):
        return self.cc_data_file

    def iterate_lmdb_cursor(self):
        for key, value in self.cc_curs:
            img_id = key.decode()
            if img_id not in self.imgs_id_to_line or self.imgs_id_to_line[img_id] not in self.cc_data:
                continue
            yield self.imgs_id_to_line[img_id], value

    def get_categorized_atts(self, att_labels, num_labels):
        if num_labels > 0:  # output labels in objs & atts entangled format
            categorized_atts = {x: [-1] * num_labels for x in self.att_categories}
            for label_idx, att_label in enumerate(att_labels):
                if att_label == -1:
                    continue
                cur_category, cur_label = self.att_to_category[att_label]
                categorized_atts[cur_category][label_idx] = cur_label
        else:
            categorized_atts = {x: [] for x in self.att_categories}
            for att_label in att_labels:
                cur_category, cur_label = self.att_to_category[att_label]
                categorized_atts[cur_category].append(cur_label)

        return categorized_atts

    def get_sg_labels(self, labels):
        obj_labels = [x[0] for x in labels]
        num_labels = len(labels)
        att_labels = None
        if self.with_atts:
            att_labels = [x[1] if len(x) > 1 else -1 for x in labels]
            if self.categorize_atts:
                att_labels = self.get_categorized_atts(att_labels, num_labels)
            else:
                att_labels = [self.atts[x] for x in att_labels]

        obj_labels = [self.objs[x] for x in obj_labels]
        return obj_labels, att_labels

    def get_raw_sents_labels(self, labels):
        obj_labels = [self.objs[x] for x in labels['objects']]
        if self.categorize_atts:
            att_labels = self.get_categorized_atts(labels['attributes'], 0)
        else:
            att_labels = [self.atts[x] for x in labels['attributes']]

        return obj_labels, att_labels

    def get_cc_item(self, idx):
        data = next(self.data_generator)
        line_id, raw_data = data
        labels = self.cc_data[line_id]

        if self.use_textual_sgs:
            obj_labels, att_labels = self.get_sg_labels(labels)
            num_labels = len(obj_labels)
        else:
            obj_labels, att_labels = self.get_raw_sents_labels(labels)
            num_labels = None

        raw_data = np.load(six.BytesIO(raw_data))
        data = raw_data.f.feat
        num_descs = data.shape[0]
        output = {
            'descs': data,
            'num_descs': num_descs,
            'obj_labels': obj_labels,
            'img_id': line_id,
            'num_labels_per_image': num_labels,
            'att_labels': att_labels
        }
        return output

    def __getitem__(self, idx):
        return self.get_cc_item(idx)

    def get_obj_labels(self):
        return self.objs

    def get_att_labels(self):
        if not self.with_atts:
            return []
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

        use_textual_sgs = (batch[0]['num_labels_per_image'] is not None)

        if use_textual_sgs:
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
        else:
            def lists_to_padded_tensor(lists):
                max_num_entries = max([len(x) for x in lists])
                if max_num_entries == 0:
                    return None
                padded_lists = [torch.Tensor(x + [-1] * (max_num_entries - len(x))) for x in lists]
                output = torch.stack(padded_lists)
                if len(output.shape) == 1:
                    output = output.unsqueeze(1)
                return output

            output['obj_labels'] = lists_to_padded_tensor(obj_labels)
            if batch[0]['att_labels'] is not None:
                att_labels = [x['att_labels'] for x in batch]
                if type(att_labels[0]) == list:  # attributes are uncategorized
                    output['att_labels'] = lists_to_padded_tensor(att_labels)
                else:
                    output['att_labels'] = {}
                    for category_name in att_labels[0].keys():
                        category_att_labels = [x[category_name] for x in att_labels]
                        output['att_labels'][category_name] = lists_to_padded_tensor(category_att_labels)
                        if output['att_labels'][category_name] is not None and len(output['att_labels'][category_name].shape) == 1:
                            abc = 123
            else:
                output['att_labels'] = None

        # other data fields
        output['img_id'] = [x['img_id'] for x in batch]

        output['num_labels_per_image'] = [x['num_labels_per_image'] for x in batch] if use_textual_sgs else None

        output['num_descs'] = torch.Tensor([x['num_descs'] for x in batch])
        output['supervision_type'] = 0  # weak supervision
        return output

    def __del__(self):
        self.cc_env.close()


def get_cc_dataloader(dset, att_categories, img_ids=None):
    dataset = MaxLossCCDataset(GQA_LABELS_ONLY, NUM_TOP_OBJS, NUM_TOP_ATTS, dset, WITH_ATTS, True,
                               att_categories, USE_TEXTUAL_SGS, img_ids)

    if dset == 'val':
        return DataLoader(dataset, **val_loader_params, collate_fn=dataset.pad_collate)
    return DataLoader(dataset, **cc_train_loader_params, collate_fn=dataset.pad_collate)
