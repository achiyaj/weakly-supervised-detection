import h5py
# from mpi4py import MPI
import os
import json
from random import shuffle
from model.config_gqa import *
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np


def pad_descriptors(data):
    output = np.zeros((MAX_NUM_OBJS, DESCRIPTORS_DIM))
    output[:data.shape[0], :] = data
    return output


class MaxLossGQAObjsDataset(Dataset):
    def __init__(self, dset):
        dset_imgs_file = relevant_imgs_file.format(dset)
        if os.path.isfile(dset_imgs_file):
            self.imgs_and_objs = json.load(open(dset_imgs_file, 'r'))
            self.objs_to_keep = [key for key in json.load(open(obj_new_id_to_name_file, 'r')).keys()]
            self.obj_name_to_new_id = json.load(open(obj_new_id_to_name_file, 'r'))
        else:
            # build relevant_imgs_and_objs.json file
            with open(obj_freqs_file, 'r') as f:
                obj_freqs = json.load(f)
            objs_to_keep = sorted([x for x in obj_freqs.items()], key=lambda x: x[1], reverse=True)[:NUM_TOP_OBJS + 1]
            self.objs_to_keep = [int(x[0]) for x in objs_to_keep if x[0] != '1702']
            obj_id_to_name = {value[0]: key for key, value in json.load(open(obj_orig_id_to_name_file, 'r')).items()}
            self.obj_name_to_new_id = {obj_id_to_name[self.objs_to_keep[i]]: i for i in range(len(self.objs_to_keep))}
            with open(obj_new_id_to_name_file, 'w') as out_f:
                json.dump(self.obj_name_to_new_id, out_f, indent=4)
            relevant_obj_names = \
                [obj_name for obj_id, obj_name in obj_id_to_name.items() if obj_id in self.objs_to_keep]

            sgs = json.load(open(sgs_file.format(dset)))
            self.imgs_and_objs = []

            for img_id, sg in tqdm(sgs.items(), desc=f'Building {dset} dataset'):
                try:
                    cur_objects = [x['name'] for x in sg['objects'].values() if x['name'] in relevant_obj_names]
                except:
                    continue
                if len(cur_objects) > 0:
                    self.imgs_and_objs += [(img_id, x) for x in set(cur_objects)]

            shuffle(self.imgs_and_objs)
            # now save to file
            with open(dset_imgs_file, 'w') as out_f:
                json.dump(self.imgs_and_objs, out_f, indent=4)

        # self.descriptors_file = h5py.File(descriptors_file, 'r', driver='mpio', comm=MPI.COMM_WORLD)
        descriptors_file_new = descriptors_file.replace('.h5', '_copy.h5') if dset == 'val' else descriptors_file
        self.descriptors_file = h5py.File(descriptors_file_new, 'r')

    def __len__(self):
        return len(self.imgs_and_objs)

    def __getitem__(self, idx):
        img_id, obj_label = self.imgs_and_objs[idx]
        data = self.descriptors_file[f'features_{img_id}'][()]
        # data = self.descriptors_file.get_key(f'features_{img_id}')
        num_descs = data.shape[0]
        data = pad_descriptors(data)
        output = {
            'descs': data,
            'num_descs': num_descs,
            'label': self.obj_name_to_new_id[obj_label],
            'img_id': img_id
        }
        return output

    def get_class_labels(self):
        return self.objs_to_keep

    def __del__(self):
        self.descriptors_file.close()


def get_dataloader(dset):
    dataset = MaxLossGQAObjsDataset(dset)
    if dset == 'val':
        return DataLoader(dataset, **val_loader_params)
    return DataLoader(dataset, **train_loader_params)
