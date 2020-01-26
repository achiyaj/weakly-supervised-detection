import json
import h5py
import numpy as np
from tqdm import tqdm
from pdb import set_trace as trace

from model.config_gqa import *


FERATURES_INPUT_FILE = '/specific/netapp5_2/gamir/datasets/gqa/orig_features_our_format_all.h5'
SGS_FILE = '/specific/netapp5_2/gamir/datasets/gqa/raw_data/{}_sceneGraphs.json'
CATEGORIES_FILE = '/specific/netapp5_2/gamir/datasets/gqa/raw_data/att_categories.json'
OBJS_FILE = '/specific/netapp5_2/gamir/datasets/gqa/objects_dict.json'
OUTPUT_FILE = '/specific/netapp5_2/gamir/achiya/vqa/misc/offline_classification/{}_{}.h5'
dsets = ['train', 'val', 'testdev']


def get_relevant_bboxes_from_sg(sg, atts_set, att_name_to_idx):
    all_coords = []
    labels = []
    for obj_id, obj_data in enumerate(sg['objects'].values()):
        relevant_atts = atts_set.intersection(set(obj_data['attributes']))
        if len(relevant_atts) != 1:
            continue
        x1, y1 = obj_data['x'], obj_data['y']
        x2, y2 = x1 + obj_data['w'], y1 + obj_data['h']
        all_coords.append(np.array([x1, y1, x2, y2], dtype=np.float32))
        labels.append(att_name_to_idx[relevant_atts.pop()])
    if len(all_coords) > 0:
        return np.stack(all_coords), labels
    return None, None


def get_objs_bboxes_from_sg(sg, obj_name_to_idx, relevant_obj_labels):
    all_coords = []
    labels = []
    for obj_id, obj_data in enumerate(sg['objects'].values()):
        if obj_data['name'] not in obj_name_to_idx or obj_data['name'] not in relevant_obj_labels:
            continue
        x1, y1 = obj_data['x'], obj_data['y']
        x2, y2 = x1 + obj_data['w'], y1 + obj_data['h']
        all_coords.append(np.array([x1, y1, x2, y2], dtype=np.float32))
        labels.append(obj_name_to_idx[obj_data['name']])
    if len(all_coords) > 0:
        return np.stack(all_coords), labels
    return None, None


def get_objs_and_atts_bboxes_from_sg(sg, relevant_obj_labels, relevant_att_labels):
    all_coords = []
    labels = []
    for obj_id, obj_data in enumerate(sg['objects'].values()):
        obj_name = obj_data['name']
        if obj_name not in relevant_obj_labels:
            continue
        x1, y1 = obj_data['x'], obj_data['y']
        x2, y2 = x1 + obj_data['w'], y1 + obj_data['h']
        all_coords.append(np.array([x1, y1, x2, y2], dtype=np.float32))
        obj_atts = [x for x in obj_data['attributes'] if x in relevant_att_labels]
        labels.append((obj_name, obj_atts))
    if len(all_coords) > 0:
        return np.stack(all_coords), labels
    return None, None


def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = np.reshape(np.tile(np.expand_dims(boxes1, 1),
                            [1, 1, np.shape(boxes2)[0]]), [-1, 4])
    b2 = np.tile(boxes2, [np.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = np.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = np.split(b2, 4, axis=1)
    y1 = np.maximum(b1_y1, b2_y1)
    x1 = np.maximum(b1_x1, b2_x1)
    y2 = np.minimum(b1_y2, b2_y2)
    x2 = np.minimum(b1_x2, b2_x2)
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = np.reshape(iou, [np.shape(boxes1)[0], np.shape(boxes2)[0]])
    return overlaps


def align_bboxes(detected_bboxes, gt_bboxes):
    iou_scores = overlaps_graph(detected_bboxes, gt_bboxes)
    assignment = np.argmax(iou_scores, axis=1)
    scores = np.max(iou_scores, axis=1)
    alignment_dict = {}
    for detected_box, gt_box in enumerate(assignment):
        if (not gt_box in alignment_dict or alignment_dict[gt_box][1] < scores[detected_box]) and \
                scores[detected_box] > IOU_THRESHOLD:
            alignment_dict[gt_box] = (detected_box, scores[detected_box])
    return alignment_dict


def get_objs_and_atts_datasets(relevant_obj_labels, relevant_att_labels, dset):
    bboxes_file = h5py.File(FERATURES_INPUT_FILE, 'r')
    dset_alignment_dict = {}
    dset_sgs = json.load(open(SGS_FILE.format(dset), 'r'))
    for img_id, cur_sg in tqdm(list(dset_sgs.items()), desc='Building dataset for {} set'.format(dset)):
        obj_gt_bboxes, labels = get_objs_and_atts_bboxes_from_sg(cur_sg, relevant_obj_labels, relevant_att_labels)
        if obj_gt_bboxes is None:
            continue
        cur_detected_bboxes = bboxes_file['bboxes_' + img_id][()]
        alignment_dict = align_bboxes(cur_detected_bboxes, obj_gt_bboxes)
        detected_box_to_label = {value[0]: labels[key] for key, value in alignment_dict.items()}
        dset_alignment_dict[img_id] = detected_box_to_label

    return dset_alignment_dict

def build_objs_datasets():
    objs = {key: value[0] for key, value in json.load(open(OBJS_FILE, 'r')).items()}
    bboxes_file = h5py.File(FERATURES_INPUT_FILE, 'r')
    relevant_obj_labels = json.load(open(obj_new_id_to_name_file, 'r')).keys()
    obj_name_to_new_id = json.load(open(obj_new_id_to_name_file, 'r'))
    obj_old_id_to_name = {value[0]: key for key, value in json.load(open(obj_orig_id_to_name_file, 'r')).items()}
    obj_old_id_to_new_id = {key: obj_name_to_new_id[value] for key, value in obj_old_id_to_name.items()
                            if value in obj_name_to_new_id.keys()}
    for dset in ['train', 'val']:
        dset_alignment_dict = {}
        dset_sgs = json.load(open(SGS_FILE.format(dset), 'r'))
        for img_id, cur_sg in tqdm(list(dset_sgs.items()), desc='Building dataset for {} set'.format(dset)):
            obj_gt_bboxes, labels = get_objs_bboxes_from_sg(cur_sg, objs, relevant_obj_labels)
            if obj_gt_bboxes is None:
                continue
            cur_detected_bboxes = bboxes_file['bboxes_' + img_id][()]
            alignment_dict = align_bboxes(cur_detected_bboxes, obj_gt_bboxes)
            detected_box_to_new_label = \
                {value[0]: obj_old_id_to_new_id[labels[key]] for key, value in alignment_dict.items()}
            dset_alignment_dict[img_id] = detected_box_to_new_label

        with open(imgs_and_objs_align_dict_file.format(dset), 'w') as f:
            json.dump(dset_alignment_dict, f, indent=4)


def build_atts_datasets():
    att_categories = {key: list(value.keys()) for key, value in json.load(open(CATEGORIES_FILE, 'r')).items()}
    att_name_to_idx = {category: {atts[i]: i for i in range(len(atts))} for category, atts in att_categories.items()}
    bboxes_file = h5py.File(FERATURES_INPUT_FILE, 'r')
    for dset in ['train', 'val']:
        att_categories_features = {key: {'features': [], 'labels': []} for key in att_categories.keys()}
        dset_sgs = json.load(open(SGS_FILE.format(dset), 'r'))
        for img_id, cur_sg in tqdm(list(dset_sgs.items()), desc='Building dataset for {} set'.format(dset)):
            cur_detected_bboxes = bboxes_file['bboxes_' + img_id][()]
            cur_features = bboxes_file['features_' + img_id][()]
            for atts_category, cur_atts in att_categories.items():
                relevant_gt_bboxes, labels = get_relevant_bboxes_from_sg(cur_sg, set(cur_atts),
                                                                         att_name_to_idx[atts_category])
                if not labels:
                    continue
                alignment_dict = align_bboxes(cur_detected_bboxes, relevant_gt_bboxes)
                att_categories_features[atts_category]['features'].append(
                    cur_features[[x[0] for x in alignment_dict.values()], :])
                att_categories_features[atts_category]['labels'] += [labels[x] for x in alignment_dict.keys()]

        # Build datasets files
        for atts_category, data in att_categories_features.items():
            print('For category: {} in {} set, {} objects were found'.format(atts_category, dset, len(data['labels'])))
            if len(data['labels']) == 0:
                continue
            all_labels = np.expand_dims(np.array(data['labels']), 1)
            all_features = np.concatenate(data['features'])
            all_features = np.concatenate((all_features, all_labels), axis=1)

            with h5py.File(OUTPUT_FILE.format(atts_category, dset), 'w') as out_f:
                out_f.create_dataset("data", data=all_features)


def main():
    build_atts_datasets()
    build_objs_datasets()



if __name__ == '__main__':
    main()



