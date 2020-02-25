import json
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import os


dsets_paths = {
    'gqa': '/specific/netapp5_2/gamir/achiya/vqa/gqa_max_loss/data/gqa/gqa_top_gqa_10000_objs_10000_atts_categorized_data_train_add_gqa.json',
    'cc_vacancy_sgs': '/specific/netapp5_2/gamir/achiya/vqa/gqa_max_loss/data/cc/cc_top_gqa_10000_objs_10000_atts_categorized_data_train_add_gqa.json',
    'cc_raw_sents': '/specific/netapp5_2/gamir/achiya/vqa/gqa_max_loss/data/cc/cc_top_gqa_10000_objs_10000_atts_categorized_data_train_add_gqa_raw_sents.json'
}


atts_file = '/specific/netapp5_2/gamir/datasets/gqa/raw_data/att_categories.json'
CATEGORIES_TO_DROP = ['hposition', 'place', 'realism', 'room', 'texture', 'vposition', 'company', 'depth', 'flavor',
                      'race', 'location', 'hardness', 'gender', 'brightness']
FREQS_PATH = '/specific/netapp5_2/gamir/achiya/vqa/gqa_max_loss/vocab_stats/{}/{}_stats.json'
DISTS_OUTPUT_PATH = '/specific/netapp5_2/gamir/achiya/vqa/gqa_max_loss/vocab_stats/{}_dist.png'


att_categories = json.load(open(atts_file))
NUM_DIST_TOP_ENTRIES = 20

dists_plot_args = {
    'width': 0.27
}


def print_labels_list_freqs(dset_name, labels_list, category_name):
    data_freqs = dict(Counter(labels_list))
    ordered_data_freqs = OrderedDict(sorted(data_freqs.items(), key=lambda x: x[1], reverse=True))
    cur_freqs_path = FREQS_PATH.format(dset_name, category_name)
    os.makedirs(os.path.dirname(cur_freqs_path), exist_ok=True)
    with open(cur_freqs_path, 'w') as out_f:
        json.dump(ordered_data_freqs, out_f, indent=4)


def get_data_stats():
    for dset_name, input_data_file in dsets_paths.items():
        print(f'\n\nGetting stats for the {dset_name} Dataset!')
        data_dict = json.load(open(input_data_file))
        if dset_name == 'gqa':
            imgs_obj_labels = [obj_data[0] for img_data in data_dict.values() for obj_data in img_data.values()]
            num_imgs = len(data_dict)
        else:  # CC data format
            if len(data_dict['objs_labels']) > 0:  # textual SGs
                imgs_obj_labels = [x for img_data in data_dict['objs_labels'] for x in img_data[1]]
                num_imgs = len(data_dict['objs_labels'])
            else:  # raw sents
                imgs_obj_labels = [x for img_data in data_dict['objs_and_atts_labels'] for x in img_data[1]['objects']]
                num_imgs = len(data_dict['objs_and_atts_labels'])

        print(f'Num relevant images in dataset is {num_imgs}')
        print(f'Avg. num. objects per img is {len(imgs_obj_labels) / num_imgs}')
        print_labels_list_freqs(dset_name, imgs_obj_labels, 'objs')

        for category in ['color', 'material', 'size']:
            cur_category_labels = att_categories[category]
            data_category_labels = []
            if dset_name == 'gqa':
                for img_data in data_dict.values():
                    cur_atts = [x for obj_data in img_data.values() for x in obj_data[1] if x in cur_category_labels]
                    data_category_labels += cur_atts
            else:  # CC data format
                for img_atts_data in data_dict['objs_and_atts_labels']:
                    if len(data_dict['objs_labels']) > 0:  # textual SGs
                        cur_atts = [obj_data[1] for obj_data in img_atts_data[1] if obj_data[1] in cur_category_labels]
                    else:  # raw sents
                        cur_atts = [x for x in img_atts_data[1]['attributes'] if x in cur_category_labels]

                    data_category_labels += cur_atts

            print(f'Avg. number of {category} labels per img is {len(data_category_labels) / num_imgs}')
            print_labels_list_freqs(dset_name, data_category_labels, category)


def get_vocab_comparison():
    dset_names = list(dsets_paths.keys())

    for category in ['objs', 'color', 'material', 'size']:
        dsets_freqs = {}
        for cur_dset_name in dset_names:
            dsets_freqs[cur_dset_name] = json.load(open(FREQS_PATH.format(cur_dset_name, category)))

        top_gqa_labels = list(dsets_freqs['gqa'].keys())[:NUM_DIST_TOP_ENTRIES]

        ind = np.arange(len(top_gqa_labels))
        bars_width = dists_plot_args['width']
        fig = plt.figure(figsize=(len(top_gqa_labels) * len(dset_names) * bars_width + 1, 10))
        ax = fig.add_subplot(111)
        legend_colors_list = []

        for dset_idx, cur_dset_name in enumerate(dset_names):
            cur_freqs = dsets_freqs[cur_dset_name]
            cur_dset_top_labels_freqs = [cur_freqs[cur_label] if cur_label in cur_freqs else 0 for cur_label in top_gqa_labels]

            cur_rect = ax.bar(ind + dset_idx * bars_width, cur_dset_top_labels_freqs, bars_width)
            legend_colors_list.append(cur_rect[0])

        ax.set_ylabel('Num Occurrences')
        ax.set_xticks(ind + bars_width)
        ax.set_xticklabels(top_gqa_labels)
        ax.legend(legend_colors_list, dset_names)
        plt.title(f'Distribution of the top {len(top_gqa_labels)} {category} labels in GQA across different datasets')
        plt.savefig(DISTS_OUTPUT_PATH.format(category))
        print(f'Finished plotting for category: {category}')


if __name__ == '__main__':
    get_data_stats()
    get_vocab_comparison()
