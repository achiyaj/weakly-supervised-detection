from nltk import word_tokenize
import json
from tqdm import tqdm
from collections import Counter

CC_CAPTIONS_FILES = {
    'train': '/specific/netapp5_2/gamir/datasets/ConceptualCaptions/Train_GCC-training.tsv',
    'val': '/specific/netapp5_2/gamir/datasets/ConceptualCaptions/Validation_GCC-1.1.0-Validation.tsv'
}
GQA_OBJS_DICT_PATH = '/specific/netapp5_2/gamir/datasets/gqa/objects_dict.json'
GQA_ATTS_DICT_PATH = '/specific/netapp5_2/gamir/datasets/gqa/attributes_dict.json'
DATA_OUTPUT_PATH = '/specific/netapp5_2/gamir/datasets/ConceptualCaptions/scene_graphs/raw_sents_data/{}_data.json'
FREQS_OUTPUT_PATH = '/specific/netapp5_2/gamir/datasets/ConceptualCaptions/scene_graphs/raw_sents_data/{}_freqs.json'


def main():
    gqa_objs = set(json.load(open(GQA_OBJS_DICT_PATH)).keys())
    gqa_atts = set(json.load(open(GQA_ATTS_DICT_PATH)).keys())
    objs_and_atts_data = {}
    for dset in ['train']:  # ['train', 'val']:
        captions_file = CC_CAPTIONS_FILES[dset]
        with open(captions_file) as f:
            lines = [x.split('\t')[0] for x in f.readlines()]

        for line_id, line in tqdm(enumerate(lines), total=len(lines)):
            line_words = word_tokenize(line)
            cur_obj_labels = []
            cur_att_labels = []
            for word in line_words:
                if word in gqa_objs:
                    cur_obj_labels.append(word)
                if word in gqa_atts:
                    cur_att_labels.append(word)

            if len(cur_obj_labels) + len(cur_att_labels) > 0:
                objs_and_atts_data[str(line_id)] = \
                    {'sent': line, 'objects': cur_obj_labels, 'attributes': cur_att_labels}

        if dset == 'train':
            # build freqs file for train set
            all_objs = [x for img_data in objs_and_atts_data.values() for x in img_data['objects']]
            all_objs_counter = dict(Counter(all_objs))
            objs_sorted_counter = \
                {k: v for k, v in sorted(all_objs_counter.items(), key=lambda item: item[1], reverse=True)}
            with open(FREQS_OUTPUT_PATH.format('objs'), 'w') as out_f:
                json.dump(objs_sorted_counter, out_f, indent=4)

            all_atts = [x for img_data in objs_and_atts_data.values() for x in img_data['attributes']]
            all_atts_counter = dict(Counter(all_atts))
            atts_sorted_counter = \
                {k: v for k, v in sorted(all_atts_counter.items(), key=lambda item: item[1], reverse=True)}
            with open(FREQS_OUTPUT_PATH.format('atts'), 'w') as out_f:
                json.dump(atts_sorted_counter, out_f, indent=4)

        with open(DATA_OUTPUT_PATH.format(dset), 'w') as out_f:
            json.dump(objs_and_atts_data, out_f, indent=2)


if __name__ == '__main__':
    main()
