import os
import json
from glob import glob
from random import shuffle
import time
from tqdm import tqdm

img_labels_html_template_path = \
    '/specific/netapp5_2/gamir/achiya/vqa/gqa_max_loss/data/general/labels_html_template.html'

output_html_path = '/specific/a/home/cc/students/cs/achiyajerbi/html/gqa_max_loss/labels_visualization/{}/{}.html'

imgs_paths = {
    'gqa': '/specific/netapp5_2/gamir/datasets/gqa/images',
    'cc': '/specific/netapp5_2/gamir/datasets/ConceptualCaptions/imgs'
}

labels_srcs = {
    'gqa': 'gqa_top_gqa_10000_objs_10000_atts_categorized_data_train_add_gqa.json',
    'cc_vacancy_sgs': 'cc_top_gqa_10000_objs_10000_atts_categorized_data_train_add_gqa.json',
    'cc_raw_sents': 'cc_top_gqa_10000_objs_10000_atts_categorized_data_train_add_gqa_raw_sents.json'
}

LABELS_BASE_PATH = '/specific/netapp5_2/gamir/achiya/vqa/gqa_max_loss/data'
NUM_LABELS_TO_EVAL = 50

cur_labels_src = 'gqa'
labels_category = 'objs'
dset_name = cur_labels_src.split('_')[0]

img_labels_html_template = open(img_labels_html_template_path).read()


def main():
    labels_data = json.load(open(os.path.join(LABELS_BASE_PATH, dset_name, labels_srcs[cur_labels_src])))
    if dset_name == 'cc':
        labels_data = labels_data['objs_labels']
        imgs_ids = [x[0] for x in labels_data]

    else:  # GQA
        imgs_ids = list(labels_data.keys())

    shuffle(imgs_ids)
    imgs_texts = []

    html_text = '<html>\n<head><title>{}</title></head>\n    <body>\n'.format(cur_labels_src)

    def get_cur_img_html_text(img_num):
        if dset_name == 'gqa':
            cur_img_path = os.path.join(imgs_paths[dset_name], f'{img_num}.jpg')
        else:  # CC
            cur_img_path = os.path.join(imgs_paths[dset_name], f'train_{img_num}.jpg')

        if not os.path.isfile(cur_img_path):
            print(f'{dset_name} Image with ID {img_num} not found!')
            return None

        img_filename = cur_img_path.split('/')[-1]
        link_img_filename = os.path.join(f'../{dset_name}_imgs_path', img_filename)
        if dset_name == 'gqa':
            img_labels = labels_data[img_num]
            obj_labels = [x[0] for x in img_labels.values()]
        else:
            obj_labels = [x[1] for x in labels_data if x[0] == img_num][0]

        if len(obj_labels) == 0:
            return None

        shuffle(obj_labels)
        cur_html_text = img_labels_html_template.replace('image_path_ph', link_img_filename)
        cur_html_text = cur_html_text.replace('cur_label', obj_labels[0])
        return cur_html_text

    num_found_imgs = 0
    cur_img_idx = 0
    pbar = tqdm(total=NUM_LABELS_TO_EVAL)
    while num_found_imgs < NUM_LABELS_TO_EVAL:
        cur_img_text = get_cur_img_html_text(imgs_ids[cur_img_idx])
        cur_img_idx += 1
        if cur_img_text is not None:
            num_found_imgs += 1
            imgs_texts.append(cur_img_text)
            pbar.update(1)

    html_text += ''.join(imgs_texts) + '    </body>\n</html>'
    cur_output_html_path = output_html_path.format(cur_labels_src, labels_category)
    os.makedirs(os.path.dirname(cur_output_html_path), exist_ok=True)
    with open(cur_output_html_path, 'w') as out_f:
        out_f.write(html_text)


if __name__ == '__main__':
    main()
