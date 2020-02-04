relevant_data_file = '/specific/netapp5_2/gamir/achiya/vqa/gqa_max_loss/data/cc/cc_top_{}{}_objs_{}_atts{}_data_{}_{}.json'
cc_freqs_path = '/specific/netapp5_2/gamir/datasets/ConceptualCaptions/scene_graphs/vacancy/{}_freqs.json'
cc_sgs_path = '/specific/netapp5_2/gamir/datasets/ConceptualCaptions/scene_graphs/vacancy/sgs_{}.json'
gqa_objs_file = '/specific/netapp5_2/gamir/datasets/gqa/objects_dict.json'
gqa_atts_file = '/specific/netapp5_2/gamir/datasets/gqa/attributes_dict.json'
cc_descriptors_file = '/specific/netapp5_2/gamir/achiya/Downloads/firefox_downloads/googlebu_att.lmdb'
gqa_descriptors_file = '/specific/netapp5_2/gamir/datasets/gqa/orig_features_our_format_all.lmdb'
line_to_img_id_file = '/specific/netapp5_2/gamir/datasets/ConceptualCaptions/data_labels/{}_img_ids.json'
ckpt_path = '/specific/netapp5_2/gamir/achiya/vqa/gqa_max_loss/exps/cc/{}/objs{}_ckpts_epoch_{}.pt'
cc_metadata_path = '/specific/netapp5_2/gamir/datasets/ConceptualCaptions/data_labels/googletalk.json'
ATT_CATEGORIES_FILE = '/specific/netapp5_2/gamir/datasets/gqa/raw_data/att_categories.json'

NUM_TOP_OBJS = 10000
NUM_TOP_ATTS = 10000
GQA_LABELS_ONLY = True
MAX_NUM_OBJS = 100
DESCRIPTORS_DIM = 2048
NUM_EPOCHS = 20
EARLY_STOPPING = 4
PRINT_EVERY = 200
VAL_EVERY = 200
NUM_VAL_EPOCHS = 5
GQA_OVERSAMPLING_RATE = 1
CC_OVERSAMPLING_RATE = 0
WITH_ATTS = True
USE_ATT_CATEGORIES = WITH_ATTS and True
DEBUG = False
CATEGORIES_TO_DROP = ['hposition', 'place', 'realism', 'room', 'texture', 'vposition', 'company', 'depth', 'flavor',
                      'race', 'location', 'hardness', 'gender', 'brightness']

cc_train_loader_params = {'batch_size': 64,
                          'shuffle': True,
                          'num_workers': 8,
                          'drop_last': True}

gqa_train_loader_params = {'batch_size': 64,
                           'shuffle': True,
                           'num_workers': 1,
                           'drop_last': True}

val_loader_params = {'batch_size': 64,
                     'shuffle': False,
                     'num_workers': 1,
                     'drop_last': True}

mlp_params = {'hidden_dim': 256,
              'input_dim': 2048}

sampling_rates = [CC_OVERSAMPLING_RATE, GQA_OVERSAMPLING_RATE]


def get_relevant_data_file(gqa_only, num_objs, num_atts, categorize_atts, dset, add_gqa):
    return relevant_data_file.format(
        'gqa_' if gqa_only else '',
        num_objs,
        num_atts,
        '_categorized' if categorize_atts else '',
        dset,
        'add_gqa' if add_gqa else ''
    )
