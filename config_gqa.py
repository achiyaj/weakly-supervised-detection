from socket import gethostname

obj_freqs_file = '/specific/netapp5_2/gamir/datasets/gqa/obj_counts.json'
obj_orig_id_to_name_file = '/specific/netapp5_2/gamir/datasets/gqa/objects_dict.json'
obj_new_id_to_name_file = '/specific/netapp5_2/gamir/achiya/vqa/gqa_max_loss/data/gqa/obj_new_id_to_name.json'
relevant_imgs_file = '/specific/netapp5_2/gamir/achiya/vqa/gqa_max_loss/data/gqa/imgs_and_objs_{}.json'
sgs_file = '/specific/netapp5_2/gamir/datasets/gqa/raw_data/{}_sceneGraphs.json'
ckpt_path = '/specific/netapp5_2/gamir/achiya/vqa/gqa_max_loss/gqa/exps/{}/objs_ckpts.pt'
imgs_and_objs_dict_file = '/specific/netapp5_2/gamir/achiya/vqa/gqa_max_loss/data/gqa/imgs_and_objs_dict_{}.json'
imgs_and_objs_align_dict_file = '/specific/netapp5_2/gamir/achiya/vqa/gqa_max_loss/data/gqa/imgs_and_objs_align_{}.json'
cm_filename = 'epoch_{}_acc_{}.png'

hostname = gethostname()
if hostname in ['rack-gamir-g01', 'rack-gamir-g02']:
    descriptors_file = '/specific/disk1/home/gamir/achiya/gqa/orig_features_our_format_all.h5'
else:
    descriptors_file = '/specific/netapp5_2/gamir/datasets/gqa/orig_features_our_format_all.h5'

NUM_TOP_OBJS = 20
NUM_EPOCHS = 20
EARLY_STOPPING = 4
PRINT_EVERY = 5
VAL_EVERY = 5
NUM_VAL_EPOCHS = 5
MAX_NUM_OBJS = 126
DESCRIPTORS_DIM = 2048
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

train_loader_params = {'batch_size': 64,
                       'shuffle': True,
                       'num_workers': 1,
                       'drop_last': True}

val_loader_params = {'batch_size': 2048,
                     'shuffle': True,
                     'num_workers': 1,
                     'drop_last': True}


mlp_params = {'hidden_dim': 256,
              'input_dim': 2048,
              'output_dim': NUM_TOP_OBJS}
