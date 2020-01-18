import json
from tqdm import tqdm


metadata = json.load(open('/specific/netapp5_2/gamir/datasets/ConceptualCaptions/data_labels/googletalk.json'))

train_data = {}
val_data = {}

for img_data in tqdm(metadata['images']):
    graph_line = img_data['id'].split('_')[0]
    if img_data['split'] == 'train':
        train_data[graph_line] = img_data['id']
    else:
        val_data[graph_line] = img_data['id']

with open('/specific/netapp5_2/gamir/datasets/ConceptualCaptions/data_labels/train_img_ids.json', 'w') as f:
    json.dump(train_data, f, indent=4)

with open('/specific/netapp5_2/gamir/datasets/ConceptualCaptions/data_labels/val_img_ids.json', 'w') as f:
    json.dump(val_data, f, indent=4)
