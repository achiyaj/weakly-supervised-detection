import json
from collections import OrderedDict
from tqdm import tqdm


cc_sgs_file = '/specific/netapp5_2/gamir/datasets/ConceptualCaptions/scene_graphs/vacancy/sgs_{}.json'
gqa_freqs = '/specific/netapp5_2/gamir/datasets/gqa/{}_dict.json'
cc_freqs_out = '/specific/netapp5_2/gamir/datasets/ConceptualCaptions/scene_graphs/vacancy/{}_freqs.json'
filtered_sgs_out = '/specific/netapp5_2/gamir/datasets/ConceptualCaptions/scene_graphs/vacancy/sgs_{}_filtered.json'


def add_or_increment(key, in_dict):
    if key in in_dict:
        in_dict[key] += 1
    else:
        in_dict[key] = 1
    return


def main():
    obj_counts = {}
    att_counts = {}
    rel_counts = {}

    with open(cc_sgs_file.format('train'), 'r') as in_f:
        cc_sgs = json.load(in_f)
    for sg in tqdm(cc_sgs.values(), desc='Building Freqs'):
        for obj in sg['objects'].values():
            add_or_increment(obj['label'], obj_counts)
            for att in obj['attributes']:
                add_or_increment(att, att_counts)
        for rel in sg['relations'].values():
            add_or_increment(rel, rel_counts)

    obj_counts_ordered = OrderedDict(sorted(obj_counts.items(), key=lambda x: x[1], reverse=True))
    att_counts_ordered = OrderedDict(sorted(att_counts.items(), key=lambda x: x[1], reverse=True))
    rel_counts_ordered = OrderedDict(sorted(rel_counts.items(), key=lambda x: x[1], reverse=True))

    with open(cc_freqs_out.format('objs'), 'w') as obj_freqs_f:
        json.dump(obj_counts_ordered, obj_freqs_f, indent=4)

    with open(cc_freqs_out.format('atts'), 'w') as att_freqs_f:
        json.dump(att_counts_ordered, att_freqs_f, indent=4)

    with open(cc_freqs_out.format('rels'), 'w') as rel_freqs_f:
        json.dump(rel_counts_ordered, rel_freqs_f, indent=4)


if __name__ == '__main__':
    main()
