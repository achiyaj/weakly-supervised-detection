import sys
sys.path.insert(0, "/specific/netapp5_2/gamir/achiya/vqa/gqa_max_loss/weakly-supervised-detection/")

from model.cc_batcher import get_cc_dataloader
from model.config_cc import *
if GQA_WEAK_SUPERVISION:
    from model.gqa_weak_supervision_batcher import get_gqa_dataloader
else:
    from model.gqa_batcher import get_gqa_dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from model.mlp_model import TrainingMLPModel
import numpy as np
import argparse
import os
from tqdm import tqdm
from utils.multiloader import MultiLoader
import json
from socket import gethostname
from glob import glob


def get_model_loss(model, data, criterion, device, optimizer, is_train):
    inputs = data['descs'].float().to(device)
    num_descs = data['num_descs'].long().to(device)
    obj_labels = data['obj_labels'].long().to(device)
    att_labels = data['att_labels']
    categorize_atts = type(att_labels) is dict

    if WITH_ATTS:
        if not categorize_atts:
            att_labels = att_labels.squeeze().long().to(device)
            att_labels_packed = att_labels[att_labels != -1]
        else:
            # att_labels = {k: v.squeeze().long().to(device) if v is not None else v for k, v in att_labels.items()}
            att_labels = {k: v.long().to(device) if v is not None else v for k, v in att_labels.items()}
    else:
        att_labels = None

    num_labels_per_image = data['num_labels_per_image']
    is_strong_supervision = (data['supervision_type'] == 1)
    # zero the parameter gradients
    if is_train:
        optimizer.zero_grad()

    # forward + backward + optimize
    obj_outputs, att_outputs = model(inputs, is_strong_supervision, num_descs, num_labels_per_image, obj_labels, att_labels)
    if is_strong_supervision:
        loss = model.collate_loss(obj_outputs, obj_labels, num_descs, criterion)
        if att_outputs is not None:
            loss += model.collate_loss(att_outputs, att_labels, num_descs, criterion)
    else:
        if type(obj_outputs) is dict:
            loss = criterion(obj_outputs['dists'], obj_outputs['labels'].to(device))
        else:
            loss = criterion(obj_outputs, obj_labels)

        if att_outputs is not None:
            if not categorize_atts:
                loss += criterion(att_outputs, att_labels_packed)
            else:
                for atts_data in att_outputs.values():
                    if len(atts_data) > 0:
                        loss += criterion(atts_data['dists'], atts_data['labels'])

    return loss


def get_restore_ckpt(cur_ckpts_template):
    ckpt_files = glob(cur_ckpts_template)
    get_epoch_num = lambda ckpt_filename: int(ckpt_filename[:-3].split('_')[-1])
    ckpt_files.sort(key=get_epoch_num)
    latest_ckpt = ckpt_files[-1]
    latest_epoch = get_epoch_num(latest_ckpt)
    return latest_ckpt, latest_epoch + 1


def main(args):
    att_categories = None
    if USE_ATT_CATEGORIES:
        att_categories = json.load(open(ATT_CATEGORIES_FILE, 'r'))
        att_categories = \
            {key: list(value.keys()) for key, value in att_categories.items() if key not in CATEGORIES_TO_DROP}

    cc_train_loader = get_cc_dataloader('train', att_categories)
    obj_labels = cc_train_loader.dataset.get_obj_labels()
    att_labels = cc_train_loader.dataset.get_att_labels()
    gqa_train_datafile = cc_train_loader.dataset.get_datafile().replace('cc', 'gqa')
    gqa_val_datafile = gqa_train_datafile.replace('train', 'val')

    cc_train_getter = lambda: get_cc_dataloader('train', att_categories)
    gqa_train_getter = lambda: get_gqa_dataloader(obj_labels, att_labels, gqa_train_datafile, 'train', att_categories)
    gqa_val_loader = get_gqa_dataloader(obj_labels, att_labels, gqa_val_datafile, 'val', att_categories)

    train_multiloader = MultiLoader([cc_train_getter, gqa_train_getter], sampling_rates)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_net = TrainingMLPModel(**mlp_params, disentangle_objs_and_atts=DISENTANGLE_OBJS_AND_ATTS,
                                    objs_output_dim=len(obj_labels), atts_output_dim=len(att_labels),
                                    att_categories=att_categories).to(device)

    optimizer = optim.Adam(training_net.parameters(), lr=3e-4)
    best_val_loss = np.inf
    best_val_epoch = 0
    num_val_batches = len(gqa_val_loader)
    if num_val_batches == 0:
        print('No validation batches!')
        return

    start_epoch = 0
    if RESTORE_FROM_CKPT:
        cur_ckpts_template = ckpts_template.format(args.exp_name, '_and_atts' if WITH_ATTS else '', '*')
        cur_ckpt_path, start_epoch = get_restore_ckpt(cur_ckpts_template)
        try:
            training_net.load_state_dict(torch.load(cur_ckpt_path))
        except RuntimeError:
            print('Current model configuration does not match the checkpoint state dict!')
            return
        print(f'Continuing to train from epoch {start_epoch - 1} checkpoint!')
    else:
        cur_ckpt_path = ckpts_template.format(args.exp_name, '_and_atts' if WITH_ATTS else '', '{}')

    os.makedirs(os.path.dirname(cur_ckpt_path), exist_ok=True)

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f'Epoch: {epoch}')
        for i, data in tqdm(enumerate(train_multiloader), total=len(train_multiloader)):
            loss = get_model_loss(training_net, data, criterion, device, optimizer, is_train=True)
            loss.backward()
            optimizer.step()

            # print statistics
            if i % PRINT_EVERY == PRINT_EVERY - 1:  # print every N mini-batches
                print('[%d, %5d] Train loss: %.3f' % (epoch + 1, i + 1, loss.item()))
                if DEBUG:
                    break

        # perform validation
        running_val_loss = 0
        torch.save(training_net.state_dict(), cur_ckpt_path.format(epoch))

        with torch.set_grad_enabled(False):
            for i, data in tqdm(enumerate(gqa_val_loader), total=len(gqa_val_loader), desc=f'Validation epoch {epoch}'):
                running_val_loss += \
                    get_model_loss(training_net, data, criterion, device, optimizer, is_train=True).item()
            cur_val_loss = running_val_loss / NUM_VAL_EPOCHS
            print('Epoch %d Val loss: %.3f' % (epoch + 1, cur_val_loss))
            if best_val_loss > cur_val_loss:
                best_val_loss = cur_val_loss
                best_val_epoch = epoch
                gqa_val_loader = get_gqa_dataloader(obj_labels, att_labels, gqa_val_datafile, 'val', att_categories)
            else:
                if best_val_epoch + EARLY_STOPPING <= epoch:
                    print('Early stopping after {} epochs'.format(epoch))
                    print(f'Best val epoch was {best_val_epoch}')
                    return


if __name__ == '__main__':
    print('Running on PC: {}'.format(gethostname()))

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='exp0', type=str, help='Where to save results')
    args = parser.parse_args()

    main(args)
