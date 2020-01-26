from model.demo_gqa_batcher import get_dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from model.mlp_model import TrainingMLPModel, InferenceMLPModel
from model.config_gqa import *
import numpy as np
from model.utils import eval_batch_prediction_gqa, plot_cm, get_cm_path
import json
import argparse
import os
from sklearn.metrics import accuracy_score
from pdb import set_trace as trace


def main(args):
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_net = TrainingMLPModel(**mlp_params).to(device)
    inference_net = InferenceMLPModel(**mlp_params).to(device)

    optimizer = optim.Adam(training_net.parameters(), lr=3e-4)

    train_loader = get_dataloader('train')
    val_loader = get_dataloader('val')

    best_val_loss = np.inf
    best_val_epoch = 0
    num_val_batches = len(val_loader)
    if num_val_batches == 0:
        print('No validation batches!')
        return

    output_path = os.path.join('exps', args.exp_name)
    os.makedirs(output_path, exist_ok=True)
    cur_ckpt_path = ckpt_path.format(args.exp_name)

    all_val_labels = json.load(open(imgs_and_objs_align_dict_file.format('val'), 'r'))
    obj_names = list(json.load(open(obj_new_id_to_name_file, 'r')).keys())

    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(train_loader):
            inputs = data['descs'].float().to(device)
            num_descs = data['num_descs'].long().to(device)
            labels = data['label'].squeeze().long().to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = training_net(inputs, num_descs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            if i % PRINT_EVERY == PRINT_EVERY - 1:  # print every N mini-batches
                print('[%d, %5d] Train loss: %.3f' % (epoch + 1, i + 1, loss.item()))
                break

        # perform validation
        running_val_loss = 0
        inference_net.load_state_dict(training_net.state_dict())
        all_val_gt_labels, all_val_preds = [], []

        for i, data in enumerate(val_loader):
            inputs = data['descs'].float().to(device)
            num_descs = data['num_descs'].long().to(device)
            labels = data['label'].squeeze().long().to(device)
            img_ids = data['img_id']
            train_outputs = training_net(inputs, num_descs, labels)
            running_val_loss += criterion(train_outputs, labels).item()

            inference_outputs = inference_net(inputs, num_descs)
            trace()
            relevant_gt_labels, pred_labels = eval_batch_prediction_gqa(all_val_labels, inference_outputs, img_ids)
            all_val_gt_labels += relevant_gt_labels
            all_val_preds += pred_labels

        accuracy = round(accuracy_score(all_val_gt_labels, all_val_preds), 5)
        plot_cm(all_val_preds, all_val_gt_labels, obj_names, get_cm_path(output_path, epoch, accuracy))

        cur_val_loss = running_val_loss / NUM_VAL_EPOCHS
        print('Epoch %d Val loss: %.3f' % (epoch + 1, cur_val_loss))
        if best_val_loss > cur_val_loss:
            best_val_loss = cur_val_loss
            best_val_epoch = epoch
            torch.save(training_net.state_dict(), cur_ckpt_path)
        else:
            if best_val_epoch + EARLY_STOPPING <= epoch:
                print('Early stopping after {} epochs'.format(epoch))
                print(f'Best val epoch was {best_val_epoch}')

                return
                # perform one last validation with best params
                # inference_net.load_state_dict(torch.load(ckpt_path))
                #
                # all_val_gt_labels, all_val_preds = [], []
                #
                # for i, data in enumerate(val_loader):
                #     inputs = data['descs'].float().to(device)
                #     num_descs = data['num_descs'].long().to(device)
                #     labels = data['label'].squeeze().long().to(device)
                #     img_ids = data['img_id']
                #     train_outputs = training_net(inputs, num_descs, labels)
                #     running_val_loss += criterion(train_outputs, labels).item()
                #
                #     inference_outputs = inference_net(inputs, num_descs)
                #     relevant_gt_labels, pred_labels = eval_batch_prediction(all_val_labels, inference_outputs, img_ids)
                #     all_val_gt_labels += relevant_gt_labels
                #     all_val_preds += pred_labels
                #
                # accuracy = round(accuracy_score(all_val_gt_labels, all_val_preds), 5)
                # plot_cm(all_val_preds, all_val_gt_labels, obj_names, get_cm_path(output_path, 'best', accuracy))

        val_loader = get_dataloader('val')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='exp0', type=str, help='Where to save results')
    args = parser.parse_args()

    main(args)
