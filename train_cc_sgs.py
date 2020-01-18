from cc_batcher import get_dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from model import TrainingMLPModel, InferenceMLPModel
from config_cc import *
import numpy as np
import argparse
import os
from tqdm import tqdm


def main(args):
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_net = TrainingMLPModel(**mlp_params).to(device)

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
    cur_ckpt_path = ckpt_path.format(args.exp_name, '_and_atts' if WITH_ATTS else '', '{}')
    os.makedirs(os.path.dirname(cur_ckpt_path), exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Training epoch {epoch}'):
            inputs = data['descs'].float().to(device)
            num_descs = data['num_descs'].long().to(device)
            obj_labels = data['obj_label'].squeeze().long().to(device)
            att_labels = data['att_label'].squeeze().long().to(device) if WITH_ATTS else None

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            obj_outputs, att_outputs = training_net(inputs, num_descs, obj_labels, att_labels)
            loss = criterion(obj_outputs, obj_labels)
            if WITH_ATTS:
                loss += criterion(att_outputs, att_labels)
            loss.backward()
            optimizer.step()

            # print statistics
            if i % PRINT_EVERY == PRINT_EVERY - 1:  # print every N mini-batches
                print('[%d, %5d] Train loss: %.3f' % (epoch + 1, i + 1, loss.item()))

        # perform validation
        running_val_loss = 0
        torch.save(training_net.state_dict(), cur_ckpt_path.format(epoch))

        with torch.set_grad_enabled(False):
            for i, data in tqdm(enumerate(val_loader), total=len(val_loader), desc=f'Validation epoch {epoch}'):
                inputs = data['descs'].float().to(device)
                num_descs = data['num_descs'].long().to(device)
                obj_labels = data['obj_label'].squeeze().long().to(device)
                att_labels = data['att_label'].squeeze().long().to(device) if WITH_ATTS else None
                obj_outputs, att_outputs = training_net(inputs, num_descs, obj_labels, att_labels)
                running_val_loss += criterion(obj_outputs, obj_labels)
                if WITH_ATTS:
                    running_val_loss += criterion(att_outputs, att_labels).item()

            cur_val_loss = running_val_loss / NUM_VAL_EPOCHS
            print('Epoch %d Val loss: %.3f' % (epoch + 1, cur_val_loss))
            if best_val_loss > cur_val_loss:
                best_val_loss = cur_val_loss
                best_val_epoch = epoch
                val_loader = get_dataloader('val')
            else:
                if best_val_epoch + EARLY_STOPPING <= epoch:
                    print('Early stopping after {} epochs'.format(epoch))
                    print(f'Best val epoch was {best_val_epoch}')
                    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='exp0', type=str, help='Where to save results')
    args = parser.parse_args()

    main(args)
