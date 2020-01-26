import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLPModel(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output = F.relu(self.fc1(x))
        output = F.relu(self.fc2(output))
        output = F.softmax(self.fc3(output), dim=2)
        return output


class TrainingMLPModel(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim, objs_output_dim, atts_output_dim=0):
        super(TrainingMLPModel, self).__init__()
        self.objs_mlp = MLPModel(hidden_dim, input_dim, objs_output_dim)
        if atts_output_dim > 0:
            self.atts_mlp = MLPModel(hidden_dim, input_dim, atts_output_dim)

    @staticmethod
    def collate_loss(preds, labels, num_descs, criterion):
        loss = 0
        for img_idx, num_img_descs in enumerate(num_descs):
            relevant_labels = labels[img_idx, :num_img_descs]
            relevant_preds = preds[img_idx, :num_img_descs, :]
            labels_mask = [i for i in range(len(relevant_labels)) if relevant_labels[i] != -1]
            if len(labels_mask) > 0:
                loss += criterion(relevant_preds[labels_mask], relevant_labels[labels_mask])

        return loss

    def forward(self, x, num_descs, num_labels_per_img, obj_labels, att_labels=None):
        objs_outputs = self.objs_mlp(x)

        unpadded_imgs_objs = [objs_outputs[i, :num_descs[i], :] for i in range(num_descs.shape[0])]
        if att_labels is not None:
            atts_outputs = self.atts_mlp(x)
            unpadded_imgs_atts = [atts_outputs[i, :num_descs[i], :] for i in range(num_descs.shape[0])]

        # list of best matching descriptors for each image's label
        matching_descs_obj_dists = []
        matching_descs_att_dists = []
        total_labels_count = 0
        is_strong_supervision = (num_labels_per_img[0] is None)
        if is_strong_supervision:
            return objs_outputs, atts_outputs
        else:
            for img_idx in range(x.shape[0]):
                for label_id in range(num_labels_per_img[img_idx]):
                    cur_obj_label = obj_labels[total_labels_count]
                    cur_att_label = att_labels[total_labels_count]
                    matching_obj_probs = unpadded_imgs_objs[img_idx][:, cur_obj_label]

                    if cur_att_label == -1:
                        matching_desc_id = matching_obj_probs.argmax()
                        matching_descs_obj_dists.append(unpadded_imgs_objs[img_idx][matching_desc_id, :])
                    else:
                        matching_att_probs = unpadded_imgs_atts[img_idx][:, cur_att_label]
                        matching_desc_id = (matching_obj_probs * matching_att_probs).argmax()
                        matching_descs_obj_dists.append(unpadded_imgs_objs[img_idx][matching_desc_id, :])
                        matching_descs_att_dists.append(unpadded_imgs_atts[img_idx][matching_desc_id, :])

                    total_labels_count += 1

            obj_dists = torch.stack(matching_descs_obj_dists)
            att_dists = None
            if len(matching_descs_att_dists) > 0:
                att_dists = torch.stack(matching_descs_att_dists)
            return obj_dists, att_dists


class InferenceMLPModel(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim, objs_output_dim, atts_output_dim=0):
        super(InferenceMLPModel, self).__init__()
        self.objs_mlp = MLPModel(hidden_dim, input_dim, objs_output_dim)
        self.predict_atts = False
        if atts_output_dim > 0:
            self.atts_mlp = MLPModel(hidden_dim, input_dim, atts_output_dim)
            self.predict_atts = True

    def forward(self, x, num_descs, predict_atts=False):
        objs_outputs = self.objs_mlp(x)

        def to_numpy(tensors_list):
            return [tensor.detach().cpu().numpy() for tensor in tensors_list]

        unpadded_imgs_objs = to_numpy([objs_outputs[i, :num_descs[i], :] for i in range(num_descs.shape[0])])
        pred_obj_labels = [np.argmax(x, axis=1).tolist() for x in unpadded_imgs_objs]
        pred_obj_probs = [np.max(x, axis=1).tolist() for x in unpadded_imgs_objs]
        pred_att_labels, pred_att_probs = None, None

        if self.predict_atts:
            atts_outputs = self.atts_mlp(x)
            unpadded_imgs_atts = to_numpy([atts_outputs[i, :num_descs[i], :] for i in range(num_descs.shape[0])])
            pred_att_labels = [np.argmax(x, axis=1).tolist() for x in unpadded_imgs_atts]
            pred_att_probs = [np.max(x, axis=1).tolist() for x in unpadded_imgs_atts]

        return pred_obj_labels, pred_obj_probs, pred_att_labels, pred_att_probs
