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
    def __init__(self, hidden_dim, input_dim, objs_output_dim, device, att_categories=None, atts_output_dim=0):
        super(TrainingMLPModel, self).__init__()
        self.objs_mlp = MLPModel(hidden_dim, input_dim, objs_output_dim).to(device)
        self.use_att_categories = not (att_categories is None)
        if atts_output_dim > 0:
            if self.use_att_categories:
                self.att_categories = att_categories
                self.atts_models_dict = \
                    {key: MLPModel(hidden_dim, input_dim, len(value)).to(device) for key, value in self.att_categories.items()}
            else:
                self.atts_mlp = MLPModel(hidden_dim, input_dim, atts_output_dim).to(device)

    @staticmethod
    def collate_loss(preds, labels, num_descs, criterion):
        loss = 0
        if type(labels) is dict:  # i.e. this are categorized attributes labels
            cur_descs_count = 0
            for img_idx, num_img_descs in enumerate(num_descs):
                for category_name, category_labels in labels.items():
                    relevant_labels = category_labels[cur_descs_count: (cur_descs_count + num_img_descs)]
                    relevant_preds = preds[img_idx][category_name]
                    labels_mask = [i for i in range(len(relevant_labels)) if relevant_labels[i] != -1]
                    if len(labels_mask) > 0:
                        loss += criterion(relevant_preds[0, labels_mask, :], relevant_labels[labels_mask])
                cur_descs_count += num_img_descs

        else:
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
            if self.use_att_categories:
                atts_outputs = []
                for i in range(num_descs.shape[0]):
                    cur_descs = x[i, :num_descs[i], :].unsqueeze(0)
                    atts_outputs.append({key: value(cur_descs) for key, value in self.atts_models_dict.items()})
            else:
                atts_outputs = self.atts_mlp(x)
                unpadded_imgs_atts = [atts_outputs[i, :num_descs[i], :] for i in range(num_descs.shape[0])]

        # list of best matching descriptors for each image's label
        matching_descs_obj_dists = []
        if self.use_att_categories:
            matching_descs_att_dists = {key: {'dists': [], 'labels': []} for key in self.att_categories.keys()}
        else:
            matching_descs_att_dists = []
        total_labels_count = 0
        is_strong_supervision = (num_labels_per_img[0] is None)

        if is_strong_supervision:
            return objs_outputs, atts_outputs
        else:
            for img_idx in range(x.shape[0]):
                for label_id in range(num_labels_per_img[img_idx]):
                    cur_obj_label = obj_labels[total_labels_count]
                    matching_obj_probs = unpadded_imgs_objs[img_idx][:, cur_obj_label]
                    if self.use_att_categories:
                        cur_att_labels = \
                            {key: att_labels[key][total_labels_count] for key in self.att_categories.keys()}
                        found_att = False
                        for key, att_label in cur_att_labels.items():
                            if att_label != -1:
                                found_att = True
                                category_and_att = (key, att_label)
                                break
                        if found_att:
                            att_category, att_label = category_and_att
                            cur_category_att_probs = atts_outputs[img_idx][att_category][0, :, :]
                            matching_att_probs = cur_category_att_probs[:, att_label]
                            matching_desc_id = (matching_obj_probs * matching_att_probs).argmax()
                            relevant_obj_dist = unpadded_imgs_objs[img_idx][matching_desc_id, :]
                            relevant_att_dist = cur_category_att_probs[matching_desc_id, :]
                            matching_descs_obj_dists.append(relevant_obj_dist)
                            matching_descs_att_dists[att_category]['dists'].append(relevant_att_dist)
                            matching_descs_att_dists[att_category]['labels'].append(att_label)
                        else:  # didn't find an attribute label
                            matching_desc_id = matching_obj_probs.argmax()
                            matching_descs_obj_dists.append(unpadded_imgs_objs[img_idx][matching_desc_id, :])
                    else:
                        cur_att_label = att_labels[total_labels_count]
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
                if self.use_att_categories:
                    att_dists = {key: {} for key in self.att_categories.keys()}
                    for att_category, att_data in matching_descs_att_dists.items():
                        if len(att_data['dists']) > 0:  # at least one distributions for this category
                            att_dists[att_category] = {
                                'dists': torch.stack(att_data['dists']),
                                'labels': torch.stack(att_data['labels'])
                            }
                else:
                    att_dists = torch.stack(matching_descs_att_dists)
            return obj_dists, att_dists


class InferenceMLPModel(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim, objs_output_dim, device, atts_output_dim=0, att_categories=None):
        super(InferenceMLPModel, self).__init__()
        self.objs_mlp = MLPModel(hidden_dim, input_dim, objs_output_dim).to(device)
        self.predict_atts = False
        if atts_output_dim > 0:
            self.predict_atts = True
            if att_categories:
                self.categorize_atts = True
                self.att_categories = att_categories
                self.atts_models_dict = {key: MLPModel(hidden_dim, input_dim, len(value)).to(device) for key, value in
                                         self.att_categories.items()}
            else:
                self.atts_mlp = MLPModel(hidden_dim, input_dim, atts_output_dim).to(device)

    def forward(self, x, num_descs):
        objs_outputs = self.objs_mlp(x)

        def to_numpy(tensors_list):
            return [tensor.detach().cpu().numpy() for tensor in tensors_list]

        unpadded_imgs_objs = to_numpy([objs_outputs[i, :num_descs[i], :] for i in range(num_descs.shape[0])])
        pred_obj_labels = [np.argmax(x, axis=1).tolist() for x in unpadded_imgs_objs]
        pred_obj_probs = [np.max(x, axis=1).tolist() for x in unpadded_imgs_objs]
        pred_att_labels, pred_att_probs = None, None

        if self.predict_atts:
            if self.categorize_atts:
                pred_att_labels = {}
                pred_att_probs = {}
                for key, model in self.atts_models_dict.items():
                    cur_category_outputs = self.atts_models_dict[key](x)
                    unpadded_imgs_atts = to_numpy(
                        [cur_category_outputs[i, :num_descs[i], :] for i in range(num_descs.shape[0])])
                    cur_pred_att_labels = [np.argmax(x, axis=1).tolist() for x in unpadded_imgs_atts][0]
                    cur_pred_att_probs = [np.max(x, axis=1).tolist() for x in unpadded_imgs_atts][0]
                    pred_att_labels[key] = cur_pred_att_labels
                    pred_att_probs[key] = cur_pred_att_probs
            else:
                atts_outputs = self.atts_mlp(x)
                unpadded_imgs_atts = to_numpy([atts_outputs[i, :num_descs[i], :] for i in range(num_descs.shape[0])])
                pred_att_labels = [np.argmax(x, axis=1).tolist() for x in unpadded_imgs_atts]
                pred_att_probs = [np.max(x, axis=1).tolist() for x in unpadded_imgs_atts]

        return pred_obj_labels, pred_obj_probs, pred_att_labels, pred_att_probs
