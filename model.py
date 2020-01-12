import torch
import torch.nn as nn
import torch.nn.functional as F
from config_gqa import *
import numpy as np


class MLPModel(torch.nn.Module):
    def __init__(self, hidden_dim, input_dim, output_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)


class TrainingMLPModel(MLPModel):
    def __init__(self, hidden_dim, input_dim, output_dim):
        MLPModel.__init__(self, hidden_dim, input_dim, output_dim)

    def forward(self, x, num_descs, labels):
        output = F.relu(self.fc1(x))
        output = F.relu(self.fc2(output))
        output = F.softmax(self.fc3(output), dim=2)

        unpadded_imgs = [output[i, :num_descs[i], :] for i in range(num_descs.shape[0])]

        # list of best matching descriptors for each image's label
        matching_descs_dists = []
        for img_idx in range(x.shape[0]):
            cur_label = labels[img_idx]
            matching_desc_id = unpadded_imgs[img_idx][:, cur_label].argmax()
            matching_descs_dists.append(unpadded_imgs[img_idx][matching_desc_id, :])

        output = torch.stack(matching_descs_dists)
        return output


class InferenceMLPModel(MLPModel):
    def __init__(self, hidden_dim, input_dim, output_dim):
        MLPModel.__init__(self, hidden_dim, input_dim, output_dim)

    def forward(self, x, num_descs):
        output = F.relu(self.fc1(x))
        output = F.relu(self.fc2(output))
        output = F.softmax(self.fc3(output), dim=2)

        unpadded_imgs = [output[i, :num_descs[i], :] for i in range(num_descs.shape[0])]
        unpadded_imgs = [x.detach().cpu().numpy() for x in unpadded_imgs]
        pred_labels = [np.argmax(x, axis=1).tolist() for x in unpadded_imgs]
        pred_probs = [np.max(x, axis=1).tolist() for x in unpadded_imgs]

        return pred_labels, pred_probs
