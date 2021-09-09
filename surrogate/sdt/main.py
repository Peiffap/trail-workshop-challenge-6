from __future__ import print_function

import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from model import SoftDecisionTree
import utils


class ImageDataset(Dataset):
    def __init__(self, model, dir_path, crop, emb_output=False):
        data_list = []
        target_list = []
        max_ind = 0
        filelist = []
        for root, subdirs, files in os.walk(dir_path):
            for file in files:
                filelist.append(os.path.join(root, file))
        random.shuffle(filelist)
        for filename in filelist:
            tensor = utils.img_to_tensor(filename, crop)
            with torch.no_grad():
                if not emb_output:
                    out = model(tensor)
                else:
                    emb, out = model(tensor)

            proba = torch.nn.functional.softmax(out[0], dim=0)

            label_ind = torch.argmax(proba).item()

            data_list.append(filename)
            target_list.append(label_ind)
            max_ind += 1
        self.data = data_list
        self.target = target_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x,y


def load_data(model, path, crop, emb_output=False):
    train_dataset = ImageDataset(model, path, crop, emb_output)
    train_loader = DataLoader(
        train_dataset,
        batch_size=50,
        shuffle=True,
        num_workers=4
    )

    return train_loader


def build_dt(args):
    model = SoftDecisionTree(args)

    return model


