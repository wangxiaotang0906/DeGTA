import numpy as np
import torch
import random

from torch_geometric.datasets import WebKB
import torch_geometric.transforms as T


# return cora dataset as pytorch geometric Data object together with 60/20/20 split, and list of cora IDs


def get_raw_text_cornell(use_text=False, seed=0):
    text = []
    data_name = 'cornell'
    dataset = WebKB('./datasets', data_name,
                        transform=T.NormalizeFeatures())
    data = dataset[0]
    data.train_mask = data.train_mask[:, seed]
    data.val_mask = data.val_mask[:, seed]
    data.test_mask = data.test_mask[:, seed]
    return data, text
