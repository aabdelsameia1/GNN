# src/data_utils.py

import torch
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch.nn as nn


def get_zinc_dataset(root='../data/ZINC', batch_size=64, subset=True):
    """
    Loads the ZINC dataset from the specified root directory.
    
    Args:
        root (str): Path to the dataset folder.
        batch_size (int): Batch size for DataLoader.
    
    Returns:
        (DataLoader, DataLoader, DataLoader): train, val, and test loaders.
    """
    train_dataset = ZINC(root, split='train', subset=subset)
    val_dataset = ZINC(root, split='val', subset=subset)
    test_dataset = ZINC(root, split='test', subset=subset)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader



def get_activation_fn(name):
    name = name.lower()
    if name == 'relu':
        return F.relu
    elif name == 'leakyrelu':
        return F.leaky_relu
    elif name == 'elu':
        return F.elu
    else:
        raise ValueError(f"Unsupported activation: {name}")


def get_activation_module(name):
    """
    For usage in nn.Sequential, we need an nn.Module (e.g. nn.ReLU).
    """
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif name == 'elu':
        return nn.ELU()
    else:
        raise ValueError(f"Unsupported activation module: {name}")