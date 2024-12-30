# src/data_utils.py

import torch
from torch_geometric.datasets import ZINC

def load_data(root_dir='data'):
    """
    Loads the ZINC dataset using PyTorch Geometric.
    
    Args:
        root_dir (str): The directory to download/store the dataset.
        
    Returns:
        tuple: A tuple (train_dataset, val_dataset, test_dataset).
    """
    train_dataset = ZINC(root=root_dir, split='train')
    val_dataset = ZINC(root=root_dir, split='val')
    test_dataset = ZINC(root=root_dir, split='test')
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    """
    Converts datasets into PyTorch dataloaders.
    
    Args:
        train_dataset: PyG or PyTorch dataset for training.
        val_dataset:   Validation dataset.
        test_dataset:  Test dataset.
        batch_size:    Batch size for each split.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    from torch_geometric.loader import DataLoader
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test to see if data loading works
    train_ds, val_ds, test_ds = load_data()
    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))
    print("Test samples:", len(test_ds))
