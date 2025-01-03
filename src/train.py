# src/train.py

import torch
import torch.nn.functional as F
import numpy as np

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_data in dataloader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()

        # Pass edge_attr to the model
        out = model(
            x=batch_data.x.float(),
            edge_index=batch_data.edge_index,
            batch=batch_data.batch,
            edge_attr=batch_data.edge_attr.float()  # <-- Added
        ).squeeze(-1)

        loss = criterion(out, batch_data.y.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_data.num_graphs

    return total_loss / len(dataloader.dataset)

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    for batch_data in dataloader:
        batch_data = batch_data.to(device)

        out = model(
            x=batch_data.x.float(),
            edge_index=batch_data.edge_index,
            batch=batch_data.batch,
            edge_attr=batch_data.edge_attr.float()  # <-- Added
        ).squeeze(-1)

        loss = criterion(out, batch_data.y.float())
        total_loss += loss.item() * batch_data.num_graphs

    return total_loss / len(dataloader.dataset)

def predict(model, dataloader, device):
    """
    Return predictions and targets for analysis (plotting, etc).
    """
    model.eval()
    all_preds, all_targets = [], []
    for batch_data in dataloader:
        batch_data = batch_data.to(device)

        out = model(
            x=batch_data.x.float(),
            edge_index=batch_data.edge_index,
            batch=batch_data.batch,
            edge_attr=batch_data.edge_attr.float()  # <-- Added
        ).squeeze(-1)

        all_preds.append(out.detach().cpu())
        all_targets.append(batch_data.y.cpu())

    return torch.cat(all_preds), torch.cat(all_targets)
