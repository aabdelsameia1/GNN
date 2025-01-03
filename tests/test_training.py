# tests/test_training.py

import torch
import torch.nn as nn
from src.model import get_model
from src.train import train_one_epoch, evaluate
from src.data_utils import get_zinc_dataset

def test_train_one_epoch():
    device = torch.device("cpu")
    train_loader, val_loader, _ = get_zinc_dataset(batch_size=8)
    sample_data = next(iter(train_loader))
    in_channels = sample_data.x.shape[1]
    
    model = get_model("gcn", in_channels=in_channels, hidden_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    initial_val_loss = evaluate(model, val_loader, criterion, device)
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    post_val_loss = evaluate(model, val_loader, criterion, device)

    # Just check that training step reduces or at least changes the loss
    assert post_val_loss <= initial_val_loss or abs(post_val_loss - initial_val_loss) < 0.01
