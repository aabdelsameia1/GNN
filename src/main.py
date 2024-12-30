# src/main.py

import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import load_data, create_dataloaders
from model import GNNModel

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        batch.x, batch.edge_index, batch.y, batch.batch = (
            batch.x.to(device),
            batch.edge_index.to(device),
            batch.y.to(device),
            batch.batch.to(device)
        )

        optimizer.zero_grad()
        # Forward pass
        outputs = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(outputs, batch.y)
        # Backward + Optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs  # accumulate total loss
    
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            batch.x, batch.edge_index, batch.y, batch.batch = (
                batch.x.to(device),
                batch.edge_index.to(device),
                batch.y.to(device),
                batch.batch.to(device)
            )
            outputs = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(outputs, batch.y)
            total_loss += loss.item() * batch.num_graphs
    
    return total_loss / len(loader.dataset)


def main():
    # Hyperparameters (adjust as needed)
    batch_size = 32
    hidden_dim = 64
    num_features = 28  # This is typical for ZINC, but verify
    num_epochs = 10
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    train_dataset, val_dataset, test_dataset = load_data(root_dir='data')
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size
    )
    
    # 2. Initialize Model
    model = GNNModel(
        input_dim=num_features,
        hidden_dim=hidden_dim,
        output_dim=1  # e.g. for a regression task on ZINC
    ).to(device)
    
    # 3. Define Loss and Optimizer
    criterion = nn.MSELoss()  # if ZINC is a regression task
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Training Loop
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # 5. Final Evaluation on Test Set
    test_loss = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

    # You could save the model here if desired
    # torch.save(model.state_dict(), "saved_model.pth")


if __name__ == "__main__":
    main()
