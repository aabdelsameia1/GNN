# # src/main.py

# import torch
# import torch.nn as nn
# from data_utils import get_zinc_dataset
# from model import get_model
# from train import train_one_epoch, evaluate
# import argparse

# def main():
#     parser = argparse.ArgumentParser(description="GNN Training")
#     parser.add_argument("--model_name", type=str, default="gcn", 
#                         help="Which model to use: gcn, gin, gat, sage, or transformer")
#     parser.add_argument("--hidden_dim", type=int, default=64)
#     parser.add_argument("--epochs", type=int, default=50)
#     parser.add_argument("--lr", type=float, default=0.001)
#     parser.add_argument("--batch_size", type=int, default=64)
#     parser.add_argument("--dropout", type=float, default=0.0,
#                         help="Dropout rate")
#     parser.add_argument("--activation", type=str, default="relu",
#                         help="Activation function: relu, leakyrelu, elu, etc.")
#     parser.add_argument("--pool", type=str, default="mean",
#                         help="Pooling method: mean, max, attention")
#     parser.add_argument("--residual", action="store_true",
#                         help="Use residual (skip) connections if set")
#     parser.add_argument("--batch_norm", action="store_true",
#                         help="Use batch normalization if set")
#     parser.add_argument("--heads", type=int, default=4,
#                         help="Number of attention heads for GAT")

#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # 1. Load data
#     train_loader, val_loader, test_loader = get_zinc_dataset(batch_size=args.batch_size)

#     # 2. Input feature dimension
#     sample_data = next(iter(train_loader))
#     in_channels = sample_data.x.shape[1]

#     # 3. Build model (pass new args to get_model)
#     model = get_model(
#         model_name=args.model_name,
#         in_channels=in_channels,
#         hidden_dim=args.hidden_dim,
#         out_channels=1,
#         dropout=args.dropout,
#         activation=args.activation,
#         pool=args.pool,
#         residual=args.residual,
#         batch_norm=args.batch_norm,
#         heads=args.heads
#     )
#     model.to(device)

#     # 4. Optimizer & loss
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     criterion = nn.MSELoss()

#     # 5. Training loop
#     for epoch in range(1, args.epochs + 1):
#         train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
#         val_loss = evaluate(model, val_loader, criterion, device)
#         print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

#     # 6. Test
#     test_loss = evaluate(model, test_loader, criterion, device)
#     print(f"Final Test Loss: {test_loss:.4f}")

# if __name__ == "__main__":
#     main()

# src/main.py

import torch
import torch.nn as nn
from data_utils import get_zinc_dataset
from model import get_model
from train import train_one_epoch, evaluate
import argparse

def main():
    parser = argparse.ArgumentParser(description="GNN Training")
    parser.add_argument("--model_name", type=str, default="gcn", 
                        help="Which model to use: gcn, gin, gat, sage, transformer, or gine")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate")
    parser.add_argument("--activation", type=str, default="relu",
                        help="Activation function: relu, leakyrelu, elu, etc.")
    parser.add_argument("--pool", type=str, default="mean",
                        help="Pooling method: mean, max, attention")
    parser.add_argument("--residual", action="store_true",
                        help="Use residual (skip) connections if set")
    parser.add_argument("--batch_norm", action="store_true",
                        help="Use batch normalization if set")
    parser.add_argument("--heads", type=int, default=4,
                        help="Number of attention heads for GAT/Transformer")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data
    train_loader, val_loader, test_loader = get_zinc_dataset(batch_size=args.batch_size)

    # 2. Determine in_channels and (if GINE) edge_dim
    sample_data = next(iter(train_loader))
    in_channels = sample_data.x.shape[1]

    # If model_name == 'gine', we might have edge_attr
    edge_dim = None
    # if args.model_name.lower() == 'gine' and hasattr(sample_data, 'edge_attr') and sample_data.edge_attr is not None:
    #     # edge_dim = sample_data.edge_attr.shape[1]
    #     edge_dim = sample_data.edge_attr.shape.numel()
    #     print(f"Detected edge_attr dimension: {edge_dim}")

    if args.model_name.lower() == 'gine':
        if sample_data.edge_attr is not None:
            if sample_data.edge_attr.dim() == 1:
                # We have shape [E], so each edge is just a scalar -> edge_dim=1
                edge_dim = 1
            else:
                # We have shape [E, X], so edge_dim=X
                edge_dim = sample_data.edge_attr.shape[1]
        else:
            edge_dim = 0

    # 3. Build model (pass new args to get_model, including edge_dim if relevant)
    model = get_model(
        model_name=args.model_name,
        in_channels=in_channels,
        hidden_dim=args.hidden_dim,
        out_channels=1,
        dropout=args.dropout,
        activation=args.activation,
        pool=args.pool,
        residual=args.residual,
        batch_norm=args.batch_norm,
        heads=args.heads,
        edge_dim=edge_dim  # Used only if model == 'gine'
    )
    model.to(device)

    # 4. Optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # 5. Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 6. Test
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
