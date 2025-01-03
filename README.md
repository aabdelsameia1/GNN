# GNN Playground

This repository contains various Graph Neural Network architectures, trained on the ZINC dataset.  
We explore GCN, GIN, and an optional Graph Transformer approach.

## Project Structure

- **data/**: Contains ZINC and other raw data.
- **images/**: Contains PDF/PNG images of graphs.
- **notebooks/**: Interactive Jupyter notebooks for exploration and experiments.
- **reports/**: Place to store logs, metrics, or final reports.
- **src/**
  - **data_utils.py**: Data loading functions.
  - **model.py**: Model architectures (GCN, GIN, Transformer).
  - **train.py**: Training and evaluation loops.
  - **main.py**: Command-line entry point to run experiments.
- **tests/**: Pytest-based unit tests.

## Getting Started

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run tests:
   ```bash
   pytest tests/
   ```
4. Train a model:
- **GIN**:
   ```bash
   python src/main.py --model_name gin --epochs 10 --batch_size 64
   ```
- **GINE** (which uses edge_attr):
   ```bash
   python src/main.py --model_name gine --epochs 10 --batch_size 64 --residual --batch_norm
   ```
- **GAT**:
   ```bash
   python src/main.py --model_name gat --epochs 5 --heads 8 --dropout 0.2
   ```
- **SAGE**:
   ```bash
   python src/main.py --model_name sage --epochs 20 --lr 0.0005
   ```

   
## Requirements

We rely on:
- Python 3.8+
- PyTorch
- PyTorch Geometric
- NumPy
- Matplotlib (optional for plotting)
- Pytest (optional for testing)

## Roadmap
- [x] GCN
- [x] GIN
- [ ] Graph Transformer (in progress)

## License

MIT.
