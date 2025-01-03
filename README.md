# GNN

This repository contains a collection of Graph Neural Network (GNN) architectures tailored for molecular property prediction on the ZINC dataset. We explore **GCN**, **GIN**, **GINE** (edge-aware GIN variant), **GAT**, **GraphSAGE**, and an **in-progress Graph Transformer** approach.

Our main focus is to compare different designs (e.g., dropout, attention pooling, residual connections, batch normalization) and observe their impact on performance, specifically for regression tasks in chemistry.

---

## Project Structure

```
.
├── data/
│   └── ZINC/                  # ZINC dataset files and any other raw data
├── images/
│   └── ...                    # PDF/PNG plots, charts, molecule grids, etc.
├── notebooks/
│   ├── Models.ipynb           # Main notebook: runs, logs, & plots final results
│   └── GraphTransformer.ipynb # Early design & experiments with Graph Transformer
├── reports/
│   └── ...                    # Logs, metrics, or final PDF reports
├── src/
│   ├── data_utils.py          # Data loading utilities
│   ├── model.py               # Model architectures (GCN, GIN, GINE, GAT, SAGE, etc.)
│   ├── train.py               # Training & evaluation loops
│   └── main.py                # Command-line entry point to run experiments
│   └── GraphTransformer.ipynb # Inital Expermints with Graph Transformers
├── tests/
│   └── ...                    # Pytest-based unit tests
└── requirements.txt
```

---

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aabdelsameia1/gnn.git
   cd gnn
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   - *Requires Python 3.8+*
   - Ensure you have PyTorch and PyTorch Geometric installed (CUDA versions as appropriate).

3. **Run tests** (optional):
   ```bash
   pytest tests/
   ```
   This runs the unit tests to verify everything is set up correctly.

4. **Train a model** using the command-line interface:

   **Available command-line arguments** (all optional unless stated otherwise):

   - `--model_name` (str, default=`"gcn"`): Which model to run. Options include:
     - `"gcn"`, `"gin"`, `"gine"`, `"gat"`, `"sage"`, `"transformer"`
   - `--hidden_dim` (int, default=`64`): Size of the hidden (latent) dimension.
   - `--epochs` (int, default=`50`): Number of training epochs.
   - `--lr` (float, default=`0.001`): Learning rate (Adam optimizer).
   - `--batch_size` (int, default=`64`): Batch size for both training and evaluation.
   - `--dropout` (float, default=`0.0`): Dropout rate for regularization.
   - `--activation` (str, default=`"relu"`): Activation function (e.g., `"leakyrelu"`, `"elu"`).
   - `--pool` (str, default=`"mean"`): Pooling method to use (`"mean"`, `"max"`, or `"attention"`).
   - `--residual` (flag): Include residual (skip) connections if set.
   - `--batch_norm` (flag): Include batch normalization if set.
   - `--heads` (int, default=`4`): Number of attention heads (for GAT or Transformer).

   **Example Commands**:

   - **GIN** (Graph Isomorphism Network):
     ```bash
     python src/main.py --model_name gin --epochs 10 --batch_size 64
     ```

   - **GINE** (GIN + edge attributes):
     ```bash
     python src/main.py --model_name gine --epochs 10 --batch_size 64 --residual --batch_norm
     ```

   - **GAT** (Graph Attention Network):
     ```bash
     python src/main.py --model_name gat --epochs 5 --heads 8 --dropout 0.2
     ```

   - **GraphSAGE**:
     ```bash
     python src/main.py --model_name sage --epochs 20 --lr 0.0005
     ```

   Adjust additional flags such as `--activation`, `--pool`, `--edge_dim`, etc., as desired.

---

## Key Results

After experimenting with 15 variations (5 architectures \(\times\) 3 parameter configurations each), these are the top highlights:

- **GINE\_V3\_Attn\_Residual\_BN** consistently achieves the \textbf{best performance} with:
  - \(\mathrm{MSE} \approx 1.0\)
  - \(\mathrm{MAE} \approx 0.51\)
  - \(R^2 \approx 0.80\)
- **Edge-aware models** (\texttt{GINE}) outperform edge-agnostic ones (\texttt{GIN}, \texttt{GCN}, etc.) by leveraging bond attributes.
- **Advanced features** (attention pooling, residual connections, batch normalization) markedly improve generalization and stability.
- **Larger training subsets** result in higher accuracy for all GNN variants, emphasizing the benefit of more data.

Check `notebooks/Models.ipynb` for detailed plots and logs of all experiments.

---

## Where to Find Results

- **notebooks/Models.ipynb**: The central experiments and final performance plots are found here, as used in our paper.  
- **Command-line runs** (`main.py`): Logs progress and metrics to your console or predefined folders.

---

## Key Features

- **Edge-Aware GNNs (GINE)**: Incorporates bond-type information (`edge_attr`) for molecular graphs, often yielding better performance on the ZINC dataset.
- **Configurable Modules**: Easily enable or disable dropout, batch normalization, residual connections, and attention pooling.
- **Multiple Architectures**: Compare GCN, GIN, GINE, GAT, GraphSAGE, and a prototype Graph Transformer under a unified training pipeline.
- **Logging & Metrics**: Track MSE, MAE, and \(R^2\) throughout training and validation; store final results in organized folders.

---

## Where to Find Results

- **notebooks/Models.ipynb**: Contains the main experiments and the final performance plots.  
  All the results and figures shown in our paper are generated here.
- **Command-line runs** (`main.py`): Output partial logs and metrics to your console or specified folders.

---

## Requirements

- Python 3.8+
- PyTorch (GPU recommended)
- PyTorch Geometric
- NumPy
- Matplotlib (for plotting)
- Pytest (for testing)

Install all dependencies via:
```bash
pip install -r requirements.txt
```

---

## Roadmap

- [x] GCN  
- [x] GIN  
- [x] **GINE (edge-aware)**  
- [x] GAT  
- [x] GraphSAGE  
- [ ] Graph Transformer (in progress)  

We plan to refine the Graph Transformer approach, experiment with deeper GNN layers, and potentially add self-supervised pretraining modules. All major experimental logs and metrics are stored in `notebooks/Models.ipynb` and in the `reports/` folder.

---

## License

This project is licensed under the [MIT License](./LICENSE)