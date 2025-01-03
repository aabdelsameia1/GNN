# tests/test_data_utils.py

import pytest
import torch
from ..src.data_utils import get_zinc_dataset

def test_zinc_dataset_loading():
    train_loader, val_loader, test_loader = get_zinc_dataset(batch_size=8)
    # Basic checks
    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0

    # Grab a batch
    batch = next(iter(train_loader))
    assert batch.x is not None
    assert batch.edge_index is not None
    assert batch.y is not None
