import pytest
import torch
import numpy as np
from stuperml.utils import _validate_splits, _split_data, _build_preprocessor, _to_tensor


def test_validate_splits_valid():
    _validate_splits(0.7, 0.2, 0.1)

def test_validate_splits_invalid_sum():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        _validate_splits(0.7, 0.2, 0.2)


def test_validate_splits_negative():
    with pytest.raises(ValueError, match="must be non-negative"):
        _validate_splits(-0.1, 0.2, 0.9)

def test_split_data_no_val():
    X = np.random.randn(100, 5)
    y = np.random.randn(100)
    train, val, test = _split_data(X, y, 0.8, 0.0, 0.2, seed=42)
    assert len(train[0]) == 80
    assert len(val[0]) == 0
    assert len(test[0]) == 20

def test_build_preprocessor():
    preprocessor = _build_preprocessor()
    assert preprocessor is not None
    # Test with sample data
    import pandas as pd
    df = pd.DataFrame({
        'num1': [1, 2, 3],
        'num2': [4.0, 5.0, 6.0],
        'cat1': ['a', 'b', 'c'],
        'cat2': [True, False, True]
    })
    X_transformed = preprocessor.fit_transform(df)
    assert X_transformed.shape[0] == 3

def test_to_tensor():
    arr = np.array([1, 2, 3])
    tensor = _to_tensor(arr)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert torch.equal(tensor, torch.tensor([1., 2., 3.]))