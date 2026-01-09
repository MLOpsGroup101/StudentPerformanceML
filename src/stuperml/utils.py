from __future__ import annotations

from pathlib import Path
from typing import Tuple

import kagglehub
import torch
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _download_csv(dataset_slug: str) -> Path:
    dataset_dir = Path(kagglehub.dataset_download(dataset_slug))
    csvs = sorted(dataset_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No .csv files found in downloaded dataset at {dataset_dir}")
    return csvs[0]


def _validate_splits(train_size: float, val_size: float, test_size: float) -> None:
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must sum to 1.0")
    if any(s < 0 for s in (train_size, val_size, test_size)):
        raise ValueError("train_size/val_size/test_size must be non-negative")


def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("scaler", StandardScaler())]),
                make_column_selector(dtype_include=["number"]),
            ),
            (
                "cat",
                Pipeline([("ohe", _make_ohe())]),
                make_column_selector(dtype_include=["object", "category", "bool"]),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _split_data(
    X_np,
    y_np,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> Tuple[Tuple, Tuple, Tuple]:
    if val_size == 0.0 and test_size == 0.0:
        return (X_np, y_np), (X_np[:0], y_np[:0]), (X_np[:0], y_np[:0])

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X_np,
        y_np,
        test_size=(1.0 - train_size),
        random_state=seed,
        shuffle=True,
    )

    tmp_total = val_size + test_size
    if tmp_total == 0.0:
        return (X_train, y_train), (X_tmp, y_tmp), (X_tmp[:0], y_tmp[:0])

    if val_size == 0.0:
        return (X_train, y_train), (X_tmp[:0], y_tmp[:0]), (X_tmp, y_tmp)

    if test_size == 0.0:
        return (X_train, y_train), (X_tmp, y_tmp), (X_tmp[:0], y_tmp[:0])

    test_fraction_of_tmp = test_size / tmp_total
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=test_fraction_of_tmp,
        random_state=seed,
        shuffle=True,
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def _to_tensor(x, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype)