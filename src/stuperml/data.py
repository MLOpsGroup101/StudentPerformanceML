from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import joblib
import kagglehub
import pandas as pd
import torch
import typer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset

from configs import data_cfg


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



class MyDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        cfg: Mapping[str, Any] = data_cfg,
    ) -> None:
        self.cfg = cfg
        self.split = split.lower()

        self.X: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None

        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: 'train', 'val', 'test'")

        x_path = self.cfg.get('data_folder') / f"X_{self.split}.pt"
        y_path = self.cfg.get('data_folder') / f"y_{self.split}.pt"
        if x_path.exists() and y_path.exists():
            self.X = torch.load(x_path)
            self.y = torch.load(y_path)

    def __len__(self) -> int:
        if self.X is None:
            raise RuntimeError("Dataset not initialized with preprocessed tensors.")
        return int(self.X.shape[0])

    def __getitem__(self, index: int):
        if self.X is None or self.y is None:
            raise RuntimeError("Dataset not initialized with preprocessed tensors.")
        return self.X[index], self.y[index]

    def preprocess(self) -> None:
        self.cfg.get('data_folder').mkdir(parents=True, exist_ok=True)

        csv_path = _download_csv("ankushnarwade/ai-impact-on-student-performance")
        df = pd.read_csv(csv_path)

        if self.cfg.get('target_col') not in df.columns:
            raise KeyError(f"Target column '{self.cfg.get('target_col')}' not found. Columns: {list(df.columns)}")

        dropped: Sequence[str] = self.cfg.get("dropped_columns", [])
        train_size = float(self.cfg.get("train_size", 0.8))
        val_size = float(self.cfg.get("val_size", 0.1))
        test_size = float(self.cfg.get("test_size", 0.1))
        seed = int(self.cfg.get("seed", 42))

        _validate_splits(train_size, val_size, test_size)

        y_np = df[self.cfg.get('target_col')].to_numpy()
        X_df = df.drop(columns=[self.cfg.get('target_col'), *dropped], errors="ignore")

        pre = _build_preprocessor()
        X_np = pre.fit_transform(X_df)

        try:
            feat_names = pre.get_feature_names_out().tolist()
        except Exception:
            feat_names = []

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = _split_data(
            X_np, y_np, train_size, val_size, test_size, seed
        )

        torch.save(_to_tensor(X_train), self.cfg.get('data_folder') / "X_train.pt")
        torch.save(_to_tensor(X_val), self.cfg.get('data_folder') / "X_val.pt")
        torch.save(_to_tensor(X_test), self.cfg.get('data_folder') / "X_test.pt")

        torch.save(_to_tensor(y_train), self.cfg.get('data_folder') / "y_train.pt")
        torch.save(_to_tensor(y_val), self.cfg.get('data_folder') / "y_val.pt")
        torch.save(_to_tensor(y_test), self.cfg.get('data_folder') / "y_test.pt")

        (self.cfg.get('data_folder') / "feature_names.json").write_text(json.dumps(feat_names))
        joblib.dump(pre, self.cfg.get('data_folder') / "preprocessor.joblib")


def preprocess() -> None:
    MyDataset(cfg=data_cfg).preprocess()


if __name__ == "__main__":
    typer.run(preprocess)