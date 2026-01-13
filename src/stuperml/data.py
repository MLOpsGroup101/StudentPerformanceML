import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from torch.utils.data import Dataset, TensorDataset
import torch
import typer

from configs import DataConfig, data_config
from stuperml.utils import (
    _download_csv,
    _validate_splits,
    _build_preprocessor,
    _to_tensor,
    _split_data,
)


class MyDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        cfg: DataConfig = data_config,
    ) -> None:
        self.cfg = cfg
        self.split = split.lower()

        self.X: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None

        if self.split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: 'train', 'val', 'test'")

        x_path = self.cfg.data_folder / f"X_{self.split}.pt"
        y_path = self.cfg.data_folder / f"y_{self.split}.pt"
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
        self.cfg.data_folder.mkdir(parents=True, exist_ok=True)

        csv_path = _download_csv("ankushnarwade/ai-impact-on-student-performance")
        df = pd.read_csv(csv_path)

        if self.cfg.target_col not in df.columns:
            raise KeyError(f"Target column '{self.cfg.target_col}' not found. Columns: {list(df.columns)}")

        dropped = self.cfg.dropped_columns
        train_size = float(self.cfg.train_size)
        val_size = float(self.cfg.val_size)
        test_size = float(self.cfg.test_size)
        seed = int(self.cfg.seed)

        _validate_splits(train_size, val_size, test_size)
        y_np = df[self.cfg.target_col].to_numpy()
        X_df = df.drop(columns=[self.cfg.target_col, *dropped], errors="ignore")

        pre = _build_preprocessor()
        X_np = pre.fit_transform(X_df)

        try:
            feat_names = pre.get_feature_names_out().tolist()
        except Exception:
            feat_names = []

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = _split_data(
            X_np, y_np, train_size, val_size, test_size, seed
        )

        torch.save(_to_tensor(X_train), self.cfg.data_folder / "X_train.pt")
        torch.save(_to_tensor(X_val), self.cfg.data_folder / "X_val.pt")
        torch.save(_to_tensor(X_test), self.cfg.data_folder / "X_test.pt")

        torch.save(_to_tensor(y_train), self.cfg.data_folder / "y_train.pt")
        torch.save(_to_tensor(y_val), self.cfg.data_folder / "y_val.pt")
        torch.save(_to_tensor(y_test), self.cfg.data_folder / "y_test.pt")

        (self.cfg.data_folder / "feature_names.json").write_text(json.dumps(feat_names))
        joblib.dump(pre, self.cfg.data_folder / "preprocessor.joblib")

    def load_data(self) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
        data_dir: Path
        data_dir = self.cfg.data_folder

        train_features = torch.load(data_dir / "X_train.pt")
        train_target = torch.load(data_dir / "y_train.pt")

        val_features = torch.load(data_dir / "X_val.pt")
        val_target = torch.load(data_dir / "y_val.pt")

        test_features = torch.load(data_dir / "X_test.pt")
        test_target = torch.load(data_dir / "y_test.pt")

        train_set = TensorDataset(train_features, train_target)
        val_set = TensorDataset(val_features, val_target)
        test_set = TensorDataset(test_features, test_target)

        return train_set, val_set, test_set


def main() -> None:
    dataset_manager = MyDataset(cfg=data_config)

    dataset_manager.preprocess()
    train_set, val_set, test_set = dataset_manager.load_data()

    for dataset in [train_set, val_set, test_set]:
        print(f"rows:{len(dataset)} \t features:{len(dataset[0][0])} \t target:{len(dataset[1][0])}")


if __name__ == "__main__":
    typer.run(main)
