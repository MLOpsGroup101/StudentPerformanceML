import json
import os
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from typing import Optional

from loguru import logger

import joblib
import pandas as pd

from torch.utils.data import Dataset, TensorDataset
import torch
import typer
from configs import DataConfig, data_config
from stuperml.logger import setup_logger
from stuperml.utils import (
    _download_csv,
    _download_csv_from_gcs,
    _validate_splits,
    _build_preprocessor,
    _to_tensor,
    _split_data,
)

setup_logger()


class MyDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        cfg: DataConfig = data_config,
    ) -> None:
        logger.debug(f"Initializing MyDataset with split='{split}' and cfg={cfg}")
        self.cfg = cfg
        self.split = split.lower()

        self.X: Optional[torch.Tensor] = None
        self.y: Optional[torch.Tensor] = None

        if self.split not in {"train", "val", "test"}:
            logger.error(f"Invalid split '{self.split}' provided.")
            raise ValueError("split must be one of: 'train', 'val', 'test'")
        logger.info(f"MyDataset initialized for split '{self.split}'.")

        x_path = self.cfg.data_folder / f"X_{self.split}.pt"
        y_path = self.cfg.data_folder / f"y_{self.split}.pt"
        if x_path.exists() and y_path.exists():
            logger.debug(f"Loading preprocessed tensors from {x_path} and {y_path}.")
            self.X = torch.load(x_path)
            self.y = torch.load(y_path)
            logger.info(f"Loaded {self.X.size(0)} samples for split '{self.split}'.")
        else:
            logger.warning(f"Preprocessed data not found at {x_path} and {y_path}. Call preprocess() first.")

    def __len__(self) -> int:
        if self.X is None:
            raise RuntimeError("Dataset not initialized with preprocessed tensors.")
        return int(self.X.shape[0])

    def __getitem__(self, index: int):
        if self.X is None or self.y is None:
            raise RuntimeError("Dataset not initialized with preprocessed tensors.")
        return self.X[index], self.y[index]

    def preprocess(self) -> None:
        logger.debug("Starting data preprocessing")
        self.cfg.data_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Data folder created/verified: {self.cfg.data_folder}")

        if self.cfg.gcs_uri:
            logger.debug("Using GCS data source")

        if (self.cfg.gcs_service_account_key 
            and "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ 
            and os.path.exists(self.cfg.gcs_service_account_key)):
            
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.cfg.gcs_service_account_key

        elif not os.path.exists(self.cfg.gcs_service_account_key):
            print(f"Notice: Service account key '{self.cfg.gcs_service_account_key}' not found. "
                "Assuming we are running in Cloud Run/Environment with auto-auth.")

            gcs_uri = self.cfg.gcs_uri
            if self.cfg.gcs_data:
                gcs_uri = f"{gcs_uri.rstrip('/')}/{self.cfg.gcs_data}"
            try:
                csv_path = _download_csv_from_gcs(gcs_uri, self.cfg.data_folder)
                logger.info(f"Downloaded CSV from GCS: {gcs_uri}")
            except Exception as e:
                logger.error(f"Failed to download CSV from GCS: {e}")
                raise
        else:
            logger.debug("Using Kaggle data source")
            csv_path = _download_csv("ankushnarwade/ai-impact-on-student-performance")
            logger.info(f"Downloaded CSV from Kaggle dataset: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.debug(f"CSV loaded into DataFrame with shape {df.shape}")

        if self.cfg.target_col not in df.columns:
            logger.error(f"Target column '{self.cfg.target_col}' not found. Available columns: {list(df.columns)}")
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
        logger.info("Preprocessing complete - data splits and preprocessor saved.")

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


def generate_report(train_set, val_set, test_set, cfg):
    report_data = []
    for name, ds in [("Train", train_set), ("Val", val_set), ("Test", test_set)]:
        X, y = ds.tensors
        report_data.append(
            {
                "Split": name,
                "Samples": len(X),
                "Features": X.shape[1],
                "Target Mean": f"{y.mean().item():.4f}",
                "Target Std": f"{y.std().item():.4f}",
                "Contains NaN": torch.isnan(X).any().item(),
            }
        )

    df_stats = pd.DataFrame(report_data)

    print("## Data Quality Report")
    print(f"\n**Data Folder:** `{cfg.data_folder}`")
    print(df_stats.to_markdown(index=False))

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    for name, ds in [("Train", train_set), ("Val", val_set), ("Test", test_set)]:
        sns.kdeplot(ds.tensors[1].numpy().flatten(), label=name, fill=True)

    plt.title("Target Distribution Across Splits")
    plt.xlabel(cfg.target_col)
    plt.legend()

    plot_path = "reports/figures/dist_plot.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"\n### Distribution Shift Check\n![Distributions](./{plot_path})")


def main() -> None:
    dataset_manager = MyDataset(cfg=data_config)

    dataset_manager.preprocess()
    train_set, val_set, test_set = dataset_manager.load_data()
    # generate_report(train_set, val_set, test_set, data_config)


if __name__ == "__main__":
    typer.run(main)
