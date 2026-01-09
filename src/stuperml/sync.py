from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import typer

from configs import data_cfg
from stuperml.data import MyDataset

_REQUIRED_FILES: Sequence[str] = (
    "X_train.pt",
    "X_val.pt",
    "X_test.pt",
    "y_train.pt",
    "y_val.pt",
    "y_test.pt",
    "feature_names.json",
    "preprocessor.joblib",
)


def _required_paths(data_folder: Path) -> Sequence[Path]:
    """Build required data file paths.

    Args:
        data_folder: Root folder for processed data artifacts.

    Returns:
        Sequence of required file paths.
    """
    return tuple(data_folder / name for name in _REQUIRED_FILES)


def _data_ready(data_folder: Path) -> bool:
    """Check whether all processed artifacts exist.

    Args:
        data_folder: Root folder for processed data artifacts.

    Returns:
        True when all required data files exist.
    """
    return all(path.exists() for path in _required_paths(data_folder))


def ensure_data(cfg: Mapping[str, Any] = data_cfg) -> bool:
    """Ensure processed data exists, downloading if needed.

    Args:
        cfg: Data configuration for preprocessing and storage.

    Returns:
        True if data was created, False if it already existed.
    """
    data_folder = Path(cfg.get("data_folder"))
    if _data_ready(data_folder):
        return False
    MyDataset(cfg=cfg).preprocess()
    return True


def main() -> None:
    """Entry point for syncing local data artifacts."""
    created = ensure_data()
    if created:
        typer.echo("Downloaded and preprocessed data artifacts.")
        return
    typer.echo("Data artifacts already present; skipping download.")


if __name__ == "__main__":
    typer.run(main)
