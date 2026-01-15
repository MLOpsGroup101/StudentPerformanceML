from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    data_folder: Path = Path("data/")
    file_names: tuple[str, ...] = ("X_train.pt", "X_val.pt", "X_test.pt", "y_train.pt", "y_val.pt", "y_test.pt")
    target_col: str = "final_score"
    dropped_columns: list[str] = field(default_factory=list)
    train_size: float = 0.8
    test_size: float = 0.1
    val_size: float = 0.1
    seed: int = 42

data_config = DataConfig()


