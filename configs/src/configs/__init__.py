# from pathlib import Path

# data_cfg = dict(
#     data_folder=Path("data/"),
#     dropped_columns=[],
#     target_col="final_score",
#     train_size=0.8,
#     test_size=0.1,
#     val_size=0.1,
# )


from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data loading and preprocessing."""

    data_folder: Path = Path("data/")
    gcs_uri: str | None = "gs://student_performance_ai_data/"
    gcs_data: str = "ai_impact_student_performance_dataset.csv"
    gcs_service_account_key: str | None = "stuperml-e4e7c60b7b19.json"
    target_col: str = "final_score"
    dropped_columns: list[str] = field(default_factory=list)
    train_size: float = 0.8
    test_size: float = 0.1
    val_size: float = 0.1
    seed: int = 42


data_config = DataConfig()
