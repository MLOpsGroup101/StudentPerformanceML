import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data loading and preprocessing."""

    data_folder: Path = Path("data/")
    gcs_uri: str | None = "gs://student_performance_ai_data/"
    gcs_data: str = "ai_impact_student_performance_dataset.csv"
    gcs_service_account_key: str | None = os.getenv("GCS_SA_KEY_PATH", "stuperml-e4e7c60b7b19.json")
    target_col: str = "final_score"
    file_names: tuple[str, ...] = ("X_train.pt", "X_val.pt", "X_test.pt", "y_train.pt", "y_val.pt", "y_test.pt")
    dropped_columns: list[str] = field(
        default_factory=lambda: [
            "student_id",
            "age",
            "gender",
            "uses_ai",
            "ai_tools_used",
            "ai_usage_purpose",
            "ai_dependency_score",
            "ai_prompts_per_week",
            "ai_ethics_score",
            "last_exam_score",
            "assignment_scores_avg",
            "concept_understanding_score",
            "study_consistency_index",
            "improvement_rate",
            "tutoring_hours",
            "class_participation_score",
            "passed",
            "performance_category",
        ]
    )
    train_size: float = 0.8
    test_size: float = 0.1
    val_size: float = 0.1
    seed: int = 42


data_config = DataConfig()
