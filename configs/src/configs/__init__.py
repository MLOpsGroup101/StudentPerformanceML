from pathlib import Path

data_cfg = dict(
    data_folder=Path("data/"),
    dropped_columns=[],
    target_col="final_score",
    train_size=0.8,
    test_size=0.1,
    val_size=0.1,
)
