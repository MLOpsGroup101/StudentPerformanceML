# Data

This page documents the dataset, preprocessing pipeline, and data artifacts used in the *stuperml* project.

## Dataset

The project uses the public dataset **"AI Impact on Student Performance"** from Kaggle. It contains approximately 8,000 student records with 26 features describing:

- Demographics (e.g., age, grade level)
- Study habits (e.g., study hours, attendance percentage)
- Academic performance indicators (e.g., prior exam scores)
- AI usage metrics (e.g., AI usage time, dependency score, ethical usage score, percentage of AI-generated content)

The primary target variable is the **final score** of each student.

By default, the dataset is downloaded from Kaggle using the helper utilities defined in `stuperml.utils`. Alternatively, a Google Cloud Storage (GCS) path can be configured in `configs/data_config` to override the data source.

## Data module (`stuperml.data`)

All data preparation logic is encapsulated in the `MyDataset` class defined in the `stuperml.data` module.

### `MyDataset`

`MyDataset` is a `torch.utils.data.Dataset` wrapper that serves two roles:

1. **Managing preprocessed tensors** for the `train`, `val`, and `test` splits.
2. **Running the preprocessing pipeline** via its `preprocess()` method.

Key behaviors:

- The constructor takes a `split` argument (`"train"`, `"val"`, or `"test"`) and a `DataConfig` instance (`configs.data_config`).
- If preprocessed tensors already exist under the configured `data_folder`, they are loaded immediately.
- If preprocessed tensors are missing, it warns that `preprocess()` should be run first.
- `__len__` and `__getitem__` expose samples in a form compatible with PyTorch `DataLoader`.

### Preprocessing pipeline

The `preprocess()` method performs the following steps:

1. **Create/verify data folder** specified by `cfg.data_folder`.
2. **Download raw CSV**:
   - If `cfg.gcs_uri` is set, it uses `_download_csv_from_gcs` to retrieve a CSV from Google Cloud Storage.
   - Otherwise, it calls `_download_csv` with the Kaggle dataset slug `"ankushnarwade/ai-impact-on-student-performance"`.
3. **Load CSV into a DataFrame** and validate that `cfg.target_col` exists.
4. **Drop configured columns** listed in `cfg.dropped_columns`.
5. **Build preprocessing pipeline** via `_build_preprocessor()` from `stuperml.utils`:
   - Numeric features: standardized using `StandardScaler`.
   - Categorical/bool features: one-hot encoded using `OneHotEncoder`.
6. **Fit the preprocessor** and transform the feature matrix `X_df`.
7. **Split into train/validation/test** using `_split_data()` with proportions from `cfg.train_size`, `cfg.val_size`, and `cfg.test_size` and random seed `cfg.seed`.
8. **Convert to tensors** using `_to_tensor` and save as:
   - `X_train.pt`, `X_val.pt`, `X_test.pt`
   - `y_train.pt`, `y_val.pt`, `y_test.pt`
9. **Persist artifacts** for downstream components:
   - `feature_names.json` (list of transformed feature names)
   - `preprocessor.joblib` (fitted scikit-learn pipeline)

### Loading data

The `load_data()` method loads the saved tensors from disk and returns three `TensorDataset` objects:

- `train_set`
- `val_set`
- `test_set`

These datasets are used by the training and evaluation scripts.

## Utility functions (`stuperml.utils`)

Several helper functions in `stuperml.utils` are used by the data module:

- `_download_csv(dataset_slug: str) -> Path`: Downloads a Kaggle dataset via `kagglehub` and returns the path to the first CSV file found.
- `_download_csv_from_gcs(gcs_uri: str, dest_dir: Path) -> Path`: Downloads a CSV from a GCS URI into a local directory. Handles URI parsing, bucket/object resolution, and existence checks.
- `_validate_splits(train_size: float, val_size: float, test_size: float)`: Ensures that the split ratios are non-negative and sum to 1.0.
- `_build_preprocessor() -> ColumnTransformer`: Builds a column transformer that scales numerical columns and one-hot encodes categorical and boolean columns.
- `_split_data(...) -> Tuple[Tuple, Tuple, Tuple]`: Splits features and targets into train/validation/test arrays using `train_test_split` with flexible handling of zero-sized splits.
- `_to_tensor(x, dtype=torch.float32) -> torch.Tensor`: Converts numpy arrays to PyTorch tensors.

These utilities are also used indirectly by evaluation and the API through the artifacts produced by `MyDataset`.

## Data quality report

Running `uv run src/stuperml/data.py` not only preprocesses the data but also generates a simple data quality report and a distribution plot:

- Prints a markdown table summarizing for each split:
  - Number of samples
  - Number of features
  - Mean and standard deviation of the target
  - Whether NaN values are present
- Saves a KDE plot of the target distribution for train/val/test to `reports/figures/dist_plot.png`.

This makes it easy to visually inspect potential distribution shift across splits.

## Summary of artifacts

After preprocessing, you should see at least the following under the configured `data_folder`:

- `X_train.pt`, `X_val.pt`, `X_test.pt`
- `y_train.pt`, `y_val.pt`, `y_test.pt`
- `feature_names.json`
- `preprocessor.joblib`

And under `reports/figures/`:

- `dist_plot.png` (target distribution across splits).
