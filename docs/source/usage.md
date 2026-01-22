# Usage

This page describes how to set up the environment, prepare the data, train the model, and serve predictions for the *stuperml* project.

## Prerequisites

- macOS (project is developed and tested primarily on macOS)
- Python environment managed via `uv` (installed on your system)
- Internet access to download the dataset (Kaggle or GCS, depending on configuration)

## Environment setup

From the project root:

1. Install dependencies and fetch the dataset using the project tasks:

```bash
uv run invoke sync
```

2. Optionally run formatting and linting:

```bash
uv run ruff format .
uv run ruff check . --fix
```

3. (Optional) Run tests to verify everything is working:

```bash
uv run pytest tests/
```

## Data preparation

Data handling is implemented in the `MyDataset` class defined in the `stuperml.data` module.

To run the full preprocessing pipeline and generate the train/validation/test splits, preprocessor, and feature name artifacts, execute:

```bash
uv run src/stuperml/data.py
```

This will:

- Download the raw CSV either from Kaggle (default) or from GCS, depending on the configuration in `configs/data_config`.
- Build a preprocessing pipeline (scaling numeric features and one-hot encoding categorical features).
- Split the dataset into train/validation/test according to the configured ratios.
- Save the following artifacts under the configured `data_folder` (by default `data/`):
  - `X_train.pt`, `X_val.pt`, `X_test.pt`
  - `y_train.pt`, `y_val.pt`, `y_test.pt`
  - `preprocessor.joblib`
  - `feature_names.json`

These artifacts are consumed later by training, evaluation, and the API.

## Training the model

Model training logic lives in `stuperml.train`. To train the neural network from the preprocessed data:

```bash
uv run src/stuperml/train.py
```

The `train` command supports several configurable parameters:

- `lr`: learning rate (default: `1e-3`)
- `batch_size`: batch size (default: `32`)
- `epochs`: number of training epochs (default: `30`)
- `verbose`: whether to print per-iteration logs

Example with custom hyperparameters:

```bash
uv run src/stuperml/train.py --lr 0.0005 --batch-size 64 --epochs 50 --verbose
```

During training the script will:

- Load train and validation tensors from the data folder.
- Initialize a `SimpleMLP` model on CPU, CUDA, or Apple MPS, depending on availability.
- Train using Mean Squared Error (MSE) loss and Adam optimizer.
- Track training and validation loss over epochs.
- Save the trained model weights to `models/model.pth`.
- Produce a training curve figure at `src/stuperml/figures/training_validation_epoch_error.png`.

## Evaluating the model

Evaluation logic is implemented in `stuperml.evaluate`. To evaluate a saved checkpoint on the test split:

```bash
uv run src/stuperml/evaluate.py --model-checkpoint models/model.pth
```

This will:

- Load the test split tensors from the data folder.
- Instantiate a new `SimpleMLP` model with the correct input size.
- Load the checkpoint weights.
- Compute the Mean Absolute Error (MAE) over the test set.
- Save a residual distribution histogram to `src/stuperml/figures/residual_distribution.png`.

If the MAE is high (greater than 10.0), a warning is logged to highlight potential performance issues.

## Serving predictions via API

The project exposes a FastAPI-based HTTP API defined in `stuperml.api`. To start the API locally:

```bash
uv run uvicorn src.stuperml.api:app --reload
```

By default the app runs on `http://127.0.0.1:8000`.

On startup, the service will:

- Load the trained model from `models/model.pth`.
- Load the preprocessor from `data/preprocessor.joblib`.
- Infer the model input size from the preprocessor or from `data/feature_names.json`.

### Health check

You can verify the service is running by calling the root endpoint:

```bash
curl http://127.0.0.1:8000/
```

Expected response:

```json
{"message": "OK", "status-code": 200}
```

### Batch prediction endpoint

To perform batch predictions, use the `/predict` endpoint with a JSON payload containing a `rows` list. Each row is a dictionary mapping feature names to values, matching the schema used during preprocessing.

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "rows": [
      {
        "Age": 18,
        "Study_Hours": 3.5,
        "Attendance": 92.0,
        "AI_Usage_Time": 2.0,
        "AI_Dependency_Score": 4,
        "AI_Ethical_Score": 8,
        "AI_Generated_Content_Percentage": 15.0
      }
    ]
  }'
```

The response contains a list of predicted final scores:

```json
{
  "predictions": [72.3]
}
```

If the payload cannot be transformed by the preprocessor (for example due to missing or unexpected features), the service returns a `400 BAD REQUEST` error with a descriptive message.

## Command summary

- Sync dependencies and fetch dataset: `uv run invoke sync`
- Run preprocessing and data report: `uv run src/stuperml/data.py`
- Train model: `uv run src/stuperml/train.py [--lr ... --batch-size ... --epochs ... --verbose]`
- Evaluate model: `uv run src/stuperml/evaluate.py --model-checkpoint models/model.pth`
- Start API: `uv run uvicorn src.stuperml.api:app --reload`
