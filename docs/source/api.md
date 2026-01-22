# API Reference

This page documents the HTTP API and provides auto-generated reference documentation for the main Python modules in the *stuperml* package.

## HTTP API (`stuperml.api`)

The FastAPI application is defined in the `stuperml.api` module. It exposes endpoints for health checking and batch prediction.

### Endpoints

#### `GET /`

- **Description**: Simple health check endpoint.
- **Response**: JSON object with keys:
  - `message`: HTTP status phrase (e.g. `"OK"`).
  - `status-code`: Numeric status code (e.g. `200`).

Example:

```bash
curl http://127.0.0.1:8000/
```

#### `POST /predict`

- **Description**: Run batch inference on a list of input rows.
- **Request body** (`PredictionRequest`):
  - `rows`: list of objects where each object is a mapping from feature name to value (`bool`, `int`, `float`, or `str`). Must contain at least one row.
- **Response body** (`PredictionResponse`):
  - `predictions`: list of floating-point predictions, one per input row.

The request features must be compatible with the fitted preprocessor (i.e. same feature names and reasonable data types). If the data cannot be transformed, the endpoint responds with a `400 BAD REQUEST` containing an error message.

On application startup, the API:

1. Loads the preprocessor from `data/preprocessor.joblib` (path defined in `configs.data_config`).
2. Infers the model input size either from `preprocessor.get_feature_names_out()` or, as a fallback, from `data/feature_names.json`.
3. Instantiates `SimpleMLP` with the inferred input size.
4. Loads weights from `models/model.pth`.

If any of the artifacts are missing, a `FileNotFoundError` or `RuntimeError` is raised during startup.

## Python API (mkdocstrings)

The sections below are auto-generated from the docstrings in the `stuperml` package using `mkdocstrings`. They provide detailed reference information about public classes, functions, and modules.

### Data

::: stuperml.data

### Models

::: stuperml.model

### Training

::: stuperml.train

### Evaluation

::: stuperml.evaluate

### HTTP API

::: stuperml.api
