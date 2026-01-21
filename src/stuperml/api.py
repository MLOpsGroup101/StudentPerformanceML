from __future__ import annotations

from contextlib import asynccontextmanager
from http import HTTPStatus
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from configs import data_config
from stuperml.model import SimpleMLP

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pth"

PREPROCESSOR_PATH = data_config.data_folder / "preprocessor.joblib"
FEATURE_NAMES_PATH = data_config.data_folder / "feature_names.json"


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load and release model artifacts for app lifecycle."""
    global _model, _preprocessor
    _model, _preprocessor = _load_model()
    try:
        yield
    finally:
        _model = None
        _preprocessor = None


app = FastAPI(lifespan=lifespan)
_model: SimpleMLP | None = None
_preprocessor: Any | None = None


class PredictionRequest(BaseModel):
    """Request payload for batch prediction."""

    rows: list[dict[str, bool | int | float | str]] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    """Response payload for batch prediction."""

    predictions: list[float]


def _load_model() -> tuple[SimpleMLP, Any]:
    """Load the preprocessor and model artifacts."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor file not found at {PREPROCESSOR_PATH}")

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    try:
        input_size = len(preprocessor.get_feature_names_out())
    except Exception:
        if FEATURE_NAMES_PATH.exists():
            feature_names = json.loads(FEATURE_NAMES_PATH.read_text())
            input_size = len(feature_names)
        else:
            raise RuntimeError("Unable to infer model input size from preprocessor or feature_names.json")

    model = SimpleMLP(input_size=input_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, preprocessor


@app.get("/")
def root() -> dict[str, Any]:
    """Health check."""
    return {"message": HTTPStatus.OK.phrase, "status-code": HTTPStatus.OK}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Run batch inference on input rows."""
    if _model is None or _preprocessor is None:
        raise HTTPException(status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail="Model not loaded")

    features_df = pd.DataFrame(request.rows)
    try:
        transformed = _preprocessor.transform(features_df)
    except Exception as exc:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(exc)) from exc

    tensor = torch.as_tensor(transformed, dtype=torch.float32)
    with torch.no_grad():
        outputs = _model(tensor).squeeze(1).tolist()

    return PredictionResponse(predictions=[float(value) for value in outputs])
