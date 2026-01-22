from __future__ import annotations

from typing import Any

import numpy as np
import torch
from fastapi.testclient import TestClient

import stuperml.api as api_module


class DummyPreprocessor:
    """Minimal preprocessor stub for API tests."""

    def transform(self, df) -> np.ndarray:
        """Return a fixed-size numeric array for predictions."""
        return np.zeros((len(df), 2), dtype=np.float32)


class DummyModel(torch.nn.Module):
    """Minimal model stub for API tests."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return zeros with the correct batch shape."""
        return torch.zeros((x.shape[0], 1), dtype=torch.float32)


def _patch_load_model(monkeypatch) -> None:
    """Patch the model loader to avoid file I/O."""
    monkeypatch.setattr(api_module, "_load_model", lambda: (DummyModel(), DummyPreprocessor()))


def test_root_healthcheck(monkeypatch) -> None:
    """Ensure the healthcheck endpoint responds successfully."""
    _patch_load_model(monkeypatch)
    with TestClient(api_module.app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "OK"


def test_predict(monkeypatch) -> None:
    """Ensure prediction endpoint returns a list of floats."""
    _patch_load_model(monkeypatch)
    with TestClient(api_module.app) as client:
        payload: dict[str, Any] = {
            "rows": [
                {"feature_a": 1.0, "feature_b": 2.0},
                {"feature_a": 3.0, "feature_b": 4.0},
            ]
        }
        response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json() == {"predictions": [0.0, 0.0]}
