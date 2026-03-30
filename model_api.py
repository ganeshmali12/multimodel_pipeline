import json
import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

APP_TITLE = "Mental Health Model API"
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"
METADATA_PATH = BASE_DIR / "best_model_metadata.json"
LABEL_ENCODER_PATH = BASE_DIR / "best_model_label_encoder.pkl"
API_KEY_NAME = "X-API-Key"
API_KEY_VALUE = os.getenv("MODEL_API_KEY")


def _load_feature_example() -> Dict[str, float]:
    if not METADATA_PATH.exists():
        return {
            "total_word_count": 120.0,
            "unique_word_count": 95.0,
            "sentiment_score": -0.35,
        }

    try:
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        feature_columns = metadata.get("feature_columns", [])
        if not feature_columns:
            raise ValueError("No feature columns found")
        return {col: 0.0 for col in feature_columns}
    except Exception:
        return {
            "total_word_count": 120.0,
            "unique_word_count": 95.0,
            "sentiment_score": -0.35,
        }


FEATURE_EXAMPLE = _load_feature_example()
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    if not API_KEY_VALUE:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server API key is not configured. Set MODEL_API_KEY environment variable.",
        )

    if not api_key or not hmac.compare_digest(api_key, API_KEY_VALUE):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )

    return api_key


class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Feature dictionary keyed by exact training feature names.",
        examples=[FEATURE_EXAMPLE],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": FEATURE_EXAMPLE
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    samples: List[Dict[str, float]] = Field(
        ...,
        description="List of feature dictionaries for batch prediction.",
        examples=[[FEATURE_EXAMPLE, FEATURE_EXAMPLE]],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "samples": [FEATURE_EXAMPLE, FEATURE_EXAMPLE]
            }
        }
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime.load()
    yield


app = FastAPI(title=APP_TITLE, version="1.0.0", lifespan=lifespan)


class ModelRuntime:
    def __init__(self) -> None:
        self.model: Any = None
        self.metadata: Dict[str, Any] = {}
        self.feature_columns: List[str] = []
        self.class_labels: List[str] = []
        self.label_encoder: Optional[Any] = None

    def load(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")

        self.model = joblib.load(MODEL_PATH)
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.feature_columns = list(self.metadata.get("feature_columns", []))
        if not self.feature_columns:
            raise ValueError("No feature_columns found in metadata.")

        self.class_labels = list(self.metadata.get("class_labels", []))

        label_encoder_file = self.metadata.get("label_encoder_file")
        if label_encoder_file and Path(label_encoder_file).exists():
            self.label_encoder = joblib.load(label_encoder_file)
        elif LABEL_ENCODER_PATH.exists():
            self.label_encoder = joblib.load(LABEL_ENCODER_PATH)


runtime = ModelRuntime()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": runtime.model is not None,
        "model_name": runtime.metadata.get("best_model_name"),
        "n_features": len(runtime.feature_columns),
        "n_classes": runtime.metadata.get("n_classes"),
    }


@app.get("/metadata", dependencies=[Depends(verify_api_key)])
def metadata() -> Dict[str, Any]:
    return {
        "dataset_used": runtime.metadata.get("dataset_used"),
        "best_model_name": runtime.metadata.get("best_model_name"),
        "best_accuracy": runtime.metadata.get("best_accuracy"),
        "best_f1_macro": runtime.metadata.get("best_f1_macro"),
        "n_features": len(runtime.feature_columns),
        "n_classes": runtime.metadata.get("n_classes"),
        "class_labels": runtime.class_labels,
        "feature_columns": runtime.feature_columns,
    }


@app.get("/sample-payload", dependencies=[Depends(verify_api_key)])
def sample_payload() -> Dict[str, Any]:
    return {
        "features": {col: 0.0 for col in runtime.feature_columns}
    }


def _build_row(features: Dict[str, float]) -> pd.DataFrame:
    expected = set(runtime.feature_columns)
    given = set(features.keys())

    missing = sorted(list(expected - given))
    extra = sorted(list(given - expected))

    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Feature schema mismatch.",
                "missing_features": missing,
                "unexpected_features": extra,
            },
        )

    ordered = {col: float(features[col]) for col in runtime.feature_columns}
    return pd.DataFrame([ordered], columns=runtime.feature_columns)


def _decode_prediction(raw_pred: Any) -> str:
    pred_value = raw_pred.item() if hasattr(raw_pred, "item") else raw_pred

    if runtime.label_encoder is not None:
        return str(runtime.label_encoder.inverse_transform([int(pred_value)])[0])

    return str(pred_value)


@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(payload: PredictionRequest) -> Dict[str, Any]:
    row = _build_row(payload.features)
    raw_pred = runtime.model.predict(row)[0]
    predicted_label = _decode_prediction(raw_pred)

    response: Dict[str, Any] = {
        "predicted_label": predicted_label,
        "raw_prediction": str(raw_pred),
    }

    if hasattr(runtime.model, "predict_proba"):
        proba = runtime.model.predict_proba(row)[0]
        if runtime.class_labels and len(runtime.class_labels) == len(proba):
            response["probabilities"] = {
                runtime.class_labels[i]: float(proba[i]) for i in range(len(proba))
            }
        else:
            response["probabilities"] = {
                str(i): float(proba[i]) for i in range(len(proba))
            }

    return response


@app.post("/predict-batch", dependencies=[Depends(verify_api_key)])
def predict_batch(payload: BatchPredictionRequest) -> Dict[str, Any]:
    if not payload.samples:
        raise HTTPException(status_code=422, detail="samples cannot be empty")

    frames = [_build_row(sample) for sample in payload.samples]
    batch_df = pd.concat(frames, ignore_index=True)

    raw_preds = runtime.model.predict(batch_df)
    labels = [_decode_prediction(pred) for pred in raw_preds]

    return {
        "count": len(labels),
        "predicted_labels": labels,
    }
