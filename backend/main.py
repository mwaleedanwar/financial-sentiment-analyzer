"""
Phase 2: FastAPI inference service for the fine-tuned financial sentiment model.

Environment:
  MODEL_PATH  Optional absolute path to the saved Hugging Face folder
              (defaults to ../financial_model next to this package).

Render / production:
  - Set MODEL_PATH to the directory that contains config.json, model.safetensors, tokenizer files.
  - Use a startup command like: uvicorn main:app --host 0.0.0.0 --port $PORT

Local dev (same repo .venv as training):
  pip install -r ../requirements-train.txt -r ../requirements-api.txt
  cd backend && uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("financial_sentiment_api")

# ---------------------------------------------------------------------------
# Paths & app
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = (BACKEND_DIR.parent / "financial_model").resolve()
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_DIR))).resolve()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Optional warmup so the first real user request is not paying cold-load latency."""
    try:
        get_classifier()
    except Exception:  # noqa: BLE001
        logger.exception("Model not ready at startup — will retry on /predict")
    yield


app = FastAPI(
    title="Financial Sentiment API",
    version="1.0.0",
    description="DistilBERT-based 3-class sentiment: Negative, Positive, Neutral.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # `allow_credentials=True` is incompatible with wildcard origins in browsers.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response contracts (explicit for OpenAPI + frontend typing)
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    """Inbound payload for a single sentence classification."""

    text: str = Field(..., min_length=1, description="Financial news sentence or tweet-like text.")


SentimentLabel = Literal["Negative", "Positive", "Neutral"]


class PredictResponse(BaseModel):
    """Argmax label + softmax confidence for the predicted class."""

    label: SentimentLabel
    confidence: float = Field(..., ge=0.0, le=1.0)


def _pipeline_device():
    """CUDA in cloud, MPS on Apple Silicon dev machines, CPU on Render free tier."""
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return -1


@lru_cache(maxsize=1)
def get_classifier():
    """
    Lazy-load the HF pipeline once per worker process.

    `lru_cache` avoids reloading weights on every request (important on free tiers).
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_PATH}. "
            "Train locally (train.py) or set MODEL_PATH to your artifact folder."
        )

    device = _pipeline_device()
    clf = pipeline(
        task="text-classification",
        model=str(MODEL_PATH),
        tokenizer=str(MODEL_PATH),
        device=device,
        top_k=1,
        truncation=True,
        max_length=128,
    )
    logger.info(
        "Model loaded from %s (device=%s)",
        MODEL_PATH,
        device,
    )
    return clf


def _unwrap_pipeline_prediction(outputs) -> dict:
    """
    HF `text-classification` return shape varies by transformers version and `top_k`:
    dict, [dict], or nested [[{...}]] for one sequence — unwrap to a single scores dict.
    """
    cur = outputs
    while isinstance(cur, list):
        if len(cur) != 1:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected classifier output shape: list length {len(cur)}.",
            )
        cur = cur[0]
    if not isinstance(cur, dict):
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected classifier output type: {type(cur).__name__}.",
        )
    return cur


@app.get("/health")
def health() -> dict[str, str]:
    """Lightweight probe for uptime monitors."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    """
    Run a single forward pass and return the best label + calibrated-ish confidence.

    The transformers pipeline returns `label` using the `id2label` mapping saved
    during fine-tuning (Negative / Positive / Neutral).
    """
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="`text` must not be empty.")

    try:
        clf = get_classifier()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    outputs = clf(text)
    result = _unwrap_pipeline_prediction(outputs)

    label = result.get("label")
    score = float(result.get("score", 0.0))
    if label not in ("Negative", "Positive", "Neutral"):
        # Fallback if config used different strings — coerce via id if present
        raise HTTPException(status_code=500, detail=f"Unexpected label from model: {label!r}")

    return PredictResponse(label=label, confidence=score)
