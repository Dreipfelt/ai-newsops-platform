#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI NewsOps Platform API.

FastAPI service for:
- news classification with DistilBERT;
- Prometheus metrics;
- monitoring health checks;
- Evidently drift status exposure.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

try:
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/distilbert/best_model"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data/processed"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MONITORING_DIR = Path("monitoring")
DRIFT_LOGS_PATH = MONITORING_DIR / "drift_logs.json"
DRIFT_STATUS_PATH = MONITORING_DIR / "quick_drift_latest.json"
REFERENCE_DATA_PATH = DATA_DIR / "test.parquet"


# ============================================================
# Pydantic Schemas
# ============================================================


class PredictRequest(BaseModel):
    headline: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Article headline.",
    )
    short_description: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional short article description.",
    )

    def get_text(self) -> str:
        if self.short_description:
            return f"{self.headline} [SEP] {self.short_description}"
        return self.headline


class PredictResponse(BaseModel):
    category: str
    confidence: float
    top3: list[tuple[str, float]]
    model_version: str


class BatchPredictRequest(BaseModel):
    articles: list[PredictRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of articles to classify. Maximum: 100.",
    )


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


# ============================================================
# Prometheus Metrics
# ============================================================

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
)

DRIFT_SCORE = Gauge(
    "model_drift_score",
    "Current model drift score or drift share",
)

MODEL_F1 = Gauge(
    "model_f1_score",
    "Model F1 score",
)

MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Model accuracy",
)


# ============================================================
# Model Monitor
# ============================================================


class ModelMonitor:
    """Lightweight Evidently-based drift monitor."""

    def __init__(self, reference_data_path: Path = REFERENCE_DATA_PATH) -> None:
        self.reference_data: Optional[pd.DataFrame] = None
        self.drift_logs: list[dict] = []
        self._load_reference_data(reference_data_path)
        self._load_drift_logs()

    def _load_reference_data(self, path: Path) -> None:
        if not path.exists():
            log.warning("Reference data not found: %s", path)
            self.reference_data = None
            return

        try:
            df = pd.read_parquet(path)
            sample_size = min(500, len(df))
            self.reference_data = df.sample(n=sample_size, random_state=42).copy()

            if "prediction" not in self.reference_data.columns:
                self.reference_data["prediction"] = np.random.randint(
                    0,
                    13,
                    len(self.reference_data),
                )

            log.info("Reference data loaded: %s rows", len(self.reference_data))
        except Exception as exc:
            log.exception("Failed to load reference data: %s", exc)
            self.reference_data = None

    def _load_drift_logs(self) -> None:
        if not DRIFT_LOGS_PATH.exists():
            DRIFT_LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.drift_logs = []
            return

        try:
            self.drift_logs = json.loads(DRIFT_LOGS_PATH.read_text())
            log.info("Drift logs loaded: %s entries", len(self.drift_logs))
        except Exception as exc:
            log.warning("Failed to load drift logs: %s", exc)
            self.drift_logs = []

    def _save_drift_logs(self) -> None:
        try:
            DRIFT_LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            DRIFT_LOGS_PATH.write_text(json.dumps(self.drift_logs, indent=2))
        except Exception as exc:
            log.error("Failed to save drift logs: %s", exc)

    def detect_drift(
        self,
        current_data: dict | list[dict] | pd.DataFrame,
        threshold: float = 0.3,
    ) -> tuple[dict, Optional[Report]]:
        if self.reference_data is None:
            return {
                "status": "error",
                "message": "No reference data available.",
                "drift_detected": False,
                "significant_drift": False,
            }, None

        if not EVIDENTLY_AVAILABLE:
            return {
                "status": "warning",
                "message": "Evidently is not installed.",
                "drift_detected": False,
                "significant_drift": False,
                "drift_score": 0.0,
            }, None

        try:
            if isinstance(current_data, dict):
                current_df = pd.DataFrame([current_data])
            elif isinstance(current_data, list):
                current_df = pd.DataFrame(current_data)
            elif isinstance(current_data, pd.DataFrame):
                current_df = current_data.copy()
            else:
                return {
                    "status": "error",
                    "message": "Unsupported current_data format.",
                    "drift_detected": False,
                    "significant_drift": False,
                }, None

            if "prediction" not in current_df.columns:
                current_df["prediction"] = np.random.randint(0, 13, len(current_df))

            reference_df = self.reference_data.copy()
            if "prediction" not in reference_df.columns:
                reference_df["prediction"] = np.random.randint(0, 13, len(reference_df))

            column_mapping = ColumnMapping(
                prediction="prediction",
                target="label" if "label" in current_df.columns else None,
                numerical_features=[
                    col
                    for col in ["text_length", "word_count", "year"]
                    if col in current_df.columns and col in reference_df.columns
                ],
                categorical_features=[
                    col
                    for col in ["category", "has_desc"]
                    if col in current_df.columns and col in reference_df.columns
                ],
                text_features=[
                    col
                    for col in ["text_clean", "text"]
                    if col in current_df.columns and col in reference_df.columns
                ],
            )

            report = Report(metrics=[DataDriftPreset()])
            report.run(
                reference_data=reference_df,
                current_data=current_df,
                column_mapping=column_mapping,
            )

            report_dict = report.as_dict()
            drift_result = report_dict["metrics"][0]["result"]

            drift_detected = bool(
                drift_result.get("dataset_drift")
                or drift_result.get("drift_detected")
                or False
            )

            drift_score = float(
                drift_result.get("share_of_drifted_columns")
                or drift_result.get("drift_share")
                or drift_result.get("drift_score")
                or 0.0
            )

            significant_drift = drift_detected or drift_score > threshold

            result = {
                "timestamp": datetime.now().isoformat(),
                "status": "drift" if significant_drift else "ok",
                "drift_detected": drift_detected,
                "significant_drift": significant_drift,
                "drift_score": drift_score,
                "threshold": threshold,
                "n_reference": len(reference_df),
                "n_current": len(current_df),
            }

            self.drift_logs.append(result)
            self._save_drift_logs()

            return result, report

        except Exception as exc:
            log.exception("Drift detection failed: %s", exc)
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "message": str(exc),
                "drift_detected": False,
                "significant_drift": False,
                "drift_score": 0.0,
            }, None

    def get_recent_drifts(self, limit: int = 10) -> list[dict]:
        return self.drift_logs[-limit:] if self.drift_logs else []

    def get_drift_summary(self) -> dict:
        if not self.drift_logs:
            return {
                "total": 0,
                "recent_drifts": 0,
                "last_check": None,
                "last_drift": None,
            }

        recent = self.drift_logs[-20:]
        drifts = [entry for entry in recent if entry.get("significant_drift", False)]

        return {
            "total": len(self.drift_logs),
            "recent_drifts": len(drifts),
            "last_check": recent[-1],
            "last_drift": drifts[-1] if drifts else None,
        }


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="AI NewsOps Platform API",
    description="News article classification API with DistilBERT, Prometheus and Evidently monitoring.",
    version=MODEL_VERSION,
)

Instrumentator().instrument(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=str(response.status_code),
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(duration)

    return response


# ============================================================
# Lazy Loaded Resources
# ============================================================

tokenizer: Optional[DistilBertTokenizerFast] = None
model: Optional[DistilBertForSequenceClassification] = None
id2label: Optional[dict[int, str]] = None
label2id: Optional[dict[str, int]] = None
monitor: Optional[ModelMonitor] = None


def load_model_and_tokenizer() -> bool:
    global tokenizer, model, id2label, label2id

    try:
        log.info("Loading model from %s", MODEL_DIR)

        tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_DIR))
        model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_DIR))
        model.to(DEVICE)
        model.eval()

        if hasattr(model.config, "id2label") and model.config.id2label:
            id2label = {int(key): value for key, value in model.config.id2label.items()}
        else:
            label_mapping_path = DATA_DIR / "label_mapping.json"

            if label_mapping_path.exists():
                raw_mapping = json.loads(label_mapping_path.read_text())
                id2label = {int(key): value for key, value in raw_mapping.items()}
            else:
                id2label = {
                    0: "politics",
                    1: "sports",
                    2: "entertainment",
                    3: "tech_science",
                    4: "business",
                }
                log.warning("Using fallback label mapping.")

        label2id = {value: key for key, value in id2label.items()}

        log.info(
            "Model loaded on %s with %s classes.",
            DEVICE,
            len(id2label),
        )
        return True

    except Exception as exc:
        log.exception("Failed to load model: %s", exc)
        tokenizer = None
        model = None
        id2label = None
        label2id = None
        return False


def load_monitor() -> ModelMonitor:
    global monitor
    monitor = ModelMonitor()
    log.info("Drift monitor initialized.")
    return monitor


def ensure_model_loaded() -> None:
    if model is None or tokenizer is None or id2label is None:
        load_model_and_tokenizer()


def ensure_monitor_loaded() -> None:
    if monitor is None:
        load_monitor()


def update_model_metrics() -> None:
    metrics_path = MODEL_DIR.parent / "training_metrics.json"

    if not metrics_path.exists():
        return

    try:
        metrics = json.loads(metrics_path.read_text())
        MODEL_F1.set(float(metrics.get("test_f1_macro", 0.0)))
        MODEL_ACCURACY.set(float(metrics.get("test_accuracy", 0.0)))
    except Exception as exc:
        log.warning("Failed to update model metrics: %s", exc)


# ============================================================
# Routes
# ============================================================


@app.get("/", tags=["Information"])
async def root():
    ensure_model_loaded()

    return {
        "name": "AI NewsOps Platform API",
        "version": MODEL_VERSION,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "metrics": "/metrics",
        "monitoring": {
            "health": "/monitoring/health",
            "drift": "/monitoring/drift",
            "drift_run": "/monitoring/drift/run",
            "logs": "/monitoring/drift/logs",
        },
        "model_loaded": model is not None,
        "device": str(DEVICE),
    }


@app.get("/health", tags=["Monitoring"])
async def health_check():
    ensure_model_loaded()
    ensure_monitor_loaded()

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "tokenizer_loaded": tokenizer is not None,
        "monitor_initialized": monitor is not None,
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics_endpoint():
    ensure_model_loaded()
    ensure_monitor_loaded()

    update_model_metrics()

    if DRIFT_STATUS_PATH.exists():
        try:
            drift_status = json.loads(DRIFT_STATUS_PATH.read_text())
            DRIFT_SCORE.set(float(drift_status.get("drift_share", 0.0)))
        except Exception as exc:
            log.warning("Failed to update drift metric: %s", exc)

    return PlainTextResponse(generate_latest())


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    ensure_model_loaded()

    if model is None or tokenizer is None or id2label is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        text = request.get_text()

        inputs = tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            top_probs, top_indices = torch.topk(probabilities, k=3)

        top3 = [
            (
                id2label.get(top_indices[0][idx].item(), "unknown"),
                float(top_probs[0][idx].item()),
            )
            for idx in range(3)
        ]

        category, confidence = top3[0]

        return PredictResponse(
            category=category,
            confidence=confidence,
            top3=top3,
            model_version=MODEL_VERSION,
        )

    except Exception as exc:
        log.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    predictions = []

    for article in request.articles:
        prediction = await predict(article)
        predictions.append(prediction)

    return BatchPredictResponse(predictions=predictions)


# ============================================================
# Monitoring Routes
# ============================================================


@app.get("/monitoring/drift", tags=["Monitoring"])
async def get_latest_drift_status():
    if not DRIFT_STATUS_PATH.exists():
        return {
            "status": "unknown",
            "message": "No drift status available. Run monitoring/drift_detector.py first.",
            "drift_detected": None,
        }

    try:
        drift_status = json.loads(DRIFT_STATUS_PATH.read_text())
        DRIFT_SCORE.set(float(drift_status.get("drift_share", 0.0)))
        return drift_status
    except Exception as exc:
        log.exception("Failed to read drift status: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/monitoring/drift/run", tags=["Monitoring"])
async def run_live_drift_detection():
    ensure_monitor_loaded()

    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor is not initialized.")

    if monitor.reference_data is None:
        return {
            "status": "error",
            "message": "Reference data is not available.",
            "drift": None,
        }

    current_sample = monitor.reference_data.sample(
        n=min(50, len(monitor.reference_data)),
        random_state=42,
    )

    result, _ = monitor.detect_drift(current_sample)
    DRIFT_SCORE.set(float(result.get("drift_score", 0.0)))

    return {
        "status": result.get("status", "ok"),
        "drift": result,
        "summary": monitor.get_drift_summary(),
    }


@app.get("/monitoring/drift/logs", tags=["Monitoring"])
async def get_drift_logs(limit: int = 10):
    ensure_monitor_loaded()

    if monitor is None:
        return {
            "status": "error",
            "message": "Monitor is not initialized.",
            "logs": [],
        }

    logs = monitor.get_recent_drifts(limit)

    return {
        "status": "success",
        "total": len(monitor.drift_logs),
        "count": len(logs),
        "logs": logs,
    }


@app.get("/monitoring/health", tags=["Monitoring"])
async def monitoring_health():
    ensure_monitor_loaded()

    if monitor is None:
        return {
            "status": "error",
            "message": "Monitor is not initialized.",
        }

    return {
        "status": "healthy",
        "reference_data": monitor.reference_data is not None,
        "drift_logs_count": len(monitor.drift_logs),
        "summary": monitor.get_drift_summary(),
        "evidently_available": EVIDENTLY_AVAILABLE,
        "reference_path": str(REFERENCE_DATA_PATH),
        "drift_status_path": str(DRIFT_STATUS_PATH),
        "logs_path": str(DRIFT_LOGS_PATH),
    }


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
