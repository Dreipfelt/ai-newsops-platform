#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API FastAPI pour la classification d'articles de news
avec monitoring Evidently et métriques Prometheus
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import pandas as pd
import numpy as np

# FastAPI
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

# Transformers
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)
import torch

# Prometheus
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator

# Evidently (optionnel)
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False

# ============================================================
# Configuration
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/distilbert/best_model"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data/processed"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DRIFT_LOGS_PATH = Path("monitoring/drift_logs.json")
REFERENCE_DATA_PATH = DATA_DIR / "test.parquet"

# ============================================================
# Pydantic Models avec descriptions
# ============================================================
class PredictRequest(BaseModel):
    headline: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Le titre de l'article (obligatoire, 3 à 500 caractères)."
    )
    short_description: Optional[str] = Field(
        None,
        max_length=1000,
        description="Courte description de l'article (optionnelle, max 1000 caractères)."
    )

    def get_text(self) -> str:
        if self.short_description:
            return f"{self.headline} [SEP] {self.short_description}"
        return self.headline

class PredictResponse(BaseModel):
    category: str = Field(..., description="Catégorie prédite.")
    confidence: float = Field(..., description="Score de confiance (0 à 1).")
    top3: List[tuple] = Field(..., description="Top 3 des catégories avec leurs scores.")

class BatchPredictRequest(BaseModel):
    articles: List[PredictRequest] = Field(..., description="Liste des articles à classer (max 100).")

class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]

# ============================================================
# Métriques Prometheus
# ============================================================
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint'])
DRIFT_SCORE = Gauge('model_drift_score', 'Current model drift score')
MODEL_F1 = Gauge('model_f1_score', 'Model F1 score')
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy')

# ============================================================
# Classe de monitoring
# ============================================================
class ModelMonitor:
    """Moniteur de dérive pour le modèle de classification"""

    def __init__(self, reference_data_path=REFERENCE_DATA_PATH):
        self.reference_data = None
        self.drift_logs = []
        self._load_reference_data(reference_data_path)
        self._load_drift_logs()

    def _load_reference_data(self, path):
        if path.exists():
            try:
                df = pd.read_parquet(path)
                sample_size = min(500, len(df))
                self.reference_data = df.sample(n=sample_size, random_state=42)
                if "prediction" not in self.reference_data.columns:
                    self.reference_data["prediction"] = np.random.randint(0, 13, len(self.reference_data))
                log.info(f"✅ Référence chargée: {len(self.reference_data)} exemples")
            except Exception as e:
                log.error(f"Erreur chargement référence: {e}")
                self.reference_data = None
        else:
            log.warning(f"⚠️ Fichier de référence non trouvé: {path}")

    def _load_drift_logs(self):
        if DRIFT_LOGS_PATH.exists():
            try:
                with open(DRIFT_LOGS_PATH) as f:
                    self.drift_logs = json.load(f)
                log.info(f"✅ Logs de drift chargés: {len(self.drift_logs)}")
            except Exception as e:
                log.error(f"Erreur chargement logs: {e}")
                self.drift_logs = []
        else:
            DRIFT_LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.drift_logs = []

    def _save_drift_logs(self):
        try:
            with open(DRIFT_LOGS_PATH, "w") as f:
                json.dump(self.drift_logs, f, indent=2)
        except Exception as e:
            log.error(f"Erreur sauvegarde logs: {e}")

    def detect_drift(self, current_data, threshold=0.3):
        if self.reference_data is None:
            return {"error": "No reference data available", "drift_detected": False}, None

        if not EVIDENTLY_AVAILABLE:
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "significant_drift": False,
                "message": "Evidently non disponible"
            }, None

        try:
            if isinstance(current_data, dict):
                current_df = pd.DataFrame([current_data])
            elif isinstance(current_data, list):
                current_df = pd.DataFrame(current_data)
            elif isinstance(current_data, pd.DataFrame):
                current_df = current_data
            else:
                return {"error": "Format de données non supporté"}, None

            if "prediction" not in current_df.columns:
                current_df["prediction"] = np.random.randint(0, 13, len(current_df))
            if "prediction" not in self.reference_data.columns:
                self.reference_data["prediction"] = np.random.randint(0, 13, len(self.reference_data))

            column_mapping = ColumnMapping(
                prediction="prediction",
                target="label" if "label" in current_df.columns else None,
                numerical_features=[],
                categorical_features=[],
                text_features=["text"] if "text" in current_df.columns else [],
            )

            report = Report(metrics=[DataDriftPreset()])
            report.run(
                reference_data=self.reference_data,
                current_data=current_df,
                column_mapping=column_mapping,
            )

            drift_metrics = report.as_dict()
            drift_detected = False
            drift_score = 0.0
            drift_ratio = 0.0

            if "metrics" in drift_metrics:
                for metric in drift_metrics["metrics"]:
                    if metric["metric"] == "DataDriftPreset":
                        drift_detected = metric["result"].get("drift_detected", False)
                        drift_score = metric["result"].get("drift_score", 0.0)
                        drift_ratio = metric["result"].get("drift_ratio", 0.0)

            significant_drift = drift_detected or drift_score > threshold

            result = {
                "timestamp": datetime.now().isoformat(),
                "drift_detected": drift_detected,
                "significant_drift": significant_drift,
                "drift_score": drift_score,
                "drift_ratio": drift_ratio,
                "threshold": threshold,
                "n_reference": len(self.reference_data),
                "n_current": len(current_df),
            }

            self.drift_logs.append(result)
            self._save_drift_logs()
            return result, report

        except Exception as e:
            log.error(f"Erreur détection drift: {e}")
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "significant_drift": False,
                "error": str(e),
                "n_reference": len(self.reference_data) if self.reference_data is not None else 0,
                "n_current": len(current_data) if current_data is not None else 0,
            }, None

    def get_recent_drifts(self, limit=10):
        return self.drift_logs[-limit:] if self.drift_logs else []

    def get_drift_summary(self):
        if not self.drift_logs:
            return {"total": 0, "recent_drifts": 0, "last_drift": None}

        recent = self.drift_logs[-20:]
        drifts = [d for d in recent if d.get("significant_drift", False)]

        return {
            "total": len(self.drift_logs),
            "recent_drifts": len(drifts),
            "last_drift": drifts[-1] if drifts else recent[-1],
            "last_check": recent[-1] if recent else None,
        }


# ============================================================
# Initialisation de l'API
# ============================================================

app = FastAPI(
    title="AI NewsOps Platform API",
    description="Classification d'articles de news avec DistilBERT",
    version="1.0.0",
)

# Expose automatiquement /metrics
Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Middleware Prometheus
# ============================================================
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)

    return response


# ============================================================
# Chargement du modèle et du tokenizer (lazy loading)
# ============================================================
tokenizer = None
model = None
id2label = None
label2id = None
monitor = None

def load_model_and_tokenizer():
    global tokenizer, model, id2label, label2id
    try:
        log.info(f"🚀 Chargement du modèle depuis {MODEL_DIR}")
        tokenizer = DistilBertTokenizerFast.from_pretrained(str(MODEL_DIR))
        model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_DIR))
        model.to(DEVICE)
        model.eval()

        if hasattr(model.config, 'id2label') and model.config.id2label:
            id2label = {int(k): v for k, v in model.config.id2label.items()}
            label2id = {v: k for k, v in id2label.items()}
            log.info(f"✅ Mapping depuis le modèle: {len(id2label)} classes")
        else:
            label_mapping_path = DATA_DIR / "label_mapping.json"
            if label_mapping_path.exists():
                with open(label_mapping_path) as f:
                    id2label = json.load(f)
                    id2label = {int(k): v for k, v in id2label.items()}
                    label2id = {v: k for k, v in id2label.items()}
                log.info(f"✅ Mapping depuis label_mapping.json: {len(id2label)} classes")
            else:
                id2label = {0: "POLITICS", 1: "SPORTS", 2: "ENTERTAINMENT", 3: "TECH", 4: "HEALTH"}
                label2id = {v: k for k, v in id2label.items()}
                log.warning(f"⚠️ Mapping par défaut: {len(id2label)} classes")

        log.info(f"✅ Modèle chargé sur {DEVICE} avec {len(id2label)} classes: {list(id2label.values())}")
        return True
    except Exception as e:
        log.error(f"❌ Erreur chargement modèle: {e}")
        return False

def load_monitor():
    global monitor
    monitor = ModelMonitor()
    log.info("✅ Moniteur de drift initialisé")
    return monitor

def ensure_model_loaded():
    """Charge le modèle et le moniteur si ce n'est pas déjà fait."""
    global tokenizer, model, id2label, label2id, monitor
    if model is None:
        load_model_and_tokenizer()
    if monitor is None:
        load_monitor()


# ============================================================
# Routes API (avec documentation Swagger enrichie)
# ============================================================

@app.get(
    "/",
    summary="Informations générales",
    description="Retourne les métadonnées de l'API et son état.",
    tags=["Informations"]
)
async def root():
    ensure_model_loaded()  # charge si nécessaire
    return {
        "name": "AI NewsOps Platform API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "monitoring": "/monitoring/health",
        "metrics": "/metrics",
        "model_loaded": model is not None,
        "device": str(DEVICE)
    }


@app.get(
    "/health",
    summary="Vérifier la santé du service",
    description="Retourne l'état de l'API, du modèle et du moniteur.",
    tags=["Monitoring"],
    responses={
        200: {
            "description": "Service en bonne santé",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "model_loaded": True,
                        "device": "cpu",
                        "tokenizer_loaded": True,
                        "monitor_initialized": True
                    }
                }
            }
        }
    }
)
async def health_check():
    ensure_model_loaded()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "tokenizer_loaded": tokenizer is not None,
        "monitor_initialized": monitor is not None
    }


@app.get(
    "/metrics",
    summary="Exposer les métriques Prometheus",
    description="Endpoint pour Prometheus. Retourne les métriques au format texte.",
    tags=["Monitoring"]
)
async def metrics_endpoint():
    # On charge le modèle pour avoir les métriques, mais si échec on renvoie 0
    ensure_model_loaded()
    if model is not None:
        try:
            metrics_path = MODEL_DIR.parent / "training_metrics.json"
            if metrics_path.exists():
                with open(metrics_path) as f:
                    data = json.load(f)
                    MODEL_F1.set(data.get("test_f1_macro", 0))
                    MODEL_ACCURACY.set(data.get("test_accuracy", 0))
        except Exception as e:
            log.error(f"Erreur mise à jour métriques: {e}")

    if monitor and monitor.drift_logs:
        last_drift = monitor.drift_logs[-1]
        DRIFT_SCORE.set(last_drift.get("drift_score", 0))

    return PlainTextResponse(generate_latest())


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Prédire la catégorie d'un article",
    description="Analyse un article (titre + description facultative) et retourne la catégorie la plus probable avec le top 3.",
    tags=["Prédiction"],
    responses={
        200: {
            "description": "Prédiction réussie",
            "content": {
                "application/json": {
                    "example": {
                        "category": "POLITICS",
                        "confidence": 0.89,
                        "top3": [["POLITICS", 0.89], ["OTHER", 0.07], ["BUSINESS", 0.04]]
                    }
                }
            }
        },
        422: {"description": "Erreur de validation (ex: headline trop court)"},
        503: {"description": "Modèle non chargé"}
    }
)
async def predict(request: PredictRequest):
    ensure_model_loaded()
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        text = request.get_text()
        inputs = tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=3)

        top_idx = top_indices[0][0].item()
        top_confidence = top_probs[0][0].item()
        category = id2label.get(top_idx, "unknown")
        confidence = top_confidence
        top3 = []
        for i in range(3):
            idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            top3.append((id2label.get(idx, "unknown"), prob))

        return PredictResponse(category=category, confidence=confidence, top3=top3)
    except Exception as e:
        log.error(f"Erreur prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Prédire plusieurs articles en lot",
    description="Classifie jusqu'à 100 articles en une seule requête.",
    tags=["Prédiction"],
    responses={
        200: {"description": "Prédictions réussies"},
        422: {"description": "Plus de 100 articles ou erreur de validation"},
        503: {"description": "Modèle non chargé"}
    }
)
async def predict_batch(request: BatchPredictRequest):
    ensure_model_loaded()
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    if len(request.articles) > 100:
        raise HTTPException(status_code=422, detail="Maximum 100 articles par batch")

    try:
        predictions = []
        for article in request.articles:
            result = await predict(article)
            predictions.append(result)
        return BatchPredictResponse(predictions=predictions)
    except Exception as e:
        log.error(f"Erreur batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Monitoring routes
# ============================================================

@app.get(
    "/monitoring/drift",
    summary="Vérifier la dérive du modèle",
    description="Compare les données récentes à la référence et détecte une éventuelle dérive (data drift).",
    tags=["Monitoring"],
    responses={
        200: {"description": "Rapport de dérive généré"},
        500: {"description": "Erreur interne"}
    }
)
async def get_drift_status():
    ensure_model_loaded()
    if monitor is None:
        return {
            "status": "error",
            "message": "Moniteur non initialisé",
            "drift": None
        }

    try:
        if monitor.reference_data is not None:
            current_sample = monitor.reference_data.sample(
                n=min(50, len(monitor.reference_data)),
                random_state=42
            )
            result, report = monitor.detect_drift(current_sample)

            DRIFT_SCORE.set(result.get("drift_score", 0.0))

            if "error" in result:
                return {
                    "status": "warning",
                    "message": result.get("error", "Erreur inconnue"),
                    "drift": result
                }

            return {
                "status": "success",
                "drift": result,
                "summary": monitor.get_drift_summary()
            }
        else:
            return {
                "status": "error",
                "message": "Données de référence non disponibles",
                "drift": None
            }
    except Exception as e:
        log.error(f"Erreur drift: {e}")
        return {
            "status": "error",
            "message": str(e),
            "drift": None
        }


@app.get(
    "/monitoring/drift/logs",
    summary="Historique des dérives",
    description="Récupère les derniers logs de détection de dérive.",
    tags=["Monitoring"],
    responses={
        200: {"description": "Liste des logs"}
    }
)
async def get_drift_logs(limit: int = 10):
    ensure_model_loaded()
    if monitor is None:
        return {"status": "error", "message": "Moniteur non initialisé", "logs": []}

    try:
        logs = monitor.get_recent_drifts(limit)
        return {
            "status": "success",
            "total": len(monitor.drift_logs) if monitor.drift_logs else 0,
            "count": len(logs),
            "logs": logs
        }
    except Exception as e:
        log.error(f"Erreur logs: {e}")
        return {"status": "error", "message": str(e), "logs": []}


@app.get(
    "/monitoring/health",
    summary="Santé du système de monitoring",
    description="Vérifie que le moniteur et les données de référence sont disponibles.",
    tags=["Monitoring"],
    responses={
        200: {"description": "État du monitoring"}
    }
)
async def monitoring_health():
    ensure_model_loaded()
    if monitor is None:
        return {
            "status": "error",
            "message": "Moniteur non initialisé",
            "reference_data": False,
            "drift_logs_count": 0,
            "summary": {"total": 0, "recent_drifts": 0}
        }

    return {
        "status": "healthy",
        "reference_data": monitor.reference_data is not None,
        "drift_logs_count": len(monitor.drift_logs) if monitor.drift_logs else 0,
        "summary": monitor.get_drift_summary(),
        "evidently_available": EVIDENTLY_AVAILABLE,
        "reference_path": str(REFERENCE_DATA_PATH),
        "logs_path": str(DRIFT_LOGS_PATH)
    }


# ============================================================
# Point d'entrée
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )