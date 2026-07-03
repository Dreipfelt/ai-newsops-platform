"""
src/api/main.py
FastAPI — AI NewsOps Platform
REST API pour la classification de news avec DistilBERT
AIA Bloc 4 — MLOps Pipeline

Endpoints :
  GET  /health        → statut de l'API et du modèle
  GET  /metrics       → métriques de performance du modèle
  POST /predict       → classification d'un article
  POST /predict/batch → classification en lot (jusqu'à 32 articles)
  GET  /docs          → Swagger UI auto-généré par FastAPI
  GET  /redoc         → ReDoc documentation
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast



# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models/distilbert/best_model"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data/processed"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "128"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
API_VERSION = "1.0.0"
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

# État global du modèle (chargé au démarrage via lifespan)
model_state = {
    "tokenizer": None,
    "model": None,
    "id2label": None,
    "loaded_at": None,
    "n_requests": 0,
    "n_errors": 0,
    "total_latency_ms": 0.0,
    "metrics": {},
}


# ─────────────────────────────────────────────────────────────
# LIFESPAN — chargement du modèle au démarrage
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle une seule fois au démarrage de l'API."""
    log.info(f"Chargement du modèle depuis {MODEL_DIR}...")

    if not MODEL_DIR.exists():
        log.error(f"Dossier modèle introuvable : {MODEL_DIR}")
        log.error("Lance d'abord : python src/models/train_cpu.py --fast")
        raise RuntimeError(f"Model directory not found: {MODEL_DIR}")

    model_state["tokenizer"] = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model_state["model"] = DistilBertForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        ignore_mismatched_sizes=False,
    )
    model_state["model"].to(DEVICE)
    model_state["model"].eval()
    model_state["loaded_at"] = datetime.utcnow().isoformat()

    # Charger le label mapping
    label_mapping_path = DATA_DIR / "label_mapping.json"
    if label_mapping_path.exists():
        with open(label_mapping_path) as f:
            raw = json.load(f)
        model_state["id2label"] = {int(k): v for k, v in raw.items()}
    else:
        # Fallback sur la config du modèle
        model_state["id2label"] = model_state["model"].config.id2label

    # Charger les métriques d'entraînement si disponibles
    metrics_path = Path("models/distilbert/training_metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            model_state["metrics"] = json.load(f)

    n_labels = len(model_state["id2label"])
    log.info(f"✅ Modèle chargé — {n_labels} classes — device: {DEVICE}")
    log.info(f"   Classes : {list(model_state['id2label'].values())}")

    yield  # L'API tourne ici

    log.info("Arrêt de l'API — libération des ressources")
    model_state["model"] = None
    model_state["tokenizer"] = None


# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI NewsOps — News Classifier API",
    description="""
## API de classification automatique d'articles de presse

Classifie un article (headline + description) dans l'une des **13 super-catégories** :
`politics`, `lifestyle`, `health_wellness`, `entertainment`, `media`,
`family_education`, `business`, `tech_science`, `other`, `international`,
`sports`, `arts_culture`, `crime`

### Modèle
- **Architecture** : DistilBERT fine-tuné (`distilbert-base-uncased`)
- **Dataset** : HuffPost News Archive (208k articles, 2012-2022)
- **Input** : `headline [SEP] short_description`

### Projet
AI NewsOps Platform · AIA Bloc 4 MLOps Certification
    """,
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    from src.monitoring.monitoring_router import router as monitoring_router
    app.include_router(monitoring_router, prefix="/monitoring", tags=["Monitoring"])
except Exception as e:
    import logging
    logging.getLogger(__name__).warning(f"Monitoring router non chargé : {e}")

# Monitoring router
try:
    from src.monitoring.monitoring_router import router as monitoring_router
    app.include_router(monitoring_router, prefix="/monitoring", tags=["Monitoring"])
except Exception as e:
    import logging
    logging.getLogger(__name__).warning(f"Monitoring router non chargé : {e}")


# ─────────────────────────────────────────────────────────────
# SCHEMAS PYDANTIC
# ─────────────────────────────────────────────────────────────
class ArticleInput(BaseModel):
    headline: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Titre de l'article",
        examples=["SpaceX launches new rocket to the Moon"],
    )
    short_description: Optional[str] = Field(
        default="",
        max_length=1000,
        description="Description courte (optionnelle mais améliore la précision)",
        examples=["Elon Musk's company successfully launched a new rocket."],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "headline": "SpaceX launches new rocket to the Moon",
                    "short_description": "Elon Musk's company successfully launched a new rocket targeting lunar orbit.",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    category: str = Field(..., description="Catégorie prédite")
    confidence: float = Field(..., description="Score de confiance (0-1)", ge=0, le=1)
    top3: list = Field(..., description="Top 3 des catégories avec leurs scores")
    input_text: str = Field(..., description="Texte envoyé au modèle")
    latency_ms: float = Field(
        ..., description="Latence de l'inférence en millisecondes"
    )
    model_version: str = Field(..., description="Version du modèle utilisé")


class BatchInput(BaseModel):
    articles: list[ArticleInput] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="Liste d'articles à classifier (max 32)",
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    loaded_at: Optional[str]
    n_requests: int
    n_errors: int
    avg_latency_ms: float
    model_version: str
    api_version: str


# ─────────────────────────────────────────────────────────────
# HELPER : inférence
# ─────────────────────────────────────────────────────────────
def run_inference(headline: str, short_description: str = "") -> dict:
    """Tokenise et classifie un article. Retourne la prédiction avec les scores."""
    tokenizer = model_state["tokenizer"]
    model = model_state["model"]
    id2label = model_state["id2label"]

    if model is None:
        raise RuntimeError("Modèle non chargé")

    # Construction du texte (même format que le preprocessing)
    text = f"{headline} [SEP] {short_description}" if short_description else headline

    t0 = time.perf_counter()

    enc = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits

    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred_id = int(np.argmax(probs))
    confidence = float(probs[pred_id])

    latency_ms = (time.perf_counter() - t0) * 1000

    # Top 3
    top3_ids = np.argsort(probs)[::-1][:3]
    top3 = [
        {"category": id2label[int(i)], "score": round(float(probs[i]), 4)}
        for i in top3_ids
    ]

    return {
        "category": id2label[pred_id],
        "confidence": round(confidence, 4),
        "top3": top3,
        "input_text": text[:200] + "..." if len(text) > 200 else text,
        "latency_ms": round(latency_ms, 2),
        "model_version": MODEL_VERSION,
    }


# ─────────────────────────────────────────────────────────────
# MIDDLEWARE — tracking des requêtes
# ─────────────────────────────────────────────────────────────
@app.middleware("http")
async def track_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if request.url.path in ("/predict", "/predict/batch"):
        model_state["n_requests"] += 1
        model_state["total_latency_ms"] += elapsed_ms
        if response.status_code >= 400:
            model_state["n_errors"] += 1

    return response


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Statut de l'API",
    tags=["Monitoring"],
)
def health():
    """
    Vérifie que l'API et le modèle sont opérationnels.
    Retourne les métriques de disponibilité et de performance.
    """
    n_req = model_state["n_requests"]
    avg_latency = model_state["total_latency_ms"] / n_req if n_req > 0 else 0.0
    return {
        "status": "healthy" if model_state["model"] is not None else "degraded",
        "model_loaded": model_state["model"] is not None,
        "device": str(DEVICE),
        "loaded_at": model_state["loaded_at"],
        "n_requests": n_req,
        "n_errors": model_state["n_errors"],
        "avg_latency_ms": round(avg_latency, 2),
        "model_version": MODEL_VERSION,
        "api_version": API_VERSION,
    }


@app.get(
    "/metrics",
    summary="Métriques du modèle",
    tags=["Monitoring"],
)
def get_metrics():
    """
    Retourne les métriques de performance du modèle entraîné
    (F1, accuracy, delta vs baseline, courbes d'entraînement).
    """
    if not model_state["metrics"]:
        return {
            "message": "Métriques non disponibles — entraîne d'abord le modèle",
            "model_loaded": model_state["model"] is not None,
        }
    return {
        "model_performance": {
            "test_f1_macro": model_state["metrics"].get("test_f1_macro"),
            "test_accuracy": model_state["metrics"].get("test_accuracy"),
            "best_val_f1": model_state["metrics"].get("best_val_f1"),
            "baseline_f1": model_state["metrics"].get("baseline_f1"),
            "delta_f1": model_state["metrics"].get("delta_f1"),
            "epochs_run": model_state["metrics"].get("epochs_run"),
            "num_labels": model_state["metrics"].get("num_labels"),
            "class_names": model_state["metrics"].get("class_names"),
        },
        "api_stats": {
            "n_requests": model_state["n_requests"],
            "n_errors": model_state["n_errors"],
            "avg_latency_ms": round(
                model_state["total_latency_ms"] / max(model_state["n_requests"], 1), 2
            ),
        },
        "training_history": model_state["metrics"].get("history", {}),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classifier un article",
    tags=["Inference"],
)
def predict(article: ArticleInput):
    """
    Classifie un article de presse dans l'une des 13 super-catégories.

    **Input** :
    - `headline` : titre de l'article (obligatoire)
    - `short_description` : description courte (optionnelle, améliore la précision)

    **Output** :
    - `category` : catégorie prédite
    - `confidence` : score de confiance (0-1)
    - `top3` : les 3 meilleures prédictions avec leurs scores
    - `latency_ms` : temps d'inférence en millisecondes
    """
    try:
        result = run_inference(article.headline, article.short_description or "")
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.error(f"Erreur inférence : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")


@app.post(
    "/predict/batch",
    summary="Classifier plusieurs articles",
    tags=["Inference"],
)
def predict_batch(batch: BatchInput):
    """
    Classifie jusqu'à **32 articles** en une seule requête.

    Retourne la liste des prédictions dans le même ordre que l'input.
    """
    results = []
    errors = []

    for i, article in enumerate(batch.articles):
        try:
            result = run_inference(article.headline, article.short_description or "")
            results.append({"index": i, "success": True, **result})
        except Exception as e:
            log.error(f"Erreur article {i} : {e}")
            errors.append({"index": i, "error": str(e)})
            results.append({"index": i, "success": False, "error": str(e)})

    return {
        "n_articles": len(batch.articles),
        "n_success": len([r for r in results if r.get("success")]),
        "n_errors": len(errors),
        "predictions": results,
    }


@app.get("/", tags=["Info"])
def root():
    """Point d'entrée — redirige vers la documentation."""
    return {
        "name": "AI NewsOps — News Classifier API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "description": "Classification automatique d'articles de presse (13 catégories)",
        "model": "distilbert-base-uncased fine-tuned on HuffPost News Archive",
    }
