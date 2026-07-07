"""
src/monitoring/monitoring_router.py
Endpoints FastAPI — Monitoring & Drift Detection
AI NewsOps Platform · AIA Bloc 4

Ajouter dans src/api/main.py :
    from src.monitoring.monitoring_router import router as monitoring_router
    app.include_router(monitoring_router, prefix="/monitoring", tags=["Monitoring"])

Endpoints :
  GET  /monitoring/drift      → vérification rapide de drift (scipy)
  GET  /monitoring/alerts     → historique des N dernières alertes
  POST /monitoring/check      → lancer tous les checks d'un coup
  GET  /monitoring/report     → lien vers le dernier rapport HTML Evidently
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

router = APIRouter()

MONITORING_DIR = Path("monitoring")
REPORTS_DIR = MONITORING_DIR / "reports"
DRIFT_LOG = MONITORING_DIR / "drift_log.jsonl"
ALERT_LOG = MONITORING_DIR / "alerts.jsonl"
QUICK_DRIFT = MONITORING_DIR / "quick_drift_latest.json"

DATA_DIR = Path("data/processed")


# ─────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────


class DriftCheckResponse(BaseModel):
    timestamp: str
    n_drifted: int
    n_features: int
    drift_share: float
    alert_level: str
    reference_size: int
    production_size: int
    features: dict


class MonitoringCheckRequest(BaseModel):
    batch_size: int = 500
    avg_latency_ms: float = 0.0
    n_requests: int = 0
    n_errors: int = 0
    run_full_report: bool = False


class MonitoringCheckResponse(BaseModel):
    timestamp: str
    n_alerts: int
    alerts: list
    drift_result: Optional[dict] = None


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────


def _load_splits():
    """Charge référence et production pour les checks."""
    import pandas as pd

    if not (DATA_DIR / "train.parquet").exists():
        raise HTTPException(
            status_code=503,
            detail="Données de référence non disponibles. Lancer le preprocessing d'abord.",
        )

    ref = pd.read_parquet(DATA_DIR / "train.parquet").sample(n=2000, random_state=42)
    prod = pd.read_parquet(DATA_DIR / "test.parquet").sample(
        n=500, random_state=int(datetime.now().timestamp()) % 1000
    )

    with open(DATA_DIR / "label_mapping.json") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}

    for df in [ref, prod]:
        df["category_name"] = df["label"].map(id2label)
        df["text_length"] = df["text"].str.len()
        df["word_count"] = df["text"].str.split().str.len()

    return (
        ref[["text", "text_length", "word_count", "category_name", "label"]],
        prod[["text", "text_length", "word_count", "category_name", "label"]],
    )


def _run_drift_background(batch_size: int, run_full: bool):
    """Tâche de fond pour ne pas bloquer l'API."""
    try:
        from src.monitoring.alerts import check_drift_alert, send_alert
        from src.monitoring.drift_detector import (
            generate_drift_report,
            load_production_data,
            load_reference_data,
            quick_drift_check,
        )

        ref = load_reference_data(sample_size=2000)
        prod = load_production_data(batch_size=batch_size)
        result = quick_drift_check(ref, prod)

        # Sauvegarder
        MONITORING_DIR.mkdir(exist_ok=True)
        with open(QUICK_DRIFT, "w") as f:
            json.dump(result, f, indent=2)

        # Vérifier et envoyer alerte si nécessaire
        alert = check_drift_alert(result)
        if alert:
            send_alert(alert)

        if run_full:
            generate_drift_report(ref, prod)

    except Exception as e:
        # Log l'erreur sans crasher l'API
        import logging

        logging.getLogger(__name__).error(f"Erreur drift background : {e}")


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────


@router.get(
    "/drift",
    response_model=DriftCheckResponse,
    summary="Vérification rapide de drift (scipy)",
)
def get_drift():
    """
    Retourne le résultat du dernier check de drift.

    Si aucun check n'a été lancé, effectue une vérification rapide en temps réel.
    Pour un check complet avec rapport HTML Evidently, utiliser POST /monitoring/check.
    """
    # Retourner le cache si disponible et récent (< 1h)
    if QUICK_DRIFT.exists():
        with open(QUICK_DRIFT) as f:
            cached = json.load(f)
        # Vérifier l'âge
        try:
            ts = datetime.fromisoformat(cached["timestamp"])
            age = (datetime.now() - ts).total_seconds()
            if age < 3600:  # < 1h → retourner le cache
                return cached
        except Exception:
            pass

    # Sinon, faire un check rapide maintenant
    try:
        from src.monitoring.drift_detector import quick_drift_check

        ref, prod = _load_splits()
        result = quick_drift_check(ref, prod)

        MONITORING_DIR.mkdir(exist_ok=True)
        with open(QUICK_DRIFT, "w") as f:
            json.dump(result, f, indent=2)

        return result

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Module de monitoring non disponible. Installer evidently et scipy.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/check",
    response_model=MonitoringCheckResponse,
    summary="Lancer tous les checks de monitoring",
)
def run_monitoring_check(
    request: MonitoringCheckRequest,
    background_tasks: BackgroundTasks,
):
    """
    Lance une vérification complète en arrière-plan :
    - Drift des données (texte + distribution des catégories)
    - Latence API
    - Taux d'erreur
    - Performance modèle

    Le rapport Evidently HTML est généré si `run_full_report=true`
    (peut prendre 30-60 secondes supplémentaires).
    """
    try:
        from src.monitoring.alerts import run_all_checks

        # Charger les métriques modèle si disponibles
        test_f1 = None
        metrics_path = Path("models/distilbert/training_metrics.json")
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            test_f1 = m.get("test_f1_macro")

        # Check drift rapide (synchrone)
        drift_result = None
        try:
            from src.monitoring.drift_detector import quick_drift_check

            ref, prod = _load_splits()
            drift_result = quick_drift_check(ref, prod)
            with open(QUICK_DRIFT, "w") as f:
                json.dump(drift_result, f, indent=2)
        except Exception:
            pass

        # Tous les checks d'alerte
        alerts = run_all_checks(
            drift_result=drift_result,
            avg_latency_ms=request.avg_latency_ms,
            n_requests=request.n_requests,
            n_errors=request.n_errors,
            test_f1=test_f1,
        )

        # Rapport Evidently en arrière-plan si demandé
        if request.run_full_report:
            background_tasks.add_task(
                _run_drift_background,
                batch_size=request.batch_size,
                run_full=True,
            )

        return {
            "timestamp": datetime.now().isoformat(),
            "n_alerts": len(alerts),
            "alerts": alerts,
            "drift_result": drift_result,
        }

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Modules de monitoring non disponibles. Vérifier l'installation.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/alerts",
    summary="Historique des alertes",
)
def get_alerts(last_n: int = 20):
    """
    Retourne les N dernières alertes enregistrées.

    Les alertes sont persistées dans `monitoring/alerts.jsonl`
    et incluent : niveau, titre, message, métriques, horodatage.
    """
    try:
        from src.monitoring.alerts import get_alert_history

        alerts = get_alert_history(last_n=last_n)
        return {
            "n_alerts": len(alerts),
            "last_n": last_n,
            "alerts": alerts,
        }
    except ImportError:
        # Fallback sans le module alerts
        if not ALERT_LOG.exists():
            return {"n_alerts": 0, "last_n": last_n, "alerts": []}
        alerts = []
        with open(ALERT_LOG) as f:
            for line in f:
                try:
                    alerts.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return {
            "n_alerts": len(alerts[-last_n:]),
            "last_n": last_n,
            "alerts": alerts[-last_n:],
        }


@router.get(
    "/report",
    summary="Dernier rapport Evidently HTML",
)
def get_latest_report():
    """
    Retourne le chemin et les métadonnées du dernier rapport Evidently.

    Pour générer un nouveau rapport : POST /monitoring/check avec run_full_report=true.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reports = sorted(
        REPORTS_DIR.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True
    )

    if not reports:
        return {
            "available": False,
            "message": "Aucun rapport disponible. Lancer POST /monitoring/check avec run_full_report=true.",
        }

    latest = reports[0]
    return {
        "available": True,
        "report_name": latest.name,
        "report_path": str(latest),
        "generated_at": datetime.fromtimestamp(latest.stat().st_mtime).isoformat(),
        "size_kb": round(latest.stat().st_size / 1024, 1),
        "n_reports": len(reports),
        "all_reports": [r.name for r in reports[:5]],
        "message": "Ouvrir le fichier HTML directement dans un navigateur.",
    }


@router.get(
    "/status",
    summary="Statut global du monitoring",
)
def get_monitoring_status():
    """
    Vue d'ensemble du système de monitoring :
    dernière exécution, statut des alertes, disponibilité des modules.
    """
    # Drift
    drift_status = {"available": False}
    if QUICK_DRIFT.exists():
        with open(QUICK_DRIFT) as f:
            drift_data = json.load(f)
        drift_status = {
            "available": True,
            "last_check": drift_data.get("timestamp"),
            "alert_level": drift_data.get("alert_level"),
            "drift_share": drift_data.get("drift_share"),
        }

    # Alertes
    n_alerts = 0
    n_critical = 0
    if ALERT_LOG.exists():
        with open(ALERT_LOG) as f:
            for line in f:
                try:
                    a = json.loads(line.strip())
                    n_alerts += 1
                    if a.get("level") == "critical":
                        n_critical += 1
                except json.JSONDecodeError:
                    continue

    # Modules disponibles
    modules = {}
    for mod in ["evidently", "scipy"]:
        try:
            __import__(mod)
            modules[mod] = "disponible"
        except ImportError:
            modules[mod] = "non installé"

    return {
        "timestamp": datetime.now().isoformat(),
        "drift": drift_status,
        "alerts": {
            "total": n_alerts,
            "critical": n_critical,
        },
        "reports": {
            "n_reports": (
                len(list(REPORTS_DIR.glob("*.html"))) if REPORTS_DIR.exists() else 0
            ),
        },
        "modules": modules,
        "config": {
            "drift_threshold": 0.15,
            "latency_threshold": "500ms",
            "error_rate_threshold": "5%",
        },
    }
