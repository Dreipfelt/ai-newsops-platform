"""
src/monitoring/alerts.py
Système d'alertes — AI NewsOps Platform
AIA Bloc 4 — MLOps Pipeline

Envoie des alertes quand :
  - Drift détecté (score > seuil)
  - Latence API > 500ms
  - Taux d'erreur API > 5%
  - F1 de production < baseline

Canaux supportés :
  - Log fichier (toujours actif)
  - Webhook HTTP (Slack, Teams, Discord, custom)
  - Email (via SMTP, optionnel)
"""

import json
import logging
import os
import smtplib
import urllib.request
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MONITORING_DIR = Path("monitoring")
ALERT_LOG = MONITORING_DIR / "alerts.jsonl"
MONITORING_DIR.mkdir(exist_ok=True)

# Chargés depuis les variables d'environnement (jamais en dur dans le code)
WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")  # Slack/Teams/Discord webhook
ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")  # destinataire email
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")

# Seuils
LATENCY_THRESHOLD_MS = 500.0
ERROR_RATE_THRESHOLD = 0.05
DRIFT_SHARE_THRESHOLD = 0.15
F1_MIN_THRESHOLD = 0.60


# ─────────────────────────────────────────────────────────────
# NIVEAUX D'ALERTE
# ─────────────────────────────────────────────────────────────
class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


LEVEL_EMOJI = {
    AlertLevel.INFO: "ℹ️",
    AlertLevel.WARNING: "⚠️",
    AlertLevel.CRITICAL: "🔴",
}


# ─────────────────────────────────────────────────────────────
# CLASSE ALERT
# ─────────────────────────────────────────────────────────────
class Alert:
    def __init__(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        metrics: Optional[dict] = None,
    ):
        self.level = level
        self.title = title
        self.message = message
        self.metrics = metrics or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "metrics": self.metrics,
        }

    def to_slack_payload(self) -> dict:
        """Formatte l'alerte pour Slack (webhook)."""
        emoji = LEVEL_EMOJI.get(self.level, "📊")
        color = {"info": "#36a64f", "warning": "#ff9500", "critical": "#ff0000"}.get(
            self.level.value, "#888888"
        )
        fields = [
            {"title": k, "value": str(v), "short": True}
            for k, v in self.metrics.items()
        ]
        return {
            "attachments": [
                {
                    "color": color,
                    "title": f"{emoji} {self.title}",
                    "text": self.message,
                    "fields": fields,
                    "footer": "AI NewsOps Platform · MLOps Monitoring",
                    "ts": int(datetime.now().timestamp()),
                }
            ]
        }

    def format_email_body(self) -> str:
        metrics_str = "\n".join(f"  • {k}: {v}" for k, v in self.metrics.items())
        return f"""
AI NewsOps Platform — Alerte Monitoring
{'=' * 50}

Niveau   : {self.level.value.upper()}
Titre    : {self.title}
Horodatage : {self.timestamp}

Message :
{self.message}

Métriques :
{metrics_str or '  (aucune)'}

{'=' * 50}
Ce message est généré automatiquement par le système de monitoring MLOps.
        """.strip()


# ─────────────────────────────────────────────────────────────
# ENVOI D'ALERTES
# ─────────────────────────────────────────────────────────────


def send_alert(alert: Alert) -> dict:
    """
    Envoie une alerte sur tous les canaux configurés.
    Retourne un dict avec le statut de chaque canal.
    """
    results = {}

    # 1. Log fichier (toujours actif)
    results["log"] = _log_alert(alert)

    # 2. Webhook (Slack/Teams/Discord)
    if WEBHOOK_URL:
        results["webhook"] = _send_webhook(alert)
    else:
        results["webhook"] = {
            "status": "skipped",
            "reason": "ALERT_WEBHOOK_URL non configuré",
        }

    # 3. Email
    if ALERT_EMAIL and SMTP_USER:
        results["email"] = _send_email(alert)
    else:
        results["email"] = {"status": "skipped", "reason": "SMTP non configuré"}

    # Log console
    level_log = {
        AlertLevel.INFO: log.info,
        AlertLevel.WARNING: log.warning,
        AlertLevel.CRITICAL: log.error,
    }.get(alert.level, log.info)

    level_log(f"ALERTE [{alert.level.value.upper()}] {alert.title}: {alert.message}")

    return results


def _log_alert(alert: Alert) -> dict:
    """Enregistre l'alerte dans le fichier JSONL."""
    try:
        with open(ALERT_LOG, "a") as f:
            f.write(json.dumps(alert.to_dict()) + "\n")
        return {"status": "ok", "path": str(ALERT_LOG)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def _send_webhook(alert: Alert) -> dict:
    """Envoie l'alerte via webhook HTTP (Slack, Teams, Discord)."""
    try:
        payload = json.dumps(alert.to_slack_payload()).encode("utf-8")
        req = urllib.request.Request(
            WEBHOOK_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return {"status": "ok", "http_code": resp.status}
    except Exception as e:
        log.error(f"Webhook échoué : {e}")
        return {"status": "error", "error": str(e)}


def _send_email(alert: Alert) -> dict:
    """Envoie l'alerte par email via SMTP."""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[NewsOps Alert] {alert.level.value.upper()} — {alert.title}"
        msg["From"] = SMTP_USER
        msg["To"] = ALERT_EMAIL
        msg.attach(MIMEText(alert.format_email_body(), "plain"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, ALERT_EMAIL, msg.as_string())

        return {"status": "ok", "recipient": ALERT_EMAIL}
    except Exception as e:
        log.error(f"Email échoué : {e}")
        return {"status": "error", "error": str(e)}


# ─────────────────────────────────────────────────────────────
# RÈGLES D'ALERTE — appelées par le monitoring
# ─────────────────────────────────────────────────────────────


def check_drift_alert(drift_result: dict) -> Optional[Alert]:
    """Génère une alerte si le drift dépasse le seuil."""
    drift_share = drift_result.get("drift_share", 0.0)
    alert_level = drift_result.get("alert_level", "ok")

    if alert_level == "ok":
        return None

    level = AlertLevel.CRITICAL if alert_level == "critical" else AlertLevel.WARNING
    n_drift = drift_result.get("n_drifted", 0)
    n_feat = drift_result.get("n_features", 0)

    drifted_features = [
        f"{k} (p={v['p_value']:.4f})"
        for k, v in drift_result.get("features", {}).items()
        if v.get("drift_detected")
    ]

    return Alert(
        level=level,
        title="Data Drift Détecté",
        message=(
            f"{n_drift}/{n_feat} features driftées ({drift_share:.1%}). "
            f"Features concernées : {', '.join(drifted_features)}. "
            "Vérifier si un retraining est nécessaire."
        ),
        metrics={
            "drift_share": f"{drift_share:.1%}",
            "n_drifted": n_drift,
            "n_features": n_feat,
            "alert_level": alert_level,
            "timestamp": drift_result.get("timestamp", ""),
        },
    )


def check_latency_alert(avg_latency_ms: float, n_requests: int) -> Optional[Alert]:
    """Génère une alerte si la latence dépasse le seuil."""
    if avg_latency_ms <= LATENCY_THRESHOLD_MS or n_requests < 10:
        return None

    return Alert(
        level=AlertLevel.WARNING,
        title="Latence API Élevée",
        message=(
            f"Latence moyenne ({avg_latency_ms:.0f}ms) dépasse le seuil "
            f"({LATENCY_THRESHOLD_MS:.0f}ms) sur {n_requests} requêtes."
        ),
        metrics={
            "avg_latency_ms": f"{avg_latency_ms:.1f}",
            "threshold_ms": f"{LATENCY_THRESHOLD_MS:.0f}",
            "n_requests": n_requests,
            "degradation_factor": f"{avg_latency_ms / LATENCY_THRESHOLD_MS:.1f}x",
        },
    )


def check_error_rate_alert(n_errors: int, n_requests: int) -> Optional[Alert]:
    """Génère une alerte si le taux d'erreur dépasse le seuil."""
    if n_requests < 10:
        return None

    error_rate = n_errors / n_requests
    if error_rate <= ERROR_RATE_THRESHOLD:
        return None

    return Alert(
        level=AlertLevel.CRITICAL,
        title="Taux d'Erreur API Élevé",
        message=(
            f"Taux d'erreur ({error_rate:.1%}) dépasse le seuil "
            f"({ERROR_RATE_THRESHOLD:.1%}) sur {n_requests} requêtes."
        ),
        metrics={
            "error_rate": f"{error_rate:.1%}",
            "n_errors": n_errors,
            "n_requests": n_requests,
            "threshold": f"{ERROR_RATE_THRESHOLD:.1%}",
        },
    )


def check_model_performance_alert(test_f1: float) -> Optional[Alert]:
    """Génère une alerte si les métriques du modèle sont trop basses."""
    if test_f1 >= F1_MIN_THRESHOLD:
        return None

    return Alert(
        level=AlertLevel.CRITICAL,
        title="Performance Modèle Dégradée",
        message=(
            f"F1 macro ({test_f1:.4f}) sous le seuil minimum "
            f"({F1_MIN_THRESHOLD:.4f}). Retraining recommandé."
        ),
        metrics={
            "current_f1": f"{test_f1:.4f}",
            "threshold_f1": f"{F1_MIN_THRESHOLD:.4f}",
            "gap": f"{F1_MIN_THRESHOLD - test_f1:.4f}",
        },
    )


def run_all_checks(
    drift_result: Optional[dict] = None,
    avg_latency_ms: float = 0.0,
    n_requests: int = 0,
    n_errors: int = 0,
    test_f1: Optional[float] = None,
) -> list:
    """Lance toutes les vérifications et envoie les alertes nécessaires."""
    alerts_sent = []

    checks = [
        check_drift_alert(drift_result) if drift_result else None,
        check_latency_alert(avg_latency_ms, n_requests),
        check_error_rate_alert(n_errors, n_requests),
        check_model_performance_alert(test_f1) if test_f1 is not None else None,
    ]

    for alert in checks:
        if alert is not None:
            result = send_alert(alert)
            alerts_sent.append(
                {
                    "alert": alert.to_dict(),
                    "results": result,
                }
            )

    if not alerts_sent:
        log.info("✅ Tous les checks sont OK — aucune alerte")

    return alerts_sent


def get_alert_history(last_n: int = 20) -> list:
    """Retourne les N dernières alertes depuis le fichier JSONL."""
    if not ALERT_LOG.exists():
        return []
    alerts = []
    with open(ALERT_LOG) as f:
        for line in f:
            try:
                alerts.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return alerts[-last_n:]
