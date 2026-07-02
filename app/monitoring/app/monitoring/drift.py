import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPerformancePreset
import logging

log = logging.getLogger(__name__)

# Chemins
REFERENCE_DATA_PATH = Path("data/processed/test.parquet")
MODEL_PATH = Path("models/distilbert/best_model")
DRIFT_LOGS_PATH = Path("monitoring/drift_logs.json")


class ModelMonitor:
    """Moniteur de dérive pour le modèle de classification"""

    def __init__(self, reference_data_path=REFERENCE_DATA_PATH):
        self.reference_data = None
        self.column_mapping = None
        self.drift_logs = []
        self._load_reference_data(reference_data_path)
        self._load_drift_logs()

    def _load_reference_data(self, path):
        """Charge les données de référence pour la détection de drift"""
        if path.exists():
            df = pd.read_parquet(path)
            # Prendre un échantillon pour la référence
            sample_size = min(5000, len(df))
            self.reference_data = df.sample(n=sample_size, random_state=42)
            # Ajouter des prédictions simulées pour la référence
            self.reference_data["prediction"] = np.random.randint(0, 13, len(self.reference_data))
            self.column_mapping = ColumnMapping(
                prediction="prediction",
                target="label",
                numerical_features=[],
                categorical_features=[],
                text_features=["text"],
            )
            log.info(f"✅ Référence chargée: {len(self.reference_data)} exemples")
        else:
            log.warning(f"⚠️ Fichier de référence non trouvé: {path}")

    def _load_drift_logs(self):
        """Charge les logs de drift existants"""
        if DRIFT_LOGS_PATH.exists():
            with open(DRIFT_LOGS_PATH) as f:
                self.drift_logs = json.load(f)
            log.info(f"✅ Logs de drift chargés: {len(self.drift_logs)} entrées")

    def _save_drift_logs(self):
        """Sauvegarde les logs de drift"""
        DRIFT_LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DRIFT_LOGS_PATH, "w") as f:
            json.dump(self.drift_logs, f, indent=2)

    def detect_drift(self, current_data, threshold=0.5):
        """Détecte la dérive entre les données de référence et les données courantes"""
        if self.reference_data is None:
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "error": "No reference data available"
            }, None

        # Convertir les données courantes en DataFrame
        if isinstance(current_data, dict):
            current_df = pd.DataFrame([current_data])
        elif isinstance(current_data, list):
            current_df = pd.DataFrame(current_data)
        elif isinstance(current_data, pd.DataFrame):
            current_df = current_data
        else:
            return {"error": "Unsupported data format"}, None

        # Ajouter une colonne de prédiction si nécessaire
        if "prediction" not in current_df.columns:
            current_df["prediction"] = np.random.randint(0, 13, len(current_df))

        # Générer le rapport de drift
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=self.reference_data,
            current_data=current_df,
            column_mapping=self.column_mapping,
        )

        # Extraire les métriques
        drift_metrics = report.as_dict()
        drift_detected = False
        drift_score = 0.0
        drift_ratio = 0.0

        if "metrics" in drift_metrics:
            for metric in drift_metrics["metrics"]:
                if metric["metric"] == "DataDriftPreset":
                    drift_detected = metric["result"]["drift_detected"]
                    drift_score = metric["result"].get("drift_score", 0.0)
                    drift_ratio = metric["result"].get("drift_ratio", 0.0)

        # Déterminer si le drift dépasse le seuil
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

        # Logger
        self.drift_logs.append(result)
        self._save_drift_logs()

        log.info(f"📊 Drift: score={drift_score:.3f}, detected={drift_detected}, significant={significant_drift}")

        return result, report

    def get_performance_report(self, predictions, labels):
        """Rapport de performance du modèle"""
        if self.reference_data is None:
            return None

        current_df = pd.DataFrame({
            "text": self.reference_data["text"].iloc[:len(predictions)],
            "label": labels,
            "prediction": predictions,
        })

        report = Report(metrics=[ClassificationPerformancePreset()])
        report.run(
            reference_data=self.reference_data,
            current_data=current_df,
            column_mapping=self.column_mapping,
        )

        return report.as_dict()

    def get_recent_drifts(self, limit=10):
        """Récupère les derniers logs de drift"""
        return self.drift_logs[-limit:] if self.drift_logs else []

    def get_drift_summary(self):
        """Résumé des dérives récentes"""
        if not self.drift_logs:
            return {"total": 0, "recent_drifts": 0, "last_drift": None}

        recent = self.drift_logs[-20:]  # Derniers 20 logs
        drifts = [log for log in recent if log.get("significant_drift", False)]
        
        return {
            "total": len(self.drift_logs),
            "recent_drifts": len(drifts),
            "last_drift": drifts[-1] if drifts else None,
            "last_check": recent[-1] if recent else None,
        }