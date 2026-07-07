"""
src/monitoring/drift_detector.py
Détection de drift — AI NewsOps Platform
AIA Bloc 4 — MLOps Pipeline

Stratégie :
  - scipy (KS test + Chi²) : check rapide, toujours disponible, utilisé par l'API
  - Evidently 0.7.x         : rapport HTML visuel optionnel (--full-report)

Usage :
  python src/monitoring/drift_detector.py
  python src/monitoring/drift_detector.py --full-report
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DATA_DIR = Path("data/processed")
MONITORING_DIR = Path("monitoring")
REPORTS_DIR = MONITORING_DIR / "reports"
DRIFT_LOG = MONITORING_DIR / "drift_log.jsonl"
QUICK_DRIFT = MONITORING_DIR / "quick_drift_latest.json"

MONITORING_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

# Seuils d'alerte
DRIFT_SHARE_THRESHOLD = 0.15  # >15% de features driftées → alerte
TEXT_LENGTH_DRIFT_PVALUE = 0.05
CATEGORY_DRIFT_PVALUE = 0.05


# ─────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────


def load_reference_data(sample_size: int = 2000) -> pd.DataFrame:
    """Charge les données de référence (train set)."""
    df = pd.read_parquet(DATA_DIR / "train.parquet")
    ref = df.sample(n=min(sample_size, len(df)), random_state=42)

    with open(DATA_DIR / "label_mapping.json") as f:
        raw = json.load(f)
    id2label = {int(k): v for k, v in raw.items()}

    ref = ref.copy()
    ref["category_name"] = ref["label"].map(id2label)
    ref["text_length"] = ref["text"].str.len()
    ref["word_count"] = ref["text"].str.split().str.len()

    log.info(f"Référence chargée : {len(ref):,} exemples")
    return ref[["text", "text_length", "word_count", "category_name", "label"]]


def load_production_data(batch_size: int = 500) -> pd.DataFrame:
    """Simule les données de production depuis le test set."""
    df = pd.read_parquet(DATA_DIR / "test.parquet")
    prod = df.sample(
        n=min(batch_size, len(df)), random_state=int(datetime.now().timestamp()) % 1000
    )

    with open(DATA_DIR / "label_mapping.json") as f:
        raw = json.load(f)
    id2label = {int(k): v for k, v in raw.items()}

    prod = prod.copy()
    prod["category_name"] = prod["label"].map(id2label)
    prod["text_length"] = prod["text"].str.len()
    prod["word_count"] = prod["text"].str.split().str.len()

    log.info(f"Production chargée : {len(prod):,} exemples")
    return prod[["text", "text_length", "word_count", "category_name", "label"]]


# ─────────────────────────────────────────────────────────────
# DRIFT CHECK RAPIDE — scipy (principal)
# ─────────────────────────────────────────────────────────────


def quick_drift_check(
    reference: pd.DataFrame,
    production: pd.DataFrame,
) -> dict:
    """
    Détection de drift avec tests statistiques scipy.
    - KS test (Kolmogorov-Smirnov) pour les features numériques
    - Chi² pour la distribution des catégories

    Retourne en < 1 seconde, sans dépendance Evidently.
    """
    results = {}

    # ── KS test sur text_length ──────────────────────────────
    ks_stat, ks_pval = stats.ks_2samp(
        reference["text_length"].values,
        production["text_length"].values,
    )
    results["text_length"] = {
        "test": "kolmogorov-smirnov",
        "statistic": round(float(ks_stat), 4),
        "p_value": round(float(ks_pval), 4),
        "drift_detected": bool(ks_pval < TEXT_LENGTH_DRIFT_PVALUE),
        "interpretation": "Longueur des textes en production",
    }

    # ── KS test sur word_count ───────────────────────────────
    ks_stat2, ks_pval2 = stats.ks_2samp(
        reference["word_count"].values,
        production["word_count"].values,
    )
    results["word_count"] = {
        "test": "kolmogorov-smirnov",
        "statistic": round(float(ks_stat2), 4),
        "p_value": round(float(ks_pval2), 4),
        "drift_detected": bool(ks_pval2 < TEXT_LENGTH_DRIFT_PVALUE),
        "interpretation": "Nombre de mots par article",
    }

    # ── Chi² sur la distribution des catégories ──────────────
    ref_counts = reference["category_name"].value_counts()
    prod_counts = production["category_name"].value_counts()
    all_cats = sorted(set(ref_counts.index) | set(prod_counts.index))

    ref_freq = np.array([ref_counts.get(c, 0) for c in all_cats], dtype=float)
    prod_freq = np.array([prod_counts.get(c, 0) for c in all_cats], dtype=float)

    # Normaliser pour avoir des fréquences attendues proportionnelles
    ref_freq = np.maximum(ref_freq, 1e-10)
    expected = ref_freq / ref_freq.sum() * prod_freq.sum()
    expected = np.maximum(expected, 1e-10)

    chi2_stat, chi2_pval = stats.chisquare(prod_freq, f_exp=expected)
    results["category_distribution"] = {
        "test": "chi-squared",
        "statistic": round(float(chi2_stat), 4),
        "p_value": round(float(chi2_pval), 4),
        "drift_detected": bool(chi2_pval < CATEGORY_DRIFT_PVALUE),
        "interpretation": "Distribution des catégories prédites",
        "top_categories_ref": ref_counts.head(5).to_dict(),
        "top_categories_prod": prod_counts.head(5).to_dict(),
    }

    # ── Résumé ───────────────────────────────────────────────
    n_drifted = sum(1 for v in results.values() if v["drift_detected"])
    drift_share = n_drifted / len(results)

    alert_level = "ok"
    if drift_share > 0:
        alert_level = "warning"
    if drift_share >= 0.67:
        alert_level = "critical"

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_drifted": n_drifted,
        "n_features": len(results),
        "drift_share": round(drift_share, 3),
        "alert_level": alert_level,
        "reference_size": len(reference),
        "production_size": len(production),
        "features": results,
        "thresholds": {
            "numerical_pvalue": TEXT_LENGTH_DRIFT_PVALUE,
            "categorical_pvalue": CATEGORY_DRIFT_PVALUE,
            "drift_share_alert": DRIFT_SHARE_THRESHOLD,
        },
    }

    # Sauvegarder
    with open(QUICK_DRIFT, "w") as f:
        json.dump(summary, f, indent=2)
    with open(DRIFT_LOG, "a") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": summary["timestamp"],
                    "alert_level": alert_level,
                    "drift_share": drift_share,
                    "n_drifted": n_drifted,
                }
            )
            + "\n"
        )

    return summary


# ─────────────────────────────────────────────────────────────
# RAPPORT EVIDENTLY 0.7.x (optionnel, pour rapport HTML visuel)
# ─────────────────────────────────────────────────────────────


def generate_evidently_report(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    report_name: Optional[str] = None,
) -> str:
    """
    Génère un rapport HTML Evidently 0.7.x.
    Import lazy pour ne pas bloquer si Evidently n'est pas compatible.
    """
    try:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = report_name or f"drift_report_{timestamp}"
        report_path = REPORTS_DIR / f"{report_name}.html"

        # Colonnes numériques uniquement pour Evidently
        ref_ev = reference[["text_length", "word_count"]].copy()
        prod_ev = production[["text_length", "word_count"]].copy()

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_ev, current_data=prod_ev)
        report.save_html(str(report_path))

        log.info(f"✅ Rapport Evidently → {report_path}")
        return str(report_path)

    except ImportError as e:
        log.warning(f"Evidently non disponible pour le rapport HTML : {e}")
        return ""
    except Exception as e:
        log.error(f"Erreur rapport Evidently : {e}")
        return ""


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────


def main(args):
    log.info("=== Drift Detection — AI NewsOps Platform ===")

    reference = load_reference_data(sample_size=2000)
    production = load_production_data(batch_size=args.batch_size)

    log.info("Vérification rapide (scipy KS + Chi²)...")
    result = quick_drift_check(reference, production)

    print("\n── Résultat ──────────────────────────────────")
    print(f"  Alert level       : {result['alert_level'].upper()}")
    print(f"  Features driftées : {result['n_drifted']}/{result['n_features']}")
    print(f"  Drift share       : {result['drift_share']:.1%}")
    print()
    for feat, res in result["features"].items():
        icon = "🔴" if res["drift_detected"] else "🟢"
        print(
            f"  {icon} {feat:<30} p={res['p_value']:.4f}  stat={res['statistic']:.4f}"
        )

    if args.full_report:
        log.info("\nGénération rapport HTML Evidently...")
        path = generate_evidently_report(reference, production)
        if path:
            print(f"\n  📊 Rapport HTML → {path}")

    print(f"\n  Résultat sauvegardé → {QUICK_DRIFT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Détection de drift")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--full-report", action="store_true")
    main(parser.parse_args())
