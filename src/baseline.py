"""
src/models/baseline.py
Modèle baseline : TF-IDF + LogisticRegression
AI NewsOps Platform · AIA Bloc 4

Objectif : F1 macro ≥ 0.82
Usage    : python src/models/baseline.py
"""

import json
import logging
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
import mlflow
import mlflow.sklearn
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR    = Path("data/processed")
REPORTS_DIR = Path("reports/figures")
MODELS_DIR  = Path("models/baseline")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_splits():
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    val   = pd.read_parquet(DATA_DIR / "val.parquet")
    test  = pd.read_parquet(DATA_DIR / "test.parquet")
    with open(DATA_DIR / "label_mapping.json") as f:
        label_mapping = json.load(f)
    # int keys → str keys dans JSON, on inverse : label_int → nom_classe
    id2label = {int(k): v for k, v in label_mapping.items()}
    log.info(f"Données chargées — train:{len(train):,} val:{len(val):,} test:{len(test):,}")
    return train, val, test, id2label


def build_pipeline() -> Pipeline:
    """
    Pipeline sklearn :
      TF-IDF (1-2 grammes, 50k features, sous-linéaire TF)
      → LogisticRegression (multi-class, class_weight=balanced, solver lbfgs)

    Choix justifiés :
    - sublinear_tf=True : atténue l'effet des mots très fréquents
    - ngram_range=(1,2) : capture les bigrammes "breaking news", "white house"
    - class_weight='balanced' : compense le déséquilibre résiduel (8x)
    - max_iter=1000 : assure la convergence sur 15 classes
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            stop_words="english",
            min_df=3,
        )),
        ("clf", LogisticRegression(
            C=5.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="multinomial",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def plot_confusion_matrix(y_true, y_pred, class_names: list, split: str):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.4, cbar_kws={"label": "Proportion"},
    )
    ax.set_xlabel("Prédit", fontsize=12)
    ax.set_ylabel("Réel", fontsize=12)
    ax.set_title(f"Matrice de confusion normalisée — Baseline TF-IDF ({split})",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    path = REPORTS_DIR / f"baseline_confusion_matrix_{split}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_per_class_f1(report_dict: dict, class_names: list, split: str):
    f1_scores = {cls: report_dict[cls]["f1-score"] for cls in class_names if cls in report_dict}
    sorted_items = sorted(f1_scores.items(), key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#D85A30" if v < 0.75 else "#5B4FCF" for _, v in sorted_items]
    bars = ax.barh([k for k, _ in sorted_items], [v for _, v in sorted_items],
                   color=colors, height=0.65)
    ax.axvline(0.75, color="#D85A30", lw=1.5, linestyle="--", label="Seuil 0.75")
    ax.axvline(np.mean(list(f1_scores.values())), color="#1D9E75", lw=1.5,
               linestyle="--", label=f"Macro avg: {np.mean(list(f1_scores.values())):.3f}")
    for bar, (_, val) in zip(bars, sorted_items):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1-score")
    ax.set_title(f"F1-score par classe — Baseline TF-IDF ({split})",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = REPORTS_DIR / f"baseline_f1_per_class_{split}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


def main(args):
    mlflow.set_experiment("news-classifier-baseline")

    train, val, test, id2label = load_splits()
    class_names = [id2label[i] for i in sorted(id2label.keys())]

    X_train, y_train = train["text_clean"], train["label"]
    X_val,   y_val   = val["text_clean"],   val["label"]
    X_test,  y_test  = test["text_clean"],  test["label"]

    with mlflow.start_run(run_name="tfidf-logreg-baseline"):

        # ── Entraînement ─────────────────────────────────────────────────────
        log.info("Entraînement du pipeline TF-IDF + LogisticRegression...")
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)
        log.info("  Entraînement terminé.")

        # ── Évaluation sur Val ────────────────────────────────────────────────
        y_val_pred = pipeline.predict(X_val)
        val_f1  = f1_score(y_val, y_val_pred, average="macro")
        val_acc = accuracy_score(y_val, y_val_pred)

        # ── Évaluation sur Test ───────────────────────────────────────────────
        y_test_pred = pipeline.predict(X_test)
        test_f1  = f1_score(y_test, y_test_pred, average="macro")
        test_acc = accuracy_score(y_test, y_test_pred)

        log.info(f"\n  ── Résultats VALIDATION ──────────────────────────────")
        log.info(f"  F1 macro : {val_f1:.4f}  |  Accuracy : {val_acc:.4f}")
        log.info(f"\n  ── Résultats TEST ────────────────────────────────────")
        log.info(f"  F1 macro : {test_f1:.4f}  |  Accuracy : {test_acc:.4f}")

        report_val  = classification_report(y_val,  y_val_pred,
                                            target_names=class_names, output_dict=True)
        report_test = classification_report(y_test, y_test_pred,
                                            target_names=class_names, output_dict=True)

        print("\n" + classification_report(y_test, y_test_pred, target_names=class_names))

        # ── MLflow logging ────────────────────────────────────────────────────
        tfidf_params = pipeline.named_steps["tfidf"].get_params()
        clf_params   = pipeline.named_steps["clf"].get_params()
        mlflow.log_params({
            "tfidf_max_features":  tfidf_params["max_features"],
            "tfidf_ngram_range":   str(tfidf_params["ngram_range"]),
            "tfidf_sublinear_tf":  tfidf_params["sublinear_tf"],
            "clf_C":               clf_params["C"],
            "clf_class_weight":    clf_params["class_weight"],
            "clf_solver":          clf_params["solver"],
            "train_size":          len(train),
        })
        mlflow.log_metrics({
            "val_f1_macro":   round(val_f1,  4),
            "val_accuracy":   round(val_acc, 4),
            "test_f1_macro":  round(test_f1, 4),
            "test_accuracy":  round(test_acc, 4),
        })

        # ── Figures ───────────────────────────────────────────────────────────
        cm_path  = plot_confusion_matrix(y_test, y_test_pred, class_names, "test")
        f1_path  = plot_per_class_f1(report_test, class_names, "test")
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(f1_path)

        # ── Sauvegarde modèle ─────────────────────────────────────────────────
        model_path = MODELS_DIR / "tfidf_logreg.pkl"
        joblib.dump(pipeline, model_path)
        mlflow.sklearn.log_model(pipeline, "model",
                                 registered_model_name="news-classifier-baseline")
        log.info(f"  Modèle sauvegardé : {model_path}")

        # ── Résumé final ──────────────────────────────────────────────────────
        print("\n" + "=" * 55)
        print("  BASELINE — RÉSUMÉ")
        print("=" * 55)
        print(f"  Val  F1 macro  : {val_f1:.4f}")
        print(f"  Val  Accuracy  : {val_acc:.4f}")
        print(f"  Test F1 macro  : {test_f1:.4f}")
        print(f"  Test Accuracy  : {test_acc:.4f}")
        target = "✅ ATTEINT" if test_f1 >= 0.82 else "❌ EN DESSOUS"
        print(f"  Objectif F1 ≥ 0.82 : {target}")
        print("=" * 55)

        return test_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline TF-IDF + LogisticRegression")
    parser.add_argument("--data-dir", default="data/processed")
    main(parser.parse_args())
