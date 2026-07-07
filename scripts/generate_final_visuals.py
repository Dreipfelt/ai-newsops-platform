"""
scripts/generate_final_visuals.py
Génère les visuels finaux pour la présentation à partir des VRAIES données
du modèle entraîné — aucune donnée inventée.

Produit :
  1. docs/visuals/confusion_matrix.png   — matrice de confusion réelle sur le test set
  2. docs/visuals/training_curves.png    — courbes loss/F1 réelles par epoch
  3. docs/visuals/class_distribution.png — distribution des 13 classes (train)

Usage :
  python scripts/generate_final_visuals.py
"""

import json
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# ─────────────────────────────────────────────────────────────
# CONFIG — palette cohérente avec les slides
# ─────────────────────────────────────────────────────────────
TEAL   = "#0D9488"
NAVY   = "#1E2761"
GOLD   = "#F59E0B"
RED    = "#DC2626"
GREEN  = "#16A34A"
SLATE  = "#475569"

MODEL_DIR   = Path("models/distilbert/best_model")
DATA_DIR    = Path("data/processed")
METRICS_FILE = Path("models/distilbert/training_metrics.json")
OUTPUT_DIR  = Path("docs/visuals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


def load_label_mapping():
    with open(DATA_DIR / "label_mapping.json") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ─────────────────────────────────────────────────────────────
# 1. MATRICE DE CONFUSION RÉELLE
# ─────────────────────────────────────────────────────────────
def generate_confusion_matrix(id2label, sample_size=2000):
    print("→ Chargement du modèle et du tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR, local_files_only=True)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
    model.eval()

    print(f"→ Chargement d'un échantillon du test set ({sample_size} exemples)...")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    if len(test_df) > sample_size:
        test_df = test_df.sample(n=sample_size, random_state=42)

    class_names = [id2label[i] for i in sorted(id2label.keys())]

    print("→ Inférence en cours (peut prendre plusieurs minutes sur CPU)...")
    preds, labels = [], []
    batch_size = 32
    texts = test_df["text"].tolist()
    true_labels = test_df["label"].tolist()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_labels = true_labels[i:i + batch_size]
            enc = tokenizer(
                batch_texts, max_length=128, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            out = model(**enc)
            batch_preds = torch.argmax(out.logits, dim=-1).tolist()
            preds.extend(batch_preds)
            labels.extend(batch_labels)
            if (i // batch_size) % 10 == 0:
                print(f"  {i}/{len(texts)} traités...")

    print("→ Génération du rapport de classification...")
    report = classification_report(labels, preds, target_names=class_names, output_dict=True)
    print(classification_report(labels, preds, target_names=class_names))

    # Matrice de confusion normalisée
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.4, cbar_kws={"label": "Proportion"},
    )
    ax.set_xlabel("Prédit", fontsize=12)
    ax.set_ylabel("Réel", fontsize=12)
    ax.set_title(
        f"Matrice de confusion — DistilBERT (test set, n={len(labels)})\n"
        f"F1 macro réel sur cet échantillon : {report['macro avg']['f1-score']:.4f}",
        fontsize=13, fontweight="bold",
    )
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Matrice de confusion → {out_path}")

    # Sauver le rapport texte aussi
    with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
        f.write(classification_report(labels, preds, target_names=class_names))

    return report


# ─────────────────────────────────────────────────────────────
# 2. COURBES D'ENTRAÎNEMENT RÉELLES
# ─────────────────────────────────────────────────────────────
def generate_training_curves():
    if not METRICS_FILE.exists():
        print("⚠️  training_metrics.json introuvable — courbes non générées")
        return

    with open(METRICS_FILE) as f:
        metrics = json.load(f)

    history = metrics.get("history", {})
    if not history:
        print("⚠️  Pas d'historique dans training_metrics.json")
        return

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, history["train_loss"], "o-", color=TEAL, lw=2.5, markersize=8, label="Train loss")
    axes[0].plot(epochs, history["val_loss"], "o-", color=RED, lw=2.5, markersize=8, label="Val loss")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].set_title("Évolution de la loss", fontsize=13, fontweight="bold")
    axes[0].set_xticks(list(epochs))
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.2)

    axes[1].plot(epochs, history["val_f1"], "o-", color=NAVY, lw=2.5, markersize=8, label="Val F1 macro")
    axes[1].plot(epochs, history["val_acc"], "o-", color=GOLD, lw=2.5, markersize=8, linestyle="--", label="Val Accuracy")
    axes[1].axhline(metrics.get("baseline_f1", 0.6515), color=SLATE, lw=1.5, linestyle=":", label=f"Baseline F1 ({metrics.get('baseline_f1', 0.6515):.4f})")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Score", fontsize=11)
    axes[1].set_title("Métriques de validation", fontsize=13, fontweight="bold")
    axes[1].set_xticks(list(epochs))
    axes[1].set_ylim(0.55, 0.85)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.2)

    fig.suptitle(
        f"DistilBERT fine-tuning — {metrics.get('epochs_run', '?')} epochs "
        f"({'mode rapide' if metrics.get('fast_mode') else 'dataset complet'})",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    out_path = OUTPUT_DIR / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Courbes d'entraînement → {out_path}")


# ─────────────────────────────────────────────────────────────
# 3. DISTRIBUTION DES CLASSES (train)
# ─────────────────────────────────────────────────────────────
def generate_class_distribution(id2label):
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    counts = train_df["label"].map(id2label).value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = [TEAL if v > 5000 else GOLD if v > 2000 else RED for v in counts.values]
    bars = ax.barh(counts.index, counts.values, color=colors, height=0.7)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=10)

    ax.set_xlabel("Nombre d'exemples (train)", fontsize=12)
    ax.set_title(
        f"Distribution des 13 super-catégories — train set (n={len(train_df):,})\n"
        f"Ratio déséquilibre : {counts.max()/counts.min():.1f}×",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out_path = OUTPUT_DIR / "class_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Distribution des classes → {out_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("═" * 60)
    print("  Génération des visuels finaux — AI NewsOps Platform")
    print("═" * 60)

    id2label = load_label_mapping()

    print("\n[1/3] Courbes d'entraînement (rapide, données déjà en mémoire)")
    generate_training_curves()

    print("\n[2/3] Distribution des classes (rapide)")
    generate_class_distribution(id2label)

    print("\n[3/3] Matrice de confusion (plus long — inférence sur 2000 exemples)")
    generate_confusion_matrix(id2label, sample_size=2000)

    print("\n" + "═" * 60)
    print(f"  ✅ Terminé — visuels dans {OUTPUT_DIR}/")
    print("═" * 60)
