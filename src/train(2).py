"""
src/models/train.py
Fine-tuning DistilBERT pour la classification de news (15 classes)
AI NewsOps Platform · AIA Bloc 4

Objectif   : F1 macro ≥ 0.88
Modèle     : distilbert-base-uncased (HuggingFace)
Usage      : python src/models/train.py [--epochs 3] [--batch-size 32] [--lr 2e-5]
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models/distilbert")
REPORTS_DIR = Path("reports/figures")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device : {DEVICE}")


# ─────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────
class NewsDataset(Dataset):
    """
    Dataset PyTorch pour DistilBERT.
    Tokenise les textes à la volée avec truncation à max_length.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 128):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device) -> float:
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        loss.backward()

        # Gradient clipping (stabilité)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            log.info(f"  Step {batch_idx+1}/{len(loader)} — loss: {loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        total_loss += outputs.loss.item()

        preds = torch.argmax(outputs.logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), f1, acc, all_preds, all_labels


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────
def plot_training_curves(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(
        epochs, history["train_loss"], "o-", color="#5B4FCF", lw=2, label="Train loss"
    )
    axes[0].plot(
        epochs, history["val_loss"], "o-", color="#D85A30", lw=2, label="Val loss"
    )
    axes[0].set_title("Loss par epoch", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CrossEntropy Loss")
    axes[0].legend()

    axes[1].plot(
        epochs, history["val_f1"], "o-", color="#1D9E75", lw=2, label="Val F1 macro"
    )
    axes[1].plot(
        epochs,
        history["val_acc"],
        "o-",
        color="#5B4FCF",
        lw=2,
        linestyle="--",
        label="Val Accuracy",
    )
    axes[1].axhline(
        0.88, color="#D85A30", lw=1.5, linestyle=":", label="Objectif F1=0.88"
    )
    axes[1].set_title("Métriques par epoch", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].legend()

    plt.suptitle(
        "DistilBERT Fine-tuning — Courbes d'entraînement",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    path = REPORTS_DIR / "distilbert_training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return str(path)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main(args):
    mlflow.set_experiment("news-classifier-distilbert")

    # ── Chargement données ────────────────────────────────────
    log.info("Chargement des données...")
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")

    with open(DATA_DIR / "label_mapping.json") as f:
        label_mapping = json.load(f)
    id2label = {int(k): v for k, v in label_mapping.items()}
    num_labels = len(id2label)
    class_names = [id2label[i] for i in sorted(id2label.keys())]

    log.info(
        f"  Train: {len(train_df):,} · Val: {len(val_df):,} · Test: {len(test_df):,}"
    )
    log.info(f"  Nombre de classes : {num_labels}")

    # ── Tokenizer & Datasets ──────────────────────────────────
    log.info(f"Chargement du tokenizer {MODEL_NAME}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_ds = NewsDataset(train_df, tokenizer, args.max_length)
    val_ds = NewsDataset(val_df, tokenizer, args.max_length)
    test_ds = NewsDataset(test_df, tokenizer, args.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ── Modèle ────────────────────────────────────────────────
    log.info("Initialisation de DistilBERT...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )
    model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Paramètres totaux    : {total_params:,}")
    log.info(f"  Paramètres entraînés : {trainable_params:,}")

    # ── Optimizer & Scheduler ─────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Run MLflow ────────────────────────────────────────────
    with mlflow.start_run(
        run_name=f"distilbert-e{args.epochs}-lr{args.lr}-bs{args.batch_size}"
    ):

        mlflow.log_params(
            {
                "model_name": MODEL_NAME,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "max_length": args.max_length,
                "warmup_steps": warmup_steps,
                "total_steps": total_steps,
                "weight_decay": 0.01,
                "num_labels": num_labels,
                "device": str(DEVICE),
                "train_size": len(train_df),
            }
        )

        history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
        best_val_f1 = 0.0
        best_model_path = MODELS_DIR / "best_model"

        # ── Boucle d'entraînement ─────────────────────────────
        for epoch in range(1, args.epochs + 1):
            log.info(f"\n{'='*55}")
            log.info(f"  EPOCH {epoch}/{args.epochs}")
            log.info(f"{'='*55}")

            train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
            val_loss, val_f1, val_acc, _, _ = evaluate(model, val_loader, DEVICE)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)
            history["val_acc"].append(val_acc)

            log.info(f"  Train loss : {train_loss:.4f}")
            log.info(f"  Val   loss : {val_loss:.4f}")
            log.info(f"  Val   F1   : {val_f1:.4f}  |  Acc : {val_acc:.4f}")

            mlflow.log_metrics(
                {
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "val_f1": round(val_f1, 4),
                    "val_acc": round(val_acc, 4),
                },
                step=epoch,
            )

            # Sauvegarder le meilleur modèle
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                model.save_pretrained(best_model_path)
                tokenizer.save_pretrained(best_model_path)
                log.info(
                    f"  ✅ Nouveau meilleur modèle sauvegardé (F1={best_val_f1:.4f})"
                )

        # ── Évaluation finale sur TEST ────────────────────────
        log.info("\nChargement du meilleur modèle pour évaluation test...")
        best_model = DistilBertForSequenceClassification.from_pretrained(
            best_model_path
        )
        best_model.to(DEVICE)

        test_loss, test_f1, test_acc, test_preds, test_labels = evaluate(
            best_model, test_loader, DEVICE
        )

        print("\n" + "=" * 60)
        print("  DISTILBERT — RÉSULTATS TEST (meilleur checkpoint)")
        print("=" * 60)
        print(classification_report(test_labels, test_preds, target_names=class_names))

        mlflow.log_metrics(
            {
                "test_f1_macro": round(test_f1, 4),
                "test_accuracy": round(test_acc, 4),
                "best_val_f1": round(best_val_f1, 4),
            }
        )

        target = "✅ ATTEINT" if test_f1 >= 0.88 else "❌ EN DESSOUS"
        log.info(f"  Test F1 macro : {test_f1:.4f}  |  Objectif F1 ≥ 0.88 : {target}")
        log.info(f"  Test Accuracy : {test_acc:.4f}")

        # ── Courbes d'entraînement ────────────────────────────
        curve_path = plot_training_curves(history)
        mlflow.log_artifact(curve_path)

        # ── Log modèle final dans le Model Registry ───────────
        mlflow.pytorch.log_model(
            best_model,
            artifact_path="model",
            registered_model_name="news-classifier-distilbert",
        )
        mlflow.log_artifact(str(best_model_path / "config.json"))
        mlflow.log_artifact(str(DATA_DIR / "label_mapping.json"))

        print("\n  Modèle enregistré dans MLflow Model Registry ✅")
        print(f"  Figures → {REPORTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DistilBERT fine-tuning")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    main(parser.parse_args())
