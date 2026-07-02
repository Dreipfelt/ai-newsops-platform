"""
Train DistilBERT on News Category Dataset (CPU/GPU compatible)
Usage: python src/models/train_cpu.py --epochs 4 --batch-size 16 --fast
"""

import os
import json
import time
import logging
import argparse
import warnings
from pathlib import Path

# ============================================================
# Configuration système AVANT les imports lourds
# ============================================================
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, accuracy_score, classification_report
import mlflow

# ============================================================
# Configuration du logging
# ============================================================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ============================================================
# Chemins et constantes
# ============================================================
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models/distilbert")
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
REPORTS_DIR = Path("reports/figures")

for d in [MODELS_DIR, CHECKPOINT_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_THREADS = torch.get_num_threads()


# ============================================================
# Dataset
# ============================================================
class NewsDataset(Dataset):
    """Dataset optimisé pour DistilBERT avec pré-tokenisation"""

    def __init__(self, df, tokenizer, max_length=128):
        self.labels = torch.tensor(df["label"].tolist(), dtype=torch.long)

        # Tokenisation batch pour performance
        enc = tokenizer(
            df["text"].tolist(),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],  # ← CLÉ CORRECTE: "labels"
        }


# ============================================================
# Fonctions d'entraînement et d'évaluation
# ============================================================
def train_epoch(model, loader, optimizer, scheduler, grad_clip=1.0):
    """Entraîne un epoch avec gradient clipping"""
    model.train()
    total_loss = 0.0
    n = len(loader)
    t0 = time.time()

    for i, batch in enumerate(loader):
        # Transfert sur device
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch)  # batch contient "labels" (au pluriel)
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += outputs.loss.item()

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / (i + 1)) * (n - i - 1) / 60
            log.info(
                f"  Step {i+1}/{n} — loss: {outputs.loss.item():.4f} — ETA: {eta:.1f}min"
            )

    return total_loss / n


@torch.no_grad()
def evaluate(model, loader):
    """Évalue le modèle sur un DataLoader"""
    model.eval()
    preds, all_labels, total_loss = [], [], 0.0

    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)  # batch contient "labels"
        total_loss += outputs.loss.item()
        preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())  # ← CLÉ CORRECTE

    return {
        "loss": total_loss / len(loader),
        "f1_macro": f1_score(all_labels, preds, average="macro"),
        "accuracy": accuracy_score(all_labels, preds),
        "predictions": preds,
        "labels": all_labels,
    }


# ============================================================
# Fonction principale
# ============================================================
def main(args):
    """Pipeline d'entraînement complet"""
    log.info(f"Device: {DEVICE} | Threads: {NUM_THREADS}")

    # 1. Chargement des données
    log.info("Chargement des données...")
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")

    if args.fast:
        train_df = train_df.sample(frac=0.15, random_state=42)
        log.warning(f"Mode --fast : train réduit à {len(train_df):,} exemples")

    with open(DATA_DIR / "label_mapping.json") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
    label2id = {v: k for k, v in id2label.items()}
    class_names = [id2label[i] for i in sorted(id2label.keys())]
    num_labels = len(id2label)

    log.info(
        f"Train:{len(train_df):,} Val:{len(val_df):,} Test:{len(test_df):,} Classes:{num_labels}"
    )

    # 2. Tokenisation
    log.info("Initialisation du tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    train_ds = NewsDataset(train_df, tokenizer)
    val_ds = NewsDataset(val_df, tokenizer)
    test_ds = NewsDataset(test_df, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0
    )

    # 3. Modèle
    log.info("Chargement du modèle DistilBERT...")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(DEVICE)

    # 4. Optimiseur et scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    log.info(
        f"Steps/epoch:{len(train_loader)} Total:{total_steps} Warmup:{warmup_steps}"
    )

    # 5. MLflow
    mlflow.set_experiment("news-classifier-distilbert")
    run_name = f"distilbert-{'fast' if args.fast else 'full'}-e{args.epochs}"

    with mlflow.start_run(run_name=run_name):
        # Log des paramètres
        mlflow.log_params(
            {
                "fast_mode": args.fast,
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
                "num_labels": num_labels,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs_max": args.epochs,
                "patience": args.patience,
                "warmup_ratio": 0.1,
                "device": str(DEVICE),
                "threads": NUM_THREADS,
            }
        )

        # 6. Boucle d'entraînement
        best_val_f1 = 0.0
        no_improve = 0
        best_path = MODELS_DIR / "best_model"
        history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
        t0 = time.time()

        for epoch in range(1, args.epochs + 1):
            log.info(f"\n{'='*50}\n  EPOCH {epoch}/{args.epochs}\n{'='*50}")

            # Train
            train_loss = train_epoch(model, train_loader, optimizer, scheduler)

            # Validation
            val_metrics = evaluate(model, val_loader)

            # Historique
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_f1"].append(val_metrics["f1_macro"])
            history["val_acc"].append(val_metrics["accuracy"])

            log.info(
                f"  Train loss:{train_loss:.4f} | "
                f"Val loss:{val_metrics['loss']:.4f} | "
                f"F1:{val_metrics['f1_macro']:.4f} | "
                f"Acc:{val_metrics['accuracy']:.4f}"
            )

            # MLflow metrics
            mlflow.log_metrics(
                {
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_metrics["loss"], 4),
                    "val_f1": round(val_metrics["f1_macro"], 4),
                    "val_accuracy": round(val_metrics["accuracy"], 4),
                },
                step=epoch,
            )

            # Checkpoint
            checkpoint_path = CHECKPOINT_DIR / f"epoch_{epoch}"
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)

            # Meilleur modèle
            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                no_improve = 0
                model.save_pretrained(best_path)
                tokenizer.save_pretrained(best_path)
                log.info(f"  ✅ Meilleur modèle sauvegardé (F1={best_val_f1:.4f})")
            else:
                no_improve += 1
                log.warning(f"  ⚠️ Pas d'amélioration ({no_improve}/{args.patience})")
                if no_improve >= args.patience:
                    log.warning(f"  🛑 Early stopping à l'epoch {epoch}")
                    break

        # 7. Évaluation finale sur le test
        total_min = (time.time() - t0) / 60
        log.info(
            f"\n🏁 Entraînement terminé en {total_min:.1f}min — Meilleur F1 val: {best_val_f1:.4f}"
        )

        best_model = DistilBertForSequenceClassification.from_pretrained(best_path)
        best_model.to(DEVICE)
        test_metrics = evaluate(best_model, test_loader)

        # Rapport détaillé
        print("\n" + "=" * 70)
        print("  RAPPORT DE CLASSIFICATION SUR LE TEST")
        print("=" * 70)
        print(
            classification_report(
                test_metrics["labels"],
                test_metrics["predictions"],
                target_names=class_names,
                digits=4,
            )
        )
        print(f"  Test F1 macro : {test_metrics['f1_macro']:.4f}")
        print(f"  Test Accuracy : {test_metrics['accuracy']:.4f}")
        print(f"  Baseline SVM  : 0.6515")
        print(f"  Delta         : {test_metrics['f1_macro'] - 0.6515:+.4f}")
        print("=" * 70)

        # 8. Sauvegarde des métriques
        mlflow.log_metrics(
            {
                "test_f1_macro": round(test_metrics["f1_macro"], 4),
                "test_accuracy": round(test_metrics["accuracy"], 4),
                "training_minutes": round(total_min, 1),
            }
        )

        metrics = {
            "model": "distilbert-base-uncased",
            "fast_mode": args.fast,
            "epochs_run": len(history["train_loss"]),
            "best_val_f1": round(best_val_f1, 4),
            "test_f1_macro": round(test_metrics["f1_macro"], 4),
            "test_accuracy": round(test_metrics["accuracy"], 4),
            "baseline_f1": 0.6515,
            "delta_f1": round(test_metrics["f1_macro"] - 0.6515, 4),
            "training_minutes": round(total_min, 1),
            "history": history,
            "class_names": class_names,
        }

        with open(MODELS_DIR / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        log.info(
            "  ✅ Métriques sauvegardées → models/distilbert/training_metrics.json"
        )
        log.info(f"  ✅ Modèle final → {best_path}")


# ============================================================
# Point d'entrée
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for News Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=4, help="Nombre d'epochs max")
    parser.add_argument("--batch-size", type=int, default=16, help="Taille de batch")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument(
        "--patience", type=int, default=2, help="Early stopping patience"
    )
    parser.add_argument(
        "--fast", action="store_true", help="Mode rapide (15% du train)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Seed pour reproductibilité
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)
