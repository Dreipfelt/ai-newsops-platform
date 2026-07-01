import os
import json
import time
import logging
import argparse
import warnings

os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch

torch.set_num_threads(12)
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score, classification_report
import mlflow

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models/distilbert")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
REPORTS_DIR = Path("reports/figures")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cpu")


class NewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        log.info(f"  Pré-tokenisation de {len(df):,} exemples...")
        enc = tokenizer(
            df["text"].tolist(),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(df["label"].tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
        }


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    n = len(loader)
    t0 = time.time()
    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"],
        )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += out.loss.item()
        if (i + 1) % 100 == 0:
            eta = (n - i - 1) / ((i + 1) / (time.time() - t0)) / 60
            log.info(
                f"  Step {i+1}/{n} — loss: {out.loss.item():.4f} — ETA: {eta:.1f}min"
            )
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, labs, total_loss = [], [], 0.0
    for batch in loader:
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"],
        )
        total_loss += out.loss.item()
        preds.extend(torch.argmax(out.logits, dim=-1).numpy())
        labs.extend(batch["label"].numpy())
    return (
        total_loss / len(loader),
        f1_score(labs, preds, average="macro"),
        accuracy_score(labs, preds),
        preds,
        labs,
    )


def main(args):
    log.info(f"Device: {DEVICE} | Threads: {torch.get_num_threads()}")
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")

    if args.fast:
        train_df = train_df.sample(frac=0.15, random_state=42)
        log.warning(f"Mode --fast : train réduit à {len(train_df):,} exemples")

    with open(DATA_DIR / "label_mapping.json") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
    class_names = [id2label[i] for i in sorted(id2label.keys())]
    num_labels = len(id2label)
    log.info(
        f"Train:{len(train_df):,} Val:{len(val_df):,} Test:{len(test_df):,} Classes:{num_labels}"
    )

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

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_labels,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
        dropout=0.2,
        seq_classif_dropout=0.2,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.06 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    log.info(
        f"Steps/epoch:{len(train_loader)} Total:{total_steps} Warmup:{warmup_steps}"
    )
    log.info(f"ETA totale estimée: ~{len(train_loader)*args.epochs*0.15/60:.0f}min")

    mlflow.set_experiment("news-classifier-distilbert-cpu")
    with mlflow.start_run(run_name=f"distilbert-cpu-fast{args.fast}-e{args.epochs}"):
        mlflow.log_params(
            {
                "fast_mode": args.fast,
                "train_size": len(train_df),
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs_max": args.epochs,
            }
        )

        history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
        best_val_f1 = 0.0
        no_improve = 0
        best_path = MODELS_DIR / "best_model"
        t0 = time.time()

        for epoch in range(1, args.epochs + 1):
            log.info(f"\n{'='*50}\n  EPOCH {epoch}/{args.epochs}\n{'='*50}")
            train_loss = train_epoch(model, train_loader, optimizer, scheduler)
            val_loss, val_f1, val_acc, _, _ = evaluate(model, val_loader)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)
            history["val_acc"].append(val_acc)
            log.info(
                f"  Train loss:{train_loss:.4f} | Val loss:{val_loss:.4f} | F1:{val_f1:.4f} | Acc:{val_acc:.4f}"
            )
            mlflow.log_metrics(
                {
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "val_f1": round(val_f1, 4),
                },
                step=epoch,
            )
            model.save_pretrained(CHECKPOINT_DIR / f"epoch_{epoch}")
            tokenizer.save_pretrained(CHECKPOINT_DIR / f"epoch_{epoch}")
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                no_improve = 0
                model.save_pretrained(best_path)
                tokenizer.save_pretrained(best_path)
                log.info(f"  ✅ Meilleur modèle sauvegardé (F1={best_val_f1:.4f})")
            else:
                no_improve += 1
                log.warning(f"  ⚠️  Pas d'amélioration ({no_improve}/{args.patience})")
                if no_improve >= args.patience:
                    log.warning(f"  🛑 Early stopping epoch {epoch}")
                    break

        total_min = (time.time() - t0) / 60
        log.info(
            f"\n🏁 Terminé en {total_min:.1f}min — meilleur F1 val:{best_val_f1:.4f}"
        )

        best_model = DistilBertForSequenceClassification.from_pretrained(best_path)
        test_loss, test_f1, test_acc, test_preds, test_labels = evaluate(
            best_model, test_loader
        )
        print("\n" + "=" * 60)
        print(classification_report(test_labels, test_preds, target_names=class_names))
        print(f"  Test F1 macro : {test_f1:.4f} | Accuracy : {test_acc:.4f}")
        print(f"  Baseline SVM  : 0.6515 | Delta : {test_f1-0.6515:+.4f}")
        print("=" * 60)
        mlflow.log_metrics(
            {
                "test_f1_macro": round(test_f1, 4),
                "test_accuracy": round(test_acc, 4),
                "training_minutes": round(total_min, 1),
            }
        )
        metrics = {
            "model": "distilbert-base-uncased",
            "fast_mode": args.fast,
            "epochs_run": len(history["train_loss"]),
            "test_f1_macro": round(test_f1, 4),
            "test_accuracy": round(test_acc, 4),
            "best_val_f1": round(best_val_f1, 4),
            "baseline_f1": 0.6515,
            "delta_f1": round(test_f1 - 0.6515, 4),
            "history": history,
            "class_names": class_names,
        }
        with open(MODELS_DIR / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        log.info("  Métriques → models/distilbert/training_metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--fast", action="store_true")
    main(parser.parse_args())
