"""
src/models/tune_optuna.py
Hyperparameter tuning DistilBERT avec Optuna
AI NewsOps Platform · AIA Bloc 4

Recherche bayésienne sur : learning_rate, weight_decay, dropout, batch_size.
Chaque trial est loggé dans MLflow pour comparaison et traçabilité complète.

Usage :
  python src/models/tune_optuna.py --n-trials 10 --epochs-per-trial 1
  python src/models/tune_optuna.py --n-trials 20 --epochs-per-trial 2 --fast
"""

import json
import logging
import argparse
import warnings
from pathlib import Path

import optuna
import mlflow
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models/distilbert")
RESULTS_DIR = MODELS_DIR / "optuna"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(12)


class NewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
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


def load_data(fast: bool, sample_frac: float = 0.10):
    """Charge un sous-échantillon pour accélérer la recherche d'hyperparamètres.

    Note méthodologique : Optuna cherche le MEILLEUR JEU D'HYPERPARAMÈTRES,
    pas le meilleur modèle final. Un sous-échantillon suffit pour comparer
    des configurations entre elles — le modèle final est ensuite réentraîné
    sur 100% des données avec les hyperparamètres gagnants (voir train_cpu.py).
    """
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val.parquet")

    if fast:
        train_df = train_df.sample(frac=sample_frac, random_state=42)
        val_df = val_df.sample(frac=min(sample_frac * 2, 1.0), random_state=42)

    with open(DATA_DIR / "label_mapping.json") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}

    return train_df, val_df, id2label


def train_one_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    for batch in loader:
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
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        out = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        preds.extend(torch.argmax(out.logits, dim=-1).numpy())
        labels.extend(batch["label"].numpy())
    return f1_score(labels, preds, average="macro")


def objective(
    trial, train_ds, val_ds, num_labels, id2label, epochs_per_trial, batch_size_fixed
):
    """
    Fonction objectif Optuna : entraîne un modèle avec des hyperparamètres
    échantillonnés et retourne le F1 macro de validation à maximiser.
    """
    # ── Espace de recherche ──────────────────────────────────
    lr = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.15)

    log.info(
        f"Trial {trial.number} — lr={lr:.2e} wd={weight_decay:.3f} "
        f"dropout={dropout:.2f} warmup={warmup_ratio:.2f}"
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("news-classifier-optuna-tuning")

    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=False):
        mlflow.log_params(
            {
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "warmup_ratio": warmup_ratio,
                "trial_number": trial.number,
            }
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size_fixed, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size_fixed * 2, shuffle=False)

        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=num_labels,
            id2label=id2label,
            label2id={v: k for k, v in id2label.items()},
            dropout=dropout,
            seq_classif_dropout=dropout,
        ).to(DEVICE)

        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = len(train_loader) * epochs_per_trial
        warmup_steps = int(warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        best_f1 = 0.0
        for epoch in range(epochs_per_trial):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
            val_f1 = evaluate(model, val_loader)
            best_f1 = max(best_f1, val_f1)

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_f1_macro": val_f1}, step=epoch
            )
            log.info(
                f"  Trial {trial.number} epoch {epoch+1}: loss={train_loss:.4f} f1={val_f1:.4f}"
            )

            # Pruning : arrêter les trials clairement mauvais avant la fin
            trial.report(val_f1, epoch)
            if trial.should_prune():
                mlflow.log_metric("pruned", 1)
                raise optuna.TrialPruned()

        mlflow.log_metric("best_val_f1", best_f1)

    return best_f1


def main(args):
    log.info("=== Hyperparameter Tuning Optuna — DistilBERT ===")
    log.info(
        f"Trials: {args.n_trials} | Epochs/trial: {args.epochs_per_trial} | Fast mode: {args.fast}"
    )

    train_df, val_df, id2label = load_data(fast=args.fast, sample_frac=args.sample_frac)
    num_labels = len(id2label)

    log.info(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Classes: {num_labels}")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_ds = NewsDataset(train_df, tokenizer)
    val_ds = NewsDataset(val_df, tokenizer)

    study = optuna.create_study(
        study_name="distilbert_newsops_tuning",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: objective(
            trial,
            train_ds,
            val_ds,
            num_labels,
            id2label,
            args.epochs_per_trial,
            args.batch_size,
        ),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # ── Résultats ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RÉSULTATS OPTUNA — Meilleur essai")
    print("=" * 60)
    print(f"  Trial n°       : {study.best_trial.number}")
    print(f"  Val F1 macro   : {study.best_value:.4f}")
    print(f"  Hyperparamètres :")
    for k, v in study.best_params.items():
        print(f"    {k:<15} : {v}")
    print("=" * 60)

    # Sauvegarder les résultats
    results = {
        "best_trial": study.best_trial.number,
        "best_val_f1": study.best_value,
        "best_params": study.best_params,
        "n_trials": args.n_trials,
        "epochs_per_trial": args.epochs_per_trial,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }
    results_path = RESULTS_DIR / "optuna_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Résultats sauvegardés → {results_path}")

    # Générer les visualisations Optuna (si plotly disponible)
    try:
        import plotly

        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_html(str(RESULTS_DIR / "optimization_history.html"))

        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_html(str(RESULTS_DIR / "param_importances.html"))

        fig3 = optuna.visualization.plot_parallel_coordinate(study)
        fig3.write_html(str(RESULTS_DIR / "parallel_coordinate.html"))

        log.info(f"Visualisations Optuna → {RESULTS_DIR}/*.html")
    except ImportError:
        log.warning("plotly non installé — visualisations HTML non générées")

    print(f"\n✅ Terminé. Utilise ces hyperparamètres pour l'entraînement final :")
    print(f"   python src/models/train_cpu.py \\")
    for k, v in study.best_params.items():
        if k == "learning_rate":
            print(f"     --lr {v} \\")
    print(
        f"   (weight_decay={study.best_params.get('weight_decay'):.3f}, "
        f"dropout={study.best_params.get('dropout'):.3f} à ajouter dans train_cpu.py)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning Optuna pour DistilBERT"
    )
    parser.add_argument(
        "--n-trials", type=int, default=10, help="Nombre d'essais Optuna"
    )
    parser.add_argument(
        "--epochs-per-trial", type=int, default=1, help="Epochs par essai"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=0.10,
        help="Fraction du train set utilisée par essai (recherche rapide)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=True,
        help="Mode rapide — sous-échantillonnage (recommandé pour Optuna)",
    )
    main(parser.parse_args())
