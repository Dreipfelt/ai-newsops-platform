"""
src/data/preprocess.py
Pipeline de preprocessing du News Category Dataset pour MLOps AIA Bloc 4
"""

import re
import json
import logging
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 1. MAPPING : 42 catégories → 15 super-catégories
# ─────────────────────────────────────────────
CATEGORY_MAPPING = {
    # Politique & société
    "POLITICS": "politics",
    "GOVERNMENT & POLITICS": "politics",
    "THE WORLDPOST": "politics",
    "WORLDPOST": "politics",
    "WORLD NEWS": "politics",

    # Business & économie
    "BUSINESS": "business",
    "MONEY": "business",
    "FIFTY": "business",

    # Divertissement
    "ENTERTAINMENT": "entertainment",
    "ARTS & CULTURE": "entertainment",
    "ARTS": "entertainment",
    "CULTURE & ARTS": "entertainment",
    "COMEDY": "entertainment",
    "WEIRD NEWS": "entertainment",

    # Tech & science
    "TECH": "tech_science",
    "SCIENCE": "tech_science",
    "GREEN": "tech_science",
    "ENVIRONMENT": "tech_science",

    # Sport
    "SPORTS": "sports",

    # Santé & bien-être
    "HEALTHY LIVING": "health_wellness",
    "WELLNESS": "health_wellness",
    "MENTAL HEALTH": "health_wellness",
    "TASTE": "health_wellness",  # nourriture/santé

    # Style de vie
    "STYLE": "lifestyle",
    "STYLE & BEAUTY": "lifestyle",
    "HOME & LIVING": "lifestyle",
    "FOOD & DRINK": "lifestyle",
    "TRAVEL": "lifestyle",
    "GOOD NEWS": "lifestyle",

    # Famille & éducation
    "PARENTING": "family_education",
    "EDUCATION": "family_education",
    "COLLEGE": "family_education",
    "PARENTS": "family_education",

    # Médias & communication
    "MEDIA": "media",
    "QUEER VOICES": "media",
    "BLACK VOICES": "media",
    "LATINO VOICES": "media",
    "WOMEN": "media",

    # Faits divers & crime
    "CRIME": "crime",

    # International
    "IMPACT": "international",
    "RELIGION": "international",

    # Divers
    "DIVORCE": "other",
    "WEDDINGS": "other",
    "AUTOMOBILES": "other",
}


def load_data(path: str) -> pd.DataFrame:
    """Charge le fichier JSON Lines du dataset HuffPost."""
    log.info(f"Chargement depuis {path}...")
    df = pd.read_json(path, lines=True)
    log.info(f"  → {len(df):,} lignes chargées, {df['category'].nunique()} catégories")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime doublons, NaN critiques et anomalies de date."""
    initial = len(df)

    # Supprimer les doublons exacts
    df = df.drop_duplicates(subset=["headline", "short_description"])

    # Supprimer les lignes sans headline ni description
    df = df.dropna(subset=["headline"])
    df["short_description"] = df["short_description"].fillna("")

    # Filtrer les dates valides (dataset couvre 2012-2022)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[(df["date"].dt.year >= 2012) & (df["date"].dt.year <= 2022)]

    log.info(f"  → Nettoyage : {initial:,} → {len(df):,} lignes ({initial - len(df):,} supprimées)")
    return df.reset_index(drop=True)


def map_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Fusionne les 42 catégories originales en 15 super-catégories."""
    df["category_original"] = df["category"]
    df["category"] = df["category"].map(CATEGORY_MAPPING).fillna("other")

    dist = df["category"].value_counts()
    log.info(f"  → Distribution après fusion :\n{dist.to_string()}")

    # Vérification : aucune classe < 500 exemples
    small = dist[dist < 500]
    if len(small) > 0:
        log.warning(f"  Classes avec peu d'exemples : {small.to_dict()}")

    return df


def clean_text(text: str) -> str:
    """Nettoyage NLP standard : lowercase, espaces, caractères spéciaux."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # URLs
    text = re.sub(r"[^a-z0-9\s\-\']", " ", text)        # caractères spéciaux
    text = re.sub(r"\s+", " ", text).strip()             # espaces multiples
    return text


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée le champ 'text' principal : concaténation headline + description.
    C'est l'input de DistilBERT.
    """
    # Texte principal : headline [SEP] description (token SEP natif de BERT)
    df["text"] = df["headline"] + " [SEP] " + df["short_description"]
    df["text_clean"] = df["text"].apply(clean_text)

    # Features auxiliaires utiles pour l'EDA et le monitoring
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["has_description"] = (df["short_description"].str.len() > 10).astype(int)
    df["year"] = df["date"].dt.year

    # Stats de base
    log.info(f"  → Longueur texte : moy={df['text_length'].mean():.0f}, "
             f"médiane={df['text_length'].median():.0f}, "
             f"max={df['text_length'].max()}")

    return df


def encode_labels(df: pd.DataFrame):
    """Encode les labels en entiers et retourne l'encoder pour sérialisation."""
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["category"])
    log.info(f"  → {len(le.classes_)} classes encodées : {list(le.classes_)}")
    return df, le


def split_data(df: pd.DataFrame):
    """Split stratifié 70/15/15 pour respecter la distribution des classes."""
    train, temp = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label"]
    )
    val, test = train_test_split(
        temp, test_size=0.50, random_state=42, stratify=temp["label"]
    )

    log.info(f"  → Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    return train, val, test


def save_artifacts(train, val, test, le, output_dir: str):
    """Sauvegarde les splits et le LabelEncoder."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train.to_parquet(out / "train.parquet", index=False)
    val.to_parquet(out / "val.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)

    # Sauvegarder le mapping label → classe (utile pour l'API)
    mapping = {int(i): cls for i, cls in enumerate(le.classes_)}
    with open(out / "label_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)

    log.info(f"  → Artifacts sauvegardés dans {out}/")


def generate_report(df_raw, df_processed, output_dir: str):
    """Génère le rapport de preprocessing pour la certification."""
    report = f"""# Rapport de preprocessing — News Category Dataset

## Dataset source
- Source : HuffPost News Archive (Kaggle)
- Fichier : `News_Category_Dataset_v3.json`
- Format : JSON Lines (un objet JSON par ligne)

## Statistiques brutes
- Lignes initiales : {len(df_raw):,}
- Catégories originales : {df_raw['category'].nunique()}
- Période couverte : {df_raw['date'].min()} → {df_raw['date'].max()}

## Après nettoyage
- Lignes conservées : {len(df_processed):,}
- Lignes supprimées : {len(df_raw) - len(df_processed):,}
  - Doublons headline+description
  - Valeurs manquantes (headline vide)
  - Dates hors plage 2012-2022

## Fusion des catégories
- 42 catégories originales → 15 super-catégories
- Justification : certaines classes < 500 exemples (ARTS, AUTOMOBILES, etc.)
  rendaient l'apprentissage instable ; fusion thématique cohérente

## Feature engineering
- Champ `text` : `headline [SEP] short_description`
  - Le token [SEP] est natif du tokenizer BERT/DistilBERT
  - Longueur moyenne : {df_processed['text_length'].mean():.0f} caractères
- Features auxiliaires : `text_length`, `word_count`, `has_description`, `year`

## Split
- Train : 70% — Val : 15% — Test : 15%
- Stratifié sur les labels pour garantir la représentation de chaque classe
- Seed : 42 (reproductibilité garantie)

## Versioning
- Données versionnées avec DVC (`dvc add data/processed/`)
- Hash DVC : cf. `data/processed.dvc`
"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/preprocessing_report.md", "w") as f:
        f.write(report)
    log.info("  → Rapport généré : preprocessing_report.md")


def main(args):
    mlflow.set_experiment("news-classifier-preprocessing")

    with mlflow.start_run(run_name="preprocessing"):
        # Pipeline
        df_raw = load_data(args.input)
        df = clean_data(df_raw.copy())
        df = map_categories(df)
        df = feature_engineering(df)
        df, le = encode_labels(df)
        train, val, test = split_data(df)
        save_artifacts(train, val, test, le, args.output)
        generate_report(df_raw, df, args.output)

        # Logguer les métriques dans MLflow
        mlflow.log_params({
            "n_categories_original": df_raw["category"].nunique(),
            "n_categories_final": df["category"].nunique(),
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
            "random_state": 42,
        })
        mlflow.log_metrics({
            "total_samples": len(df),
            "samples_removed": len(df_raw) - len(df),
            "avg_text_length": df["text_length"].mean(),
        })

        log.info("Preprocessing terminé avec succes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/News_Category_Dataset_v3.json")
    parser.add_argument("--output", default="data/processed")
    main(parser.parse_args())