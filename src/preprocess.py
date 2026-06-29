"""
src/data/preprocess.py
Preprocessing pipeline — News Category Dataset
AI NewsOps Platform · AIA Bloc 4
"""

import re
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow
import kagglehub

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# MAPPING 42 → 15 super-catégories
# ─────────────────────────────────────────────────────────────
CATEGORY_MAPPING = {
    "POLITICS": "politics",           "GOVERNMENT & POLITICS": "politics",
    "THE WORLDPOST": "politics",      "WORLDPOST": "politics",
    "WORLD NEWS": "politics",
    "BUSINESS": "business",           "MONEY": "business",
    "FIFTY": "business",
    "ENTERTAINMENT": "entertainment", "ARTS & CULTURE": "entertainment",
    "ARTS": "entertainment",          "CULTURE & ARTS": "entertainment",
    "COMEDY": "entertainment",        "WEIRD NEWS": "entertainment",
    "TECH": "tech_science",           "SCIENCE": "tech_science",
    "GREEN": "tech_science",          "ENVIRONMENT": "tech_science",
    "SPORTS": "sports",
    "HEALTHY LIVING": "health_wellness", "WELLNESS": "health_wellness",
    "MENTAL HEALTH": "health_wellness",  "TASTE": "health_wellness",
    "STYLE": "lifestyle",             "STYLE & BEAUTY": "lifestyle",
    "HOME & LIVING": "lifestyle",     "FOOD & DRINK": "lifestyle",
    "TRAVEL": "lifestyle",            "GOOD NEWS": "lifestyle",
    "PARENTING": "family_education",  "EDUCATION": "family_education",
    "COLLEGE": "family_education",    "PARENTS": "family_education",
    "MEDIA": "media",                 "QUEER VOICES": "media",
    "BLACK VOICES": "media",          "LATINO VOICES": "media",
    "WOMEN": "media",
    "CRIME": "crime",
    "IMPACT": "international",        "RELIGION": "international",
    "DIVORCE": "other",               "WEDDINGS": "other",
    "AUTOMOBILES": "other",
}


# ─────────────────────────────────────────────────────────────
# ÉTAPES DU PIPELINE
# ─────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    log.info(f"Chargement : {path}")
    df = pd.read_json(path, lines=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    log.info(f"  {len(df):,} lignes · {df['category'].nunique()} catégories")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.drop_duplicates(subset=["headline", "short_description"])
    df = df.dropna(subset=["headline"])
    df["short_description"] = df["short_description"].fillna("")
    df = df[(df["date"].dt.year >= 2012) & (df["date"].dt.year <= 2022)]
    df = df[df["headline"].str.len() >= 10]
    df = df.reset_index(drop=True)
    log.info(f"  Nettoyage : {n0:,} → {len(df):,} lignes ({n0 - len(df):,} supprimées)")
    return df


def map_categories(df: pd.DataFrame) -> pd.DataFrame:
    df["category_original"] = df["category"]
    df["category"] = df["category"].map(CATEGORY_MAPPING).fillna("other")
    dist = df["category"].value_counts()
    log.info(f"  Catégories : {df['category'].nunique()} super-catégories")
    log.info(f"  Min: {dist.min():,} ({dist.idxmin()}) · Max: {dist.max():,} ({dist.idxmax()})")
    return df


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s\-\']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Feature principale pour DistilBERT
    df["text"] = df["headline"] + " [SEP] " + df["short_description"]
    df["text_clean"] = df["text"].apply(clean_text)

    # Features auxiliaires (monitoring, EDA)
    df["text_length"]   = df["text"].str.len()
    df["word_count"]    = df["text"].str.split().str.len()
    df["has_desc"]      = (df["short_description"].str.len() > 10).astype(int)
    df["year"]          = df["date"].dt.year

    log.info(f"  Texte — moy: {df['text_length'].mean():.0f} chars · "
             f"médiane: {df['text_length'].median():.0f} chars")
    return df


def encode_labels(df: pd.DataFrame):
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["category"])
    log.info(f"  Labels : {len(le.classes_)} classes → {list(le.classes_)}")
    return df, le


def split_data(df: pd.DataFrame):
    train, temp = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label"])
    val, test   = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp["label"])
    log.info(f"  Split — train: {len(train):,} · val: {len(val):,} · test: {len(test):,}")
    return train, val, test


def save_artifacts(train, val, test, le, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train.to_parquet(out / "train.parquet", index=False)
    val.to_parquet(out / "val.parquet",     index=False)
    test.to_parquet(out / "test.parquet",   index=False)

    mapping = {int(i): cls for i, cls in enumerate(le.classes_)}
    with open(out / "label_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)

    log.info(f"  Artifacts → {out}/")
    return mapping


def main(args):
    mlflow.set_experiment("news-classifier-preprocessing")

    with mlflow.start_run(run_name="preprocess-v1"):

        df_raw = load_data(args.input)
        df     = clean_data(df_raw.copy())
        df     = map_categories(df)
        df     = feature_engineering(df)
        df, le = encode_labels(df)
        train, val, test = split_data(df)
        mapping = save_artifacts(train, val, test, le, args.output)

        # Log MLflow
        mlflow.log_params({
            "n_categories_original": df_raw["category"].nunique(),
            "n_categories_final":    df["category"].nunique(),
            "train_size":            len(train),
            "val_size":              len(val),
            "test_size":             len(test),
            "random_state":          42,
            "max_year":              2022,
            "min_year":              2012,
        })
        mlflow.log_metrics({
            "total_samples":    len(df),
            "samples_removed":  len(df_raw) - len(df),
            "avg_text_length":  round(df["text_length"].mean(), 1),
            "imbalance_ratio":  round(
                df["category"].value_counts().iloc[0] /
                df["category"].value_counts().iloc[-1], 1
            ),
        })
        mlflow.log_dict(mapping, "label_mapping.json")

        log.info("Preprocessing terminé.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing pipeline")
    parser.add_argument("--input",  default="data/raw/News_Category_Dataset_v3.json",
                        help="Chemin vers le fichier JSON brut")
    parser.add_argument("--output", default="data/processed",
                        help="Dossier de sortie des artifacts")
    main(parser.parse_args())
