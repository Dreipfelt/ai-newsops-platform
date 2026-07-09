"""
src/data/preprocess.py
Preprocessing pipeline — News Category Dataset
AI NewsOps Platform · AIA Bloc 4

Transforme le dataset brut HuffPost (42 catégories) en splits train/val/test
propres, avec fusion en 13 super-catégories, versionnés via DVC.

Ce script est la source de vérité canonique du mapping catégoriel. Toute
modification du CATEGORY_MAPPING doit être suivie d'une régénération complète
des splits (`python src/data/preprocess.py`) pour éviter toute désynchronisation
entre les fichiers parquet et le modèle entraîné — voir le post-mortem du bug
`arts_culture` documenté dans le README (section Known Limitations).

Usage :
  python src/data/preprocess.py
  python src/data/preprocess.py --input data/raw/News_Category_Dataset_v3.json --output data/processed
"""

import re
import json
import logging
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# MAPPING CANONIQUE — 42 catégories originales → 13 super-catégories
#
# Chaque catégorie source de HuffPost est explicitement mappée. Toute
# catégorie fréquente doit être mappée explicitement pour éviter une
# dilution silencieuse dans "other" (c'est exactement ce qui est arrivé
# à ARTS/ARTS & CULTURE dans une version antérieure de ce mapping).
# ─────────────────────────────────────────────────────────────
CATEGORY_MAPPING = {
    # Politique & actualités nationales
    "POLITICS": "politics",
    "THE WORLDPOST": "politics",
    "WORLD NEWS": "politics",
    "WORLDPOST": "politics",
    "U.S. NEWS": "politics",
    # Bien-être & santé
    "WELLNESS": "health_wellness",
    "HEALTHY LIVING": "health_wellness",
    "MENTAL HEALTH": "health_wellness",
    "TASTE": "health_wellness",
    # Divertissement
    "ENTERTAINMENT": "entertainment",
    "COMEDY": "entertainment",
    "WEIRD NEWS": "entertainment",
    # Style de vie
    "TRAVEL": "lifestyle",
    "STYLE & BEAUTY": "lifestyle",
    "FOOD & DRINK": "lifestyle",
    "HOME & LIVING": "lifestyle",
    "GOOD NEWS": "lifestyle",
    "STYLE": "lifestyle",
    # Famille & éducation
    "PARENTING": "family_education",
    "PARENTS": "family_education",
    "COLLEGE": "family_education",
    "EDUCATION": "family_education",
    # Médias & voix minoritaires
    "QUEER VOICES": "media",
    "BLACK VOICES": "media",
    "WOMEN": "media",
    "MEDIA": "media",
    "LATINO VOICES": "media",
    # Business & finance
    "BUSINESS": "business",
    "MONEY": "business",
    "FIFTY": "business",
    # Sport
    "SPORTS": "sports",
    # International & société
    "IMPACT": "international",
    "RELIGION": "international",
    # Technologie & science
    "GREEN": "tech_science",
    "SCIENCE": "tech_science",
    "TECH": "tech_science",
    "ENVIRONMENT": "tech_science",
    # Arts & culture — catégorie distincte, NE PAS fusionner dans entertainment
    "ARTS": "arts_culture",
    "ARTS & CULTURE": "arts_culture",
    "CULTURE & ARTS": "arts_culture",
    # Faits divers
    "CRIME": "crime",
    # Divers
    "WEDDINGS": "other",
    "DIVORCE": "other",
}

# 13 super-catégories attendues — utilisé pour valider le résultat final
EXPECTED_CATEGORIES = {
    "arts_culture",
    "business",
    "crime",
    "entertainment",
    "family_education",
    "health_wellness",
    "international",
    "lifestyle",
    "media",
    "other",
    "politics",
    "sports",
    "tech_science",
}


def load_data(path: str) -> pd.DataFrame:
    """Charge le dataset brut HuffPost (JSON Lines) et parse les dates."""
    log.info(f"Chargement : {path}")
    df = pd.read_json(path, lines=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    found = set(df["category"].unique())
    known = set(CATEGORY_MAPPING.keys())
    unmapped = found - known
    if unmapped:
        log.warning(
            f"  Catégories source non mappées explicitement (-> 'other') : {sorted(unmapped)}"
        )

    log.info(f"  {len(df):,} lignes - {df['category'].nunique()} categories source")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplique, retire les valeurs manquantes critiques, filtre la periode valide."""
    n0 = len(df)
    df = df.drop_duplicates(subset=["headline", "short_description"])
    df = df.dropna(subset=["headline"])
    df["short_description"] = df["short_description"].fillna("")
    df = df[(df["date"].dt.year >= 2012) & (df["date"].dt.year <= 2022)]
    df = df[df["headline"].str.len() >= 10]
    df = df.reset_index(drop=True)
    log.info(
        f"  Nettoyage : {n0:,} -> {len(df):,} lignes ({n0 - len(df):,} supprimees)"
    )
    return df


def map_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique CATEGORY_MAPPING et conserve la categorie source dans
    `category_original` -- indispensable pour tout audit futur.
    """
    df["category_original"] = df["category"]
    df["category"] = df["category"].map(CATEGORY_MAPPING).fillna("other")

    dist = df["category"].value_counts()
    ratio = dist.iloc[0] / dist.iloc[-1]
    log.info(
        f"  {df['category'].nunique()} super-categories - ratio desequilibre : {ratio:.1f}x"
    )

    actual = set(dist.index)
    missing = EXPECTED_CATEGORIES - actual
    unexpected = actual - EXPECTED_CATEGORIES
    if missing:
        log.error(f"  Categories manquantes apres mapping : {missing}")
        raise ValueError(f"Mapping incomplet -- categories manquantes : {missing}")
    if unexpected:
        log.error(f"  Categories inattendues apres mapping : {unexpected}")
        raise ValueError(f"Mapping incoherent -- categories inattendues : {unexpected}")

    print("\n  Distribution finale des super-categories :")
    for cat, cnt in dist.items():
        bar = "#" * int(cnt / 1500)
        print(f"    {cat:<20} {cnt:>6,}  {bar}")

    return df


def clean_text(text: str) -> str:
    """Nettoyage leger : minuscule, retrait URLs et caracteres speciaux."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z0-9\s\-']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit la feature texte principale : headline + [SEP] + description.
    Le token [SEP] est natif du vocabulaire BERT/DistilBERT.
    """
    df["text"] = df["headline"] + " [SEP] " + df["short_description"]
    df["text_clean"] = df["text"].apply(clean_text)
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["has_desc"] = (df["short_description"].str.len() > 10).astype(int)
    df["year"] = df["date"].dt.year

    log.info(
        f"  Texte - moyenne: {df['text_length'].mean():.0f} caracteres - "
        f"mediane: {df['text_length'].median():.0f} caracteres"
    )
    return df


def encode_labels(df: pd.DataFrame):
    """Encode les 13 categories en labels entiers 0-12, ordre alphabetique deterministe."""
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["category"])
    log.info(f"  {len(le.classes_)} classes : {list(le.classes_)}")
    return df, le


def split_data(df: pd.DataFrame):
    """Split stratifie 70/15/15, seed fixe pour reproductibilite totale."""
    train, temp = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label"]
    )
    val, test = train_test_split(
        temp, test_size=0.50, random_state=42, stratify=temp["label"]
    )
    log.info(f"  Split - train:{len(train):,} - val:{len(val):,} - test:{len(test):,}")
    return train, val, test


def save_artifacts(train, val, test, le, output_dir: str) -> dict:
    """Sauvegarde les 3 splits en parquet et le mapping label<->categorie en JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train.to_parquet(out / "train.parquet", index=False)
    val.to_parquet(out / "val.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)

    mapping = {int(i): cls for i, cls in enumerate(le.classes_)}
    with open(out / "label_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2)

    log.info(f"  Artifacts -> {out}/")
    return mapping


def validate_consistency(output_dir: str, mapping: dict):
    """
    Controle de coherence final : verifie que la colonne `category` (texte)
    correspond bien a `label` (entier) via le mapping fraichement genere.
    Ce controle est celui qui aurait detecte immediatement le bug
    arts_culture/entertainment -- il tourne desormais a chaque preprocessing.
    """
    id2label = mapping
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(Path(output_dir) / f"{split}.parquet")
        df["label_name"] = df["label"].map(id2label)
        coherence = (df["category"] == df["label_name"]).mean()
        if coherence < 1.0:
            raise ValueError(
                f"Incoherence detectee dans {split}.parquet : "
                f"category vs label ne correspondent qu'a {coherence:.1%}"
            )
        log.info(f"  Coherence category<->label verifiee sur {split} : {coherence:.1%}")


def generate_report(df_raw, df_processed, output_dir: str):
    """Genere un rapport markdown resumant les decisions de preprocessing."""
    dist = df_processed["category"].value_counts()
    report = f"""# Rapport de preprocessing -- News Category Dataset

## Source
- Dataset : HuffPost News Archive (Kaggle, `rmisra/news-category-dataset`)
- Fichier : `News_Category_Dataset_v3.json`
- Format : JSON Lines

## Statistiques
- Lignes brutes : {len(df_raw):,}
- Lignes apres nettoyage : {len(df_processed):,}
- Categories source : {df_raw['category'].nunique()}
- Super-categories finales : {df_processed['category'].nunique()}
- Ratio desequilibre (max/min) : {dist.iloc[0] / dist.iloc[-1]:.1f}x

## Decisions cles
1. Fusion 42 -> 13 categories -- reduit le ratio de desequilibre de ~170x a ~13x
2. `arts_culture` mappee separement -- jamais fusionnee dans `entertainment`
3. Feature texte : `headline [SEP] short_description`, token natif BERT
4. Split 70/15/15 stratifie, seed=42, reproductible via `dvc repro`

## Distribution finale
```
{dist.to_string()}
```
"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/preprocessing_report.md", "w") as f:
        f.write(report)
    log.info("  Rapport genere : preprocessing_report.md")


def main(args):
    mlflow.set_experiment("news-classifier-preprocessing")

    with mlflow.start_run(run_name="preprocess"):
        df_raw = load_data(args.input)
        df = clean_data(df_raw.copy())
        df = map_categories(df)
        df = feature_engineering(df)
        df, le = encode_labels(df)
        train, val, test = split_data(df)
        mapping = save_artifacts(train, val, test, le, args.output)
        validate_consistency(args.output, mapping)
        generate_report(df_raw, df, args.output)

        mlflow.log_params(
            {
                "n_categories_original": df_raw["category"].nunique(),
                "n_categories_final": df["category"].nunique(),
                "train_size": len(train),
                "val_size": len(val),
                "test_size": len(test),
                "random_state": 42,
            }
        )
        mlflow.log_metrics(
            {
                "total_samples": len(df),
                "samples_removed": len(df_raw) - len(df),
                "avg_text_length": round(df["text_length"].mean(), 1),
                "imbalance_ratio": round(
                    df["category"].value_counts().iloc[0]
                    / df["category"].value_counts().iloc[-1],
                    1,
                ),
            }
        )
        mlflow.log_dict(mapping, "label_mapping.json")

        log.info("Preprocessing termine -- donnees coherentes et validees.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocessing -- News Category Dataset"
    )
    parser.add_argument(
        "--input",
        default="data/raw/News_Category_Dataset_v3.json",
        help="Chemin vers le fichier JSON brut HuffPost",
    )
    parser.add_argument(
        "--output",
        default="data/processed",
        help="Dossier de sortie des artifacts (parquet + label_mapping.json)",
    )
    main(parser.parse_args())
