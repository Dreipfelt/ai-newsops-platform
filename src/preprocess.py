from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = Path("data/raw/News_Category_Dataset_v3.json")
PROCESSED_DIR = Path("data/processed")

TOP_N_CATEGORIES = 10
RANDOM_STATE = 42


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download it from Kaggle and place it in data/raw/."
        )

    return pd.read_json(path, lines=True)


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"headline", "short_description", "category", "date"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.copy()

    df["headline"] = df["headline"].fillna("").astype(str)
    df["short_description"] = df["short_description"].fillna("").astype(str)
    df["category"] = df["category"].fillna("").astype(str)

    df["text"] = (
        df["headline"].str.strip()
        + ". "
        + df["short_description"].str.strip()
    )

    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

    df = df[df["text"].str.len() > 20]
    df = df[df["category"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"])

    top_categories = df["category"].value_counts().head(TOP_N_CATEGORIES).index
    df = df[df["category"].isin(top_categories)]

    df = df[["text", "category", "date"]]

    return df.reset_index(drop=True)


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["category"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp_df["category"],
    )

    return train_df, val_df, test_df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_dataset(RAW_DATA_PATH)
    clean_df = preprocess_dataset(raw_df)

    train_df, val_df, test_df = split_dataset(clean_df)

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(f"Processed dataset saved to {PROCESSED_DIR}")
    print(f"Train: {len(train_df)} rows")
    print(f"Validation: {len(val_df)} rows")
    print(f"Test: {len(test_df)} rows")
    print(f"Categories: {sorted(clean_df['category'].unique())}")


if __name__ == "__main__":
    main()