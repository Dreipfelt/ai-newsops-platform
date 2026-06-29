from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline


TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/val.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "baseline_tfidf_logreg.joblib"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_PATH.exists() or not VAL_PATH.exists():
        raise FileNotFoundError(
            "Processed datasets not found. Run `make preprocess` first."
        )

    return pd.read_csv(TRAIN_PATH), pd.read_csv(VAL_PATH)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_data()

    X_train = train_df["text"]
    y_train = train_df["category"]

    X_val = val_df["text"]
    y_val = val_df["category"]

    model = build_pipeline()
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    metrics = {
        "accuracy": accuracy_score(y_val, predictions),
        "precision_macro": precision_score(y_val, predictions, average="macro"),
        "recall_macro": recall_score(y_val, predictions, average="macro"),
        "f1_macro": f1_score(y_val, predictions, average="macro"),
    }

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")
    print("Validation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()