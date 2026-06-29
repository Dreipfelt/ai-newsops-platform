from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline


TRAIN_PATH = Path("data/processed/train.csv")
VAL_PATH = Path("data/processed/val.csv")
MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODEL_PATH = MODEL_DIR / "baseline_tfidf_logreg.joblib"
EXPERIMENT_NAME = "news-classification-baseline"


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


def compute_metrics(y_true: pd.Series, predictions: list[str]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision_macro": precision_score(y_true, predictions, average="macro"),
        "recall_macro": recall_score(y_true, predictions, average="macro"),
        "f1_macro": f1_score(y_true, predictions, average="macro"),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_data()

    X_train = train_df["text"]
    y_train = train_df["category"]

    X_val = val_df["text"]
    y_val = val_df["category"]

    model = build_pipeline()

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="tfidf_logreg_baseline"):
        mlflow.log_param("model_type", "TF-IDF + LogisticRegression")
        mlflow.log_param("max_features", 50000)
        mlflow.log_param("ngram_range", "(1, 2)")
        mlflow.log_param("min_df", 2)
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("validation_rows", len(val_df))
        mlflow.log_param("num_classes", y_train.nunique())

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        metrics = compute_metrics(y_val, predictions)

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        joblib.dump(model, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model_artifact")

        mlflow.sklearn.log_model(
            sk_model=model,
            name="baseline_tfidf_logreg_model",
        )

        labels_path = REPORTS_DIR / "baseline_labels.txt"
        labels_path.write_text("\n".join(sorted(y_train.unique())))
        mlflow.log_artifact(str(labels_path), artifact_path="metadata")

        print(f"Model saved to {MODEL_PATH}")
        print("Validation metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        print("MLflow run logged successfully.")


if __name__ == "__main__":
    main()