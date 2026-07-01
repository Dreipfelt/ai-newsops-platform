from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

TEST_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("models/baseline_tfidf_logreg.joblib")
REPORTS_DIR = Path("reports")


def main() -> None:
    if not TEST_PATH.exists():
        raise FileNotFoundError("Test dataset not found. Run `make preprocess` first.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `make train` first.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(TEST_PATH)
    model = joblib.load(MODEL_PATH)

    X_test = test_df["text"]
    y_test = test_df["category"]

    predictions = model.predict(X_test)

    report = classification_report(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    (REPORTS_DIR / "classification_report.txt").write_text(report)
    pd.DataFrame(matrix).to_csv(REPORTS_DIR / "confusion_matrix.csv", index=False)

    print(report)
    print(f"Reports saved to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
