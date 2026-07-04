from pathlib import Path
import json

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


REFERENCE_PATH = Path("data/processed/train.parquet")
CURRENT_PATH = Path("data/processed/test.parquet")
OUTPUT_PATH = Path("monitoring/quick_drift_latest.json")

COLUMNS = [
    "category",
    "date",
    "text_length",
    "word_count",
    "has_desc",
    "year",
]


def main() -> None:
    reference_df = pd.read_parquet(REFERENCE_PATH)[COLUMNS].copy()
    current_df = pd.read_parquet(CURRENT_PATH)[COLUMNS].copy()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    result = report.as_dict()

    dataset_drift = result["metrics"][0]["result"]

    output = {
        "drift_detected": dataset_drift.get("dataset_drift", False),
        "drift_share": dataset_drift.get("share_of_drifted_columns", 0.0),
        "number_of_columns": dataset_drift.get("number_of_columns"),
        "number_of_drifted_columns": dataset_drift.get("number_of_drifted_columns"),
        "status": "drift" if dataset_drift.get("dataset_drift", False) else "ok",
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
