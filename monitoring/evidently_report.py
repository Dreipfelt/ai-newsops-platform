from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report


REFERENCE_PATH = Path("data/processed/train.parquet")
CURRENT_PATH = Path("data/processed/test.parquet")
REPORT_PATH = Path("monitoring/reports/data_drift_report.html")


def main() -> None:
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"Reference dataset not found: {REFERENCE_PATH}")

    if not CURRENT_PATH.exists():
        raise FileNotFoundError(f"Current dataset not found: {CURRENT_PATH}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    reference_df = pd.read_parquet(REFERENCE_PATH)
    current_df = pd.read_parquet(CURRENT_PATH)

    print(reference_df.columns.tolist())
    print(current_df.columns.tolist())
    
    columns = ["category", "date"]
    reference_df = reference_df[columns].copy()
    current_df = current_df[columns].copy()

    report = Report(
        metrics=[
            DataQualityPreset(),
            DataDriftPreset(),
        ]
    )

    report.run(reference_data=reference_df, current_data=current_df)
    report.save_html(str(REPORT_PATH))

    print(f"Evidently report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
