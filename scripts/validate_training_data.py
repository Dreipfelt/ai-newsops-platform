"""Validate processed training datasets before retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED_FILES = ("train.parquet", "val.parquet", "test.parquet", "label_mapping.json")
REQUIRED_COLUMNS = {"text", "label"}


def validate_data_dir(data_dir: Path) -> dict[str, object]:
    missing = [name for name in REQUIRED_FILES if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed files: {', '.join(missing)}")

    with (data_dir / "label_mapping.json").open("r", encoding="utf-8") as fh:
        label_mapping = json.load(fh)

    summary: dict[str, object] = {"n_labels": len(label_mapping), "splits": {}}
    valid_labels = {int(k) for k in label_mapping.keys()}

    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.parquet"
        df = pd.read_parquet(path)
        missing_columns = REQUIRED_COLUMNS - set(df.columns)
        if missing_columns:
            raise ValueError(f"{path} missing columns: {sorted(missing_columns)}")
        if df.empty:
            raise ValueError(f"{path} is empty")
        unknown_labels = set(df["label"].dropna().astype(int).unique()) - valid_labels
        if unknown_labels:
            raise ValueError(f"{path} has unknown labels: {sorted(unknown_labels)}")
        summary["splits"][split] = {"rows": int(len(df))}

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    args = parser.parse_args()
    validate_data_dir(Path(args.data_dir))


if __name__ == "__main__":
    main()
