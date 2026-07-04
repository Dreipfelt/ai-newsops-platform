"""Decide whether retraining should run based on drift report signals."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def extract_drift_score(report: dict) -> float:
    for key in ("drift_score", "dataset_drift_score", "share_drifted_columns"):
        value = report.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    if isinstance(report.get("metrics"), dict):
        for key in ("drift_score", "share_drifted_columns"):
            value = report["metrics"].get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return 0.0


def build_decision(report: dict, threshold: float, force: bool = False) -> dict:
    score = extract_drift_score(report)
    should_retrain = force or score >= threshold or bool(report.get("drift_detected"))
    reason = "forced" if force else "drift_threshold_exceeded" if should_retrain else "no_significant_drift"
    return {
        "should_retrain": should_retrain,
        "reason": reason,
        "drift_score": score,
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-report", default="monitoring/quick_drift_latest.json")
    parser.add_argument("--output", default="monitoring/retrain_decision.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    threshold = float(os.getenv("DRIFT_THRESHOLD", "0.30"))
    decision = build_decision(load_json(Path(args.drift_report)), threshold, args.force)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
