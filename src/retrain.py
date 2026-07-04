"""Run a lightweight retraining entrypoint for scheduled orchestration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def read_decision(path: Path, force: bool = False) -> dict:
    if force:
        return {"should_retrain": True, "reason": "forced"}
    if not path.exists():
        return {"should_retrain": True, "reason": "missing_decision_file"}
    return json.loads(path.read_text(encoding="utf-8"))


def write_run_summary(output_path: Path, status: str, reason: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": status,
        "reason": reason,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", ""),
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision-file", default="monitoring/retrain_decision.json")
    parser.add_argument("--output", default="reports/retraining_run.json")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--train-command",
        default=os.getenv("RETRAIN_COMMAND", "python src/train_baseline.py"),
        help="Command used to train the candidate model.",
    )
    args = parser.parse_args()

    decision = read_decision(Path(args.decision_file), args.force)
    if not decision.get("should_retrain", False):
        write_run_summary(Path(args.output), "skipped", decision.get("reason", "not_required"))
        return

    result = subprocess.run(args.train_command, shell=True, check=False)
    if result.returncode != 0:
        write_run_summary(Path(args.output), "failed", "training_command_failed")
        sys.exit(result.returncode)

    write_run_summary(Path(args.output), "completed", decision.get("reason", "scheduled"))


if __name__ == "__main__":
    main()
