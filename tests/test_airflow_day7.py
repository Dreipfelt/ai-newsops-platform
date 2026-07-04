from pathlib import Path

from src.retrain_trigger import build_decision, extract_drift_score


def test_airflow_dag_file_exists_and_declares_pipeline_tasks():
    dag_file = Path("airflow/dags/retraining_dag.py")
    assert dag_file.exists()
    content = dag_file.read_text(encoding="utf-8")
    for task_id in [
        "validate_data",
        "check_drift",
        "retrain_candidate",
        "register_model",
    ]:
        assert task_id in content


def test_extract_drift_score_from_simple_report():
    assert extract_drift_score({"drift_score": 0.42}) == 0.42


def test_build_decision_retrains_when_threshold_exceeded():
    decision = build_decision({"drift_score": 0.42}, threshold=0.30)
    assert decision["should_retrain"] is True
    assert decision["reason"] == "drift_threshold_exceeded"


def test_build_decision_skips_when_no_drift():
    decision = build_decision({"drift_score": 0.10}, threshold=0.30)
    assert decision["should_retrain"] is False
    assert decision["reason"] == "no_significant_drift"
