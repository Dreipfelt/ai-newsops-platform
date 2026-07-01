"""
tests/test_api.py
Tests unitaires et d'intégration — AI NewsOps API
Coverage cible : ≥ 70%

Usage : pytest tests/ -v --cov=src --cov-report=term-missing
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from fastapi.testclient import TestClient


# ─────────────────────────────────────────────────────────────
# FIXTURES — Mock du modèle pour les tests (pas besoin de GPU)
# ─────────────────────────────────────────────────────────────

def make_mock_model_state():
    """Crée un état de modèle mocké réaliste."""
    return {
        "tokenizer":   MagicMock(),
        "model":       MagicMock(),
        "id2label":    {
            0: "arts_culture", 1: "business", 2: "crime",
            3: "entertainment", 4: "family_education", 5: "health_wellness",
            6: "international", 7: "lifestyle", 8: "media",
            9: "other", 10: "politics", 11: "sports", 12: "tech_science",
        },
        "loaded_at":        "2026-01-01T00:00:00",
        "n_requests":       0,
        "n_errors":         0,
        "total_latency_ms": 0.0,
        "metrics": {
            "test_f1_macro":  0.6805,
            "test_accuracy":  0.7359,
            "best_val_f1":    0.6691,
            "baseline_f1":    0.6515,
            "delta_f1":       0.029,
            "epochs_run":     4,
            "num_labels":     13,
            "class_names":    ["politics", "sports", "tech_science"],
            "history": {
                "train_loss": [1.29, 0.76, 0.53, 0.38],
                "val_loss":   [0.95, 0.90, 0.94, 0.99],
                "val_f1":     [0.63, 0.66, 0.67, 0.67],
                "val_acc":    [0.70, 0.72, 0.72, 0.72],
            }
        },
    }


def make_mock_logits(pred_class: int = 10, n_classes: int = 13):
    """Crée des logits mockés avec une prédiction dominante."""
    import torch
    logits = torch.zeros(1, n_classes)
    logits[0, pred_class] = 5.0  # score élevé pour la classe prédite
    return MagicMock(logits=logits)


@pytest.fixture
def client():
    """Client de test FastAPI avec modèle mocké."""
    import src.api.main as api_module

    mock_state = make_mock_model_state()

    # Mock run_inference directement pour isoler les tests API
    def mock_run_inference(headline, short_description=""):
        return {
            "category":      "politics",
            "confidence":    0.9973,
            "top3":          [
                {"category": "politics",      "score": 0.9973},
                {"category": "media",         "score": 0.0006},
                {"category": "entertainment", "score": 0.0004},
            ],
            "input_text":    f"{headline} [SEP] {short_description}",
            "latency_ms":    75.0,
            "model_version": "1.0.0",
        }

    with patch.object(api_module, "model_state", mock_state), \
         patch.object(api_module, "run_inference", side_effect=mock_run_inference):
        from fastapi.testclient import TestClient
        test_client = TestClient(api_module.app)
        yield test_client


# ─────────────────────────────────────────────────────────────
# TESTS — Root & Health
# ─────────────────────────────────────────────────────────────

class TestRoot:
    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_api_info(self, client):
        data = response = client.get("/").json()
        assert "name" in data
        assert "docs" in data
        assert "predict" in data

    def test_docs_accessible(self, client):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/predict" in schema["paths"]
        assert "/health" in schema["paths"]


class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_schema(self, client):
        data = client.get("/health").json()
        required_fields = [
            "status", "model_loaded", "device",
            "n_requests", "n_errors", "avg_latency_ms",
            "model_version", "api_version",
        ]
        for field in required_fields:
            assert field in data, f"Champ manquant : {field}"

    def test_health_status_healthy(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_version(self, client):
        data = client.get("/health").json()
        assert data["api_version"] == "1.0.0"
        assert data["model_version"] == "1.0.0"


# ─────────────────────────────────────────────────────────────
# TESTS — /predict
# ─────────────────────────────────────────────────────────────

class TestPredict:
    def test_predict_with_headline_only(self, client):
        response = client.post("/predict", json={
            "headline": "Senate passes new climate legislation"
        })
        assert response.status_code == 200

    def test_predict_with_headline_and_description(self, client):
        response = client.post("/predict", json={
            "headline": "Senate passes new climate legislation",
            "short_description": "Democrats and Republicans reach a compromise."
        })
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
        data = client.post("/predict", json={
            "headline": "Lakers win NBA championship"
        }).json()
        required = ["category", "confidence", "top3", "input_text", "latency_ms", "model_version"]
        for field in required:
            assert field in data, f"Champ manquant : {field}"

    def test_predict_confidence_between_0_and_1(self, client):
        data = client.post("/predict", json={
            "headline": "Apple launches new MacBook Pro"
        }).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_top3_has_3_items(self, client):
        data = client.post("/predict", json={
            "headline": "Scientists discover new exoplanet"
        }).json()
        assert len(data["top3"]) == 3

    def test_predict_top3_scores_sum_to_approx_1(self, client):
        data = client.post("/predict", json={
            "headline": "Federal Reserve raises interest rates"
        }).json()
        total = sum(item["score"] for item in data["top3"])
        assert total <= 1.01  # top3, pas 100%

    def test_predict_category_is_string(self, client):
        data = client.post("/predict", json={
            "headline": "New study reveals benefits of exercise"
        }).json()
        assert isinstance(data["category"], str)
        assert len(data["category"]) > 0

    def test_predict_empty_headline_returns_422(self, client):
        response = client.post("/predict", json={"headline": ""})
        assert response.status_code == 422

    def test_predict_missing_headline_returns_422(self, client):
        response = client.post("/predict", json={"short_description": "Only description"})
        assert response.status_code == 422

    def test_predict_headline_too_short_returns_422(self, client):
        response = client.post("/predict", json={"headline": "Hi"})
        assert response.status_code == 422

    def test_predict_input_text_contains_sep(self, client):
        data = client.post("/predict", json={
            "headline": "Breaking news today",
            "short_description": "Something happened"
        }).json()
        assert "[SEP]" in data["input_text"]


# ─────────────────────────────────────────────────────────────
# TESTS — /predict/batch
# ─────────────────────────────────────────────────────────────

class TestPredictBatch:
    def test_batch_single_article(self, client):
        response = client.post("/predict/batch", json={
            "articles": [{"headline": "Senate votes on budget"}]
        })
        assert response.status_code == 200

    def test_batch_multiple_articles(self, client):
        response = client.post("/predict/batch", json={
            "articles": [
                {"headline": "Senate votes on budget"},
                {"headline": "Lakers win championship"},
                {"headline": "Apple releases new iPhone"},
            ]
        })
        assert response.status_code == 200

    def test_batch_response_schema(self, client):
        data = client.post("/predict/batch", json={
            "articles": [
                {"headline": "Senate votes on budget"},
                {"headline": "Lakers win championship"},
            ]
        }).json()
        assert "n_articles" in data
        assert "n_success" in data
        assert "n_errors" in data
        assert "predictions" in data

    def test_batch_count_correct(self, client):
        data = client.post("/predict/batch", json={
            "articles": [
                {"headline": "Senate votes on budget"},
                {"headline": "Lakers win championship"},
                {"headline": "Apple releases new iPhone"},
            ]
        }).json()
        assert data["n_articles"] == 3
        assert data["n_success"] == 3
        assert data["n_errors"] == 0

    def test_batch_empty_list_returns_422(self, client):
        response = client.post("/predict/batch", json={"articles": []})
        assert response.status_code == 422

    def test_batch_too_many_articles_returns_422(self, client):
        articles = [{"headline": f"Article {i} about something important"} for i in range(33)]
        response = client.post("/predict/batch", json={"articles": articles})
        assert response.status_code == 422

    def test_batch_predictions_ordered(self, client):
        data = client.post("/predict/batch", json={
            "articles": [
                {"headline": "First article about politics"},
                {"headline": "Second article about sports"},
            ]
        }).json()
        assert data["predictions"][0]["index"] == 0
        assert data["predictions"][1]["index"] == 1


# ─────────────────────────────────────────────────────────────
# TESTS — /metrics
# ─────────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_schema(self, client):
        data = client.get("/metrics").json()
        assert "model_performance" in data
        assert "api_stats" in data

    def test_metrics_f1_present(self, client):
        data = client.get("/metrics").json()
        perf = data["model_performance"]
        assert "test_f1_macro" in perf
        assert "test_accuracy" in perf
        assert "baseline_f1" in perf
        assert "delta_f1" in perf

    def test_metrics_f1_is_float(self, client):
        data = client.get("/metrics").json()
        f1 = data["model_performance"]["test_f1_macro"]
        assert isinstance(f1, float)
        assert 0.0 <= f1 <= 1.0


# ─────────────────────────────────────────────────────────────
# TESTS — Preprocessing utilities
# ─────────────────────────────────────────────────────────────

class TestPreprocessing:
    """Tests des fonctions de preprocessing (indépendants du modèle)."""

    def test_category_mapping_exhaustive(self):
        """Vérifie que toutes les catégories connues sont mappées."""
        import sys
        sys.path.insert(0, ".")
        try:
            from src.data.preprocess import CATEGORY_MAPPING
            known_categories = {
                "POLITICS", "THE WORLDPOST", "WORLD NEWS", "WORLDPOST", "U.S. NEWS",
                "WELLNESS", "HEALTHY LIVING", "ENTERTAINMENT", "COMEDY",
                "TRAVEL", "STYLE & BEAUTY", "FOOD & DRINK", "HOME & LIVING",
                "PARENTING", "PARENTS", "COLLEGE", "EDUCATION",
                "QUEER VOICES", "BLACK VOICES", "WOMEN", "MEDIA", "LATINO VOICES",
                "BUSINESS", "MONEY", "SPORTS", "IMPACT", "RELIGION",
                "GREEN", "SCIENCE", "TECH", "ARTS", "CRIME", "WEDDINGS", "DIVORCE",
            }
            for cat in known_categories:
                assert cat in CATEGORY_MAPPING, f"Catégorie non mappée : {cat}"
        except ImportError:
            pytest.skip("preprocess.py non disponible dans cet environnement")

    def test_label_mapping_file_exists(self):
        """Vérifie que le label_mapping.json existe."""
        path = Path("data/processed/label_mapping.json")
        if not path.exists():
            pytest.skip("label_mapping.json non disponible (run preprocessing first)")
        with open(path) as f:
            mapping = json.load(f)
        assert len(mapping) == 13, f"Attendu 13 classes, trouvé {len(mapping)}"

    def test_processed_splits_exist(self):
        """Vérifie que les splits parquet existent."""
        for split in ["train", "val", "test"]:
            path = Path(f"data/processed/{split}.parquet")
            if not path.exists():
                pytest.skip(f"{split}.parquet non disponible")
            import pandas as pd
            df = pd.read_parquet(path)
            assert len(df) > 0
            assert "text" in df.columns
            assert "label" in df.columns
