"""
tests/test_api.py
Tests unitaires et d'intégration — AI NewsOps API
Coverage cible : ≥ 70%

Corrigé pour matcher l'architecture réelle de src/api/main.py après fusion
avec app/main.py : variables globales séparées (model, tokenizer, id2label,
monitor) plutôt qu'un dict model_state unique.

Usage : pytest tests/ -v --cov=src --cov-report=term-missing
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# FIXTURES — Mock du modèle pour les tests (pas besoin de GPU)
# ─────────────────────────────────────────────────────────────

ID2LABEL = {
    0: "arts_culture", 1: "business", 2: "crime",
    3: "entertainment", 4: "family_education", 5: "health_wellness",
    6: "international", 7: "lifestyle", 8: "media",
    9: "other", 10: "politics", 11: "sports", 12: "tech_science",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def make_mock_torch_output(pred_class: int = 10, n_classes: int = 13):
    """Crée une sortie de modèle mockée avec des logits dominants sur pred_class."""
    import torch
    logits = torch.zeros(1, n_classes)
    logits[0, pred_class] = 8.0
    output = MagicMock()
    output.logits = logits
    return output


@pytest.fixture
def client():
    """
    Client de test FastAPI avec modèle, tokenizer et monitor mockés.

    L'architecture réelle de src/api/main.py utilise des variables globales
    au niveau module (model, tokenizer, id2label, label2id, monitor) plutôt
    qu'un dict unique — le patch cible donc directement ces globals.
    """
    import src.api.main as api_module

    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.to.return_value = mock_model
    mock_model.config.id2label = ID2LABEL

    def mock_call(**kwargs):
        return make_mock_torch_output(pred_class=LABEL2ID["politics"])

    mock_model.side_effect = mock_call
    mock_model.__call__ = mock_call

    mock_tokenizer = MagicMock()

    def mock_tokenize(text, **kwargs):
        import torch
        return {
            "input_ids": torch.zeros(1, 128, dtype=torch.long),
            "attention_mask": torch.ones(1, 128, dtype=torch.long),
        }

    mock_tokenizer.side_effect = mock_tokenize
    mock_tokenizer.__call__ = mock_tokenize

    mock_monitor = MagicMock()
    mock_monitor.reference_data = None
    mock_monitor.drift_logs = []
    mock_monitor.get_drift_summary.return_value = {
        "total": 0, "recent_drifts": 0, "last_check": None, "last_drift": None,
    }
    mock_monitor.get_recent_drifts.return_value = []

    with patch.object(api_module, "model", mock_model), \
         patch.object(api_module, "tokenizer", mock_tokenizer), \
         patch.object(api_module, "id2label", ID2LABEL), \
         patch.object(api_module, "label2id", LABEL2ID), \
         patch.object(api_module, "monitor", mock_monitor), \
         patch.object(api_module, "ensure_model_loaded", lambda: None), \
         patch.object(api_module, "ensure_monitor_loaded", lambda: None), \
         patch("torch.no_grad"), \
         patch("torch.softmax") as mock_softmax, \
         patch("torch.topk") as mock_topk:

        import torch as real_torch

        def softmax_side_effect(logits, dim=-1):
            probs = real_torch.zeros_like(logits)
            probs[0, LABEL2ID["politics"]] = 0.95
            probs[0, LABEL2ID["media"]] = 0.03
            probs[0, LABEL2ID["tech_science"]] = 0.02
            return probs

        mock_softmax.side_effect = softmax_side_effect

        def topk_side_effect(probs, k=3):
            top_probs = real_torch.tensor([[0.95, 0.03, 0.02]])
            top_indices = real_torch.tensor([[
                LABEL2ID["politics"], LABEL2ID["media"], LABEL2ID["tech_science"],
            ]])
            return top_probs, top_indices

        mock_topk.side_effect = topk_side_effect

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
        data = client.get("/").json()
        assert "name" in data
        assert "docs" in data

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
        for field in ["status", "model_loaded", "device", "tokenizer_loaded"]:
            assert field in data, f"Champ manquant : {field}"

    def test_health_status_healthy(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True


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
        for field in ["category", "confidence", "top3", "model_version"]:
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
            ]
        })
        assert response.status_code == 200

    def test_batch_response_schema(self, client):
        data = client.post("/predict/batch", json={
            "articles": [{"headline": "Senate votes on budget"}]
        }).json()
        assert "predictions" in data

    def test_batch_count_correct(self, client):
        data = client.post("/predict/batch", json={
            "articles": [
                {"headline": "Senate votes on budget"},
                {"headline": "Lakers win championship"},
            ]
        }).json()
        assert len(data["predictions"]) == 2

    def test_batch_empty_list_returns_422(self, client):
        response = client.post("/predict/batch", json={"articles": []})
        assert response.status_code == 422

    def test_batch_too_many_articles_returns_422(self, client):
        articles = [{"headline": f"Article {i} about something important"} for i in range(101)]
        response = client.post("/predict/batch", json={"articles": articles})
        assert response.status_code == 422


# ─────────────────────────────────────────────────────────────
# TESTS — /metrics
# ─────────────────────────────────────────────────────────────

class TestMetrics:
    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_is_prometheus_format(self, client):
        text = client.get("/metrics").text
        assert "http_requests_total" in text


# ─────────────────────────────────────────────────────────────
# TESTS — /monitoring/*
# ─────────────────────────────────────────────────────────────

class TestMonitoring:
    def test_monitoring_health_returns_200(self, client):
        response = client.get("/monitoring/health")
        assert response.status_code == 200

    def test_monitoring_drift_returns_200(self, client):
        response = client.get("/monitoring/drift")
        assert response.status_code == 200


# ─────────────────────────────────────────────────────────────
# TESTS — Preprocessing utilities (indépendants du modèle)
# ─────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_category_mapping_exhaustive(self):
        """Vérifie que toutes les catégories connues sont mappées."""
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
        """Vérifie que le label_mapping.json existe et a bien 13 classes."""
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
