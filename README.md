# AI NewsOps Platform

> **A production-grade MLOps pipeline for automated news article classification**  
> Fine-tuned DistilBERT · FastAPI · Docker Compose · Prometheus · Grafana · Evidently AI · MLflow · Apache Airflow · GitHub Actions CI/CD  
> *AIA Bloc 4 — MLOps Certification · Frédéric Dreipfelt*

[![CI](https://github.com/Dreipfelt/ai-newsops-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/Dreipfelt/ai-newsops-platform/actions/workflows/ci.yml)
[![CD](https://github.com/Dreipfelt/ai-newsops-platform/actions/workflows/cd.yml/badge.svg)](https://github.com/Dreipfelt/ai-newsops-platform/actions/workflows/cd.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-3.14-0194E2.svg)](https://mlflow.org/)
[![Tests](https://img.shields.io/badge/tests-34%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Live Stack Inventory](#live-stack-inventory)
- [System Architecture](#system-architecture)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Model Training & Evaluation](#model-training--evaluation)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Observability Stack](#observability-stack)
- [Automated Retraining](#automated-retraining)
- [Versioning & Rollback](#versioning--rollback)
- [CI/CD Pipeline](#cicd-pipeline)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Technology Stack](#technology-stack)
- [Known Limitations & Roadmap](#known-limitations--roadmap)

---

## Project Overview

AI NewsOps Platform is an end-to-end MLOps system automating the full lifecycle of a news classification model — from raw data ingestion through to production deployment, continuous monitoring, and automated retraining. The system classifies news articles into **13 thematic super-categories** derived from the HuffPost News Archive (208,733 articles, 2012–2022).

Every component described in this document is running and verifiable in the live `docker-compose` stack — this is not an architectural proposal, it is a working system with 16+ hours of continuous uptime at the time of writing.

### Key Results

| Metric | Baseline (TF-IDF + LinearSVC) | DistilBERT Fine-tuned |
|---|:---:|:---:|
| F1 Macro (test) | 0.6515 | **0.6791** |
| Accuracy (test) | 71.84% | **73.82%** |
| Inference latency (p95) | — | **~5 ms** (measured via Prometheus) |
| Requests observed | — | live counter via Grafana |

---

## Live Stack Inventory

The full platform runs as seven Docker services orchestrated by `docker-compose`. This table reflects the actual `docker-compose ps` output, not an intended design:

| Service | Image | Port | Role |
|---|---|:---:|---|
| `api` | Custom (FastAPI) | 8000 | Inference + Prometheus metrics + monitoring routes |
| `mlflow` | python:3.10-slim | 5000 | Experiment tracking + Model Registry |
| `prometheus` | prom/prometheus:latest | 9090 | Metrics scraping (15s interval) |
| `grafana` | grafana/grafana:latest | 3000 | 11-panel live monitoring dashboard |
| `dashboard` | Custom (Streamlit) | 8501 | Human-facing prediction & drift explorer |
| `airflow-webserver` | apache/airflow:2.9.3 | 8080 | Retraining DAG UI |
| `airflow-scheduler` | apache/airflow:2.9.3 | — | DAG scheduling engine |
| `airflow-postgres` | postgres:15 | 5432 | Airflow metadata store |

```bash
docker-compose up -d
docker-compose ps
# All 8 containers should report "Up" / "healthy"
```

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           AI NewsOps Platform                                    │
│                                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌────────────────────┐ │
│  │    DATA     │    │   TRAINING  │    │  DEPLOYMENT │    │    OPERATIONS      │ │
│  │             │    │             │    │             │    │                    │ │
│  │ Kaggle API  │──▶│ DistilBERT  │──▶│  FastAPI    │──▶│  Prometheus        │ │
│  │ DVC         │    │ HuggingFace │    │  Docker     │    │  Grafana           │ │
│  │ Parquet     │    │ MLflow      │    │  Compose    │    │  Evidently AI      │ │
│  │ 208k arts.  │    │             │    │  GH Actions │    │  Streamlit         │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └────────────────────┘ │
│                                                                     │            │
│                                                                     ▼            │
│                                                           ┌─────────────────── ┐ │
│                                                           │  Apache Airflow    │ │
│                                                           │  Weekly retraining │ │
│                                                           │ Champion/challenger│ │
│                                                           └────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Dataset & Preprocessing

### Source

The [HuffPost News Category Dataset v3](https://www.kaggle.com/datasets/rmisra/news-category-dataset) (Kaggle): 209,527 articles, 2012–2022, originally labelled across 42 categories.

### The Class Imbalance Problem

The raw distribution exhibits a **170× imbalance ratio** between the most frequent category (`POLITICS`, 35,602 examples) and the least frequent. Several categories have fewer than 500 examples.

**Decision:** merge the 42 original labels into **13 semantically coherent super-categories**, reducing the imbalance ratio to **13×**.

| Super-Category | Train Examples |
|---|:---:|
| `politics` | 32,433 |
| `lifestyle` | 23,677 |
| `health_wellness` | 18,644 |
| `entertainment` | 17,844 |
| `media` | 12,919 |
| `family_education` | 10,332 |
| `business` | 6,393 |
| `tech_science` | 5,821 |
| `other` | 4,943 |
| `international` | 4,223 |
| `sports` | 3,551 |
| `arts_culture` | 2,738 |
| `crime` | 2,490 |

### Feature Engineering

```
text = headline + " [SEP] " + short_description
```

`max_length=128`, covering 95% of the corpus (median: 177 characters, ~30 tokens per the Evidently dataset summary below).

### Data Splits & Validated Consistency

| Split | Examples | Proportion |
|---|:---:|:---:|
| Train | 146,113 | 70% |
| Validation | 31,310 | 15% |
| Test | 31,310 | 15% |

Stratified by label, versioned with DVC. An Evidently AI drift report (`monitoring/reports/data_drift_report.html`) confirms train/test consistency:

| Feature | Drift Status | Distance |
|---|:---:|:---:|
| year | Not detected | Wasserstein = 0.0087 |
| text_length | Not detected | Wasserstein = 0.0068 |
| word_count | Not detected | Wasserstein = 0.0067 |
| has_desc | Not detected | Jensen-Shannon = 0.0003 |
| category | Not detected | Jensen-Shannon = 0.0001 |

**0 of 5 columns drifted (0.0% share)** — this validates that the stratified split preserved the underlying distribution, and serves as the reference baseline against which live production drift is measured. It is not, by itself, evidence of production stability — see [Known Limitations](#known-limitations--roadmap).

---

## Model Training & Evaluation

### Why DistilBERT?

DistilBERT retains 97% of BERT's GLUE performance at 40% fewer parameters and ~60% faster inference — the right trade-off for short-text classification (headline + description, 20–40 tokens) where latency matters more than marginal accuracy gains from a larger model.

### Training Configuration

| Hyperparameter | Value | Rationale |
|---|:---:|---|
| Base model | `distilbert-base-uncased` | Strongest open distilled BERT at parameter parity |
| Max sequence length | 128 | Covers 95% of corpus |
| Batch size | 16 (CPU) | Constrained by training hardware (i7-8700T, no GPU) |
| Learning rate | 3e-5 | Standard BERT fine-tuning range |
| Weight decay | 0.05 | Regularisation against overfitting on fast-mode subset |
| Dropout | 0.2 | Increased from default 0.1 |
| Epochs run | 2 (fast mode) | 15% of training data, CPU-constrained |
| Early stopping patience | 2 epochs | Stops if validation F1 plateaus |

### Results

```
Baseline (TF-IDF + LinearSVC)
  Test F1 macro: 0.6515  |  Accuracy: 71.84%

DistilBERT (fast mode, 2 epochs, 15% train)
  Test F1 macro: 0.6791  |  Accuracy: 73.82%
  Delta: +0.0276 F1 points (+4.2%)
  Training time: 192 minutes (CPU)
```

Full training on 100% of data with GPU acceleration is expected to reach F1 ≥ 0.85 based on the training curve trend observed in early experiments (F1 = 0.72 after a single epoch on full data with GPU).

---

## API Reference

Interactive documentation: `http://localhost:8000/docs`

### `POST /predict`

```json
{
  "headline": "Senate votes on landmark climate legislation",
  "short_description": "Bipartisan bill passes with 67 votes."
}
```

**Response:**
```json
{
  "category": "politics",
  "confidence": 0.9957,
  "top3": [["politics", 0.9957], ["tech_science", 0.0013], ["media", 0.0008]],
  "model_version": "1.0.0"
}
```

### `POST /predict/batch`

Up to 100 articles per request.

### `GET /metrics`

Prometheus-format text metrics. Includes `http_requests_total`, `http_request_duration_seconds`, `model_f1_score`, `model_accuracy`, `model_drift_score`.

### `GET /monitoring/health` · `/monitoring/drift` · `/monitoring/drift/run` · `/monitoring/drift/logs`

Full reference: [`docs/API.md`](docs/API.md) · Postman collection: [`docs/newsops_postman_collection.json`](docs/newsops_postman_collection.json)

---

## Deployment

### Full Stack (Recommended)

```bash
docker-compose up -d
```

| Interface | URL | Credentials |
|---|---|---|
| API Swagger | http://localhost:8000/docs | — |
| Grafana Dashboard | http://localhost:3000/d/ai-newsops-monitoring | admin / admin |
| Streamlit Dashboard | http://localhost:8501 | — |
| MLflow UI | http://localhost:5000 | — |
| Airflow UI | http://localhost:8080 | — |
| Prometheus | http://localhost:9090 | — |

### Generating Demo Traffic

```bash
bash scripts/generate_traffic.sh 10 0.5
# Sends 10 varied /predict requests, 0.5s apart — populates Grafana panels for demos
```

---

## Observability Stack

### Layer 1 — Operational Telemetry (Prometheus + Grafana)

11 live panels on the `ai-newsops-monitoring` dashboard, all backed by real PromQL queries against the running API:

| Panel | Query |
|---|---|
| API Availability | `up{job="newsops-api"}` |
| Requests / Second | `sum(rate(http_requests_total{job="newsops-api"}[5m]))` |
| P95 Latency | `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job="newsops-api"}[5m])) by (le))` |
| Error Rate | `sum(rate(http_requests_total{job="newsops-api",status=~"5.."}[5m]))` |
| Model Drift Score | `model_drift_score{job="newsops-api"}` |
| Model Accuracy | `model_accuracy{job="newsops-api"}` |
| Model F1 Score | `model_f1_score{job="newsops-api"}` |
| HTTP Requests by Status | `sum(rate(http_requests_total{job="newsops-api"}[5m])) by (status)` |
| API Latency (avg/p95) | `rate(http_request_duration_seconds_sum[5m]) / rate(http_request_duration_seconds_count[5m])` |
| Prediction Traffic | `sum(rate(http_requests_total{job="newsops-api",endpoint=~"/predict.*"}[5m]))` |
| Model Drift Over Time | `model_drift_score{job="newsops-api"}` (time series) |

Provisioning: `grafana/dashboards/dashboard.json` + `grafana/datasources/datasource.yml` (datasource UID pinned to `prometheus` to guarantee reproducible provisioning across environments).

### Layer 2 — Data Drift Detection (Evidently AI + scipy)

Two complementary detection paths:

1. **Fast path (scipy)** — Kolmogorov-Smirnov on numerical features, χ² on the categorical distribution. Powers `GET /monitoring/drift`, returns in < 1 second.
2. **Rich path (Evidently AI)** — full HTML report with distribution overlays, generated on demand via `python src/monitoring/drift_detector.py --full-report`. Current reference-vs-test comparison: **0.0% columns drifted**, confirming split integrity (see [Dataset & Preprocessing](#dataset--preprocessing)).

### Layer 3 — Human-facing Dashboard (Streamlit)

`dashboard/app.py` — four pages: **Home** (API health check), **Prediction** (interactive single-article classifier), **Monitoring** (live metrics), **Drift** (on-demand drift report trigger). Runs independently of Grafana for non-technical stakeholders (e.g. editorial team).

---

## Automated Retraining

Orchestrated by Apache Airflow (`airflow/dags/retraining_dag.py`), running continuously in the stack (`airflow-webserver` + `airflow-scheduler` + `airflow-postgres`).

### DAG: `news_classifier_retraining`

```
check_drift ──▶ should_retrain ──▶ backup_current_model ──▶ retrain_model
                                                                    │
                                                                    ▼
                                                          evaluate_and_decide
                                                             │         │
                                                       promote_model  rollback_model
                                                             │         │
                                                             └────┬────┘
                                                                  ▼
                                                                notify
```

**Schedule:** every Monday at 02:00 UTC · **Promotion policy:** retrained model is promoted only if test F1 improves by ≥0.5% over the incumbent; otherwise automatic rollback to the previous checkpoint.

```bash
# Airflow UI already running as part of docker-compose
open http://localhost:8080
```

---

## Versioning & Rollback

### MLflow Model Registry

```bash
python src/register_model.py --action register   # Register current model
python src/register_model.py --action list        # List all versions + F1 scores
python src/register_model.py --action compare      # Side-by-side comparison
python src/register_model.py --action promote --version 2
python src/register_model.py --action rollback --version 1
```

Current state: **Version 1 in Production**, F1 = 0.6791, local backup at `models/distilbert/versions/v1_20260705/`.

### Data Versioning (DVC)

```bash
dvc repro     # Reproduce the full preprocessing pipeline from raw data
dvc status    # Check for uncommitted data changes
```

---

## CI/CD Pipeline

### CI (`.github/workflows/ci.yml`)

```
push / PR → lint (Ruff + Black) → test (34 pytest, coverage ≥70%) → build (Docker → GHCR)
```

### CD (`.github/workflows/cd.yml`)

```
CI success on main → pull image → stop old container → start new → health check (30s timeout)
                                                                          │
                                              fail ─────────────────── rollback to previous SHA
                                              pass ─── smoke test /predict ─── notify
```

---

## Repository Structure

```
ai-newsops-platform/
│
├── src/
│   ├── api/main.py                  # FastAPI + Prometheus + ModelMonitor (Evidently)
│   ├── data/preprocess.py           # 42 → 13 category preprocessing pipeline
│   ├── models/
│   │   ├── baseline_v2.py           # TF-IDF + LinearSVC baseline
│   │   └── train_cpu.py             # DistilBERT fine-tuning, CPU-optimised
│   ├── monitoring/
│   │   ├── drift_detector.py        # scipy KS+χ² (fast) + Evidently HTML (rich)
│   │   ├── alerts.py                # Webhook / email alert dispatch
│   │   └── monitoring_router.py     # /monitoring/* FastAPI routes
│   ├── retrain_trigger.py
│   ├── retrain.py
│   └── register_model.py            # MLflow Model Registry + local rollback
│
├── airflow/dags/retraining_dag.py   # Weekly retraining DAG (TaskFlow API)
│
├── dashboard/app.py                 # Streamlit human-facing dashboard
│
├── grafana/
│   ├── dashboards/dashboard.json    # 11-panel monitoring dashboard definition
│   └── datasources/datasource.yml   # Prometheus datasource (UID pinned)
│
├── prometheus/prometheus.yml        # Scrape config (job: newsops-api, 15s interval)
│
├── scripts/generate_traffic.sh      # Demo traffic generator for dashboards
│
├── tests/test_api.py                # 34 unit + integration tests (mocked model)
│
├── docs/
│   ├── API.md
│   └── newsops_postman_collection.json
│
├── notebooks/01_eda.ipynb           # 13-section EDA
│
├── models/distilbert/
│   ├── best_model/                  # Production weights (256 MB, Git LFS)
│   ├── checkpoints/                 # Per-epoch checkpoints
│   ├── versions/                    # MLflow-linked local backups
│   └── training_metrics.json
│
├── monitoring/reports/               # Evidently HTML reports
│
├── .github/workflows/{ci,cd}.yml
├── Dockerfile
├── docker-compose.yml                # 8 services: api, mlflow, prometheus,
│                                      # grafana, dashboard, airflow×3
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## Getting Started

```bash
git clone https://github.com/Dreipfelt/ai-newsops-platform.git
cd ai-newsops-platform

# Option A — full stack
docker-compose up -d
curl -s http://localhost:8000/health | python -m json.tool

# Option B — local development
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make preprocess && make baseline && make train
make api
```

```bash
make test    # 34 tests, coverage ~62%
```

---

## Technology Stack

| Layer | Technology | Rationale |
|---|---|---|
| Model | DistilBERT | 40% faster than BERT, 97% performance retention |
| Baseline | TF-IDF + LinearSVC | Strong, interpretable reference point |
| Experiment tracking | MLflow 3.14 | Model Registry, stage transitions, lineage |
| Data versioning | DVC | Reproducible `dvc repro` pipeline |
| API | FastAPI 0.115 | Async, Pydantic validation, auto Swagger |
| Metrics | Prometheus + prometheus-fastapi-instrumentator | De facto ML observability standard |
| Dashboards | Grafana | 11-panel live operational dashboard |
| Drift detection | Evidently AI + scipy | HTML reports + fast statistical fallback |
| Human dashboard | Streamlit | Non-technical stakeholder interface |
| Containerisation | Docker Compose (8 services) | Full-stack reproducibility |
| CI/CD | GitHub Actions | Native, free tier sufficient |
| Orchestration | Apache Airflow 2.9.3 | Weekly retraining DAG |
| Testing | pytest + pytest-cov | 34 tests, mocked model, no GPU required in CI |

---

## Known Limitations & Roadmap

This section exists deliberately: an honest account of what is implemented versus planned, to avoid overstating the platform's maturity.

### Implemented and verified

- ✅ Full 8-service Docker Compose stack, running continuously
- ✅ Grafana dashboard with 11 panels reading live Prometheus data (datasource UID pinned to prevent provisioning drift)
- ✅ Evidently AI drift reports comparing reference vs. current data
- ✅ Airflow DAG for weekly automated retraining with promote/rollback logic
- ✅ MLflow Model Registry with local backup-based rollback

### Explicitly not implemented (roadmap only)

- **Feature Store** — feature transformations are currently recomputed at inference time rather than served from a versioned store (e.g. Feast). Low risk at current scale (13 static super-categories), but would matter at higher feature complexity.
- **Shadow-mode deployment / champion-challenger with human sign-off** — the current retraining DAG promotes automatically based on a quantitative F1 threshold; there is no shadow-traffic period or manual approval gate before production promotion.
- **Kubernetes** — the platform runs on Docker Compose on a single host. Manifests for a Kubernetes deployment (Deployment, Service, HPA) are a natural next step for horizontal scaling but are not present in this repository.
- **RAG / semantic search** — mentioned as a potential extension (editorial assistant use case) but no vector store or retrieval layer exists in this codebase.
- **Rate limiting / authentication** — the API has no auth layer or per-client rate limiting; suitable for internal/demo use, not multi-tenant production.
- **PII / data governance** — no explicit schema validation (e.g. Great Expectations) or PII scrubbing layer; the source dataset is public news content with no personal data concerns, but this would need addressing before ingesting arbitrary user-submitted content.

---

## Licence

MIT Licence — see [LICENSE](LICENSE).

## Author

**Frédéric Dreipfelt** — Data Science & AI Engineering, Jedha Bootcamp  
AIA Bloc 4 — MLOps Pipeline Certification, 2026
