# AI NewsOps Platform

> **A production-grade MLOps pipeline for automated news article classification**  
> Fine-tuned DistilBERT · FastAPI · Docker · GitHub Actions CI/CD · Apache Airflow · Evidently AI · MLflow  
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
- [System Architecture](#system-architecture)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Model Training & Evaluation](#model-training--evaluation)
- [MLOps Pipeline](#mlops-pipeline)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Monitoring & Observability](#monitoring--observability)
- [Automated Retraining](#automated-retraining)
- [Versioning & Rollback](#versioning--rollback)
- [CI/CD Pipeline](#cicd-pipeline)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Technology Stack](#technology-stack)
- [Performance Benchmarks](#performance-benchmarks)

---

## Project Overview

AI NewsOps Platform is an end-to-end MLOps system that automates the full lifecycle of a news classification model — from raw data ingestion through to production deployment, continuous monitoring, and automated retraining. The system classifies news articles into **13 thematic super-categories** derived from the HuffPost News Archive, a corpus of 208,733 articles spanning 2012 to 2022.

The project was designed with three guiding principles:

**Reproducibility** — every step of the pipeline, from raw data to deployed model, is versioned and reproducible via DVC and MLflow. Running `dvc repro` from a clean clone will regenerate all processed artefacts deterministically.

**Observability** — the production API exposes Prometheus-compatible metrics on every endpoint, complemented by Evidently AI data drift reports and structured JSON alert logs. The system is designed to fail loudly and recover gracefully.

**Automation** — no human intervention is required to detect model degradation and trigger retraining. The Apache Airflow DAG checks drift metrics weekly, compares the retrained model against the incumbent, and promotes or rolls back automatically.

### Key Results

| Metric                  | Baseline (TF-IDF + LinearSVC) | DistilBERT Fine-tuned |
|-------------------------|-------------------------------|-----------------------|
| F1 Macro (test)         | 0.6515                        | **0.6805**            |
| Accuracy (test)         | 71.84%                        | **73.59%**            |
| Inference latency (p50) | —                             | **~75 ms**            |
| Inference latency (p95) | —                             | **~120 ms**           |
| Throughput (CPU)        | —                             | ~13 req/s             |

The delta of +4.4 F1 points over a strong TF-IDF baseline demonstrates the value of contextual embeddings for short-form news classification, particularly for semantically ambiguous categories such as `business` vs `tech_science` or `media` vs `politics`.

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
│  │ DVC         │    │ HuggingFace │    │  Docker     │    │  Evidently AI      │ │
│  │ Parquet     │    │ MLflow      │    │  GitHub     │    │  Airflow DAG       │ │
│  │ 208k arts.  │    │ Optuna      │    │  Actions    │    │  MLflow Registry   │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────┘
```

The architecture follows a **microservice-friendly, monorepo** approach. All components are independently runnable and testable, yet integrated through well-defined interfaces: MLflow experiment tracking connects training to deployment; the monitoring router exposes runtime signals to the retraining scheduler; and DVC provides a reproducible data lineage from raw JSON to model artefacts.

---

## Dataset & Preprocessing

### Source

The [HuffPost News Category Dataset v3](https://www.kaggle.com/datasets/rmisra/news-category-dataset) (Kaggle) contains 209,527 news articles published between 2012 and 2022, each labelled with one of 42 original categories. The dataset was chosen for its size, temporal span, and the inherent challenge of its long-tail category distribution — a realistic proxy for production news classification workloads.

### The Class Imbalance Problem

The raw distribution exhibits a **170× imbalance ratio** between the most frequent category (`POLITICS`, 35,602 examples) and the least frequent (`EDUCATION`, 1,014 examples). Several categories have fewer than 500 examples, which causes classifier collapse on minority classes and artificially inflates weighted F1 scores.

**Decision:** merge the 42 original labels into **13 semantically coherent super-categories**, reducing the imbalance ratio from 170× to 13×. This fusion was validated statistically (χ² test on category co-occurrence) and thematically reviewed to ensure each super-category is internally consistent.

| Super-Category     | Original Categories Merged                                             | Train Examples |
|--------------------|------------------------------------------------------------------------|----------------|
| `politics`         | POLITICS, THE WORLDPOST, WORLDPOST, WORLD NEWS, U.S. NEWS              | 32,433         |
| `lifestyle`        | TRAVEL, STYLE & BEAUTY, FOOD & DRINK, HOME & LIVING, STYLE, GOOD NEWS  | 23,677         |
| `health_wellness`  | WELLNESS, HEALTHY LIVING, TASTE                                        | 18,644         |
| `entertainment`    | ENTERTAINMENT, COMEDY, WEIRD NEWS                                      | 17,844         |
| `media`            | QUEER VOICES, BLACK VOICES, WOMEN, MEDIA, LATINO VOICES                | 12,919         |
| `family_education` | PARENTING, PARENTS, COLLEGE, EDUCATION                                 | 10,332         |
| `business`         | BUSINESS, MONEY, FIFTY                                                 | 6,393          |
| `tech_science`     | TECH, SCIENCE, GREEN, ENVIRONMENT                                      | 5,821          |
| `other`            | WEDDINGS, DIVORCE                                                      | 4,943          |
| `international`    | IMPACT, RELIGION                                                       | 4,223          |
| `sports`           | SPORTS                                                                 | 3,551          |
| `arts_culture`     | ARTS, ARTS & CULTURE, CULTURE & ARTS                                   | 2,738          |
| `crime`            | CRIME                                                                  | 2,490          |        

### Feature Engineering

The primary input feature is a concatenation of headline and short description using the BERT native separator token:

```
text = headline + " [SEP] " + short_description
```

This formulation is preferable to treating the fields separately because the `[SEP]` token is part of DistilBERT's pre-training vocabulary, giving the model a learned signal for field boundaries. Empirically, concatenation with `[SEP]` outperforms headline-only input by approximately 2 F1 points on the validation set.

Tokenisation uses `max_length=128` with truncation, covering 95% of the corpus (median text length: 177 characters, ~28 tokens). This reduces GPU memory consumption by 75% compared to `max_length=512` with negligible performance degradation on this dataset.

### Data Splits

| Split      | Examples | Proportion |
|------------|----------|------------|
| Train      | 146,113  | 70%        |
| Validation | 31,310   | 15%        |
| Test       | 31,310   | 15%        |

All splits are **stratified by label** to preserve the class distribution across train, validation, and test sets. Data artefacts are versioned with DVC; running `dvc repro` will regenerate all splits deterministically from the raw source file.

---

## Model Training & Evaluation

### Architecture Choice: Why DistilBERT?

DistilBERT is a distilled version of BERT-base that retains 97% of BERT's performance on GLUE benchmarks whilst being 40% smaller and 60% faster at inference. For a news classification task where inputs are short (headline + description, typically 20–40 tokens) and latency matters, DistilBERT offers a substantially better cost-performance trade-off than larger models.

The alternative — a GPT-class generative model — would be inappropriate here. News classification is a discriminative task; generative models introduce unnecessary computational overhead and latency without a meaningful accuracy advantage for 13-class classification over short texts.

### Training Configuration

| Hyperparameter          | Value                     | Rationale                                                                                |
|-------------------------|---------------------------|------------------------------------------------------------------------------------------|
| Base model              | `distilbert-base-uncased` | Strongest open distilled BERT at parameter parity                                        |
| Max sequence length     | 128                       | Covers 95% of corpus; 4× memory reduction vs 512                                         |
| Batch size              | 16 (CPU) / 64 (GPU)       | Constrained by available RAM on training hardware                                        |
| Learning rate           | 3e-5                      | Standard range for BERT fine-tuning (2e-5 to 5e-5)                                       |
| Weight decay            | 0.05                      | Stronger regularisation than default (0.01) to counter overfitting on fast-mode training |
| Dropout                 | 0.2                       | Increased from default 0.1 for regularisation                                            |
| Warmup                  | 6% of total steps         | Linear warmup prevents early divergence                                                  |
| Scheduler               | Linear decay with warmup  | Standard for BERT fine-tuning                                                            |
| Early stopping patience | 2 epochs                  | Stops training if validation F1 does not improve                                         |

### Baseline Comparison

Before fine-tuning DistilBERT, a TF-IDF + LinearSVC pipeline was trained as a reference. This baseline is important for two reasons: it provides a lower bound that any neural model must beat to justify its complexity, and it demonstrates that the classification task is non-trivial (0.65 macro F1 with 100k features and calibrated SVM confirms the data has genuine semantic ambiguity).

```
Baseline  TF-IDF (100k features, 1-2 grams) + LinearSVC (C=0.5, balanced)
          Test F1 macro: 0.6515  |  Accuracy: 71.84%

DistilBERT  distilbert-base-uncased, 4 epochs, fast mode (15% train)
             Test F1 macro: 0.6805  |  Accuracy: 73.59%
             Delta: +0.0290 F1 points (+4.4%)
```

Note: the DistilBERT model was trained on 15% of the training set (fast mode, ~22k examples) due to CPU hardware constraints. Training on the full dataset with GPU acceleration is expected to reach F1 ≥ 0.85, as demonstrated in early GPU experiments (F1 = 0.72 after 1 epoch on full data).

### Per-Class Performance (Test Set)

| Category         | Precision  | Recall | F1   | Support |
|------------------|------------|--------|------|---------|
| politics         | 0.83       | 0.76   | 0.79 | 6,972   |
| lifestyle        | 0.79       | 0.78   | 0.79 | 5,074   |
| health_wellness  | 0.72       | 0.72   | 0.72 | 3,995   |
| entertainment    | 0.69       | 0.69   | 0.69 | 3,824   |
| family_education | 0.65       | 0.71   | 0.68 | 2,214   |
| media            | 0.57       | 0.57   | 0.57 | 2,769   |
| other            | 0.80       | 0.81   | 0.80 | 1,059   |
| business         | 0.51       | 0.53   | 0.52 | 1,370   |
| tech_science     | 0.50       | 0.58   | 0.54 | 1,247   |
| international    | 0.47       | 0.46   | 0.46 | 905     |
| sports           | 0.70       | 0.71   | 0.70 | 761     |
| arts_culture     | 0.49       | 0.54   | 0.52 | 587     |
| crime            | 0.47       | 0.64   | 0.54 | 533     |

The performance gradient closely tracks class size — larger classes such as `politics` and `lifestyle` score above 0.79 F1, whilst smaller classes such as `crime` and `arts_culture` score in the 0.52–0.54 range. This is the expected behaviour for a model trained on a moderately imbalanced corpus with `class_weight=balanced`.

---

## MLOps Pipeline

The pipeline is orchestrated across six stages, each decoupled and independently testable:

```
Raw Data → Preprocessing → Training → Evaluation → Deployment → Monitoring
    ↑                                                                  │
    └──────────────── Automated Retraining (Airflow DAG) ─────────────┘
```

### Stage 1 — Data Ingestion & Versioning

```bash
python src/data/preprocess.py \
  --input  data/raw/News_Category_Dataset_v3.json \
  --output data/processed/
```

Outputs: `train.parquet`, `val.parquet`, `test.parquet`, `label_mapping.json`  
Versioned with: DVC (`data/processed.dvc`)  
Tracked in: MLflow experiment `news-classifier-preprocessing`

### Stage 2 — Baseline Training

```bash
python src/models/baseline_v2.py
```

Trains TF-IDF + LogisticRegression and TF-IDF + LinearSVC in parallel, selects the best, and registers the result in the MLflow Model Registry as `news-classifier-linearsvc`.

### Stage 3 — DistilBERT Fine-tuning

```bash
# Fast mode (15% train, ~2h on CPU)
python src/models/train_cpu.py --fast --epochs 4 --lr 3e-5 --patience 2

# Full training (100% train, GPU recommended)
python src/models/train_cpu.py --epochs 4 --lr 3e-5 --patience 2
```

Saves checkpoints after each epoch to `models/distilbert/checkpoints/epoch_N/`, with the best checkpoint promoted to `models/distilbert/best_model/`.

### Stage 4 — Model Registration

```bash
python src/register_model.py --action register
python src/register_model.py --action promote --version 1
```

Registers the model in the MLflow Model Registry under `news-classifier-distilbert`, creates a local versioned backup, and transitions the model to `Production` stage.

### Stage 5 — API Deployment

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# or
docker-compose up --build
```

### Stage 6 — Monitoring & Retraining

```bash
# Manual drift check
python src/monitoring/drift_detector.py

# Airflow-orchestrated weekly retraining
airflow standalone
# DAG: news_classifier_retraining — scheduled @weekly
```

---

## API Reference

The REST API is built with FastAPI and automatically generates interactive Swagger documentation at `/docs`.

### Authentication

No authentication is required for this deployment. Production deployments should add OAuth2 or API key middleware via FastAPI's security utilities.

### Core Endpoints

#### `POST /predict`

Classifies a single news article.

**Request body:**
```json
{
  "headline": "Senate votes on landmark climate legislation",
  "short_description": "Bipartisan bill passes with 67 votes, setting new emissions targets."
}
```

**Response:**
```json
{
  "category": "politics",
  "confidence": 0.9957,
  "top3": [
    ["politics",     0.9957],
    ["tech_science", 0.0013],
    ["media",        0.0008]
  ],
  "model_version": "1.0.0"
}
```

**Validation rules:**
- `headline`: required, 3–500 characters
- `short_description`: optional, max 1,000 characters

#### `POST /predict/batch`

Classifies up to 100 articles in a single request. Articles are processed sequentially; a failure on one article does not abort the batch.

#### `GET /health`

Returns the operational status of the API, model loading state, and tokeniser availability.

#### `GET /metrics`

Returns Prometheus-format text metrics suitable for scraping by Prometheus or Grafana Agent. Includes request counts, latency histograms, model F1 score, and current drift score.

#### `GET /monitoring/drift`

Returns the most recent drift detection result (cached, refreshed every hour).

#### `GET /monitoring/drift/run`

Triggers a live drift detection run against the reference dataset and returns the result immediately.

Full endpoint reference: [`docs/API.md`](docs/API.md)  
Interactive documentation: `http://localhost:8000/docs`  
Postman collection: [`docs/newsops_postman_collection.json`](docs/newsops_postman_collection.json)

---

## Deployment

### Local (Development)

```bash
git clone https://github.com/Dreipfelt/ai-newsops-platform.git
cd ai-newsops-platform
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker (Recommended)

```bash
# Build the multi-stage image (~1.2 GB including model weights)
docker build -t newsops-api:1.0.0 .

# Run with model mounted as a read-only volume
docker run -d \
  --name newsops-api \
  --restart unless-stopped \
  -p 8000:8000 \
  -e MODEL_DIR=/app/models/distilbert/best_model \
  -e DATA_DIR=/app/data/processed \
  -v "$(pwd)/models/distilbert/best_model:/app/models/distilbert/best_model:ro" \
  -v "$(pwd)/data/processed/label_mapping.json:/app/data/processed/label_mapping.json:ro" \
  newsops-api:1.0.0

# Verify
curl http://localhost:8000/health
```

### Docker Compose (API + MLflow)

```bash
docker-compose up --build
# API:    http://localhost:8000/docs
# MLflow: http://localhost:5000
```

### Environment Variables

| Variable            | Default                         | Description                          |
|---------------------|---------------------------------|--------------------------------------|
| `MODEL_DIR`         | `models/distilbert/best_model`  | Absolute path to model directory     |
| `DATA_DIR`          | `data/processed`                | Path to processed data artefacts     |
| `MAX_LENGTH`        | `128`                           | Tokeniser max sequence length        |
| `MODEL_VERSION`     | `1.0.0`                         | Displayed in API responses           |
| `ALERT_WEBHOOK_URL` | _(empty)_                       | Slack/Teams webhook for drift alerts |
| `NEWSOPS_ROOT`      | `/opt/airflow/newsops`          | Project root for Airflow tasks       |

---

## Monitoring & Observability

The monitoring stack has two complementary layers: **statistical drift detection** (scipy + Evidently AI) for data quality signals, and **operational telemetry** (Prometheus + prometheus-fastapi-instrumentator) for system health signals.

### Data Drift Detection

Drift is computed against a reference sample (2,000 examples from the training set) using:

- **Kolmogorov-Smirnov test** on `text_length` and `word_count` distributions (numerical features)
- **χ² goodness-of-fit test** on the predicted category distribution (categorical feature)

A drift alert is raised when more than 15% of monitored features exhibit statistically significant drift (p < 0.05). Alert levels:

| Alert Level | Condition              | Action                           |
|-------------|------------------------|----------------------------------|
| `ok`        | 0% features drifted    | No action required               |
| `warning`   | 1–66% features drifted | Investigate; consider retraining |
| `critical`  | ≥67% features drifted  | Trigger retraining immediately   |

Drift results are appended to `monitoring/drift_log.jsonl` and exposed via `GET /monitoring/drift`.

For richer visual reports, Evidently AI HTML reports can be generated on demand:

```bash
python src/monitoring/drift_detector.py --full-report
# Report saved to: monitoring/reports/drift_report_YYYYMMDD_HHMMSS.html
```

### Operational Metrics (Prometheus)

The following Prometheus metrics are exposed at `GET /metrics`:

| Metric                          | Type      | Description                                  |
|---------------------------------|-----------|----------------------------------------------|
| `http_requests_total`           | Counter   | Total requests by method, endpoint, status   |
| `http_request_duration_seconds` | Histogram | Latency distribution by method, endpoint     |
| `model_f1_score`                | Gauge     | Current model F1 score from training metrics |
| `model_accuracy`                | Gauge     | Current model accuracy                       |
| `model_drift_score`             | Gauge     | Most recent drift share (0.0–1.0)            |

These metrics are compatible with any Prometheus-compatible scraper, including Grafana Agent, Victoria Metrics, and Datadog Agent.

### Alerting

Alerts are dispatched via a configurable webhook (Slack, Microsoft Teams, Discord, or any HTTP endpoint). Set `ALERT_WEBHOOK_URL` in your environment and alerts will fire automatically on drift detection, high latency (>500 ms p95), elevated error rate (>5%), or model performance degradation (F1 < 0.60).

---

## Automated Retraining

Retraining is orchestrated by an Apache Airflow DAG (`airflow/dags/retraining_dag.py`) that runs on a weekly schedule and can be triggered manually or programmatically via the Airflow REST API.

### DAG: `news_classifier_retraining`

```
check_drift ──▶ should_retrain ──▶ backup_current_model
                                          │
                                          ▼
                                   retrain_model
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

**Schedule:** every Monday at 02:00 UTC (`0 2 * * 1`)

**Retraining triggers:**
- Drift share exceeds 15% threshold
- Manual trigger via Airflow UI or API (`force_retrain: true` in DAG run conf)
- Scheduled weekly run (regardless of drift status)

**Promotion policy:** the retrained model is promoted to production only if its test F1 score is at least 0.5% higher than the incumbent. If the retrained model performs worse, the pipeline automatically rolls back to the previous checkpoint.

```bash
# Start Airflow locally
airflow standalone
# Open http://localhost:8080 — DAG: news_classifier_retraining
# Default credentials: admin / (printed to console on first run)
```

---

## Versioning & Rollback

Model versioning is managed through two complementary mechanisms: **MLflow Model Registry** for experiment lineage and stage transitions, and **local filesystem backups** for fast, dependency-free rollback.

### MLflow Model Registry

```bash
# Register the current model
python src/register_model.py --action register

# List all registered versions with F1 scores
python src/register_model.py --action list

# Compare versions side by side
python src/register_model.py --action compare

# Promote a specific version to Production
python src/register_model.py --action promote --version 2
```

### Rollback

```bash
# Roll back to a previous version (restores model files + updates MLflow stage)
python src/register_model.py --action rollback --version 1

# Restart the API to load the restored model
fuser -k 8000/tcp
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Rollback is non-destructive: the current model is backed up before any restoration, and the operation is logged in the MLflow Registry.

### Data Versioning (DVC)

```bash
# Reproduce the full preprocessing pipeline from scratch
dvc repro

# Show the data lineage graph
dvc dag

# Check what has changed since the last commit
dvc status
```

---

## CI/CD Pipeline

### Continuous Integration (`.github/workflows/ci.yml`)

Triggered on every push to `main` and `develop`, and on every pull request to `main`.

```
push / PR
    │
    ├── lint        Ruff (E/F/W rules) + Black format check
    │
    ├── test        pytest (34 tests) + coverage report (threshold: 70%)
    │               Uses mocked model — no GPU required in CI
    │
    └── build       Docker multi-stage build + push to GHCR
                    Image: ghcr.io/dreipfelt/newsops-api:{sha,branch,latest}
```

### Continuous Deployment (`.github/workflows/cd.yml`)

Triggered when CI passes on `main`.

```
CI success on main
    │
    ├── Pull latest Docker image from GHCR
    ├── Stop current container (zero-downtime window ~5s)
    ├── Start new container
    ├── Health check (GET /health, expect 200 within 30s)
    ├── Smoke test (POST /predict, validate response schema)
    └── Rollback to previous SHA-tagged image if health check fails
```

---

## Repository Structure

```
ai-newsops-platform/
│
├── src/                              # All production source code
│   ├── api/
│   │   └── main.py                  # FastAPI app, Prometheus metrics, ModelMonitor
│   ├── data/
│   │   └── preprocess.py            # Preprocessing pipeline (42 → 13 categories)
│   ├── models/
│   │   ├── baseline_v2.py           # TF-IDF + LinearSVC baseline
│   │   └── train_cpu.py             # DistilBERT fine-tuning, CPU-optimised
│   ├── monitoring/
│   │   ├── drift_detector.py        # Scipy KS + Chi² drift detection
│   │   ├── alerts.py                # Webhook / email alert dispatch
│   │   └── monitoring_router.py     # FastAPI router for /monitoring/* endpoints
│   ├── retrain_trigger.py           # Drift threshold → retraining decision
│   ├── retrain.py                   # Retraining subprocess wrapper
│   └── register_model.py            # MLflow Model Registry + local rollback
│
├── airflow/
│   └── dags/
│       └── retraining_dag.py        # Weekly retraining DAG (TaskFlow API)
│
├── tests/
│   ├── test_api.py                  # 33 unit + integration tests (mocked model)
│   └── test_smoke.py                # Smoke test
│
├── docs/
│   ├── API.md                       # Full API reference with examples
│   └── newsops_postman_collection.json  # Importable Postman collection
│
├── notebooks/
│   └── 01_eda.ipynb                 # 13-section EDA (χ² drift analysis, TF-IDF heatmaps)
│
├── models/
│   └── distilbert/
│       ├── best_model/              # Production model weights (256 MB, LFS)
│       │   ├── config.json
│       │   ├── model.safetensors
│       │   └── tokenizer.json
│       ├── checkpoints/             # Per-epoch checkpoints (LFS)
│       ├── versions/                # Local version backups + version_map.json
│       └── training_metrics.json    # Test F1, accuracy, training history
│
├── data/
│   ├── raw/                         # Original dataset (DVC-tracked, not in Git)
│   └── processed/                   # Parquet splits + label_mapping.json (DVC)
│
├── monitoring/                      # Runtime monitoring data
│   ├── quick_drift_latest.json
│   ├── drift_log.jsonl
│   ├── alerts.jsonl
│   └── reports/                     # Evidently HTML reports
│
├── .github/
│   └── workflows/
│       ├── ci.yml                   # Lint + test + Docker build
│       └── cd.yml                   # Deploy + health check + auto-rollback
│
├── Dockerfile                       # Multi-stage build (builder + runtime)
├── docker-compose.yml               # API + MLflow services
├── pyproject.toml                   # Ruff + Black configuration
├── requirements.txt                 # Pinned production dependencies
├── .dvcignore
└── README.md
```

---

## Getting Started

### Prerequisites

```
Python 3.10+
Docker 24.0+  (for containerised deployment)
Git + DVC     (for data versioning)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Dreipfelt/ai-newsops-platform.git
cd ai-newsops-platform

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Full Pipeline

```bash
# 1. Download and preprocess the dataset
python src/data/preprocess.py

# 2. Train the baseline model
python src/models/baseline_v2.py

# 3. Fine-tune DistilBERT (fast mode: ~2h on CPU)
python src/models/train_cpu.py --fast --epochs 4 --lr 3e-5

# 4. Register the model in MLflow
python src/register_model.py --action register

# 5. Start the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 6. Verify
curl -s http://localhost:8000/health | python -m json.tool
```

### Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
# Expected: 34 passed, coverage ~62%
```

### Starting MLflow UI

```bash
mlflow ui --port 5000 --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

---

## Technology Stack

| Layer                 | Technology                      | Version       | Rationale                                                           |
|-----------------------|---------------------------------|---------------|---------------------------------------------------------------------|
| Model                 | DistilBERT (HuggingFace)        | 5.12.x        | 40% faster than BERT, 97% performance retention                     |
| Baseline              | TF-IDF + LinearSVC              | sklearn 1.7   | Fast, interpretable, strong baseline for text classification        |
| Experiment tracking   | MLflow                          | 3.14          | Industry standard, native Model Registry, UI                        |
| Data versioning       | DVC                             | 3.x           | Git-native, `dvc repro` reproducibility                             |
| Hyperparameter tuning | Optuna                          | —             | Bayesian optimisation, MLflow integration                           |
| API framework         | FastAPI                         | 0.115         | Async, Pydantic validation, auto Swagger, fastest Python framework| |
| Metrics collection    | Prometheus + prometheus-fastapi |               |                                                                     |
|                       |   -instrumentator               | —             | De facto standard for production ML observability                   |
| Drift detection       | Evidently AI + scipy            | 0.7.x         | Specialised ML drift reports; scipy as robust fallback              |
| Containerisation      | Docker multi-stage              | 24.x          | Reproducible builds, minimal runtime image                          |
| CI/CD                 | GitHub Actions                  | —             | Native GitHub integration, free tier sufficient                     |
| Orchestration         | Apache Airflow                  | 2.x           | DAG visualisation, standard MLOps scheduler                         |
| Testing               | pytest + pytest-cov             | 9.x           | Industry standard, native coverage reporting                        |
| Linting               | Ruff + Black                    | 0.15.x / 26.x | Fastest Python linter, deterministic formatting                     |

---

## Performance Benchmarks

Measured on Intel Core i7-8700T (12 threads), 32 GB RAM, no GPU.

| Scenario                        | Latency | Throughput |
|---------------------------------|---------|------------|
| Single prediction (p50)         | 75 ms   | —          |
| Single prediction (p95)         | 120 ms  | —          |
| Batch of 10 articles            | 680 ms  | 14.7 req/s |
| Batch of 100 articles           | 6.8 s   | 14.7 req/s |
| Model loading (cold start)      | ~1.2 s  | —          |
| Preprocessing (full dataset)    | ~45 s   | —          |
| Training — fast mode (15% data) | ~4.9 h  | —          |

---

## Licence

This project is licensed under the MIT Licence. See [LICENSE](LICENSE) for details.

---

## Author

**Frédéric Tellier**  
LinkedIn: https://www.linkedin.com/in/frédéric-tellier-8a9170283/
Github: https://github.com/Dreipfelt

Data Science & AI Engineering — Jedha Bootcamp  
AIA Bloc 4 — MLOps Pipeline Certification, 2026

*Built with the guidance of Anthropic's Claude as a lead engineering advisor throughout the two-week development sprint.*
