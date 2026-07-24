# Specification Document — AI NewsOps Platform

**Project** AI NewsOps Platform — Automated news article classification
**Client** NewsMedia Inc. (fictional client, academic context)
**Author** Frédéric Dreipfelt
**Certification** AIA RNCP41993 — Block 4: Building, deploying and operating AI solutions
**Version** 2.0 — July 2026
**Status** Approved for implementation

---

## Table of Contents

1. [Context and Business Problem](#1-context-and-business-problem)
2. [Business Case and Cost Analysis](#2-business-case-and-cost-analysis)
3. [Scope](#3-scope)
4. [Functional Requirements](#4-functional-requirements)
5. [Non-Functional Requirements](#5-non-functional-requirements)
6. [Data Specification](#6-data-specification)
7. [Technical Architecture](#7-technical-architecture)
8. [Accessibility Requirements](#8-accessibility-requirements)
9. [Security, Privacy and Compliance](#9-security-privacy-and-compliance)
10. [CI/CD and Deployment](#10-cicd-and-deployment)
11. [Monitoring, SLI and SLO](#11-monitoring-sli-and-slo)
12. [Automated Retraining Policy](#12-automated-retraining-policy)
13. [Constraints, Assumptions and Risks](#13-constraints-assumptions-and-risks)
14. [Acceptance Criteria](#14-acceptance-criteria)
15. [Out of Scope and Roadmap](#15-out-of-scope-and-roadmap)

---

## 1. Context and Business Problem

NewsMedia Inc. operates an editorial platform ingesting approximately **10,000 news articles per day** from wire services and partner publications. Each article must be assigned an editorial category before it enters the content management system, so that it can be routed to the correct desk, surfaced in the appropriate section, and indexed for archival retrieval.

This categorisation is currently performed manually by the editorial desk. The process does not scale: it introduces a delay between ingestion and publication, produces inconsistent taxonomy across editors, and consumes editorial time that would be better spent on commissioning and fact-checking.

**Objective.** Deliver a production-grade machine learning service that classifies incoming articles automatically into a controlled taxonomy, with sufficient reliability that human review is required only for low-confidence cases, and with the operational instrumentation required to run it unattended.

**Success in one sentence.** The editorial desk should stop categorising articles by hand, and the engineering team should be able to detect and correct model degradation before an editor notices it.

---

## 2. Business Case and Cost Analysis

All figures below are estimates stated for the purpose of sizing the project. They are not audited financial data.

### 2.1 Cost of the current manual process

| Parameter | Value |
|---|---|
| Articles to classify per day | 10,000 |
| Average handling time per article | ~20 seconds |
| Daily editorial effort | ~55 hours |
| Full-time equivalents required (7 productive hours/day) | **≈ 7.9 FTE** |
| Fully loaded cost per FTE | €45,000 / year |
| **Annual cost of manual classification** | **≈ €357,000** |

### 2.2 Cost of the automated platform

| Item | Basis | Annual estimate |
|---|---|---|
| Model training | One-off, ~4 hours CPU on existing hardware | negligible |
| Inference runtime | 1 vCPU / 2 GB container, running continuously | ~€420 |
| Observability stack (Prometheus, Grafana, MLflow) | Self-hosted alongside the API | ~€300 |
| Orchestration (Airflow, PostgreSQL metadata) | Self-hosted | ~€240 |
| Engineering maintenance | 0.2 FTE | ~€9,000 |
| **Total annual operating cost** | | **≈ €9,960** |

### 2.3 Expected return

The platform is not expected to replace human judgement entirely. Predictions below a confidence threshold are routed to an editor for review; on the current model, approximately **15% of volume** is expected to fall into this category.

| Metric | Value |
|---|---|
| Volume automated without human intervention | ~85% |
| Residual manual effort | ≈ 1.2 FTE (≈ €54,000/year) |
| Gross annual saving | ≈ €303,000 |
| Net annual saving after platform cost | **≈ €293,000** |
| Payback period | **under one month** |

### 2.4 Non-financial benefits

Beyond direct cost displacement, automation delivers a consistent taxonomy that manual classification cannot guarantee across a rotating editorial team; near-instantaneous categorisation, removing a queueing delay from the publication pipeline; and a structured, machine-readable content layer that becomes the foundation for later capabilities such as semantic search or recommendation.

### 2.5 Cost of inaction

Continuing manually means the €357,000 annual cost persists and scales linearly with ingestion volume. At a projected 15% year-on-year growth in article volume, the manual approach would require an additional FTE within eighteen months.

---

## 3. Scope

### 3.1 In scope

The delivery covers an inference API exposing single and batch classification; a trained and versioned classification model; a reproducible data preparation pipeline; a continuous integration and deployment pipeline; a monitoring stack covering both service health and model behaviour; an automated retraining pipeline with promotion and rollback logic; and human-facing dashboards for both technical and editorial stakeholders.

### 3.2 Out of scope for this release

Article ingestion from wire services (the client's existing CMS supplies articles to the API); summarisation, entity extraction or any generative capability; multilingual support (English only); and end-user authentication (see §9.2).

---

## 4. Functional Requirements

| ID | Requirement | Priority |
|---|---|:---:|
| FR-01 | The system shall classify a single article, supplied as a headline and an optional short description, into exactly one of thirteen editorial categories. | Must |
| FR-02 | The system shall return, alongside the predicted category, a confidence score and the three highest-scoring candidate categories. | Must |
| FR-03 | The system shall accept batch requests of up to 100 articles in a single call. | Must |
| FR-04 | The system shall expose the model version used for each prediction, to allow downstream traceability. | Must |
| FR-05 | The system shall reject malformed input with an explicit validation error rather than returning a low-quality prediction. | Must |
| FR-06 | The system shall expose its own health status and operational metrics for external monitoring. | Must |
| FR-07 | The system shall expose the current data drift status and the history of drift evaluations. | Should |
| FR-08 | The system shall provide a human-facing dashboard allowing a non-technical user to test a classification interactively. | Should |

### 4.1 Category taxonomy

The source dataset carries 42 raw categories with a 170:1 imbalance ratio between the largest and smallest. These are consolidated into **thirteen super-categories** — `politics`, `lifestyle`, `health_wellness`, `entertainment`, `media`, `family_education`, `business`, `tech_science`, `other`, `international`, `sports`, `arts_culture`, `crime` — reducing the imbalance ratio to 13:1.

This consolidation is a deliberate modelling decision, not a data-cleaning convenience: several source categories held fewer than 500 examples, at which point a classifier cannot learn them reliably and macro-averaged metrics become dominated by noise on those classes.

---

## 5. Non-Functional Requirements

### 5.1 Performance targets and their justification

Performance targets are calibrated against the **actual deployment hardware** — a 12-thread CPU host with no GPU — and against the **actual business volume**, rather than against aspirational figures.

**Throughput sizing.** 10,000 articles per day, concentrated in an eight-hour editorial window, corresponds to an average sustained load of **0.35 requests per second**. Applying a conservative peak factor of 10 to accommodate wire-service bursts gives a peak requirement of **3.5 requests per second**. A single-replica deployment measured at 7.62 requests per second therefore provides approximately **2.2× headroom over peak demand**. A higher target would not serve the business need and would only misrepresent the sizing exercise.

| ID | Requirement | Target | Measured |
|---|---|---|---|
| NFR-01 | Latency, single isolated request (p95) | < 100 ms | ~5 ms |
| NFR-02 | Latency under 15 concurrent users (p95) | < 2,000 ms | 1,800 ms |
| NFR-03 | Sustained throughput, single replica | ≥ 5 req/s | 7.62 req/s |
| NFR-04 | Error rate under sustained load | < 1% | 0.00% |
| NFR-05 | Cold start (model load to first response) | < 60 s | ~15 s |
| NFR-06 | Batch of 100 articles | < 10 s | ~6.8 s |

### 5.2 Model quality targets

| ID | Requirement | Target | Measured |
|---|---|---|---|
| NFR-07 | Macro F1 on held-out test set | ≥ 0.65 | 0.6791 |
| NFR-08 | Accuracy on held-out test set | ≥ 0.70 | 0.7382 |
| NFR-09 | Improvement over TF-IDF baseline | ≥ +2% relative | +4.2% |

The macro F1 target of 0.65 is set deliberately above the measured TF-IDF + LinearSVC baseline of 0.6515, so that the target cannot be met without the neural model justifying its additional complexity. Macro averaging is chosen over weighted averaging precisely because it refuses to let strong performance on `politics` (F1 0.83, 444 test examples) mask weaker performance on `arts_culture` (F1 0.52, 33 test examples).

### 5.3 Reliability and maintainability

| ID | Requirement |
|---|---|
| NFR-10 | The full stack shall be reproducible from a clean clone via a single `docker compose up` command. |
| NFR-11 | Data preparation shall be deterministic and reproducible from the raw source (fixed random seed, versioned with DVC). |
| NFR-12 | Every model promoted to production shall be traceable to the experiment run, hyperparameters and dataset version that produced it. |
| NFR-13 | A failed deployment shall roll back automatically without human intervention. |
| NFR-14 | Automated test coverage shall be maintained on the API and data preparation layers. |

---

## 6. Data Specification

### 6.1 Source

| Attribute | Value |
|---|---|
| Dataset | HuffPost News Category Dataset v3 |
| Provider | Kaggle (`rmisra/news-category-dataset`) |
| Volume | 209,527 articles |
| Period | 2012–2022 |
| Language | English |
| Licence | See `DATA_LICENSE.md` |
| Personal data | None — public editorial content, author names are bylines already published |

### 6.2 Preparation pipeline

Deduplication on headline and description; removal of records with a missing or trivially short headline; restriction to the 2012–2022 window; consolidation of 42 categories into 13; construction of the model input as `headline [SEP] short_description`, exploiting the separator token native to the BERT vocabulary; and a stratified 70/15/15 split with a fixed seed.

After cleaning, **208,733 records** are retained: 146,113 training, 31,310 validation, 31,310 test.

### 6.3 Data integrity control

The preparation script enforces two blocking validations on every run. First, the resulting category set must match the expected thirteen exactly — a missing or unexpected category raises an exception rather than producing silently degraded data. Second, the textual category column and the encoded integer label must agree for 100% of rows in all three splits.

This second check was added following a defect in which 2,743 `arts_culture` articles had been silently absorbed into `entertainment` by a stale mapping, desynchronising the parquet files from the trained model. The symptom surfaced only during an evaluation run producing a near-random macro F1 of 0.02. The control now makes that class of defect impossible to introduce without an immediate, loud failure.

---

## 7. Technical Architecture

| Layer | Technology | Rationale |
|---|---|---|
| Model | DistilBERT (`distilbert-base-uncased`), fine-tuned | Retains ~97% of BERT's GLUE performance at 40% fewer parameters and ~60% faster inference — the correct trade-off for short-text classification on CPU |
| Baseline | TF-IDF + LinearSVC | Provides a strong, interpretable reference point that the neural model must beat to justify itself |
| Hyperparameter search | Optuna (TPE sampler, median pruning) | Bayesian search with native MLflow integration |
| Experiment tracking | MLflow | Model Registry with stage transitions and full run lineage |
| Data versioning | DVC | Git-native, reproducible pipeline |
| API | FastAPI | Async-native, Pydantic validation, auto-generated OpenAPI documentation |
| Containerisation | Docker Compose (8 services) | Environment parity between development and deployment |
| Metrics | Prometheus + `prometheus-fastapi-instrumentator` | De facto standard for production ML observability |
| Dashboards | Grafana (11 panels) | Live operational and model-health visualisation |
| Drift detection | Evidently AI + SciPy | Rich HTML reporting alongside a fast statistical path |
| Stakeholder interface | Streamlit | Non-technical access for the editorial team |
| Orchestration | Apache Airflow | Scheduled retraining with visual DAG monitoring |
| CI/CD | GitHub Actions | Native integration, no additional infrastructure |

### 7.1 Model configuration

The production model uses a maximum sequence length of 128 tokens, covering approximately 95% of the corpus whilst reducing memory consumption fourfold compared with 512. Training used a learning rate of 3e-05, weight decay of 0.05 and dropout of 0.2.

A systematic Optuna search over eight trials subsequently identified a superior configuration — learning rate 3.82e-05, weight decay 0.021, dropout 0.136, warmup ratio 0.028 — reaching a validation macro F1 of 0.6561 on a 10% sample. The search attributed **84% of performance variance to the learning rate alone** (fANOVA), against 9% for dropout, 5% for weight decay and 2% for warmup. Applying this configuration to a full training run is scheduled as the immediate next iteration (§15).

---

## 8. Accessibility Requirements

Accessibility is a delivery requirement, not a post-hoc consideration. The applicable standards are **RGAA 4.1** (French public-sector reference) and **WCAG 2.1 level AA**, on which RGAA is based.

### 8.1 Scope of application

The platform exposes three interfaces with materially different accessibility obligations.

The **REST API** is a machine interface. It returns structured JSON with no visual, colour or spatial dependency, and is consumable by any assistive technology through the calling application. Accessibility obligations at this layer fall on the consuming client, not on the API — but the API must not *obstruct* accessible consumption, which drives requirement A-01 below.

The **Streamlit dashboard** is the editorial-facing interface and carries the fullest obligation, as it is intended for non-technical users who may rely on assistive technology.

The **Grafana dashboards** are an internal engineering interface with a narrower user base, but the same principles apply.

### 8.2 Requirements

| ID | Requirement | WCAG ref. | Interface | Status |
|---|---|:---:|---|:---:|
| A-01 | No information shall be conveyed by colour alone. Drift and health states shall be returned as explicit text values (`ok`, `warning`, `critical`), not only as a colour. | 1.4.1 | API, all dashboards | Implemented |
| A-02 | All interactive controls shall be reachable and operable via keyboard alone, with a visible focus indicator. | 2.1.1, 2.4.7 | Streamlit | Implemented |
| A-03 | Text and meaningful interface elements shall meet a minimum contrast ratio of 4.5:1. | 1.4.3 | Streamlit, Grafana | Implemented |
| A-04 | Every dashboard panel shall carry a descriptive title stating what is measured and in which unit, so that its purpose is intelligible without interpreting the visualisation. | 1.1.1 | Grafana | Implemented |
| A-05 | Data presented graphically shall also be available in a non-graphical form (JSON endpoint, CSV export, or tabular view). | 1.1.1 | API, Streamlit | Implemented |
| A-06 | Page language shall be declared programmatically. | 3.1.1 | Streamlit | Implemented |
| A-07 | Error messages shall identify the field in error and describe the correction required, in plain language. | 3.3.1, 3.3.3 | API, Streamlit | Implemented |
| A-08 | Content shall remain usable at 200% zoom without loss of function or horizontal scrolling. | 1.4.4 | Streamlit | Partial |
| A-09 | Static analytical figures (confusion matrix, hyperparameter importance plots) shall be accompanied by a textual description of their content and conclusion. | 1.1.1 | Documentation | Partial |

### 8.3 Design decisions taken for accessibility

Returning drift status as a text enumeration rather than a colour code (A-01) was a deliberate API design choice: it means an editorial user consuming the status through a screen reader receives the same information as an engineer looking at a red panel in Grafana.

Publishing the underlying data of every visualisation in machine-readable form (A-05) means that no conclusion in this platform is accessible *only* through a chart. The confusion matrix figure is accompanied by `docs/visuals/classification_report.txt`; the Optuna plots are accompanied by `models/distilbert/optuna/optuna_results.json`.

### 8.4 Known gaps and honest limitations

No formal RGAA audit has been conducted; the assessment above is a self-evaluation against the reference criteria and would need third-party confirmation before a public-sector deployment.

Streamlit does not expose full control over ARIA landmarks and heading hierarchy, which limits how precisely the dashboard structure can be communicated to a screen reader. A production deployment serving users with assistive technology as a primary audience would justify replacing Streamlit with a bespoke accessible front end.

Requirement A-08 is marked partial because zoom behaviour has been verified visually but not tested with assistive technology. Requirement A-09 is partial because textual descriptions exist for the principal figures but not exhaustively.

---

## 9. Security, Privacy and Compliance

### 9.1 Data protection

The source corpus contains published editorial content with no personal data as defined by GDPR Article 4. Author bylines are already public. No special-category data is processed. No data subject rights procedure is therefore required for the training corpus.

Should the platform later ingest user-submitted content, a personal-data screening layer would become mandatory before ingestion — this is recorded as a roadmap item, not an implemented control.

### 9.2 Authentication and access control

The platform is deployed **without authentication**, which is appropriate for the internal, single-tenant, non-production context of this certification and is stated here explicitly rather than left implicit.

Production deployment would require, at minimum: OAuth 2.0 or API-key authentication on all write and inference endpoints; per-client rate limiting (indicative target: 100 requests per minute); TLS 1.3 termination; and network isolation of the observability stack from the public interface.

### 9.3 Audit trail

Every prediction is logged with a timestamp, the predicted category, the confidence score and the model version. Retention is set at twelve months. Article text is not retained in prediction logs.

---

## 10. CI/CD and Deployment

### 10.1 Continuous integration

Triggered on every push and pull request to `main`: static analysis (Ruff, Black), automated test suite (34 tests, executed against a mocked model so that no GPU or model artefact is required on the runner), and container image build.

### 10.2 Continuous deployment

Triggered on successful integration against `main`: image pull, container replacement, post-deployment health check with a 30-second timeout, and a functional smoke test issuing a real prediction request. **Failure of either verification triggers an automatic rollback** to the previously tagged image, without human intervention (NFR-13).

---

## 11. Monitoring, SLI and SLO

### 11.1 Service level indicators

The following indicators are measured continuously by Prometheus and displayed in Grafana. Values in the *Measured* column are drawn from the load-test campaign documented in `docs/load_test_comparison.md` and from the training metrics in `models/distilbert/training_metrics.json`.

| SLI | Objective | Measured | Met |
|---|---|---|:---:|
| Availability (single host, observation period) | ≥ 99% | stable over multi-day continuous operation | ✅ |
| Latency p95, isolated request | < 100 ms | ~5 ms | ✅ |
| Latency p95, 15 concurrent users | < 2,000 ms | 1,800 ms | ✅ |
| Sustained throughput, single replica | ≥ 5 req/s | 7.62 req/s | ✅ |
| Error rate under sustained load | < 1% | 0.00% | ✅ |
| Macro F1 on test set | ≥ 0.65 | 0.6791 | ✅ |
| Accuracy on test set | ≥ 0.70 | 0.7382 | ✅ |
| Drift detection latency | < 30 min | evaluated on demand and on schedule | ✅ |

### 11.2 Alert thresholds

| Alert | Threshold | Automated action |
|---|---|---|
| Elevated latency | p95 > 3,000 ms sustained | Investigate CPU saturation; reduce batch size |
| Elevated error rate | > 1% of 5xx responses | Restart API container; verify model loading |
| Data drift detected | drift score > 0.30 | Trigger retraining DAG; record drift event |
| Model quality degraded | macro F1 < 0.65 | Block candidate promotion |

### 11.3 Two-layer drift detection

Detection operates on two complementary paths. A **fast statistical path** (SciPy) applies a Kolmogorov–Smirnov test to numerical text features and a chi-squared test to the predicted category distribution, returning in under one second and driving the API endpoint. A **rich reporting path** (Evidently AI) produces an HTML report with distribution overlays for human review, generated on demand.

An alert is raised when more than 15% of monitored features exhibit statistically significant drift, or when the aggregate drift score exceeds 0.30.

---

## 12. Automated Retraining Policy

Retraining is orchestrated by an Apache Airflow DAG (`newsops_retraining_pipeline`) running on a daily schedule and additionally triggerable on demand.

The pipeline validates incoming data, evaluates drift, and proceeds to retraining only if drift exceeds the configured threshold or a manual trigger is present — avoiding pointless compute when the data distribution is stable. A candidate model is then trained, evaluated against the incumbent, and subjected to the promotion policy below.

**Promotion policy.** A candidate is promoted to production only if its macro F1 exceeds the incumbent's by at least 0.5 percentage points, or remains within a 1-point tolerance band (permitting promotion of a model trained on fresher data at equivalent quality). A candidate scoring below that band triggers an automatic rollback to the previous checkpoint. No regressive model can reach production without quantitative validation.

Every retraining outcome — promoted or rolled back — is recorded in the MLflow Model Registry with its metrics and lineage.

---

## 13. Constraints, Assumptions and Risks

### 13.1 Constraints

| ID | Constraint | Consequence |
|---|---|---|
| C-01 | Deployment hardware is a 12-thread CPU host with no GPU | Training and inference budgets are calibrated accordingly; performance targets in §5.1 reflect this |
| C-02 | Delivery timeframe is a fixed academic window | Training runs use reduced data samples where a full run would exceed the schedule; this is stated wherever it applies |
| C-03 | Single-host deployment | Horizontal scaling is bounded by shared CPU contention (see R-02) |

### 13.2 Assumptions

Articles arrive in English with a populated headline; the client's CMS handles ingestion and calls the API; and the editorial taxonomy of thirteen categories remains stable over the retraining cycle.

### 13.3 Risks

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|:---:|:---:|---|
| R-01 | Weak performance on low-support categories (`arts_culture`, `international`) degrades editorial trust | High | Medium | Macro-averaged metrics prevent the issue being masked; low-confidence predictions routed to human review |
| R-02 | Horizontal scaling does not deliver expected gains on shared CPU | Confirmed | Medium | Empirically measured and documented (`docs/load_test_comparison.md`); resolution requires physically isolated compute per replica, recorded in roadmap |
| R-03 | Editorial taxonomy changes, invalidating the trained model | Medium | High | Category mapping centralised in a single canonical location with blocking validation; retraining pipeline available on demand |
| R-04 | Data pipeline desynchronises from the trained model | Occurred once | High | Blocking consistency validation added to the preparation script (§6.3) |
| R-05 | Drift goes undetected between scheduled evaluations | Low | Medium | On-demand drift endpoint plus scheduled DAG evaluation |

Risk R-02 and R-04 are recorded as *confirmed* and *occurred* rather than hypothetical, because both materialised during delivery and were measured and remediated. Recording them as theoretical would misrepresent the project's history.

---

## 14. Acceptance Criteria

The delivery is considered to meet this specification when all of the following hold simultaneously.

| # | Criterion | Verification method |
|:---:|---|---|
| 1 | The full stack starts from a clean clone with a single command and reports all services healthy | `docker compose --profile airflow up -d` then `scripts/verify_all.sh` |
| 2 | The API classifies a submitted article and returns category, confidence and top-3 | `POST /predict` |
| 3 | All quality targets in §5.2 are met on the held-out test set | `models/distilbert/training_metrics.json` |
| 4 | All performance targets in §5.1 are met under the documented load profile | `docs/load_test_comparison.md` |
| 5 | The CI pipeline passes on the current `main` | GitHub Actions status |
| 6 | A failed deployment rolls back automatically | CD workflow rollback step |
| 7 | The monitoring stack reports live model and service metrics | Grafana dashboard, 11 panels |
| 8 | Drift can be evaluated on demand and the history retrieved | `GET /monitoring/drift`, `GET /monitoring/drift/logs` |
| 9 | The retraining DAG executes end to end with promotion or rollback | Airflow UI, `newsops_retraining_pipeline` |
| 10 | Data preparation is reproducible and passes its integrity validation | `python src/data/preprocess.py` |
| 11 | Accessibility requirements marked *Implemented* in §8.2 are verifiable | Manual inspection against the stated WCAG criteria |

---

## 15. Out of Scope and Roadmap

The following are deliberately **not implemented** in this release. They are recorded here so that the boundary of the delivery is explicit and cannot be mistaken for an oversight.

| Item | Rationale for exclusion | Priority |
|---|---|:---:|
| Full training run with Optuna-optimal hyperparameters on 100% of data | Compute budget within the delivery window (C-02); configuration identified and ready to apply | High |
| Authentication and rate limiting | Not required for the single-tenant internal context; specified in §9.2 for production | High |
| Kubernetes multi-node deployment | Requires infrastructure beyond the single available host; empirically justified by R-02 | Medium |
| Feature store (e.g. Feast) | Feature transformations are inexpensive and recomputed at inference; low value at thirteen static categories | Medium |
| Shadow-mode deployment with human sign-off | Current promotion is quantitative and automatic; a shadow period would require production traffic volume not available here | Medium |
| Schema validation layer (e.g. Great Expectations) | Partially covered by the integrity controls in §6.3; a dedicated layer would be warranted with multiple upstream data sources | Low |
| Semantic search / RAG capability | A separate product capability, not a prerequisite for classification | Low |
| Multilingual support | Corpus and business need are English-only | Low |

---

**Document control**

| Version | Date | Change |
|---|---|---|
| 1.0 | July 2026 | Initial specification |
| 2.0 | July 2026 | Added business case (§2); performance targets recalibrated against measured hardware capability and business volume (§5.1); accessibility requirements expanded to full RGAA/WCAG mapping with honest gap statement (§8); risks R-02 and R-04 recorded as materialised; acceptance criteria formalised (§14) |
