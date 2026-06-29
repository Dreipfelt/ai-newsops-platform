cat > docs/architecture.md <<'EOF'
# Architecture

## 1. Project overview

AI NewsOps Platform is an MLOps-oriented platform designed for a news agency use case.

The objective is to help journalists process large volumes of news articles by providing:

- automatic article classification;
- summarization;
- semantic search;
- duplicate detection;
- RAG-based question answering;
- monitoring;
- retraining;
- model versioning;
- deployment automation.

The project starts with a simple and reliable FastAPI service, then evolves toward a complete MLOps architecture.

---

## 2. Business context

News agencies receive and process a large number of articles, dispatches, and external sources every day.

Journalists need to quickly answer questions such as:

- What category does this article belong to?
- Is this article a duplicate of another source?
- What are the main entities and topics?
- Can I search previous articles semantically?
- Can the system summarize the relevant background?
- Is the model still performing correctly over time?

AI NewsOps Platform addresses this through a modular AI and MLOps architecture.

---

## 3. Current architecture

Current implemented scope:

```text
User / Developer
      |
      v
FastAPI application
      |
      v
Health endpoint
      |
      v
Pytest validation
      |
      v
Docker image
      |
      v
GitHub Actions CI

Implemented components:

Component	    Status	        Description
FastAPI API	    Implemented	    Main API service
/health         endpoint	    Implemented	Service health check
Pytest	        Implemented	    Local and CI tests
Dockerfile	    Implemented	    API container image
Docker Compose	Implemented	    Local container orchestration
GitHub Actions	Implemented	    CI test and Docker build

4. Target architecture

Target production-oriented architecture:

News Dataset / RSS Feeds
        |
        v
Data Ingestion
        |
        v
Data Validation
        |
        v
Preprocessing
        |
        +---------------------+
        |                     |
        v                     v
Classification Model     Embedding Model
        |                     |
        v                     v
MLflow Registry          Qdrant Vector DB
        |                     |
        +----------+----------+
                   |
                   v
              FastAPI API
                   |
        +----------+----------+
        |          |          |
        v          v          v
 Classification  RAG     Duplicate Detection
        |
        v
Journalist Dashboard
        |
        v
Feedback + Logs
        |
        v
Monitoring with Evidently
        |
        v
Retraining Pipeline
        |
        v
Model Versioning + Rollback
5. Main components
5.1 API layer

Technology: FastAPI

Responsibilities:

expose AI services;
validate request and response schemas;
provide health checks;
expose prediction endpoints;
provide API documentation through OpenAPI.

Current endpoint:

GET /health

Target endpoints:

POST /classify
POST /summarize
POST /semantic-search
POST /duplicates
POST /rag/query
POST /feedback
GET  /metrics
5.2 Data layer

Local data is excluded from Git.

Expected local dataset path:

data/raw/News_Category_Dataset_v3.json

The dataset is downloaded manually from Kaggle and documented in DATA_LICENSE.md.

Planned data stages:

data/raw/
data/processed/
data/reference/

Purpose:

raw data storage;
cleaned training data;
reference data for drift monitoring;
production-like data simulation.
5.3 Model layer

The model layer will evolve in stages.

Stage 1 — Baseline model
TF-IDF
  |
  v
Logistic Regression

Purpose:

establish a simple benchmark;
produce interpretable metrics;
validate the end-to-end training workflow.
Stage 2 — Embedding-based classifier
Sentence Transformer
  |
  v
Article embeddings
  |
  v
Classifier

Purpose:

improve semantic understanding;
reuse embeddings for semantic search and duplicate detection;
reduce duplication between ML and RAG pipelines.
Stage 3 — RAG assistant
Question
  |
  v
Embedding
  |
  v
Qdrant retrieval
  |
  v
LLM response with sources

Purpose:

provide a journalist assistant grounded in indexed news articles;
reduce hallucination risk by returning sources.
5.4 Vector database

Technology: Qdrant

Planned responsibilities:

store article embeddings;
support semantic search;
support duplicate detection;
support RAG retrieval.
5.5 MLOps layer

Planned tools:

Need	Tool
Experiment tracking	MLflow
Model registry	MLflow Registry
Monitoring	Evidently
Containerization	Docker
CI/CD	GitHub Actions
Orchestration	Airflow
Deployment	Kubernetes
Vector search	Qdrant
6. CI/CD architecture

Current CI workflow:

Push to main
    |
    v
GitHub Actions
    |
    +--> Install dependencies
    |
    +--> Run tests
    |
    +--> Build Docker image

This ensures that each change pushed to the repository remains testable and containerizable.

7. Deployment strategy

Current deployment mode:

Local Docker Compose

Target deployment modes:

Docker Compose for local development
Kubernetes for production-like demonstration

Planned Kubernetes components:

Namespace
Deployment
Service
ConfigMap
Secret
Rolling update
Rollback
8. Monitoring strategy

The monitoring layer will track both technical and ML indicators.

Technical metrics:

API latency;
request count;
error rate;
container health.

ML metrics:

prediction distribution;
confidence distribution;
input text length;
category drift;
embedding drift;
model performance when labels are available.

Drift detection will compare reference data with recent production-like data.

9. Retraining strategy

Retraining can be triggered by:

new labeled articles;
detected data drift;
model performance degradation;
journalist feedback.

Target retraining flow:

New data / Drift alert
        |
        v
Preprocessing
        |
        v
Training
        |
        v
Evaluation
        |
        v
Compare with production model
        |
        +--> Accept and register model
        |
        +--> Reject model

A model is promoted only if it improves or maintains the expected performance threshold.

10. Rollback strategy

Rollback will be handled at two levels:

Model rollback through MLflow model versions.
Application rollback through Docker image tags or Kubernetes rollout undo.

Target command example:

kubectl rollout undo deployment/ai-newsops-api
11. Design principles

This project follows these principles:

keep the API simple and testable;
separate API code from ML pipelines;
keep data and models out of Git;
use Docker for reproducibility;
validate every change with CI;
add MLOps tools incrementally;
avoid unnecessary complexity before the core pipeline works.

EOF


## `docs/api.md`

```bash
cat > docs/api.md <<'EOF'
# API Documentation

## 1. Overview

AI NewsOps Platform exposes its services through a FastAPI application.

The API is designed to support newsroom-oriented AI features such as:

- article classification;
- summarization;
- semantic search;
- duplicate detection;
- RAG-based question answering;
- monitoring;
- feedback collection.

FastAPI automatically exposes OpenAPI documentation.

Local Swagger UI:

```text
http://localhost:8000/docs
```
## 2. Base URL

Local development:

http://localhost:8000

Docker Compose:

http://localhost:8000

## 3. Current endpoints
### 3.1 Health check
GET /health

Checks that the API service is running.

Example request
curl http://localhost:8000/health
Example response
{
  "status": "ok",
  "service": "ai-newsops-api",
  "version": "0.1.0"
}
Response fields
Field	Type	Description
status	string	Health status
service	string	Service name
version	string	API version

## 4. Planned endpoints

The following endpoints are planned for the complete MLOps platform.

### 4.1 Classify article
POST /classify

Classifies a news article into an editorial category.

Request body
{
  "headline": "European leaders announce new AI regulation",
  "short_description": "The new regulation introduces stricter rules for high-risk AI systems."
}
Response body
{
  "category": "POLITICS",
  "confidence": 0.87,
  "model_version": "baseline-v1"
}
Response fields
Field	        Type	Description
category	    string	Predicted article category
confidence	    number	Model confidence score
model_version	string	Version of the model used

### 4.2 Summarize article
POST /summarize

Generates a short editorial summary of an article.

Request body
{
  "headline": "Central bank announces interest rate decision",
  "article": "Full article text goes here..."
}
Response body
{
  "summary": "The central bank announced its latest interest rate decision amid inflation concerns.",
  "summary_type": "short",
  "model_version": "summarizer-v1"
}

### 4.3 Semantic search
POST /semantic-search

Searches articles by semantic similarity instead of keyword matching.

Request body
{
  "query": "European AI regulation and technology companies",
  "top_k": 5
}
Response body
{
  "results": [
    {
      "article_id": "article-001",
      "headline": "EU announces new AI regulation",
      "score": 0.91
    }
  ]
}

### 4.4 Duplicate detection
POST /duplicates

Detects whether an article is similar to already indexed articles.

Request body
{
  "headline": "Explosion reported in city center",
  "short_description": "Authorities are investigating an explosion reported downtown.",
  "threshold": 0.9
}
Response body
{
  "is_duplicate": true,
  "matches": [
    {
      "article_id": "article-042",
      "headline": "Blast reported in downtown area",
      "score": 0.93
    }
  ]
}

### 4.5 RAG query
POST /rag/query

Answers a journalist question using retrieved news articles as sources.

Request body
{
  "question": "What are the latest developments on AI regulation in Europe?",
  "top_k": 5
}
Response body
{
  "answer": "European AI regulation is currently focused on high-risk AI systems, transparency obligations, and compliance requirements for technology companies.",
  "sources": [
    {
      "article_id": "article-001",
      "headline": "EU announces new AI regulation",
      "score": 0.91
    }
  ]
}

### 4.6 Feedback
POST /feedback

Collects human feedback from journalists.

Request body
{
  "article_id": "article-001",
  "predicted_category": "POLITICS",
  "corrected_category": "TECH",
  "comment": "The article is mostly about AI companies, not politics."
}
Response body
{
  "status": "received"
}

Feedback will later be used for monitoring and retraining.

### 4.7 Metrics
GET /metrics

Exposes technical and ML monitoring metrics.

Planned metrics:

request count;
latency;
error count;
prediction distribution;
confidence distribution;
drift indicators.

## 5. Error format

Target error format:

{
  "error": "Invalid request",
  "details": "The field 'headline' is required."
}

## 6. Versioning

The API version is exposed in the health endpoint.

Current version:

0.1.0

Future breaking changes should use versioned routes:

/api/v1/classify
/api/v1/rag/query


## 7. Local validation

Run the API locally:

make api

Test the health endpoint:

curl http://localhost:8000/health

Run automated tests:

make test

EOF


## Validation + commit

```bash
git add docs/architecture.md docs/api.md
git commit -m "Add architecture and API documentation"
git push
```