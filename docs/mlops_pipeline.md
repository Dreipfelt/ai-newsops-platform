# MLOps Pipeline

## 1. Objective

The purpose of AI NewsOps Platform is to automate the complete lifecycle of an NLP model used inside a news agency.

The platform continuously ingests articles, trains and serves machine learning models, indexes semantic embeddings, monitors production quality, and automatically retrains models when performance degrades.

---

# 2. Global Architecture

```text
                        +----------------------+
                        |    News Dataset      |
                        |  Kaggle / RSS Feeds  |
                        +----------+-----------+
                                   |
                                   v
                     +---------------------------+
                     |      Data Ingestion       |
                     +------------+--------------+
                                  |
                                  v
                     +---------------------------+
                     |     Data Validation       |
                     +------------+--------------+
                                  |
                                  v
                     +---------------------------+
                     |     Preprocessing         |
                     +------------+--------------+
                                  |
             +--------------------+--------------------+
             |                                         |
             v                                         v
+---------------------------+          +---------------------------+
| Baseline Classification   |          | Sentence Transformer      |
+-------------+-------------+          +-------------+-------------+
              |                                          |
              v                                          v
       Model Evaluation                        Article Embeddings
              |                                          |
              +--------------------+---------------------+
                                   |
                                   v
                          +-------------------+
                          |      MLflow       |
                          | Tracking/Registry |
                          +---------+---------+
                                    |
                                    v
                      +---------------------------+
                      |     Model Validation      |
                      +-------------+-------------+
                                    |
                    Accepted         |        Rejected
                                    |
                                    v
                        +----------------------+
                        |   Docker Image       |
                        +----------+-----------+
                                   |
                                   v
                     +---------------------------+
                     |    GitHub Actions CI/CD   |
                     +------------+--------------+
                                  |
                                  v
                        +----------------------+
                        |     Kubernetes       |
                        +----------+-----------+
                                   |
            +----------------------+----------------------+
            |                      |                      |
            v                      v                      v
    Classification API     Semantic Search         RAG API
            |                      |                      |
            +-----------+----------+----------------------+
                        |
                        v
                  +-------------+
                  |   Qdrant    |
                  +------+------+ 
                         |
                         v
                    Journalist
                         |
                         v
                  Human Feedback
                         |
                         v
                +----------------+
                |   Evidently    |
                +--------+-------+
                         |
            Drift        |
                         v
                 +---------------+
                 |    Airflow    |
                 +-------+-------+
                         |
                         v
                   Automatic
                    Retraining
```

# 3. Pipeline Stages

## Stage 1 — Data ingestion

Source:

News Category Dataset
Future RSS feeds
Future newsroom feeds

Output:

data/raw/

## Stage 2 — Data validation

Validation rules:

missing values
duplicated articles
invalid dates
invalid categories
empty headlines

Output:

Validated dataset


## Stage 3 — Preprocessing

Operations:

text normalization
category cleaning
feature engineering
train/validation/test split

Output:

data/processed/

## Stage 4 — Model training

Current model

TF-IDF

↓

Logistic Regression

Future model

Sentence Transformers

↓

Embedding Classifier

Artifacts:

models/

## Stage 5 — Experiment tracking

Tool

MLflow

Stores

parameters
metrics
artifacts
model versions

## Stage 6 — Model Registry

Only validated models are promoted.

Workflow

Training

    ↓

Evaluation

    ↓

Validation

    ↓

Registry

    ↓

Production

Rollback remains possible at any time.


## Stage 7 — Containerization

Each validated model is deployed inside Docker.

The Docker image contains:

FastAPI
trained model
inference code

## Stage 8 — Continuous Integration

GitHub Actions automatically executes

unit tests
API tests
Docker build

Future additions

lint
formatting
security scan

## Stage 9 — Kubernetes Deployment

Target platform

k3s

Resources

Deployment
Service
ConfigMap
Secret
Horizontal Pod Autoscaler

Deployment strategy

Rolling Update

Rollback

kubectl rollout undo

## Stage 10 — Vector Search

Technology

Qdrant

Stores

Article embeddings

Provides

semantic search
duplicate detection
RAG retrieval

## Stage 11 — RAG

Pipeline

Question

    ↓

Embedding

    ↓

Qdrant Retrieval

    ↓

    LLM

    ↓

Grounded Answer

Only retrieved articles are used as context.


## Stage 12 — Monitoring

Technology

Evidently

Monitored metrics

Technical

latency
request count
API errors

ML

prediction distribution
confidence
category drift
embedding drift
text length drift

## Stage 13 — Feedback

Journalists can correct

category
summary
retrieved articles

Feedback is stored for future retraining.


## Stage 14 — Retraining

Triggered by

drift
new labelled data
scheduled retraining

Technology

Apache Airflow

Pipeline

New data

    ↓

Validation

    ↓

Training

    ↓

Evaluation

    ↓

MLflow

    ↓

Deployment

# 4. Versioning

Data

DVC (planned)

Models

MLflow Registry

Code

Git

Containers

Docker

# 5. Production Flow
User

    ↓

FastAPI

    ↓

Classifier

    ↓

Qdrant

    ↓

LLM

↓

Response

    ↓

Monitoring

    ↓

Retraining

# 6. Technologies
Layer	        Technology
API	            FastAPI
ML	            Scikit-learn
Embeddings	    Sentence Transformers
Vector DB	    Qdrant
Tracking	    MLflow
Monitoring	    Evidently
Workflow	    Apache Airflow
Container	    Docker
CI/CD	        GitHub Actions
Orchestration	Kubernetes (k3s)

# 7. Project Status
Component	    Status
Dataset	        ✅
Preprocessing	🚧
Baseline model	🚧
API	            ✅
Docker	        ✅
GitHub Actions	✅
MLflow	        ⏳
Qdrant	        ⏳
RAG	            ⏳
Evidently	    ⏳
Airflow	        ⏳
Kubernetes	    ⏳                    
