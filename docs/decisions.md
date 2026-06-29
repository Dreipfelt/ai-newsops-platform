# Architecture Decision Records (ADR)

## Introduction

This document records the major architectural decisions made during the development of AI NewsOps Platform.

Each decision explains:

* the context;  
* the available alternatives;  
* the selected solution;  
* the rationale behind the choice.  

Recording these decisions improves maintainability and makes future evolution easier.

---

# ADR-001 — FastAPI for Model Serving

## Status

Accepted

## Context

The project requires a lightweight REST API capable of serving machine learning models while automatically generating API documentation.

## Options considered

* Flask  
* Django REST Framework  
* FastAPI  

## Decision

FastAPI was selected.

## Rationale

Advantages:

* native async support;  
* automatic OpenAPI documentation;  
* excellent performance;  
* strong typing with Pydantic;  
* easy integration with Docker.  

## Consequences

The API layer is simple, scalable, and easy to test.

---

# ADR-002 — Scikit-learn for the Baseline Model

## Status

Accepted

## Context

A reliable baseline classifier is required before introducing transformer models.

## Options considered

* Naive Bayes  
* Random Forest  
* Logistic Regression  
* XGBoost  

## Decision

TF-IDF + Logistic Regression.

## Rationale

* fast training;  
* interpretable;  
* widely used NLP baseline;  
* strong benchmark for future transformer models.  

## Consequences

The baseline provides a reference against which future models can be compared.

---

# ADR-003 — Sentence Transformers

## Status

Accepted

## Context

The project aims to support semantic search and Retrieval-Augmented Generation (RAG).

## Options considered

* TF-IDF vectors  
* Word2Vec  
* FastText  
* Sentence Transformers  

## Decision

Sentence Transformers.

## Rationale

Advantages:

* semantic embeddings;  
* state-of-the-art retrieval quality;  
* reusable for classification, search, and RAG;  
* efficient inference.  

## Consequences

A single embedding representation serves multiple platform components.

---

# ADR-004 — Qdrant as Vector Database

## Status

Accepted

## Context

The platform requires efficient similarity search over article embeddings.

## Options considered

* FAISS  
* ChromaDB  
* Weaviate  
* Elasticsearch  
* Qdrant  

## Decision

Qdrant.

## Rationale

Advantages:

* production-ready;  
* REST and gRPC APIs;  
* Docker-native;  
* filtering capabilities;  
* excellent integration with RAG pipelines.  

## Consequences

Qdrant becomes the semantic search engine for the platform.

---

# ADR-005 — MLflow for Experiment Tracking

## Status

Accepted

## Context

Experiments, models, and metrics must be versioned and reproducible.

## Options considered

* DVC only  
* Weights & Biases  
* Neptune  
* MLflow  

## Decision

MLflow.

## Rationale

Advantages:

* open source; 
* experiment tracking; 
* model registry; 
* artifact management; 
* deployment friendly. 

## Consequences

Every model version is traceable and reproducible.

---

# ADR-006 — Evidently for Monitoring

## Status

Accepted

## Context

Production models require continuous monitoring.

## Options considered

* Custom monitoring  
* WhyLabs  
* Arize AI  
* Evidently  

## Decision

Evidently.

## Rationale

Advantages:

* open source;  
* dedicated ML monitoring;  
* drift detection;  
* classification reports;  
* seamless integration into Python pipelines.  

## Consequences

Monitoring becomes reproducible and fully integrated into the MLOps workflow.

---

# ADR-007 — Apache Airflow for Orchestration

## Status

Accepted

## Context

Retraining pipelines must be automated.

## Options considered

* Cron  
* Prefect  
* Kubeflow Pipelines  
* Apache Airflow  

## Decision

Apache Airflow.

## Rationale

Advantages:

* mature scheduler;  
* DAG visualization;  
* retry management;  
* production-ready;  
* extensible.  

## Consequences

Retraining workflows are reproducible and observable.

---

# ADR-008 — Kubernetes for Deployment

## Status

Accepted

## Context

The final platform should demonstrate production-grade deployment.

## Options considered

* Docker only  
* Docker Swarm  
* Nomad  
* Kubernetes  

## Decision

Kubernetes (k3s).

## Rationale

Advantages:

* industry standard;  
* rolling updates;  
* self-healing;  
* autoscaling;  
* service discovery.  

## Consequences

The platform demonstrates enterprise deployment practices while remaining lightweight through k3s.

---

# ADR-009 — GitHub Actions for CI/CD

## Status

Accepted

## Context

Every code change should be automatically validated.

## Options considered

* Jenkins  
* GitLab CI  
* Azure DevOps  
* GitHub Actions  

## Decision

GitHub Actions.

## Rationale

Advantages:

* native GitHub integration;  
* simple YAML workflows;  
* Docker support;  
* free for public repositories.  

## Consequences

Each commit is automatically tested and validated before deployment.

---

# ADR-010 — Modular Repository Structure

## Status

Accepted

## Context

The project combines API development, machine learning, monitoring, documentation, and infrastructure.

## Decision

Adopt a modular repository structure.

```
app/
src/
tests/
docs/
dashboard/
.github/
```

## Rationale

Advantages:

* separation of concerns;  
* easier maintenance;  
* scalability;  
* onboarding of new contributors.  

## Consequences

The repository remains organized as the platform grows.

---

# Future ADRs

Future decisions will document topics such as:

* multilingual support;  
* LLM selection;  
* authentication strategy;  
* cloud deployment;  
* active learning;  
* model explainability.  

