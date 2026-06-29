# Deployment Guide

# 1. Overview

AI NewsOps Platform is designed to be deployed incrementally, following modern MLOps practices.

The deployment strategy progresses through four stages:

1. Local development  
2. Docker containerization  
3. Continuous Integration / Continuous Deployment  
4. Kubernetes deployment  

This approach ensures reproducibility, scalability, and maintainability.

---

# 2. Local Development

The project can be executed locally using a Python virtual environment.

## Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run tests

```bash
make test
```

## Start the API

```bash
make api
```

Swagger UI

```text
http://localhost:8000/docs
```

---

# 3. Docker Deployment

The API is containerized using Docker.

Build the image

```bash
docker build -t ai-newsops-api .
```

Run the container

```bash
docker run -p 8000:8000 ai-newsops-api
```

Health check

```bash
curl http://localhost:8000/health
```

Expected response

```json
{
    "status":"ok",
    "service":"ai-newsops-api",
    "version":"0.1.0"
}
```

---

# 4. Docker Compose

For local orchestration the project uses Docker Compose.

Start

```bash
docker compose up --build
```

Stop

```bash
docker compose down
```

Current services

```text
FastAPI API
```

Future services

```text
FastAPI
MLflow
Qdrant
Evidently
Airflow
Grafana
Prometheus
```

---

# 5. CI/CD Pipeline

Continuous Integration is provided by GitHub Actions.

Each push triggers:

```text
Git Push

    ↓

Checkout

    ↓

Install dependencies

    ↓

Run tests

    ↓

Build Docker image

    ↓

Success / Failure
```

Future improvements:

- Ruff  
- Black  
- Bandit  
- Security scanning  
- Automatic deployment  

---

# 6. MLflow Deployment

The future MLflow server will be deployed as a dedicated service.

Responsibilities:

- experiment tracking  
- model registry  
- artifact storage  
- model versioning  

Deployment mode:

Docker

Production:

Kubernetes Deployment

---

# 7. Qdrant Deployment

Qdrant stores article embeddings.

Responsibilities:

- semantic search  
- duplicate detection  
- RAG retrieval  

Deployment:

Docker

Production:

Kubernetes

Persistent Volume enabled.

---

# 8. Evidently Deployment

Evidently continuously evaluates model quality.

Monitored metrics:

- prediction drift  
- category drift  
- embedding drift  
- text statistics  

Deployment:

Dedicated monitoring service.

---

# 9. Airflow Deployment

Apache Airflow orchestrates retraining.

Scheduled DAGs:

- preprocessing  
- training  
- evaluation  
- model registration  

Future DAGs:

```text
daily_retraining

weekly_retraining

drift_retraining
```

---

# 10. Kubernetes Deployment

Target cluster:

k3s

Main resources:

```text
Namespace

Deployment

Service

Ingress

ConfigMap

Secret

PersistentVolumeClaim
```

Application layout

```text
Ingress

    ↓

FastAPI

    ↓

MLflow

    ↓

Qdrant

    ↓

Monitoring
```

Scaling

Horizontal Pod Autoscaler

Rolling updates enabled.

---

# 11. Rollback Strategy

Rollback exists at multiple levels.

Application

```bash
kubectl rollout undo deployment/ai-newsops-api
```

Model

MLflow Registry

Container

Previous Docker image tag

---

# 12. Deployment Strategy

Current state

```text
Developer

↓

Git

↓

GitHub

↓

GitHub Actions

↓

Docker

↓

Local execution
```

Target production architecture

```text
Developer

↓

GitHub

↓

GitHub Actions

↓

Docker Registry

↓

Kubernetes

↓

FastAPI

↓

MLflow

↓

Qdrant

↓

Evidently

↓

Airflow

↓

Journalists
```

---

# 13. Security

Planned security measures

- Secrets stored in Kubernetes Secrets      
- Environment variables     
- HTTPS ingress     
- Non-root Docker containers        
- Image vulnerability scanning      
- API authentication        

---

# 14. Disaster Recovery

Recovery strategy

- Git repository        
- Docker images     
- MLflow Registry       
- Qdrant Persistent Volumes     

Rollback can be performed without retraining the model.

---

# 15. Deployment Roadmap

| Stage             | Status |
|-------------------|--------|
| FastAPI           |   ✅   |
| Docker            |   ✅   |
| Docker Compose    |   ✅   |
| GitHub Actions    |   ✅   |
| MLflow            |   ⏳   |
| Qdrant            |   ⏳   |
| Evidently         |   ⏳   |
| Airflow           |   ⏳   |
| Kubernetes        |   ⏳   |