# AI NewsOps Platform

AI NewsOps Platform is an MLOps project for a news agency use case.

The platform aims to ingest news articles, classify them, prepare them for semantic search, and expose intelligence services through a FastAPI API.

## Business problem

News agencies receive a high volume of articles and news dispatches every day. Journalists need to quickly classify, search, deduplicate, and summarize this content.

This project demonstrates how an MLOps pipeline can support that workflow with a production-oriented architecture.

## Current features

- Exploratory data analysis on the News Category Dataset v3
- Data preprocessing pipeline
- Baseline text classification model
- FastAPI application skeleton
- Unit tests
- Docker-ready structure
- CI/CD with GitHub Actions
- Dataset license documentation

## Target features

- News classification
- Article summarization
- Semantic search
- Duplicate detection
- RAG assistant for journalists
- Model tracking with MLflow
- Monitoring with Evidently
- Automated retraining
- Kubernetes deployment

## Dataset

This project uses the News Category Dataset v3 by Rishabh Misra.

Dataset source:

https://www.kaggle.com/datasets/rmisra/news-category-dataset

Dataset license:

Creative Commons Attribution 4.0 International (CC BY 4.0)

https://creativecommons.org/licenses/by/4.0/

The dataset is not included in this repository. Users must download it separately from the official Kaggle source.

Expected local path:

```text
data/raw/News_Category_Dataset_v3.json
```

## Project structure
ai-newsops-platform/   
├── app/   
├── src/   
├── tests/   
├── notebooks/   
├── docs/   
├── Dockerfile   
├── docker-compose.yml   
├── Makefile   
├── requirements.txt   
└── README.md   
 
## Installation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Run tests
pytest

or:

make test
Run preprocessing
make preprocess
Train baseline model
make train
Run the API locally
make api

Then open:

http://localhost:8000/docs
Run with Docker
docker build -t ai-newsops-api .
docker run -p 8000:8000 ai-newsops-api

Or:

docker compose up --build
Planned API endpoints
GET  /health
POST /classify
POST /summarize
POST /semantic-search
POST /duplicates
POST /rag/query
POST /feedback
GET  /metrics
MLOps scope

This project demonstrates:

Data preparation
Model training
Model evaluation
API deployment
Docker packaging
CI/CD validation
Monitoring
Model versioning
Automated retraining
Rollback strategy
License

The source code is released under the license provided in LICENSE.

Dataset licensing details are documented in DATA_LICENSE.md.
EOF