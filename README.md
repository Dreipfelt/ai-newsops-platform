# AI NewsOps Platform

> **AI-powered Media Intelligence Platform for News Monitoring, NLP and Editorial Automation**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-green)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![MLflow](https://img.shields.io/badge/MLflow-MLOps-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-13ADC7)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI/CD-2088FF?logo=githubactions)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active%20Development-success)

🚧 This project is actively evolving toward a production-grade RAG system for media intelligence use cases.
---
## Overview

AI NewsOps Platform is an end-to-end NLP + MLOps system for media intelligence.

It is inspired by 20 years of experience in multilingual editorial monitoring and demonstrates how AI can automate classification, retrieval and summarisation of news content.

Although developed as part of the **Architecte Intelligence Artificielle (RNCP38777)** certification, the project is intentionally structured as a production-grade application following modern MLOps principles.

---
## Highlights

- Fine-tuned DistilBERT for news classification (F1: ~0.71)
- End-to-end NLP pipeline from ingestion to inference API
- Production-grade FastAPI microservice (Docker-ready)
- Embedding-based semantic search engine
- RAG pipeline designed for newsroom assistant use cases
- Duplicate detection using vector similarity
- MLflow experiment tracking (20+ runs logged)
- Reproducible training pipeline with DVC
- CI/CD automation with GitHub Actions

---
# Business Context

Newsrooms and media intelligence platforms process thousands of articles every day.

Editors and analysts must quickly:

* classify incoming news
* detect duplicate publications
* retrieve relevant information
* summarize multiple sources
* prepare editorial reports
* monitor emerging topics

AI NewsOps Platform demonstrates how NLP, Machine Learning and Large Language Models can automate these repetitive tasks while maintaining reproducibility, traceability and deployment readiness.

---

# Why this project?

This project is directly inspired by **20 years of professional experience in multilingual media monitoring and editorial intelligence**.

Rather than building a generic NLP demonstration, its objective is to design practical AI services addressing real editorial workflows encountered in news agencies and media intelligence companies.

This unique combination of:

* media expertise
* multilingual text processing
* Data Science
* MLOps
* NLP

is the core value of the project.

---

# Key Features

## Implemented

* Exploratory Data Analysis (EDA)
* Data preprocessing pipeline
* Baseline text classification
* DistilBERT fine-tuning
* Embedding generation
* Duplicate detection
* FastAPI backend
* Streamlit dashboard
* Docker deployment
* MLflow experiment tracking
* DVC dataset versioning
* Unit testing
* GitHub Actions CI/CD
* Technical documentation

---

## Under Active Development

* Retrieval-Augmented Generation (RAG)
* Semantic search
* Retrieval evaluation
* LLM-assisted newsroom workflows
* Monitoring with Evidently
* Automated retraining
* Kubernetes deployment
* Production optimization

---

# Architecture

```text
                  News Sources
                       │
                       ▼
               Data Ingestion
                       │
                       ▼
               Data Preprocessing
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
 Duplicate Detection        News Classification
                                   │
                                   ▼
                        DistilBERT Fine-tuned Model
                                   │
                                   ▼
                        Embedding Generation
                                   │
                                   ▼
                    Semantic Search / RAG
                                   │
                                   ▼
                      LLM Summarization
                                   │
                                   ▼
                           FastAPI Backend
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
             Streamlit Dashboard            REST API
```

---

# Tech Stack

## Programming

* Python 3.10

---

## Machine Learning

* Scikit-learn
* Transformers
* Hugging Face
* DistilBERT
* Sentence Transformers
* NumPy
* Pandas

---

## NLP

* Text Classification
* Embeddings
* Duplicate Detection
* Semantic Search
* Retrieval-Augmented Generation (in progress)
* LLM Integration

---

## MLOps

* MLflow
* Docker
* Docker Compose
* DVC
* GitHub Actions

---

## Backend

* FastAPI
* REST API
* Pydantic

---

## Frontend

* Streamlit

---

# Project Structure

```text
ai-newsops-platform/

├── app/                # FastAPI application & NLP services
├── dashboard/          # Streamlit interface
├── data/               # Raw & processed datasets (DVC)
├── docs/               # Technical documentation
├── models/             # Trained models
├── notebooks/          # Research notebooks
├── reports/            # Evaluation reports & figures
├── src/                # Training & experimentation scripts
├── tests/              # Unit tests
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── README.md
```

---

# Current NLP Pipeline

Current pipeline includes:

* dataset ingestion
* preprocessing
* text normalization
* baseline ML models
* DistilBERT fine-tuning
* embedding generation
* duplicate detection
* API deployment

The architecture has been designed to progressively integrate Retrieval-Augmented Generation (RAG) and semantic retrieval capabilities.

---

# API (planned)

| Method | Endpoint         | Description           |
| ------ | ---------------- | --------------------- |
| GET    | /health          | Health check          |
| POST   | /classify        | News classification   |
| POST   | /summarize       | Article summarization |
| POST   | /semantic-search | Semantic retrieval    |
| POST   | /duplicates      | Duplicate detection   |
| POST   | /rag/query       | RAG assistant         |
| POST   | /feedback        | User feedback         |
| GET    | /metrics         | Monitoring metrics    |

---

# Installation

Clone the repository

```bash
git clone https://github.com/Dreipfelt/ai-newsops-platform.git
cd ai-newsops-platform
```

Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run tests

```bash
pytest
```

or

```bash
make test
```

Run preprocessing

```bash
make preprocess
```

Train the baseline model

```bash
make train
```

Run the API

```bash
make api
```

Open the API documentation

```text
http://localhost:8000/docs
```

Run with Docker

```bash
docker compose up --build
```

---

# Dataset

This project uses the **News Category Dataset v3** by Rishabh Misra.

Source

https://www.kaggle.com/datasets/rmisra/news-category-dataset

License

Creative Commons Attribution 4.0 International (CC BY 4.0)

The dataset is **not distributed** with this repository.

Expected location

```text
data/raw/News_Category_Dataset_v3.json
```

---

# MLOps Scope

The project demonstrates:

* reproducible data preparation
* model training
* model evaluation
* experiment tracking
* API deployment
* Docker packaging
* CI/CD validation
* monitoring
* model versioning
* automated retraining
* deployment readiness

---

# Roadmap

## Short Term

* Improve Retrieval-Augmented Generation pipeline
* Add retrieval evaluation metrics
* Benchmark embedding models
* Improve semantic search

## Mid Term

* LLM orchestration
* Knowledge base
* Vector database integration
* Model optimization

## Long Term

* Production-ready RAG
* Kubernetes deployment
* Continuous evaluation
* On-premise inference
* Multi-agent newsroom workflows

---

# Documentation

Additional documentation is available in the **docs/** directory:

* Architecture
* API
* Deployment
* Monitoring
* Model Card
* Retraining
* MLOps Pipeline
* Technical Decisions
* Dataset Documentation

---

# License

The source code is released under the license provided in the **LICENSE** file.

Dataset licensing information is available in **DATA_LICENSE.md**.

---

## Author

**Frédéric Tellier**

* LinkedIn: https://www.linkedin.com/in/frédéric-tellier-8a9170283/
* Portfolio: https://github.com/Dreipfelt
---

Data Scientist • Data Engineer • NLP • MLOps

Building AI systems for multilingual media intelligence, editorial automation and production-ready NLP.

