# Specification Document — AI NewsOps Platform

## 1. Overview

Client : NewsMedia Inc.
Problem : Manually classifying 10k articles per day is impossible
Solution : Fine-tuned DistilBERT API, 13 categories

## 2. Functional Requirements
Metric	Specification
Latency P95	< 10 ms
Accuracy	≥ 70% F1 macro
Throughput	≥ 100 req/s
Uptime	99.9%

## 3. Technical Specifications
Model : Fine-tuned DistilBERT, 4 epochs
Infrastructure : FastAPI + Docker Compose
Monitoring : Prometheus + Grafana + Evidently AI
Orchestration : Apache Airflow daily retraining

## 4. CI/CD Pipeline
GitHub Actions : lint → test (34 tests) → build → push

## 5. SLA/SLO
Availability : 99.9%
Latency : P95 < 10 ms
Accuracy : F1 ≥ 0.70
Drift Alert : < 30 minutes after detection

## 6. Accessibility (WCAG 2.1)
Streamlit Dashboard : Keyboard navigation, high contrast
Grafana : Tooltips, alt text on charts
API : Simple JSON, no visual dependency

## 7. Compliance
Dataset : Public (no PII)
Audit trail : 12 months log retention
No authentication locally (OAuth2 for production)