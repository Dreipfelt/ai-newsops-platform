# Model Card

# AI NewsOps Platform – News Classification Model

---

# 1. Model Overview

| Property       | Value                           |
| -------------- | ------------------------------- |
| Model Name     | AI NewsOps News Classifier      |
| Version        | 0.1.0                           |
| Status         | Development                     |
| Task           | Multi-class Text Classification |
| Domain         | News Articles                   |
| Language       | English                         |
| Framework      | Scikit-learn                    |
| Future Version | Sentence Transformers           |

---

# 2. Purpose

The objective of this model is to automatically classify news articles into editorial categories.

The model is intended to assist journalists by reducing manual categorization effort and improving content organization.

Future versions will also support semantic search and Retrieval-Augmented Generation (RAG).

---

# 3. Intended Users

Primary users include:
  
* journalists;  
* editors;  
* newsroom managers;  
* data engineers;  
* machine learning engineers.  

---

# 4. Intended Use Cases

The model supports:

* automatic news categorization;  
* newsroom organization;  
* semantic indexing;  
* duplicate detection;  
* retrieval for RAG systems.  

---

# 5. Out-of-Scope Use

This model is **not** intended for:

* legal decision making;  
* medical diagnosis;  
* political recommendation;  
* misinformation detection;  
* fake news detection.  

---

# 6. Dataset

Training dataset:

News Category Dataset v3

Source:

https://www.kaggle.com/datasets/rmisra/news-category-dataset

License:

CC BY 4.0

Approximate characteristics:

* 210k articles  
* 42 categories  
* English language  
* Huffington Post articles  

---

# 7. Input

Expected input:

```json
{
    "headline": "...",
    "short_description": "..."
}
```

The inference pipeline combines both fields into a single text representation.

---

# 8. Output

Example:

```json
{
    "prediction": "BUSINESS",
    "confidence": 0.94
}
```

Future versions may also return:

* top-3 predictions;  
* confidence distribution;  
* retrieved semantic neighbours.  

---

# 9. Training Pipeline

Training stages:

1. Data validation  
2. Text preprocessing  
3. Feature extraction  
4. Model training  
5. Evaluation  
6. Registration in MLflow  

---

# 10. Baseline Model

Current approach:

TF-IDF

↓

Logistic Regression

Reason:

* fast training;  
* interpretable;  
* strong NLP baseline.  

---

# 11. Future Model

Planned architecture:

Sentence Transformer

    ↓

Dense Embeddings

    ↓

Embedding Classifier

Advantages:

* semantic understanding;  
* better retrieval;  
* RAG compatibility.  

---

# 12. Evaluation Metrics

The following metrics are reported:

* Accuracy  
* Precision  
* Recall  
* F1-score  
* Confusion Matrix  

Future versions will include:

* calibration curves;  
* confidence analysis;  
* per-category performance.  

---

# 13. Monitoring

The model is monitored in production using Evidently.

Indicators include:

* prediction drift;  
* category drift;  
* embedding drift;  
* latency;  
* confidence evolution.  

---

# 14. Versioning

Model versions are managed using MLflow.

Each version stores:

* training dataset;  
* preprocessing version;  
* hyperparameters;  
* metrics;  
* artifacts;  
* training timestamp.  

---

# 15. Known Limitations

Current limitations include:

* English language only;  
* historical news data;  
* headline and short description only;  
* no full article body;  
* limited journalist feedback.  

---

# 16. Potential Biases

Potential sources of bias:

* editorial choices from a single publisher;  
* historical category imbalance;  
* temporal evolution of topics;  
* language-specific vocabulary.  

These biases should be considered when interpreting predictions.

---

# 17. Ethical Considerations

The model is designed to assist journalists, not replace editorial judgment.

Final publication decisions remain under human responsibility.

Human validation is encouraged for critical content.

---

# 18. Future Improvements

Planned enhancements:

* multilingual support;  
* larger transformer models;  
* retrieval-augmented classification;  
* active learning from journalist feedback;  
* continual learning.  

---

# 19. Ownership

Project:

AI NewsOps Platform

Repository:

GitHub

Maintainer:

Frédéric Tellier

Certification:

AIA – Block 4

---

# 20. Model Lifecycle

```text
Dataset

    ↓

Preprocessing

    ↓

Training

    ↓

Evaluation

    ↓

MLflow Registry

    ↓

Docker

    ↓

FastAPI

    ↓

Monitoring

    ↓

Retraining
```
