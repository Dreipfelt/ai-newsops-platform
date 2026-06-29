# Retraining Strategy

# 1. Objective

The objective of the retraining pipeline is to ensure that the production model remains accurate as new articles, editorial topics, and language evolve over time.

Rather than retraining on a fixed schedule only, AI NewsOps Platform combines scheduled retraining with event-driven retraining triggered by monitoring signals.

---

# 2. Retraining Architecture

```text
                New Articles
                      |
                      v
              Data Validation
                      |
                      v
             Data Preprocessing
                      |
                      v
              Feature Engineering
                      |
                      v
               Model Training
                      |
                      v
              Model Evaluation
                      |
          +-----------+-----------+
          |                       |
          v                       v
      Candidate Worse        Candidate Better
          |                       |
          |                  MLflow Registry
          |                       |
          |                       v
          |               Production Model
          |                       |
          +-----------+-----------+
                      |
                      v
                Deployment
```

---

# 3. Retraining Triggers

The retraining pipeline can be started for several reasons.

## Scheduled retraining

Example frequencies:

- daily  
- weekly  
- monthly  
  
Used to continuously refresh the model.

---

## Data drift

Detected by Evidently.

Examples:

- category distribution changes;  
- vocabulary evolution;  
- article length changes;  
- publication pattern changes.  

---

## Performance degradation

Retraining is initiated if:

- accuracy decreases;  
- F1-score decreases;  
- precision decreases;  
- recall decreases.  

Thresholds are configurable.

---

## Human feedback

Journalists may submit corrections through the API.

Example:

```text
Predicted:
BUSINESS

Correct:
TECH
```

Corrected examples are added to the next training dataset.

---

# 4. Retraining Workflow

```text
Reference Dataset
        |
        v
New Articles
        |
        v
Validation
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
        +-------------------+
        |                   |
        v                   v
Reject              Register in MLflow
                            |
                            v
                     Deploy Candidate
```

---

# 5. Training Pipeline

Each retraining execution performs:

1. Load new data  
2. Validate data  
3. Clean text  
4. Build features  
5. Train candidate model  
6. Evaluate performance  
7. Compare with production model  
8. Register model if improved  
9. Deploy automatically (optional)  

---

# 6. Evaluation Metrics

The following metrics are computed for every model.

| Metric            | Purpose                |
|-------------------|------------------------|
| Accuracy          | Overall performance    |
| Precision         | False positive control |
| Recall            | False negative control |
| F1-score          | Balanced evaluation    |
| Confusion Matrix  | Error analysis         |

The new model must satisfy predefined acceptance criteria before deployment.

---

# 7. Model Comparison

MLflow stores all candidate models.

Comparison includes:

- metrics;  
- parameters;  
- artifacts;  
- training date;  
- dataset version.  

Only the best validated model becomes the production version.

---

# 8. Model Registry

MLflow Registry maintains:

- Development  
- Staging  
- Production  
- Archived  

Example lifecycle:

```text
Training

    ↓

Registered

    ↓

Staging

    ↓

Production

    ↓

Archived
```

---

# 9. Rollback

Rollback is possible if:

- production errors increase;  
- monitoring detects severe drift;  
- journalists report poor predictions.  

Rollback methods:

- previous MLflow version; 
- previous Docker image; 
- Kubernetes rollout undo. 

---

# 10. Airflow Orchestration

Airflow coordinates the retraining workflow.

Example DAG:

```text
Start

    ↓

Load Dataset

    ↓

Validation

    ↓

Preprocessing

    ↓

Training

    ↓   

Evaluation

    ↓

Register Model

    ↓

Deploy

    ↓

End
```

Future DAGs:

- scheduled_retraining  
- drift_retraining  
- feedback_retraining  

---

# 11. Dataset Versioning

Each training run records:

- dataset version;  
- preprocessing version;  
- feature version;  
- model version.  

Future implementation:

DVC.

---

# 12. Reproducibility

Each experiment stores:

- random seed;  
- hyperparameters;  
- Git commit hash;  
- dataset version;  
- Python package versions.  

This guarantees reproducible experiments.

---

# 13. Safety Checks

Deployment occurs only if:

- training completed successfully;  
- evaluation completed successfully;  
- candidate model outperforms production;  
- API tests succeed;  
- Docker image builds successfully.  

Otherwise the candidate model is rejected.

---

# 14. Future Improvements

Planned enhancements include:

- automatic hyperparameter optimization;  
- shadow deployment;  
- canary deployment;  
- A/B testing;  
- reinforcement from journalist feedback.  

---

# 15. Current Status

| Component             | Status     |
|-----------------------|------------|
| Baseline Training     | 🚧         |
| MLflow Tracking       | ⏳ Planned |
| Airflow DAG           | ⏳ Planned |
| Automatic Retraining  | ⏳ Planned |
| Rollback              | ⏳ Planned |
| Dataset Versioning    | ⏳ Planned |
