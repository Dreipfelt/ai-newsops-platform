# Monitoring Strategy

# 1. Objective

The objective of the monitoring layer is to ensure that the AI NewsOps Platform remains reliable, performant, and accurate after deployment.

Monitoring covers:

- infrastructure health;        
- API performance;      
- machine learning performance;     
- data quality;     
- data drift;       
- embedding drift;      
- automatic retraining triggers.        

Monitoring is designed to detect issues before they impact journalists.

---

# 2. Monitoring Architecture

```text
                FastAPI API
                      |
                      v
             Prediction Requests
                      |
                      v
              Logging & Metrics
                      |
      +---------------+----------------+
      |                                |
      v                                v
 Technical Metrics              ML Metrics
      |                                |
      +---------------+----------------+
                      |
                      v
                 Evidently
                      |
      +---------------+----------------+
      |                                |
      v                                v
 Dashboard                    Drift Alerts
      |                                |
      +---------------+----------------+
                      |
                      v
              Airflow Retraining
```

---

# 3. Technical Monitoring

Technical monitoring ensures that the platform remains operational.

Metrics collected:

| Metric            | Description                   |
|-------------------|-------------------------------|
| API latency       | Response time                 |  
| Request count     | Number of API requests        |
| Error rate        | Failed requests               |
| HTTP status codes | Distribution of responses     |
| CPU usage         | API container utilization     |
| Memory usage      | Container memory consumption  |
| Container health  | Docker/Kubernetes health      |

Example thresholds:

| Metric            | Alert threshold |
|-------------------|-----------------|
| Latency           | > 500 ms        |
| Error rate        | > 5 %           |
| CPU               | > 80 %          |
| Memory            | > 80 %          |

---

# 4. Machine Learning Monitoring

The ML layer monitors model quality over time.

Tracked metrics:

- prediction distribution;      
- confidence distribution;      
- class imbalance;      
- category frequency;       
- prediction entropy.       

Purpose:

Detect unexpected model behaviour before users notice it.

---

# 5. Data Drift Monitoring

Incoming production-like data is continuously compared with the reference dataset.

Drift indicators include:

- category distribution;  
- publication date distribution;  
- article length;  
- vocabulary changes;  
- missing values.  

Example:

```text
Reference Dataset

    ↓

Incoming Articles

    ↓

Statistical Comparison

    ↓

Drift Score
```

---

# 6. Embedding Drift

Because semantic search and RAG rely on embeddings, embedding quality must also be monitored.

Indicators:

- embedding distance distribution;  
- centroid shift;  
- nearest neighbour stability;  
- cosine similarity distribution.  

If embedding drift exceeds the defined threshold, retraining may be scheduled.

---

# 7. Evidently

Evidently is used to generate monitoring reports.

Reports include:

- Data Drift Report  
- Target Drift Report  
- Classification Report  
- Data Quality Report  

Generated reports are archived for auditability.

---

# 8. Human Feedback

Journalists can provide feedback through the API.

Examples:

- incorrect category;  
- poor summary;  
- irrelevant semantic search;  
- incorrect RAG answer.  

Feedback is stored and later incorporated into retraining datasets.

---

# 9. Alert Strategy

Alerts are triggered when predefined thresholds are exceeded.

Examples:

| Event             | Action                 |
|-------------------|------------------------|
| High latency      | Notify administrator   |
| High error rate   | Investigate API        |
| Data drift        | Generate report        |
| Embedding drift   | Schedule retraining    |
| Model degradation | Compare model versions |

---

# 10. Retraining Trigger

Monitoring feeds the retraining pipeline.

```text
Prediction

    ↓

Monitoring

    ↓

Drift Detection

    ↓

Alert

    ↓

Airflow DAG

    ↓

Retraining
```

Retraining is not automatic unless validation criteria are satisfied.

---

# 11. Dashboards

Future dashboards will display:

Technical metrics:

- request volume;  
- latency;  
- error rate;  
- resource usage.  

ML metrics:

- prediction distribution;  
- drift indicators;  
- confidence histogram;  
- category evolution.  

Potential visualization tools:

- Evidently  
- Grafana  
- Prometheus  
  
---

# 12. Monitoring Frequency

| Metric                    | Frequency |
|---------------------------|-----------|
| API health                | Real time |
| Latency                   | Real time |
| Error rate                | Real time |
| Data drift                | Daily     |
| Embedding drift           | Daily     |
| Classification quality    | Weekly    |
| Retraining evaluation     | On demand |

---

# 13. Monitoring Workflow

```text
Incoming Articles
        |
        v
Prediction
        |
        v
Metrics Collection
        |
        +----------------+
        |                |
        v                v
Technical         ML Monitoring
        |                |
        +--------+-------+
                 |
                 v
            Evidently
                 |
        +--------+--------+
        |                 |
        v                 v
Dashboard          Drift Alert
        |                 |
        +--------+--------+
                 |
                 v
            Airflow DAG
                 |
                 v
         Retraining Pipeline
```

---

# 14. Future Improvements

Planned enhancements include:

- Prometheus integration;  
- Grafana dashboards;  
- Slack or email alerts;  
- model-specific dashboards;  
- monitoring of RAG answer quality;  
- monitoring of retrieval precision;  
- monitoring of hallucination rate.  

---

# 15. Current Status

| Component             | Status     |
|-----------------------|------------|
| API Health            |   ✅       |
| Docker Health         |   ✅       |
| CI Monitoring         |   ✅       |
| Evidently             | ⏳ Planned |
| Prometheus            | ⏳ Planned |
| Grafana               | ⏳ Planned |
| Drift Detection       | ⏳ Planned |
| Automatic Alerts      | ⏳ Planned |
| Automatic Retraining  | ⏳ Planned |
