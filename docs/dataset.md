# Dataset Documentation

## 1. Overview

AI NewsOps Platform is built on the **News Category Dataset v3** created by **Rishabh Misra**.

The dataset contains news articles collected from the Huffington Post over several years and is intended for text classification research.

For this project, the dataset also serves as the basis for:

- semantic search;  
- duplicate detection;  
- retrieval-augmented generation (RAG);  
- monitoring and drift detection;  
- automated retraining.  

---

## 2. Dataset source

Dataset:

https://www.kaggle.com/datasets/rmisra/news-category-dataset

Original publication:

https://arxiv.org/abs/2209.11429

License:

Creative Commons Attribution 4.0 International (CC BY 4.0)

The dataset is **not redistributed** in this repository.

---

## 3. Dataset characteristics

Approximate statistics:

| Property      | Value         |
|---------------|---------------|
| Articles      | ~210,000      |
| Categories    | 42            |
| Format        | JSON Lines    |
| Language      | English       |

Each record contains:

| Field             | Description           |
|-------------------|-----------------------|
| category          | News category         |
| headline          | Article title         |
| short_description | Short summary         |
| authors           | Article author(s)     |
| date              | Publication date      |
| link              | Original article URL  |

---

## 4. Local storage

The dataset should be downloaded manually.

Expected location:

```text
data/raw/News_Category_Dataset_v3.json
```

## 5. Dataset lifecycle
Kaggle
   |
   v
data/raw
   |
   v
Validation
   |
   v
Preprocessing
   |
   v
Processed dataset
   |
   +----------------+
   |                |
   v                v
Classifier      Embeddings

## 6. Data validation

Validation checks include:

missing values;
duplicate rows;
invalid publication dates;
empty headlines;
empty descriptions;
category consistency.

Rows failing validation may be removed or corrected during preprocessing.

## 7. Preprocessing pipeline

The preprocessing stage performs:

Missing values
remove empty headlines;
replace missing descriptions with an empty string.
Text normalization
lowercase conversion;
whitespace normalization;
removal of duplicated spaces.
Feature engineering

The project creates additional features such as:

article length;
headline length;
publication year;
publication month;
combined text field:
headline + short_description

## 8. Category management

The original dataset contains 42 editorial categories.

Some categories are highly represented while others contain relatively few samples.

Two strategies are evaluated:

Strategy 1

Keep the original taxonomy.

Advantages:

preserves editorial granularity.

Disadvantages:

strong class imbalance.

Strategy 2

Merge similar categories.

Example:

ARTS
ARTS & CULTURE
↓

ARTS

Purpose:

reduce imbalance;
improve classifier robustness.

## 9. Data split

Training data is split into:

Training
80%

Validation
10%

Test
10%

The split is stratified to preserve class proportions.

## 10. Embeddings

Each article will be transformed into a dense embedding using a Sentence Transformer model.

Pipeline:

Article

↓

Sentence Transformer

↓

Embedding

↓

Qdrant

Embeddings will be reused for:

semantic search;
duplicate detection;
RAG retrieval.

## 11. Monitoring dataset

A reference dataset will be stored after training.

Future production-like data will be compared against this reference to detect:

category drift;
embedding drift;
text length drift;
vocabulary drift.

Monitoring will be performed with Evidently.

## 12. Retraining

Retraining may be triggered by:

significant drift;
new labeled articles;
journalist feedback;
scheduled retraining.

The retraining pipeline is planned to run with Apache Airflow.

## 13. Limitations

The dataset presents several limitations:

English only;
historical news;
no journalist feedback labels;
no article body (headline + short description only).

Despite these limitations, it is well suited for demonstrating a production-oriented NLP MLOps pipeline.
