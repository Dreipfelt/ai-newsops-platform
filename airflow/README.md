# Airflow retraining DAG

Jour 7 ajoute une orchestration Airflow locale pour le pipeline de retraining.

## DAG

`airflow/dags/retraining_dag.py` exécute :

1. validation des datasets `data/processed` ;
2. lecture du dernier rapport de drift ;
3. décision de retraining ;
4. entraînement candidat ;
5. enregistrement/comparaison MLflow.

Le DAG est volontairement fin : Airflow orchestre, les scripts Python portent la logique métier.

## Lancer localement

```bash
docker compose --profile airflow up airflow-init
docker compose --profile airflow up airflow-webserver airflow-scheduler
```

Interface : <http://localhost:8080>

Identifiants par défaut : `airflow` / `airflow`.

## Variables utiles

- `DRIFT_THRESHOLD` : seuil de drift, défaut `0.30`
- `MIN_CANDIDATE_F1` : F1 minimale attendue, défaut `0.65`
- `MLFLOW_TRACKING_URI` : URI MLflow, défaut `http://mlflow:5000`
- `RETRAIN_COMMAND` : commande d'entraînement, défaut `python src/train_baseline.py`
