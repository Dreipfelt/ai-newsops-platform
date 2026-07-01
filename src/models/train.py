import pandas as pd
import mlflow
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LogisticRegression,
)  # Changed from RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris  # Import iris dataset

if __name__ == "__main__":
    experiment_name = "baby-ml-model"
    mlflow.set_experiment(experiment_name)

    print("Training a simple Logistic Regression model...")

    start_time = time.time()

    mlflow.sklearn.autolog()

    print("Loading Iris dataset...")
    iris = load_iris()
    X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    y = pd.Series(data=iris.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Use a single run context
    with mlflow.start_run() as run:
        model = Pipeline(
            steps=[
                ("standard_scaler", StandardScaler()),
                ("logistic_regression", LogisticRegression()),
            ]
        )

        model.fit(X_train, y_train)

    print("...Training and logging complete!")
    print(f"---Total execution time: {time.time()-start_time:.4f} seconds")
