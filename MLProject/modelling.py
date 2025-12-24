import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    # MLflow local tracking (WAJIB SAMA CI)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("CI_Heart_Disease")

    # Autolog metric saja
    mlflow.sklearn.autolog(log_models=False)

    # Load data
    df = pd.read_csv("heart_preprocessing.csv")
    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", acc)

    # 🔥 INI SATU-SATUNYA MODEL YANG DIPAKAI DOCKER
    mlflow.sklearn.log_model(
        model,
        artifact_path="random_forest_model"
    )

    print("Training selesai. Accuracy:", acc)


if __name__ == "__main__":
    main()
