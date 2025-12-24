import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("CI_Heart_Disease")

    mlflow.sklearn.autolog(log_models=False)

    df = pd.read_csv("heart_preprocessing.csv")
    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", acc)

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")

    # 🔥 INI PENTING
    mlflow.sklearn.log_model(model, artifact_path="model")

    print("Training selesai. Accuracy:", acc)

if __name__ == "__main__":
    main()
