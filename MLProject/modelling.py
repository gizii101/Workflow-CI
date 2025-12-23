import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")

def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("CI_Heart_Disease")

    # Autolog BOLEH, tapi matikan log model
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

    # 🔥 INI YANG WAJIB ADA
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_model"
    )

    print("Training selesai. Accuracy:", acc)

if __name__ == "__main__":
    main()
