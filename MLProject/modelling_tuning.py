import pandas as pd
import mlflow
import mlflow.sklearn
import json
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():

    # =========================================
    # 1. MLFLOW EXPERIMENT (JANGAN set_tracking_uri)
    # =========================================
    mlflow.set_experiment("Heart_Disease_Classification")

    # =========================================
    # 2. LOAD DATA
    # =========================================
    df = pd.read_csv("heart_preprocessing.csv")

    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # =========================================
    # 3. GRID SEARCH
    # =========================================
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "criterion": ["gini", "entropy"]
    }

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    print("Training + Hyperparameter Tuning dimulai...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # =====================================
    # 4. METRICS
    # =====================================
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # =====================================
    # 5. SAVE MODEL KE FILESYSTEM
    # =====================================
    os.makedirs("random_forest_model", exist_ok=True)
    joblib.dump(best_model, "random_forest_model/model.pkl")

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="random_forest_model"
    )

    # =====================================
    # 6. EXTRA ARTIFACTS
    # =====================================
    with open("performance_report.txt", "w") as f:
        f.write("=== Heart Disease Classification Report ===\n")
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"Precision : {prec:.4f}\n")
        f.write(f"Recall    : {rec:.4f}\n")
        f.write(f"F1-Score  : {f1:.4f}\n")
        f.write(f"Best Params: {grid_search.best_params_}\n")

    mlflow.log_artifact("performance_report.txt")

    with open("best_config.json", "w") as f:
        json.dump(grid_search.best_params_, f, indent=4)

    mlflow.log_artifact("best_config.json")

    print("SELESAI. Accuracy:", acc)


if __name__ == "__main__":
    main()
