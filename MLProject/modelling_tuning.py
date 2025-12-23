import pandas as pd
import mlflow
import mlflow.sklearn
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():

    # =========================================
    # 1. DAGSHUB + MLFLOW CONFIG
    # =========================================
    mlflow.set_tracking_uri(
        "https://dagshub.com/m004d5y1199/Heart-Disease-Classification-Analysis.mlflow"
    )
    mlflow.set_experiment("Heart_Disease_Classification")

    # =========================================
    # 2. LOAD PREPROCESSED DATASET
    # =========================================
    data_path = "heart_preprocessing.csv"

    df = pd.read_csv(data_path)

    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================================
    # 3. AUTOLOG (MODEL DIMATIKAN)
    # =========================================
    mlflow.sklearn.autolog(log_models=False)

    # =========================================
    # 4. PARAMETER GRID
    # =========================================
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "criterion": ["gini", "entropy"]
    }

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    # =========================================
    # 5. START RUN
    # =========================================
    with mlflow.start_run(run_name="RF_HeartDisease_Tuning"):

        print("Training + Hyperparameter Tuning dimulai...")
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # =====================================
        # 6. MANUAL METRICS
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
        # 7. LOG MODEL (MLmodel, model.pkl, dll)
        # =====================================
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="random_forest_model"
        )

        # =====================================
        # 8. EXTRA ARTEFACTS
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

        print("-" * 40)
        print("TRAINING & LOGGING SELESAI (DAGSHUB)")
        print(f"Accuracy: {acc:.4f}")
        print("-" * 40)


if __name__ == "__main__":
    main()