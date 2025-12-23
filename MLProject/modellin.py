import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def main():
    # =========================================
    # 1. SET TRACKING URI (LOCAL FILE)
    # =========================================
    mlflow.set_tracking_uri("file:./mlruns")

    # =========================================
    # 2. SET EXPERIMENT NAME
    # =========================================
    mlflow.set_experiment("CI_Heart_Disease_Training")

    # Aktifkan autolog (sesuai CI + Basic)
    mlflow.sklearn.autolog()

    # =========================================
    # 3. LOAD DATASET
    # =========================================
    df = pd.read_csv("heart_preprocessing.csv")

    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]

    # =========================================
    # 4. TRAIN TEST SPLIT
    # =========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================================
    # 5. START MLFLOW RUN
    # =========================================
    with mlflow.start_run(run_name="RF_HeartDisease_Baseline"):

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        # =====================================
        # 6. EVALUATION
        # =====================================
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Simpan classification report sebagai artifact
        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)

        mlflow.log_artifact("classification_report.txt")

        print("-" * 40)
        print("CI Training Selesai")
        print("Accuracy:", acc)
        print("-" * 40)


if __name__ == "__main__":
    main()
