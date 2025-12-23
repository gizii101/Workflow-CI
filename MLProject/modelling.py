import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("CI_Heart_Disease_Training")

    # Autolog cukup, JANGAN start_run
    mlflow.sklearn.autolog()

    df = pd.read_csv("heart_preprocessing.csv")
    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("classification_report.txt")

    print("CI training selesai. Accuracy:", acc)


if __name__ == "__main__":
    main()
