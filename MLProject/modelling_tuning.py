import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    # PENTING: jangan pakai start_run()
    mlflow.set_experiment("Heart_Disease_Classification")

    # Data HARUS relatif ke MLProject/
    df = pd.read_csv("heart_preprocessing.csv")

    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)

    # ⬇️ INI KUNCI UTAMA
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

if __name__ == "__main__":
    main()
