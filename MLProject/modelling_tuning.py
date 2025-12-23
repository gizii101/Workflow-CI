import pandas as pd
import json
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():

    df = pd.read_csv("heart_preprocessing.csv")

    X = df.drop(columns=["HeartDisease"])
    y = df["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "criterion": ["gini", "entropy"]
    }

    model = RandomForestClassifier(random_state=42)

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    os.makedirs("random_forest_model", exist_ok=True)
    joblib.dump(best_model, "random_forest_model/model.pkl")

    with open("performance_report.txt", "w") as f:
        f.write(f"Accuracy: {acc}\nPrecision: {prec}\nRecall: {rec}\nF1: {f1}\n")
        f.write(f"Best Params: {grid.best_params_}\n")

    with open("best_config.json", "w") as f:
        json.dump(grid.best_params_, f, indent=4)


if __name__ == "__main__":
    main()
