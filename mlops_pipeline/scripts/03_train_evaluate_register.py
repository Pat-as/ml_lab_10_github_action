import sys
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from mlflow.artifacts import download_artifacts

ACCURACY_THRESHOLD = 0.90


def train_evaluate_register(preprocessing_run_id: str, C: float = 1.0):
    mlflow.set_experiment("Email Spam - Train/Evaluate/Register CI")

    with mlflow.start_run(run_name=f"logreg_C_{C}"):
        print(f"Starting training (C={C})...")
        mlflow.set_tag("ml.step", "model_training_evaluation")
        mlflow.log_param("preprocessing_run_id", preprocessing_run_id)

        local_artifacts = download_artifacts(
            run_id=preprocessing_run_id,
            artifact_path="processed_data"
        )

        train_df = pd.read_csv(os.path.join(local_artifacts, "train.csv"))
        test_df = pd.read_csv(os.path.join(local_artifacts, "test.csv"))

        X_train = train_df.drop(columns=[train_df.columns[-1]])
        y_train = train_df.iloc[:, -1]
        X_test = test_df.drop(columns=[test_df.columns[-1]])
        y_test = test_df.iloc[:, -1]

        pipe = Pipeline([
            ("clf", LogisticRegression(C=C, max_iter=2000, solver="liblinear"))
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)

        mlflow.sklearn.log_model(pipe, "spam_classifier_pipeline")

        if acc >= ACCURACY_THRESHOLD:
            print("Threshold met. Registering model...")
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/spam_classifier_pipeline"
            reg = mlflow.register_model(model_uri, "email-spam-classifier-prod")
            print(f"Registered model: {reg.name}, version {reg.version}")
        else:
            print("Threshold not met; skip registering.")

        print("Training finished.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/03_train_evaluate_register.py <preprocessing_run_id> [C]")
        sys.exit(1)

    run_id = sys.argv[1]
    C = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    train_evaluate_register(run_id, C)
