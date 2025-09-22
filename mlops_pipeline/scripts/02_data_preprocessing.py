import os
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List

DEFAULT_DATA_PATH = os.environ.get(
    "SPAM_DATA_CSV",
    os.path.join(os.path.dirname(__file__), "..", "data", "emails.csv")
)


def _guess_target_column(df: pd.DataFrame) -> str:
    # try common target names
    candidates = ["Prediction", "prediction", "Class", "class", "Label", "label", "Spam", "spam", "target"]
    for c in candidates:
        if c in df.columns:
            return c
    # else use last column
    return df.columns[-1]


def _drop_identifier_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    # drop non-numeric columns except target
    drop_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            drop_cols.append(col)
            continue
        # heuristic by name
        lower = str(col).lower()
        if any(k in lower for k in ["email", "name", "no", "id"]):
            drop_cols.append(col)
    return df.drop(columns=drop_cols, errors="ignore")


def load_kaggle_spam_dataframe(csv_path: str = None) -> Tuple[pd.DataFrame, str]:
    csv_path = csv_path or DEFAULT_DATA_PATH
    df = pd.read_csv(csv_path)
    target_col = _guess_target_column(df)
    df = _drop_identifier_columns(df, target_col)
    # ensure target is the last column for convenience
    cols = [c for c in df.columns if c != target_col] + [target_col]
    df = df[cols]
    return df, target_col


def split_X_y(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def preprocess_data(test_size=0.25, random_state=42, csv_path: str = None):
    mlflow.set_experiment("Email Spam - Data Preprocessing CI CI")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting data preprocessing run: {run_id}")
        mlflow.set_tag("ml.step", "data_preprocessing")

        df, target_col = load_kaggle_spam_dataframe(csv_path)

        # fill missing numeric with 0
        df = df.fillna(0)

        X, y = split_X_y(df, target_col)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        out_dir = "processed_data"
        os.makedirs(out_dir, exist_ok=True)

        # store columns used (to help client scripts reproduce feature order)
        pd.DataFrame({"feature": X.columns.tolist()}).to_csv(
            os.path.join(out_dir, "features.csv"), index=False
        )

        pd.concat([X_train, y_train], axis=1).to_csv(
            os.path.join(out_dir, "train.csv"), index=False
        )
        pd.concat([X_test, y_test], axis=1).to_csv(
            os.path.join(out_dir, "test.csv"), index=False
        )

        mlflow.log_param("target_column", target_col)
        mlflow.log_param("test_size", test_size)
        mlflow.log_metric("train_rows", len(X_train))
        mlflow.log_metric("test_rows", len(X_test))
        mlflow.log_artifacts(out_dir, artifact_path="processed_data")

        print("-" * 60)
        print("Saved processed_data/ (train.csv, test.csv, features.csv) and logged as artifacts.")
        print(f"Preprocessing Run ID: {run_id}")
        print("-" * 60)


if __name__ == "__main__":
    preprocess_data()
