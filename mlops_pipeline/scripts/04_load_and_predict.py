import os
import numpy as np
import pandas as pd
import mlflow

# Load the same feature order used in training if available
DEFAULT_FEATURES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "processed_data", "features.csv"
)
DEFAULT_DATA_PATH = os.environ.get(
    "SPAM_DATA_CSV",
    os.path.join(os.path.dirname(__file__), "..", "data", "emails.csv")
)

MODEL_NAME = "email-spam-classifier-prod"
MODEL_STAGE = "Staging"


def _prepare_sample():
    if os.path.exists(DEFAULT_FEATURES_PATH):
        # Load the feature list used in training
        feat_df = pd.read_csv(DEFAULT_FEATURES_PATH)
        features = feat_df["feature"].tolist()
    else:
        raise FileNotFoundError("Feature definition file not found. Please ensure features.csv exists.")

    # Load raw data
    raw = pd.read_csv(DEFAULT_DATA_PATH).fillna(0)

    # Filter and align columns to match the trained model
    raw = raw[[c for c in raw.columns if c in features]]
    sample = raw.reindex(columns=features, fill_value=0)  # Ensure order + missing features = 0
    return sample.iloc[[0]]  # Return first row as DataFrame


def run_client():
    print(f"Loading model '{MODEL_NAME}' from stage '{MODEL_STAGE}'...")

    try:
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    except mlflow.exceptions.MlflowException as e:
        print(f"Error loading model: {e}")
        print("Ensure a model version is in the requested stage in MLflow UI.")
        return

    sample = _prepare_sample()
    pred = model.predict(sample)  # Don't use .values — pass DataFrame directly

    print("-" * 40)
    print("Sample features (first row, aligned to training columns):")
    print(sample.iloc[0].to_dict())
    print(f"Predicted Label: {pred[0]} (1=spam, 0=ham)")
    print("-" * 40)


if __name__ == "__main__":
    run_client()
