import mlflow 
import pandas as pd 
from collections import Counter 
import os 
import pandas as pd 
from typing import Tuple, List 
DEFAULT_DATA_PATH = os.environ.get("SPAM_DATA_CSV",
os.path.join(os.path.dirname(__file__), "..", "data", "emails.csv")) 

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
        if any(k in lower for k in ["email", "name", "no", "id"]):  drop_cols.append(col) 
    return df.drop(columns=drop_cols, errors="ignore") 

def load_kaggle_spam_dataframe(csv_path: str = None) -> Tuple[pd.DataFrame, str]: 
    csv_path = csv_path or DEFAULT_DATA_PATH 
    df = pd.read_csv(csv_path) 
    target_col = _guess_target_column(df) 
    df = _drop_identifier_columns(df, target_col) 
    # ensure target is the last column for convenience  cols = [c for c in df.columns if c != target_col] + [target_col]  df = df[cols] 
    return df, target_col 
def split_X_y(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col]) 
    y = df[target_col] 
    return X, y 

def validate_data(csv_path: str = None): 
    mlflow.set_experiment("Email Spam - Data Validation")  
    with mlflow.start_run(): 
        print("Starting data validation run...") 
        mlflow.set_tag("ml.step", "data_validation") 
    df, target_col = load_kaggle_spam_dataframe(csv_path)
    n_rows, n_cols = df.shape 
    missing = int(df.isnull().sum().sum()) 
    class_counts = Counter(df[target_col].values.tolist())
    n_classes = len(class_counts) 
   
    print(f"Data path: {csv_path or DEFAULT_DATA_PATH}")  
    print(f"Target column: {target_col}") 
    print(f"Shape: {n_rows} x {n_cols}") 
    print(f"Missing values: {missing}") 
    print(f"Class distribution: {class_counts}") 
    
    mlflow.log_param("target_column", target_col) 
    mlflow.log_metric("num_rows", n_rows) 
    mlflow.log_metric("num_cols", n_cols) 
    mlflow.log_metric("missing_values", missing) 
    mlflow.log_param("class_counts", str(class_counts)) 
    mlflow.log_param("n_classes", n_classes) 
    
    status = "Success" if (missing == 0 and n_classes == 2) else "Check" 
    mlflow.log_param("validation_status", status)  
    print(f"Validation status: {status}") 
    print("Data validation run finished.") 
if __name__ == "__main__": 
    validate_data()
