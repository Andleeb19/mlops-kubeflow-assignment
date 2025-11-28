"""
Task 2: Kubeflow Pipeline Components
These are component definitions for Kubeflow Pipelines.
Note: For Task 3, we use MLflow instead of Kubeflow for orchestration.
"""

from pathlib import Path


def data_extraction(repo_url: str, data_path: str, rev: str) -> str:
    """
    Fetch the DVC tracked dataset from the remote Git repository.
    
    Inputs:
    - repo_url: Git repository URL
    - data_path: Path to data file in repo
    - rev: Git revision/branch
    
    Outputs:
    - output_csv: Path to extracted CSV file
    """
    import os
    import subprocess
    import tempfile
    import shutil

    tmp_dir = tempfile.mkdtemp()
    try:
        dest_path = os.path.join(tmp_dir, "dataset.csv")
        subprocess.run(
            [
                "dvc",
                "get",
                repo_url,
                data_path,
                "--rev",
                rev,
                "--out",
                dest_path,
            ],
            check=True,
        )
        return dest_path
    finally:
        pass  # Keep file for now


def data_preprocessing(raw_data_path: str) -> tuple:
    """
    Clean and split the dataset.
    
    Inputs:
    - raw_data: Path to raw CSV file
    
    Outputs:
    - train_output: Path to training data CSV
    - test_output: Path to test data CSV
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from pathlib import Path
    import mlflow

    # Initialize MLflow
    mlflow.set_tracking_uri("file:///tmp/mlruns")
    mlflow.set_experiment("boston-housing-mlops")
    
    with mlflow.start_run(run_name="data_preprocessing", nested=True):
        df = pd.read_csv(raw_data_path).dropna()
        target_column = "TARGET"

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
        train_df[target_column] = y_train.reset_index(drop=True)
        test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
        test_df[target_column] = y_test.reset_index(drop=True)

        # Save to artifacts directory
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        
        train_path = str(artifacts_dir / "train_data.csv")
        test_path = str(artifacts_dir / "test_data.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Log preprocessing metrics to MLflow
        mlflow.log_metric("train_samples", len(train_df))
        mlflow.log_metric("test_samples", len(test_df))
        mlflow.log_metric("features_count", len(X.columns))
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        
        return train_path, test_path


def model_training(train_data_path: str) -> str:
    """
    Train a Random Forest model using MLflow.
    
    Inputs:
    - train_data: Path to training data CSV
    
    Outputs:
    - model_output: Path to saved model file
    """
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from pathlib import Path
    import mlflow
    import mlflow.sklearn

    # Initialize MLflow
    mlflow.set_tracking_uri("file:///tmp/mlruns")
    mlflow.set_experiment("boston-housing-mlops")
    
    with mlflow.start_run(run_name="model_training", nested=True):
        df = pd.read_csv(train_data_path)
        y = df["TARGET"]
        X = df.drop(columns=["TARGET"])

        # Model parameters
        n_estimators = 200
        random_state = 42
        
        model = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=random_state, 
            n_jobs=-1
        )
        model.fit(X, y)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("model_type", "RandomForestRegressor")
        
        # Log training metrics
        train_score = model.score(X, y)
        mlflow.log_metric("train_r2_score", train_score)
        
        # Save model artifact
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        model_path = str(artifacts_dir / "model.joblib")
        joblib.dump(model, model_path)
        
        # Log the saved model path
        mlflow.log_artifact(model_path, "model_artifact")
        
        return model_path


def model_evaluation(test_data_path: str, model_path: str) -> str:
    """
    Evaluate the trained model and persist metrics using MLflow.
    
    Inputs:
    - test_data: Path to test data CSV
    - model_artifact: Path to model file
    
    Outputs:
    - metrics_output: Path to metrics JSON file
    """
    import json
    import joblib
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    from pathlib import Path
    import mlflow
    import mlflow.sklearn

    # Initialize MLflow
    mlflow.set_tracking_uri("file:///tmp/mlruns")
    mlflow.set_experiment("boston-housing-mlops")
    
    with mlflow.start_run(run_name="model_evaluation", nested=True):
        df = pd.read_csv(test_data_path)
        y_true = df["TARGET"]
        X = df.drop(columns=["TARGET"])

        model = joblib.load(model_path)
        preds = model.predict(X)

        # Calculate metrics
        mae = float(mean_absolute_error(y_true, preds))
        r2 = float(r2_score(y_true, preds))
        mse = float(mean_squared_error(y_true, preds))
        rmse = float(mse ** 0.5)

        metrics = {
            "mae": mae,
            "r2": r2,
            "mse": mse,
            "rmse": rmse,
        }
        
        # Log metrics to MLflow
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2_score", r2)
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_rmse", rmse)
        
        # Save metrics to file
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        metrics_path = str(artifacts_dir / "metrics.json")
        
        with open(metrics_path, "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        
        # Log metrics file as artifact
        mlflow.log_artifact(metrics_path, "metrics")
        
        return metrics_path


def compile_components(output_dir: Path | None = None) -> None:
    """Compile component definitions into YAML specs for Task 2."""
    output_dir = output_dir or (Path(__file__).resolve().parents[1] / "components")
    output_dir.mkdir(exist_ok=True)

    components_info = {
        "data_extraction": {
            "description": "Fetch the DVC tracked dataset from the remote Git repository",
            "inputs": ["repo_url: str", "data_path: str", "rev: str"],
            "outputs": ["output_csv: str"]
        },
        "data_preprocessing": {
            "description": "Clean and split the dataset using MLflow tracking",
            "inputs": ["raw_data: str"],
            "outputs": ["train_output: str", "test_output: str"]
        },
        "model_training": {
            "description": "Train a Random Forest model using MLflow",
            "inputs": ["train_data: str"],
            "outputs": ["model_output: str"]
        },
        "model_evaluation": {
            "description": "Evaluate the trained model and persist metrics using MLflow",
            "inputs": ["test_data: str", "model_artifact: str"],
            "outputs": ["metrics_output: str"]
        },
    }

    for name, info in components_info.items():
        output_file = output_dir / f"{name}.yaml"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Kubeflow Pipeline Component: {name}\n")
            f.write(f"# Defined in: src/pipeline_components.py\n")
            f.write(f"# Function: {name}\n\n")
            f.write(f"# Description: {info['description']}\n\n")
            f.write(f"# Inputs:\n")
            for inp in info['inputs']:
                f.write(f"#   - {inp}\n")
            f.write(f"\n# Outputs:\n")
            for out in info['outputs']:
                f.write(f"#   - {out}\n")
            f.write(f"\n# Note: This component is implemented as a Python function.\n")
            f.write(f"# For Task 3, MLflow is used for orchestration instead of Kubeflow.\n")
        
        print(f"Created component YAML: {output_file}")


if __name__ == "__main__":
    compile_components()
