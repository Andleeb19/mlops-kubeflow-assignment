"""
Kubeflow Pipeline Components for Task 3
These are proper Kubeflow components using kfp.dsl.component decorator.
"""

from kfp import dsl
from kfp.dsl import InputPath, OutputPath

BASE_IMAGE = "python:3.10-slim"
COMMON_PACKAGES = [
    "pandas==2.1.4",
    "scikit-learn==1.3.2",
    "joblib==1.3.2",
    "numpy==1.26.4",
    "dvc==3.51.0",
    "mlflow==2.11.0",
]


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=COMMON_PACKAGES,
)
def data_extraction(
    repo_url: str,
    data_path: str,
    rev: str,
    output_csv: OutputPath(str),
) -> None:
    """Fetch the DVC tracked dataset from the remote Git repository."""
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
        shutil.copy(dest_path, output_csv)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=COMMON_PACKAGES,
)
def data_preprocessing(
    raw_data: InputPath(str),
    train_output: OutputPath(str),
    test_output: OutputPath(str),
) -> None:
    """Clean and split the dataset."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import mlflow

    # Initialize MLflow
    mlflow.set_tracking_uri("file:///tmp/mlruns")
    mlflow.set_experiment("boston-housing-mlops")
    
    with mlflow.start_run(run_name="data_preprocessing", nested=True):
        df = pd.read_csv(raw_data).dropna()
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

        train_df.to_csv(train_output, index=False)
        test_df.to_csv(test_output, index=False)
        
        # Log preprocessing metrics to MLflow
        mlflow.log_metric("train_samples", len(train_df))
        mlflow.log_metric("test_samples", len(test_df))
        mlflow.log_metric("features_count", len(X.columns))
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=COMMON_PACKAGES,
)
def model_training(
    train_data: InputPath(str),
    model_output: OutputPath(str),
) -> None:
    """Train a Random Forest model using MLflow."""
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    import mlflow
    import mlflow.sklearn

    # Initialize MLflow
    mlflow.set_tracking_uri("file:///tmp/mlruns")
    mlflow.set_experiment("boston-housing-mlops")
    
    with mlflow.start_run(run_name="model_training", nested=True):
        df = pd.read_csv(train_data)
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
        joblib.dump(model, model_output)
        
        # Log the saved model path
        mlflow.log_artifact(model_output, "model_artifact")


@dsl.component(
    base_image=BASE_IMAGE,
    packages_to_install=COMMON_PACKAGES,
)
def model_evaluation(
    test_data: InputPath(str),
    model_artifact: InputPath(str),
    metrics_output: OutputPath(str),
) -> None:
    """Evaluate the trained model and persist metrics using MLflow."""
    import json
    import joblib
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    import mlflow
    import mlflow.sklearn

    # Initialize MLflow
    mlflow.set_tracking_uri("file:///tmp/mlruns")
    mlflow.set_experiment("boston-housing-mlops")
    
    with mlflow.start_run(run_name="model_evaluation", nested=True):
        df = pd.read_csv(test_data)
        y_true = df["TARGET"]
        X = df.drop(columns=["TARGET"])

        model = joblib.load(model_artifact)
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
        with open(metrics_output, "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        
        # Log metrics file as artifact
        mlflow.log_artifact(metrics_output, "metrics")


