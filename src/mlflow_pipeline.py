"""
Task 3: MLflow Pipeline Orchestration
This replaces Kubeflow Pipelines with MLflow for pipeline orchestration.
"""

import mlflow
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import pipeline components
import pipeline_components as pc


def run_mlflow_pipeline(
    repo_url: str = None,
    data_path: str = "data/raw_data.csv",
    rev: str = "main",
    use_dvc: bool = False
):
    """
    Run the complete ML pipeline using MLflow for orchestration.
    
    Args:
        repo_url: Git repository URL (if using DVC get)
        data_path: Path to data file
        rev: Git revision/branch
        use_dvc: Whether to use DVC get or local file
    """
    # Set MLflow tracking
    mlflow.set_tracking_uri("file:///tmp/mlruns")
    mlflow.set_experiment("boston-housing-mlops")
    
    # Start parent run for the entire pipeline
    with mlflow.start_run(run_name="mlflow_pipeline_run") as parent_run:
        print("=" * 60)
        print("Starting MLflow Pipeline")
        print("=" * 60)
        
        # Step 1: Data Extraction
        print("\n[Step 1/4] Data Extraction...")
        if use_dvc and repo_url:
            raw_data_path = pc.data_extraction(repo_url, data_path, rev)
        else:
            # Use local data file
            raw_data_path = data_path
            if not Path(raw_data_path).exists():
                raise FileNotFoundError(f"Data file not found: {raw_data_path}")
        
        mlflow.log_param("data_source", "dvc" if use_dvc else "local")
        mlflow.log_param("data_path", raw_data_path)
        print(f"✓ Data extracted to: {raw_data_path}")
        
        # Step 2: Data Preprocessing
        print("\n[Step 2/4] Data Preprocessing...")
        train_path, test_path = pc.data_preprocessing(raw_data_path)
        mlflow.log_param("train_data_path", train_path)
        mlflow.log_param("test_data_path", test_path)
        print(f"✓ Training data: {train_path}")
        print(f"✓ Test data: {test_path}")
        
        # Step 3: Model Training
        print("\n[Step 3/4] Model Training...")
        model_path = pc.model_training(train_path)
        mlflow.log_param("model_path", model_path)
        print(f"✓ Model saved to: {model_path}")
        
        # Step 4: Model Evaluation
        print("\n[Step 4/4] Model Evaluation...")
        metrics_path = pc.model_evaluation(test_path, model_path)
        mlflow.log_param("metrics_path", metrics_path)
        print(f"✓ Metrics saved to: {metrics_path}")
        
        # Log pipeline completion
        mlflow.log_param("pipeline_status", "completed")
        
        print("\n" + "=" * 60)
        print("MLflow Pipeline Completed Successfully!")
        print("=" * 60)
        print(f"\nView results:")
        print(f"  - MLflow UI: Run 'mlflow ui' then visit http://localhost:5000")
        print(f"  - Model: {model_path}")
        print(f"  - Metrics: {metrics_path}")
        print(f"  - Tracking URI: file:///tmp/mlruns")
        
        return {
            "raw_data": raw_data_path,
            "train_data": train_path,
            "test_data": test_path,
            "model": model_path,
            "metrics": metrics_path,
            "run_id": parent_run.info.run_id
        }


if __name__ == "__main__":
    import sys
    
    # Default: use local data file
    use_dvc = "--dvc" in sys.argv
    repo_url = None
    
    if use_dvc:
        # Example: python src/mlflow_pipeline.py --dvc
        # You would set repo_url here
        repo_url = "https://github.com/your-username/mlops-kubeflow-assignment.git"
    
    results = run_mlflow_pipeline(
        repo_url=repo_url,
        data_path="data/raw_data.csv",
        rev="main",
        use_dvc=use_dvc
    )
    
    print(f"\nPipeline Run ID: {results['run_id']}")

