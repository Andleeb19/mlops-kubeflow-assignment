"""
Standalone MLflow training script for Boston Housing dataset.
This can be run independently to test MLflow tracking.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path

# Set MLflow tracking URI (local file system)
mlflow.set_tracking_uri("file:///tmp/mlruns")
mlflow.set_experiment("boston-housing-mlops")

def main():
    # Load data
    data_path = Path("data/raw_data.csv")
    if not data_path.exists():
        print(f"Error: {data_path} not found. Please run DVC pull first.")
        return
    
    df = pd.read_csv(data_path).dropna()
    target_column = "TARGET"
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Start MLflow run
    with mlflow.start_run(run_name="standalone_training"):
        # Model parameters
        n_estimators = 200
        max_depth = None
        random_state = 42
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        train_preds = model.predict(X_train_scaled)
        test_preds = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, train_preds)
        train_r2 = r2_score(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        test_r2 = r2_score(y_test, test_preds)
        test_mse = mean_squared_error(y_test, test_preds)
        test_rmse = test_mse ** 0.5
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("test_size", 0.2)
        
        # Log metrics
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("train_r2_score", train_r2)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_r2_score", test_r2)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        joblib.dump(model, model_dir / "model.joblib")
        mlflow.log_artifact(str(model_dir / "model.joblib"), "model_artifact")
        
        print("=" * 50)
        print("MLflow Training Complete!")
        print("=" * 50)
        print(f"Train MAE: {train_mae:.4f}")
        print(f"Train R2: {train_r2:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Test R2: {test_r2:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"\nMLflow UI: Run 'mlflow ui' to view experiments")
        print(f"Tracking URI: file:///tmp/mlruns")

if __name__ == "__main__":
    main()

