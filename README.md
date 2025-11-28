# MLOps Kubeflow Assignment

This repo implements the Assignment #4 requirements for the Cloud MLOps course. It demonstrates end-to-end MLOps practices covering data versioning with DVC, Kubeflow Pipelines orchestration on Minikube, and CI automation via Jenkins or GitHub Actions.

## Project Overview

- **Dataset**: Boston Housing (regression target treated as continuous variable).
- **Model**: RandomForestRegressor with standard scaling and an 80/20 train/test split.
- **Pipeline**: Four Kubeflow components (data extraction, preprocessing, training, evaluation) orchestrated into a single DAG and compiled to `pipeline.yaml`.
- **MLflow Integration**: MLflow is integrated into Kubeflow components for experiment tracking, model logging, and metrics tracking.
- **CI**: Jenkins declarative pipeline validates dependency installation and pipeline compilation.

## Repository Structure

```
.
├── components/              # Auto-generated Kubeflow component specs
├── data/                    # Contains raw dataset tracked by DVC
├── src/
│   ├── model_training.py    # Standalone training script (local runs)
│   ├── mlflow_training.py   # Standalone MLflow training script
│   └── pipeline_components.py  # Kubeflow components with MLflow integration
├── pipeline.py              # Kubeflow pipeline definition & compiler entrypoint
├── Jenkinsfile              # Three-stage Jenkins CI pipeline
├── Dockerfile               # Base image for custom components (optional)
├── requirements.txt
├── README.md
└── data/raw_data.csv.dvc    # DVC metadata for dataset
```

## Local Setup

1. **Clone & create environment**
   ```bash
   git clone https://github.com/Andleeb19/mlops-kubeflow-assignment.git
   cd mlops-kubeflow-assignment
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **DVC configuration**
   ```bash
   dvc init  # only on first clone
   dvc remote add -d localremote <absolute-path-to-remote-storage>
   dvc pull
   ```
   > Replace `<absolute-path-to-remote-storage>` with your storage (local folder, S3 bucket, etc.).

3. **Verify dataset**
   ```bash
   dvc status
   python src/model_training.py
   ```

4. **Test MLflow tracking (optional)**
   ```bash
   python src/mlflow_training.py
   mlflow ui  # View MLflow UI at http://localhost:5000
   ```

## MLflow Integration

This project uses **MLflow** for experiment tracking and model management within Kubeflow components. MLflow provides:

- **Experiment Tracking**: All runs are logged with parameters, metrics, and artifacts
- **Model Logging**: Models are automatically logged and versioned
- **Metrics Tracking**: Training and evaluation metrics are tracked across runs

### MLflow Features Used:

1. **In Kubeflow Components** (`src/pipeline_components.py`):
   - `mlflow.set_experiment()`: Creates/uses experiment named "boston-housing-mlops"
   - `mlflow.start_run()`: Tracks each component execution
   - `mlflow.log_param()`: Logs hyperparameters (n_estimators, random_state, etc.)
   - `mlflow.log_metric()`: Logs metrics (MAE, R2, RMSE, etc.)
   - `mlflow.sklearn.log_model()`: Logs trained models
   - `mlflow.log_artifact()`: Logs model files and metrics JSON

2. **Standalone MLflow Script** (`src/mlflow_training.py`):
   - Can be run independently to test MLflow tracking
   - Useful for local development and testing

### Viewing MLflow Experiments:

```bash
# Start MLflow UI
mlflow ui

# Access at http://localhost:5000
# View experiments, runs, metrics, and models
```

### MLflow Tracking Location:

- Default: `file:///tmp/mlruns` (local file system)
- Can be changed to remote tracking server (MLflow server, S3, etc.)

## Kubeflow Pipeline Walkthrough

1. **Compile components (creates YAMLs under `components/`):**
   ```bash
   python src/pipeline_components.py
   ```
2. **Compile the DAG:**
   ```bash
   python pipeline.py
   ```
   Upload `pipeline.yaml` in the Kubeflow Pipelines UI.

3. **Run on Minikube:**
   - Start minikube: `minikube start --cpus=4 --memory=8192 --disk-size=40g`
   - Deploy Kubeflow (standalone KFP recommended).
   - Open the KFP dashboard (`kubectl port-forward svc/ml-pipeline-ui 8080:80 -n kubeflow`).
   - Upload `pipeline.yaml`, set parameters (`repo_url`, `data_path`, `rev`), and launch a run.
   - Monitor run graph and step logs to capture screenshots for Deliverable 3.

## Jenkins/GitHub Workflow

1. Create a Pipeline job pointing to this repo and choose the included `Jenkinsfile`.
2. Ensure the build node has Python 3.10+, Docker (optional), and DVC installed.
3. Trigger the job:
   - **Stage 1**: Checkout via Jenkins SCM.
   - **Stage 2**: Create venv & install `requirements.txt`.
   - **Stage 3**: Compile Kubeflow pipeline (runs `python pipeline.py`).
4. Capture the console output showing all stages succeeded.

For GitHub Actions, replicate the same stages inside `.github/workflows/ci.yml` if desired.

## Deliverables Checklist

- [x] Repo structure + DVC tracking screenshots (`git status`, `dvc status`, `dvc push`).
- [x] Component code + YAML outputs under `components/`.
- [x] Pipeline compiled to `pipeline.yaml` for upload to Kubeflow UI.
- [x] Jenkinsfile ready for CI automation.
- [x] README documenting setup, pipeline run, and CI instructions.

## GitHub Commands Reference

```bash
# Initial setup
git init
git remote add origin https://github.com/Andleeb19/mlops-kubeflow-assignment.git

# Regular workflow
git status
git add .
git commit -m "Implement Kubeflow MLOps assignment"
git push origin main
```


