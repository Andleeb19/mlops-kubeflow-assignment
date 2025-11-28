"""
Task 3: Kubeflow Pipeline Definition
This file defines the complete pipeline using Kubeflow Pipelines SDK.
"""

from kfp import dsl
from kfp.compiler import Compiler
from src import kubeflow_components as kc


@dsl.pipeline(
    name="boston-housing-mlops-pipeline",
    description="Complete ML pipeline for Boston Housing dataset: data extraction, preprocessing, training, and evaluation.",
)
def ml_pipeline(
    repo_url: str = "https://github.com/your-username/mlops-kubeflow-assignment.git",
    data_path: str = "data/raw_data.csv",
    rev: str = "main",
):
    """
    Main pipeline definition.
    
    Args:
        repo_url: Git repository URL containing DVC-tracked data
        data_path: Path to data file in the repository
        rev: Git revision/branch to use
    """
    # Step 1: Data Extraction
    extraction_task = kc.data_extraction(
        repo_url=repo_url,
        data_path=data_path,
        rev=rev,
    )
    
    # Step 2: Data Preprocessing
    preprocessing_task = kc.data_preprocessing(
        raw_data=extraction_task.outputs["output_csv"],
    )
    
    # Step 3: Model Training
    training_task = kc.model_training(
        train_data=preprocessing_task.outputs["train_output"],
    )
    
    # Step 4: Model Evaluation
    evaluation_task = kc.model_evaluation(
        test_data=preprocessing_task.outputs["test_output"],
        model_artifact=training_task.outputs["model_output"],
    )


def compile_pipeline(output_path: str = "pipeline.yaml") -> None:
    """Compile the pipeline to YAML file."""
    Compiler().compile(ml_pipeline, output_path)
    print(f"Pipeline compiled successfully to: {output_path}")


if __name__ == "__main__":
    compile_pipeline()
