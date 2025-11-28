from pathlib import Path

from kfp import dsl
from kfp.compiler import Compiler

from src import pipeline_components as pc


@dsl.pipeline(
    name="boston-housing-mlops",
    description="Train and evaluate a Random Forest on the Boston housing dataset.",
)
def ml_pipeline(
    repo_url: str,
    data_path: str = "data/raw_data.csv",
    rev: str = "main",
):
    extraction = pc.data_extraction(
        repo_url=repo_url,
        data_path=data_path,
        rev=rev,
    )

    preprocessing = pc.data_preprocessing(
        raw_data=extraction.outputs["output_csv"]
    )

    training = pc.model_training(
        train_data=preprocessing.outputs["train_output"]
    )

    pc.model_evaluation(
        test_data=preprocessing.outputs["test_output"],
        model_artifact=training.outputs["model_output"],
    )


def compile_pipeline() -> Path:
    output_path = Path("pipeline.yaml")
    Compiler().compile(ml_pipeline, package_path=str(output_path))
    return output_path


if __name__ == "__main__":
    compile_pipeline()

