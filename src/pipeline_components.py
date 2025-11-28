from __future__ import annotations

from pathlib import Path

from kfp import components, dsl

BASE_IMAGE = "python:3.10-slim"
COMMON_PACKAGES = [
    "pandas==2.1.4",
    "scikit-learn==1.3.2",
    "joblib==1.3.2",
    "numpy==1.26.4",
    "dvc==3.51.0",
]


@dsl.component(base_image=BASE_IMAGE, packages_to_install=COMMON_PACKAGES)
def data_extraction(
    repo_url: str,
    data_path: str,
    rev: str,
    output_csv: dsl.OutputPath("CSV"),
):
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


@dsl.component(base_image=BASE_IMAGE, packages_to_install=COMMON_PACKAGES)
def data_preprocessing(
    raw_data: dsl.InputPath("CSV"),
    train_output: dsl.OutputPath("CSV"),
    test_output: dsl.OutputPath("CSV"),
):
    """Clean and split the dataset."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

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


@dsl.component(base_image=BASE_IMAGE, packages_to_install=COMMON_PACKAGES)
def model_training(
    train_data: dsl.InputPath("CSV"),
    model_output: dsl.OutputPath("Model"),
):
    """Train a Random Forest model."""
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    df = pd.read_csv(train_data)
    y = df["TARGET"]
    X = df.drop(columns=["TARGET"])

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)

    joblib.dump(model, model_output)


@dsl.component(base_image=BASE_IMAGE, packages_to_install=COMMON_PACKAGES)
def model_evaluation(
    test_data: dsl.InputPath("CSV"),
    model_artifact: dsl.InputPath("Model"),
    metrics_output: dsl.OutputPath("Metrics"),
):
    """Evaluate the trained model and persist metrics."""
    import json
    import joblib
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, r2_score

    df = pd.read_csv(test_data)
    y_true = df["TARGET"]
    X = df.drop(columns=["TARGET"])

    model = joblib.load(model_artifact)
    preds = model.predict(X)

    metrics = {
        "mae": float(mean_absolute_error(y_true, preds)),
        "r2": float(r2_score(y_true, preds)),
    }
    with open(metrics_output, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)


def compile_components(output_dir: Path | None = None) -> None:
    """Compile component definitions into YAML specs."""
    output_dir = output_dir or (Path(__file__).resolve().parents[1] / "components")
    output_dir.mkdir(exist_ok=True)

    comp_funcs = {
        "data_extraction": data_extraction,
        "data_preprocessing": data_preprocessing,
        "model_training": model_training,
        "model_evaluation": model_evaluation,
    }

    for name, func in comp_funcs.items():
        spec = components.create_component_from_func(func)
        spec.save(str(output_dir / f"{name}.yaml"))


if __name__ == "__main__":
    compile_components()

