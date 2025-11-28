import kfp
from kfp import dsl
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics

# [Same component code as before - the @component decorators]

@component(
    base_image='python:3.9',
    packages_to_install=['pandas', 'scikit-learn']
)
def data_extraction(output_data: Output[Dataset]):
    """Extract data"""
    import pandas as pd
    from sklearn.datasets import load_boston
    
    print("Loading Boston housing dataset...")
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['target'] = boston.target
    
    df.to_csv(output_data.path, index=False)
    print(f"Data saved. Shape: {df.shape}")

@component(
    base_image='python:3.9',
    packages_to_install=['pandas', 'scikit-learn']
)
def data_preprocessing(input_data: Input[Dataset], 
                       train_data: Output[Dataset],
                       test_data: Output[Dataset]):
    """Preprocess and split data"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    df = pd.read_csv(input_data.path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df['target'] = y_train.values
    train_df.to_csv(train_data.path, index=False)
    
    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['target'] = y_test.values
    test_df.to_csv(test_data.path, index=False)
    
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

@component(
    base_image='python:3.9',
    packages_to_install=['pandas', 'scikit-learn', 'joblib']
)
def model_training(train_data: Input[Dataset], model_artifact: Output[Model]):
    """Train model"""
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    import joblib
    
    train_df = pd.read_csv(train_data.path)
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, model_artifact.path)
    print("Model trained and saved")

@component(
    base_image='python:3.9',
    packages_to_install=['pandas', 'scikit-learn', 'joblib']
)
def model_evaluation(test_data: Input[Dataset], model_artifact: Input[Model], metrics: Output[Metrics]):
    """Evaluate model"""
    import pandas as pd
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import joblib
    import json
    
    test_df = pd.read_csv(test_data.path)
    model = joblib.load(model_artifact.path)
    
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    y_pred = model.predict(X_test)
    
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    metrics.log_metric("rmse", rmse)
    metrics.log_metric("mae", mae)
    metrics.log_metric("r2_score", r2)

@pipeline(
    name='Boston Housing ML Pipeline',
    description='Complete ML pipeline for Boston housing price prediction'
)
def boston_housing_pipeline():
    extract_task = data_extraction()
    preprocess_task = data_preprocessing(input_data=extract_task.outputs['output_data'])
    train_task = model_training(train_data=preprocess_task.outputs['train_data'])
    eval_task = model_evaluation(
        test_data=preprocess_task.outputs['test_data'],
        model_artifact=train_task.outputs['model_artifact']
    )

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(
        pipeline_func=boston_housing_pipeline,
        package_path='pipeline.yaml'
    )
    print("✅ Pipeline compiled successfully to pipeline.yaml")