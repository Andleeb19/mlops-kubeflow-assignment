import kfp
from pipeline_local import boston_housing_pipeline

# This will show the pipeline graph
print("Pipeline structure:")
print("=" * 50)
print("1. data_extraction → Loads Boston housing dataset")
print("2. data_preprocessing → Scales and splits data")  
print("3. model_training → Trains Random Forest model")
print("4. model_evaluation → Evaluates and logs metrics")
print("=" * 50)

# Compile to check it's valid
kfp.compiler.Compiler().compile(
    pipeline_func=boston_housing_pipeline,
    package_path='pipeline.yaml'
)

print("\n✅ Pipeline compiled successfully!")
print("📄 Check pipeline.yaml for the full pipeline definition")