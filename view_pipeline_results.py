"""
Helper script to view MLflow pipeline results
Run this after executing the MLflow pipeline
"""

import mlflow
import json
from pathlib import Path

def view_pipeline_results():
    """Display pipeline results in a formatted way."""
    
    # Set tracking URI
    mlflow.set_tracking_uri("file:///tmp/mlruns")
    
    print("=" * 70)
    print("MLflow Pipeline Results Viewer")
    print("=" * 70)
    
    # Get experiment
    try:
        experiment = mlflow.get_experiment_by_name("boston-housing-mlops")
        if experiment is None:
            print("❌ Experiment 'boston-housing-mlops' not found.")
            print("   Run the pipeline first: python src/mlflow_pipeline.py")
            return
        
        print(f"\n📊 Experiment: {experiment.name}")
        print(f"   Experiment ID: {experiment.experiment_id}")
        print(f"   Artifact Location: {experiment.artifact_location}")
        
        # Get all runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=5
        )
        
        if runs.empty:
            print("\n❌ No runs found in this experiment.")
            return
        
        print(f"\n📈 Found {len(runs)} run(s)")
        print("-" * 70)
        
        # Display parent runs (pipeline runs)
        parent_runs = runs[runs['tags.mlflow.parentRunId'].isna()]
        
        for idx, run in parent_runs.iterrows():
            print(f"\n🔷 Pipeline Run: {run['tags.mlflow.runName']}")
            print(f"   Run ID: {run['run_id']}")
            print(f"   Status: {run['status']}")
            print(f"   Start Time: {run['start_time']}")
            
            # Get full run details
            full_run = mlflow.get_run(run['run_id'])
            
            if full_run.data.params:
                print(f"\n   📝 Parameters:")
                for key, value in full_run.data.params.items():
                    print(f"      • {key}: {value}")
            
            if full_run.data.metrics:
                print(f"\n   📊 Metrics:")
                for key, value in full_run.data.metrics.items():
                    print(f"      • {key}: {value:.4f}")
            
            # Get nested runs
            nested_runs = runs[runs['tags.mlflow.parentRunId'] == run['run_id']]
            if not nested_runs.empty:
                print(f"\n   🔹 Nested Runs ({len(nested_runs)}):")
                for _, nested in nested_runs.iterrows():
                    run_name = nested.get('tags.mlflow.runName', 'unnamed')
                    status = nested['status']
                    print(f"      • {run_name}: {status}")
                    
                    # Show metrics for nested runs
                    nested_full = mlflow.get_run(nested['run_id'])
                    if nested_full.data.metrics:
                        for key, value in nested_full.data.metrics.items():
                            print(f"        - {key}: {value:.4f}")
        
        print("\n" + "=" * 70)
        
        # Check artifacts
        artifacts_dir = Path("artifacts")
        if artifacts_dir.exists():
            print("\n📁 Local Artifacts:")
            for file in artifacts_dir.iterdir():
                if file.is_file():
                    size = file.stat().st_size
                    print(f"   • {file.name} ({size:,} bytes)")
                    
                    # Display metrics.json content
                    if file.name == "metrics.json":
                        try:
                            with open(file, 'r') as f:
                                metrics = json.load(f)
                                print(f"\n   📊 Metrics from {file.name}:")
                                for key, value in metrics.items():
                                    if isinstance(value, float):
                                        print(f"      • {key}: {value:.4f}")
                                    else:
                                        print(f"      • {key}: {value}")
                        except Exception as e:
                            print(f"      (Could not read: {e})")
        else:
            print("\n⚠️  Artifacts directory not found")
        
        print("\n" + "=" * 70)
        print("\n💡 To view in MLflow UI, run: mlflow ui")
        print("   Then open: http://localhost:5000")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you've run the pipeline first:")
        print("  python src/mlflow_pipeline.py")


if __name__ == "__main__":
    view_pipeline_results()

