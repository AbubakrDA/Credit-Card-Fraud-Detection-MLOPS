import os
import sys
import mlflow
import mlflow.sklearn

# Ensure the root directory is in the path so we can import 'src'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Import our modular logic
from src.config import DATA_PATH, MLFLOW_TRACKING_URI
from src.data import load_data, split_data
from src.preprocess import build_preprocessor, apply_preprocessing
from src.train import get_models, train_model
from src.evaluate import evaluate_model
from src.model_selection import select_best_model
from src.utils import save_confusion_matrix
from sklearn.pipeline import Pipeline

def run_standalone_pipeline():
    """
    Executes the multi-model training pipeline directly without Airflow.
    
    Why this matters:
    This allows Windows users to populate their MLflow registry and test the 
    end-to-end flow immediately, while keeping the modular 'src' code 
    identical to the production Airflow definition.
    """
    print("--- STARTING STANDALONE TRAINING PIPELINE ---")
    
    # 1. Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Credit_Card_Fraud_Detection")
    
    # 2. Data Loading
    print(f"Loading data from: {DATA_PATH}")
    df = load_data(str(DATA_PATH))
    X_train, X_test, y_train, y_test = split_data(df)
    
    # 3. Preprocessing
    print("Building and fitting preprocessor...")
    preprocessor = build_preprocessor()
    X_train_prep, X_test_prep = apply_preprocessing(preprocessor, X_train, X_test)
    
    # 4. Multi-Model Loop
    models = get_models()
    run_metrics = []
    
    with mlflow.start_run(run_name="Standalone_Execution") as parent_run:
        for model_name, model in models.items():
            print(f"\n>>> Running {model_name}...")
            
            with mlflow.start_run(run_name=f"{model_name}_standalone", nested=True) as child_run:
                # Train
                model.fit(X_train_prep, y_train)
                
                # Create the inference-safe pipeline
                full_pipeline = Pipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", model)
                ])
                
                # Evaluate
                metrics, cm = evaluate_model(model, X_test_prep, y_test)
                
                # Log to MLflow
                mlflow.log_metrics(metrics)
                mlflow.log_params(model.get_params())
                
                # Artifacts
                cm_path = os.path.join(BASE_DIR, "logs")
                os.makedirs(cm_path, exist_ok=True)
                cm_filename = os.path.join(cm_path, f"{model_name}_cm_standalone.png")
                save_confusion_matrix(cm, model_name, cm_filename)
                mlflow.log_artifact(cm_filename)
                
                # Model Registration
                mlflow.sklearn.log_model(
                    sk_model=full_pipeline, 
                    artifact_path="fraud_pipeline_model"
                )
                
                print(f"    Metrics: {metrics}")
                run_metrics.append({
                    "run_id": child_run.info.run_id,
                    "model_name": model_name,
                    "metrics": metrics
                })
        
        # 5. Model Selection
        print("\n--- Model Selection ---")
        best_run_id = select_best_model(run_metrics)
        best_model_info = next(m for m in run_metrics if m["run_id"] == best_run_id)
        
        print(f"BEST MODEL SELECTED: {best_model_info['model_name']}")
        print(f"MLFLOW RUN ID: {best_run_id}")
        
        # Log result
        mlflow.log_param("best_model_run_id", best_run_id)
        mlflow.log_param("best_model_name", best_model_info['model_name'])

    print("\n--- PIPELINE COMPLETE ---")
    print("Run 'mlflow ui' to see results.")

if __name__ == "__main__":
    run_standalone_pipeline()
