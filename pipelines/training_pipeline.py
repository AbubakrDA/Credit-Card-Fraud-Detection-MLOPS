import os
import sys
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline

from airflow import DAG
from airflow.operators.python import PythonOperator

# Add the project directory path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.config import DATA_PATH, MLFLOW_TRACKING_URI
from src.data import load_data, split_data
from src.preprocess import build_preprocessor, apply_preprocessing
from src.train import get_models
from src.evaluate import evaluate_model
from src.model_selection import select_best_model
from src.utils import save_confusion_matrix

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Credit_Card_Fraud_Detection")

def run_training_pipeline(**kwargs):
    """
    Main Airflow Task: Loads data, trains multiple models, evaluates, and selects best model.
    
    Why this matters in MLOps:
    This entire logical flow should execute identically whether triggered manually 
    or via a scheduler. It registers the chosen best model directly into MLflow so the
    FastAPI service can securely dynamically pull the updated "champion" model.
    """
    ti = kwargs['ti']
    
    print("Step 1: Loading Data...")
    df = load_data(str(DATA_PATH))
    
    print("Step 2: Splitting Data...")
    X_train, X_test, y_train, y_test = split_data(df)
    
    print("Step 3: Preprocessing Data...")
    preprocessor = build_preprocessor()
    X_train_prep, X_test_prep = apply_preprocessing(preprocessor, X_train, X_test)
    
    # Track stats for logging
    models = get_models()
    run_metrics = []
    
    print("Step 4: Training & Evaluating Models with MLflow...")
    # Parent run so all sub-models stay grouped
    with mlflow.start_run(run_name="Full_Training_Pipeline") as parent_run:
        
        for model_name, model in models.items():
            run_name = f"{model_name}_training"
            
            with mlflow.start_run(run_name=run_name, nested=True) as child_run:
                print(f"-> Training {model_name}...")
                
                # Fit the core estimator
                model.fit(X_train_prep, y_train)
                
                # We package the preprocessor and estimator together so inference is seamless.
                # If we don't do this, we'd have to deploy feature scripts alongside the model.
                full_pipeline = Pipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", model)
                ])
                
                print(f"-> Evaluating {model_name}...")
                metrics, cm = evaluate_model(model, X_test_prep, y_test)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                mlflow.log_params(model.get_params())
                mlflow.log_param("model_type", model_name)
                
                # Save & Log the Confusion Matrix locally then artifact
                cm_path = os.path.join(BASE_DIR, "logs")
                os.makedirs(cm_path, exist_ok=True)
                cm_filename = os.path.join(cm_path, f"{model_name}_cm.png")
                save_confusion_matrix(cm, model_name, cm_filename)
                mlflow.log_artifact(cm_filename)
                
                # Log the actual pipeline so the API can use it
                mlflow.sklearn.log_model(
                    sk_model=full_pipeline, 
                    artifact_path="fraud_pipeline_model"
                )
                
                run_metrics.append({
                    "run_id": child_run.info.run_id,
                    "model_name": model_name,
                    "metrics": metrics
                })
                print(f"   {model_name} Metrics: {metrics}")
        
        print("Step 5: Selecting the Best Model...")
        best_run_id = select_best_model(run_metrics)
        best_model_name = next(m["model_name"] for m in run_metrics if m["run_id"] == best_run_id)
        
        print(f"*** Best Model Selected: {best_model_name} ({best_run_id}) ***")
        
        # Log which model won the parent run
        mlflow.log_param("best_model_run_id", best_run_id)
        mlflow.log_param("best_model_name", best_model_name)
        
        # We push the best run_id to XCom for downstream steps logically (if any)
        # Even though FastAPI will just search MLflow directly later.
        ti.xcom_push(key='best_model_run_id', value=best_run_id)


# Define Airflow DAG
default_args = {
    'owner': 'mlops',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

dag = DAG(
    'fraud_detection_training',
    default_args=default_args,
    description='Trains multiple models and selects the best one based on PR-AUC',
    schedule_interval='@weekly',
    catchup=False
)

train_evaluate_task = PythonOperator(
    task_id='run_training_pipeline',
    python_callable=run_training_pipeline,
    provide_context=True,
    dag=dag,
)

train_evaluate_task
