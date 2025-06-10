import mlflow
import mlflow.sklearn
import datetime
import os



def setup_mlflow(tracking_uri=None, experiment_name="KMeans_Clustering"):
    
    # Set tracking URI
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Set or create the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)
    
    mlflow.set_experiment(experiment_name)
    
    return experiment_id


def log_model_to_mlflow(model_result):

    # Generate a timestamp for the model name
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"kmeans_model_{current_time}"
    
    # Start a new MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        
        # Log model parameters
        params = model_result["parameters"].copy()
        params["training_date"] = datetime.datetime.now().strftime("%Y-%m-%d")
        mlflow.log_params(params)
        
        # Log model metrics
        mlflow.log_metrics(model_result["metrics"])
        
        # Log the model itself
        mlflow.sklearn.log_model(model_result["model"], model_name)
        
        return {
            "run_id": run_id,
            "model_name": model_name
        }

def get_tracking_uri():
    return mlflow.get_tracking_uri()