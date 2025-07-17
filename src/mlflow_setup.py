import mlflow
import mlflow.sklearn
import datetime



def setup_mlflow(TRACKING_URI, EXPERIMENT_NAME):
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME) 
    else:
        experiment_id = experiment.experiment_id



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