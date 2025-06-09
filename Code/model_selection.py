import mlflow
import mlflow.sklearn

def select_best_model_for_prediction(
    tracking_uri,
    experiment_name,
    metric="silhouette_score", 
    high_metric=True
):
    """
    Select the best model from MLflow based on a metric and return for prediction.
    
    Args:
        tracking_uri: The URI of the MLflow tracking server
        experiment_name: Name of the MLflow experiment
        metric: Metric for evaluation
        higher_is_better: Whether higher metric values are better (default: True)
        
    Returns:
        A tuple of (model, model_info) where model_info contains metadata about the model
        Returns (None, None) if no suitable model is found
    """
    # Set the tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    
    # Get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found")
        return None, None
    
    # Generate all runs with the specified metric
    sort_order = "DESC" if high_metric else "ASC"
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"metrics.{metric} IS NOT NULL",
        order_by=[f"metrics.{metric} {sort_order}"]
    )
    
    # Generate the best run (first row after sorting)
    best_run = runs.iloc[0]
    best_run_id = best_run.run_id
    
    # Generate metric value and parameters
    best_metric_value = best_run[f"metrics.{metric}"]
    parameters = {k.replace("params.", ""): v for k, v in best_run.items() if k.startswith("params.")}
    
    # Generate model artifact path
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(best_run_id)
    model_dirs = [artifact.path for artifact in artifacts if artifact.is_dir]
    
    # Load the model
    model_path = model_dirs[0]
    try:
        model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/{model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
    # Collect model information
    model_info = {
        "run_id": best_run_id,
        "model_path": model_path,
        "metric_name": metric,
        "metric_value": best_metric_value,
        "parameters": parameters
    }
    
    print(f"Best model with {metric} = {best_metric_value:.4f}")
    print(f"Run ID: {best_run_id}")
    print(f"Parameters: {parameters}")
    
    return model, model_info