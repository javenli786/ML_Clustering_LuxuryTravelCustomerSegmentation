import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import mlflow
import mlflow.sklearn
import datetime
import os
from typing import Dict, Any, List

mlflow.set_tracking_uri("file:///your/absolute/path/to/mlruns")

def train_kmeans_model(
    X_pca_2, 
    n_clusters=4, 
    init='k-means++', 
    max_iter=500, 
    n_init=10, 
    random_state=30, 
    algorithm='elkan'):
    
    # Generate a unique model name with timestamp
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"kmeans_model_{current_time}"
    
    # Start MLflow run
    with mlflow.start_run() as run:
        # Log parameters
        params = {
            "n_clusters": n_clusters,
            "init": init,
            "max_iter": max_iter,
            "n_init": n_init,
            "random_state": random_state,
            "algorithm": algorithm,
            "training_date": datetime.datetime.now().strftime("%Y-%m-%d")
        }
        mlflow.log_params(params)
        
        # Train the model and generate predictions
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
            algorithm=algorithm
        )
        
        y_pred = kmeans.fit_predict(X_pca_2)
        
        # Calculate metrics
        silhouette = silhouette_score(X_pca_2, y_pred)
        calinski = calinski_harabasz_score(X_data, y_pred)
        davies = davies_bouldin_score(X_data, y_pred)
            
        # Log metrics
        metrics = {
            "silhouette_score": silhouette,
            "calinski_harabasz_score": calinski,
            "davies_bouldin_score": davies
            }
        mlflow.log_metrics(metrics)
        
        # Log model with timestamp
        mlflow.sklearn.log_model(kmeans, model_name)
        
        return {
            "model": kmeans,
            "predictions": y_pred,
            "run_id": run.info.run_id,
            "model_name": model_name
        }


def select_best_model(
    experiment_name: str, 
    metric: str = "silhouette_score", 
    ascending: bool = False
):

    # Set experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment {experiment_name} not found.")
        return None
    
    # Get all runs in the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs.empty:
        print("No runs found in the experiment.")
        return None
    
    # For davies_bouldin_score, lower is better
    if metric == "davies_bouldin_score":
        ascending = True
    
    # Sort runs by the metric
    sorted_runs = runs.sort_values(f"metrics.{metric}", ascending=ascending)
    
    if sorted_runs.empty:
        return None
    
    # Get the run ID of the best model
    best_run_id = sorted_runs.iloc[0]["run_id"]
    
    # Get the artifacts of the run to find the model name
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(best_run_id)
    model_names = [artifact.path for artifact in artifacts if artifact.is_dir]
    
    if not model_names:
        return {"run_id": best_run_id, "model_name": "kmeans_model"}
    
    return {"run_id": best_run_id, "model_name": model_names[0]}



from mlflow.tracking import MlflowClient

def promote_best_kmeans_model(metric_name="silhouette_score", higher_is_better=True):
    client = MlflowClient()
    experiment = client.get_experiment_by_name("KMeans_Clustering")
    
    best_score = float('-inf') if higher_is_better else float('inf')
    best_run_id = None

    for run in client.search_runs(experiment_ids=[experiment.experiment_id]):
        if metric_name in run.data.metrics:
            score = run.data.metrics[metric_name]
            if (higher_is_better and score > best_score) or (not higher_is_better and score < best_score):
                best_score = score
                best_run_id = run.info.run_id

    if best_run_id:
        model_uri = f"runs:/{best_run_id}/kmeans_models"
        mlflow.register_model(model_uri, "Best_KMeans_Model")
        print(f"Promoted model from run {best_run_id} to registry.")
