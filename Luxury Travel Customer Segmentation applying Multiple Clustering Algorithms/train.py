import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import mlflow
import mlflow.sklearn
import os
import logging
import joblib
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
print('123')

def train_model(X, n_clusters=4, random_state=30, model_version=None):
    
    
    mlflow.set_experiment("Clustering_LuxuryTravelCustomerSegmentation")
    
    # Create the model version
    if model_version is None:
        model_version = f"training_{datetime.now().strftime('%Y_%m_%d')}"
        
    run_name = f"KMeans_{n_clusters}_clusters_{model_version}"
    logger.info(f"Training KMeans model: {run_name}")
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("algorithm", "KMeans")
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("init", "k-means++")
        mlflow.log_param("max_iter", 500)
        mlflow.log_param("n_init", 10)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("algorithm_type", "elkan")
        mlflow.log_param("model_version", model_version)
        mlflow.log_param("training_date", datetime.now().strftime("%Y-%m-%d"))
        
        # Train KMeans
        kmeans = KMeans(
            n_clusters=n_clusters, 
            init='k-means++', 
            max_iter=500, 
            n_init=10, 
            random_state=random_state, 
            algorithm='elkan'
        )
        
        # Measure training time
        start_time = time.time()
        kmeans.fit(X)
        training_time = time.time() - start_time
        
        # Get cluster labels
        labels = kmeans.labels_
        
        # Calculate evaluation metrics
        silhouette = silhouette_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
        inertia = kmeans.inertia_
        
        # Log metrics
        logger.info(f"KMeans with {n_clusters} clusters - Silhouette: {silhouette:.4f}, "
                   f"Davies-Bouldin: {db_score:.4f}, Calinski-Harabasz: {ch_score:.4f}, "
                   f"Inertia: {inertia:.4f}, Training time: {training_time:.2f}s")
        
        mlflow.log_metric("silhouette_score", silhouette)
        mlflow.log_metric("davies_bouldin_score", db_score)
        mlflow.log_metric("calinski_harabasz_score", ch_score)
        mlflow.log_metric("inertia", inertia)
        mlflow.log_metric("training_time", training_time)
        
        # Calculate cluster distribution
        cluster_counts = np.bincount(labels)
        for i, count in enumerate(cluster_counts):
            mlflow.log_metric(f"cluster_{i}_size", count)
            mlflow.log_metric(f"cluster_{i}_percentage", count / len(labels) * 100)
            
        # Log model
        mlflow.sklearn.log_model(kmeans, "model")
        
        # Get run ID
        run_id = mlflow.active_run().info.run_id
        
        # Store metrics in a dictionary
        metrics = {
            'silhouette': silhouette,
            'davies_bouldin': db_score,
            'calinski_harabasz': ch_score,
            'inertia': inertia,
            'training_time': training_time,
            'run_id': run_id,
            'model_version': model_version
        }
        
    return kmeans, labels, metrics, run_id

def get_best_model_from_mlflow(experiment_name="Luxury_Travel_Customer_Segmentation", 
                              metric="silhouette_score", mode="max"):
    """
    Get the best model from MLflow based on a specific metric.
    
    Parameters:
    -----------
    experiment_name : str, optional
        Name of the MLflow experiment
    metric : str, optional
        Metric to use for model selection
    mode : str, optional
        'max' if higher values are better, 'min' if lower values are better
        
    Returns:
    --------
    tuple
        (best_model, best_run_id, best_metric_value)
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        logger.error(f"Experiment {experiment_name} not found")
        return None, None, None
    
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    
    if not runs:
        logger.warning(f"No runs found for experiment {experiment_name}")
        return None, None, None
    
    best_run = None
    best_metric_value = float('-inf') if mode == 'max' else float('inf')
    
    for run in runs:
        if metric not in run.data.metrics:
            continue
        
        current_value = run.data.metrics[metric]
        
        if (mode == 'max' and current_value > best_metric_value) or \
           (mode == 'min' and current_value < best_metric_value):
            best_metric_value = current_value
            best_run = run
    
    if best_run is None:
        logger.warning(f"No runs found with metric {metric}")
        return None, None, None
    
    best_run_id = best_run.info.run_id
    
    try:
        # Load the model
        model_uri = f"runs:/{best_run_id}/model"
        best_model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded best model from run {best_run_id} with {metric} = {best_metric_value}")
        return best_model, best_run_id, best_metric_value
    except Exception as e:
        logger.error(f"Error loading model from run {best_run_id}: {e}")
        return None, best_run_id, best_metric_value

def compare_and_select_best_model(new_model, new_metrics, metric="silhouette_score", mode="max"):
    """
    Compare a newly trained model with the best existing model in MLflow.
    
    Parameters:
    -----------
    new_model : sklearn.cluster.KMeans
        Newly trained KMeans model
    new_metrics : dict
        Metrics for the newly trained model
    metric : str, optional
        Metric to use for comparison
    mode : str, optional
        'max' if higher values are better, 'min' if lower values are better
        
    Returns:
    --------
    tuple
        (selected_model, selected_run_id, is_new_model_better)
    """
    # Normalize metric name
    mlflow_metric = metric
    if metric == "silhouette":
        mlflow_metric = "silhouette_score"
    elif metric == "davies_bouldin":
        mlflow_metric = "davies_bouldin_score"
    elif metric == "calinski_harabasz":
        mlflow_metric = "calinski_harabasz_score"
    
    # Get the best existing model from MLflow
    best_model, best_run_id, best_metric_value = get_best_model_from_mlflow(
        metric=mlflow_metric, mode=mode
    )
    
    if best_model is None:
        logger.info("No existing model found. Using newly trained model.")
        return new_model, new_metrics['run_id'], True
    
    # Get the corresponding metric value for the new model
    new_metric_value = new_metrics.get(metric.replace("_score", ""))
    
    # Compare the metrics
    if mode == "max":
        is_new_better = new_metric_value > best_metric_value
    else:  # mode == "min"
        is_new_better = new_metric_value < best_metric_value
    
    if is_new_better:
        logger.info(f"New model is better: {metric}={new_metric_value:.4f} vs {best_metric_value:.4f}")
        return new_model, new_metrics['run_id'], True
    else:
        logger.info(f"Existing model is better: {metric}={best_metric_value:.4f} vs {new_metric_value:.4f}")
        return best_model, best_run_id, False

def train_and_select_best_model(X, n_clusters=4, random_state=30, 
                               metric="silhouette_score", mode="max",
                               model_dir="models"):
    """
    Train a new model, compare with existing models, and select the best one.
    
    Parameters:
    -----------
    X : numpy.ndarray
        The data for clustering
    n_clusters : int, optional
        Number of clusters for KMeans
    random_state : int, optional
        Random seed for reproducibility
    metric : str, optional
        Metric to use for model comparison
    mode : str, optional
        'max' if higher values are better, 'min' if lower values are better
    model_dir : str, optional
        Directory to save models
        
    Returns:
    --------
    tuple
        (selected_model, cluster_labels, is_newly_trained)
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Create version string based on current date
    model_version = f"scheduled_{datetime.now().strftime('%Y_%m_%d')}"
    
    # Train new model
    logger.info(f"Training new model with version: {model_version}")
    new_model, new_labels, new_metrics, run_id = train_kmeans_model(
        X, n_clusters=n_clusters, random_state=random_state, model_version=model_version
    )
    
    # Compare with existing models and select the best
    selected_model, selected_run_id, is_new_better = compare_and_select_best_model(
        new_model, new_metrics, metric=metric, mode=mode
    )
    
    # Save the selected model
    model_path = os.path.join(model_dir, "selected_model.joblib")
    joblib.dump(selected_model, model_path)
    logger.info(f"Saved selected model to {model_path}")
    
    # Also save a timestamped version
    timestamp_model_path = os.path.join(model_dir, f"model_{datetime.now().strftime('%Y%m%d')}.joblib")
    joblib.dump(selected_model, timestamp_model_path)
    logger.info(f"Saved timestamped model to {timestamp_model_path}")
    
    # If the new model was selected, use its labels, otherwise predict with the selected model
    if is_new_better:
        selected_labels = new_labels
        logger.info("Using newly trained model (better performance)")
    else:
        selected_labels = selected_model.predict(X)
        logger.info("Using existing model (better performance)")
    
    # Save the selection results to a log file
    results_log_path = os.path.join(model_dir, "model_selection_history.txt")
    with open(results_log_path, 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: ")
        f.write(f"New model: {metric}={new_metrics.get(metric.replace('_score', '')):.4f}, ")
        f.write(f"Selected model: run_id={selected_run_id}, ")
        f.write(f"Is new better: {is_new_better}\n")
    
    return selected_model, selected_labels, is_new_better