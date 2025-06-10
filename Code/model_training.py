import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score



def train_clustering_model(
    X_data, 
    n_clusters=4, 
    init='k-means++', 
    max_iter=500, 
    n_init=10, 
    random_state=30, 
    algorithm='elkan'
):
    
    # Create and train the KMeans model
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
        algorithm=algorithm
    )
    
    y_pred = kmeans.fit_predict(X_data)
    
    # Calculate evaluation metrics
    metrics = {}
    
    return {
        "model": kmeans,
        "predictions": y_pred,
        "metrics": metrics,
        "parameters": {
            "n_clusters": n_clusters,
            "init": init,
            "max_iter": max_iter,
            "n_init": n_init,
            "random_state": random_state,
            "algorithm": algorithm
        }
    }