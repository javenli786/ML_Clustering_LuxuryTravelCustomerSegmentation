import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import datetime



def train_kmeans_model(X_data, params=None):
    import mlflow
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # Set default parameters
    parameters = {
        'n_clusters': 4,
        'init': 'k-means++',
        'max_iter': 500,
        'n_init': 10,
        'random_state': 30,
        'algorithm': 'elkan'
    }
    
    now = datetime.now().strftime("%Y%m%d_%H%M")
    
    with mlflow.start_run(run_name=f"KMeans_Model{now}"):
        # Log parameters
        mlflow.log_params(parameters)
        
        # Create and train the KMeans model
        kmeans = KMeans(**parameters)
        
        y_pred = kmeans.fit_predict(X_data)
        
        # Calculate silhouette score
        if len(set(y_pred)) > 1:
            sil_score = silhouette_score(X_data, y_pred)
        else:
            sil_score = 0
            
        # Log metrics
        mlflow.log_metric("silhouette_score", sil_score)
        
        # Log model
        mlflow.sklearn.log_model(model=kmeans, registered_model_name="LuxuryTravelCustomerSegmentation_KMeans_Model")
        
        # Calculate evaluation metrics
        metrics = {
            "silhouette_score": sil_score
        }
        
        return metrics