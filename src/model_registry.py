import mlflow
from mlflow.tracking import MlflowClient

def register_model(model_name):
    
    """
    Transition the best model version to production and archive others

    Args:
        model_name (str): Name of the model to transition
    Returns:
        best_version (str): The version of the model transitioned to production
    """

    client = MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    
    best_version = None
    best_sil_score = 0

    # Transition the Best Model Version to Production
    for v in versions:
        if v.run_id:
            run = client.get_run(v.run_id)
            val_sil_score = run.data.metrics.get("silihouette_score", float('inf'))
            
            if val_sil_score > best_sil_score:
                best_sil_score = val_sil_score
                best_version = v.version
    
    if best_version:
        client.transition_model_version_stage(
            name=model_name,
            version=best_version,
            stage="Production"
        )
        
        # Archive other versions
        for v in versions:
            if v.version != best_version and v.current_stage != "Archived":
                client.transition_model_version_stage(
                    name=model_name,
                    version=v.version,
                    stage="Archived"
                )
        
        print(f"{best_version} ({best_sil_score:.6f}) transitioned to production")
        return best_version
    else:
        print(f"No production-level model for {model_name}")
        return None