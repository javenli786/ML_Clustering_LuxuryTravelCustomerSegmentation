import mlflow
from mlflow.tracking import MlflowClient

def generate_predictions(model_name, input_data):
    """
    Predict with the production model version
    
    Args:
        model_name (str): Name of the model
        input_data (dict): Input data for prediction
        
    Returns:
        dict: Prediction results.
    """
    client = MlflowClient()
    
    # Generate the production model version
    production_version = client.get_model_version(model_name, "Production")
    
    if not production_version:
        print(f"No production version found for model {model_name}")
    
    # Load the model
    model_uri = f"models:/{model_name}/{production_version.version}"
    model = mlflow.keras.load_model(model_uri)
    
    # Generate predictions
    predictions = model.predict(input_data)
    
    return predictions