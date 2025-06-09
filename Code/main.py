import os
import pandas as pd

from data_preparation import prepare_data
from data_exploration import explore_data
from data_preprocessing import handle_outliers, apply_pca
from model_training import train_clustering_model
from mlflow_tracking import setup_mlflow, log_model_to_mlflow, get_tracking_uri
from model_selection import select_best_model_for_prediction

def main():
    # Set data paths
    guest_data_path = os.path.join("..", "data", "guest_data.xlsx")
    booking_data_path = os.path.join("..", "data", "booking_data.xlsx")

    print('Data preparation')
    df_guest_train, df_booking, df_gb = prepare_data(guest_data_path, booking_data_path)

    print('Data exploration')
    explore_data(df_guest_train, df_gb, df_booking)

    print('Outlier handling and dimensionality reduction')
    df_guest_rfm_outlier = handle_outliers(df_guest_train)
    X_pca_2 = apply_pca(df_guest_rfm_outlier)

    print("Mlflow setup")
    tracking_uri = get_tracking_uri()
    experiment_name = "KMeans_Clustering"
    setup_mlflow(tracking_uri, experiment_name)

    print("Best model selection")
    best_model, model_info = select_best_model_for_prediction(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        metric="silhouette_score",
        high_metric=True
    )

    if best_model is None:
        print("Best model not found.")
        return

    print("Model prediction")
    predictions = best_model.predict(X_pca_2)
    df_result = X_pca_2.copy()
    df_result["Cluster"] = predictions

    # ! Revise
    output_path = os.path.join("..", "data", "prediction_result.csv")
    df_result.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()