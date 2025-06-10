import os
import pandas as pd

from data_preparation import prepare_data
from data_exploration import explore_data
from data_preprocessing import handle_outliers, apply_pca
from model_training import train_clustering_model
from mlflow_tracking import setup_mlflow, log_model_to_mlflow, get_tracking_uri
from model_selection import select_best_model_for_prediction
from data_upload import upload_to_database

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

    # Merge predictions with the original data
    
    # Extract customer data with bookings
    df_customer_result = df_guest_train[df_guest_train['GuestWithBooking'] == 1].copy()
    
    # Merge outlier data
    df_customer_result = pd.merge(
        df_customer_result,
        df_guest_rfm_outlier[['ContactId', 'Outlier']],
        on='ContactId',
        how='left'
    )


    df_customer_result['Clustering_SegmentTitle'] = None
    df_customer_result.loc[df_customer_result['Outlier'] == 1, 'Clustering_SegmentTitle'] = predictions

    # Fill customer marked as outliers with Clustering_SegmentTitle 4
    df_customer_result['Clustering_SegmentTitle'].fillna(4, inplace=True)
    
    
    # Merge customer data without bookings
    df_customer_result = pd.concat([
        df_customer_result,
        df_guest_train[df_guest_train['GuestWithBooking'] == 0]
    ], axis=0)
    
    
    # Fill Clustering_SegmentTitle for customers without booking data
    df_customer_result['Clustering_SegmentTitle'].fillna(5, inplace=True)

    
    # Merge with booking data
    booking_cols = ['Adults', 'Child', 'Infant', 'Nights', 'ContactId', 'MetaGroupName']
    df_gb_result = pd.merge(
        df_customer_result,
        df_booking[booking_cols],
        on='ContactId',
        how='left'
    )
    
    # Upload to SQL database
    table_name = f'clustering_result_{pd.Timestamp.now().strftime("%Y%m%d")}'
    db_uri = """ enter database URI """
    df = df_gb_result.copy()
    upload_to_database(df, table_name, db_url, if_exists='replace', index=False)
    
if __name__ == "__main__":
    main()