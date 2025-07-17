import os
import pandas as pd

from data_preparation import prepare_data
from data_exploration import explore_data
from data_preprocessing import handle_outliers, apply_pca
from model_training import train_kmeans_model
from mlflow_setup import setup_mlflow, log_model_to_mlflow, get_tracking_uri
from model_registry import register_model
from model_prediction import generate_predictions

TRACKING_URI='http://192.168.1.162:5000'
EXPERIMENT_NAME='LuxuryTravelCustomerSegmentation'
MODEL_NAME='LuxuryTravelCustomerSegmentation_KMeans_Model'
def main(retrain=True):

    df_guest_train, df_booking, df_gb = prepare_data(guest_data_path, booking_data_path)

    explore_data(df_guest_train, df_gb, df_booking)

    df_guest_rfm_outlier = handle_outliers(df_guest_train)
    X_pca_2 = apply_pca(df_guest_rfm_outlier)
    
    setup_mlflow(TRACKING_URI=TRACKING_URI, EXPERIMENT_NAME=EXPERIMENT_NAME)
    
    if retrain:
        train_kmeans_model(X_pca_2)
        register_model()
    
    else:
        predictions = generate_predictions(model_name=MODEL_NAME,input_date=X_pca_2)

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
    
if __name__ == "__main__":
    main()