import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

"""
Segment outliers into  a group with Isolation Forest
"""
def handle_outliers(df_guest_train):
    
    df_guest_rfm = df_guest_train[df_guest_train['GuestWithBooking'] == 1][['ContactId', 'Days', 'EnquiryN', 'QuoteN', 'BookN', 'CommercialValueGBP']]
    
    df_guest_rfm['CommercialValueGBP'].fillna(0, inplace = True)
    df_guest_rfm['Days'].fillna(df_guest_rfm['Days'].max(), inplace = True)
    
    
    # Isolation Forest   
    iso_forest = IsolationForest(n_estimators = 100, contamination = 0.1, random_state = 30)

    X_IsoForest = df_guest_rfm[['Days', 'EnquiryN', 'QuoteN', 'BookN', 'CommercialValueGBP']]
    iso_forest.fit(X_IsoForest)
    predictions_outlier = iso_forest.predict(X_IsoForest)

    df_guest_rfm_outlier = df_guest_rfm.copy()
    df_guest_rfm_outlier['Outlier'] = predictions_outlier

    return df_guest_rfm_outlier

def apply_pca(df_guest_rfm_outlier):
    """
    Apply PCA to the data
    """
    df_guest_rfm = df_guest_rfm_outlier[df_guest_rfm_outlier['Outlier'] == 1].drop('Outlier', axis = 1)
    
    X_pca_input = df_guest_rfm[df_guest_rfm_outlier['Outlier'] == 1].drop('ContactId', axis = 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca_input)

    pca = PCA(n_components = 5)

    X_pca = pca.fit_transform(X_scaled)
    X_pca = pd.DataFrame(X_pca)
    X_pca_2 = X_pca.iloc[:, 0:2]
    
    return X_pca_2
