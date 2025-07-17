import numpy as np
import pandas as pd

def prepare_data(guest_data, booking_data):
    
    """
    Import the data
    """

    df_guest = pd.read_excel(guest_data)
    df_booking = pd.read_excel(booking_data)
    
    """
    Observe the data
    """
    
    # Head
    df_guest.head(5)
    df_booking.head(5)
    
    # Info
    df_guest.info()
    df_booking.info()
    
    # Missing values
    df_guest.isnull().sum()
    df_booking.isnull().sum()
    
    #Unique values
    for c in df_guest.columns:
        print(c)
        print(df_guest[c].unique())
        print('')
    
    for c in df_booking.columns:
        print(c)
        print(df_booking[c].unique())
        print('')
    

    """
    Preprocess the data
    """
    
    # 'df_guest'
    # Fill in missing values
    df_guest['SegmentTitle'] = df_guest['SegmentTitle'].fillna('Unknown')
    df_guest['BusinessUnitName'] = df_guest['BusinessUnitName'].fillna('Unknown')
    df_guest['Country'] = df_guest['Country'].fillna('Unknown')
    df_guest['LoyaltyClub'] = df_guest['LoyaltyClub'].fillna('NoLoyaltyClub')

    # 'df_booking'
    df_booking['Nights'] = df_booking['Nights'].apply(lambda x: abs(x) if not isinstance(x, str) else x)
    booktype_dict = {'Fam': 'Family or friends'}
    df_booking['BookType'] = df_booking['BookType'].replace(booktype_dict)
    businessunit_dict = {
        'Scott Dunn Hong Kong': 'ScottDunn Hong Kong',
        'CH-Hong Kong': 'ScottDunn Hong Kong',
        'Scott Dunn China': 'ScottDunn China',
        'Scott Dunn Dubai': 'ScottDunn Dubai',
        'CH-Singapore': 'ScottDunn Singapore'
    }
    df_booking['BusinessUnit'] = df_booking['BusinessUnit'].replace(businessunit_dict)
    df_booking['BookDate'] = pd.to_datetime(df_booking['BookDate'], errors='coerce')
    df_booking['EnquiryDate'] = pd.to_datetime(df_booking['EnquiryDate'], errors='coerce')
    df_booking['QuoteDate'] = pd.to_datetime(df_booking['QuoteDate'], errors='coerce')
    df_booking['CommercialValueGBP'] = df_booking['CommercialValueGBP'].fillna(0)

    # Merge the guest and booking data
    df_gb = pd.merge(df_guest, df_booking, on='ContactId')

    """
    Feature Engineering
    """
    
    # Create the training data
    df_guest_train = df_guest.copy()

    
    # Recency
    df_gb['LaterDate'] = df_gb[['BookDate', 'EnquiryDate', 'QuoteDate']].max(axis=1)
    latest_date = df_gb.groupby('ContactId')['LaterDate'].max().reset_index()
    
    df_guest_train = pd.merge(df_guest_train, latest_date, on=['ContactId'], how='outer')
    df_guest_train.rename(columns={'LaterDate': 'LatestDate'}, inplace=True)
    max_date = df_guest_train['LatestDate'].max()
    df_guest_train['Days'] = df_guest_train['LatestDate'].apply(lambda x: (max_date - x).days if pd.notnull(x) else None)

    # Frequency
    enquiry_n = df_booking['EnquiryDate'].notnull().groupby(df_booking['ContactId']).sum()
    quote_n = df_booking['QuoteDate'].notnull().groupby(df_booking['ContactId']).sum()
    book_n = df_booking['BookDate'].notnull().groupby(df_booking['ContactId']).sum()
    df_guest_train = pd.merge(df_guest_train, enquiry_n, on='ContactId', how='outer')
    df_guest_train = pd.merge(df_guest_train, quote_n, on='ContactId', how='outer')
    df_guest_train = pd.merge(df_guest_train, book_n, on='ContactId', how='outer')
    df_guest_train.rename({'EnquiryDate': 'EnquiryN', 'QuoteDate': 'QuoteN', 'BookDate': 'BookN'}, axis=1, inplace=True)

    # Monetary
    CommercialValue_sum = df_gb.groupby('ContactId')['CommercialValueGBP'].sum()
    df_guest_train = pd.merge(df_guest_train, CommercialValue_sum, on='ContactId', how='outer')

    # Mark the customers with booking records
    df_guest_train['GuestWithBooking'] = np.where(
        df_guest_train[['Days', 'QuoteN', 'EnquiryN', 'BookN', 'CommercialValueGBP']].notnull().any(axis=1), 1, 0
    )

    return df_guest_train, df_booking, df_gb