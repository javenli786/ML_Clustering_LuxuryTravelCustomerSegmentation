import pandas as pd
from sqlalchemy import create_engine, engine, text


def upload_df_to_sql(df, table_name, db_url, if_exists='replace', index=False):
    """
    Upload a DataFrame to a SQL database

    Args:
        df (pd.DataFrame): DataFrame to upload
        table_name (str): Table name
        db_url (str): Database URL
        if_exists (str): 'replace', 'append', or 'fail'
        index (bool): Whether to write DataFrame index as a column
    """
    engine = create_engine(db_url)
    df.to_sql(table_name, engine, if_exists=if_exists, index=index)

    print(f"{table_name} uploaded to {db_url} (if_exists='{if_exists}')")
