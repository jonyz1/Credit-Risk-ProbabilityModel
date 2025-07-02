import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# For encoding with WoE (if needed later)
# from woe import WoETransformer

def create_aggregate_features(df):
    """
    Generate aggregate features per CustomerId
    """
    agg_df = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'std'],
        'TransactionId': 'count',
    }).reset_index()

    agg_df.columns = ['CustomerId', 'TotalAmount', 'MeanAmount', 'StdAmount', 'TransactionCount']
    
    return agg_df

def extract_time_features(df):
    """
    Extract features like transaction hour, day, month
    """
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    df['TransactionDay'] = df['TransactionStartTime'].dt.day
    df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    df['TransactionYear'] = df['TransactionStartTime'].dt.year

    return df

def build_pipeline(numeric_features, categorical_features):
    """
    Create transformation pipeline for preprocessing
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor
def create_proxy_target(df, snapshot_date=None):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).reset_index()

    rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Example logic to determine high risk
    high_risk_cluster = rfm.groupby('Cluster')['Recency'].mean().idxmax()
    
    rfm['is_high_risk'] = (rfm['Cluster'] == high_risk_cluster).astype(int)
    
    return df.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
