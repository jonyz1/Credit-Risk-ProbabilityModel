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
