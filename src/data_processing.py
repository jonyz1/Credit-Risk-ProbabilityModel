import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from xverse.transformer import WOE
from datetime import datetime

# Custom transformer for extracting temporal features
class TemporalFeatureExtractor:
    def __init__(self, datetime_column='TransactionStartTime'):
        self.datetime_column = datetime_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.datetime_column] = pd.to_datetime(X[self.datetime_column])
        X['TransactionHour'] = X[self.datetime_column].dt.hour
        X['TransactionDay'] = X[self.datetime_column].dt.day
        X['TransactionMonth'] = X[self.datetime_column].dt.month
        X['TransactionYear'] = X[self.datetime_column].dt.year
        return X

# Custom transformer for aggregating features and calculating RFM metrics
class CustomerAggregator:
    def __init__(self, group_by='CustomerId', amount_col='Amount', value_col='Value', datetime_col='TransactionStartTime', snapshot_date=None):
        self.group_by = group_by
        self.amount_col = amount_col
        self.value_col = value_col
        self.datetime_col = datetime_col
        self.snapshot_date = snapshot_date or datetime.now()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
        
        # Calculate RFM metrics
        rfm = X.groupby(self.group_by).agg({
            self.datetime_col: lambda x: (self.snapshot_date - x.max()).days,  # Recency: Days since last transaction
            self.amount_col: ['count', 'sum', 'mean', 'std'],  # Frequency and Monetary metrics
            self.value_col: ['sum', 'mean']
        }).reset_index()
        
        # Flatten column names
        rfm.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in rfm.columns]
        rfm = rfm.rename(columns={
            f'{self.datetime_col}_<lambda>': 'Recency',
            f'{self.amount_col}_count': 'Frequency',
            f'{self.amount_col}_sum': 'Monetary',
            f'{self.amount_col}_mean': 'Amount_mean',
            f'{self.amount_col}_std': 'Amount_std',
            f'{self.value_col}_sum': 'Value_sum',
            f'{self.value_col}_mean': 'Value_mean'
        })
        return rfm

# Custom transformer for K-Means clustering and high-risk labeling
class RFMClusterer:
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

    def fit(self, X, y=None):
        # Scale RFM features for clustering
        rfm_cols = ['Recency', 'Frequency', 'Monetary']
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X[rfm_cols])
        self.kmeans.fit(X_scaled)
        return self

    def transform(self, X):
        X = X.copy()
        rfm_cols = ['Recency', 'Frequency', 'Monetary']
        X_scaled = self.scaler.transform(X[rfm_cols])
        
        # Assign cluster labels
        X['Cluster'] = self.kmeans.predict(X_scaled)
        
        # Identify high-risk cluster (low Frequency, low Monetary, high Recency)
        cluster_stats = X.groupby('Cluster')[rfm_cols].mean()
        high_risk_cluster = cluster_stats[
            (cluster_stats['Frequency'] == cluster_stats['Frequency'].min()) &
            (cluster_stats['Monetary'] == cluster_stats['Monetary'].min())
        ].index[0]
        
        # Assign high-risk label
        X['is_high_risk'] = (X['Cluster'] == high_risk_cluster).astype(int)
        return X

# Function to load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Function to create preprocessing pipeline
def create_preprocessing_pipeline(snapshot_date=None):
    # Define numerical and categorical columns
    numerical_cols = ['Recency', 'Frequency', 'Monetary', 'Amount_mean', 'Amount_std', 'Value_sum', 'Value_mean', 'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']
    categorical_cols = ['ProductCategory', 'ChannelId', 'PricingStrategy']
    
    # Numerical pipeline: Impute missing values and standardize
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: Impute missing values and one-hot encode
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])
    
    # Full pipeline: Temporal features, RFM aggregation, clustering, preprocessing, and WoE
    pipeline = Pipeline([
        ('temporal', TemporalFeatureExtractor()),
        ('aggregator', CustomerAggregator(snapshot_date=snapshot_date)),
        ('clusterer', RFMClusterer()),
        ('preprocessor', preprocessor),
        ('woe', WOE())  # Apply Weight of Evidence transformation
    ])
    
    return pipeline

# Main function to process data
def process_data(input_path, output_path, snapshot_date=None):
    # Load data
    df = load_data(input_path)
    
    # Initialize pipeline
    pipeline = create_preprocessing_pipeline(snapshot_date=snapshot_date)
    
    # Fit and transform data
    processed_data = pipeline.fit_transform(df)
    
    # Convert output to DataFrame
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    processed_df = pd.DataFrame(processed_data, columns=feature_names)
    
    # Add CustomerId and is_high_risk back to the processed data
    aggregator = pipeline.named_steps['aggregator']
    clusterer = pipeline.named_steps['clusterer']
    temp_df = aggregator.transform(df)
    temp_df = clusterer.transform(temp_df)
    processed_df['CustomerId'] = temp_df['CustomerId']
    processed_df['is_high_risk'] = temp_df['is_high_risk']
    
    # Save processed data
    processed_df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    
    return processed_df

if __name__ == "__main__":
    input_file = '../data/raw/xente_data.csv'  # Update path as needed
    output_file = '../data/processed/processed_data.csv'
    snapshot_date = datetime(2025, 7, 2)  # Set snapshot date for Recency calculation
    processed_df = process_data(input_file, output_file, snapshot_date)