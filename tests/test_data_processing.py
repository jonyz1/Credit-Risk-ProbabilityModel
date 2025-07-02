import pandas as pd
import pytest
from src.data_processing import TemporalFeatureExtractor, CustomerAggregator

# Test fixture for sample data
@pytest.fixture
def sample_data():
    data = {
        'CustomerId': ['C1', 'C1', 'C2'],
        'TransactionStartTime': ['2025-06-01 10:00:00', '2025-06-02 12:00:00', '2025-06-01 15:00:00'],
        'Amount': [100, -50, 200],
        'Value': [100, 50, 200]
    }
    return pd.DataFrame(data)

# Test TemporalFeatureExtractor
def test_temporal_feature_extractor(sample_data):
    transformer = TemporalFeatureExtractor()
    transformed_df = transformer.transform(sample_data)
    
    # Check if new temporal columns are added
    expected_columns = ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']
    for col in expected_columns:
        assert col in transformed_df.columns, f"{col} not found in transformed data"
    
    # Check if TransactionHour is correct for first row
    assert transformed_df['TransactionHour'].iloc[0] == 10, "Incorrect TransactionHour"

# Test CustomerAggregator
def test_customer_aggregator(sample_data):
    transformer = CustomerAggregator(snapshot_date=pd.to_datetime('2025-07-02'))
    transformed_df = transformer.transform(sample_data)
    
    # Check if aggregated columns are present
    expected_columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary', 'Amount_mean', 'Amount_std', 'Value_sum', 'Value_mean']
    for col in expected_columns:
        assert col in transformed_df.columns, f"{col} not found in aggregated data"
    
    # Check if aggregation for CustomerId 'C1' is correct
    c1_data = transformed_df[transformed_df['CustomerId'] == 'C1']
    assert c1_data['Frequency'].iloc[0] == 2, "Incorrect Frequency for C1"
    assert c1_data['Monetary'].iloc[0] == 50, "Incorrect Monetary for C1"