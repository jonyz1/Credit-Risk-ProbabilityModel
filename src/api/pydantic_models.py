from pydantic import BaseModel

class PredictionInput(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float
    Amount_mean: float
    Amount_std: float
    Value_sum: float
    Value_mean: float
    TransactionHour: float
    TransactionDay: float
    TransactionMonth: float
    TransactionYear: float
    ProductCategory_airtime: float = 0.0
    ProductCategory_data_bundles: float = 0.0
    ProductCategory_financial_services: float = 0.0
    ProductCategory_movies: float = 0.0
    ProductCategory_other: float = 0.0
    ProductCategory_retail: float = 0.0
    ProductCategory_transport: float = 0.0
    ProductCategory_tv: float = 0.0
    ProductCategory_utility_bill: float = 0.0
    ChannelId_ChannelId_2: float = 0.0
    ChannelId_ChannelId_3: float = 0.0
    ChannelId_ChannelId_5: float = 0.0
    PricingStrategy_1: float = 0.0
    PricingStrategy_2: float = 0.0
    PricingStrategy_4: float = 0.0

class PredictionOutput(BaseModel):
    risk_probability: float