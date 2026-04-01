import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer

from src.config import PCA_COLS, AMOUNT_COL, TIME_COL

class TimeTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms the 'Time' column into 'Hour of Day'.
    
    Why this matters in MLOps:
    Raw elapsed seconds from the first transaction have no predictive power 
    for unseen future events. Cyclical 'hour' allows the model to capture 
    patterns like "most fraud happens at 3 AM". Encapsulating this in a 
    custom Sklearn Transformer ensures the same transformation is perfectly 
    reproduced during FastAPI inference without duplicated code.
    """
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_out = X.copy()
        # Time is given in seconds elapsed since first transaction.
        # We assume 1 day = 86400 seconds. We map it to hour.
        X_out[TIME_COL] = (X_out[TIME_COL] % 86400) / 3600.0
        return X_out

def build_preprocessor() -> ColumnTransformer:
    """
    Builds the Scikit-Learn preprocessing pipeline.
    
    Why this matters in MLOps:
    A ColumnTransformer guarantees that scaling rules learned on the training 
    dataset are cleanly serialized and applied to production API payloads.
    - Amount is Highly Skewed: We use RobustScaler to ignore the extreme outliers ($25k).
    - PCA columns: They are already scaled by definition of PCA. We pass them through.
    """
    time_pipe = Pipeline(steps=[
        ("time_transform", TimeTransformer()),
        ("time_scale", RobustScaler())
    ])
    
    amount_pipe = Pipeline(steps=[
        ("amount_scale", RobustScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("time", time_pipe, [TIME_COL]),
            ("amount", amount_pipe, [AMOUNT_COL]),
            ("pca", "passthrough", PCA_COLS)
        ]
    )
    
    return preprocessor

def apply_preprocessing(preprocessor, X_train, X_test):
    """
    Fits preprocessing on train, transforms train and test.
    """
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
    return X_train_prep, X_test_prep
