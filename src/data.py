import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import TEST_SIZE, RANDOM_STATE, TARGET_COL

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the credit card fraud dataset.
    
    Why this matters in MLOps:
    Having a centralized data loading function ensures that pipeline scripts 
    and exploratory notebooks all read from the same source of truth.
    """
    df = pd.read_csv(filepath)
    return df

def split_data(df: pd.DataFrame):
    """
    Splits the dataframe into training and testing sets.
    
    Why this matters in MLOps:
    We use stratify=y heavily because of the extreme class imbalance (0.17%).
    If we don't, we might end up with no fraud cases in our test set, rendering 
    our evaluation useless.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y  # Essential for imbalanced datasets
    )
    
    return X_train, X_test, y_train, y_test
