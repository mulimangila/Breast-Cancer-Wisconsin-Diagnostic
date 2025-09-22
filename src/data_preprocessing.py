import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.logger import logger

def load_data(file_path):
    logger.info(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def preprocess_data(df):
    logger.info("Starting data preprocessing...")
 
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    
    df["diagnosis"] = (df["diagnosis"] == "M").astype(int)
    logger.info("'diagnosis' column encoded as binary.")

    missing_data = df.isnull().sum()
    if missing_data.any():
        logger.warning(f"Missing values found in columns: {missing_data[missing_data > 0].index.tolist()}")
    
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    
    logger.info("Scaling the features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("Features scaled using StandardScaler.")
    
    return X_scaled, y

def split_data(X, y):
    """
    Splits the data into training and test sets (80% train, 20% test).
    """
    logger.info("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logger.info(f"Data split: {X_train.shape[0]} train samples and {X_test.shape[0]} test samples.")
    return X_train, X_test, y_train, y_test
