import os
import pickle
import pandas as pd
import numpy as np

def load_trained_model(model_path):
    """Load the trained model from a pickle file."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def load_processed_data(file_path):
    """Load processed data from pickle."""
    return pd.read_pickle(file_path)

def prepare_test_data(df):
    """
    Prepare the dataframe for testing:
    - Drop columns that were removed during training.
    - Return X (features) and, if available, the true label.
    """
    # List of columns that were dropped during training:
    cols_to_drop = ["Device-ID/User-ID", "Timestamp"]
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=col)
    
    # The target column is "Alert Triggered (Yes/No)". Separate it out.
    if "Alert Triggered (Yes/No)" in df.columns:
        X = df.drop(columns=["Alert Triggered (Yes/No)"])
        y = df["Alert Triggered (Yes/No)"]
    else:
        X = df
        y = None

    # Convert remaining features to numeric, coercing errors to NaN, then fill missing values with 0.
    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(0, inplace=True)
    
    return X, y

if __name__ == "__main__":
    # Paths to the model and the processed data
    model_path = "ml_models/health_model.pkl"
    processed_data_path = "data/processed_health.pkl"

    # Ensure the processed data file exists
    if not os.path.exists(processed_data_path):
        print(f"Processed data not found. Please run `python utils/data_handler.py` to generate it.")
        exit()
    
    # Load the trained model and the processed data
    model = load_trained_model(model_path)
    df = load_processed_data(processed_data_path)
    
    # Prepare the data (drop columns, convert types, etc.)
    X, y = prepare_test_data(df)
    
    # Randomly select a sample row from X
    random_sample = X.sample(1)
    # Get the true label if available
    true_value = None
    if y is not None:
        true_value = y.loc[random_sample.index[0]]
    
    # Predict using the trained model
    prediction = model.predict(random_sample.values)[0]
    
    print("Random Test Sample:")
    print(random_sample)
    print()
    print("Prediction:", prediction)
    
    if true_value is not None:
        print("Actual value:", true_value)