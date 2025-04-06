import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np

def train_reminder_model():
    """Train a reminder behavior prediction model using processed reminder data."""
    # Check if processed reminder data exists
    if not os.path.exists("data/processed_reminder.pkl"):
        print("Error: Processed reminder data file not found. Run `python utils/data_handler.py` first.")
        exit()
    
    # Load the processed reminder data
    with open("data/processed_reminder.pkl", "rb") as f:
        df = pickle.load(f)
    
    # Print columns to inspect the dataframe
    print("Dataframe Columns:", df.columns.tolist())
    
    # Drop columns that are not needed which could be non-numeric.
    # Adjust column names as appropriate – here we drop common non-numeric fields.
    cols_to_drop = ["Device-ID/User-ID", "Scheduled Time", "Timestamp"]
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    
    # Ensure the target column exists; in this case, "Acknowledged (Yes/No)"
    if "Acknowledged (Yes/No)" not in df.columns:
        print("Error: 'Acknowledged (Yes/No)' column not found in the data!")
        exit()
    
    # Separate features and target
    X = df.drop(columns=["Acknowledged (Yes/No)"])
    y = df["Acknowledged (Yes/No)"]
    
    # Convert remaining features to numeric (forcing non-numeric values to NaN, then fill them)
    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(0, inplace=True)
    
    print("Training Data Preview:\n", X.head())
    
    # Split the data for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Save the trained model
    model_path = "ml_models/reminder_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print("Reminder model trained and saved at", model_path)

if __name__ == "__main__":
    train_reminder_model()