import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np

def train_safety_model():
    """Train a fall detection model using processed safety data."""
    # Check if processed safety data exists
    if not os.path.exists("data/processed_safety.pkl"):
        print("Error: Processed safety data file not found. Run `python utils/data_handler.py` first.")
        exit()
    
    # Load the processed safety data
    with open("data/processed_safety.pkl", "rb") as f:
        df = pickle.load(f)
    
    # Inspect the dataframe columns for debugging
    print("Dataframe Columns:", df.columns.tolist())
    
    # Drop non-numeric columns that are not needed (adjust the names as necessary)
    cols_to_drop = ["Device-ID/User-ID", "Timestamp"]
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    
    # Check that the target column exists
    if "Fall Detected (Yes/No)" not in df.columns:
        print("Error: 'Fall Detected (Yes/No)' column not found in the data!")
        exit()
    
    # Separate features and target
    X = df.drop(columns=["Fall Detected (Yes/No)"])
    y = df["Fall Detected (Yes/No)"]
    
    # Convert all remaining feature columns to numeric;
    # non-numeric values become NaN (then filled with 0)
    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(0, inplace=True)
    
    print("Training Data Preview:\n", X.head())
    
    # Split data and train the model (using a linear SVM here)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = SVC(kernel="linear", probability=True, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model
    model_path = "ml_models/safety_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print("Safety model trained and saved at", model_path)

if __name__ == "__main__":
    train_safety_model()