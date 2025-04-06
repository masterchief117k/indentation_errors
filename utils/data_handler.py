import pandas as pd
import numpy as np

def clean_data(df):
    """
    Cleans a given DataFrame by replacing NaN and infinite values with safe defaults.
    """
    df.replace([np.nan, np.inf, -np.inf], "", inplace=True)
    return df

def prepare_json(data_dict):
    """
    Ensures all values in the JSON dictionary are properly formatted before sending requests.
    """
    for key, value in data_dict.items():
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            data_dict[key] = 0  # Replace invalid float values with zero
    return data_dict
import pandas as pd
import numpy as np
import pickle

def clean_data(df):
    """ Replaces NaN & infinite values with defaults. """
    df.replace([np.nan, np.inf, -np.inf], "", inplace=True)
    return df

def encode_categorical(df, categorical_cols):
    """ Converts categorical columns into numeric encoding for ML models. """
    df_encoded = df.copy()
    for col in categorical_cols:
        df_encoded[col] = df_encoded[col].astype("category").cat.codes
    return df_encoded

def save_processed_data(df, filename):
    """ Saves processed data as a pickle file for ML training. """
    with open(f"data/{filename}", "wb") as f:
        pickle.dump(df, f)

if __name__ == "__main__":
    # Load datasets
    df_health = clean_data(pd.read_csv("data/health_monitoring.csv"))
    df_safety = clean_data(pd.read_csv("data/safety_monitoring.csv"))
    df_reminder = clean_data(pd.read_csv("data/daily_reminder.csv"))

    # Encode categorical features
    df_health = encode_categorical(df_health, ["Heart Rate Below/Above Threshold (Yes/No)", "SpO₂ Below Threshold (Yes/No)"])
    df_safety = encode_categorical(df_safety, ["Fall Detected (Yes/No)"])
    df_reminder = encode_categorical(df_reminder, ["Acknowledged (Yes/No)"])

    # Save processed data
    save_processed_data(df_health, "processed_health.pkl")
    save_processed_data(df_safety, "processed_safety.pkl")
    save_processed_data(df_reminder, "processed_reminder.pkl")
    print(df_health.head())
    print(df_safety.head())
    print(df_reminder.head())
    print("Data preprocessing completed and saved!")