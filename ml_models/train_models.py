import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder


def train_health_model():
    """ Train a health monitoring prediction model. """
    
    # Load the processed data
    with open("data/processed_health.pkl", "rb") as f:
        df = pickle.load(f)
    
    # Inspect the dataframe columns
    print("Dataframe Columns:", df.columns.tolist())
    
    # Drop columns that are known to be non-numeric
    # Adjust column names if needed (e.g., "Device-ID/User-ID", "Timestamp")
    cols_to_drop = ["Device-ID/User-ID", "Timestamp"]
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    
    # Encode categorical features (if not already encoded)
    label_enc = LabelEncoder()
    categorical_cols = ["Heart Rate Below/Above Threshold (Yes/No)", "SpO₂ Below Threshold (Yes/No)"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = label_enc.fit_transform(df[col])
    
    # At this point, our target is: "Alert Triggered (Yes/No)"
    if "Alert Triggered (Yes/No)" not in df.columns:
        print("Error: 'Alert Triggered (Yes/No)' column not found in the data!")
        exit()
    
    # Separate features and target
    X = df.drop(columns=["Alert Triggered (Yes/No)"])  # Features
    y = df["Alert Triggered (Yes/No)"]  # Target

    # Force all remaining features to numeric.
    # Non-numeric values will become NaN; we then fill them with zero.
    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(0, inplace=True)
    
    # Print a preview of training data
    print("Training Data Preview:\n", X.head())

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the trained model
    model_path = "ml_models/health_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print("Health model trained and saved at", model_path)


def train_safety_model():
    """ Train a fall detection model. """
    with open("data/processed_safety.pkl", "rb") as f:
        df = pickle.load(f)

    X = df.drop(columns=["Fall Detected (Yes/No)"])
    y = df["Fall Detected (Yes/No)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel="linear")
    model.fit(X_train, y_train)

    with open("ml_models/safety_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Safety model trained and saved!")

def train_reminder_model():
    """ Train a behavior prediction model. """
    with open("data/processed_reminder.pkl", "rb") as f:
        df = pickle.load(f)

    X = df.drop(columns=["Acknowledged (Yes/No)"])
    y = df["Acknowledged (Yes/No)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)

    with open("ml_models/reminder_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Reminder model trained and saved!")

if __name__ == "__main__":
    train_health_model()
    train_safety_model()
    train_reminder_model()