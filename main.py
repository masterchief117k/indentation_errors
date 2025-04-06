import pandas as pd
import requests
from utils.data_handler import clean_data, prepare_json


import requests
import pandas as pd

df_safety = pd.read_csv("data/safety_monitoring.csv")

# Convert dataframe row to dictionary
safety_data = df_safety.iloc[0].to_dict()

print("Sending Safety Data:", safety_data)  # Debug print

response = requests.post("http://127.0.0.1:5000/safety", json=safety_data)
print("Raw Response:", response.text)  # Debug Flask's response
print("Response Status Code:", response.status_code)

import requests
import pandas as pd

df_safety = pd.read_csv("data/safety_monitoring.csv")

# Convert dataframe row to dictionary
safety_data = df_safety.iloc[0].to_dict()
print("Sending Safety Data:", safety_data)  # Debug print before sending

response = requests.post("http://127.0.0.1:5000/safety", json=safety_data)
print("Safety Agent Response:", response.text)
# Load and clean data
df_health = clean_data(pd.read_csv("data/health_monitoring.csv"))
df_safety = clean_data(pd.read_csv("data/safety_monitoring.csv"))
df_reminder = clean_data(pd.read_csv("data/daily_reminder.csv"))

# Convert to valid JSON format
health_data = prepare_json(df_health.iloc[0].to_dict())
safety_data = prepare_json(df_safety.iloc[0].to_dict())
reminder_data = prepare_json(df_reminder.iloc[0].to_dict())

# Send data to Flask API
headers = {"Content-Type": "application/json"}
response = requests.post("http://127.0.0.1:5000/safety", json=safety_data, headers=headers)
health_response = requests.post("http://127.0.0.1:5000/health", json=health_data)
safety_response = requests.post("http://127.0.0.1:5000/safety", json=safety_data)
reminder_response = requests.post("http://127.0.0.1:5000/reminder", json=reminder_data)

# Print responses
print("Health Agent:", health_response.json())
print("Safety Agent:", safety_response.json())
print("Reminder Agent:", reminder_response.json())