import pandas as pd
from utils.data_handler import clean_data, prepare_json

class HealthMonitoringAgent:
    def __init__(self, health_data_file):
        try:
            self.data = pd.read_csv(health_data_file)
            self.data = clean_data(self.data)  # Clean data before processing
        except FileNotFoundError:
            print(f"Error: File {health_data_file} not found.")
            self.data = pd.DataFrame()  # Empty dataframe for safety

    def check_health(self):
        alerts = []
        if self.data.empty:
            return [{"error": "No valid health data available"}]
        
        for _, row in self.data.iterrows():
            if row.get("SpO₂ Below Threshold (Yes/No)") == "Yes" or row.get("Glucose Levels Below/Above Threshold (Yes/No)") == "Yes":
                alerts.append({"alert": "Health anomaly detected!", "data": prepare_json(row.to_dict())})
        
        return alerts

# 🛠️ Add a main function to execute
if __name__ == "__main__":
    agent = HealthMonitoringAgent("../data/health_monitoring.csv")  # Adjust path if needed
    alerts = agent.check_health()
    
    if alerts:
        for alert in alerts:
            print(alert)
    else:
        print("No health anomalies detected.")