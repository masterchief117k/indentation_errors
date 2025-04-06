import pandas as pd

class SafetyMonitoringAgent:
    def __init__(self, safety_data_file):
        self.data = pd.read_csv(safety_data_file)

    def detect_falls(self):
        alerts = []
        for _, row in self.data.iterrows():
            if row["Fall Detected (Yes/No)"] == "Yes" or row["Impact Force Level"] == "Yes":
                alerts.append({"alert": "Fall detected! Emergency response needed.", "data": row.to_dict()})
        return alerts

if __name__ == "__main__":
    safety_agent = SafetyMonitoringAgent("../data/safety_monitoring.csv")
    print(safety_agent.detect_falls())