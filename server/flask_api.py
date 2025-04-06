from flask import Flask, request, jsonify
from flask import Flask, request, jsonify
import sys
import os

# Add project root to sys.path so that ml_models becomes importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml_models.model_utils import predict_health, predict_safety, predict_reminder


app = Flask(__name__)

@app.route('/health', methods=['POST'])
def health_monitor():
    data = request.json
    if data.get("SpO₂ Below Threshold") == "Yes" or data.get("Glucose Levels Below Threshold") == "Yes":
        return jsonify({"alert": "Health anomaly detected! Notifying caregiver."})
    return jsonify({"status": "Vitals are normal."})
@app.route('/safety', methods=['POST'])
def safety_monitor():
    try:
        data = request.json  # Extract JSON request data
        
        print("Received Safety Data:", data)  # Debugging print statement
        
        if not data:
            return jsonify({"error": "No JSON data received"}), 400  # Handle missing JSON
        
        if data.get("Fall Detected (Yes/No)") == "Yes" or data.get("Impact Force Level", 0) == "Yes":
            return jsonify({"alert": "Fall detected! Emergency response triggered."})

        return jsonify({"status": "No safety issues."})
    
    except Exception as e:
        print(f"Error in /safety endpoint: {str(e)}")
        return jsonify({"error": f"Internal Server Error - {str(e)}"}), 500

@app.route('/health/predict', methods=['POST'])
def health_predict():
    data = request.json
    prediction = predict_health(list(data.values()))
    return jsonify({"prediction": prediction})

@app.route('/safety/predict', methods=['POST'])
def safety_predict():
    data = request.json
    prediction = predict_safety(list(data.values()))
    return jsonify({"prediction": prediction})

@app.route('/reminder/predict', methods=['POST'])
def reminder_predict():
    data = request.json
    prediction = predict_reminder(list(data.values()))
    return jsonify({"prediction": prediction})




@app.route('/reminder', methods=['POST'])
def reminder_agent():
    data = request.json
    if data.get("Acknowledged") == "No":
        return jsonify({"alert": "Reminder not acknowledged! Follow-up required."})
    return jsonify({"status": "Reminder acknowledged."})
if __name__ == '__main__':
    app.run(port=5000)