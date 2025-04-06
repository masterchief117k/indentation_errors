import pickle

def load_model(model_path):
    """ Loads a trained ML model from file. """
    with open(model_path, "rb") as f:
        return pickle.load(f)

health_model = load_model("ml_models/health_model.pkl")
safety_model = load_model("ml_models/safety_model.pkl")
reminder_model = load_model("ml_models/reminder_model.pkl")

def predict_health(data):
    """ Make a prediction using the health model. """
    return health_model.predict([data])[0]

def predict_safety(data):
    """ Make a prediction using the safety model. """
    return safety_model.predict([data])[0]

def predict_reminder(data):
    """ Make a prediction using the reminder model. """
    return reminder_model.predict([data])[0]