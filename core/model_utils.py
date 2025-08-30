from joblib import load
import os

MODEL_PATH = "models/forex_model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")
    model = load(MODEL_PATH)
    if model is None:
        raise ValueError("❌ Loaded model is empty or invalid.")
    return model
