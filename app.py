import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib

app = Flask(__name__, static_folder="static", template_folder="templates")

# Define paths
MODEL_PATH = os.path.join("models", "iris_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
METADATA_PATH = os.path.join("models", "metadata.json")
EVAL_PLOTS_DIR = os.path.join("models", "evaluation_plots")

# Global variables for model resources
model = None
scaler = None
metadata = None

def load_resources():
    global model, scaler, metadata
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(METADATA_PATH)):
        raise FileNotFoundError("Trained model files are missing. Run src/train.py first.")
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

# Load model, scaler, and metadata on startup
try:
    load_resources()
    print("Model resources loaded successfully.")
except Exception as e:
    print(f"Warning: Failed to load model resources: {e}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    if not metadata:
        return jsonify({"status": "error", "message": "Model metadata not loaded"}), 500
    return jsonify(metadata)

@app.route("/api/predict", methods=["POST"])
def predict():
    if not model or not scaler or not metadata:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No input data provided"}), 400
            
        # Extract features
        required_fields = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        inputs = []
        for field in required_fields:
            if field not in data:
                return jsonify({"status": "error", "message": f"Missing field: {field}"}), 400
            try:
                val = float(data[field])
                if val <= 0:
                    return jsonify({"status": "error", "message": f"Value for {field} must be positive"}), 400
                inputs.append(val)
            except ValueError:
                return jsonify({"status": "error", "message": f"Invalid number value for field: {field}"}), 400
                
        # Format input array
        input_arr = np.array([inputs])
        
        # Scale input
        scaled_input = scaler.transform(input_arr)
        
        # Predict class index
        pred_idx = int(model.predict(scaled_input)[0])
        classes = metadata["classes"]
        predicted_species = classes[pred_idx]
        
        # Predict probabilities
        probabilities_dict = {}
        confidence = 1.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(scaled_input)[0]
            confidence = float(probs[pred_idx])
            for idx, p in enumerate(probs):
                probabilities_dict[classes[idx]] = float(p)
        else:
            probabilities_dict[predicted_species] = 1.0
            
        return jsonify({
            "status": "success",
            "prediction": predicted_species,
            "confidence": confidence,
            "probabilities": probabilities_dict
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/plots/<filename>", methods=["GET"])
def get_plot(filename):
    if filename not in ["confusion_matrix.png", "feature_importance.png", "feature_correlation.png", "pca_projection.png"]:
        return jsonify({"status": "error", "message": "File not found"}), 404
    return send_from_directory(EVAL_PLOTS_DIR, filename)

if __name__ == "__main__":
    # In production, we would use a proper WSGI server like gunicorn,
    # but for local development and demo, Flask dev server is ideal.
    app.run(host="127.0.0.1", port=5000, debug=True)
