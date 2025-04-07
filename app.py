from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model and scaler
try:
    model = joblib.load('isolation_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")

@app.route('/')
def home():
    return "Anomaly Detection API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.json.get("data", [])
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Convert to NumPy array
        data_array = np.array(data)

        # Validate input shape
        if data_array.ndim != 2:
            return jsonify({'error': 'Invalid input format. Expected 2D array'}), 400

        # Scale the data
        scaled_data = scaler.transform(data_array)

        # Make predictions
        predictions = model.predict(scaled_data)

        # Prepare response
        response = {
            "predictions": [
                {"index": i, "label": int(pred), "status": "Anomaly" if pred == -1 else "Normal"}
                for i, pred in enumerate(predictions)
            ]
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)