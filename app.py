from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    model = None

@app.route('/')
def home():
    return jsonify({
        "message": "HireSafe Backend API",
        "version": "1.0.0",
        "team": ["Grishma Thakare", "Pranav Kale", "Kirthis Shetty", "Shweta Shetty", "Arham Khan"]
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not found"}), 500
    
    try:
        data = request.get_json()
        
        # Extract features from the request
        # This is a basic example - adjust based on your actual model features
        features = [
            data.get('title', ''),
            data.get('location', ''),
            data.get('department', ''),
            data.get('salary_range', ''),
            data.get('company_profile', ''),
            data.get('description', ''),
            data.get('requirements', ''),
            data.get('benefits', ''),
            data.get('telecommuting', 0),
            data.get('has_company_logo', 1),
            data.get('has_questions', 0),
            data.get('employment_type', ''),
            data.get('required_experience', ''),
            data.get('required_education', ''),
            data.get('industry', ''),
            data.get('function', '')
        ]
        
        # Make prediction (you'll need to preprocess the data according to your model)
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0]
        
        return jsonify({
            "prediction": int(prediction),
            "is_fraud": bool(prediction == 1),
            "confidence": float(max(probability)),
            "fraud_probability": float(probability[1])
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
