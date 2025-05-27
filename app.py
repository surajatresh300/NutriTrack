from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load the trained model
model_path = os.path.join('ML Project', 'macro_predictor_model.pkl')
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully.")
except FileNotFoundError:
    logger.error(f"Model file not found: {model_path}")
    model = None

# Expected columns as used during training
expected_columns = ['age', 'gender', 'height', 'weight', 'activity_level', 'goal']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()

    if not data:
        return jsonify({'error': 'No JSON data received'}), 400

    if not all(key in data for key in expected_columns):
        return jsonify({'error': 'Missing required input fields'}), 400

    try:
        # Construct input dataframe directly
        input_data = pd.DataFrame({key: [data[key]] for key in expected_columns})

        # Check column order and missing
        input_data = input_data[expected_columns]

        logger.info(f"Input data:\n{input_data}")

        # Apply the preprocessing steps (with handle_unknown='ignore' to avoid errors on unknown categories)
        preprocessor = model.named_steps['preprocessor']
        input_data_transformed = preprocessor.transform(input_data)

        predictions = model.predict(input_data)

        result = {
            'protein': round(predictions[0][0], 1),
            'carbs': round(predictions[0][1], 1),
            'fats': round(predictions[0][2], 1),
            'water': round(predictions[0][3], 2)
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False') == 'True'
    app.run(debug=debug_mode)
