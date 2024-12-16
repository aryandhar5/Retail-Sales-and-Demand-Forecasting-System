from flask import Flask, request, jsonify, render_template
import joblib  # Use joblib to load the model
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model_path = 'model/xgboost_model.pkl'

try:
    # Use joblib to load the model
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found at {model_path}. Please check the path.")
except Exception as e:
    raise Exception(f"An error occurred while loading the model: {str(e)}")

@app.route('/')
def home():
    # Render the homepage (index.html)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data from the form
        input_data = request.form.to_dict()

        # Check if all necessary fields are present in the input
        required_fields = ['Gender', 'Product_Category', 'Age', 'Quantity', 'Price_per_Unit']
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")

        # Convert input data to a DataFrame
        data = {key: [value] for key, value in input_data.items()}
        data = pd.DataFrame(data)

        # Convert input values to the correct data type (if necessary)
        try:
            data = data.astype({
                'Gender': float,
                'Product_Category': float,
                'Age': float,
                'Quantity': float,
                'Price_per_Unit': float
            })
        except ValueError as ve:
            return jsonify({'error': f"Invalid input values: {str(ve)}"})

        # Make prediction using the loaded model
        prediction = model.predict(data)

        # Format the prediction result
        output = f"Predicted Total Sales: {prediction[0]:,.2f}"

        # Return prediction on the webpage
        return render_template('index.html', prediction_text=output)

    except ValueError as ve:
        # Specific error for invalid or missing input
        return jsonify({'error': str(ve)})

    except Exception as e:
        # General error handling for other issues
        return jsonify({'error': f"An error occurred: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
