"""
Loan Prediction Application - Indian Rupee Version with EMI Calculator
"""
from flask import Flask, render_template, request, jsonify
import pandas as pd
from model import LoanPredictionModel
import os

# Create Flask application
application = Flask(__name__)
app = application  # Alias for local development

# Load the trained model
model = LoanPredictionModel()
model_path = 'loan_model.pkl'

# Check if model file exists and load it
if not os.path.exists(model_path):
    print(f"ERROR: Model file '{model_path}' not found!")
    print(f"Current directory: {os.getcwd()}")
else:
    try:
        model.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()


@app.route('/')
def home():
    """
    Home page
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction request with Indian Rupee format
    """
    try:
        # Check if model is loaded properly
        if not hasattr(model, 'model') or not hasattr(model, 'scaler'):
            return render_template('result.html', 
                                   error="Model not loaded properly. Please restart the application and ensure loan_model.pkl exists.")
        
        # Get data from form
        age = int(request.form['age'])
        income = float(request.form['income'])
        loan_amount = float(request.form['loan_amount'])
        credit_score = int(request.form['credit_score'])
        
        # Validate inputs
        if age < 18 or age > 100:
            return render_template('result.html', 
                                   error="Age must be between 18 and 100")
        
        if income <= 0:
            return render_template('result.html', 
                                   error="Income must be a positive number")
        
        if loan_amount <= 0:
            return render_template('result.html', 
                                   error="Loan amount must be a positive number")
        
        if credit_score < 300 or credit_score > 900:
            return render_template('result.html', 
                                   error="Credit score must be between 300 and 900")
        
        # Create input dataframe for model
        # Cap credit score at 850 for model compatibility (trained on 300-850 range)
        model_credit_score = min(credit_score, 850)
        
        input_data = pd.DataFrame({
            'Age': [age],
            'Income': [income],
            'LoanAmount': [loan_amount],
            'CreditScore': [model_credit_score]
        })
        
        print(f"Input data: Age={age}, Income={income}, Loan={loan_amount}, CIBIL={credit_score}")
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        print(f"Prediction: {prediction} (1=Approved, 0=Rejected)")
        print(f"Probabilities: Reject={probability[0]:.2%}, Approve={probability[1]:.2%}")
        
        # Prepare result
        result = {
            'age': age,
            'income': income,
            'loan_amount': loan_amount,
            'credit_score': credit_score,
            'prediction': int(prediction),
            'approved': bool(prediction == 1),
            'approval_probability': float(probability[1] * 100),
            'rejection_probability': float(probability[0] * 100)
        }
        
        return render_template('result.html', result=result)
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error in prediction:")
        print(error_detail)
        return render_template('result.html', 
                               error=f"An error occurred: {str(e)}. Please check if the model is trained.")


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for prediction
    """
    try:
        data = request.get_json()
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'Age': [data['age']],
            'Income': [data['income']],
            'LoanAmount': [data['loan_amount']],
            'CreditScore': [min(data['credit_score'], 850)]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Return JSON response
        return jsonify({
            'prediction': int(prediction),
            'approved': bool(prediction == 1),
            'approval_probability': float(probability[1]),
            'rejection_probability': float(probability[0])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/health')
def health():
    """
    Health check endpoint
    """
    model_loaded = hasattr(model, 'model') and hasattr(model, 'scaler')
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded
    })


if __name__ == "__main__":
    # Local development
    print("Starting Loan Prediction System...")
    print(f"Current directory: {os.getcwd()}")
    print(f"Model file exists: {os.path.exists('loan_model.pkl')}")
    app.run(debug=True, host='0.0.0.0', port=8080)
