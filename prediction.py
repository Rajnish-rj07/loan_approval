"""
Prediction Script - Load the trained model and make predictions
"""
import pandas as pd
import numpy as np
from model import LoanPredictionModel


def load_trained_model(model_path='loan_model.pkl'):
    """
    Load the trained model
    """
    model = LoanPredictionModel()
    model.load_model(model_path)
    return model


def predict_single(model, age, income, loan_amount, credit_score):
    """
    Make a prediction for a single applicant
    """
    # Create input dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'Income': [income],
        'LoanAmount': [loan_amount],
        'CreditScore': [credit_score]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    return prediction, probability


def display_prediction(age, income, loan_amount, credit_score, prediction, probability):
    """
    Display prediction results in a formatted way
    """
    print("\n" + "="*60)
    print("LOAN PREDICTION RESULT")
    print("="*60)
    
    print("\nApplicant Information:")
    print(f"  Age: {age} years")
    print(f"  Income: ${income:,.2f}")
    print(f"  Loan Amount: ${loan_amount:,.2f}")
    print(f"  Credit Score: {credit_score}")
    
    print("\n" + "-"*60)
    
    if prediction == 1:
        print("\n✓ LOAN APPROVED")
        print(f"  Approval Probability: {probability[1] * 100:.2f}%")
        print(f"  Rejection Probability: {probability[0] * 100:.2f}%")
    else:
        print("\n✗ LOAN REJECTED")
        print(f"  Rejection Probability: {probability[0] * 100:.2f}%")
        print(f"  Approval Probability: {probability[1] * 100:.2f}%")
    
    print("\n" + "="*60 + "\n")


def predict_batch(model, csv_file):
    """
    Make predictions for multiple applicants from a CSV file
    """
    # Load data
    df = pd.read_csv(csv_file)
    
    # Extract features
    X = df[['Age', 'Income', 'LoanAmount', 'CreditScore']]
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Add predictions to dataframe
    df['Prediction'] = predictions
    df['Approval_Probability'] = probabilities[:, 1]
    df['Prediction_Label'] = df['Prediction'].map({0: 'Rejected', 1: 'Approved'})
    
    return df


def main():
    """
    Main function for making predictions
    """
    print("\n" + "="*60)
    print("LOAN PREDICTION SYSTEM")
    print("="*60)
    
    # Load the trained model
    print("\nLoading trained model...")
    model = load_trained_model('loan_model.pkl')
    
    print("\nModel loaded successfully!")
    print("\nYou can now make predictions.")
    
    # Example predictions
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    
    # Example 1: Likely to be approved
    print("\n--- Example 1 ---")
    age1, income1, loan1, credit1 = 25, 75000, 20000, 680
    pred1, prob1 = predict_single(model, age1, income1, loan1, credit1)
    display_prediction(age1, income1, loan1, credit1, pred1, prob1)
    
    # Example 2: Likely to be rejected
    print("\n--- Example 2 ---")
    age2, income2, loan2, credit2 = 23, 25000, 35000, 520
    pred2, prob2 = predict_single(model, age2, income2, loan2, credit2)
    display_prediction(age2, income2, loan2, credit2, pred2, prob2)
    
    # Example 3: Good profile
    print("\n--- Example 3 ---")
    age3, income3, loan3, credit3 = 30, 100000, 25000, 720
    pred3, prob3 = predict_single(model, age3, income3, loan3, credit3)
    display_prediction(age3, income3, loan3, credit3, pred3, prob3)


if __name__ == "__main__":
    main()
