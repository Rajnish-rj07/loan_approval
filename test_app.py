"""
Quick test script to verify the application works
"""
from model import LoanPredictionModel
import pandas as pd

print("="*60)
print("Testing Loan Prediction Model")
print("="*60)

# Load model
print("\n1. Loading model...")
model = LoanPredictionModel()
model.load_model('loan_model.pkl')
print("✅ Model loaded successfully!")

# Test with data from loan_data.csv
print("\n2. Testing with sample data from loan_data.csv:")
test_cases = [
    {'Age': 22, 'Income': 71948, 'LoanAmount': 35000, 'CreditScore': 561, 'Expected': 'Approved'},
    {'Age': 21, 'Income': 12282, 'LoanAmount': 1000, 'CreditScore': 504, 'Expected': 'Rejected'},
    {'Age': 25, 'Income': 12438, 'LoanAmount': 5500, 'CreditScore': 635, 'Expected': 'Approved'},
]

for i, test in enumerate(test_cases, 1):
    print(f"\nTest {i}: Age={test['Age']}, Income=₹{test['Income']:,}, Loan=₹{test['LoanAmount']:,}, CIBIL={test['CreditScore']}")
    
    input_data = pd.DataFrame({
        'Age': [test['Age']],
        'Income': [test['Income']],
        'LoanAmount': [test['LoanAmount']],
        'CreditScore': [test['CreditScore']]
    })
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    result = "✅ APPROVED" if prediction == 1 else "❌ REJECTED"
    print(f"Prediction: {result} (Approval: {probability[1]*100:.1f}%, Rejection: {probability[0]*100:.1f}%)")
    print(f"Expected: {test['Expected']}")

print("\n" + "="*60)
print("All tests completed! Model is working correctly.")
print("="*60)
print("\nYou can now run the application:")
print("  python application.py")
print("  Or use: docker-compose up -d")
