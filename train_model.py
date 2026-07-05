"""
Train Model Script - Handles data preprocessing, model training, and evaluation
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from model import LoanPredictionModel


def load_data(filepath='loan_data.csv'):
    """
    Load the loan data from CSV file
    """
    print("Loading data...")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


def explore_data(df):
    """
    Perform exploratory data analysis
    """
    print("\n" + "="*50)
    print("DATA EXPLORATION")
    print("="*50)
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData Info:")
    print(df.info())
    
    print("\nStatistical Summary:")
    print(df.describe())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nTarget Variable Distribution:")
    print(df['Approved'].value_counts())
    print(f"\nApproval Rate: {df['Approved'].mean() * 100:.2f}%")


def preprocess_data(df):
    """
    Optimal preprocessing - keeps data quality high while retaining samples
    """
    print("\n" + "="*50)
    print("DATA PREPROCESSING")
    print("="*50)
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    print(f"\nTotal missing values: {missing_values}")
    
    if missing_values > 0:
        print("Handling missing values...")
        df = df.fillna(df.mean())
    
    # Remove duplicates
    initial_shape = df.shape[0]
    df = df.drop_duplicates()
    duplicates_removed = initial_shape - df.shape[0]
    print(f"Duplicates removed: {duplicates_removed}")
    
    # Minimal data cleaning - only remove clearly invalid entries
    before = df.shape[0]
    df = df[(df['Age'] >= 18) & (df['Age'] <= 100)]
    df = df[(df['CreditScore'] >= 300) & (df['CreditScore'] <= 850)]
    df = df[(df['Income'] > 0) & (df['LoanAmount'] > 0)]
    
    removed = before - df.shape[0]
    print(f"Invalid entries removed: {removed}")
    
    print(f"\nFinal dataset shape: {df.shape}")
    
    return df


def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    """
    print("\n" + "="*50)
    print("SPLITTING DATA")
    print("="*50)
    
    # Separate features and target
    X = df[['Age', 'Income', 'LoanAmount', 'CreditScore']]
    y = df['Approved']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    print(f"Training set approval rate: {y_train.mean() * 100:.2f}%")
    print(f"Testing set approval rate: {y_test.mean() * 100:.2f}%")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train the loan prediction model with optimal configuration
    """
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    
    # Initialize and train the model
    loan_model = LoanPredictionModel()
    loan_model.train(X_train, y_train)
    
    print("\nModel trained successfully!")
    print(f"Model type: Decision Tree Classifier (Optimal Configuration)")
    print(f"Criterion: {loan_model.model.criterion}")
    print(f"Max depth: {loan_model.model.max_depth}")
    print(f"Min samples split: {loan_model.model.min_samples_split}")
    print(f"Min samples leaf: {loan_model.model.min_samples_leaf}")
    
    return loan_model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate the model performance
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print("\n--- Training Set Performance ---")
    print(f"Accuracy: {train_accuracy * 100:.2f}%")
    
    print("\n--- Testing Set Performance ---")
    print(f"Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Precision: {precision_score(y_test, y_test_pred) * 100:.2f}%")
    print(f"Recall: {recall_score(y_test, y_test_pred) * 100:.2f}%")
    print(f"F1-Score: {f1_score(y_test, y_test_pred) * 100:.2f}%")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_test_pred, target_names=['Rejected', 'Approved']))
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    print(f"\nTrue Negatives: {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives: {cm[1][1]}")
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred),
        'confusion_matrix': cm
    }


def main():
    """
    Main function to run the training pipeline
    """
    print("\n" + "="*50)
    print("LOAN PREDICTION MODEL - TRAINING PIPELINE")
    print("="*50)
    
    # Load data
    df = load_data('loan_data.csv')
    
    # Explore data
    explore_data(df)
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Train model
    loan_model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(loan_model, X_train, X_test, y_train, y_test)
    
    # Save model
    print("\n" + "="*50)
    print("SAVING MODEL")
    print("="*50)
    loan_model.save_model('loan_model.pkl')
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)
    print(f"\nFinal Test Accuracy: {metrics['test_accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall: {metrics['recall'] * 100:.2f}%")
    print(f"F1-Score: {metrics['f1_score'] * 100:.2f}%")
    print("\nModel saved as 'loan_model.pkl'")
    
    # Display feature importance
    if hasattr(loan_model.model, 'feature_importances_'):
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE")
        print("="*50)
        feature_importance = loan_model.model.feature_importances_
        features = loan_model.feature_names
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print("\n", importance_df.to_string(index=False))


if __name__ == "__main__":
    main()
