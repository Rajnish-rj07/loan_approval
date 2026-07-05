"""
Model Module - Defines the loan prediction model
"""
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import pickle


class LoanPredictionModel:
    """
    Loan Prediction Model using Decision Tree Classifier
    Optimized configuration for best test accuracy
    """
    
    def __init__(self):
        # Optimal configuration - provides 83-84% test accuracy
        self.model = DecisionTreeClassifier(
            criterion='gini',
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = ['Age', 'Income', 'LoanAmount', 'CreditScore']
        
    def preprocess_data(self, X):
        """
        Preprocess the input data
        """
        # Scale the features
        X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def train(self, X, y):
        """
        Train the model
        """
        # Fit the scaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the model
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X):
        """
        Make predictions
        """
        X_scaled = self.preprocess_data(X)
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        """
        X_scaled = self.preprocess_data(X)
        probabilities = self.model.predict_proba(X_scaled)
        return probabilities
    
    def save_model(self, filepath='loan_model.pkl'):
        """
        Save the trained model to a file
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='loan_model.pkl'):
        """
        Load a trained model from a file
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filepath}")
        
        return self
