# Loan Prediction System - Machine Learning Project

**Developer:** Rajnish Kumar Maurya (2201234)

## Project Overview

This is a comprehensive machine learning project that predicts loan approval decisions using a Decision Tree Classifier. The system includes data preprocessing, model training, evaluation, and deployment through a Flask web application.

## Project Structure

```
Loan_Prediction_Rajnish_2201234/
│
├── loan_data.csv           # Dataset containing loan application data
├── app.py                  # Flask web application
├── model.py                # Model class definition
├── train_model.py          # Training script with preprocessing & evaluation
├── prediction.py           # Script for making predictions
├── loan_model.pkl          # Trained model (generated after training)
├── requirements.txt        # Python dependencies
│
├── templates/              # HTML templates for web app
│   ├── index.html         # Home page
│   └── result.html        # Result page
│
├── static/                # Static files
│   └── style.css          # CSS styling
│
└── README.md              # Project documentation
```

## Features

### 1. Data Preprocessing
- Loading and exploring loan data
- Handling missing values
- Removing duplicates
- Detecting and handling outliers using IQR method
- Feature scaling with StandardScaler

### 2. Model Training
- Decision Tree Classifier implementation
- Train-test split with stratification
- Model hyperparameter tuning
- Model persistence using pickle

### 3. Model Evaluation
- Accuracy, Precision, Recall, F1-Score metrics
- Confusion Matrix analysis
- Classification Report
- Training vs Testing performance comparison

### 4. Deployment
- Flask web application
- User-friendly web interface
- RESTful API endpoint
- Real-time predictions

## Dataset

The `loan_data.csv` contains the following features:

- **Age**: Applicant's age
- **Income**: Annual income in dollars
- **LoanAmount**: Requested loan amount
- **CreditScore**: Credit score (300-850)
- **Approved**: Target variable (0 = Rejected, 1 = Approved)

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Steps

1. **Clone or download the project**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Required packages:
   - flask==2.3.3
   - pandas==2.0.3
   - numpy==1.24.3
   - scikit-learn==1.3.0
   - matplotlib==3.7.2
   - seaborn==0.12.2

## Usage

### 1. Train the Model

Run the training script to preprocess data, train the model, and evaluate performance:

```bash
python train_model.py
```

This will:
- Load and explore the data
- Preprocess the dataset
- Split data into training and testing sets
- Train the Decision Tree model
- Evaluate model performance
- Save the model as `loan_model.pkl`

### 2. Make Predictions (Console)

Use the prediction script to make predictions from the command line:

```bash
python prediction.py
```

This will show example predictions with detailed results.

### 3. Run Web Application

Start the Flask web server:

```bash
python app.py
```

Then open your browser and navigate to:
```
http://127.0.0.1:5000/
```

### 4. Using the Web Interface

1. Enter applicant details:
   - Age (18-100)
   - Annual Income ($)
   - Loan Amount ($)
   - Credit Score (300-850)

2. Click "Predict Loan Approval"

3. View the prediction result with:
   - Approval/Rejection decision
   - Probability scores
   - Applicant details summary

## API Usage

The application also provides a RESTful API endpoint:

### Endpoint: `/api/predict`
**Method:** POST  
**Content-Type:** application/json

**Request Body:**
```json
{
    "age": 25,
    "income": 75000,
    "loan_amount": 20000,
    "credit_score": 680
}
```

**Response:**
```json
{
    "prediction": 1,
    "approved": true,
    "approval_probability": 0.85,
    "rejection_probability": 0.15
}
```

## Model Details

### Algorithm: Decision Tree Classifier

**Hyperparameters:**
- `criterion`: 'gini'
- `max_depth`: 10
- `min_samples_split`: 20
- `min_samples_leaf`: 10
- `random_state`: 42

### Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions

## Project Workflow

1. **Data Collection**: Load loan application data from CSV
2. **Data Exploration**: Analyze data distribution and statistics
3. **Data Preprocessing**: Clean and prepare data for training
4. **Data Splitting**: Divide into training and testing sets
5. **Model Training**: Train Decision Tree Classifier
6. **Model Evaluation**: Assess model performance
7. **Model Saving**: Persist trained model to disk
8. **Deployment**: Deploy model via Flask web application

## Technologies Used

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning library
- **Flask**: Web application framework
- **HTML/CSS**: Frontend interface
- **Matplotlib/Seaborn**: Data visualization (for training analysis)

## Future Enhancements

- Add more machine learning algorithms for comparison
- Implement feature importance visualization
- Add model retraining functionality through web interface
- Implement user authentication and history tracking
- Deploy to cloud platform (AWS, Azure, Heroku)
- Add data visualization dashboard
- Implement A/B testing for different models

## Troubleshooting

### Model not found error
If you get a "Model not found" error when running the web app:
```bash
python train_model.py
```
This will create the `loan_model.pkl` file.

### Import errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Port already in use
If port 5000 is already in use, modify `app.py`:
```python
app.run(debug=True, port=5001)  # Change port number
```

## Contact

**Developer:** Rajnish Kumar Maurya  
**ID:** 2201234

## License

This project is developed for educational purposes as part of a Machine Learning course.

---

**Note:** This is a demonstration project. For production use, additional security measures, error handling, and testing should be implemented.
