#!/usr/bin/env python
# coding: utf-8

# # Hotel Booking Cancellation Prediction using XGBoost
# 
# This script implements XGBoost for hotel booking cancellation prediction.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('../data/hotel_bookings.csv')

# Display basic information about the dataset
print(f"Dataset Shape: {df.shape}")
print(df.head())

# Create a copy of the data to avoid any changes to original data
data = df.copy()

# Data preprocessing
# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Fill missing values appropriately
data['children'].fillna(0, inplace=True)
data['country'].fillna('Unknown', inplace=True)
data['agent'].fillna(0, inplace=True)
data['company'].fillna(0, inplace=True)

print("\nChecking data types:")
print(data.dtypes)

# Convert 'is_canceled' to a binary target variable
y = data['is_canceled']

# Convert 'arrival_date' to datetime (combining the year, month, day columns)
print("\nCreating date features...")
data['arrival_date'] = pd.to_datetime(
    data['arrival_date_year'].astype(str) + '-' + 
    data['arrival_date_month'] + '-' + 
    data['arrival_date_day_of_month'].astype(str),
    errors='coerce'
)

# Extract month from arrival_date
data['arrival_month'] = data['arrival_date'].dt.month

# Create a feature for lead time categories
data['lead_time_category'] = pd.cut(
    data['lead_time'], 
    bins=[0, 30, 90, 180, 365, float('inf')],
    labels=['Last Minute', 'Short', 'Medium', 'Long', 'Very Long']
)

print("\nProcessing categorical features...")
# Feature selection - dropping unnecessary columns
cols_to_drop = [
    'arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month',
    'arrival_date_week_number', 'reservation_status', 'reservation_status_date',
    'arrival_date', 'is_canceled'
]
X = data.drop(cols_to_drop, axis=1)

# Handle categorical variables
categorical_columns = X.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Split the data into training and testing sets
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Define a function to visualize confusion matrix
def plot_confusion_matrix(model, X, y, title='Confusion Matrix'):
    """
    Plot confusion matrix with percentages
    """
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    # Calculate percentages
    total = cm.sum()
    percentage_cm = (cm / total) * 100
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Reds)
    
    # Add labels with counts and percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]}\n{percentage_cm[i, j]:.2f}%", 
                    ha='center', va='center', color='white' if percentage_cm[i, j] > 10 else 'black',
                    fontsize=12, fontweight='bold')
    
    plt.title(title, fontsize=15)
    plt.show()
    
    # Return the confusion matrix values
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp

# Function to evaluate model and print metrics
def evaluate_model(model, X, y, dataset_name=""):
    """
    Evaluate model performance and print metrics
    """
    y_pred = model.predict(X)
    print(f"\nModel evaluation on {dataset_name} data:")
    print(classification_report(y, y_pred))
    
    # Get confusion matrix values
    tn, fp, fn, tp = plot_confusion_matrix(model, X, y, f"XGBoost - {dataset_name} Confusion Matrix")
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    print(f"Confusion Matrix Values:")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN): {tn}")
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return accuracy, precision, recall, f1, tn, fp, fn, tp

# XGBoost Model Training
print("\nTraining initial XGBoost model...")
# Initial XGBoost Model with default parameters
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    n_jobs=-1
)

# Train the model
xgb_model.fit(X_train, y_train)

# Evaluate model on training data
train_metrics = evaluate_model(xgb_model, X_train, y_train, "Training")

# Evaluate model on test data
test_metrics = evaluate_model(xgb_model, X_test, y_test, "Test")

# Display feature importance
print("\nFeature Importance Analysis:")
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_model.feature_importances_
})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False).reset_index(drop=True)
print(feature_importance.head(20))

# Plot feature importance
plt.figure(figsize=(12, 10))
xgb.plot_importance(xgb_model, max_num_features=15)
plt.title('Feature Importance (XGBoost)', fontsize=15)
plt.show()

# Hyperparameter Tuning with GridSearchCV
print("\nPerforming hyperparameter tuning (this may take some time)...")

# Define parameter grid (using a smaller grid for demo purposes)
param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.2],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0]
}

# Use a smaller subset for hyperparameter tuning to save time
X_tune, _, y_tune, _ = train_test_split(X_train, y_train, test_size=0.7, random_state=42)

# GridSearchCV with cross-validation
grid_search = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic', random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

# Fit the grid search
grid_search.fit(X_tune, y_tune)

# Best parameters
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best F1 score: {grid_search.best_score_:.4f}")

# Train the final model with the best parameters
print("\nTraining final model with best parameters...")
final_model = xgb.XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    **grid_search.best_params_
)

# Train on the full training set
final_model.fit(X_train, y_train)

# Evaluate the final model on training data
print("\nFinal model evaluation:")
final_train_metrics = evaluate_model(final_model, X_train, y_train, "Training (Tuned)")

# Evaluate final model on test data
final_test_metrics = evaluate_model(final_model, X_test, y_test, "Test (Tuned)")

# Save the final model
model_filename = 'hotel_cancellation_prediction_xgboost_model.joblib'
joblib.dump(final_model, model_filename)
print(f"\nModel saved as '{model_filename}'")

# Model comparison
print("\nModel Performance Comparison:")
comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'XGBoost (Default)': [test_metrics[0], test_metrics[1], test_metrics[2], test_metrics[3]],
    'XGBoost (Tuned)': [final_test_metrics[0], final_test_metrics[1], final_test_metrics[2], final_test_metrics[3]]
})
print(comparison_df)

# Function for prediction
def predict_cancellation(data_dict):
    """
    Function to predict the probability of cancellation given input features
    """
    # Create a dataframe from input data
    input_df = pd.DataFrame([data_dict])
    
    # Ensure all features are properly processed (same as in training)
    for col in categorical_columns:
        if col in input_df.columns and col in X.columns:
            le = LabelEncoder()
            le.fit(data[col].astype(str))
            input_df[col] = le.transform(input_df[col].astype(str))
    
    # Ensure the input has all required columns
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Default value
    
    # Select only the features used in training
    input_df = input_df[X.columns]
    
    # Make prediction
    cancellation_prob = final_model.predict_proba(input_df)[0, 1]
    will_cancel = cancellation_prob > 0.5
    
    return {
        "will_cancel": bool(will_cancel),
        "cancellation_probability": float(cancellation_prob),
        "model": "XGBoost (Tuned)"
    }

# Example usage
example_booking = {
    'hotel': 'Resort Hotel',
    'lead_time': 150,
    'stays_in_weekend_nights': 2,
    'stays_in_week_nights': 3,
    'adults': 2,
    'children': 1,
    'meal': 'BB',
    'market_segment': 'Online TA',
    'reserved_room_type': 'A',
    'deposit_type': 'No Deposit',
    'customer_type': 'Transient',
    'adr': 125.0,
    'required_car_parking_spaces': 0,
    'total_of_special_requests': 1
}

print("\nBusiness insights from the XGBoost model:")
print("1. Key factors in cancellation prediction (top features):")
for feature, importance in feature_importance.head(10).values:
    print(f"   - {feature}: {importance:.4f}")

print("\n2. Recommendations for hotels based on model findings:")
print("   - Focus on bookings with high lead times as they're more likely to be canceled")
print("   - Consider different cancellation policies for different market segments")
print("   - Implement dynamic pricing strategies for periods with high cancellation risks")
print("   - Use the cancellation probability to optimize overbooking strategies")

print("\nXGBoost hotel booking cancellation prediction model complete!") 