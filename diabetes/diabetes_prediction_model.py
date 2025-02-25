import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pickle
import os

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
print("Loading the diabetes dataset...")
df = pd.read_csv('../data/diabetes.csv')

# Display basic information about the dataset
print("\nDataset Information:")
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

print("\nSummary statistics:")
print(df.describe())

# Check for missing values (represented as 0 in some columns)
print("\nChecking for zeros in columns that shouldn't have zeros:")
for column in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    zero_count = (df[column] == 0).sum()
    print(f"{column}: {zero_count} zeros ({zero_count/len(df)*100:.2f}%)")

# Data preprocessing
print("\nPreprocessing the data...")

# Replace zeros with NaN for columns where zero doesn't make sense
columns_to_process = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in columns_to_process:
    df[column] = df[column].replace(0, np.nan)

# Fill missing values with the median of each column
for column in columns_to_process:
    median_value = df[column].median()
    df[column] = df[column].fillna(median_value)

# Split the data into features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training and evaluation
print("\nTraining and evaluating multiple models...")

# Define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Evaluate models using cross-validation
results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    print(f"{name}: CV Accuracy = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Find the best model based on cross-validation results
best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
print(f"\nBest model based on cross-validation: {best_model_name}")

# Hyperparameter tuning for the best model
print(f"\nPerforming hyperparameter tuning for {best_model_name}...")

if best_model_name == 'Logistic Regression':
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs']
    }
    best_model = LogisticRegression(max_iter=1000, random_state=42)
elif best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    best_model = RandomForestClassifier(random_state=42)
elif best_model_name == 'SVM':
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }
    best_model = SVC(probability=True, random_state=42)
else:  # Gradient Boosting
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    }
    best_model = GradientBoostingClassifier(random_state=42)

# Perform grid search
grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# Train the best model on the entire training set
best_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate the model
print("\nModel Evaluation on Test Set:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
print("ROC curve saved as 'roc_curve.png'")

# Feature importance (if applicable)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance plot saved as 'feature_importance.png'")
    print("\nFeature Importance:")
    print(feature_importance)
elif best_model_name == 'Logistic Regression':
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': best_model.coef_[0]
    }).sort_values('Coefficient', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
    plt.title('Feature Coefficients')
    plt.tight_layout()
    plt.savefig('feature_coefficients.png')
    print("Feature coefficients plot saved as 'feature_coefficients.png'")
    print("\nFeature Coefficients:")
    print(feature_importance)

# Save the model and preprocessing components
print("\nSaving the model and preprocessing components...")

# Create a dictionary with all components needed for prediction
model_components = {
    'model': best_model,
    'scaler': scaler,
    'feature_names': X.columns.tolist(),
    'model_type': best_model_name
}

# Save the model components as a pickle file
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model_components, f)

print("Model saved as 'diabetes_model.pkl'")

# Create a simple function to demonstrate how to use the saved model
print("\nCreating a demo script for using the saved model...")

demo_script = """
import pickle
import pandas as pd
import numpy as np

def predict_diabetes(data, model_path='diabetes_model.pkl'):
    # Load the model components
    with open(model_path, 'rb') as f:
        model_components = pickle.load(f)
    
    model = model_components['model']
    scaler = model_components['scaler']
    feature_names = model_components['feature_names']
    
    # Ensure data has the correct features
    data = pd.DataFrame(data, columns=feature_names)
    
    # Apply the same preprocessing
    for column in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        if column in data:
            data[column] = data[column].replace(0, np.nan)
    
    # Fill missing values with the median from the training data
    for column in data.columns:
        if data[column].isna().any():
            # In a real application, you would use the median from the training data
            # Here we'll just use the median of the provided data
            data[column] = data[column].fillna(data[column].median())
    
    # Scale the features
    X_scaled = scaler.transform(data)
    
    # Make predictions
    prediction = model.predict(X_scaled)
    probability = model.predict_proba(X_scaled)[:, 1]
    
    return prediction, probability

# Example usage
if __name__ == "__main__":
    # Sample data (features should be in the same order as during training)
    sample_data = [
        [6, 148, 72, 35, 0, 33.6, 0.627, 50],  # Sample 1
        [1, 85, 66, 29, 0, 26.6, 0.351, 31]    # Sample 2
    ]
    
    predictions, probabilities = predict_diabetes(sample_data)
    
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"Sample {i+1}:")
        print(f"  Prediction: {'Diabetic' if pred == 1 else 'Non-diabetic'}")
        print(f"  Probability of diabetes: {prob:.4f}")
"""

with open('diabetes_prediction_demo.py', 'w') as f:
    f.write(demo_script)

print("Demo script saved as 'diabetes_prediction_demo.py'")
print("\nDone! The diabetes prediction model has been created and saved.")
