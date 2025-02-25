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
