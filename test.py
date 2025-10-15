import joblib

# Load the trained model
model = joblib.load("C:/Users/T8674/Desktop/NASA-turbofan-ML-Project-AIOPS/Turbofan/artifact/model_trainer/2025-09-09-20-07-40/trained_model/model.pkl")

# Replace this with actual0.5, 300, 0.02, 0.1, 0.3 input data used during training
sample_input = [[1,31,642.58,1581.22,1398.91,554.42,2388.08,9056.4,47.23,521.79,2388.06,8130.11,8.4024,393,38.81,23.3552]]  # Example values

# Make a prediction
print(model.predict(sample_input))