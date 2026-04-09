import streamlit as st
import joblib
import pandas as pd

# 1. Load the model and feature list
data = joblib.load("insurance_model.joblib")
# Check if data is a dict (from cell 1ce02f40) or just the model
if isinstance(data, dict):
    model = data['model']
    features = data['features']
else:
    model = data
    # Fallback to current kernel features if available
    features = [] 

st.title("Insurance Cost Predictor")

# 2. User Inputs (Matching your training features)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
smoker = st.selectbox("Smoker", ["No", "Yes"])

# 3. Prediction logic
if st.button("Predict"):
    # Convert smoker to numeric
    smoker_val = 1 if smoker == "Yes" else 0
    
    # Create input dataframe
    input_dict = {'age': [age], 'bmi': [bmi], 'smoker': [smoker_val]}
    input_df = pd.DataFrame(input_dict)
    
    # Add missing columns with 0s to match model expectations
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # Ensure columns are in the same order as training
    input_df = input_df[features]
    
    prediction = model.predict(input_df)
    st.success(f"Estimated 12-Month Insurance Cost: ${prediction[0]:,.2f}")
