
import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the trained model
try:
    model = joblib.load('best_model.pkl')
except FileNotFoundError:
    st.error("Error: best_model.pkl not found. Make sure it's in the same directory as app.py")
    st.stop()

st.title("Salary Prediction App")
st.write("Enter the details to predict the salary using the trained Random Forest Regressor model.")

# Input fields for numerical features
age = st.slider("Age", min_value=23, max_value=53, value=35, step=1) # Min/Max from df.describe()
years_of_experience = st.slider("Years of Experience", min_value=0, max_value=25, value=7, step=1) # Min/Max from df.describe()

# Categorical features - using derived mappings from the training notebook
# Gender: 0 for Female, 1 for Male (based on df.head() after encoding)
gender_options = {'Female': 0, 'Male': 1}
gender_selection = st.radio("Gender", list(gender_options.keys()))
gender_encoded = gender_options[gender_selection]

# Education Level: 0 for Bachelor's, 1 for Master's, 2 for PhD (based on df.head() after encoding)
education_options = {"Bachelor's Degree": 0, "Master's Degree": 1, "PhD": 2}
education_selection = st.selectbox("Education Level", list(education_options.keys()))
education_encoded = education_options[education_selection]

# Job Title: This was encoded numerically. For a real app, you would have the full mapping.
st.info("Please enter the numerical encoded value for 'Job Title'. In a production app, you'd provide a dropdown with all job titles and their corresponding encoded numbers.")
job_title_encoded = st.number_input("Job Title (Encoded Numerical Value)", min_value=0, value=0, step=1)

# Create a DataFrame for prediction
# Ensure the column order matches the training data: 'Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'
input_data = pd.DataFrame([[
    age,
    gender_encoded,
    education_encoded,
    job_title_encoded,
    years_of_experience
]], columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

if st.button("Predict Salary"): 
    # Make prediction
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Salary: ${prediction:,.2f}")

st.write("---")
st.write("Disclaimer: This prediction is based on a simplified model and categorical encodings. For accurate predictions, ensure the 'Job Title' encoding matches your original training data.")
