import streamlit as st
import numpy as np
import pickle

# Load models
with open('diabetes_model.pkl', 'rb') as file:
    diabetes_model = pickle.load(file)

with open('heart_disease_model.pkl', 'rb') as file:
    heart_disease_model = pickle.load(file)

# App Title
st.title("🩺 Healthcare Prediction Models")
st.write(
    """
    Welcome to the **Healthcare Prediction App**! Select a prediction model from the sidebar to begin.
    """
)

# Sidebar navigation
model_choice = st.sidebar.radio(
    "Choose a model:",
    ("Diabetes Prediction", "Heart Disease Prediction"),
)

# Function for Diabetes Prediction
def diabetes_prediction():
    st.header("Diabetes Prediction 🩺")
    st.write("Enter the details below:")

    # Input fields in columns with clearly visible labels
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input(label="Number of Pregnancies", min_value=0, max_value=20, value=1)
        blood_pressure = st.number_input(label="Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
        insulin = st.number_input(label="Insulin Level", min_value=0, max_value=900, value=80)
        diabetes_pedigree_function = st.number_input(
            label="Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, step=0.01, format="%.2f"
        )
    with col2:
        glucose = st.number_input(label="Glucose Level", min_value=0, max_value=300, value=120)
        skin_thickness = st.number_input(label="Skin Thickness (mm)", min_value=0, max_value=100, value=20)
        bmi = st.number_input(label="BMI (Body Mass Index)", min_value=0.0, max_value=100.0, value=25.0, step=0.1, format="%.1f")
        age = st.number_input(label="Age", min_value=0, max_value=120, value=30)

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

    if st.button("Predict Diabetes"):
        prediction = diabetes_model.predict(input_data)
        if prediction[0] == 1:
            st.error("🚨 The model predicts that the patient is likely to have diabetes.")
        else:
            st.success("✅ The model predicts that the patient is unlikely to have diabetes.")

# Function for Heart Disease Prediction
def heart_disease_prediction():
    st.header("Heart Disease Prediction ❤️")
    st.write("Enter the details below:")

    # Input fields in columns with clearly visible labels
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input(label="Age", min_value=0, max_value=120, value=45)
        cp = st.number_input(label="Chest Pain Type (0-3)", min_value=0, max_value=3, value=1)
        chol = st.number_input(label="Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
        restecg = st.number_input(label="Resting ECG Results (0-2)", min_value=0, max_value=2, value=1)
        exang = st.selectbox(label="Exercise Induced Angina", options=["No", "Yes"])
        exang = 1 if exang == "Yes" else 0
        ca = st.number_input(label="Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
    with col2:
        sex = st.selectbox(label="Sex", options=["Female", "Male"])
        sex = 1 if sex == "Male" else 0
        trestbps = st.number_input(label="Resting Blood Pressure (mm Hg)", min_value=0, max_value=200, value=120)
        fbs = st.selectbox(label="Fasting Blood Sugar > 120 mg/dl", options=["No", "Yes"])
        fbs = 1 if fbs == "Yes" else 0
        thalach = st.number_input(label="Maximum Heart Rate Achieved", min_value=0, max_value=220, value=150)
        oldpeak = st.number_input(label="ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f")
        slope = st.number_input(label="Slope of Peak Exercise ST Segment (0-2)", min_value=0, max_value=2, value=1)
        thal = st.number_input(label="Thalassemia (0-3)", min_value=0, max_value=3, value=2)

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    if st.button("Predict Heart Disease"):
        prediction = heart_disease_model.predict(input_data)
        if prediction[0] == 1:
            st.error("🚨 The model predicts that the patient is likely to have heart disease.")
        else:
            st.success("✅ The model predicts that the patient is unlikely to have heart disease.")

# Display selected model
if model_choice == "Diabetes Prediction":
    diabetes_prediction()
elif model_choice == "Heart Disease Prediction":
    heart_disease_prediction()
