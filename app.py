import streamlit as st
import numpy as np
import pickle


# Load models
with open('diabetes_model.pkl', 'rb') as file:
    diabetes_model = pickle.load(file)

with open('heart_disease_model.pkl', 'rb') as file:
    heart_disease_model = pickle.load(file)

# Add a header image

# App Title and Intro
st.title("🩺 **Predictive Pulse**")
st.write(
    """
    ### Welcome to the Predictive Pulse App!
    This app is designed to provide quick, reliable, and accurate predictions for common health conditions like **Diabetes** and **Heart Disease**.
    
    🧠 **Powered by Machine Learning:**  
    Our models analyze your health data to predict risks based on cutting-edge algorithms trained on medical datasets.
    
    🚀 **How to Use:**  
    1. Select a prediction model from the sidebar.  
    2. Enter the required health parameters.  
    3. Click on the **Predict** button to view the result.  
    
    💡 **Purpose of the App:**  
    - **Assist Early Diagnosis:** Help users identify potential health risks early.  
    - **Empower Users:** Provide data-driven insights into health.  
    - **Improve Decision-Making:** Support doctors and patients with advanced tools.  
    
    ---
    """
)
st.sidebar.image(
        "https://tse4.mm.bing.net/th?id=OIG1.kyAa_Vh0ZKIb2zjPEJ9n&pid=ImgGn",
        use_container_width=True
    )

# Sidebar navigation
st.sidebar.title("Navigation")
model_choice = st.sidebar.radio(
    "Choose a model:",
    ("Diabetes Prediction", "Heart Disease Prediction"),
)


# Sidebar Contact and About Us

st.sidebar.markdown("""
        <style>
            .about-us-title {
                font-size: 60px;
                color: #0073e6;
                font-weight: bold;
                text-align: center;
            }
            .about-us-text {
                font-size: 14px;
                color: #333;
                line-height: 1.6;
                padding: 5px;
            }
            .about-us-container {
                background-color: #f4f7fc;
                border-radius: 10px;
                padding: 15px;
                margin-top: 20px;
            }
        </style>
        <div class="about-us-container">
            <h2 class="about-us-title">About Us</h2>
            <p class="about-us-text">
                Welcome to **Predictive Pulse**, a platform leveraging cutting-edge AI and machine learning models to predict health conditions.
            </p>
            <p class="about-us-text">
                Our models can predict the likelihood of diseases like diabetes, heart disease, and cancer based on various health metrics.
            </p>
            <p class="about-us-text">
                Join us in revolutionizing healthcare by integrating AI-powered predictions into everyday medical decisions.
            </p>
        </div>
    """, unsafe_allow_html=True)


# Load the trained models


# Load your dataset (using a sample dataset for illustration purposes)
# Replace with your actual dataset

# Display the graph







# Function for Diabetes Prediction
def diabetes_prediction():
    st.header("Diabetes Prediction 🩺")
    st.image("https://camo.githubusercontent.com/484bb6ba99bf5fbccd82f484c5bcd286edd8c296ab091919ba58310ddabfe3a8/68747470733a2f2f7265732e636c6f7564696e6172792e636f6d2f67726f6865616c74682f696d6167652f75706c6f61642f635f66696c6c2c665f6175746f2c666c5f6c6f7373792c685f3635302c715f6175746f2c775f313038352f76313538313639353638312f4443554b2f436f6e74656e742f6361757365732d6f662d64696162657465732e706e67", use_container_width=True)
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
    st.image("https://editor.analyticsvidhya.com/uploads/95051Cardiovascular-Disease.jpg", use_container_width=True)
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
        slope = st.number_input(label="Slope of Peak Exercise ST Segment (0-3)", min_value=0, max_value=3, value=1)
        thal = st.number_input(label="Thalassemia (0-7)", min_value=0, max_value=7, value=2)

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
    
