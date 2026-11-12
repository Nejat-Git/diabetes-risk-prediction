
import os
import joblib
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

# Load trained model
MODEL_PATH = os.path.join(
    os.path.dirname(_file_),
    "Models",
    "adaboost_final.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Application heading
st.title("🩺 Diabetes Risk Prediction")
st.markdown(
    "### An educational machine learning application "
    "using the AdaBoost algorithm."
)

st.divider()

st.subheader("Enter the patient information")

# Input form
with st.form("diabetes_form"):

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies", min_value=0, step=1, value=2
        )

        glucose = st.number_input(
            "Glucose", min_value=0.0, value=120.0
        )

        blood_pressure = st.number_input(
            "Blood Pressure", min_value=0.0, value=70.0
        )

        skin_thickness = st.number_input(
            "Skin Thickness", min_value=0.0, value=25.0
        )

    with col2:
        insulin = st.number_input(
            "Insulin", min_value=0.0, value=80.0
        )

        bmi = st.number_input(
            "BMI", min_value=0.0, value=28.5
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            value=0.45
        )

        age = st.number_input(
            "Age", min_value=1, step=1, value=35
        )

    submitted = st.form_submit_button(
        "Predict Diabetes Risk",
        use_container_width=True
    )

# Prediction
if submitted:

    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }])

    probability = model.predict_proba(input_data)[0][1]

    risk_score = int(probability * 5 + 0.5)
    risk_score = max(0, min(5, risk_score))

    risk_labels = {
        0: "Very Low Risk",
        1: "Low Risk",
        2: "Lower-Intermediate Risk",
        3: "Upper-Intermediate Risk",
        4: "High Risk",
        5: "Very High Risk"
    }

    st.divider()
    st.subheader("Prediction Results")

    result_col1, result_col2, result_col3 = st.columns(3)

    result_col1.metric(
        "Risk Score",
        f"{risk_score} / 5"
    )

    result_col2.metric(
        "Risk Level",
        risk_labels[risk_score]
    )

    result_col3.metric(
        "Model Probability",
        f"{probability * 100:.2f}%"
    )

    st.info(
        "This is an educational machine learning prediction, "
        "not a medical diagnosis. The 0–5 score is a simple "
        "conversion of model probability and is not a clinically "
        "validated risk scale. Consult a qualified healthcare "
        "professional for medical advice."
    )

st.divider()

st.caption(
    "Diabetes Risk Prediction | AdaBoost Machine Learning Project"
)
