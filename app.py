import os
import joblib
import pandas as pd
import streamlit as st

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)

# ----------------------------------
# CUSTOM DESIGN / CSS
# ----------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #F4F8FB;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

#main-header {
    background: linear-gradient(135deg, #102A43, #174A65);
    padding: 35px 20px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 25px;
}

#main-header h1 {
    color: white;
    font-size: 32px;
    margin-bottom: 10px;
}

#main-header p {
    color: #D9EAF2;
    font-size: 16px;
    margin: 5px;
}

.section-title {
    color: #102A43;
    font-size: 20px;
    font-weight: 700;
    margin-top: 12px;
    margin-bottom: 12px;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background-color: #087E8B;
    color: white;
    border: none;
    border-radius: 9px;
    height: 50px;
    font-size: 17px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background-color: #066773;
    color: white;
    border: none;
}

div[data-testid="stMetric"] {
    background-color: white;
    border: 1px solid #D5E4EA;
    padding: 18px;
    border-radius: 12px;
}

div[data-testid="stForm"] {
    background-color: white;
    border: 1px solid #D5E4EA;
    border-radius: 14px;
    padding: 22px;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

h2, h3 {
    color: #102A43;
}

.footer {
    text-align: center;
    color: #627D98;
    font-size: 13px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------
# LOAD TRAINED MODEL
# ----------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "Models",
    "adaboost_final.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ----------------------------------
# APPLICATION HEADER
# ----------------------------------

st.markdown("""
<div id="main-header">
    <h1>🩺 Diabetes Risk Prediction</h1>
    <p>Machine Learning Based Risk Estimation System</p>
    <p>Powered by the AdaBoost Algorithm</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "### Enter Health Information\n"
    "Provide the requested values to generate a model-based score."
)

# ----------------------------------
# INPUT FORM
# ----------------------------------

with st.form("diabetes_form"):

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<p class="section-title">Personal Information</p>',
            unsafe_allow_html=True
        )

        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0,
            step=1,
            value=2
        )

        age = st.number_input(
            "Age",
            min_value=1,
            step=1,
            value=35
        )

        st.markdown(
            '<p class="section-title">Glucose and Blood Pressure</p>',
            unsafe_allow_html=True
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0.0,
            value=120.0
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0.0,
            value=70.0
        )

    with col2:

        st.markdown(
            '<p class="section-title">Body Measurements</p>',
            unsafe_allow_html=True
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0.0,
            value=25.0
        )

        insulin = st.number_input(
            "Insulin",
            min_value=0.0,
            value=80.0
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            value=28.5
        )

        st.markdown(
            '<p class="section-title">Other Information</p>',
            unsafe_allow_html=True
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            value=0.45
        )
    submitted = st.form_submit_button(
        "Predict Diabetes Risk",
        use_container_width=True
    )

# ----------------------------------
# PREDICTION
# ----------------------------------

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

    st.markdown("### Prediction Results")

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
        "Educational disclaimer: This application is an "
        "educational machine-learning project using a specific "
        "dataset. Its output is not a medical diagnosis. "
        "The 0–5 score is a simple conversion of model probability "
        "and is not a clinically validated risk scale. "
        "Consult a qualified healthcare professional for medical advice."
    )

# ----------------------------------
# FOOTER
# ----------------------------------

st.markdown("""
<div class="footer">
    Diabetes Risk Prediction | AdaBoost Machine Learning Project
</div>
""", unsafe_allow_html=True)
