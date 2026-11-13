# Predicting Diabetes Risk Using Boosting Algorithms
## Project Overview



This project develops a machine learning-based system for estimating diabetes risk using boosting algorithms. It uses patient health-related input features to generate a risk score from 0 to 5.

The system is built using Python and Streamlit, with a tuned AdaBoost classifier selected as the final prediction model.

### Objectives

- Develop a diabetes risk prediction model using boosting algorithms.

- Preprocess and analyze the diabetes dataset.

- Train and evaluate different machine learning models.

- Compare model performance using classification metrics.

- Build an interactive web application for risk estimation.

### Dataset

The project uses the Pima Indians Diabetes Dataset, containing 768 records and 8 input features, with a binary target variable (Outcome).

The input features include:

- Pregnancies

- Glucose

- Blood Pressure

- Skin Thickness

- Insulin

- BMI

- Diabetes Pedigree Function

- Age

### Methodology

1. Dataset collection and exploration.

2. Data preprocessing and handling missing-like zero values.

3. Stratified train-test splitting (80% training and 20% testing).

4. Model training using AdaBoost and Random Forest.

5. Hyperparameter tuning using GridSearchCV.

6. Model evaluation and comparison.

7. Development of an interactive Streamlit application.

### Models and Evaluation

Three models were evaluated:



| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Baseline AdaBoost | 74.68% | 0.818 |
| Tuned AdaBoost | 76.62% | 0.821 |
| Random Forest | 74.03% | 0.812 |

The tuned AdaBoost model was selected as the final model based on its overall evaluation performance.

### Application Features

- Interactive health-feature input form.

- Machine learning-based prediction.

- Risk score ranging from 0 to 5.

- Corresponding risk-level label.

- Estimated probability display.

- User-friendly Streamlit interface.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit
- GitHub

## Project Structure

```text
diabetes-risk-prediction/
│
├── Models/
│   ├── README.md
│   └── adaboost_final.pkl
│
├── app.py
├── requirements.txt
└── README.md

```

### How to Run Locally

1. Clone the repository:

git clone https://github.com/Nejat-Git/diabetes-risk-prediction.git

2. Open the project directory:

cd diabetes-risk-prediction

3. Install the required packages:

pip install -r requirements.txt

4. Run the application:

streamlit run app.py

## Live Application

[Open Diabetes Risk Prediction App](https://diabetes-risk-prediction-9sszwgqsvtow9fsvfy6qu8.streamlit.app/)

### Limitations

This project is an academic machine learning application. Its predictions are based on the dataset used for training and should not be interpreted as medical diagnoses or clinically validated risk assessments.

The risk score is an educational transformation of the model's predicted probability.

### Author

**Nejat Mohammed Awol**

Computer Science Master's Student
**Addis Ababa University**

GitHub: https://github.com/Nejat-Git
