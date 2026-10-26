# Student Success Analytics — Exam Score Predictor

## Project Overview

An interactive **Streamlit application** that allows users to explore datasets, visualize numerical relationships, and predict student exam scores using **multiple linear regression**.

The project demonstrates how to turn a Python machine learning model into a publicly accessible web application.

## Dataset

**Source:** [Kaggle — Predicting Student Test Scores (S6E1)](https://www.kaggle.com/competitions/playground-series-s6e1)

The dataset contains 630,000 synthetically generated student records.

**Default model:**
- **Inputs:** Study hours and class attendance
- **Target:** Exam score
- **Algorithm:** Multiple Linear Regression (Scikit-learn)

## Business Application

The model estimates exam scores to explore academic performance patterns and potential student support needs.

**Prediction risks:**
- **Overprediction:** Students needing academic assistance may be overlooked.
- **Underprediction:** Tutoring resources may be allocated unnecessarily.

## Live Application

**Streamlit:** https://hwk1studentsuccessanalytics-46887.streamlit.app/

## Limitations

The dataset is synthetic, and the model has not been validated using real student records. Predictions are for educational purposes only.
