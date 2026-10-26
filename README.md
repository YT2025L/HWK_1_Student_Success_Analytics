# Student Success Analytics — Exam Score Predictor

### 46-887: Machine Learning for Business Applications | HWK 1

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

## Key Features

- Upload CSV files and preview data.
- Explore data using interactive line, scatter, and histogram charts.
- Select two numerical inputs and one prediction target.
- Train a linear regression model and view its equation and R².
- Enter custom values to generate predictions.
- Display warnings for inputs outside the training range.

## Business Application

The model estimates exam scores to explore academic performance patterns and potential student support needs.

**Prediction risks:**
- **Overprediction:** Students needing academic assistance may be overlooked.
- **Underprediction:** Tutoring resources may be allocated unnecessarily.

## Technologies

Python | Pandas | Plotly | Scikit-learn | Streamlit | GitHub

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Live Application

**Streamlit:** [Add your deployed app URL here]

## Limitations

The dataset is synthetic, and the model has not been validated using real student records. Predictions are for educational purposes only and should not be used for actual academic decisions.
