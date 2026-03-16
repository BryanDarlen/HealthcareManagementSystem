# Healthcare No-Show Prediction System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue" alt="Python">
  <img src="https://img.shields.io/badge/LightGBM-3.3.5-green" alt="LightGBM">
  <img src="https://img.shields.io/badge/FastAPI-0.108.0-teal" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-1.29.0-red" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

# Healthcare No-Show Prediction

An end-to-end machine learning project for predicting whether a patient will miss a scheduled medical appointment. The project includes data preparation, feature engineering, model training with LightGBM, an optional FastAPI prediction API, and a Streamlit dashboard for interactive exploration.

## Project Overview

Missed appointments create operational inefficiency and financial loss in healthcare systems. This project predicts appointment no-show risk using patient, scheduling, and historical behavior features so that clinics can prioritize reminders and interventions for high-risk cases.

## What This Project Includes

- Raw appointment dataset
- Data cleaning and preprocessing pipeline
- Feature engineering pipeline
- LightGBM no-show prediction model
- Streamlit dashboard for interactive analysis
- FastAPI app for prediction serving
- Saved model artifacts for reproducibility and demo use

## Repository Structure

```text
healthcare-no-show-prediction/
├── app/
│   └── dashboard.py
├── data/
│   ├── raw/
│   │   └── medical_appointment.csv
│   ├── processed/
│   │   └── cleaned_data.csv
│   └── features/
│       └── engineered_features.csv
├── docs/
│   └── figures/
├── models/
│   ├── lightbgm_model.pkl
│   ├── lightbgm_features.json
│   └── lightbgm_results.json
├── notebooks/
│   └── 01_EDA.ipynb
├── src/
│   ├── api/
│   │   └── main.py
│   ├── data/
│   │   └── data_loader.py
│   ├── features/
│   │   └── feature_engineer.py
│   └── models/
│       └── train.py
├── tests/
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── README.md
├── requirements.txt
└── verify_data.py
