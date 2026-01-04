# Housing Price Regression (ML)

This project implements a supervised machine learning pipeline to predict housing prices using location and socio-economic features from a structured housing dataset.

## Overview
The dataset contains geographic, demographic, and housing-related attributes such as longitude, latitude, median income, population, and proximity to the ocean. The goal is to model the relationship between these features and median house value using regression techniques.

## Dataset
- 20,640 records
- Numerical features: location, housing age, rooms, population, income
- Categorical feature: ocean proximity
- Target variable: median house value

## Methodology
- Data loading and exploration using Pandas
- Data cleaning and handling of missing values
- Exploratory data analysis and visualization
- Feature preprocessing
- Model training and evaluation using:
  - Linear Regression
  - Random Forest Regressor
  - Grid Search–optimized Random Forest

## Model Persistence
Trained models are saved using `joblib` for reuse and comparison:
- `linear_regression_model.pkl`
- `random_forest_model.pkl`
- `best_random_forest_model.pkl`
- `all_housing_models.pkl` (dictionary containing all models)

## Tech Stack
- Python
- NumPy
- Pandas
- Matplotlib / Seaborn
- Scikit-learn
- Joblib

## What This Project Demonstrates
- Supervised regression on structured tabular data
- Feature handling and exploratory analysis
- Model comparison and hyperparameter optimization
- Basic model persistence for reuse

## Notes
This project focuses on core machine learning fundamentals rather than deployment or production-scale systems.
