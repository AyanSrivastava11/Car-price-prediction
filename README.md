# Predicting Used Car Prices

This project involves predicting the prices of used cars using a dataset from Kaggle. The notebook covers data exploration, preprocessing, and modeling to forecast car prices based on various features.

## Table of Contents
- [Project Overview](#project-overview)
- [Libraries Used](#libraries-used)
  
## Project Overview
In this project, I work with a Kaggle dataset to predict the prices of used cars. The notebook includes the following steps:
1. Data exploration and preprocessing.
2. Training and evaluating machine learning models.
3. Comparing model performance.

## Project Overview
This project consists of the following main components:

### 1. Data Preprocessing
In this module, I performed data cleaning and preprocessing to prepare the dataset for modeling. This included handling missing values, encoding categorical features, and normalizing numerical features.

#### Data Cleaning
- Removed or imputed missing values.
- Encoded categorical features using techniques like one-hot encoding.

#### Feature Engineering
- Created new features based on existing ones to improve model performance.
- Normalized numerical features to ensure consistent scaling.

### 2. Model Development
In this module, I developed and trained a regression model to predict the price of used cars.

#### Model Selection
- Experimented with various regression algorithms including Linear Regression, Decision Trees, and Gradient Boosting.

#### Training
- Split the dataset into training and testing sets.
- Trained the model using the training set and tuned hyperparameters for optimal performance.

### 3. Model Evaluation
Evaluated the model's performance using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.

## Technologies Used
- **Programming Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Tools**: Jupyter Notebooks, Google Colab

## Installation

### Prerequisites
- Python 3.x
- Pip

### Steps

1. Clone the repository:
    ```bash
    git clone <repository-link>
    ```

2. Navigate to the project directory:
    ```bash
    cd used-car-prediction-model
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the application:
    ```bash
    python app.py
    ```

## Usage

1. Use the `data_preprocessing.py` file for data cleaning and preprocessing.
2. Use the `model_training.py` file for model training and evaluation.
3. The results and predictions can be generated using the `predict.py` file.

For further instructions on dataset usage, model training, and testing, refer to the individual module documentation in the repository.

## Libraries Used
```python
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_scoreThe following libraries are used in this project:


