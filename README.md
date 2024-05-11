# PRODIGY_ML_01
# House Price Prediction

This repository contains code for predicting house prices using a linear regression model. The model is trained on features such as square footage, number of bedrooms, and bathrooms.

## Requirements

- Python 3
- pandas
- numpy
- scikit-learn

## Installation

1. Clone this repository:
2. Navigate to the project directory
3. Install the required dependencies


## Usage

1. Place your training data in a CSV file named `train.csv` with columns for features and target variable (e.g., 'GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'SalePrice').
2. Place your test data in a CSV file named `test.csv` with columns for features only (e.g., 'GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath').
3. Run the script `house_price_prediction.py` to train the model and generate predictions.

```bash
python house_price_prediction.py

