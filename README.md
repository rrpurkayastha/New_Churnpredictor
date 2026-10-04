# Churn Predictor AI

This is a simple machine learning project that predicts whether a customer is likely to leave a company.

The project uses a Random Forest Classifier and has a simple Streamlit interface.

## What It Does

The app allows you to:

* Upload a customer CSV file
* View the data
* Train a machine learning model
* Enter information about a new customer
* Predict whether the customer will churn
* See the churn probability
* See the customer's risk level
* Get some possible reasons for the risk
* Get some suggested actions

## Technologies Used

* Python
* Pandas
* Streamlit
* Scikit-learn

## Machine Learning Model

The project uses a **Random Forest Classifier**.

Before training the model, the data is prepared by:

* Filling in missing values
* Scaling numerical values
* Converting categorical values into numbers
* Encoding the churn result

## Risk Levels

The app uses the churn probability to give a risk level.

| Churn Probability | Risk   |
| ----------------- | ------ |
| 0 - 40%           | Low    |
| 40 - 70%          | Medium |
| 70 - 100%         | High   |

## How to Run

Install the required libraries:

```bash
pip install pandas streamlit scikit-learn
```

Run the program:

```bash
streamlit run app.py
```

Then upload a CSV file and enter the customer information.

## Future Improvements

Some things I could add later are:

* Better model evaluation
* Graphs and charts
* Better explanations for predictions
* More accurate risk levels
* Downloadable prediction results
* A better user interface

## Author

Rajasmit Purkayastha
