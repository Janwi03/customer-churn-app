# Customer Churn Prediction with IBM Dataset

## Overview

This repository contains a machine learning project focused on predicting customer churn using IBM's widely-used Telco Customer Churn dataset. The goal is to identify customers who are likely to discontinue a service, enabling businesses to take proactive retention actions.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [EDA & Preprocessing](#eda--preprocessing)
- [Modeling](#modeling)
- [Results](#results)
- [Installation & Usage](#installation--usage)
- [Requirements](#requirements)
- [Conclusion](#conclusion)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Dataset

- **Source:** [IBM Sample Data Sets](https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/)
- **Description:** The dataset contains information about various customers including demographics, account information, service usage, and a “Churn” label indicating whether the customer has left (Yes/No).

## Project Structure

```
├── .streamlit/
│   └── config.toml
├── .venv/
├── data/
│   └── Telco_customer_churn.xlsx
├── models/
│   ├── churn_model.pkl
│   ├── encoder.pkl
│   └── scaler.pkl
└── src/
    ├── __pycache__/
    ├── predict.py
    ├── preprocess.py
    └── train.py
```

## EDA & Preprocessing

- Explored data distributions, missing values, and relationships between features and churn.
- Handled missing data, encoded categorical variables, and normalized numerical features.
- Performed feature selection and engineering for improved model performance.

## Modeling

- **Algorithms used:** Logistic Regression, Decision Tree, Random Forest, XGBoost, etc.
- **Model evaluation:** Used metrics such as Accuracy, Precision, Recall, F1-score, and ROC-AUC for model assessment.
- **Best model:** (Specify—for example, Random Forest performed best with an ROC-AUC of X.XX)

## Results

- Detailed analysis of model performance.
- Insights on factors influencing customer churn (feature importance).
- Recommendations for business actions based on model findings.

## Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/customer-churn-prediction.git
   cd customer-churn-prediction
   ```
2. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the notebook or script**
   - For Jupyter Notebook:
     ```bash
     jupyter notebook notebooks/eda_and_modelling.ipynb
     ```
   - For Python script:
     ```bash
     python main.py
     ```

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost (optional)

(See `requirements.txt` for full list)

## Conclusion

This project demonstrates the process of data-driven churn prediction, leveraging machine learning models to enhance customer retention strategies. The approach can be further extended using advanced techniques such as hyperparameter tuning, deep learning, or deployment as a REST API.

## Acknowledgements

- IBM for providing the dataset.
- scikit-learn, Pandas, and related open-source libraries.
- [Your inspiration or references, if any.]

## Contact

For any questions or suggestions, contact [your.email@example.com](mailto:your.email@example.com).
