# Predictive Analytics & AI Model Deployment - Credit Default Risk

## Project Overview
This project is an end-to-end Machine Learning pipeline that predicts whether a credit card client will default on their payment next month. It covers data acquisition, preprocessing, feature engineering, model training, hyperparameter comparison, and deployment via FastAPI and a Streamlit dashboard.

## Problem Statement
Credit card defaults are a massive liability for financial institutions. Accurately predicting default risk allows banks to manage limits dynamically and proactively offer repayment plans.

## Business Context
We aim to predict default risk to minimize financial losses while maintaining customer relationships.

## Objectives
- Perform an EDA on the UCI Credit Card Defaults Dataset.
- Engineer financial ratio features (Credit Utilization, Payment Ratio).
- Compare multiple models (Logistic Regression, Random Forest, HistGradientBoosting).
- Deploy the best-performing model as a FastAPI endpoint.
- Provide an interactive Streamlit Dashboard.

## Dataset
- **Source:** UCI Machine Learning Repository (Default of Credit Card Clients)
- **Target Variable:** `default payment next month` (0 = No, 1 = Yes)
- **Features:** 23 initial features (demographics, repayment history, bill statements). Engineered features include Utilization Ratio and Payment Delay metrics.

## Technologies Used
- Python 3
- Pandas, NumPy
- Scikit-learn
- FastAPI, Uvicorn
- Streamlit
- Seaborn, Matplotlib

## Architecture
1. **Data Pipeline:** `src/data_loader.py` -> `src/preprocessing.py` -> `src/feature_engineering.py` -> `src/split_data.py`.
2. **Model Training:** `src/train.py` handles ColumnTransformers and fits multiple models, selecting the highest ROC-AUC.
3. **Deployment:** FastAPI (`app/main.py`) exposes `/predict`. Streamlit (`dashboard/app.py`) provides the UI.

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd cognevance_predictive_analytics_ai
   ```

2. **Virtual Environment Setup**
   *(Note: Ensure your environment does not have Application Control policies blocking Python DLLs, otherwise scikit-learn will fail to import).*
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Dataset Setup**
   Run the data acquisition script to fetch the latest UCI data:
   ```bash
   python src/data_loader.py
   python src/validation.py
   ```

4. **Data Preparation**
   ```bash
   python src/preprocessing.py
   python src/eda.py
   python src/feature_engineering.py
   python src/split_data.py
   ```

5. **Training**
   ```bash
   python src/train.py
   ```
   *This evaluates models and saves the best pipeline to `models/best_model_pipeline.joblib`.*

6. **Running the API**
   ```bash
   python -m uvicorn app.main:app --reload
   ```

7. **Running the Dashboard**
   ```bash
   python -m streamlit run dashboard/app.py
   ```

## Example API Request
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
         "limit_balance": 50000,
         "sex": 2,
         "education": 2,
         "marriage": 2,
         "age": 25,
         "pay_1": 0, "pay_2": 0, "pay_3": 0, "pay_4": 0, "pay_5": 0, "pay_6": 0,
         "bill_amt1": 20000, "bill_amt2": 19000, "bill_amt3": 0, "bill_amt4": 0, "bill_amt5": 0, "bill_amt6": 0,
         "pay_amt1": 2000, "pay_amt2": 0, "pay_amt3": 0, "pay_amt4": 0, "pay_amt5": 0, "pay_amt6": 0,
         "utilization_ratio": 0.4,
         "payment_ratio_1": 0.1,
         "avg_delay": 0.0,
         "max_delay": 0.0
     }'
```

## Example API Response
```json
{
  "prediction": 0,
  "probability": 0.15,
  "risk_level": "Low"
}
```

## Limitations & Future Improvements
- **Limitation:** The dataset has high class imbalance. We used stratification and balanced weights, but SMOTE could be explored.
- **Limitation:** Windows Application Control Policy blocked `scikit-learn` in our initial internship local run. The pipeline is built to be run in an unrestricted VM.
- **Future:** Incorporate deep learning (Keras) if dataset grows exponentially.

## Internship Context
Developed as Project 3 for Cognevance Technologies.

## License
MIT
