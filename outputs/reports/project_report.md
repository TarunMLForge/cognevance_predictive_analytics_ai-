# Project Report: Predictive Analytics & AI Model Deployment

## 1. Introduction
This report outlines the end-to-end development of a machine learning predictive pipeline for Cognevance Technologies (Project 3). The project builds a deployable predictive model alongside a web dashboard.

## 2. Problem Statement
Predicting whether a credit card client will default on their next payment.

## 3. Business Context
Banks need reliable risk assessments to automatically alter credit limits and target clients with early debt counseling. A False Negative (missing a default) is very costly, so we prioritize Recall and ROC-AUC.

## 4. Dataset
**Default of Credit Card Clients Dataset (UCI)**
- Size: 30,000 instances
- 23 original features spanning demographic info, past payment status, and bill amounts.

## 5. Data Collection
Data was programmatically fetched using the `ucimlrepo` library to ensure reproducibility, removing the need for manual CSV downloads.

## 6. Data Validation
- Missing values: 0
- Duplicate rows: 35
- Target Distribution: 77.88% (0), 22.12% (1).

## 7. Data Preprocessing
- Duplicates dropped.
- Columns renamed to semantically clear names (`limit_balance`, `pay_1`, etc.).
- Undocumented categories (e.g., `0`, `5`, `6` in education) were grouped into 'Other'.

## 8. Exploratory Data Analysis
- We observed that defaults are higher in clients with lower credit limits.
- Delayed payment statuses (especially `pay_1` >= 2) strongly correlate with final default.
- Demographics alone (Sex, Marriage) do not hold as strong a predictive signal as the financial history.

## 9. Feature Engineering
We introduced new domain-specific features:
- `utilization_ratio`: `bill_amt1 / limit_balance` (How maxed out the card is).
- `payment_ratio_1`: `pay_amt1 / bill_amt2` (Fraction of the previous bill paid).
- `avg_delay`: Average months delayed across the 6-month period.
- `max_delay`: Highest delay reached in the last 6 months.

## 10. Model Development
Models prepared in the pipeline:
- **Logistic Regression (Baseline)**
- **Random Forest Classifier**
- **HistGradientBoostingClassifier** (sklearn's LightGBM equivalent)

## 11. Model Comparison
*(NOT VERIFIED YET - Awaiting unrestricted environment execution)*
The code compares these models based on:
- ROC-AUC
- F1-Score
- Precision & Recall

## 12. Hyperparameter Tuning
*(NOT VERIFIED YET)*
The models are equipped with `class_weight='balanced'` in the provided scripts to combat the class imbalance automatically. Future iterations incorporate `GridSearchCV`.

## 13. Final Model Selection
*(NOT VERIFIED YET)*
The script `src/train.py` dynamically saves the model with the highest ROC-AUC score.

## 14. Predictive Analytics
The final deployed pipeline processes incoming raw features identically to the training set via a `ColumnTransformer`.

## 15. API/Deployment
Deployed using **FastAPI**. 
Endpoints:
- `GET /health`
- `POST /predict` (accepts 27 features, returns `{prediction, probability, risk_level}`)

## 16. Dashboard
Deployed using **Streamlit**. It provides a 2-tab interface:
- Tab 1: Live prediction form for entering client data and fetching risk probabilities.
- Tab 2: Dashboard reviewing the EDA charts generated during Phase 5.

## 17. Evaluation
*(NOT VERIFIED YET)*

## 18. Results
*(NOT VERIFIED YET - Expected: HistGradientBoosting generally achieves ~0.78 ROC-AUC on this dataset).*

## 19. Limitations
- **Environment Policy Block:** On the local development machine, Windows Application Control blocked the load of `scipy` DLLs, preventing `scikit-learn` from compiling models. The codebase is fully written but execution is deferred.
- **Data Limitations:** Bill amounts can be negative (due to overpayment/refunds), which complicates ratio calculations (handled by taking absolutes + smoothing factor).

## 20. Future Improvements
- Implement SMOTE for synthetic oversampling.
- Explore deep learning models if larger tabular temporal data becomes available.
- Cloud deployment (e.g., AWS Elastic Beanstalk or Render).

## 21. Conclusion
The architectural pipeline is complete, professional, modular, and GitHub-ready, satisfying all requirements of the internship project, pending execution on an unrestricted environment.
