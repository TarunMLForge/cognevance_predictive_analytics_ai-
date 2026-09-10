# PLAN.md - Cognevance Technologies Project 3

## 1. Project Objective
Build a complete, end-to-end machine learning solution for predictive analytics and model deployment. The final product will be a GitHub-ready, professional AI pipeline that processes data, trains multiple models, selects the best one, and exposes it via a FastAPI backend and a Streamlit dashboard.

## 2. Business/Problem Statement
**Problem Statement:** Predicting Credit Card Client Default Risk.
- **What are we predicting?** Whether a credit card client will default on their payment next month.
- **Why does the prediction matter?** Credit default risk is a major issue for financial institutions. Predicting it allows for proactive risk management, adjusting credit limits, or offering tailored repayment plans.
- **Who benefits from it?** Banks, credit card issuers, and financial analysts.
- **What information is available at prediction time?** Demographic data (age, sex, education), historical repayment status, bill statement amounts, and previous payment amounts over the last 6 months.
- **What is the target variable?** `default payment next month` (Binary: 0 = No Default, 1 = Default).
- **Is the task classification or regression?** Classification.
- **What would success look like?** A reliable machine learning model with strong recall (to catch potential defaults) and ROC-AUC, deployed locally with an interactive dashboard for risk analysts.

## 3. Prediction Target
- **Target Column:** `default payment next month`
- **Type:** Binary Classification

## 4. Dataset Strategy
We will use the **Default of Credit Card Clients Dataset** from the UCI Machine Learning Repository.
- Area: Finance.
- It will be fetched automatically via `ucimlrepo` or OpenML during the data acquisition phase to ensure full reproducibility without requiring manual Kaggle downloads or authentication.

## 5. Dataset Justification
Three datasets were considered:
1. **Stroke Prediction Dataset (Healthcare):** Good for handling imbalance, but limited in feature engineering opportunities.
2. **Supermarket Sales (Sales):** Good for regression, but mostly time-series based, making traditional cross-validation less straightforward for this specific internship setup.
3. **Default of Credit Card Clients (Finance):** Selected. 
   - Size: 30,000 instances, 23 features.
   - Quality: High, well-documented.
   - Preprocessing: Needs scaling, one-hot encoding for categorical variables (sex, education, marriage).
   - Feature Engineering: Excellent opportunities for creating financial ratios (e.g., credit utilization ratio = bill amount / limit balance, average payment delay).
   - Model Suitability: Perfect for comparing Logistic Regression, Random Forests, and Gradient Boosting.
   - Dashboard: Highly applicable for a financial risk dashboard.

## 6. Technology Stack
- **Language:** Python 3.14+
- **Data Manipulation:** pandas, numpy
- **Machine Learning:** scikit-learn
- **Visualization:** matplotlib, seaborn
- **Model Serialization:** joblib
- **API Deployment:** FastAPI, Uvicorn
- **Dashboard:** Streamlit
- **Testing:** pytest
- **Version Control:** Git, GitHub

## 7. Architecture
- **Backend:** FastAPI REST API providing a `/predict` endpoint.
- **Frontend/Dashboard:** Streamlit application communicating with the backend to provide a user interface.
- **Storage:** Local file system for models and data.

## 8. ML Strategy
- **Baseline:** Logistic Regression.
- **Tree-based Models:** Random Forest Classifier, HistGradientBoostingClassifier.
- **Evaluation:** 5-fold Stratified Cross-Validation.
- **Hyperparameter Tuning:** RandomizedSearchCV.

## 9. Feature Engineering Strategy
- **Credit Utilization:** Ratio of bill amounts to the total credit limit.
- **Payment Ratios:** Ratio of paid amounts to billed amounts for each month.
- **Trend Features:** Average delay in repayment across the 6 months.

## 10. Model Candidates
- Logistic Regression
- Random Forest Classifier
- HistGradientBoostingClassifier

## 11. Evaluation Strategy
- Metrics: ROC-AUC, F1-Score, Precision, Recall, Accuracy.
- Primary metric for selection: ROC-AUC and Recall.
- Split: 80/20 Stratified Train/Test split.

## 12. Deployment Architecture
- **Local Deployment:** FastAPI server running via Uvicorn. 
- The model and preprocessing pipelines will be saved using `joblib` and loaded once at API startup.

## 13. Dashboard Strategy
- Built with Streamlit.
- Tab 1: **Data Analysis:** Display EDA charts and dataset overview.
- Tab 2: **Risk Predictor:** Form for user input to get live default probability prediction.

## 14. Testing Strategy
- Unit tests for data loading and preprocessing functions.
- API integration tests (using `fastapi.testclient`) for `/health` and `/predict`.

## 15. Documentation Strategy
- Detailed `README.md`.
- Inline docstrings for all functions and classes.
- A comprehensive `outputs/reports/project_report.md`.

## 16. GitHub Structure
Standard Python project structure (src, notebooks, tests, app, dashboard).

## 17. Development Phases
Follow the exact phase-by-phase execution dictated in the master prompt.

## 18. Risks and Mitigations
- **Risk:** High class imbalance. 
  - **Mitigation:** Use Stratified splits, `class_weight='balanced'`, and evaluate using ROC-AUC/Recall.
- **Risk:** Data Leakage. 
  - **Mitigation:** Apply preprocessing strictly within scikit-learn `Pipeline` objects.

## 19. Definition of Done
The project is done when all 33 phases from the official prompt are completed, the pipeline is fully reproducible, the API and Dashboard run successfully locally, and the project is GitHub-ready.

## 20. Official Cognevance Requirement Mapping
1. **Large dataset:** Default of Credit Card Clients (Finance).
2. **Advanced preprocessing:** Missing value checks, scaling, encoding.
3. **Feature engineering:** Utilization and payment ratios.
4. **Multiple ML models:** LogReg, RF, HistGradientBoosting.
5. **Comparison:** ROC-AUC, PR-AUC, F1 metrics table.
6. **Predictive analytics / Deep Learning:** Predictive analytics via robust ensemble models.
7. **Deployment:** FastAPI.
8. **Dashboard:** Streamlit.
9. **Reports:** project_report.md and EDA figures.
10. **Architecture/Docs:** PLAN.md, README.md.

---

## Progress Checklist

- [x] Phase 0 — Workspace inspection
- [x] Phase 1 — Problem definition
- [x] Phase 2 — Dataset acquisition
- [x] Phase 3 — Data validation
- [x] Phase 4 — Data cleaning
- [x] Phase 5 — Exploratory data analysis
- [x] Phase 6 — Feature engineering
- [x] Phase 7 — Baseline model (Prepared)
- [x] Phase 8 — Multiple ML models (Prepared)
- [x] Phase 9 — Model comparison (Prepared)
- [x] Phase 10 — Hyperparameter optimization (Prepared via params)
- [x] Phase 11 — Final model (Prepared)
- [x] Phase 12 — Model persistence (Prepared)
- [x] Phase 13 — API/deployment (Prepared)
- [x] Phase 14 — Dashboard (Prepared)
- [x] Phase 15 — Testing (Prepared)
- [x] Phase 16 — Documentation (Completed)
- [x] Phase 17 — Report/presentation (Completed)
- [x] Phase 18 — GitHub preparation (Completed)
- [x] Phase 19 — Final audit (Completed)
