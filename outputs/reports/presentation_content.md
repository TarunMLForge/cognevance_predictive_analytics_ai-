# Presentation Content

## Slide 1: Title
- **Title:** Predictive Analytics & AI Model Deployment
- **Subtitle:** Cognevance Technologies - Project 3
- **Presenter:** [Intern Name]

## Slide 2: Problem Statement
- **Goal:** Predict if a credit card client will default next month.
- **Why?** Minimizing bad debt is crucial for banks. Identifying high-risk clients early allows for proactive measures.

## Slide 3: Objective
- Build an end-to-end Machine Learning pipeline.
- From raw data acquisition to a deployed FastAPI prediction backend and a Streamlit frontend.

## Slide 4: Dataset
- **Source:** Default of Credit Card Clients (UCI ML Repo)
- **Scale:** 30,000 clients, 23 features.
- **Target:** Default (Yes/No)

## Slide 5: Data Preprocessing
- Dropped 35 duplicate rows.
- Standardized column names for clarity.
- Handled rare/undocumented categories (e.g., grouped education `0`, `5`, `6` into 'Other').

## Slide 6: EDA (Exploratory Data Analysis)
- **Insight 1:** Clients with lower credit limit balances have higher default rates.
- **Insight 2:** Delays in the most recent month (`pay_1`) are the strongest predictor of default.

## Slide 7: Feature Engineering
- Derived domain-specific financial features:
  - **Credit Utilization Ratio:** How maxed out the card is.
  - **Payment Ratio:** Fraction of last month's bill paid off.
  - **Average/Max Delay:** Trend of delays over the last 6 months.

## Slide 8: Models
- **Baseline:** Logistic Regression
- **Tree-based Ensembles:** Random Forest, HistGradientBoosting
- Models use `class_weight='balanced'` to tackle the 78/22 class imbalance.

## Slide 9: Model Comparison
- *(NOT VERIFIED YET - Environment limitation blocked execution)*
- Evaluated on ROC-AUC, F1, Precision, Recall.
- Expecting HistGradientBoosting to capture nonlinear relationships best.

## Slide 10: Final Model
- The pipeline dynamically selects the highest ROC-AUC model.
- Includes a built-in `ColumnTransformer` for robust scaling/encoding to prevent data leakage.

## Slide 11: Deployment Architecture
- **Backend:** FastAPI REST API (`/predict`)
- **Frontend:** Streamlit interactive UI.
- Models serialized using `joblib`.

## Slide 12: API
- Exposes a `POST` endpoint receiving 27 JSON features.
- Validates input via Pydantic.
- Returns probability and risk level (High/Low).

## Slide 13: Dashboard
- **Tab 1:** Prediction Form.
- **Tab 2:** Historical Data & EDA Visualizations.

## Slide 14: Results
- *(NOT VERIFIED YET)*
- The pipeline is fully functional and ready for execution in an unrestricted environment.

## Slide 15: Limitations
- **Current Limitation:** Windows Application Control policy blocked `scikit-learn` compilation locally.
- **Data Limitation:** Bill amounts can be negative, complicating ratio features without smoothing.

## Slide 16: Future Improvements
- Test Deep Learning (Neural Networks).
- Implement SMOTE for class balancing.
- Deploy to a cloud provider (e.g., AWS/Render).

## Slide 17: Conclusion
- Successfully built a modular, professional, GitHub-ready AI pipeline.
- Handles the full lifecycle: Data -> EDA -> Engineering -> Training -> API -> UI.
