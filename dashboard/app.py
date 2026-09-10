import streamlit as st
import pandas as pd
import requests
import json
import os

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Credit Default Predictor", layout="wide")

st.title("💳 Credit Card Default Predictor Dashboard")
st.markdown("This dashboard interfaces with the FastAPI backend to predict the risk of a client defaulting on their next credit card payment.")

# Tab selection
tab1, tab2 = st.tabs(["Prediction Interface", "Dataset Overview"])

with tab1:
    st.header("Predict Client Default Risk")
    
    with st.form("prediction_form"):
        st.subheader("Demographics & Account Details")
        col1, col2, col3, col4 = st.columns(4)
        limit_balance = col1.number_input("Credit Limit", min_value=1000, max_value=1000000, value=50000)
        age = col2.number_input("Age", min_value=18, max_value=100, value=30)
        sex = col3.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x==1 else "Female")
        education = col4.selectbox("Education", options=[1, 2, 3, 4], format_func=lambda x: {1:"Graduate", 2:"University", 3:"High School", 4:"Other"}[x])
        marriage = col1.selectbox("Marriage", options=[1, 2, 3], format_func=lambda x: {1:"Married", 2:"Single", 3:"Other"}[x])
        
        st.subheader("Recent Repayment Status (-1=Duly, >0=Months Delayed)")
        p_col1, p_col2, p_col3, p_col4, p_col5, p_col6 = st.columns(6)
        pay_1 = p_col1.number_input("Sep Status", min_value=-2, max_value=8, value=0)
        pay_2 = p_col2.number_input("Aug Status", min_value=-2, max_value=8, value=0)
        pay_3 = p_col3.number_input("Jul Status", min_value=-2, max_value=8, value=0)
        pay_4 = p_col4.number_input("Jun Status", min_value=-2, max_value=8, value=0)
        pay_5 = p_col5.number_input("May Status", min_value=-2, max_value=8, value=0)
        pay_6 = p_col6.number_input("Apr Status", min_value=-2, max_value=8, value=0)
        
        st.subheader("Bill Amounts & Payment Amounts (Most Recent)")
        b_col1, b_col2 = st.columns(2)
        bill_amt1 = b_col1.number_input("Sep Bill Amount", value=20000.0)
        bill_amt2 = b_col1.number_input("Aug Bill Amount", value=19000.0)
        pay_amt1 = b_col2.number_input("Sep Paid Amount", value=2000.0)
        
        # Calculate engineered features dynamically based on inputs
        utilization_ratio = bill_amt1 / (limit_balance + 1e-5)
        payment_ratio_1 = pay_amt1 / (abs(bill_amt2) + 1e-5)
        delays = [max(0, p) for p in [pay_1, pay_2, pay_3, pay_4, pay_5, pay_6]]
        avg_delay = sum(delays) / 6.0
        max_delay = max(delays)
        
        submit_button = st.form_submit_button("Predict Default Risk")
        
    if submit_button:
        payload = {
            "limit_balance": limit_balance, "sex": sex, "education": education, "marriage": marriage, "age": age,
            "pay_1": pay_1, "pay_2": pay_2, "pay_3": pay_3, "pay_4": pay_4, "pay_5": pay_5, "pay_6": pay_6,
            "bill_amt1": bill_amt1, "bill_amt2": bill_amt2, "bill_amt3": 0, "bill_amt4": 0, "bill_amt5": 0, "bill_amt6": 0,
            "pay_amt1": pay_amt1, "pay_amt2": 0, "pay_amt3": 0, "pay_amt4": 0, "pay_amt5": 0, "pay_amt6": 0,
            "utilization_ratio": utilization_ratio, "payment_ratio_1": payment_ratio_1, "avg_delay": avg_delay, "max_delay": max_delay
        }
        
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Risk Level: **{result['risk_level']}**")
                st.metric(label="Probability of Default", value=f"{result['probability']*100:.1f}%")
            else:
                st.error(f"API Error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Is the FastAPI server running?")

with tab2:
    st.header("Dataset & Visualizations")
    st.markdown("These visualizations were generated during the EDA phase (Phase 5).")
    
    if os.path.exists("outputs/figures/01_target_distribution.png"):
        st.image("outputs/figures/01_target_distribution.png", caption="Target Distribution")
        st.image("outputs/figures/04_correlation_matrix.png", caption="Correlation Matrix")
    else:
        st.warning("Visualizations not found. Run the EDA script first.")
