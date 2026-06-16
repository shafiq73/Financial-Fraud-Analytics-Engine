import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="Financial Fraud Detection Engine",
    page_icon="🛡️",
    layout="centered"
)

# 2. Load the Serialized Artifacts safely (Updated with your exact file names)
@st.cache_resource
def load_artifacts():
    model = joblib.load("xgb_fraud_pickle.pkl") # Aapki naye file ka naam
    scaler = joblib.load("robust_scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    st.error(f"Error loading model artifacts: {e}")

# 3. UI Header
st.title("🛡️ Financial Fraud Analytics Engine")
st.write("An end-to-end Machine Learning pipeline utilizing calibrated classification models to detect anomalous credit card transactions in real-time.")
st.markdown("---")

if artifacts_loaded:
    st.subheader("💳 Input Transaction Metrics")
    st.write("Enter the transaction details below to evaluate the risk probability score.")

    # Form to take inputs
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            time_input = st.number_input("Transaction Time (Seconds)", min_value=0.0, value=3600.0)
        with col2:
            amount_input = st.number_input("Transaction Amount ($)", min_value=0.0, value=125.50)
            
        st.markdown("**Anonymized V-Features (PCA Components V1 - V28):**")
        
        # Creating a grid for V1 to V28 features
        v_inputs = []
        v_cols = st.columns(4)
        
        for i in range(1, 29):
            col_index = (i - 1) % 4
            with v_cols[col_index]:
                v_val = st.number_input(f"V{i}", value=0.0, key=f"V{i}")
                v_inputs.append(v_val)
                
        # Submit button
        submit_btn = st.form_submit_button("Analyze Transaction Risk", type="primary")

    # 4. Prediction Logic
    if submit_btn:
        raw_features = np.array([[time_input, amount_input]])
        scaled_features = scaler.transform(raw_features)
        
        final_input_vector = [scaled_features[0][0]] + v_inputs + [scaled_features[0][1]]
        input_df = pd.DataFrame([final_input_vector])
        
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # 5. Display Results
        st.markdown("### 📊 Diagnostics Evaluation Report")
        
        fraud_probability = prediction_proba[1] * 100
        genuine_probability = prediction_proba[0] * 100
        
        if prediction == 1:
            st.error(f"🚨 **Alert: Highly Suspect / Fraudulent Transaction Detected!**")
            st.metric(label="Fraud Risk Probability", value=f"{fraud_probability:.2f}%")
        else:
            st.success(f"✅ **Clear: Transaction is Statistically Genuine.**")
            st.metric(label="Genuine Probability Confidence", value=f"{genuine_probability:.2f}%")

else:
    st.info("💡 Deployment Tip: Please ensure 'xgb_fraud_pickle.pkl' and 'robust_scaler.pkl' are placed in the same repository root directory.")

# Footer
st.markdown("---")
st.caption("Engineered by Shafiq Ahmed | FinTech Risk Intelligence Workflow")
