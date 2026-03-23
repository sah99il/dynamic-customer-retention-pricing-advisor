import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")


def call_api(payload: dict) -> dict:
    resp = requests.post(API_URL, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def validate_inputs(tenure: int, monthly: float, total: float) -> tuple[bool, str]:
    if tenure < 0 or monthly < 0:
        return False, "Please enter valid values"
    if tenure == 0 and monthly == 0.0:
        return False, "Please fill all required fields"
    if total > 200000.0:
        return False, "Please enter valid values"
    return True, ""


def interpret_churn(prob: float) -> tuple[str, str, str]:
    if prob > 0.7:
        return "High Risk", "Offer Discount", "error"
    if prob >= 0.4:
        return "Medium Risk", "Engage Customer", "warning"
    return "Low Risk", "Safe Customer", "success"


st.set_page_config(page_title="Customer Retention & Pricing Advisor", layout="centered")

st.title("Customer Retention & Pricing Advisor")
st.caption("Predict churn risk and get retention recommendations")

left, right = st.columns(2)

with left:
    st.subheader("Customer Info")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=0, step=1)
    monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=10000.0, value=0.0, step=1.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)

with right:
    st.subheader("Service Details")
    internet = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"], index=0)
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        index=0,
    )
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)

tenure_i = int(tenure)
monthly_f = float(monthly)
total_charges = float(tenure_i) * monthly_f

payload = {
    "tenure": tenure_i,
    "MonthlyCharges": monthly_f,
    "TotalCharges": total_charges,
    "Contract": str(contract),
    "InternetService": ("No" if internet == "No" else str(internet)),
    "PaymentMethod": str(payment),
    "PaperlessBilling": str(paperless),
}

st.divider()

st.subheader("Prediction")

if st.button("\U0001F680 Predict Churn", type="primary", use_container_width=True):
    ok, message = validate_inputs(tenure_i, monthly_f, total_charges)
    if not ok:
        st.warning(message) if "valid" in message.lower() else st.info(message)
    else:
        with st.spinner("Predicting..."):
            try:
                result = call_api(payload)
                prob_raw = result.get("churn_probability")
                prob = float(prob_raw)
                if not (0.0 <= prob <= 1.0):
                    raise ValueError("Invalid probability")
            except (requests.exceptions.RequestException, ValueError, TypeError):
                st.warning("\u26A0\ufe0f Backend not available")
            else:
                risk, recommendation, tone = interpret_churn(prob)
                st.metric("Churn Probability", f"{prob * 100:.1f}%")
                st.write(f"Risk Level: **{risk}**")
                st.write(f"Recommendation: **{recommendation}**")
                getattr(st, tone)(f"{risk} \u2192 {recommendation}")

st.divider()
st.caption("Built with FastAPI + Streamlit")
