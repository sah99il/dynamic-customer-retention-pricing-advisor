import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")


def call_api(payload: dict) -> dict:
    resp = requests.post(API_URL, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


st.set_page_config(page_title="Customer Retention & Pricing Advisor", layout="centered")

st.title("Customer Retention & Pricing Advisor")
st.caption("Predict churn risk and get retention recommendations")


def show_risk(prob: float) -> None:
    if prob > 0.8:
        st.error("\U0001F525 High Risk \u2192 Give Discount")
    elif prob > 0.6:
        st.warning("\u26A0\ufe0f Medium Risk \u2192 Offer Plan")
    elif prob > 0.4:
        st.info("\U0001F642 Low Risk \u2192 Engagement Offer")
    else:
        st.success("\u2705 Safe Customer")


left, right = st.columns(2)

with left:
    st.subheader("Customer Info")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=0, step=1)
    monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=10000.0, value=0.0, step=1.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)

with right:
    st.subheader("Service Details")
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=1)
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        index=0,
    )
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)

payload = {
    "tenure": int(tenure),
    "MonthlyCharges": float(monthly),
    "TotalCharges": float(tenure) * float(monthly),
    "Contract": str(contract),
    "InternetService": str(internet),
    "PaymentMethod": str(payment),
    "PaperlessBilling": str(paperless),
}

st.divider()

if st.button("\U0001F680 Predict Churn"):
    if payload["tenure"] < 0 or payload["MonthlyCharges"] < 0 or payload["TotalCharges"] < 0:
        st.warning("Please enter valid values")
    elif payload["MonthlyCharges"] == 0.0:
        st.info("Please fill all required fields")
    else:
        with st.spinner("Predicting..."):
            try:
                result = call_api(payload)
                churn_prob = float(result.get("churn_probability"))
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError):
                st.warning("\u26A0\ufe0f Backend not available")
            except Exception:
                st.warning("\u26A0\ufe0f Backend not available")
            else:
                st.metric("Churn probability", f"{churn_prob:.4f}")
                show_risk(churn_prob)

st.divider()
st.caption("Built with FastAPI + Streamlit")
