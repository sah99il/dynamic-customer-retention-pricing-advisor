from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_artifact(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {path.as_posix()}")
    return joblib.load(path)


def _tenure_group(tenure: pd.Series) -> pd.Series:
    conditions = [
        tenure < 12,
        (tenure >= 12) & (tenure < 24),
        (tenure >= 24) & (tenure < 48),
        tenure >= 48,
    ]
    choices = [0, 1, 2, 3]
    return pd.Series(np.select(conditions, choices, default=0), index=tenure.index).astype("int64")


def _num_services(df: pd.DataFrame) -> pd.Series:
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    service_cols = [c for c in service_cols if c in df.columns]
    if not service_cols:
        return pd.Series(0, index=df.index, dtype="int64")

    used = pd.DataFrame(index=df.index)
    for col in service_cols:
        if col == "InternetService":
            used[col] = (df[col].astype(str).str.strip().str.lower() != "no").astype("int64")
        else:
            used[col] = (df[col].astype(str).str.strip().str.lower() == "yes").astype("int64")

    return used.sum(axis=1).astype("int64")


def make_features(df: pd.DataFrame, monthly_median: float) -> pd.DataFrame:
    out = df.copy()

    # Basic cleaning consistent with training folds creation
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")

    required = ["tenure", "MonthlyCharges", "TotalCharges", "Contract"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required input fields: {missing}")

    out["avg_monthly_spend"] = out["TotalCharges"] / (out["tenure"] + 1.0)
    out["tenure_group"] = _tenure_group(out["tenure"])
    out["high_value_customer"] = (out["MonthlyCharges"] > monthly_median).astype("int64")
    out["num_services"] = _num_services(out)
    out["ContractOrdinal"] = (
        out["Contract"]
        .map({"Month-to-month": 0, "One year": 1, "Two year": 2})
        .fillna(-1)
        .astype("int64")
    )
    out["engagement_score"] = out["tenure"] * out["MonthlyCharges"]
    out = out.drop(columns=["Contract"])

    return out


def recommend_action(churn_prob: float) -> str:
    if churn_prob > 0.8:
        return "High risk: give strong discount"
    if churn_prob > 0.6:
        return "Medium risk: suggest long-term plan"
    if churn_prob > 0.4:
        return "Low risk: send engagement offer"
    return "Safe customer"


def predict_churn(customer: dict) -> float:
    model = load_artifact("models/model.pkl")
    feature_cols = load_artifact("models/features.pkl")

    # Use training-data median to match how `high_value_customer` was computed.
    folds_path = Path("data/train_folds.csv")
    if folds_path.exists():
        folds = pd.read_csv(folds_path)
        monthly_median = float(folds["MonthlyCharges"].median())
        total_median = float(pd.to_numeric(folds["TotalCharges"], errors="coerce").median())
    else:
        monthly_median = float(customer.get("MonthlyCharges", 0.0))
        total_median = float(customer.get("TotalCharges", 0.0))

    df = pd.DataFrame([customer])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(total_median)

    feat = make_features(df, monthly_median=monthly_median)
    X = pd.get_dummies(feat, drop_first=True)
    X = X.reindex(columns=feature_cols, fill_value=0)

    prob = float(model.predict_proba(X)[:, 1][0])
    return prob


if __name__ == "__main__":
    # Sample input (single customer)
    example_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.7,
        "TotalCharges": "478.5",
    }

    churn_prob = predict_churn(example_customer)
    action = recommend_action(churn_prob)

    print(f"Churn probability: {churn_prob:.4f}")
    print(f"Recommended action: {action}")

    # Example output:
    # Churn probability: 0.4925
    # Recommended action: Low risk: send engagement offer
