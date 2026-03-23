from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "model.pkl"
FEATURES_PATH = APP_DIR / "models" / "features.pkl"
FOLDS_PATH = APP_DIR / "data" / "train_folds.csv"


def load_artifact(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path.as_posix()}")
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


def make_features(df: pd.DataFrame, monthly_median: float, total_median: float) -> pd.DataFrame:
    out = df.copy()

    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(total_median)

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


def recommend_action(prob: float) -> str:
    if prob > 0.8:
        return "Give discount"
    if prob > 0.6:
        return "Offer long-term plan"
    if prob > 0.4:
        return "Engagement offer"
    return "Safe"


def load_reference_medians() -> tuple[float, float]:
    if not FOLDS_PATH.exists():
        return 0.0, 0.0

    df = pd.read_csv(FOLDS_PATH)
    monthly_median = float(pd.to_numeric(df["MonthlyCharges"], errors="coerce").median())
    total_median = float(pd.to_numeric(df["TotalCharges"], errors="coerce").median())
    return monthly_median, total_median


app = FastAPI(title="Customer Retention & Pricing Advisor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    model = load_artifact(MODEL_PATH)
    feature_cols = load_artifact(FEATURES_PATH)

    if not hasattr(model, "predict_proba"):
        raise TypeError("Loaded model does not implement predict_proba().")

    if not isinstance(feature_cols, list) or not all(isinstance(c, str) for c in feature_cols):
        raise TypeError("features.pkl must be a list[str].")

    monthly_median, total_median = load_reference_medians()

    app.state.model = model
    app.state.feature_cols = feature_cols
    app.state.monthly_median = monthly_median
    app.state.total_median = total_median


class CustomerInput(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str
    PaymentMethod: str
    PaperlessBilling: str


@app.post("/predict")
def predict(payload: CustomerInput):
    customer = payload.model_dump()
    df = pd.DataFrame([customer])

    monthly_median = float(getattr(app.state, "monthly_median", 0.0)) or float(df["MonthlyCharges"].iloc[0])
    total_median = float(getattr(app.state, "total_median", 0.0)) or float(df["TotalCharges"].iloc[0])

    feat = make_features(df, monthly_median=monthly_median, total_median=total_median)
    X = pd.get_dummies(feat, drop_first=True)
    X = X.reindex(columns=app.state.feature_cols, fill_value=0)

    prob = float(app.state.model.predict_proba(X)[:, 1][0])
    action = recommend_action(prob)

    return {"churn_probability": prob, "action": action}
