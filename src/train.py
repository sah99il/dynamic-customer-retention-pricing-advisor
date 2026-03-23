import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


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
    # "Using service" means paid/active, not "No ..." placeholders.
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
    for col in service_cols:
        if col not in df.columns:
            # Keep it resilient if dataset differs slightly.
            service_cols = [c for c in service_cols if c in df.columns]
            break

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
    _require_columns(df, ["tenure", "MonthlyCharges", "TotalCharges", "Contract"])

    out = df.copy()

    # Basic numeric are already present: tenure, MonthlyCharges, TotalCharges
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

    # Replace original Contract with ordinal version to avoid duplicated signal.
    out = out.drop(columns=["Contract"])

    return out


def one_hot_align(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_X = pd.get_dummies(train_df, drop_first=True)
    valid_X = pd.get_dummies(valid_df, drop_first=True)
    valid_X = valid_X.reindex(columns=train_X.columns, fill_value=0)
    return train_X, valid_X


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/train_folds.csv", help="Path to fold CSV.")
    args = parser.parse_args()

    data_path = Path(args.input)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find '{data_path}'. Run: python src/create_folds.py --input data/telco.csv"
        )

    df = pd.read_csv(data_path)
    _require_columns(df, ["Churn", "kfold"])

    target_col = "Churn"
    fold_col = "kfold"

    fold_scores: list[float] = []
    for fold in sorted(df[fold_col].unique()):
        train_df = df[df[fold_col] != fold].reset_index(drop=True)
        valid_df = df[df[fold_col] == fold].reset_index(drop=True)

        monthly_median = float(train_df["MonthlyCharges"].median())

        train_feat = make_features(train_df.drop(columns=[target_col, fold_col]), monthly_median)
        valid_feat = make_features(valid_df.drop(columns=[target_col, fold_col]), monthly_median)

        train_X, valid_X = one_hot_align(train_feat, valid_feat)
        y_train = train_df[target_col].values
        y_valid = valid_df[target_col].values

        model = LogisticRegression(solver="liblinear", max_iter=1000)
        model.fit(train_X, y_train)

        valid_pred = model.predict_proba(valid_X)[:, 1]
        score = roc_auc_score(y_valid, valid_pred)
        fold_scores.append(float(score))
        print(f"Fold {fold}: ROC-AUC={score:.4f}")

    print(f"Mean ROC-AUC: {np.mean(fold_scores):.4f}")

    # Final model on full data
    monthly_median_full = float(df["MonthlyCharges"].median())
    full_feat = make_features(df.drop(columns=[target_col, fold_col]), monthly_median_full)
    full_X = pd.get_dummies(full_feat, drop_first=True)
    y_full = df[target_col].values

    final_model = LogisticRegression(solver="liblinear", max_iter=1000)
    final_model.fit(full_X, y_full)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    with open(models_dir / "model.pkl", "wb") as f:
        pickle.dump(final_model, f)

    with open(models_dir / "features.pkl", "wb") as f:
        pickle.dump(list(full_X.columns), f)

    print(f"Saved model to: {models_dir / 'model.pkl'}")
    print(f"Saved features to: {models_dir / 'features.pkl'}")


if __name__ == "__main__":
    main()

