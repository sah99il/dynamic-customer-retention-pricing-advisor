import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


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
    _require_columns(df, ["Churn"])

    target_col = "Churn"
    fold_col = "kfold"
    drop_cols = [c for c in [target_col, fold_col] if c in df.columns]

    X_raw = df.drop(columns=drop_cols)
    y = df[target_col].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def eval_model(model_name: str, model_factory):
        scores: list[float] = []
        for fold, (train_idx, valid_idx) in enumerate(skf.split(X_raw, y)):
            train_df = df.iloc[train_idx].reset_index(drop=True)
            valid_df = df.iloc[valid_idx].reset_index(drop=True)

            monthly_median = float(train_df["MonthlyCharges"].median())

            train_feat = make_features(train_df.drop(columns=drop_cols), monthly_median)
            valid_feat = make_features(valid_df.drop(columns=drop_cols), monthly_median)

            train_X, valid_X = one_hot_align(train_feat, valid_feat)
            y_train = train_df[target_col].values
            y_valid = valid_df[target_col].values

            model = model_factory()
            model.fit(train_X, y_train)

            valid_pred = model.predict_proba(valid_X)[:, 1]
            score = roc_auc_score(y_valid, valid_pred)
            scores.append(float(score))
            print(f"{model_name} | Fold {fold}: ROC-AUC={score:.4f}")

        mean_score = float(np.mean(scores))
        print(f"{model_name} | Mean ROC-AUC: {mean_score:.4f}")
        return mean_score

    lr_mean = eval_model(
        "Logistic Regression",
        lambda: LogisticRegression(solver="liblinear", max_iter=1000),
    )
    rf_mean = eval_model(
        "Random Forest",
        lambda: RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=1),
    )

    if rf_mean > lr_mean:
        best_name = "Random Forest"
        best_factory = lambda: RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=1)
    else:
        best_name = "Logistic Regression"
        best_factory = lambda: LogisticRegression(solver="liblinear", max_iter=1000)

    print(f"Best model: {best_name}")

    # Final model on full data (save ONLY the best)
    monthly_median_full = float(df["MonthlyCharges"].median())
    full_feat = make_features(df.drop(columns=drop_cols), monthly_median_full)
    full_X = pd.get_dummies(full_feat, drop_first=True)
    y_full = y

    final_model = best_factory()
    final_model.fit(full_X, y_full)

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, models_dir / "model.pkl")
    joblib.dump(list(full_X.columns), models_dir / "features.pkl")

    print(f"Saved model to: {models_dir / 'model.pkl'}")
    print(f"Saved features to: {models_dir / 'features.pkl'}")


if __name__ == "__main__":
    main()
