import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _find_input_csv(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists():
        return path

    # Convenience fallback for this repo.
    fallback = Path("data/telco_churn.csv")
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Could not find input CSV at '{path_str}' or '{fallback}'.")


def load_and_clean(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "Churn" not in df.columns:
        raise ValueError("Expected 'Churn' column in input data.")

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype("int64")

    if "TotalCharges" not in df.columns:
        raise ValueError("Expected 'TotalCharges' column in input data.")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    return df


def add_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    df["kfold"] = -1

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, valid_idx) in enumerate(skf.split(X=df, y=df["Churn"])):
        df.loc[valid_idx, "kfold"] = fold

    if (df["kfold"] == -1).any():
        raise RuntimeError("Fold assignment failed for some rows.")

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/telco.csv", help="Path to raw Telco CSV.")
    parser.add_argument("--output", default="data/train_folds.csv", help="Path to save fold CSV.")
    args = parser.parse_args()

    input_csv = _find_input_csv(args.input)
    df = load_and_clean(input_csv)
    df = add_folds(df, n_splits=5, seed=42)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved folds to: {output_path}")


if __name__ == "__main__":
    main()

