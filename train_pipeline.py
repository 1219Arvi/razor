"""
Training pipeline, extracted from ML_pipeline.ipynb into importable
functions so main.py (and anything else) can call it without re-running
a notebook or retraining the model over and over.

The notebook is still fine to keep around for exploratory CV comparisons
(logreg vs gradient boosting) — this module is the "production" path:
prepare features -> split -> train -> persist -> evaluate -> pick a
cost-optimal threshold.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import agent_logic

RAW_MODEL_CSV = "data/interim/synthetic_orders_model.csv"
PREPPED_CSV = "data/processed/synthetic_orders_prepped.csv"
X_TRAINPOOL_CSV = "data/splits/X_trainpool.csv"
Y_TRAINPOOL_CSV = "data/splits/y_trainpool.csv"
X_TEST_CSV = "data/splits/X_test_FINAL.csv"
Y_TEST_CSV = "data/splits/y_test_FINAL.csv"
TEST_PREDICTIONS_CSV = "data/outputs/test_predictions.csv"
THRESHOLD_SWEEP_CSV = "data/outputs/threshold_cost_analysis.csv"
MODEL_PATH = "models/log_reg_final.joblib"

CATEGORICAL_COLS = ["category", "payment_method", "delivery_pincode_tier", "time_of_day_ordered"]
ID_COLS = ["order_id", "customer_id", "order_date"]


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Impute + one-hot encode a raw synthetic_orders_model-shaped dataframe.
    Returns (encoded_df, mean_return_rate). `mean_return_rate` must be kept
    around — it's needed again at inference time to impute new orders the
    same way the training data was imputed.

    This is also the function two_tier_agent's edge-case test and
    razorpay_test_mode/audit.py should call instead of re-implementing the
    impute/encode steps inline.
    """
    df = df.copy()
    df["has_return_history"] = df["customer_past_return_rate"].notna().astype(int)
    mean_return_rate = df["customer_past_return_rate"].mean()
    df["customer_past_return_rate"] = df["customer_past_return_rate"].fillna(mean_return_rate)

    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    df_encoded = df_encoded.drop(columns=[c for c in ID_COLS if c in df_encoded.columns])
    return df_encoded, mean_return_rate


def align_features(df_encoded: pd.DataFrame, feature_names) -> pd.DataFrame:
    """
    Reindexes a freshly-encoded dataframe (e.g. a single new order, or the
    Razorpay test-mode batch) onto the exact column set the model was
    trained on, filling any missing one-hot columns with 0.
    """
    return df_encoded.reindex(columns=feature_names, fill_value=0)


def load_and_prepare(raw_csv: str = RAW_MODEL_CSV, save: bool = True) -> tuple[pd.DataFrame, float]:
    df = pd.read_csv(raw_csv)
    df_encoded, mean_return_rate = prepare_features(df)
    if save:
        _ensure_parent_dir(PREPPED_CSV)
        df_encoded.to_csv(PREPPED_CSV, index=False)
    return df_encoded, mean_return_rate


def split_data(df_encoded: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, save: bool = True):
    X = df_encoded.drop(columns=["returned"])
    y = df_encoded["returned"]

    X_trainpool, X_test, y_trainpool, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if save:
        for path in (X_TRAINPOOL_CSV, Y_TRAINPOOL_CSV, X_TEST_CSV, Y_TEST_CSV):
            _ensure_parent_dir(path)
        X_trainpool.to_csv(X_TRAINPOOL_CSV, index=False)
        y_trainpool.to_csv(Y_TRAINPOOL_CSV, index=False)
        X_test.to_csv(X_TEST_CSV, index=False)
        y_test.to_csv(Y_TEST_CSV, index=False)

    return X_trainpool, X_test, y_trainpool, y_test


def build_log_reg_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
    ])


def train_models(X_trainpool: pd.DataFrame, y_trainpool: pd.Series, random_state: int = 42):
    """Fits the primary model (logreg) and the comparison model (GB)."""
    log_reg_final = build_log_reg_pipeline(random_state)
    log_reg_final.fit(X_trainpool, y_trainpool)

    gb_final = GradientBoostingClassifier(random_state=random_state)
    gb_final.fit(X_trainpool, y_trainpool)

    return log_reg_final, gb_final


def evaluate_on_test(log_reg_final, gb_final, X_test, y_test, save: bool = True) -> pd.DataFrame:
    log_reg_probs = log_reg_final.predict_proba(X_test)[:, 1]
    gb_probs = gb_final.predict_proba(X_test)[:, 1]

    print("=== FINAL HELD-OUT TEST SET RESULTS (evaluated once) ===\n")
    print("Logistic Regression (PRIMARY MODEL):")
    print(f"  ROC-AUC: {roc_auc_score(y_test, log_reg_probs):.4f}")
    print(f"  PR-AUC:  {average_precision_score(y_test, log_reg_probs):.4f}")
    print("\nGradient Boosting (comparison model):")
    print(f"  ROC-AUC: {roc_auc_score(y_test, gb_probs):.4f}")
    print(f"  PR-AUC:  {average_precision_score(y_test, gb_probs):.4f}")

    preds = pd.DataFrame({"y_true": y_test.values, "log_reg_prob": log_reg_probs, "gb_prob": gb_probs})
    if save:
        _ensure_parent_dir(TEST_PREDICTIONS_CSV)
        preds.to_csv(TEST_PREDICTIONS_CSV, index=False)
        print(f"\nSaved: {TEST_PREDICTIONS_CSV}")
    return preds


def save_model(model, path: str = MODEL_PATH) -> None:
    _ensure_parent_dir(path)
    joblib.dump(model, path)
    print(f"Saved trained model: {path}")


def load_model(path: str = MODEL_PATH) -> Pipeline:
    """
    Loads the persisted logreg Pipeline. This replaces the pattern of
    retraining log_reg_final from X_trainpool/y_trainpool in
    two_tier_agent.ipynb and razorpay_test_mode/audit.py — call this
    instead, once run_full_training() has been run at least once.
    """
    return joblib.load(path)


def find_cost_optimal_threshold(
    probs: pd.Series,
    y_true: pd.Series,
    order_value: pd.Series,
    fp_cost: float = agent_logic.FLAG_COST,
    thresholds=np.arange(0.05, 0.95, 0.01),
    save: bool = True,
) -> tuple[pd.Series, pd.DataFrame]:
    """Sweeps decision thresholds and returns the cheapest one, plus the full sweep."""
    results = []
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        fp_mask = (y_pred == 1) & (y_true == 0)
        fn_mask = (y_pred == 0) & (y_true == 1)

        total_fp_cost = fp_mask.sum() * fp_cost
        total_fn_cost = agent_logic.fn_cost(order_value[fn_mask]).sum()
        total_cost = total_fp_cost + total_fn_cost

        results.append({
            "threshold": round(t, 2),
            "n_flagged": int(y_pred.sum()),
            "false_positives": int(fp_mask.sum()),
            "false_negatives": int(fn_mask.sum()),
            "fp_cost": total_fp_cost,
            "fn_cost": round(total_fn_cost, 2),
            "total_cost": round(total_cost, 2),
        })

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["total_cost"].idxmin()]

    if save:
        _ensure_parent_dir(THRESHOLD_SWEEP_CSV)
        results_df.to_csv(THRESHOLD_SWEEP_CSV, index=False)

    return best_row, results_df


def run_full_training(raw_csv: str = RAW_MODEL_CSV, model_path: str = MODEL_PATH):
    """
    End-to-end: prepare -> split -> train -> persist -> evaluate -> cost
    search. This is what main.py calls for the "train" stage.
    """
    df_encoded, mean_return_rate = load_and_prepare(raw_csv)
    X_trainpool, X_test, y_trainpool, y_test = split_data(df_encoded)

    log_reg_final, gb_final = train_models(X_trainpool, y_trainpool)
    save_model(log_reg_final, model_path)

    preds = evaluate_on_test(log_reg_final, gb_final, X_test, y_test)

    order_value_test = X_test["order_value"].reset_index(drop=True)
    best_row, sweep_df = find_cost_optimal_threshold(
        preds["log_reg_prob"], preds["y_true"], order_value_test
    )
    print("\n=== Cost-optimal threshold ===")
    print(best_row.to_string())

    return {
        "log_reg_final": log_reg_final,
        "gb_final": gb_final,
        "X_test": X_test,
        "y_test": y_test,
        "preds": preds,
        "mean_return_rate": mean_return_rate,
        "best_threshold_row": best_row,
        "threshold_sweep": sweep_df,
    }


if __name__ == "__main__":
    run_full_training()