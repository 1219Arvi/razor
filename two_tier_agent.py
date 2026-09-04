"""
Two-tier agent: turns model probabilities into auto_approve /
flag_for_review decisions with explanations, and writes an audit log.

This replaces two_tier_agent.ipynb's script logic. Notable fix versus the
original notebook: cell 3 there read "data/splits/synthetic_orders_model.csv"
and "data/splits/synthetic_orders_prepped.csv", which don't exist (the real
files are under data/interim/ and data/processed/ respectively) - that only
"worked" with a stale kernel already holding the right variables in memory.
Fixed here to read the correct paths.
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

import agent_logic
import train_pipeline

ORDERS_WITH_ACTIONS_CSV = "data/outputs/orders_with_actions.csv"
ORDERS_WITH_EXPLANATIONS_CSV = "data/outputs/orders_with_explanations.csv"
AUDIT_LOG_CSV = "data/logs/audit_log.csv"


def assign_actions(predictions_csv: str = train_pipeline.TEST_PREDICTIONS_CSV, save: bool = True) -> pd.DataFrame:
    preds = pd.read_csv(predictions_csv)
    preds["action"] = preds["log_reg_prob"].apply(agent_logic.decide_action)

    print("=== Action distribution across test set ===")
    print(preds["action"].value_counts())
    n_flagged = (preds["action"] == "flag_for_review").sum()
    print(f"\nOrders flagged for review: {n_flagged} / {len(preds)}")

    if save:
        os.makedirs(os.path.dirname(ORDERS_WITH_ACTIONS_CSV), exist_ok=True)
        preds.to_csv(ORDERS_WITH_ACTIONS_CSV, index=False)
        print(f"\nSaved: {ORDERS_WITH_ACTIONS_CSV}")
    return preds


def add_explanations(
    preds: pd.DataFrame,
    model_path: str = train_pipeline.MODEL_PATH,
    x_test_csv: str = train_pipeline.X_TEST_CSV,
    save: bool = True,
) -> pd.DataFrame:
    log_reg_final = train_pipeline.load_model(model_path)
    scaler = log_reg_final.named_steps["scaler"]
    model = log_reg_final.named_steps["model"]

    X_test = pd.read_csv(x_test_csv)
    feature_names = X_test.columns

    explanations = [
        agent_logic.explain_order(X_test.iloc[[i]], scaler, model, feature_names)
        for i in range(len(X_test))
    ]
    preds = preds.copy()
    preds["explanation"] = explanations

    if save:
        preds.to_csv(ORDERS_WITH_EXPLANATIONS_CSV, index=False)
        print(f"Saved: {ORDERS_WITH_EXPLANATIONS_CSV}")
    return preds


def build_audit_log(
    preds: pd.DataFrame,
    raw_model_csv: str = train_pipeline.RAW_MODEL_CSV,
    prepped_csv: str = train_pipeline.PREPPED_CSV,
    save: bool = True,
) -> pd.DataFrame:
    """
    Recovers order_id/customer_id for the test rows (lost during encoding)
    by redoing the same stratified split on the *original* raw data, then
    builds the final audit log with those IDs + decisions + explanations.
    """
    original_df = pd.read_csv(raw_model_csv)
    df_prepped = pd.read_csv(prepped_csv)

    X_full = df_prepped.drop(columns=["returned"])
    y_full = df_prepped["returned"]

    _, test_idx_df, _, _ = train_test_split(
        original_df, y_full, test_size=0.2, random_state=42, stratify=y_full
    )

    order_ids = test_idx_df["order_id"].values
    customer_ids = test_idx_df["customer_id"].values

    assert len(order_ids) == len(preds), "Mismatch: order_id count doesn't match predictions count"

    from datetime import datetime, timezone

    audit_log = pd.DataFrame({
        "order_id": order_ids,
        "customer_id": customer_ids,
        "risk_probability": preds["log_reg_prob"].round(4),
        "action_taken": preds["action"],
        "explanation": preds["explanation"],
        "decision_threshold_used": agent_logic.DECISION_THRESHOLD,
        "model_version": "logreg_v1_step2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    print("=== Audit log sample ===")
    print(audit_log.head(10).to_string(index=False))
    print(f"\nTotal decisions logged: {len(audit_log)}")
    print(f"Action breakdown:\n{audit_log['action_taken'].value_counts()}")

    if save:
        os.makedirs(os.path.dirname(AUDIT_LOG_CSV), exist_ok=True)
        audit_log.to_csv(AUDIT_LOG_CSV, index=False)
        print(f"\nSaved: {AUDIT_LOG_CSV}")

    return audit_log


def run_two_tier_agent() -> pd.DataFrame:
    preds = assign_actions()
    preds = add_explanations(preds)
    build_audit_log(preds)
    return preds


if __name__ == "__main__":
    run_two_tier_agent()