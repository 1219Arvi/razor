"""
Runs the trained return-risk model against real Razorpay test-mode
orders (created by orders.py) and produces an audit log — the same
decide/explain logic as the synthetic two-tier agent, just pointed at
real order IDs instead of the held-out test set.
"""

import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Let this script be run from anywhere (repo root or this folder) and
# still find the root-level shared modules.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import agent_logic
import train_pipeline

ENRICHED_ORDERS_CSV = os.path.join(_REPO_ROOT, "data/test_mode/razorpay_enriched_orders.csv")
AUDIT_LOG_CSV = os.path.join(_REPO_ROOT, "data/test_mode/razorpay_audit_log.csv")


def run_audit(
    enriched_orders_csv: str = ENRICHED_ORDERS_CSV,
    model_path: str = os.path.join(_REPO_ROOT, train_pipeline.MODEL_PATH),
    save: bool = True,
) -> pd.DataFrame:
    log_reg_final = train_pipeline.load_model(model_path)
    scaler = log_reg_final.named_steps["scaler"]
    model = log_reg_final.named_steps["model"]

    # feature_names come from the trainpool the model was actually fit on
    X_trainpool = pd.read_csv(os.path.join(_REPO_ROOT, train_pipeline.X_TRAINPOOL_CSV))
    feature_names = X_trainpool.columns

    mean_return_rate = pd.read_csv(
        os.path.join(_REPO_ROOT, train_pipeline.RAW_MODEL_CSV)
    )["customer_past_return_rate"].mean()

    rzp_df = pd.read_csv(enriched_orders_csv)

    rzp_df["has_return_history"] = rzp_df["customer_past_return_rate"].notna().astype(int)
    rzp_df["customer_past_return_rate"] = rzp_df["customer_past_return_rate"].fillna(mean_return_rate)

    categorical_cols = ["category", "payment_method", "delivery_pincode_tier", "time_of_day_ordered"]
    rzp_encoded = pd.get_dummies(rzp_df, columns=categorical_cols)
    rzp_encoded = train_pipeline.align_features(rzp_encoded, feature_names)

    probs = model.predict_proba(scaler.transform(rzp_encoded))[:, 1]
    actions = [agent_logic.decide_action(p) for p in probs]
    explanations = [
        agent_logic.explain_order(rzp_encoded.iloc[[i]], scaler, model, feature_names)
        for i in range(len(rzp_encoded))
    ]

    audit_log = pd.DataFrame({
        "razorpay_order_id": rzp_df["razorpay_order_id"],
        "razorpay_status": rzp_df["razorpay_status"],
        "order_value": rzp_df["order_value"],
        "risk_probability": np.round(probs, 4),
        "action_taken": actions,
        "explanation": explanations,
        "decision_threshold_used": agent_logic.DECISION_THRESHOLD,
        "model_version": "logreg_v1_step2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    print("=== Agent decisions on real Razorpay test-mode orders ===\n")
    for _, row in audit_log.iterrows():
        print(f"{row['razorpay_order_id']}  (Rs.{row['order_value']:.2f})  "
              f"prob={row['risk_probability']:.3f} -> {row['action_taken']}")
        print(f"  {row['explanation']}\n")

    if save:
        os.makedirs(os.path.dirname(AUDIT_LOG_CSV), exist_ok=True)
        audit_log.to_csv(AUDIT_LOG_CSV, index=False)
        print(f"Saved: {AUDIT_LOG_CSV}")

    return audit_log


if __name__ == "__main__":
    run_audit()