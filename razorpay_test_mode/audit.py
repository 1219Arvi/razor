import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datetime import datetime, timezone

rzp_df = pd.read_csv("data/test_mode/razorpay_enriched_orders.csv")

X_trainpool = pd.read_csv("data/splits/X_trainpool.csv")
y_trainpool = pd.read_csv("data/splits/y_trainpool.csv").squeeze()

log_reg_final = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])
log_reg_final.fit(X_trainpool, y_trainpool)
scaler = log_reg_final.named_steps["scaler"]
model = log_reg_final.named_steps["model"]
feature_names = X_trainpool.columns

mean_return_rate = pd.read_csv("data/interim/synthetic_orders_model.csv")["customer_past_return_rate"].mean()

DECISION_THRESHOLD = 0.16

def decide_action(probability, threshold=DECISION_THRESHOLD):
    return "flag_for_review" if probability >= threshold else "auto_approve"

def readable_label(feature_name, scaled_value=None, raw_value=None):
    manual_overrides = {
        "discount_pct": "large discount", "size_variant_flag": "size-dependent item",
        "is_first_time_buyer": "first-time buyer", "customer_past_return_rate": "customer's return history",
        "customer_past_orders": "customer's order count", "days_to_deliver": "delivery time",
    }
    if feature_name == "order_value":
        return "unusually low order value" if scaled_value is not None and scaled_value < 0 else "unusually high order value"
    if feature_name == "has_return_history":
        return "no prior return history available" if raw_value == 0 else "established return history"
    if feature_name in manual_overrides:
        return manual_overrides[feature_name]
    if feature_name.startswith("category_"):
        return f"{feature_name.replace('category_', '')} category"
    if feature_name.startswith("payment_method_"):
        return f"{feature_name.replace('payment_method_', '')} payment"
    if feature_name.startswith("delivery_pincode_tier_"):
        return f"{feature_name.replace('delivery_pincode_tier_', '')} delivery area"
    if feature_name.startswith("time_of_day_ordered_"):
        return f"ordered in the {feature_name.replace('time_of_day_ordered_', '')}"
    return feature_name

def explain_order(feature_row):
    scaled_values = scaler.transform(feature_row)[0]
    contributions = scaled_values * model.coef_[0]
    raw_values = feature_row.iloc[0]
    contrib_series = pd.Series(contributions, index=feature_names)
    scaled_series = pd.Series(scaled_values, index=feature_names)
    one_hot_prefixes = ("category_", "payment_method_", "delivery_pincode_tier_", "time_of_day_ordered_")

    def is_valid_driver(feat):
        if feat.startswith(one_hot_prefixes):
            return raw_values[feat] == 1
        return True

    candidates = contrib_series[contrib_series > 0.05].sort_values(ascending=False)
    candidates = candidates[[is_valid_driver(f) for f in candidates.index]]
    top_positive = candidates.head(3)
    reasons = [readable_label(f, scaled_series[f], raw_values[f]) for f in top_positive.index]
    return "Elevated risk driven by: " + ", ".join(reasons) if reasons else "No strong individual risk drivers identified."


rzp_df["has_return_history"] = rzp_df["customer_past_return_rate"].notna().astype(int)
rzp_df["customer_past_return_rate"] = rzp_df["customer_past_return_rate"].fillna(mean_return_rate)

categorical_cols = ["category", "payment_method", "delivery_pincode_tier", "time_of_day_ordered"]
rzp_encoded = pd.get_dummies(rzp_df, columns=categorical_cols)
rzp_encoded = rzp_encoded.reindex(columns=feature_names, fill_value=0)


probs = model.predict_proba(scaler.transform(rzp_encoded))[:, 1]
actions = [decide_action(p) for p in probs]
explanations = [explain_order(rzp_encoded.iloc[[i]]) for i in range(len(rzp_encoded))]

audit_log = pd.DataFrame({
    "razorpay_order_id": rzp_df["razorpay_order_id"],
    "razorpay_status": rzp_df["razorpay_status"],
    "order_value": rzp_df["order_value"],
    "risk_probability": np.round(probs, 4),
    "action_taken": actions,
    "explanation": explanations,
    "decision_threshold_used": DECISION_THRESHOLD,
    "model_version": "logreg_v1_step2",
    "timestamp": datetime.now(timezone.utc).isoformat(),
})

print("=== Agent decisions on real Razorpay test-mode orders ===\n")
for _, row in audit_log.iterrows():
    print(f"{row['razorpay_order_id']}  (Rs.{row['order_value']:.2f})  "
          f"prob={row['risk_probability']:.3f} -> {row['action_taken']}")
    print(f"  {row['explanation']}\n")

audit_log.to_csv("data/test_mode/razorpay_audit_log.csv", index=False)
print("Saved: data/test_mode/razorpay_audit_log.csv")