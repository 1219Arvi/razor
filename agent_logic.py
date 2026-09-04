"""
Shared agent decision logic.

Single source of truth for the threshold, the action rules, the cost
formulas, and the human-readable explanation logic. Every script that
turns a model probability into a decision (metrics.py, two_tier_agent,
three_tier_agent, razorpay_test_mode/audit.py) should import from here
instead of redefining these — that's what let three separate copies of
`explain_order` drift in the original notebooks.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------
# Thresholds (chosen from the cost-optimal search in train_pipeline.py /
# the old ML_pipeline.ipynb cell 4)
# ---------------------------------------------------------------------
DECISION_THRESHOLD = 0.16  # two-tier: auto_approve vs flag_for_review

# ---------------------------------------------------------------------
# Cost model (₹). These are the same constants used in the cost sweep,
# kept here too so every stage that needs to *cost* a decision (not just
# make one) uses identical numbers.
# ---------------------------------------------------------------------
FLAG_COST = 40  # flat cost per false alarm / per review: friction overhead


def fn_cost(order_value):
    """Cost of a missed return: reverse logistics + restocking + refund fees."""
    return 150 + 50 + 0.02 * order_value


def hold_cost(order_value):
    """Cost of holding an order for manual verification (three-tier only)."""
    return 20 + 60 + 0.05 * order_value


# ---------------------------------------------------------------------
# Action rules
# ---------------------------------------------------------------------
def decide_action(probability: float, threshold: float = DECISION_THRESHOLD) -> str:
    """
    Turns a model probability into one of two bounded actions. This is the
    entire two-tier 'decision menu' the agent is allowed to choose from -
    nothing outside these two strings can ever be returned.
    """
    return "flag_for_review" if probability >= threshold else "auto_approve"


def three_tier_action(probability: float, t1: float, t2: float) -> str:
    """Three-tier version: adds a 'hold_for_verification' middle ground."""
    if probability < t1:
        return "auto_approve"
    elif probability < t2:
        return "flag_for_review"
    else:
        return "hold_for_verification"


# ---------------------------------------------------------------------
# Explanation logic
# ---------------------------------------------------------------------
_MANUAL_OVERRIDES = {
    "discount_pct": "large discount",
    "size_variant_flag": "size-dependent item",
    "is_first_time_buyer": "first-time buyer",
    "customer_past_return_rate": "customer's return history",
    "customer_past_orders": "customer's order count",
    "days_to_deliver": "delivery time",
}

_ONE_HOT_PREFIXES = (
    "category_",
    "payment_method_",
    "delivery_pincode_tier_",
    "time_of_day_ordered_",
)


def readable_label(feature_name: str, scaled_value: float = None, raw_value=None) -> str:
    """Turns an internal feature/column name into a human-readable phrase."""
    if feature_name == "order_value":
        return (
            "unusually low order value"
            if scaled_value is not None and scaled_value < 0
            else "unusually high order value"
        )
    if feature_name == "has_return_history":
        return (
            "no prior return history available"
            if raw_value == 0
            else "established return history"
        )
    if feature_name in _MANUAL_OVERRIDES:
        return _MANUAL_OVERRIDES[feature_name]
    for prefix in _ONE_HOT_PREFIXES:
        if feature_name.startswith(prefix):
            label = feature_name.replace(prefix, "")
            if prefix == "category_":
                return f"{label} category"
            if prefix == "payment_method_":
                return f"{label} payment"
            if prefix == "delivery_pincode_tier_":
                return f"{label} delivery area"
            if prefix == "time_of_day_ordered_":
                return f"ordered in the {label}"
    return feature_name


def explain_order(feature_row: pd.DataFrame, scaler, model, feature_names, top_n: int = 3) -> str:
    """
    Explains a single-row prediction by ranking the (scaled feature value x
    learned coefficient) contributions, restricted to features that are
    actually "on" for this row (for one-hot columns) and positive-contributing.

    `scaler` / `model` are the fitted StandardScaler / LogisticRegression
    steps pulled out of the trained Pipeline (see train_pipeline.load_model).
    """
    scaled_values = scaler.transform(feature_row)[0]
    contributions = scaled_values * model.coef_[0]
    raw_values = feature_row.iloc[0]

    contrib_series = pd.Series(contributions, index=feature_names)
    scaled_series = pd.Series(scaled_values, index=feature_names)

    def is_valid_driver(feat):
        if feat.startswith(_ONE_HOT_PREFIXES):
            return raw_values[feat] == 1
        return True

    candidates = contrib_series[contrib_series > 0.05].sort_values(ascending=False)
    candidates = candidates[[is_valid_driver(f) for f in candidates.index]]
    top_positive = candidates.head(top_n)

    if top_positive.empty:
        return "No strong individual risk drivers identified."

    reasons = [
        readable_label(feat, scaled_series[feat], raw_values[feat])
        for feat in top_positive.index
    ]
    return "Elevated risk driven by: " + ", ".join(reasons)