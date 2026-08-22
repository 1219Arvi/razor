"""
Synthetic Return-Risk Dataset Generator
========================================

Generates realistic, India-context e-commerce order data with a
correlated (but noisy, non-trivial) return-risk label.

Design goals
------------
1. Ground-truth risk formula is known, so a trained model's feature
   importances can be checked against it later (explainability check).
2. Noise is injected deliberately so the problem isn't trivially
   separable. Target: model ROC-AUC in the ~0.75-0.85 range, not ~0.99.
3. Base return rate is auto-calibrated to ~18-22%, realistic for a
   blended Indian e-commerce catalog (fashion-heavy).

Grounding in real data (UCI Online Retail)
-------------------------------------------
Two parameters below were fitted against real transaction data instead
of guessed, and one directional assumption was corrected after checking
real data. See each section for the specific grounding note.

    * Order value distribution shape (sigma) - fitted via
      scipy.stats.lognorm.fit on real UCI order values.
    * Customer order-frequency distribution - fitted via a negative
      binomial simulation search against real UCI per-customer order
      counts.
    * Order value -> return risk direction - corrected after real data
      showed higher-value orders are LESS likely to be cancelled
      (originally assumed the opposite).

Last verified run (seed=42)
----------------------------
    Base return rate:            0.200   (target 0.18-0.22)
    Return rate - fashion:       0.306   Return rate - grocery:  0.063
    Return rate - COD:           0.293   Return rate - card:     0.157
    Return rate - first-time:    0.332   Return rate - repeat:   0.179
    Sanity-check model ROC-AUC:  0.755   PR-AUC:                 0.479
    Customer pool sizing:        662/662 customers used, 5000 orders
"""

import numpy as np
import pandas as pd
import uuid
from datetime import datetime, timedelta

# =======================================================================
# Configuration
# =======================================================================

RNG = np.random.default_rng(42)

N_ORDERS = 5000

# GROUNDED: sized so expected total orders (N_CUSTOMERS x mean orders
# per customer) lands near N_ORDERS with no post-hoc trimming needed.
# An earlier version generated a larger pool and trimmed down to
# N_ORDERS - that was rejected because random trimming disproportionately
# erased low-order-count customers, distorting the fitted shape. Sizing
# the pool correctly up front avoids that problem entirely.
#   N_CUSTOMERS = N_ORDERS / target_mean_orders_per_customer
#               = 5000 / 7.55 (real UCI anchor) =~ 662
N_CUSTOMERS = 662

# GROUNDED: fitted via simulation search against real UCI per-customer
# order-count anchors (mean orders/customer = 7.55, one-time-buyer
# fraction = 0.246). These draws are used as literal order counts per
# customer, NOT as RNG.choice sampling weights - using them as weights
# was tried first and rejected, since weighted-choice sampling reshapes
# the distribution and no longer reproduces the fitted anchors.
NEG_BINOM_R = 0.55
NEG_BINOM_P = 0.08

# GROUNDED: shape (sigma) fitted via scipy.stats.lognorm.fit on real UCI
# order values (raw fit: mu=5.596, sigma=1.086). Location (mu) was
# rescaled to a market-appropriate INR median of ~1200 instead of a
# direct GBP->INR conversion, because UCI reflects UK wholesale buying
# behavior (small businesses buying in bulk) which would otherwise
# inflate synthetic order values well past typical Indian consumer
# e-commerce norms.
ORDER_VALUE_MU = 7.09
ORDER_VALUE_SIGMA = 1.086
ORDER_VALUE_MIN, ORDER_VALUE_MAX = 250, 20_000

TARGET_BASE_RETURN_RATE = 0.20

CATEGORIES = ["fashion", "electronics", "home", "beauty", "grocery"]
CATEGORY_PROBS = [0.38, 0.20, 0.17, 0.15, 0.10]
CATEGORY_RISK_OFFSET = {
    "fashion": 0.6, "electronics": -0.3, "home": 0.0,
    "beauty": 0.2, "grocery": -0.9,
}

PINCODE_TIERS = ["metro", "tier2", "tier3"]
TIER_PROBS = [0.45, 0.35, 0.20]
TIER_RISK_OFFSET = {"metro": -0.1, "tier2": 0.1, "tier3": 0.3}

PAYMENT_METHODS = ["COD", "UPI", "card", "netbanking"]
PAYMENT_PROBS = [0.34, 0.40, 0.20, 0.06]  # COD still large in India, UPI now dominant

TIME_SLOTS = ["morning", "afternoon", "evening", "night"]
TIME_PROBS = [0.20, 0.30, 0.35, 0.15]

ORDER_DATE_START = datetime(2025, 1, 1)
ORDER_DATE_WINDOW_DAYS = 300


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# =======================================================================
# Step 1 - Customer pool
# =======================================================================
# Each customer gets a stable ID and a latent "return propensity" -
# some people just return more than others, independent of any single
# order's features. This lets return behavior correlate across a
# customer's orders, not just react to each order in isolation.

customer_ids = [str(uuid.uuid4()) for _ in range(N_CUSTOMERS)]
customer_latent_propensity = RNG.beta(2, 6, size=N_CUSTOMERS)  # skewed low


# =======================================================================
# Step 2 - Assign orders to customers (grounded order-frequency)
# =======================================================================
# Draw each customer's order count directly from the fitted negative
# binomial, build the order list by repeating customer indices that many
# times, then shuffle. The pool was sized in Step 0 so this rarely needs
# more than a small trim/pad to land exactly on N_ORDERS.

customer_order_counts = RNG.negative_binomial(NEG_BINOM_R, NEG_BINOM_P, size=N_CUSTOMERS) + 1
order_customer_idx = np.repeat(np.arange(N_CUSTOMERS), customer_order_counts)
RNG.shuffle(order_customer_idx)

pre_trim_total = len(order_customer_idx)
if pre_trim_total >= N_ORDERS:
    order_customer_idx = order_customer_idx[:N_ORDERS]
else:
    shortfall = N_ORDERS - pre_trim_total
    extra = RNG.choice(N_CUSTOMERS, size=shortfall)
    order_customer_idx = np.concatenate([order_customer_idx, extra])
    RNG.shuffle(order_customer_idx)

print(
    f"Customer pool sizing check: {len(np.unique(order_customer_idx))} unique "
    f"customers used out of {N_CUSTOMERS} pool, {len(order_customer_idx)} total "
    f"orders (pre-trim/pad total was {pre_trim_total})"
)


# =======================================================================
# Step 3 - Generate per-order features
# =======================================================================

def make_order_row(i: int) -> dict:
    """Build one synthetic order record."""
    cust_idx = order_customer_idx[i]
    customer_id = customer_ids[cust_idx]
    latent_propensity = customer_latent_propensity[cust_idx]

    # Orders placed by this customer earlier in the sequence so far.
    past_orders = int(np.sum(order_customer_idx[:i] == cust_idx))
    is_first_time = int(past_orders == 0)

    # Past return rate is only meaningful once a customer has history.
    # Left as NaN for first-time buyers - this is a deliberate, realistic
    # missing-data case a real model has to handle, not a bug to patch.
    if past_orders > 0:
        past_return_rate = float(np.clip(
            latent_propensity + RNG.normal(0, 0.08), 0, 1
        ))
    else:
        past_return_rate = np.nan

    category = RNG.choice(CATEGORIES, p=CATEGORY_PROBS)
    pincode_tier = RNG.choice(PINCODE_TIERS, p=TIER_PROBS)
    payment_method = RNG.choice(PAYMENT_METHODS, p=PAYMENT_PROBS)
    time_slot = RNG.choice(TIME_SLOTS, p=TIME_PROBS)

    order_value = float(np.clip(
        RNG.lognormal(mean=ORDER_VALUE_MU, sigma=ORDER_VALUE_SIGMA),
        ORDER_VALUE_MIN, ORDER_VALUE_MAX,
    ))
    discount_pct = float(np.clip(RNG.beta(2, 5), 0, 0.75))

    # Fashion is almost always size-dependent; a slice of beauty
    # (e.g. clothing-adjacent accessories) is too.
    size_variant_flag = int(
        (category == "fashion" and RNG.random() < 0.85)
        or (category == "beauty" and RNG.random() < 0.15)
    )

    days_to_deliver = int(np.clip(
        RNG.poisson(3) + (2 if pincode_tier == "tier3" else 0), 1, 12
    ))
    order_date = ORDER_DATE_START + timedelta(
        days=int(RNG.integers(0, ORDER_DATE_WINDOW_DAYS))
    )

    return {
        "order_id": str(uuid.uuid4()),
        "customer_id": customer_id,
        "order_date": order_date.date().isoformat(),
        "order_value": round(order_value, 2),
        "category": category,
        "discount_pct": round(discount_pct, 3),
        "payment_method": payment_method,
        "is_first_time_buyer": is_first_time,
        "customer_past_orders": past_orders,
        "customer_past_return_rate": (
            round(past_return_rate, 3) if not np.isnan(past_return_rate) else np.nan
        ),
        "delivery_pincode_tier": pincode_tier,
        "size_variant_flag": size_variant_flag,
        "days_to_deliver": days_to_deliver,
        "time_of_day_ordered": time_slot,
        # Ground-truth column, kept for validation only - dropped before
        # the model-training CSV is written.
        "_latent_propensity": latent_propensity,
    }


df = pd.DataFrame(make_order_row(i) for i in range(N_ORDERS))


# =======================================================================
# Step 4 - Compute the ground-truth risk label
# =======================================================================
# Combine weighted features into a logit, add noise, convert to a
# probability, then sample the binary `returned` outcome. Weights below
# are the deliberately injected "true" relationships - used later to
# sanity-check that a trained model's feature importances roughly
# recover them.

order_value_norm = (df["order_value"] - df["order_value"].mean()) / df["order_value"].std()
is_cod = (df["payment_method"] == "COD").astype(int)

# First-time buyers have no observed past_return_rate (NaN by design).
# Fill with the population mean so the formula doesn't break on NaN -
# this mirrors what a real downstream model has to do too.
past_return_filled = df["customer_past_return_rate"].fillna(
    df["customer_past_return_rate"].mean()
)
past_return_inverse = 1 - past_return_filled  # higher = safer customer

category_offset = df["category"].map(CATEGORY_RISK_OFFSET)
tier_offset = df["delivery_pincode_tier"].map(TIER_RISK_OFFSET)
noise = RNG.normal(0, 0.6, size=N_ORDERS)

logit = (
      1.1 * is_cod                                  # COD -> higher risk
    + 1.8 * df["discount_pct"]                       # heavy discounting -> higher risk
    + 0.9 * df["is_first_time_buyer"]                 # no trust/fit history -> higher risk
    + 0.7 * df["size_variant_flag"]                   # size-dependent items -> higher risk
    - 0.4 * order_value_norm                          # GROUNDED: real UCI data shows higher-
                                                        # value orders are LESS likely to be
                                                        # cancelled (71% -> 1% across value
                                                        # quintiles) - sign flipped from the
                                                        # original (wrong) guess
    - 1.3 * past_return_inverse                        # loyal, low-return customer -> safer
    + category_offset                                  # category-specific base risk
    + tier_offset                                      # delivery-tier-specific base risk
    + 0.3 * (df["days_to_deliver"] > 5).astype(int)    # slow delivery -> mild extra risk
    + 0.5 * df["_latent_propensity"]                   # personal propensity bleeds through
    + noise                                             # deliberate noise so it's not trivial
)


def find_bias_for_target_rate(logit_values, target_rate, tol=0.002, max_iter=100):
    """Binary-search a bias term so mean(sigmoid(logit + bias)) ~= target_rate."""
    lo, hi = -10.0, 10.0
    mid = 0.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        rate = sigmoid(logit_values + mid).mean()
        if abs(rate - target_rate) < tol:
            return mid
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return mid


bias = find_bias_for_target_rate(logit.values, TARGET_BASE_RETURN_RATE)
print(f"Calibrated bias: {bias:.4f} (target base rate {TARGET_BASE_RETURN_RATE})")

p_return = sigmoid(logit + bias)
df["p_return_true"] = p_return.round(4)      # ground truth probability, validation only
df["returned"] = RNG.binomial(1, p_return)   # the actual label


# =======================================================================
# Step 5 - Report + save
# =======================================================================

print(f"Base return rate: {df['returned'].mean():.3f}")
print(f"Return rate by category:\n{df.groupby('category')['returned'].mean()}")
print(f"Return rate by payment method:\n{df.groupby('payment_method')['returned'].mean()}")
print(f"Return rate, first-time vs repeat:\n{df.groupby('is_first_time_buyer')['returned'].mean()}")

# Full version, includes ground-truth columns - keep for your own
# validation/debugging, do NOT train a model on this file.
df.to_csv("return_risk/synthetic_orders_full.csv", index=False)

# Production version - what the model should actually train on
# (ground-truth leakage columns removed).
model_df = df.drop(columns=["_latent_propensity", "p_return_true"])
model_df.to_csv("return_risk/synthetic_orders_model.csv", index=False)

print("\nSaved:")
print(" - synthetic_orders_full.csv  (includes ground truth, for validation only)")
print(" - synthetic_orders_model.csv (use this for training)")