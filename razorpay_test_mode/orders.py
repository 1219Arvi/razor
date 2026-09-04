import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import razorpay
from dotenv import load_dotenv

RNG_SEED = 42
ORDER_VALUE_MU, ORDER_VALUE_SIGMA = 7.09, 1.086

CATEGORIES = ["fashion", "electronics", "home", "beauty", "grocery"]
CATEGORY_PROBS = [0.38, 0.20, 0.17, 0.15, 0.10]
PAYMENT_METHODS = ["COD", "UPI", "card", "netbanking"]
PAYMENT_PROBS = [0.34, 0.40, 0.20, 0.06]
PINCODE_TIERS = ["metro", "tier2", "tier3"]
TIER_PROBS = [0.45, 0.35, 0.20]

OUTPUT_CSV = "data/test_mode/razorpay_enriched_orders.csv"


def _get_client() -> razorpay.Client:
    load_dotenv()
    key_id = os.getenv("RAZORPAY_KEY_ID")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET")
    if not key_id or not key_secret:
        raise RuntimeError("RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET not found - check your .env file.")
    return razorpay.Client(auth=(key_id, key_secret))


def create_test_orders(n_orders: int = 25, seed: int = RNG_SEED, save: bool = True, sleep_seconds: float = 0.1) -> pd.DataFrame:
    """
    Creates `n_orders` real Razorpay *test-mode* orders via the API, and
    attaches locally-fabricated feature columns (category, discount, etc.)
    to each real order ID so the trained model has something to score.
    """
    client = _get_client()
    rng = np.random.default_rng(seed)
    records = []

    print(f"Creating {n_orders} real Razorpay test-mode orders...\n")

    for i in range(n_orders):
        order_value = float(np.clip(rng.lognormal(ORDER_VALUE_MU, ORDER_VALUE_SIGMA), 250, 20000))
        amount_paise = int(round(order_value * 100))  # Razorpay wants paise, not rupees

        try:
            razorpay_order = client.order.create({
                "amount": amount_paise,
                "currency": "INR",
                "receipt": f"return_risk_batch_{i}",
                "notes": {"purpose": "return-risk-agent"},
            })
        except Exception as e:
            print(f"  Order {i} FAILED: {e}")
            continue

        is_first_time = int(rng.random() < 0.30)
        category = rng.choice(CATEGORIES, p=CATEGORY_PROBS)
        record = {
            "razorpay_order_id": razorpay_order["id"],
            "razorpay_status": razorpay_order["status"],
            "order_value": round(order_value, 2),
            "category": category,
            "discount_pct": round(float(np.clip(rng.beta(2, 5), 0, 0.75)), 3),
            "payment_method": rng.choice(PAYMENT_METHODS, p=PAYMENT_PROBS),
            "is_first_time_buyer": is_first_time,
            "customer_past_orders": 0 if is_first_time else int(rng.poisson(3)) + 1,
            "customer_past_return_rate": (
                np.nan if is_first_time else round(float(np.clip(rng.beta(2, 6), 0, 1)), 3)
            ),
            "delivery_pincode_tier": rng.choice(PINCODE_TIERS, p=TIER_PROBS),
            "size_variant_flag": int(category == "fashion" and rng.random() < 0.85),
            "days_to_deliver": int(np.clip(rng.poisson(3), 1, 12)),
            "time_of_day_ordered": rng.choice(["morning", "afternoon", "evening", "night"]),
        }

        records.append(record)
        print(f"  [{i + 1}/{n_orders}] Created {razorpay_order['id']}  "
              f"amount=Rs.{order_value:.2f}  status={razorpay_order['status']}")

        time.sleep(sleep_seconds)

    df = pd.DataFrame(records)
    print(f"\nSuccessfully created and enriched {len(df)} / {n_orders} orders")
    print(df.head())

    if save:
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved: {OUTPUT_CSV}")

    return df


if __name__ == "__main__":
    create_test_orders()