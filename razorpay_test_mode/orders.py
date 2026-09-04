import os
import razorpay
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta
import time

load_dotenv()
client = razorpay.Client(auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET")))

RNG = np.random.default_rng(42)
N_ORDERS = 25 

ORDER_VALUE_MU, ORDER_VALUE_SIGMA = 7.09, 1.086

CATEGORIES = ["fashion", "electronics", "home", "beauty", "grocery"]
CATEGORY_PROBS = [0.38, 0.20, 0.17, 0.15, 0.10]
PAYMENT_METHODS = ["COD", "UPI", "card", "netbanking"]
PAYMENT_PROBS = [0.34, 0.40, 0.20, 0.06]
PINCODE_TIERS = ["metro", "tier2", "tier3"]
TIER_PROBS = [0.45, 0.35, 0.20]

records = []

print(f"Creating {N_ORDERS} real Razorpay test-mode orders...\n")

for i in range(N_ORDERS):
    order_value = float(np.clip(RNG.lognormal(ORDER_VALUE_MU, ORDER_VALUE_SIGMA), 250, 20000))
    amount_paise = int(round(order_value * 100))  # Razorpay wants paise, not rupees

    try:
        razorpay_order = client.order.create({
            "amount": amount_paise,
            "currency": "INR",
            "receipt": f"return_risk_batch_{i}",
            "notes": {"purpose": "return-risk-agent"}
        })
    except Exception as e:
        print(f"  Order {i} FAILED: {e}")
        continue

    is_first_time = int(RNG.random() < 0.30)
    record = {
        "razorpay_order_id": razorpay_order["id"],
        "razorpay_status": razorpay_order["status"],
        "order_value": round(order_value, 2),
        "category": RNG.choice(CATEGORIES, p=CATEGORY_PROBS),
        "discount_pct": round(float(np.clip(RNG.beta(2, 5), 0, 0.75)), 3),
        "payment_method": RNG.choice(PAYMENT_METHODS, p=PAYMENT_PROBS),
        "is_first_time_buyer": is_first_time,
        "customer_past_orders": 0 if is_first_time else int(RNG.poisson(3)) + 1,
        "customer_past_return_rate": (
            np.nan if is_first_time else round(float(np.clip(RNG.beta(2, 6), 0, 1)), 3)
        ),
        "delivery_pincode_tier": RNG.choice(PINCODE_TIERS, p=TIER_PROBS),
        "size_variant_flag": 0, 
        "days_to_deliver": int(np.clip(RNG.poisson(3), 1, 12)),
        "time_of_day_ordered": RNG.choice(["morning", "afternoon", "evening", "night"]),
    }
    record["size_variant_flag"] = int(record["category"] == "fashion" and RNG.random() < 0.85)

    records.append(record)
    print(f"  [{i+1}/{N_ORDERS}] Created {razorpay_order['id']}  "
          f"amount=Rs.{order_value:.2f}  status={razorpay_order['status']}")

    time.sleep(0.1)  

df = pd.DataFrame(records)
print(f"\nSuccessfully created and enriched {len(df)} / {N_ORDERS} orders")
print(df.head())

df.to_csv("data/test_mode/razorpay_enriched_orders.csv", index=False)
print("\nSaved: data/test_mode/razorpay_enriched_orders.csv")