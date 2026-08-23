import os
import razorpay
from dotenv import load_dotenv
import time

print("[1] Loading .env...")
load_dotenv()

key_id = os.getenv("RAZORPAY_KEY_ID")
key_secret = os.getenv("RAZORPAY_KEY_SECRET")

id_ok = key_id is not None
secret_ok = key_secret is not None
print("[2] Key ID loaded:", id_ok)
print("[3] Key Secret loaded:", secret_ok)

if not id_ok or not secret_ok:
    print("STOPPING: .env not loaded - check the .env file is in the same folder as this notebook.")
else:
    print("[4] Creating Razorpay client...")
    client = razorpay.Client(auth=(key_id, key_secret))

    print("[5] Attempting order.create() call...")
    start = time.time()
    try:
        order = client.order.create({
            "amount": 100000,
            "currency": "INR",
            "receipt": "receipt_debug_test",
        })
        elapsed = time.time() - start
        print("[6] SUCCESS in", round(elapsed, 2), "seconds")
        print(order)
    except Exception as e:
        elapsed = time.time() - start
        print("[6] FAILED after", round(elapsed, 2), "seconds")
        print("Error type:", type(e).__name__)
        print("Error message:", e)