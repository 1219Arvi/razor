import pandas as pd
import numpy as np

df9 = pd.read_csv("UCI_dSet/online_retail_09_10.csv", encoding="ISO-8859-1")
df10 = pd.read_csv("UCI_dSet/online_retail_10_11.csv", encoding="ISO-8859-1")

df9 = df9.dropna(subset=["CustomerID"])
df10 = df10.dropna(subset=["CustomerID"])

df9["is_cancellation"] = df9["InvoiceNo"].astype(str).str.startswith("C")
df10["is_cancellation"] = df10["InvoiceNo"].astype(str).str.startswith("C")

df9["line_value"] = df9["Quantity"] * df9["UnitPrice"]
df10["line_value"] = df10["Quantity"] * df10["UnitPrice"]

orders = pd.concat([df9, df10]).groupby("InvoiceNo").agg(
    customer_id=("CustomerID", "first"),
    order_value=("line_value", "sum"),
    n_items=("StockCode", "nunique"),
    total_qty=("Quantity", "sum"),
    invoice_date=("InvoiceDate", "first"),
    is_cancellation=("is_cancellation", "first"),
    country=("Country", "first"),
).reset_index()

print(orders.shape)
print(orders.head())