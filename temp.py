import pandas as pd
import numpy as np

df9 = pd.read_csv("UCI_dSet/online_retail_09_10.csv", encoding="ISO-8859-1")
df10 = pd.read_csv("UCI_dSet/online_retail_10_11.csv", encoding="ISO-8859-1")
print(df9.columns.tolist())
print(df9.head())