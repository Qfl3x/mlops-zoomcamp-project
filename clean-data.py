"""
Splits data into current and future sets, as well as split the current set into training, validation and test sets
Copies the result to the train and monitoring directories as well as to the GCS bucket."""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_BUCKET = os.getenv("DATA_BUCKET")
df_orig = pd.read_csv("data/telecom_customer_churn.csv")

df_orig["client_churned"] = df_orig["Customer Status"] == "Churned"
df_orig = df_orig[df_orig["Customer Status"] != "Joined"]
columns_to_drop = [
    "City",
    "Latitude",
    "Longitude",
    "Churn Category",
    "Churn Reason",
    "Customer Status",
]
df_orig = df_orig.drop(columns_to_drop, axis=1)  # Drop All columns with nans
df_orig.dropna(axis=1, how="any", inplace=True)


df_curr, df_future = train_test_split(df_orig, test_size=0.2)
df_future.to_csv("data/future.csv")

df_train, df_test = train_test_split(
    df_curr, test_size=0.4, stratify=df_curr["client_churned"]
)
df_test, df_val = train_test_split(
    df_test, test_size=0.5, stratify=df_test["client_churned"]
)

df_val.to_csv("data/val.csv")
df_train.to_csv("data/train.csv")
df_test.to_csv("data/test.csv")

os.system(f"gsutil -m cp data/*.csv gs://{DATA_BUCKET}/")
os.system("cp -r data/ train/")
os.system("cp -r data/ monitoring/")
