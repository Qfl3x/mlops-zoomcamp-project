import base64
import json
import os
import pickle
import sys

import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
MODEL_BUCKET = os.getenv("MODEL_BUCKET")
DATA_BUCKET = os.getenv("DATA_BUCKET")


categorical = [
    "Gender",
    "Zip Code",
    "Offer",
    "Contract",
    "Phone Service",
    "Internet Service",
    "Paperless Billing",
    "Payment Method",
]
numerical = [
    "Age",
    "Number of Dependents",
    "Number of Referrals",
    "Tenure in Months",
    "Monthly Charge",
    "Total Charges",
    "Total Refunds",
    "Total Extra Data Charges",
    "Total Revenue",
]
target = "client_churned"


def download_files():
    """
    Download Files from GS Bucket"""
    storage_client = storage.Client()
    storage_client_data = storage.Client()

    model_bucket = storage_client.bucket(MODEL_BUCKET)
    data_bucket = storage_client_data.bucket(DATA_BUCKET)

    preprocessors_blob = model_bucket.blob("preprocessors.pkl")
    model_blob = model_bucket.blob("model.xgb")

    preprocessors_blob.download_to_filename("/tmp/preprocessors.pkl")
    model_blob.download_to_filename("/tmp/model.xgb")

    data_blob = data_bucket.blob("future.csv")
    data_blob.download_to_filename("/tmp/future.csv")


def upload_output():

    storage_client = storage.Client()
    bucket = storage_client.bucket(DATA_BUCKET)
    blob = bucket.blob("prediction.csv")

    blob.upload_from_filename("/tmp/prediction.csv")


def read_data():
    df = pd.read_csv("/tmp/future.csv")
    return df


def predict_binary(probs):
    return (probs >= 0.5).astype("bool")


def preprocess_data(df):
    with open("/tmp/preprocessors.pkl", "rb") as file:
        (dv, scaler) = pickle.load(file)

    df = df.copy()
    df.drop("Customer ID", axis=1, inplace=True)
    target_vec = df[target]
    df.drop(target, axis=1, inplace=True)
    df[categorical] = df[categorical].astype("str")
    df[numerical] = scaler.transform(df[numerical])
    df_dict = df.to_dict(orient="records")
    X = dv.transform(df_dict)
    return X


def predict(X):

    booster = xgb.Booster({"verbosity": 0})
    booster.load_model("/tmp/model.xgb")

    X_DMatrix = xgb.DMatrix(X)
    return predict_binary(booster.predict(X_DMatrix))


def endpoint(event, context):
    """
    Main function to be exported.
    Takes the event and outputs the prediction and sends it to the Pull stream"""
    # ride = base64.b64decode(event["data"]).decode("utf-8")
    # ride = json.loads(ride)
    if event["data"] != "debug":
        download_files()

    df = read_data()
    X = preprocess_data(df)
    preds = predict(X)
    df["churn_prediction"] = preds
    df.to_csv("/tmp/prediction.csv")
    if event["data"] != "debug":
        upload_output()
