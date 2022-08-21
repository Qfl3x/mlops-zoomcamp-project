"""
Prefect Flow that does the Batch Monitoring"""
import json
import os
import pickle

import xgboost as xgb
import pandas as pd
from prefect import flow, task
from pymongo import MongoClient
import pyarrow.parquet as pq

from evidently import ColumnMapping

from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, ClassificationPerformanceTab

from evidently.model_profile import Profile
from evidently.model_profile.sections import (
    DataDriftProfileSection,
    ClassificationPerformanceProfileSection,
)


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


def preprocess_data(df):
    """Preprocesses data"""
    with open("./model/preprocessors.pkl", "rb") as file:
        (dv, scaler) = pickle.load(file)

    df = df.copy()
    df.drop("Customer ID", axis=1, inplace=True)
    target_vec = df[target]
    df.drop(target, axis=1, inplace=True)
    df[categorical] = df[categorical].astype("str")
    df[numerical] = scaler.transform(df[numerical])
    df_dict = df.to_dict(orient="records")
    X = dv.transform(df_dict)
    return [X, target_vec]


def predict_binary(probs):
    return (probs >= 0.5).astype("bool")


@task
def load_reference_data(filename):
    """
    Loads the reference filename data."""
    booster = xgb.Booster({"verbosity": 0})
    booster.load_model("./model/model.xgb")

    reference_data = pd.read_csv(filename)
    # Create features
    [X, target_vec] = preprocess_data(reference_data)
    X_DMatrix = xgb.DMatrix(X)
    # add target column
    reference_data["churn_prediction"] = predict_binary(booster.predict(X_DMatrix))
    return reference_data


@task
def fetch_data():
    """
    Gets Output data from the GCS bucket, keeps Customer ID"""
    os.system("gsutil cp gs://mlops-project-data/prediction.csv ./data/prediction.csv")
    df = pd.read_csv("./data/prediction.csv")
    return df


@task
def run_evidently(ref_data, data):
    """
    Runs Evidently metrics on the data"""
    data.drop(["Unnamed: 0.1", "Unnamed: 0"], axis=1, inplace=True)
    data["churn_prediction"] = data["churn_prediction"].astype(int)
    data[target] = data[target].astype(int)
    ref_data[target] = ref_data[target].astype(int)

    profile = Profile(
        sections=[DataDriftProfileSection(), ClassificationPerformanceProfileSection()]
    )
    mapping = ColumnMapping(
        prediction="churn_prediction",
        numerical_features=numerical,
        categorical_features=categorical,
        datetime_features=[],
        target=target,
    )
    profile.calculate(ref_data, data, mapping)

    dashboard = Dashboard(
        tabs=[DataDriftTab(), ClassificationPerformanceTab(verbose_level=0)]
    )
    dashboard.calculate(ref_data, data, mapping)
    return json.loads(profile.json()), dashboard


@task
def save_html_report(result):
    result[1].save("/home/qfl3x/project/monitoring/evidently_report.html")


@flow
def batch_analyze():
    # upload_target("./data/prediction.csv")
    ref_data = load_reference_data("./data/train.csv")
    data = fetch_data()
    result = run_evidently(ref_data, data)
    save_html_report(result)
