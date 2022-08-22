"""
Prefect Flow that does the Batch Monitoring"""
import json
import os
import pickle

import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from dotenv import load_dotenv
from evidently import ColumnMapping
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import ClassificationPerformanceTab, DataDriftTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import (
    ClassificationPerformanceProfileSection,
    DataDriftProfileSection,
)
from pymongo import MongoClient
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from sklearn.metrics import fbeta_score

from prefect import flow, task

load_dotenv()

DATA_BUCKET = os.getenv("DATA_BUCKET")
SENDGRID_INTEGRATION = os.getenv("SENDGRID_INTEGRATION", False)
if SENDGRID_INTEGRATION != False:
    SENDGRID_INTEGRATION = True

EMAIL = os.getenv("PERSONAL_EMAIL")

PROJECT_PATH = os.getenv("PROJECT_PATH")
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
    os.system(f"gsutil cp gs://{DATA_BUCKET}/prediction.csv ./data/prediction.csv")
    df = pd.read_csv("./data/prediction.csv")
    return df


@task
def assess_fbeta(ref_data, data):
    y_true = data[target].tolist()
    y_pred = data["churn_prediction"].tolist()

    ref_y_true = ref_data[target].tolist()
    ref_y_pred = ref_data["churn_prediction"].tolist()

    ref_fbeta = fbeta_score(ref_y_true, ref_y_pred, beta=5)

    data_fbeta = fbeta_score(y_true, y_pred, beta=5)
    return ref_fbeta - data_fbeta


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
    result[1].save(f"{PROJECT_PATH}/evidently_report.html")


@task
def send_email(score_diff):

    message = Mail(
        from_email=EMAIL,
        to_emails=EMAIL,
        subject="ALERT: FBeta score too low.",
        html_content=f"fbeta score difference: {score_diff}",
    )
    try:
        sg = SendGridAPIClient(os.environ.get("SENDGRID_API_KEY"))
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(e.message)


@flow
def batch_analyze():
    ref_data = load_reference_data("./data/train.csv")
    data = fetch_data()
    result = run_evidently(ref_data, data)
    save_html_report(result)
    fbeta_diff = assess_fbeta(ref_data, data)
    if fbeta_diff > 0.15 and SENDGRID_INTEGRATION:
        send_email(score_diff)
