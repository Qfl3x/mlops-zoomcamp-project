import os
import pickle
import sys

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import StandardScaler

import mlflow

MODEL_BUCKET = os.getenv("MODEL_BUCKET")

REMOTE_TRACKING_IP = os.getenv("REMOTE_IP", "localhost")
MLFLOW_TRACKING_URI = f"http://{REMOTE_TRACKING_IP}:5000"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

EXPERIMENT_NAME = "chosen-models-churn"
MODEL_NAME = "churn-prediction"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

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


def read_data():
    df_test = pd.read_csv("data/test.csv")
    df_test.drop("Customer ID", axis=1, inplace=True)
    return df_test


def preprocess_data(df, dv=None, scaler=None, train=True):

    df = df.copy()
    target_vec = df[target]
    df.drop(target, axis=1, inplace=True)
    df[categorical] = df[categorical].astype("str")
    if train:
        scaler = StandardScaler()
        df[numerical] = scaler.fit_transform(df[numerical])

        df_dict = df.to_dict(orient="records")
        dv = DictVectorizer()
        X = dv.fit_transform(df_dict)
        return (scaler, dv, X, target_vec)
    else:
        df[numerical] = scaler.transform(df[numerical])
        df_dict = df.to_dict(orient="records")
        X = dv.transform(df_dict)
        return (X, target_vec)


run_id = sys.argv[1]
mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{run_id}/model/preprocessors.pkl", dst_path="./"
)

with open("preprocessors.pkl", "rb") as file:
    (dv, scaler) = pickle.load(file)

df_test = read_data()
X_test, y_test = preprocess_data(df_test, dv=dv, scaler=scaler, train=False)
print(X_test.shape)

logged_model = f"runs:/{run_id}/model"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
y_pred = loaded_model.predict(X_test)

test_score = fbeta_score(y_test, y_pred, beta=5)
try:
    client.create_registered_model(name=MODEL_NAME)
except:
    pass
description = f"test score: {test_score}"
mv = client.create_model_version(
    name=MODEL_NAME, source=logged_model, run_id=run_id, description=description
)
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=mv.version,
    stage="production",
    archive_existing_versions=True,
)

# Update Model to bucket

mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{run_id}/model/preprocessors.pkl", dst_path="../model"
)
mlflow.artifacts.download_artifacts(
    artifact_uri=f"runs:/{run_id}/model/model.xgb", dst_path="../model"
)

# os.system(f"gsutil cp model/* gs://{MODEL_BUCKET}/")
