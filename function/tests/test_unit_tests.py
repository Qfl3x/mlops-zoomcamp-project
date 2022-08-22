"""
Base Unit tests"""

import base64
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pytest
import scipy
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

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

    os.system("mkdir tmp")
    os.system("cp ./model/* ./tmp")
    os.system("cp ./data/future.csv ./tmp")


@pytest.mark.offlinenodata
def test_preprocessors():

    download_files()
    with open("./tmp/preprocessors.pkl", "rb") as file:
        (dv, scaler) = pickle.load(file)

    assert (type(dv) == DictVectorizer) and (type(scaler) == StandardScaler)
    os.system("rm -rf ./tmp/")


@pytest.mark.offlinenodata
def test_booster():

    download_files()

    booster = xgb.Booster({"verbosity": 0})
    booster.load_model("./tmp/model.xgb")

    assert type(booster) == xgb.Booster
    os.system("rm -rf ./tmp/")


def read_data():
    df = pd.read_csv("./tmp/future.csv")
    return df


def predict_binary(probs):
    return (probs >= 0.5).astype("bool")


def preprocess_data(df):
    with open("./tmp/preprocessors.pkl", "rb") as file:
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
    booster.load_model("./tmp/model.xgb")

    X_DMatrix = xgb.DMatrix(X)
    return predict_binary(booster.predict(X_DMatrix))


@pytest.mark.offlinedata
def test_preprocessing():

    download_files()

    df = read_data()
    X = preprocess_data(df)

    assert type(X) == scipy.sparse._csr.csr_matrix
    os.system("rm -rf ./tmp/")


@pytest.mark.offlinedata
def test_prediction():

    download_files()

    df = read_data()
    X = preprocess_data(df)
    preds = predict(X)
    assert type(preds) == np.ndarray
