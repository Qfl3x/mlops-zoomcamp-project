"""
Online Integration test"""
import base64
import json
import os
import subprocess
from pathlib import Path

import pytest
import requests
from dotenv import load_dotenv
from google.cloud import storage
from requests.packages.urllib3.util.retry import Retry

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
MODEL_BUCKET = os.getenv("MODEL_BUCKET")
DATA_BUCKET = os.getenv("DATA_BUCKET")


@pytest.mark.online
def test_download_files():
    """
    Download Files from GS Bucket"""

    os.system("mkdir tmp")
    storage_client = storage.Client()
    storage_client_data = storage.Client()

    model_bucket = storage_client.bucket(MODEL_BUCKET)
    data_bucket = storage_client_data.bucket(DATA_BUCKET)

    preprocessors_blob = model_bucket.blob("preprocessors.pkl")
    model_blob = model_bucket.blob("model.xgb")

    preprocessors_blob.download_to_filename("./tmp/preprocessors.pkl")
    model_blob.download_to_filename("./tmp/model.xgb")

    data_blob = data_bucket.blob("future.csv")
    data_blob.download_to_filename("./tmp/future.csv")

    files_in_tmp = os.listdir("./tmp")
    assert (
        ("future.csv" in files_in_tmp)
        and ("preprocessors.pkl" in files_in_tmp)
        and ("model.xgb" in files_in_tmp)
    )
    os.system("rm -rf ./tmp/")


def download_files():
    """
    Download Files from GS Bucket"""
    os.system("mkdir tmp")
    storage_client_data = storage.Client()

    data_bucket = storage_client_data.bucket(DATA_BUCKET)

    data_blob = data_bucket.blob("prediction.csv")
    data_blob.download_to_filename("./tmp/prediction.csv")


@pytest.mark.online
def test_function_online():
    os.system("mkdir tmp")
    storage_client_data = storage.Client()

    data_bucket = storage_client_data.bucket(DATA_BUCKET)

    data_blob = data_bucket.blob("prediction.csv")
    first_date = data_blob

    os.system("cd ../../ && ./upload-files.sh")

    data_blob_new = data_bucket.blob("prediction.csv")

    second_date = data_blob.updated

    assert second_date != first_date
