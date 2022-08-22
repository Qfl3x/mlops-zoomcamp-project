"""
Offline Integration test"""
import os
import subprocess
from pathlib import Path

import pytest
import requests
from dotenv import load_dotenv
from requests.packages.urllib3.util.retry import Retry

load_dotenv()


@pytest.mark.offline
def test_function_offline():

    # Initiate the files in tmp for the function
    os.system("cp ../../model/* /tmp")
    os.system("cp ../../data/future.csv /tmp")

    port = 8080

    pubsub_message = {"data": {"data": "debug"}}

    current_path = Path(os.path.dirname(__file__))

    parent_path = current_path.parent
    process = subprocess.Popen(
        [
            "functions-framework",
            "--target",
            "endpoint",
            "--signature-type",
            "event",
            "--port",
            str(port),
        ],
        cwd=parent_path,
        stdout=subprocess.PIPE,
    )

    url = f"http://localhost:{port}/"

    retry_policy = Retry(total=6, backoff_factor=1)
    retry_adapter = requests.adapters.HTTPAdapter(max_retries=retry_policy)

    session = requests.Session()
    session.mount(url, retry_adapter)

    response = session.post(url, json=pubsub_message)

    # Stop the functions framework process
    process.kill()
    process.wait()
    out, err = process.communicate()

    files_in_tmp = os.listdir("/tmp/")
    assert "prediction.csv" in files_in_tmp
