#!/bin/bash

mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root gs://project-mlflow/artifacts --host 0.0.0.0

