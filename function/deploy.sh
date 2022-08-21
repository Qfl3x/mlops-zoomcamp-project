#!/usr/bin/env bash

gcloud functions deploy endpoint \
    --trigger-bucket $DATA_BUCKET \
    --runtime python39
