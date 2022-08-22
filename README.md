# MLOps Zoomcamp final project: Offline Churn Prediction for a Telecom company.

## Data Science part:

### Problem Statement:

In this project, I'll use this Kaggle dataset: https://www.kaggle.com/datasets/shilongzhuang/telecom-customer-churn-by-maven-analytics?resource=download

and try to predict whether or not a client is going to churn or not in this quarter based on several parameters on the client themselves, as well as their consumption figures.

To assess the performance of the model, I'll be using an f5 score (fbeta with beta=5). The reasons for this are:

+ The Dataset is highly skewed (As usual for Churn prediction) towards clients not churning
+ In this instance, Recall is more important than precision. Falsely detecting a client as churned and potentially giving them a free offer is of much less severity than losing a client for the Telecom company

As such, it is clear that our beta for the fbeta metric has to be greater than 1. I've chosen 5 arbitarily since while we don't care too much about precision, it should still be taken into account.

### Model Used:

I'll be using an XGBoost Classifier whose hyperparameters I'll get using Hyperoptimization.

## Cleaning Data and uploading it:

I put the dataset csv from Kaggle in the `data` folder in the parent directory and run `python clean_data.py`, this removes bad columns and cuts the data into training, validation, test and future (set on which the deployment will run) sets. It also uploads the files to a Bucket for the Deployed Function as well as copy the folder to the `train` and `monitoring` folders.

## Training the model:

To train the model, I go through 5 steps:

1. Read the data, preprocess it using a Scaler and a DictVectorizer
2. Run a Hyperoptimzer for XGBoost hyperparameters, store results (Without the models) in the `hpo-xgboost-churn` experiment
3. Get the best (default 10) runs from the previous experiment and retrain them and log the entire model along with inference time and validation metrics with preprocessors in the `chosen-models-churn` experiment along with a version tag
4. Manually choose the best model, copy its run ID
5. Run `python register_model.py <ID>` to register the model, measure its test metric and upload its model files to a GCS Bucket.

## Deploying the model:

The model is deployed using a Cloud Functions Function that's set to be triggered by the data bucket. If one of the Data Bucket's files is uploaded or changed, the Function will automatically get the model and preprocessors from the model bucket and run them on the `future.csv` file on the Data Bucket.

## Monitoring:
For monitoring, I've used a simple EvidentlyAI HTML report since there is no online deployment. In case the reference score is considerably better than the current score, it will send an email alert.

# Running the files:

For running the files, you can either use the provided Makefile for easy deployment only. Or Train everything then deploy.

### Requirements:
`terraform`
`pipenv`
`make` (See your distribution's wiki)
Fully functioning `gcloud` tools. (Setup with `gcloud init`)

Before running anything, it is necessary to manually setup certain parameters for security and size reasons:
1. Download the `telecom_customer_churn.csv` file from Kaggle
2. Create two service-accounts, give one absolute admin control as it will be used by Terraform, and add another service-account that will be used to manage permissions around the files
3. For each service-account, download a JSON key. Put the Terraform's JSON key in `infrastructure` as `terraform-account.json` (or modify `main.tf` and the scripts). For the other account keep the JSON anywhere.
4. Create a `baseenv` file in the root directory. In it put 4 fields:
   + `GOOGLE_APPLICATION_CREDENTIALS`: The Absolute Path to the JSON key for the host account (Non-Terraform)
   + `GOOGLE_ACCOUNT_NAME`: email address of the host account
   + `REMOTE_IP`: (Optional) Remote IP of the server on which you're running MLFlow (Note: Prefect will need to be configured accordingly if the agent is running on a different host). If blank, will default to `localhost`
   + `SENDGRID_API_KEY`: (Optional) API key for SendGrid for automated email
   + `SENDGRID_INTEGRATION`: (Optional) Set as any value to skip the SendGrid automated email
   + `PERSONAL_EMAIL`: (Optional) In case SendGrid is configured, put your personal email here
5. You may need to change the prefix value in `variables.tf` in `infrastructure` to avoid creating resources with the same name as mine!

## Deploy everything:

Once all is set, simply use `make build` and you will have a working version that uses the provided xgboost model! To test it, you can use either `upload-files.sh` to upload files to the data bucket created by terraform (The environment variables are automatically setup), or use `pytest -m online` to start the online tests that will test your buckets as well as trigger the function. You can see the names of the buckets created by terraform at your `.env` files.

## Prefect Setup:

For training and monitoring, it is necessary to setup Prefect. An appropriate version of prefect was installed when running `pipenv install`. Run Prefect using `prefect orion start --host 0.0.0.0`.

*Note*: In my case, even though prefect was hosted on 0.0.0.0, accessing it remotely was impossible, the page loads but there is no data. I used an ssh localhost tunnel instead: `ssh -NL 4200:localhost:4200 IP_GOES_HERE`. Then access Prefect UI in `localhost:4200`.

In the Prefect UI. Two Blocks must be set.
1. The first Block is the Block where all the deployments will be stored. It can be a LocalFileSystem or a Remote Bucket of your choice.
2. For training, it is also necessary to setup a `String` Block called `value-counter` and give it the value `1` (Any value will work). This value corresponds to the tag that will be set on the MLFlow runs. Keep Prefect running for Monitoring and training.

## Monitoring:

For Monitoring, go to `monitoring`, modify the  and execute `./prefect-deploy.sh YOUR_BLOCK` and pass the first created Block where the deployment will be stored, example `./prefect-deploy.sh gcs/main`.

To get the evidently run an agent in the local machine (Where `gsutil` is setup) and run the deployment using Prefect UI (In a real-world scenario the deployment will be scheduled every month or so). An HTML report should be created at the project's root directory.

## Training:

For Training, go to the `mlflow` directory and run `mlflow` UI in the configuration that suits you (A `./run_mlflow.sh` script exists with hardcoded backend and artifact store). Once done, run `./prefect-deploy.sh YOUR_BLOCK` to build and apply the deployment to Prefect then in Prefect run the deployment with default values.

The deployment will create two experiments:
1. `hpo-xgboost-churn`: Which will temporarily contain the runs of the hyperoptimizer
2. `chosen-models-churn`: Will will contain the top 10 models with their validation metrics and a version tag

Next choose the model that suits you best based on the metrics provided and run `pipenv run python register_model.py RUN_ID`, where `RUN_ID` is the ID of the run chosen, example: `pipenv run python register_model.py c4879e24e478401b982969d30db7c398`. The chosen model will be registered in the model registry with a test set score, and downloaded to the `model` folder in the root directory.

## Testing:

For Testing, two test masks exist: `online`, `offlinenodata` and `offlinedata`. `online` tests (ran with `pytest -m online`) will test the infrastructure, while `offline` tests will test everything else. `offlinedata` will run data-dependent tests, while `offlinenodata` will run non-data-dependent tests.

## Destroying the Infrastructure:

To destroy the infrastructure, go to `infrastructure` and run `./terraform-destroy.sh`
