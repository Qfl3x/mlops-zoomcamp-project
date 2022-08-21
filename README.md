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
For monitoring, I've used a simple EvidentlyAI HTML report since there is no online deployment.

# Running the files:
