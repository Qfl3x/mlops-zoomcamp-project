"""
Hyper optimizer. Logs to MLflow at the IP address below
"""
import os
import time
import warnings

warnings.filterwarnings("ignore")
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll import scope
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from prefect.blocks.system import String
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import fbeta_score, recall_score
from sklearn.preprocessing import StandardScaler

import mlflow
from prefect import flow, task

REMOTE_TRACKING_IP = os.getenv("REMOTE_IP", "localhost")
MLFLOW_TRACKING_URI = f"http://{REMOTE_TRACKING_IP}:5000"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

HPO_EXPERIMENT_NAME = "hpo-xgboost-churn"
EXPERIMENT_NAME = "chosen-models-churn"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(HPO_EXPERIMENT_NAME)

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


@task
def read_data():
    """
    Reads data and removes Customer ID's which are unnecessary at this point"""
    df_train = pd.read_csv("data/train.csv")
    df_val = pd.read_csv("data/val.csv")
    df_train.drop("Customer ID", axis=1, inplace=True)
    df_val.drop("Customer ID", axis=1, inplace=True)
    return [df_train, df_val]


@task
def preprocess_data(df, dv=None, scaler=None, train=True):
    """
    Preprocesses data using a Scaler and a DictVectorizer.
    train: Whether or not this is a training pass to train the preprocessors"""

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
        return [scaler, dv, X, target_vec]
    else:
        df[numerical] = scaler.transform(df[numerical])
        df_dict = df.to_dict(orient="records")
        X = dv.transform(df_dict)
        return [X, target_vec]


search_space = {
    "max_depth": scope.int(hp.quniform("max_depth", 4, 100, 1)),
    "learning_rate": hp.loguniform("learning_rate", -3, 0),
    "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
    "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
    "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
    "objective": "binary:logistic",
    "seed": 42,
}


def predict_binary(probs):
    return (probs >= 0.5).astype("int")


@task
def hyperoptimizer(X_train, y_train, X_val, y_val):
    """
    Does the Hyperparamater passes over the data"""

    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)
            model = xgb.XGBClassifier(verbosity=0)
            model.set_params(**params)
            model.fit(X_train, y_train)
            y_pred_probs = model.predict(X_val)
            y_pred = predict_binary(y_pred_probs)
            fbeta = fbeta_score(y_val, y_pred, beta=5)
            precision = fbeta_score(y_val, y_pred, beta=0)
            mlflow.log_metric("fbeta", fbeta)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall_score(y_val, y_pred))

        return {"loss": -fbeta, "status": STATUS_OK}

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials(),
    )
    return 0


@task
def train_and_log_model(params, X_train, y_train, X_val, y_val, tag):
    """
    Once given the parameters of the model, it retrains and saves all output along with a version tag"""

    with mlflow.start_run():
        mlflow.set_tag("version_tag", tag)
        params = space_eval(search_space, params)
        model = xgb.XGBClassifier(verbosity=0)
        model.set_params(**params)
        model.fit(X_train, y_train)

        # evaluate model on the validation set
        start_time = time.time()
        y_pred_probs = model.predict(X_val)
        end_time = time.time()
        inference_time = end_time - start_time
        y_pred = predict_binary(y_pred_probs)
        fbeta = fbeta_score(y_val, y_pred, beta=5)
        precision = fbeta_score(y_val, y_pred, beta=0)
        mlflow.log_metric("fbeta", fbeta)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall_score(y_val, y_pred))
        mlflow.log_metric("Inference time", inference_time)
        mlflow.log_artifact("preprocessors.pkl", artifact_path="model")


@flow
def run(log_top, X_train, y_train, X_val, y_val, *args):
    """
    Passes over the best log_top experiments and calls train_and_log_params on each"""
    # Setup version tag
    current_tag_block = String.load("version-counter")
    print(current_tag_block)
    current_tag = int(current_tag_block.value)

    # retrieve the top_n model runs and log the models to MLflow
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=log_top,
        order_by=["metrics.fbeta DESC"],
    )
    for run in runs:
        train_and_log_model(
            params=run.data.params,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            tag=current_tag,
        )
    os.system("rm preprocessors.pkl")
    # Clean the HPO Experiment, we only need the chosen models
    experiment_id = client.get_experiment_by_name(HPO_EXPERIMENT_NAME).experiment_id
    all_runs = client.search_runs(experiment_ids=experiment_id)
    for mlflow_run in all_runs:
        client.delete_run(mlflow_run.info.run_id)
    # Updates Version tag
    new_tag = String(value=f"{current_tag + 1}")
    new_tag.save(name="version-counter", overwrite=True)


@task
def save_preprocessors(dv, scaler):
    with open("preprocessors.pkl", "wb") as file:
        pickle.dump((dv, scaler), file)
    return 0


@flow
def main():
    """
    Main function. Reads data, preprocesses it and gives out the best models"""
    [df_train, df_val] = read_data()
    [scaler, dv, X_train, y_train] = preprocess_data(df_train)
    [X_val, y_val] = preprocess_data(df_val, dv, scaler, train=False)
    preprocessor_return_code = save_preprocessors(dv, scaler)
    hpo_return_code = hyperoptimizer(X_train, y_train, X_val, y_val)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.xgboost.autolog()
    run(10, X_train, y_train, X_val, y_val, hpo_return_code, preprocessor_return_code)
