export $(cat .env)
pipenv run prefect deployment build train.py:main --name main -sb $1
pipenv run prefect deployment apply main-deployment.yaml
