export $(cat .env)
pipenv run prefect deployment build batch_monitoring.py:batch_analyze --name analyze -sb $1
pipenv run prefect deployment apply batch_analyze-deployment.yaml
