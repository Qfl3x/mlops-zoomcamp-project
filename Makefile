##
# MLOps-Zoomcamp Churn Prediction
#
# @file
# @version 0.1

data:
	pipenv run python ./clean-data.py
terraform:
	cp baseenv infrastructure/.env
	cd infrastructure && terraform init
	cd infrastructure && ./terraform-apply.sh
dotenv:
	./initiate_dotenvs.sh
build: data terraform dotenv
	./upload-files.sh
plan:
	cp baseenv infrastructure/.env
	cd infrastructure && terraform init
	cd infrastructure && ./terraform-plan.sh
# end
