export $(cat .env)
export GOOGLE_APPLICATION_CREDENTIALS=./terraform-account.json
terraform plan -var="gcp_project_id="$PROJECT_ID"" -var="service-account="$GOOGLE_ACCOUNT_NAME""
