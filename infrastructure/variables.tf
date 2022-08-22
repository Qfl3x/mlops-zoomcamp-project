variable "gcp_region" {
  description = "Region in GCP"
  default = "eu-west3"
}

variable "gcp_project_id" {
  description = "Project ID in GCP"
}

variable "gcp_credentials" {
  default = "./terraform-account.json"
}
variable "prefix" {
  description = "Prefix for the services"
  default = "terraform-mlops-project"
}

variable "credentials" {
  default = "./terraform-account.json"
}
variable "service-account" {
}

variable "model_bucket_name" {
  default = "churn-model-bucket"
}

variable "data_bucket_name" {
  default = "churn-model-data"
}
