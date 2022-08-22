terraform {
  required_version = ">=1.0"
  backend "gcs" {
    bucket = "mlops-project-terraform-state"
    prefix = "tfstate-stg"
    credentials = "terraform-account.json"
  }
  required_providers {
    google = {
      source = "hashicorp/google"
  }
  }
}

provider "google" {
  project = var.gcp_project_id
  region  = "europe-west3"
  zone    = "europe-west3-c"
  #credentials = var.gcp_credentials
}
module "model_bucket" {
  source = "./modules/gs"
  bucket_name = "${var.prefix}-${var.model_bucket_name}"
  service-account = var.service-account
}

module "data_bucket" {
  source = "./modules/gs"
  bucket_name = "${var.prefix}-${var.data_bucket_name}"
  service-account = var.service-account
}


# Write .env files
resource "local_file" "env_file" {

  content = <<EOT
PROJECT_ID=${var.gcp_project_id}
MODEL_BUCKET=${var.prefix}-${var.model_bucket_name}
DATA_BUCKET=${var.prefix}-${var.data_bucket_name}
EOT

  filename = "../.env-terraform"
}

resource null_resource "finalize_env_file" {
  provisioner "local-exec" {
    command = "cd ../ && cat baseenv .env-terraform > .env"
  }
  depends_on = [local_file.env_file]
}

resource null_resource "func_env_file" {
  provisioner "local-exec" {
    command = "cd ../ && cp .env-terraform function/.env"
  }
  depends_on = [local_file.env_file]
}
resource null_resource "copy_env_file" {
  provisioner "local-exec" {
    command = "cd ../ && ./initiate_dotenvs.sh"
  }
  depends_on = [null_resource.finalize_env_file]
}

module "function" {
  source = "./modules/function"
  prefix = var.prefix
  project_id = var.gcp_project_id
  model_bucket_name = module.model_bucket.bucket_name
  data_bucket_name = module.data_bucket.bucket_name
  depends_on = [ module.model_bucket, module.data_bucket, null_resource.func_env_file ]
}
