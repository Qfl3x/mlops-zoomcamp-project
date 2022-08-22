resource null_resource "zip_function_files" {
  provisioner "local-exec" {
    command = "touch function.zip && rm function.zip && zip -urj function.zip ../function/main.py ../function/requirements.txt ../function/project-host.json"
  }
}

resource "google_storage_bucket_object" "function_archive"{
  name = "function.zip"
  bucket = var.model_bucket_name
  source = "./function.zip"
  depends_on = [ null_resource.zip_function_files ]
}


resource "google_cloudfunctions_function" "function" {
  name = "${var.prefix}-churn-prediction"
  runtime = "python39"

  entry_point = "endpoint"

  source_archive_bucket = var.model_bucket_name
  source_archive_object = google_storage_bucket_object.function_archive.name

  event_trigger {
    event_type = "google.storage.object.finalize"
    resource = var.data_bucket_name
  }

  environment_variables = {
    PROJECT_ID = var.project_id
    MODEL_BUCKET = var.model_bucket_name
    DATA_BUCKET = var.data_bucket_name
  }
}
