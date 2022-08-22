resource "google_storage_bucket" "bucket" {
  name          = var.bucket_name
  location      = "EU"
  force_destroy = true
}

output "bucket_name" {
  value = google_storage_bucket.bucket.name
}

resource "google_storage_bucket_iam_member" "member" {
  bucket        = google_storage_bucket.bucket.name
  role          = "roles/storage.objectAdmin"
  member        = "serviceAccount:${var.service-account}"
}
