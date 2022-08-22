variable "bucket_name" {
  default = "model_bucket"
}

variable "service-account" {
  type = string
  description = "Service Account associated with the whole function service"
}
