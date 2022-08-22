variable "push_topic_name" {
  default = "backend_push_topic"
}

variable "pull_topic_name" {
  default = "backend_pull_topic"
}

variable "subscriber_name" {
  default = "subscriber"
}

variable "topic_name" {
  type = string
  default = "topic"
}

variable "service-account" {
  type = string
}
