
resource "google_pubsub_topic" "topic" {
  name = "${var.topic_name}"

  message_retention_duration = "86600s"
}

resource "google_pubsub_topic_iam_member" "member" {
  topic = google_pubsub_topic.topic.name
  role = "roles/pubsub.publisher"
  member = "serviceAccount:${var.service-account}"
}
# resource "google_pubsub_topic" "backend_pull_stream" {
#   name = "${var.prefix}-${var.pull_stream_name}"

#   message_retention_duration = "86600s"
# }
output "topic_name" {
  value = google_pubsub_topic.topic.name
}
