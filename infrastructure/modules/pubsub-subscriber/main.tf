
resource "google_pubsub_subscription" "subscriber" {
  name  = var.subscriber_name
  topic = var.topic
}

resource "google_pubsub_subscription_iam_member" "member" {
  subscription = google_pubsub_subscription.subscriber.name
  role = "roles/pubsub.subscriber"
  member = "serviceAccount:${var.service-account}"
}
