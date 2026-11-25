variable "project_id" {
  type = string
}

variable "region" {
  type        = string
  default     = "us-east1"
  description = "us-east1, us-west1 or us-central1 for the Cloud Storage free tier"
  validation {
    condition     = contains(["us-east1", "us-west1", "us-central1"], var.region)
    error_message = "Use a region with the Cloud Storage free tier: us-east1, us-west1 or us-central1."
  }
}

variable "timezone" {
  type    = string
  default = "America/New_York"
}

variable "bucket_name" {
  type        = string
  description = "Globally unique name for the model bucket"
}

variable "create_firestore" {
  type        = bool
  default     = true
  description = "false if the project already has a (default) Firestore database in Native mode"
}

variable "spreadsheet_id" {
  type = string
}

variable "source_tab" {
  type    = string
  default = "Enriched_Data"
}

variable "output_tab" {
  type    = string
  default = "Predictions"
}

variable "github_repository" {
  type        = string
  description = "owner/name, with the exact capitalization shown on GitHub"
}

variable "human_invokers" {
  type        = list(string)
  default     = []
  description = "Who may call the API by hand, e.g. [\"user:you@example.com\"]"
}

variable "trainer_job_name" {
  type    = string
  default = "workout-trainer"
}

variable "training_check_cron" {
  type    = string
  default = "0 4 * * *"
}

variable "predictions_cron" {
  type    = string
  default = "0 6 * * *"
}

variable "scheduler_paused" {
  type        = bool
  default     = true
  description = "Start paused; set to false once a manual run has been checked"
}

variable "alerts_enabled" {
  type        = bool
  default     = false
  description = "Set to true after adding a version to the workout-smtp-password secret"
}

variable "alert_email" {
  type    = string
  default = ""
}

variable "smtp_user" {
  type    = string
  default = ""
}

variable "billing_account" {
  type        = string
  default     = ""
  description = "XXXXXX-XXXXXX-XXXXXX to create the alert budget; empty to skip"
}

variable "monthly_alert_usd" {
  type    = number
  default = 1
}

variable "extra_env" {
  type        = map(string)
  default     = {}
  description = "Any configs/pipeline.yaml override, e.g. { WML_TRAINING__PERFORMANCE__MAX_MAPE = \"4.0\" }"
}

variable "placeholder_service_image" {
  type    = string
  default = "us-docker.pkg.dev/cloudrun/container/hello"
}

variable "placeholder_job_image" {
  type    = string
  default = "us-docker.pkg.dev/cloudrun/container/job"
}
