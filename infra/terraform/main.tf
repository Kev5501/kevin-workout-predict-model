terraform {
  required_version = ">= 1.6"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 6.0, < 8.0"
    }
  }
  backend "gcs" {
    prefix = "workout-ml" # bucket is passed at init: terraform init -backend-config="bucket=..."
  }
}

provider "google" {
  project               = var.project_id
  region                = var.region
  billing_project       = var.project_id
  user_project_override = true # needed by the billing budget API
}

locals {
  services = [
    "run.googleapis.com", "cloudscheduler.googleapis.com", "firestore.googleapis.com", "storage.googleapis.com",
    "secretmanager.googleapis.com", "iam.googleapis.com", "iamcredentials.googleapis.com", "sts.googleapis.com",
    "sheets.googleapis.com", "cloudresourcemanager.googleapis.com", "billingbudgets.googleapis.com",
  ]
  common_env = {
    WML_APP__BACKEND           = "gcp"
    WML_APP__TIMEZONE          = var.timezone
    WML_GCP__PROJECT           = var.project_id
    WML_GCP__REGION            = var.region
    WML_GCP__BUCKET            = google_storage_bucket.models.name
    WML_GCP__TRAINER_JOB       = var.trainer_job_name
    WML_SHEETS__SPREADSHEET_ID = var.spreadsheet_id
    WML_SHEETS__SOURCE_TAB     = var.source_tab
    WML_SHEETS__OUTPUT_TAB     = var.output_tab
  }
}

resource "google_project_service" "enabled" {
  for_each           = toset(local.services)
  service            = each.value
  disable_on_destroy = false
}

# ---------------------------------------------------------------- storage
resource "google_storage_bucket" "models" {
  name                        = var.bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false
  versioning {
    enabled = true
  }
  soft_delete_policy {
    retention_duration_seconds = 0 # versioning already protects files; soft delete is billed
  }
  depends_on = [google_project_service.enabled]
}

resource "google_firestore_database" "default" {
  count           = var.create_firestore ? 1 : 0
  name            = "(default)" # only the default database has a free quota
  location_id     = var.region
  type            = "FIRESTORE_NATIVE"
  deletion_policy = "ABANDON"
  depends_on      = [google_project_service.enabled]
}

resource "google_secret_manager_secret" "smtp_password" {
  secret_id = "workout-smtp-password" # the value is added by hand, so it never enters Terraform state
  replication {
    auto {}
  }
  depends_on = [google_project_service.enabled]
}

# ---------------------------------------------------------------- identities
resource "google_service_account" "api" {
  account_id   = "workout-api"
  display_name = "Workout ML API"
}

resource "google_service_account" "trainer" {
  account_id   = "workout-trainer"
  display_name = "Workout ML trainer"
}

resource "google_service_account" "scheduler" {
  account_id   = "workout-scheduler"
  display_name = "Workout ML scheduler"
}

resource "google_project_iam_member" "firestore" {
  for_each = {
    api     = google_service_account.api.email
    trainer = google_service_account.trainer.email
  }
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${each.value}"
}

resource "google_storage_bucket_iam_member" "read" {
  for_each = {
    api     = google_service_account.api.email
    trainer = google_service_account.trainer.email
  }
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${each.value}"
}

resource "google_storage_bucket_iam_member" "create" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectCreator" # can add versions, can't change or delete them
  member = "serviceAccount:${google_service_account.trainer.email}"
}

resource "google_secret_manager_secret_iam_member" "smtp" {
  secret_id = google_secret_manager_secret.smtp_password.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.trainer.email}"
}

# ---------------------------------------------------------------- compute
resource "google_cloud_run_v2_service" "api" {
  name                = "workout-api"
  location            = var.region
  deletion_protection = false

  template {
    service_account                  = google_service_account.api.email
    timeout                          = "300s"
    max_instance_request_concurrency = 4
    scaling {
      min_instance_count = 0
      max_instance_count = 1
    }
    containers {
      image = var.placeholder_service_image # replaced by GitHub Actions
      resources {
        limits            = { cpu = "1", memory = "1Gi" }
        cpu_idle          = true # billed only while handling requests
        startup_cpu_boost = true
      }
      dynamic "env" {
        for_each = merge(local.common_env, var.extra_env)
        content {
          name  = env.key
          value = env.value
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [template[0].containers[0].image, client, client_version]
  }
  depends_on = [google_project_service.enabled]
}

resource "google_cloud_run_v2_job" "trainer" {
  name                = var.trainer_job_name
  location            = var.region
  deletion_protection = false

  template {
    parallelism = 1
    task_count  = 1
    template {
      service_account = google_service_account.trainer.email
      timeout         = "1800s" # hard cap on the cost of one run
      max_retries     = 0
      containers {
        image = var.placeholder_job_image # replaced by GitHub Actions
        resources {
          limits = { cpu = "4", memory = "4Gi" }
        }
        dynamic "env" {
          for_each = merge(local.common_env, var.extra_env, var.alerts_enabled ? {
            WML_ALERTS__ENABLED   = "true"
            WML_ALERTS__TO        = var.alert_email
            WML_ALERTS__SMTP_USER = var.smtp_user
          } : {})
          content {
            name  = env.key
            value = env.value
          }
        }
        dynamic "env" {
          for_each = var.alerts_enabled ? [1] : []
          content {
            name = "WML_ALERTS__SMTP_PASSWORD"
            value_source {
              secret_key_ref {
                secret  = google_secret_manager_secret.smtp_password.secret_id
                version = "latest"
              }
            }
          }
        }
      }
    }
  }

  lifecycle {
    ignore_changes = [template[0].template[0].containers[0].image, client, client_version]
  }
  depends_on = [google_project_service.enabled, google_secret_manager_secret_iam_member.smtp]
}

resource "google_cloud_run_v2_job_iam_member" "api_starts_trainer" {
  name     = google_cloud_run_v2_job.trainer.name
  location = var.region
  role     = "roles/run.jobsExecutorWithOverrides" # the API passes --mode and --trigger
  member   = "serviceAccount:${google_service_account.api.email}"
}

resource "google_cloud_run_v2_service_iam_member" "invokers" {
  for_each = merge(
    { scheduler = "serviceAccount:${google_service_account.scheduler.email}" },
    { for m in var.human_invokers : m => m },
  )
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = each.value
}

# ---------------------------------------------------------------- schedule
resource "google_cloud_scheduler_job" "daily" {
  for_each = {
    training-check = { schedule = var.training_check_cron, path = "/v1/training/check" }
    predictions    = { schedule = var.predictions_cron, path = "/v1/predictions/run" }
  }
  name             = "workout-${each.key}"
  region           = var.region
  schedule         = each.value.schedule
  time_zone        = var.timezone
  paused           = var.scheduler_paused
  attempt_deadline = "320s"
  retry_config {
    retry_count = 1
  }
  http_target {
    http_method = "POST"
    uri         = "${google_cloud_run_v2_service.api.uri}${each.value.path}"
    oidc_token {
      service_account_email = google_service_account.scheduler.email
      audience              = google_cloud_run_v2_service.api.uri
    }
  }
  depends_on = [google_project_service.enabled]
}

# ---------------------------------------------------------------- alert budget (warns, doesn't cap)
resource "google_billing_budget" "alert" {
  count           = var.billing_account == "" ? 0 : 1
  billing_account = var.billing_account
  display_name    = "workout-ml alert"
  budget_filter {
    projects = ["projects/${data.google_project.this.number}"]
  }
  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.monthly_alert_usd)
    }
  }
  dynamic "threshold_rules" {
    for_each = [0.5, 1.0]
    content {
      threshold_percent = threshold_rules.value
    }
  }
  depends_on = [google_project_service.enabled]
}

data "google_project" "this" {
  project_id = var.project_id
}
