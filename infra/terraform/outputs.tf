output "api_url" {
  value = google_cloud_run_v2_service.api.uri
}

output "share_spreadsheet_with" {
  description = "Share the spreadsheet with both addresses (api: Editor, trainer: Viewer)"
  value = {
    editor = google_service_account.api.email
    viewer = google_service_account.trainer.email
  }
}

output "github_variables" {
  description = "Repository variables for GitHub Actions"
  value = {
    GCP_PROJECT      = var.project_id
    GCP_REGION       = var.region
    GCP_WIF_PROVIDER = google_iam_workload_identity_pool_provider.github.name
    GCP_DEPLOYER     = google_service_account.deployer.email
  }
}
