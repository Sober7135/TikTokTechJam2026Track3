mod codex;
mod store;

use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, anyhow};
use axum::body::{Body, to_bytes};
use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE};
use axum::http::{Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use benchmark_contracts::{
    ApiError, ArtifactUploadResponse, EnqueueJobRequest, EnqueueJobResponse, JobKind,
    JobResultUpload, JobView, LeaseActionRequest, LeaseRequest, LeasedJob, VerificationSpec,
    VerificationTemplate,
};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tower_http::trace::TraceLayer;
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

use crate::store::Store;

const MAX_BODY_BYTES: usize = 1024 * 1024;
const MAX_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;
static ARTIFACT_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug)]
struct Config {
    bind: String,
    admin_token: String,
    worker_token: String,
    codex_enabled: bool,
    codex_command: String,
    analysis_root: PathBuf,
    artifact_root: PathBuf,
    artifact_base_url: String,
}

#[derive(Clone, Debug)]
struct AppState {
    store: Arc<Store>,
    config: Arc<Config>,
}

#[derive(Debug)]
struct HttpError {
    status: StatusCode,
    message: String,
}

impl HttpError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn internal(error: impl std::fmt::Display) -> Self {
        error!(%error, "control-plane request failed");
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal server error")
    }
}

impl IntoResponse for HttpError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ApiError {
                error: self.message,
            }),
        )
            .into_response()
    }
}

type HttpResult<T> = std::result::Result<T, HttpError>;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let state = state_from_env()?;
    let bind = state.config.bind.clone();
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .with_context(|| format!("failed to bind {bind}"))?;
    info!(%bind, "control plane listening");

    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("control-plane HTTP server failed")
}

fn state_from_env() -> Result<AppState> {
    let artifact_base_url = env_value("CONTROL_PLANE_ARTIFACT_BASE_URL", "http://127.0.0.1:8080")
        .trim_end_matches('/')
        .to_owned();
    validate_service_url(&artifact_base_url, "CONTROL_PLANE_ARTIFACT_BASE_URL")?;
    let config = Config {
        bind: env_value("CONTROL_PLANE_BIND", "127.0.0.1:8080"),
        admin_token: required_secret("CONTROL_PLANE_ADMIN_TOKEN")?,
        worker_token: required_secret("CONTROL_PLANE_WORKER_TOKEN")?,
        codex_enabled: env_value("CODEX_ANALYSIS_ENABLED", "false") == "true",
        codex_command: env_value("CODEX_COMMAND", "codex"),
        analysis_root: PathBuf::from(env_value("CODEX_ANALYSIS_ROOT", "analysis-workspaces")),
        artifact_root: PathBuf::from(env_value("CONTROL_PLANE_ARTIFACT_ROOT", "artifacts")),
        artifact_base_url,
    };
    let lease_seconds = env_value("CONTROL_PLANE_LEASE_SECONDS", "30")
        .parse::<u64>()
        .context("CONTROL_PLANE_LEASE_SECONDS must be an integer")?;
    let max_attempts = env_value("CONTROL_PLANE_MAX_ATTEMPTS", "2")
        .parse::<u32>()
        .context("CONTROL_PLANE_MAX_ATTEMPTS must be an integer")?;
    let state_path = env_value("CONTROL_PLANE_STATE", "control-plane-state.json");
    let store = Store::open(
        state_path,
        lease_seconds.saturating_mul(1_000),
        max_attempts,
    )?;
    Ok(AppState {
        store: Arc::new(store),
        config: Arc::new(config),
    })
}

fn router(state: AppState) -> Router {
    let admin_routes = Router::new()
        .route("/v1/jobs", post(enqueue_job))
        .route("/v1/jobs/{job_id}", get(get_job))
        .route("/v1/artifacts", post(upload_artifact))
        .route_layer(middleware::from_fn_with_state(state.clone(), require_admin));
    let worker_routes = Router::new()
        .route("/v1/workers/lease", post(lease_job))
        .route("/v1/jobs/{job_id}/heartbeat", post(heartbeat_job))
        .route("/v1/jobs/{job_id}/result", post(complete_job))
        .route("/v1/artifacts/{sha256}", get(download_artifact))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            require_worker,
        ));

    Router::new()
        .route("/healthz", get(health))
        .merge(admin_routes)
        .merge(worker_routes)
        .fallback(not_found)
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn not_found() -> HttpError {
    HttpError::new(StatusCode::NOT_FOUND, "route not found")
}

async fn enqueue_job(
    State(state): State<AppState>,
    payload: std::result::Result<Json<EnqueueJobRequest>, JsonRejection>,
) -> HttpResult<(StatusCode, Json<EnqueueJobResponse>)> {
    let request = decode_json(payload)?;
    let store = Arc::clone(&state.store);
    let response = store_call(move || store.enqueue(request))
        .await?
        .map_err(|error| HttpError::new(StatusCode::BAD_REQUEST, error.to_string()))?;
    let status = if response.deduplicated {
        StatusCode::OK
    } else {
        StatusCode::CREATED
    };
    Ok((status, Json(response)))
}

async fn upload_artifact(
    State(state): State<AppState>,
    body: Body,
) -> HttpResult<(StatusCode, Json<ArtifactUploadResponse>)> {
    let bytes = to_bytes(body, MAX_ARTIFACT_BYTES)
        .await
        .map_err(|_| HttpError::new(StatusCode::PAYLOAD_TOO_LARGE, "artifact exceeds 64 MiB"))?;
    if bytes.is_empty() {
        return Err(HttpError::new(
            StatusCode::BAD_REQUEST,
            "artifact must not be empty",
        ));
    }
    let artifact_root = state.config.artifact_root.clone();
    let sha256 = store_call(move || store_artifact(&artifact_root, &bytes))
        .await?
        .map_err(HttpError::internal)?;
    let url = format!("{}/v1/artifacts/{sha256}", state.config.artifact_base_url);
    Ok((
        StatusCode::CREATED,
        Json(ArtifactUploadResponse { url, sha256 }),
    ))
}

async fn download_artifact(
    State(state): State<AppState>,
    Path(sha256): Path<String>,
) -> HttpResult<Response> {
    if !valid_sha256(&sha256) {
        return Err(HttpError::new(StatusCode::BAD_REQUEST, "invalid SHA-256"));
    }
    let path = state.config.artifact_root.join(format!("{sha256}.tar.gz"));
    let bytes = store_call(move || {
        fs::read(&path).with_context(|| format!("failed to read artifact {}", path.display()))
    })
    .await?
    .map_err(|_| HttpError::new(StatusCode::NOT_FOUND, "artifact not found"))?;
    Ok(([(CONTENT_TYPE, "application/gzip")], bytes).into_response())
}

fn store_artifact(root: &PathBuf, bytes: &[u8]) -> Result<String> {
    let sha256 = format!("{:x}", Sha256::digest(bytes));
    fs::create_dir_all(root)
        .with_context(|| format!("failed to create artifact root {}", root.display()))?;
    let destination = root.join(format!("{sha256}.tar.gz"));
    if destination.exists() {
        return Ok(sha256);
    }
    let sequence = ARTIFACT_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temporary = root.join(format!(".{sha256}.{}.{}.tmp", std::process::id(), sequence));
    fs::write(&temporary, bytes)
        .with_context(|| format!("failed to write artifact {}", temporary.display()))?;
    fs::rename(&temporary, &destination)
        .with_context(|| format!("failed to publish artifact {}", destination.display()))?;
    Ok(sha256)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

async fn lease_job(
    State(state): State<AppState>,
    payload: std::result::Result<Json<LeaseRequest>, JsonRejection>,
) -> HttpResult<Response> {
    let request = decode_json(payload)?;
    let store = Arc::clone(&state.store);
    let lease = store_call(move || store.lease(&request.worker_id, &request.hardware_profile))
        .await?
        .map_err(|error| HttpError::new(StatusCode::CONFLICT, error.to_string()))?;
    Ok(match lease {
        Some(job) => Json::<LeasedJob>(job).into_response(),
        None => StatusCode::NO_CONTENT.into_response(),
    })
}

async fn get_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
) -> HttpResult<Json<JobView>> {
    let store = Arc::clone(&state.store);
    let job = store_call(move || store.get(&job_id))
        .await?
        .map_err(|error| HttpError::new(StatusCode::NOT_FOUND, error.to_string()))?;
    Ok(Json(job))
}

async fn heartbeat_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    payload: std::result::Result<Json<LeaseActionRequest>, JsonRejection>,
) -> HttpResult<Json<JobView>> {
    let request = decode_json(payload)?;
    let store = Arc::clone(&state.store);
    let job = store_call(move || store.heartbeat(&job_id, &request.lease_id))
        .await?
        .map_err(|error| HttpError::new(StatusCode::CONFLICT, error.to_string()))?;
    Ok(Json(job))
}

async fn complete_job(
    State(state): State<AppState>,
    Path(job_id): Path<String>,
    payload: std::result::Result<Json<JobResultUpload>, JsonRejection>,
) -> HttpResult<Json<JobView>> {
    let upload = decode_json(payload)?;
    let store = Arc::clone(&state.store);
    let (job, newly_completed) = store_call(move || store.complete(&job_id, upload))
        .await?
        .map_err(|error| HttpError::new(StatusCode::CONFLICT, error.to_string()))?;
    if state.config.codex_enabled && newly_completed && job.result.is_some() {
        match job.kind {
            JobKind::Benchmark => spawn_initial_analysis(state, job.clone()),
            JobKind::Verification => spawn_verification_analysis(state, job.clone()),
        }
    }
    Ok(Json(job))
}

fn decode_json<T: DeserializeOwned>(
    payload: std::result::Result<Json<T>, JsonRejection>,
) -> HttpResult<T> {
    payload
        .map(|Json(value)| value)
        .map_err(|error| HttpError::new(error.status(), error.body_text()))
}

async fn require_admin(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> HttpResult<Response> {
    authorize(&request, &state.config.admin_token)?;
    Ok(next.run(request).await)
}

async fn require_worker(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> HttpResult<Response> {
    authorize(&request, &state.config.worker_token)?;
    Ok(next.run(request).await)
}

fn authorize(request: &Request<Body>, expected: &str) -> HttpResult<()> {
    let supplied = request
        .headers()
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .unwrap_or_default();
    if constant_time_eq(supplied.as_bytes(), expected.as_bytes()) {
        Ok(())
    } else {
        Err(HttpError::new(StatusCode::UNAUTHORIZED, "unauthorized"))
    }
}

async fn store_call<T, F>(operation: F) -> HttpResult<Result<T>>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(HttpError::internal)
}

fn spawn_initial_analysis(state: AppState, job: JobView) {
    tokio::task::spawn_blocking(move || {
        match codex::analyze(
            &state.config.codex_command,
            &state.config.analysis_root,
            &job,
        ) {
            Ok(mut analysis) => {
                if job.kind == JobKind::Benchmark {
                    match enqueue_requested_verification(&state.store, &job, &analysis) {
                        Ok(Some(child_id)) => {
                            if let Some(object) = analysis.as_object_mut() {
                                object.insert(
                                    "verification_job_id".to_owned(),
                                    Value::String(child_id),
                                );
                            }
                        }
                        Ok(None) => {}
                        Err(error) => {
                            error!(job_id = %job.job_id, %error, "Codex verification request was not enqueued")
                        }
                    }
                }
                if let Err(error) = state.store.set_analysis(&job.job_id, analysis) {
                    error!(job_id = %job.job_id, %error, "failed to store Codex analysis");
                }
            }
            Err(error) => {
                error!(job_id = %job.job_id, %error, "Codex analysis failed");
            }
        }
    });
}

fn spawn_verification_analysis(state: AppState, verification: JobView) {
    tokio::task::spawn_blocking(move || {
        let Some(parent_id) = verification.request.parent_job_id.as_deref() else {
            error!(job_id = %verification.job_id, "verification job omitted parent_job_id");
            return;
        };
        let parent = match state.store.get(parent_id) {
            Ok(parent) => parent,
            Err(error) => {
                error!(job_id = %verification.job_id, %error, "failed to load verification parent");
                return;
            }
        };
        match codex::analyze_verification(
            &state.config.codex_command,
            &state.config.analysis_root,
            &parent,
            &verification,
        ) {
            Ok(mut analysis) => {
                if let Some(object) = analysis.as_object_mut() {
                    object.insert(
                        "verification_job_id".to_owned(),
                        Value::String(verification.job_id.clone()),
                    );
                    object.insert("verification_completed".to_owned(), Value::Bool(true));
                }
                if let Err(error) = state.store.set_analysis(&parent.job_id, analysis) {
                    error!(job_id = %parent.job_id, %error, "failed to store final Codex analysis");
                }
            }
            Err(error) => {
                error!(job_id = %verification.job_id, %error, "final Codex analysis failed");
            }
        }
    });
}

fn enqueue_requested_verification(
    store: &Store,
    parent: &JobView,
    analysis: &Value,
) -> Result<Option<String>> {
    let Some(request) = analysis
        .get("verification_request")
        .filter(|value| !value.is_null())
    else {
        return Ok(None);
    };
    let template = match request.get("template").and_then(Value::as_str) {
        Some("repeat_noisy_case") => VerificationTemplate::RepeatNoisyCase,
        Some("stricter_accuracy") => VerificationTemplate::StricterAccuracy,
        _ => {
            return Err(anyhow!(
                "Codex requested a non-allowlisted verification template"
            ));
        }
    };
    let reason = request
        .get("reason")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("verification request omitted reason"))?;
    let expected_evidence = request
        .get("expected_evidence")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("verification request omitted expected_evidence"))?;

    let mut child = parent.request.clone();
    child.parent_job_id = Some(parent.job_id.clone());
    child.verification = Some(VerificationSpec {
        template: template.clone(),
        reason: reason.to_owned(),
        expected_evidence: expected_evidence.to_owned(),
    });
    match template {
        VerificationTemplate::RepeatNoisyCase => {
            child.options.repeats = child.options.repeats.saturating_mul(2).min(10_000);
            child.options.benchmark_rounds =
                child.options.benchmark_rounds.saturating_mul(2).min(100);
        }
        VerificationTemplate::StricterAccuracy => {
            child.options.rtol *= 0.5;
            child.options.atol *= 0.5;
            child.options.accuracy_trials =
                child.options.accuracy_trials.saturating_mul(2).min(100);
        }
    }
    let response = store.enqueue(child)?;
    Ok(Some(response.job_id))
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0_u8, |diff, (a, b)| diff | (a ^ b))
        == 0
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            error!(%error, "failed to install Ctrl-C handler");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(error) => error!(%error, "failed to install SIGTERM handler"),
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
    info!("shutdown signal received");
}

fn env_value(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_owned())
}

fn required_secret(name: &str) -> Result<String> {
    let value = env::var(name).with_context(|| format!("{name} must be set"))?;
    if value.len() < 16 {
        return Err(anyhow!("{name} must contain at least 16 characters"));
    }
    Ok(value)
}

fn validate_service_url(value: &str, name: &str) -> Result<()> {
    if !(value.starts_with("https://")
        || value.starts_with("http://127.0.0.1:")
        || value.starts_with("http://localhost:"))
    {
        return Err(anyhow!(
            "{name} must use HTTPS, except for loopback development"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path as FilePath, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use axum::body::{Body, to_bytes};
    use axum::http::{Request, header};
    use benchmark_contracts::{API_SCHEMA_VERSION, BenchmarkOptions, SourceBundle};
    use tower::ServiceExt;

    use super::*;

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(1);
    const ADMIN_TOKEN: &str = "admin-token-123456789";
    const WORKER_TOKEN: &str = "worker-token-12345678";

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let path = env::temp_dir().join(format!(
                "techjam-http-test-{}-{}",
                std::process::id(),
                TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).expect("create test directory");
            Self(path)
        }

        fn path(&self) -> &FilePath {
            &self.0
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).expect("remove test directory");
        }
    }

    fn test_state() -> (TestDirectory, AppState) {
        let directory = TestDirectory::new();
        let store = Store::open(directory.path().join("state.json"), 60_000, 2).expect("store");
        let config = Config {
            bind: "127.0.0.1:0".to_owned(),
            admin_token: ADMIN_TOKEN.to_owned(),
            worker_token: WORKER_TOKEN.to_owned(),
            codex_enabled: false,
            codex_command: "codex".to_owned(),
            analysis_root: directory.path().join("analysis"),
            artifact_root: directory.path().join("artifacts"),
            artifact_base_url: "http://127.0.0.1:8080".to_owned(),
        };
        (
            directory,
            AppState {
                store: Arc::new(store),
                config: Arc::new(config),
            },
        )
    }

    fn valid_job() -> EnqueueJobRequest {
        EnqueueJobRequest {
            schema_version: API_SCHEMA_VERSION,
            repository: "owner/repo".to_owned(),
            pull_request: 42,
            head_sha: "a".repeat(40),
            benchmark_version: "v1".to_owned(),
            hardware_profile: "rtx-4070".to_owned(),
            benchmark_image: format!("bench@sha256:{}", "b".repeat(64)),
            source: SourceBundle {
                url: "https://example.invalid/source.tar.gz".to_owned(),
                sha256: "c".repeat(64),
                strip_components: 0,
            },
            pull_request_context: None,
            options: BenchmarkOptions::default(),
            parent_job_id: None,
            verification: None,
        }
    }

    fn post(path: &str, token: Option<&str>, body: Body) -> Request<Body> {
        let mut builder = Request::post(path).header(header::CONTENT_TYPE, "application/json");
        if let Some(token) = token {
            builder = builder.header(header::AUTHORIZATION, format!("Bearer {token}"));
        }
        builder.body(body).expect("request")
    }

    #[tokio::test]
    async fn health_is_public() {
        let (_directory, state) = test_state();
        let response = router(state)
            .oneshot(
                Request::get("/healthz")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn enqueue_requires_admin_token() {
        let (_directory, state) = test_state();
        let response = router(state)
            .oneshot(post("/v1/jobs", None, Body::from("{}")))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn worker_routes_require_worker_token() {
        let (_directory, state) = test_state();
        let response = router(state)
            .oneshot(post(
                "/v1/workers/lease",
                Some(ADMIN_TOKEN),
                Body::from("{}"),
            ))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn unknown_routes_return_json_error() {
        let (_directory, state) = test_state();
        let response = router(state)
            .oneshot(
                Request::get("/unknown")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = to_bytes(response.into_body(), MAX_BODY_BYTES)
            .await
            .expect("response body");
        let error: ApiError = serde_json::from_slice(&body).expect("JSON error response");
        assert_eq!(error.error, "route not found");
    }

    #[tokio::test]
    async fn malformed_json_is_a_bad_request() {
        let (_directory, state) = test_state();
        let response = router(state)
            .oneshot(post(
                "/v1/jobs",
                Some(ADMIN_TOKEN),
                Body::from("{not-json}"),
            ))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn request_body_is_bounded() {
        let (_directory, state) = test_state();
        let response = router(state)
            .oneshot(post(
                "/v1/jobs",
                Some(ADMIN_TOKEN),
                Body::from(vec![b' '; MAX_BODY_BYTES + 1]),
            ))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn enqueue_reports_created_then_deduplicated() {
        let (_directory, state) = test_state();
        let app = router(state);
        let body = serde_json::to_vec(&valid_job()).expect("serialize job");
        let first = app
            .clone()
            .oneshot(post(
                "/v1/jobs",
                Some(ADMIN_TOKEN),
                Body::from(body.clone()),
            ))
            .await
            .expect("first response");
        assert_eq!(first.status(), StatusCode::CREATED);

        let second = app
            .oneshot(post("/v1/jobs", Some(ADMIN_TOKEN), Body::from(body)))
            .await
            .expect("second response");
        assert_eq!(second.status(), StatusCode::OK);
        let body = to_bytes(second.into_body(), MAX_BODY_BYTES)
            .await
            .expect("response body");
        let response: EnqueueJobResponse = serde_json::from_slice(&body).expect("JSON response");
        assert!(response.deduplicated);
    }

    #[tokio::test]
    async fn artifact_upload_and_worker_download_are_authenticated() {
        let (_directory, state) = test_state();
        let app = router(state);
        let upload = app
            .clone()
            .oneshot(
                Request::post("/v1/artifacts")
                    .header(header::AUTHORIZATION, format!("Bearer {ADMIN_TOKEN}"))
                    .header(header::CONTENT_TYPE, "application/gzip")
                    .body(Body::from("archive-bytes"))
                    .expect("upload request"),
            )
            .await
            .expect("upload response");
        assert_eq!(upload.status(), StatusCode::CREATED);
        let body = to_bytes(upload.into_body(), MAX_BODY_BYTES)
            .await
            .expect("upload body");
        let artifact: ArtifactUploadResponse =
            serde_json::from_slice(&body).expect("artifact response");

        let unauthorized = app
            .clone()
            .oneshot(
                Request::get(format!("/v1/artifacts/{}", artifact.sha256))
                    .body(Body::empty())
                    .expect("unauthorized request"),
            )
            .await
            .expect("unauthorized response");
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);

        let download = app
            .oneshot(
                Request::get(format!("/v1/artifacts/{}", artifact.sha256))
                    .header(header::AUTHORIZATION, format!("Bearer {WORKER_TOKEN}"))
                    .body(Body::empty())
                    .expect("download request"),
            )
            .await
            .expect("download response");
        assert_eq!(download.status(), StatusCode::OK);
        let body = to_bytes(download.into_body(), MAX_ARTIFACT_BYTES)
            .await
            .expect("download body");
        assert_eq!(&body[..], b"archive-bytes");
    }
}
