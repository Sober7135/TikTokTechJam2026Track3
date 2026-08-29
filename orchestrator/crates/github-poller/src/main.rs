use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use benchmark_contracts::{
    API_SCHEMA_VERSION, ArtifactUploadResponse, BenchmarkOptions, EnqueueJobRequest,
    EnqueueJobResponse, JobState, JobView, PullRequestContext, SourceBundle,
};
use reqwest::{Method, Response, StatusCode};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

const COMMENT_MARKER: &str = "<!-- techjam-gpu-benchmark -->";
const MAX_JSON_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
const MAX_SOURCE_BYTES: usize = 64 * 1024 * 1024;
const MAX_COMMENT_BYTES: usize = 60 * 1024;

#[derive(Clone, Debug)]
struct Config {
    github_token: String,
    github_repository: String,
    github_api_base: String,
    github_api_version: String,
    control_plane_url: String,
    control_plane_admin_token: String,
    benchmark_version: String,
    hardware_profile: String,
    benchmark_image: String,
    options: BenchmarkOptions,
    state_path: PathBuf,
    poll_interval: Duration,
    once: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct GitHubPull {
    number: u64,
    title: String,
    body: Option<String>,
    html_url: String,
    draft: bool,
    user: GitHubUser,
    head: GitHubHead,
    base: GitHubBase,
}

#[derive(Clone, Debug, Deserialize)]
struct GitHubUser {
    login: String,
}

#[derive(Clone, Debug, Deserialize)]
struct GitHubHead {
    sha: String,
}

#[derive(Clone, Debug, Deserialize)]
struct GitHubBase {
    #[serde(rename = "ref")]
    name: String,
}

#[derive(Clone, Debug, Deserialize)]
struct CommentResponse {
    id: u64,
}

#[derive(Clone, Debug, Deserialize)]
struct CommentSummary {
    id: u64,
    body: Option<String>,
    user: GitHubUser,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct PersistentState {
    schema_version: u32,
    pulls: BTreeMap<u64, PullState>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PullState {
    head_sha: String,
    job_id: String,
    comment_id: Option<u64>,
    comment_digest: Option<String>,
}

struct Poller {
    config: Config,
    http: reqwest::Client,
    state: PersistentState,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    let config = Config::from_env()?;
    let mut poller = Poller::new(config)?;
    loop {
        match poller.run_once().await {
            Ok(()) => {}
            Err(error) if poller.config.once => return Err(error),
            Err(error) => error!(%error, "GitHub poll failed"),
        }
        if poller.config.once {
            return Ok(());
        }
        tokio::time::sleep(poller.config.poll_interval).await;
    }
}

impl Config {
    fn from_env() -> Result<Self> {
        let github_api_base = env_value("GITHUB_API_BASE", "https://api.github.com")
            .trim_end_matches('/')
            .to_owned();
        validate_url(&github_api_base, "GITHUB_API_BASE")?;
        let control_plane_url = required("CONTROL_PLANE_URL")?
            .trim_end_matches('/')
            .to_owned();
        validate_url(&control_plane_url, "CONTROL_PLANE_URL")?;
        let github_repository = required("GITHUB_REPOSITORY")?;
        split_repository(&github_repository)?;
        let options = match env::var("GITHUB_POLLER_OPTIONS_PATH") {
            Ok(path) => serde_json::from_slice(
                &fs::read(&path).with_context(|| format!("failed to read {path}"))?,
            )
            .with_context(|| format!("failed to decode benchmark options from {path}"))?,
            Err(_) => BenchmarkOptions::default(),
        };
        Ok(Self {
            github_token: required_secret("GITHUB_TOKEN")?,
            github_repository,
            github_api_base,
            github_api_version: env_value("GITHUB_API_VERSION", "2026-03-10"),
            control_plane_url,
            control_plane_admin_token: required_secret("CONTROL_PLANE_ADMIN_TOKEN")?,
            benchmark_version: required("BENCHMARK_VERSION")?,
            hardware_profile: required("BENCHMARK_HARDWARE_PROFILE")?,
            benchmark_image: required("BENCHMARK_IMAGE")?,
            options,
            state_path: PathBuf::from(env_value("GITHUB_POLLER_STATE", "github-poller-state.json")),
            poll_interval: seconds("GITHUB_POLL_SECONDS", 30)?,
            once: env_value("GITHUB_POLLER_ONCE", "false") == "true",
        })
    }
}

impl Poller {
    fn new(config: Config) -> Result<Self> {
        let state = load_state(&config.state_path)?;
        let http = reqwest::Client::builder()
            .user_agent("techjam-benchmark-github-poller/0.1.0")
            .timeout(Duration::from_secs(120))
            .build()
            .context("failed to build HTTP client")?;
        Ok(Self {
            config,
            http,
            state,
        })
    }

    async fn run_once(&mut self) -> Result<()> {
        let comment_author = self.authenticated_user().await?.login;
        let pulls = self.list_pulls().await?;
        let mut first_error = None;
        for pull in pulls.into_iter().filter(|pull| !pull.draft) {
            if let Err(error) = self.reconcile_pull(&pull, &comment_author).await {
                error!(pull_request = pull.number, %error, "failed to reconcile pull request");
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    async fn reconcile_pull(&mut self, pull: &GitHubPull, comment_author: &str) -> Result<()> {
        let needs_job = self
            .state
            .pulls
            .get(&pull.number)
            .is_none_or(|state| state.head_sha != pull.head.sha);
        if needs_job {
            let archive = self.download_source(&pull.head.sha).await?;
            let artifact = self.upload_artifact(archive).await?;
            let request = self.enqueue_request(pull, artifact);
            request.validate().map_err(|message| anyhow!(message))?;
            let enqueued: EnqueueJobResponse = self
                .control_plane_json(Method::POST, "/v1/jobs", Some(&request))
                .await?;
            let comment_id = self
                .state
                .pulls
                .get(&pull.number)
                .and_then(|state| state.comment_id);
            self.state.pulls.insert(
                pull.number,
                PullState {
                    head_sha: pull.head.sha.clone(),
                    job_id: enqueued.job_id.clone(),
                    comment_id,
                    comment_digest: None,
                },
            );
            persist_state(&self.config.state_path, &self.state)?;
            info!(pull_request = pull.number, job_id = %enqueued.job_id, "benchmark job reconciled");
        }

        let job_id = self
            .state
            .pulls
            .get(&pull.number)
            .map(|state| state.job_id.clone())
            .ok_or_else(|| anyhow!("poller state omitted reconciled job"))?;
        let root: JobView = self
            .control_plane_json(Method::GET, &format!("/v1/jobs/{job_id}"), None::<&Value>)
            .await?;
        let verification = if let Some(child_id) = root
            .analysis
            .as_ref()
            .and_then(|value| value.get("verification_job_id"))
            .and_then(Value::as_str)
        {
            Some(
                self.control_plane_json(
                    Method::GET,
                    &format!("/v1/jobs/{child_id}"),
                    None::<&Value>,
                )
                .await?,
            )
        } else {
            None
        };
        let body = render_comment(pull, &root, verification.as_ref());
        let digest = sha256_hex(body.as_bytes());
        let current = self
            .state
            .pulls
            .get(&pull.number)
            .ok_or_else(|| anyhow!("poller state omitted pull request"))?;
        if current.comment_digest.as_deref() == Some(&digest) {
            return Ok(());
        }
        let comment_id = self
            .upsert_comment(pull.number, current.comment_id, comment_author, &body)
            .await?;
        let current = self
            .state
            .pulls
            .get_mut(&pull.number)
            .ok_or_else(|| anyhow!("poller state omitted pull request"))?;
        current.comment_id = Some(comment_id);
        current.comment_digest = Some(digest);
        persist_state(&self.config.state_path, &self.state)
    }

    fn enqueue_request(
        &self,
        pull: &GitHubPull,
        artifact: ArtifactUploadResponse,
    ) -> EnqueueJobRequest {
        EnqueueJobRequest {
            schema_version: API_SCHEMA_VERSION,
            repository: self.config.github_repository.clone(),
            pull_request: pull.number,
            head_sha: pull.head.sha.clone(),
            benchmark_version: self.config.benchmark_version.clone(),
            hardware_profile: self.config.hardware_profile.clone(),
            benchmark_image: self.config.benchmark_image.clone(),
            source: SourceBundle {
                url: artifact.url,
                sha256: artifact.sha256,
                strip_components: 1,
            },
            pull_request_context: Some(PullRequestContext {
                title: bounded_text(&pull.title, 512),
                body: bounded_text(pull.body.as_deref().unwrap_or_default(), 16 * 1024),
                author: bounded_text(&pull.user.login, 128),
                html_url: bounded_text(&pull.html_url, 2_048),
                base_ref: bounded_text(&pull.base.name, 256),
            }),
            options: self.config.options.clone(),
            parent_job_id: None,
            verification: None,
        }
    }

    async fn list_pulls(&self) -> Result<Vec<GitHubPull>> {
        let url = format!(
            "{}/repos/{}/pulls?state=open&sort=updated&direction=asc&per_page=100",
            self.config.github_api_base, self.config.github_repository
        );
        self.github_json(Method::GET, &url, None::<&Value>).await
    }

    async fn authenticated_user(&self) -> Result<GitHubUser> {
        let url = format!("{}/user", self.config.github_api_base);
        self.github_json(Method::GET, &url, None::<&Value>).await
    }

    async fn download_source(&self, head_sha: &str) -> Result<Vec<u8>> {
        if !valid_sha(head_sha) {
            bail!("GitHub returned an invalid pull-request head SHA");
        }
        let url = format!(
            "{}/repos/{}/tarball/{head_sha}",
            self.config.github_api_base, self.config.github_repository
        );
        let response = self
            .github_request(Method::GET, &url)
            .send()
            .await
            .context("failed to download GitHub source archive")?;
        read_success_bytes(response, "download source", MAX_SOURCE_BYTES).await
    }

    async fn upload_artifact(&self, archive: Vec<u8>) -> Result<ArtifactUploadResponse> {
        let url = format!("{}/v1/artifacts", self.config.control_plane_url);
        let response = self
            .http
            .post(url)
            .bearer_auth(&self.config.control_plane_admin_token)
            .header("Content-Type", "application/gzip")
            .body(archive)
            .send()
            .await
            .context("failed to upload source artifact")?;
        decode_json(response, "upload artifact").await
    }

    async fn upsert_comment(
        &self,
        pull_request: u64,
        stored_id: Option<u64>,
        comment_author: &str,
        body: &str,
    ) -> Result<u64> {
        if let Some(comment_id) = stored_id
            && let Some(id) = self.update_comment(comment_id, body).await?
        {
            return Ok(id);
        }
        if let Some(comment_id) = self.find_comment(pull_request, comment_author).await? {
            return self
                .update_comment(comment_id, body)
                .await?
                .ok_or_else(|| anyhow!("GitHub comment disappeared during update"));
        }
        let url = format!(
            "{}/repos/{}/issues/{pull_request}/comments",
            self.config.github_api_base, self.config.github_repository
        );
        let response: CommentResponse = self
            .github_json(Method::POST, &url, Some(&serde_json::json!({"body": body})))
            .await?;
        Ok(response.id)
    }

    async fn update_comment(&self, comment_id: u64, body: &str) -> Result<Option<u64>> {
        let url = format!(
            "{}/repos/{}/issues/comments/{comment_id}",
            self.config.github_api_base, self.config.github_repository
        );
        let response = self
            .github_request(Method::PATCH, &url)
            .json(&serde_json::json!({"body": body}))
            .send()
            .await
            .context("failed to update GitHub comment")?;
        if response.status() == StatusCode::NOT_FOUND {
            return Ok(None);
        }
        Ok(Some(
            decode_json::<CommentResponse>(response, "update comment")
                .await?
                .id,
        ))
    }

    async fn find_comment(&self, pull_request: u64, comment_author: &str) -> Result<Option<u64>> {
        let url = format!(
            "{}/repos/{}/issues/{pull_request}/comments?per_page=100&sort=created&direction=desc",
            self.config.github_api_base, self.config.github_repository
        );
        let comments: Vec<CommentSummary> =
            self.github_json(Method::GET, &url, None::<&Value>).await?;
        Ok(comments.into_iter().find_map(|comment| {
            (comment.user.login == comment_author
                && comment
                    .body
                    .as_deref()
                    .is_some_and(|body| body.contains(COMMENT_MARKER)))
            .then_some(comment.id)
        }))
    }

    async fn github_json<T: DeserializeOwned, B: Serialize + ?Sized>(
        &self,
        method: Method,
        url: &str,
        body: Option<&B>,
    ) -> Result<T> {
        let mut request = self.github_request(method, url);
        if let Some(body) = body {
            request = request.json(body);
        }
        let response = request.send().await.context("GitHub API request failed")?;
        decode_json(response, "GitHub API request").await
    }

    fn github_request(&self, method: Method, url: &str) -> reqwest::RequestBuilder {
        self.http
            .request(method, url)
            .bearer_auth(&self.config.github_token)
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", &self.config.github_api_version)
    }

    async fn control_plane_json<T: DeserializeOwned, B: Serialize + ?Sized>(
        &self,
        method: Method,
        path: &str,
        body: Option<&B>,
    ) -> Result<T> {
        let url = format!("{}{path}", self.config.control_plane_url);
        let mut request = self
            .http
            .request(method, url)
            .bearer_auth(&self.config.control_plane_admin_token);
        if let Some(body) = body {
            request = request.json(body);
        }
        let response = request
            .send()
            .await
            .context("control-plane request failed")?;
        decode_json(response, "control-plane request").await
    }
}

fn render_comment(pull: &GitHubPull, root: &JobView, verification: Option<&JobView>) -> String {
    let mut body = format!(
        "{COMMENT_MARKER}\n## GPU Benchmark\n\n**{}**\n\n| Field | Value |\n| --- | --- |\n| Status | `{}` |\n| Commit | `{}` |\n| Job | `{}` |\n| Benchmark | `{}` |\n| Hardware | `{}` |\n",
        markdown_text(&pull.title, 512),
        job_state_name(&root.state),
        shorten(&root.request.head_sha, 12),
        markdown_text(&root.job_id, 128),
        markdown_text(&root.request.benchmark_version, 128),
        markdown_text(&root.request.hardware_profile, 64),
    );
    if let Some(result) = &root.result {
        body.push_str("\n### Deterministic result\n\n");
        body.push_str(&format!(
            "- Conclusion: **{}**\n",
            if result.success { "passed" } else { "failed" }
        ));
        if let Some(passed) = result
            .benchmark_result
            .get("correctness_passed")
            .and_then(Value::as_bool)
        {
            body.push_str(&format!(
                "- Correctness: **{}**\n",
                if passed { "passed" } else { "failed" }
            ));
        }
        render_cases(&mut body, &result.benchmark_result);
    }
    if let Some(verification) = verification {
        body.push_str(&format!(
            "\n### Verification\n\n- `{}`: `{}`\n",
            markdown_text(&verification.job_id, 128),
            job_state_name(&verification.state)
        ));
    }
    if let Some(analysis) = &root.analysis {
        body.push_str("\n### AI-generated interpretation\n\n");
        if let Some(summary) = analysis.get("summary").and_then(Value::as_str) {
            body.push_str(&markdown_text(summary, 2_000));
            body.push('\n');
        }
        render_string_list(
            &mut body,
            "Likely bottlenecks",
            analysis.get("likely_bottlenecks"),
        );
        render_string_list(
            &mut body,
            "Recommendations",
            analysis.get("recommendations"),
        );
        if analysis
            .get("verification_completed")
            .and_then(Value::as_bool)
            == Some(true)
        {
            body.push_str("\nVerification evidence was included in this final review.\n");
        }
    }
    if root.state == JobState::Superseded {
        body.push_str("\n> Superseded by a newer pull-request commit.\n");
    }
    truncate_comment(body)
}

fn render_cases(body: &mut String, result: &Value) {
    let Some(cases) = result.get("cases").and_then(Value::as_array) else {
        return;
    };
    body.push_str("\n| Case | Status | Correct | Speedup |\n| ---: | --- | :---: | ---: |\n");
    for case in cases.iter().take(32) {
        let case_id = case.get("case_id").and_then(Value::as_u64).unwrap_or(0);
        let status = case
            .get("status")
            .and_then(Value::as_str)
            .map_or_else(|| "unknown".to_owned(), |value| markdown_text(value, 32));
        let correctness = match case.get("correctness_passed").and_then(Value::as_bool) {
            Some(true) => "yes",
            Some(false) => "no",
            None => "n/a",
        };
        let speedup = case
            .pointer("/performance/speedup")
            .and_then(Value::as_f64)
            .map_or_else(|| "n/a".to_owned(), |value| format!("{value:.3}x"));
        body.push_str(&format!(
            "| {case_id} | {status} | {correctness} | {speedup} |\n"
        ));
    }
}

fn render_string_list(body: &mut String, heading: &str, value: Option<&Value>) {
    let Some(items) = value.and_then(Value::as_array) else {
        return;
    };
    if items.is_empty() {
        return;
    }
    body.push_str(&format!("\n**{heading}**\n\n"));
    for item in items.iter().take(8).filter_map(Value::as_str) {
        body.push_str(&format!("- {}\n", markdown_text(item, 500)));
    }
}

fn markdown_text(value: &str, max_chars: usize) -> String {
    let sanitized = value.replace("<!--", "&lt;!--").replace("-->", "--&gt;");
    let mut output = String::new();
    for character in sanitized.chars().take(max_chars) {
        match character {
            '@' => output.push_str("@\u{200b}"),
            '\n' | '\r' | '\t' => output.push(' '),
            '\\' | '`' | '*' | '_' | '{' | '}' | '[' | ']' | '<' | '>' | '#' | '|' => {
                output.push('\\');
                output.push(character);
            }
            character if character.is_control() => {}
            character => output.push(character),
        }
    }
    output
}

fn truncate_comment(mut body: String) -> String {
    const NOTICE: &str = "\n\n_Output truncated by the GitHub poller._\n";
    if body.len() <= MAX_COMMENT_BYTES {
        return body;
    }
    let mut end = MAX_COMMENT_BYTES - NOTICE.len();
    while !body.is_char_boundary(end) {
        end -= 1;
    }
    body.truncate(end);
    body.push_str(NOTICE);
    body
}

fn job_state_name(state: &JobState) -> &'static str {
    match state {
        JobState::Queued => "queued",
        JobState::Leased => "leased",
        JobState::Running => "running",
        JobState::Succeeded => "succeeded",
        JobState::Failed => "failed",
        JobState::Cancelled => "cancelled",
        JobState::TimedOut => "timed_out",
        JobState::Superseded => "superseded",
    }
}

async fn decode_json<T: DeserializeOwned>(response: Response, operation: &str) -> Result<T> {
    let bytes = read_success_bytes(response, operation, MAX_JSON_RESPONSE_BYTES).await?;
    serde_json::from_slice(&bytes).with_context(|| format!("{operation} returned invalid JSON"))
}

async fn read_success_bytes(
    mut response: Response,
    operation: &str,
    limit: usize,
) -> Result<Vec<u8>> {
    let status = response.status();
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        bail!("{operation} response exceeds size limit");
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .with_context(|| format!("failed reading {operation} response"))?
    {
        if bytes.len().saturating_add(chunk.len()) > limit {
            bail!("{operation} response exceeds size limit");
        }
        bytes.extend_from_slice(&chunk);
    }
    if !status.is_success() {
        let message = String::from_utf8_lossy(&bytes);
        bail!(
            "{operation} failed with {status}: {}",
            message.chars().take(1_000).collect::<String>()
        );
    }
    Ok(bytes)
}

fn load_state(path: &Path) -> Result<PersistentState> {
    if !path.exists() {
        return Ok(PersistentState {
            schema_version: 1,
            pulls: BTreeMap::new(),
        });
    }
    let state: PersistentState = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
    )
    .with_context(|| format!("failed to decode {}", path.display()))?;
    if state.schema_version != 1 {
        bail!("unsupported poller state schema {}", state.schema_version);
    }
    Ok(state)
}

fn persist_state(path: &Path, state: &PersistentState) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let temporary = path.with_extension("json.tmp");
    let mut bytes = serde_json::to_vec_pretty(state)?;
    bytes.push(b'\n');
    fs::write(&temporary, bytes)
        .with_context(|| format!("failed to write {}", temporary.display()))?;
    fs::rename(&temporary, path).with_context(|| format!("failed to replace {}", path.display()))
}

fn bounded_text(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_owned();
    }
    let mut end = max_bytes;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_owned()
}

fn sha256_hex(value: &[u8]) -> String {
    format!("{:x}", Sha256::digest(value))
}

fn shorten(value: &str, length: usize) -> &str {
    value.get(..length).unwrap_or(value)
}

fn valid_sha(value: &str) -> bool {
    matches!(value.len(), 40 | 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn split_repository(repository: &str) -> Result<(&str, &str)> {
    repository
        .split_once('/')
        .filter(|(owner, repo)| {
            !owner.is_empty()
                && !repo.is_empty()
                && !repo.contains('/')
                && repository.bytes().all(|byte| {
                    byte.is_ascii_alphanumeric() || b"-_.".contains(&byte) || byte == b'/'
                })
        })
        .ok_or_else(|| anyhow!("GITHUB_REPOSITORY must have owner/name form"))
}

fn validate_url(value: &str, name: &str) -> Result<()> {
    if !(value.starts_with("https://")
        || value.starts_with("http://127.0.0.1:")
        || value.starts_with("http://localhost:"))
    {
        bail!("{name} must use HTTPS, except for loopback tests");
    }
    Ok(())
}

fn required(name: &str) -> Result<String> {
    let value = env::var(name).with_context(|| format!("{name} must be set"))?;
    if value.is_empty() {
        bail!("{name} must not be empty");
    }
    Ok(value)
}

fn required_secret(name: &str) -> Result<String> {
    let value = required(name)?;
    if value.len() < 16 {
        bail!("{name} must contain at least 16 characters");
    }
    Ok(value)
}

fn env_value(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_owned())
}

fn seconds(name: &str, default: u64) -> Result<Duration> {
    let value = env::var(name).map_or(Ok(default), |value| value.parse::<u64>())?;
    if value == 0 {
        bail!("{name} must be positive");
    }
    Ok(Duration::from_secs(value))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::body::Bytes;
    use axum::extract::{Path as AxumPath, State};
    use axum::http::{HeaderMap, StatusCode as AxumStatus};
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use benchmark_contracts::{JobKind, JobResultUpload};

    use super::*;

    fn pull() -> GitHubPull {
        GitHubPull {
            number: 7,
            title: "Optimize @team <!-- marker -->".to_owned(),
            body: Some("Fuses attention".to_owned()),
            html_url: "https://github.com/owner/repo/pull/7".to_owned(),
            draft: false,
            user: GitHubUser {
                login: "student".to_owned(),
            },
            head: GitHubHead {
                sha: "a".repeat(40),
            },
            base: GitHubBase {
                name: "main".to_owned(),
            },
        }
    }

    fn job() -> JobView {
        JobView {
            job_id: "job-1".to_owned(),
            kind: JobKind::Benchmark,
            state: JobState::Succeeded,
            request: EnqueueJobRequest {
                schema_version: API_SCHEMA_VERSION,
                repository: "owner/repo".to_owned(),
                pull_request: 7,
                head_sha: "a".repeat(40),
                benchmark_version: "v1".to_owned(),
                hardware_profile: "rtx-4070".to_owned(),
                benchmark_image: format!("bench@sha256:{}", "b".repeat(64)),
                source: SourceBundle {
                    url: "https://example.invalid/source.tar.gz".to_owned(),
                    sha256: "c".repeat(64),
                    strip_components: 1,
                },
                pull_request_context: None,
                options: BenchmarkOptions::default(),
                parent_job_id: None,
                verification: None,
            },
            attempts: 1,
            created_at_ms: 1,
            updated_at_ms: 2,
            superseded_by: None,
            result: Some(JobResultUpload {
                lease_id: "lease-1".to_owned(),
                success: true,
                failure_category: None,
                benchmark_result: serde_json::json!({
                    "correctness_passed": true,
                    "cases": [{
                        "case_id": 2,
                        "status": "succeeded",
                        "correctness_passed": true,
                        "performance": {"speedup": 1.25}
                    }]
                }),
                log_excerpt: String::new(),
            }),
            analysis: Some(serde_json::json!({
                "summary": "Good result @team",
                "likely_bottlenecks": ["launch overhead"],
                "recommendations": ["profile kernels"],
                "confidence": "medium",
                "verification_request": null
            })),
        }
    }

    #[test]
    fn comment_contains_result_and_sanitizes_untrusted_text() {
        let body = render_comment(&pull(), &job(), None);
        assert!(body.starts_with(COMMENT_MARKER));
        assert!(body.contains("| 2 | succeeded | yes | 1.250x |"));
        assert!(body.contains("AI-generated interpretation"));
        assert!(body.contains("@\u{200b}team"));
        assert!(!body.contains("<!-- marker -->"));
    }

    #[test]
    fn bounded_text_preserves_utf8_boundaries() {
        assert_eq!(bounded_text("a界b", 2), "a");
        assert_eq!(bounded_text("a界b", 4), "a界");
    }

    #[derive(Clone)]
    struct MockState {
        base_url: String,
        artifact_uploads: Arc<AtomicUsize>,
        enqueues: Arc<AtomicUsize>,
        comments: Arc<AtomicUsize>,
    }

    fn authorized(headers: &HeaderMap, token: &str) -> bool {
        headers
            .get("Authorization")
            .and_then(|value| value.to_str().ok())
            == Some(&format!("Bearer {token}"))
    }

    async fn mock_pulls(headers: HeaderMap) -> (AxumStatus, Json<Value>) {
        if !authorized(&headers, "github-token-123456789") {
            return (AxumStatus::UNAUTHORIZED, Json(serde_json::json!({})));
        }
        (
            AxumStatus::OK,
            Json(serde_json::json!([{
                "number": 7,
                "title": "Optimize attention",
                "body": "Fuse kernels",
                "html_url": "https://github.com/owner/repo/pull/7",
                "draft": false,
                "user": {"login": "student"},
                "head": {"sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
                "base": {"ref": "main"}
            }])),
        )
    }

    async fn mock_user(headers: HeaderMap) -> (AxumStatus, Json<Value>) {
        if !authorized(&headers, "github-token-123456789") {
            return (AxumStatus::UNAUTHORIZED, Json(serde_json::json!({})));
        }
        (
            AxumStatus::OK,
            Json(serde_json::json!({"login": "benchmark-owner"})),
        )
    }

    async fn mock_archive(headers: HeaderMap) -> (AxumStatus, &'static [u8]) {
        if !authorized(&headers, "github-token-123456789") {
            return (AxumStatus::UNAUTHORIZED, b"");
        }
        (AxumStatus::OK, b"github-tarball")
    }

    async fn mock_upload(
        State(state): State<MockState>,
        headers: HeaderMap,
        body: Bytes,
    ) -> (AxumStatus, Json<Value>) {
        if !authorized(&headers, "admin-token-123456789") || body != b"github-tarball"[..] {
            return (AxumStatus::UNAUTHORIZED, Json(serde_json::json!({})));
        }
        state.artifact_uploads.fetch_add(1, Ordering::Relaxed);
        (
            AxumStatus::CREATED,
            Json(serde_json::json!({
                "url": format!("{}/v1/artifacts/{}", state.base_url, "c".repeat(64)),
                "sha256": "c".repeat(64)
            })),
        )
    }

    async fn mock_enqueue(
        State(state): State<MockState>,
        headers: HeaderMap,
        Json(request): Json<EnqueueJobRequest>,
    ) -> (AxumStatus, Json<Value>) {
        if !authorized(&headers, "admin-token-123456789") {
            return (AxumStatus::UNAUTHORIZED, Json(serde_json::json!({})));
        }
        assert_eq!(request.source.strip_components, 1);
        assert_eq!(
            request
                .pull_request_context
                .as_ref()
                .map(|context| context.title.as_str()),
            Some("Optimize attention")
        );
        state.enqueues.fetch_add(1, Ordering::Relaxed);
        (
            AxumStatus::CREATED,
            Json(serde_json::json!({
                "job_id": "job-1",
                "state": "queued",
                "deduplicated": false
            })),
        )
    }

    async fn mock_job(
        headers: HeaderMap,
        AxumPath(_job_id): AxumPath<String>,
    ) -> (AxumStatus, Json<Value>) {
        if !authorized(&headers, "admin-token-123456789") {
            return (AxumStatus::UNAUTHORIZED, Json(serde_json::json!({})));
        }
        (
            AxumStatus::OK,
            Json(serde_json::to_value(job()).expect("serialize job")),
        )
    }

    async fn mock_list_comments(headers: HeaderMap) -> (AxumStatus, Json<Value>) {
        if !authorized(&headers, "github-token-123456789") {
            return (AxumStatus::UNAUTHORIZED, Json(serde_json::json!({})));
        }
        (AxumStatus::OK, Json(serde_json::json!([])))
    }

    async fn mock_create_comment(
        State(state): State<MockState>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> (AxumStatus, Json<Value>) {
        if !authorized(&headers, "github-token-123456789")
            || !body
                .get("body")
                .and_then(Value::as_str)
                .is_some_and(|body| body.contains(COMMENT_MARKER))
        {
            return (AxumStatus::UNAUTHORIZED, Json(serde_json::json!({})));
        }
        state.comments.fetch_add(1, Ordering::Relaxed);
        (AxumStatus::CREATED, Json(serde_json::json!({"id": 42})))
    }

    #[tokio::test]
    async fn poller_reconciles_once_and_persists_deduplication_state() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock server");
        let base_url = format!("http://{}", listener.local_addr().expect("mock address"));
        let mock = MockState {
            base_url: base_url.clone(),
            artifact_uploads: Arc::new(AtomicUsize::new(0)),
            enqueues: Arc::new(AtomicUsize::new(0)),
            comments: Arc::new(AtomicUsize::new(0)),
        };
        let app = Router::new()
            .route("/user", get(mock_user))
            .route("/repos/owner/repo/pulls", get(mock_pulls))
            .route("/repos/owner/repo/tarball/{sha}", get(mock_archive))
            .route("/v1/artifacts", post(mock_upload))
            .route("/v1/jobs", post(mock_enqueue))
            .route("/v1/jobs/{job_id}", get(mock_job))
            .route(
                "/repos/owner/repo/issues/7/comments",
                get(mock_list_comments).post(mock_create_comment),
            )
            .with_state(mock.clone());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("mock server");
        });
        let state_path = std::env::temp_dir().join(format!(
            "techjam-poller-test-{}-{}.json",
            std::process::id(),
            sha256_hex(base_url.as_bytes())
        ));
        let config = Config {
            github_token: "github-token-123456789".to_owned(),
            github_repository: "owner/repo".to_owned(),
            github_api_base: base_url.clone(),
            github_api_version: "2026-03-10".to_owned(),
            control_plane_url: base_url,
            control_plane_admin_token: "admin-token-123456789".to_owned(),
            benchmark_version: "v1".to_owned(),
            hardware_profile: "rtx-4070".to_owned(),
            benchmark_image: format!("bench@sha256:{}", "b".repeat(64)),
            options: BenchmarkOptions::default(),
            state_path: state_path.clone(),
            poll_interval: Duration::from_secs(30),
            once: true,
        };
        let mut poller = Poller::new(config).expect("poller");
        poller.run_once().await.expect("first poll");
        poller.run_once().await.expect("second poll");
        assert_eq!(mock.artifact_uploads.load(Ordering::Relaxed), 1);
        assert_eq!(mock.enqueues.load(Ordering::Relaxed), 1);
        assert_eq!(mock.comments.load(Ordering::Relaxed), 1);
        assert_eq!(load_state(&state_path).expect("state").pulls.len(), 1);

        server.abort();
        let _ = fs::remove_file(state_path);
    }
}
