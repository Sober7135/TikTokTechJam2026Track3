use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use benchmark_contracts::{
    FailureCategory, JobResultUpload, LeaseActionRequest, LeaseRequest, LeasedJob,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
const RESULT_LIMIT_BYTES: u64 = 1024 * 1024;
const LOG_LIMIT_BYTES: usize = 64 * 1024;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug)]
struct Config {
    control_plane_url: String,
    artifact_base_url: String,
    worker_token: String,
    worker_id: String,
    hardware_profile: String,
    benchmark_image: String,
    gpu_device: String,
    poll_interval: Duration,
    heartbeat_interval: Duration,
    job_timeout: Duration,
    allow_file_source: bool,
    once: bool,
}

fn main() -> Result<()> {
    let config = Config::from_env()?;
    eprintln!(
        "worker {} polling {} for {}",
        config.worker_id, config.control_plane_url, config.hardware_profile
    );
    loop {
        match lease_job(&config) {
            Ok(Some(job)) => {
                eprintln!("leased {} ({:?})", job.job_id, job.kind);
                let upload = match execute_job(&config, &job) {
                    Ok(upload) => upload,
                    Err(error) => JobResultUpload {
                        lease_id: job.lease_id.clone(),
                        success: false,
                        failure_category: Some(FailureCategory::Infrastructure),
                        benchmark_result: json!({
                            "schema_version": 1,
                            "error": error.to_string()
                        }),
                        log_excerpt: format!("worker infrastructure error: {error}"),
                    },
                };
                upload_result(&config, &job.job_id, &upload)?;
                eprintln!("uploaded result for {}", job.job_id);
                if config.once {
                    return Ok(());
                }
            }
            Ok(None) => {
                if config.once {
                    return Ok(());
                }
                thread::sleep(config.poll_interval);
            }
            Err(error) => {
                eprintln!("lease request failed: {error}");
                if config.once {
                    return Err(error);
                }
                thread::sleep(config.poll_interval);
            }
        }
    }
}

impl Config {
    fn from_env() -> Result<Self> {
        let control_plane_url = required("CONTROL_PLANE_URL")?;
        if !(control_plane_url.starts_with("https://")
            || control_plane_url.starts_with("http://127.0.0.1:")
            || control_plane_url.starts_with("http://localhost:"))
        {
            bail!("CONTROL_PLANE_URL must use HTTPS, except for loopback development");
        }
        let control_plane_url = control_plane_url.trim_end_matches('/').to_owned();
        let artifact_base_url = env_value("WORKER_ARTIFACT_BASE_URL", &control_plane_url)
            .trim_end_matches('/')
            .to_owned();
        if !(artifact_base_url.starts_with("https://")
            || artifact_base_url.starts_with("http://127.0.0.1:")
            || artifact_base_url.starts_with("http://localhost:"))
        {
            bail!("WORKER_ARTIFACT_BASE_URL must use HTTPS, except for loopback development");
        }
        let worker_token = required("CONTROL_PLANE_WORKER_TOKEN")?;
        validate_curl_config_value(&worker_token, "CONTROL_PLANE_WORKER_TOKEN")?;
        Ok(Self {
            control_plane_url,
            artifact_base_url,
            worker_token,
            worker_id: required("WORKER_ID")?,
            hardware_profile: required("WORKER_HARDWARE_PROFILE")?,
            benchmark_image: required("WORKER_BENCHMARK_IMAGE")?,
            gpu_device: env_value("WORKER_GPU_DEVICE", "0"),
            poll_interval: seconds("WORKER_POLL_SECONDS", 5)?,
            heartbeat_interval: seconds("WORKER_HEARTBEAT_SECONDS", 10)?,
            job_timeout: seconds("WORKER_JOB_TIMEOUT_SECONDS", 3600)?,
            allow_file_source: env_value("WORKER_ALLOW_FILE_SOURCE", "false") == "true",
            once: env_value("WORKER_ONCE", "false") == "true",
        })
    }
}

fn lease_job(config: &Config) -> Result<Option<LeasedJob>> {
    let body = LeaseRequest {
        worker_id: config.worker_id.clone(),
        hardware_profile: config.hardware_profile.clone(),
    };
    let response = post_json(config, "/v1/workers/lease", &body)?;
    match response.status {
        200 => Ok(Some(response.json()?)),
        204 => Ok(None),
        status => bail!("control plane returned {status}: {}", response.text()),
    }
}

fn heartbeat(config: &Config, job: &LeasedJob) -> Result<()> {
    let response = post_json(
        config,
        &format!("/v1/jobs/{}/heartbeat", job.job_id),
        &LeaseActionRequest {
            lease_id: job.lease_id.clone(),
        },
    )?;
    if response.status != 200 {
        bail!(
            "heartbeat returned {}: {}",
            response.status,
            response.text()
        );
    }
    Ok(())
}

fn upload_result(config: &Config, job_id: &str, upload: &JobResultUpload) -> Result<()> {
    upload.validate().map_err(|message| anyhow!(message))?;
    let response = post_json(config, &format!("/v1/jobs/{job_id}/result"), upload)?;
    if response.status != 200 {
        bail!(
            "result upload returned {}: {}",
            response.status,
            response.text()
        );
    }
    Ok(())
}

fn execute_job(config: &Config, job: &LeasedJob) -> Result<JobResultUpload> {
    if job.request.hardware_profile != config.hardware_profile {
        bail!("job hardware profile does not match worker");
    }
    if job.request.benchmark_image != config.benchmark_image {
        bail!("job image does not exactly match WORKER_BENCHMARK_IMAGE");
    }
    heartbeat(config, job)?;

    let work = TempDir::new("techjam-job")?;
    let archive = work.path().join("source.tar.gz");
    let source = work.path().join("source");
    let output = work.path().join("output");
    fs::create_dir(&source)?;
    fs::create_dir(&output)?;
    fetch_source(config, &job.request.source.url, &archive)?;
    verify_sha256(&archive, &job.request.source.sha256)?;
    extract_archive(&archive, &source, job.request.source.strip_components)?;

    let log_path = work.path().join("benchmark.log");
    let log = File::create(&log_path)?;
    let log_error = log.try_clone()?;
    let container_name = format!("techjam-{}-{}", config.worker_id, job.job_id);
    let source_mount = format!("{}:/workspace:ro", source.canonicalize()?.display());
    let output_mount = format!("{}:/output:rw", output.canonicalize()?.display());
    let options = &job.request.options;
    let case = &options.case;

    let mut command = Command::new("docker");
    command.args([
        "run",
        "--rm",
        "--name",
        &container_name,
        "--gpus",
        &format!("device={}", config.gpu_device),
        "--network",
        "none",
        "--read-only",
        "--pids-limit",
        "512",
        "--memory",
        "16g",
        "--shm-size",
        "1g",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=4g",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--volume",
        &source_mount,
        "--volume",
        &output_mount,
        "--workdir",
        "/workspace",
        &config.benchmark_image,
        "python",
        "torch_transformer_benchmark.py",
        "--device",
        "cuda:0",
        "--dtype",
        &options.dtype,
        "--batch-size",
        &case.batch_size.to_string(),
        "--seq-len",
        &case.seq_len.to_string(),
        "--d-model",
        &case.d_model.to_string(),
        "--heads",
        &case.heads.to_string(),
        "--ffn-dim",
        &case.ffn_dim.to_string(),
        "--layers",
        &case.layers.to_string(),
        "--padding-ratio",
        &case.padding_ratio.to_string(),
        "--accuracy-trials",
        &options.accuracy_trials.to_string(),
        "--rtol",
        &options.rtol.to_string(),
        "--atol",
        &options.atol.to_string(),
        "--warmup",
        &options.warmup.to_string(),
        "--repeats",
        &options.repeats.to_string(),
        "--benchmark-rounds",
        &options.benchmark_rounds.to_string(),
        "--seed",
        &options.seed.to_string(),
        "--json-output",
        "/output/result.json",
    ]);
    if case.causal {
        command.arg("--causal");
    }
    command
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(log_error));
    let mut child = command
        .spawn()
        .context("failed to start Docker benchmark")?;

    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if started.elapsed() >= config.job_timeout {
            kill_container(&container_name);
            let _ = child.kill();
            let _ = child.wait();
            return Ok(failure_upload(
                job,
                FailureCategory::Timeout,
                json!({"schema_version": 1, "error": "benchmark timed out"}),
                &log_path,
            ));
        }
        thread::sleep(config.heartbeat_interval);
        if let Err(error) = heartbeat(config, job) {
            kill_container(&container_name);
            let _ = child.kill();
            let _ = child.wait();
            bail!("lost lease while benchmark was running: {error}");
        }
    };
    Ok(finish_upload(
        job,
        status,
        &output.join("result.json"),
        &log_path,
    ))
}

fn finish_upload(
    job: &LeasedJob,
    status: ExitStatus,
    result_path: &Path,
    log_path: &Path,
) -> JobResultUpload {
    let result = read_result(result_path).unwrap_or_else(|error| {
        json!({"schema_version": 1, "error": format!("missing or invalid result JSON: {error}")})
    });
    if status.success() {
        JobResultUpload {
            lease_id: job.lease_id.clone(),
            success: true,
            failure_category: None,
            benchmark_result: result,
            log_excerpt: read_log(log_path),
        }
    } else {
        let category = if status.code() == Some(2) {
            FailureCategory::Correctness
        } else {
            FailureCategory::Build
        };
        failure_upload(job, category, result, log_path)
    }
}

fn failure_upload(
    job: &LeasedJob,
    category: FailureCategory,
    result: Value,
    log_path: &Path,
) -> JobResultUpload {
    JobResultUpload {
        lease_id: job.lease_id.clone(),
        success: false,
        failure_category: Some(category),
        benchmark_result: result,
        log_excerpt: read_log(log_path),
    }
}

fn fetch_source(config: &Config, url: &str, destination: &Path) -> Result<()> {
    if let Some(path) = url.strip_prefix("file://") {
        if !config.allow_file_source {
            bail!("file:// source is disabled");
        }
        fs::copy(path, destination)
            .with_context(|| format!("failed to copy source bundle {path}"))?;
        return Ok(());
    }
    let is_control_plane_artifact =
        url.starts_with(&format!("{}/v1/artifacts/", config.artifact_base_url));
    if !url.starts_with("https://") && !is_control_plane_artifact {
        bail!("source URL must use HTTPS or the configured control-plane artifact endpoint");
    }
    validate_curl_config_value(url, "source URL")?;
    let mut curl_config = format!("url = \"{url}\"\n");
    if is_control_plane_artifact {
        curl_config.push_str(&format!(
            "header = \"Authorization: Bearer {}\"\n",
            config.worker_token
        ));
    }
    let mut child = Command::new("curl")
        .args([
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--proto",
            if is_control_plane_artifact {
                "=http,https"
            } else {
                "=https"
            },
            "--config",
            "-",
            "--output",
        ])
        .arg(destination)
        .stdin(Stdio::piped())
        .spawn()
        .context("failed to start curl for source download")?;
    child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("curl stdin unavailable"))?
        .write_all(curl_config.as_bytes())?;
    let status = child.wait()?;
    if !status.success() {
        bail!("source download failed with {status}");
    }
    Ok(())
}

fn verify_sha256(path: &Path, expected: &str) -> Result<()> {
    let output = Command::new("sha256sum")
        .arg(path)
        .output()
        .context("failed to start sha256sum")?;
    if !output.status.success() {
        bail!("sha256sum failed");
    }
    let actual = String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();
    if actual != expected.to_ascii_lowercase() {
        bail!("source bundle SHA-256 mismatch");
    }
    Ok(())
}

fn extract_archive(archive: &Path, destination: &Path, strip_components: u8) -> Result<()> {
    let status = Command::new("tar")
        .args([
            "--extract",
            "--gzip",
            "--no-same-owner",
            "--no-same-permissions",
            "--strip-components",
            &strip_components.to_string(),
            "--file",
        ])
        .arg(archive)
        .arg("--directory")
        .arg(destination)
        .status()
        .context("failed to start tar")?;
    if !status.success() {
        bail!("source extraction failed with {status}");
    }
    Ok(())
}

fn read_result(path: &Path) -> Result<Value> {
    let metadata = fs::metadata(path).with_context(|| format!("missing {}", path.display()))?;
    if metadata.len() > RESULT_LIMIT_BYTES {
        bail!("result JSON exceeds 1 MiB");
    }
    let value: Value = serde_json::from_slice(&fs::read(path)?)?;
    if !value.is_object() {
        bail!("result JSON must be an object");
    }
    Ok(value)
}

fn read_log(path: &Path) -> String {
    let bytes = fs::read(path).unwrap_or_default();
    let bounded = &bytes[..bytes.len().min(LOG_LIMIT_BYTES)];
    String::from_utf8_lossy(bounded).into_owned()
}

fn kill_container(name: &str) {
    let _ = Command::new("docker")
        .args(["kill", name])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

struct HttpResponse {
    status: u16,
    body: Vec<u8>,
}

impl HttpResponse {
    fn json<T: DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_slice(&self.body).context("control plane returned invalid JSON")
    }

    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

fn post_json(config: &Config, path: &str, body: &impl Serialize) -> Result<HttpResponse> {
    let temp = TempDir::new("techjam-http")?;
    let body_path = temp.path().join("body.json");
    let mut body_file = File::create(&body_path).context("failed to create request body file")?;
    serde_json::to_writer(&mut body_file, body)?;
    body_file.flush()?;
    let url = format!("{}{path}", config.control_plane_url);
    validate_curl_config_value(&url, "request URL")?;
    let curl_config = format!(
        "url = \"{url}\"\nrequest = \"POST\"\nheader = \"Authorization: Bearer {}\"\nheader = \"Content-Type: application/json\"\n",
        config.worker_token
    );
    let mut child = Command::new("curl")
        .args([
            "--silent",
            "--show-error",
            "--max-time",
            "45",
            "--config",
            "-",
            "--data-binary",
        ])
        .arg(format!("@{}", body_path.display()))
        .args(["--write-out", "\n%{http_code}"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .context("failed to start curl")?;
    child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("curl stdin unavailable"))?
        .write_all(curl_config.as_bytes())?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        bail!("curl failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    let marker = output
        .stdout
        .iter()
        .rposition(|byte| *byte == b'\n')
        .ok_or_else(|| anyhow!("curl omitted HTTP status"))?;
    let status = std::str::from_utf8(&output.stdout[marker + 1..])?
        .trim()
        .parse::<u16>()?;
    Ok(HttpResponse {
        status,
        body: output.stdout[..marker].to_vec(),
    })
}

fn validate_curl_config_value(value: &str, name: &str) -> Result<()> {
    if value
        .bytes()
        .any(|byte| matches!(byte, b'\r' | b'\n' | b'"' | b'\\'))
    {
        bail!("{name} contains characters unsafe for curl configuration");
    }
    Ok(())
}

fn seconds(name: &str, default: u64) -> Result<Duration> {
    let seconds = env::var(name).map_or(Ok(default), |value| value.parse::<u64>())?;
    if seconds == 0 {
        bail!("{name} must be positive");
    }
    Ok(Duration::from_secs(seconds))
}

fn required(name: &str) -> Result<String> {
    let value = env::var(name).with_context(|| format!("{name} must be set"))?;
    if value.is_empty() {
        bail!("{name} must not be empty");
    }
    Ok(value)
}

fn env_value(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_owned())
}

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(prefix: &str) -> Result<Self> {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = env::temp_dir().join(format!("{prefix}-{}-{sequence}", std::process::id()));
        fs::create_dir(&path).with_context(|| format!("failed to create {}", path.display()))?;
        Ok(Self { path })
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.path) {
            eprintln!(
                "failed to remove temporary directory {}: {error}",
                self.path.display()
            );
        }
    }
}
