use std::env;
use std::ffi::{OsStr, OsString};
use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{Read, Write};
use std::os::unix::ffi::OsStringExt;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::CommandExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const SCHEMA_VERSION: u32 = 2;
const DEFAULT_TIMEOUT_SECONDS: u64 = 3600;
const MAX_TIMEOUT_SECONDS: u64 = 3600;
const MAX_OUTSTANDING_JOBS: usize = 4;
const QUEUE_POLL_INTERVAL: Duration = Duration::from_millis(200);
const RESULT_LIMIT_BYTES: u64 = 8 * 1024 * 1024;
const ENVIRONMENT_PROBE_LIMIT_BYTES: usize = 1024 * 1024;
const PYTHON_ENVIRONMENT_PROBE: &str = r#"
import importlib.metadata as metadata
import json
import platform

packages = sorted(
    (distribution.metadata.get("Name") or "", distribution.version or "")
    for distribution in metadata.distributions()
)
print(json.dumps(
    {"packages": packages, "python_version": platform.python_version()},
    sort_keys=True,
    separators=(",", ":"),
))
"#;
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum JobState {
    Preparing,
    AwaitingHook,
    Queued,
    Running,
    Succeeded,
    Failed,
    TimedOut,
    Abandoned,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct EnvironmentIdentity {
    uv_lock_sha256: String,
    python_executable_sha256: String,
    python_inventory_sha256: String,
    python_version: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct JobRecord {
    schema_version: u32,
    job_id: String,
    state: JobState,
    owner_pid: u32,
    benchmark_pid: Option<u32>,
    created_at_unix_ms: u64,
    #[serde(default)]
    queued_at_unix_ms: Option<u64>,
    started_at_unix_ms: Option<u64>,
    finished_at_unix_ms: Option<u64>,
    workspace: PathBuf,
    base_commit: String,
    snapshot_sha256: Option<String>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    python: Option<PathBuf>,
    #[serde(default)]
    environment: Option<EnvironmentIdentity>,
    #[serde(default = "default_gpu_device")]
    gpu_device: String,
    #[serde(default = "default_timeout_seconds")]
    timeout_seconds: u64,
    benchmark_args: Vec<String>,
    result_path: PathBuf,
    log_path: PathBuf,
    exit_code: Option<i32>,
    error: Option<String>,
}

#[derive(Debug)]
struct RunOptions {
    workspace: PathBuf,
    python: Option<PathBuf>,
    gpu_device: String,
    timeout: Duration,
    session_id: Option<String>,
    benchmark_args: Vec<OsString>,
}

#[derive(Debug)]
enum Cli {
    Submit(RunOptions),
    HookStop,
    List { workspace: PathBuf },
    Show { workspace: PathBuf, job_id: String },
    Cancel { workspace: PathBuf, job_id: String },
    Help,
}

#[derive(Debug)]
struct Repository {
    root: PathBuf,
    state_root: PathBuf,
}

fn main() {
    let exit_code = match run_cli(env::args_os().skip(1).collect()) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("benchmarkctl: {error:#}");
            1
        }
    };
    std::process::exit(exit_code);
}

fn run_cli(args: Vec<OsString>) -> Result<i32> {
    match parse_cli(args)? {
        Cli::Submit(options) => submit_benchmark(options),
        Cli::HookStop => run_stop_hook(),
        Cli::List { workspace } => {
            list_jobs(&Repository::discover(&workspace)?)?;
            Ok(0)
        }
        Cli::Show { workspace, job_id } => {
            show_job(&Repository::discover(&workspace)?, &job_id)?;
            Ok(0)
        }
        Cli::Cancel { workspace, job_id } => {
            cancel_job(&Repository::discover(&workspace)?, &job_id)?;
            Ok(0)
        }
        Cli::Help => {
            print_help();
            Ok(0)
        }
    }
}

fn parse_cli(args: Vec<OsString>) -> Result<Cli> {
    let mut args = args.into_iter();
    let Some(command) = args.next() else {
        return Ok(Cli::Help);
    };
    if command == "help" || command == "--help" || command == "-h" {
        return Ok(Cli::Help);
    }

    match command.to_str() {
        Some("submit") => parse_submit(args.collect()),
        Some("hook-stop") => {
            if args.next().is_some() {
                bail!("hook-stop does not accept arguments");
            }
            Ok(Cli::HookStop)
        }
        Some("list") => {
            let workspace = parse_workspace_only(args.collect())?;
            Ok(Cli::List { workspace })
        }
        Some("show") => parse_show(args.collect()),
        Some("cancel") => parse_cancel(args.collect()),
        _ => bail!("unknown command {:?}; use --help", command),
    }
}

fn parse_submit(args: Vec<OsString>) -> Result<Cli> {
    Ok(Cli::Submit(parse_benchmark_options(args)?))
}

fn parse_benchmark_options(args: Vec<OsString>) -> Result<RunOptions> {
    let mut workspace = env::current_dir()?;
    let mut python = None;
    let mut gpu_device = "0".to_owned();
    let mut timeout = Duration::from_secs(DEFAULT_TIMEOUT_SECONDS);
    let mut session_id = None;
    let mut benchmark_args = Vec::new();
    let mut index = 0;

    while index < args.len() {
        match args[index].to_str() {
            Some("--") => {
                benchmark_args.extend_from_slice(&args[index + 1..]);
                break;
            }
            Some("--workspace") => {
                workspace = PathBuf::from(option_value(&args, &mut index, "--workspace")?);
            }
            Some("--python") => {
                python = Some(PathBuf::from(option_value(&args, &mut index, "--python")?));
            }
            Some("--gpu-device") => {
                gpu_device = option_value(&args, &mut index, "--gpu-device")?
                    .to_string_lossy()
                    .into_owned();
            }
            Some("--timeout-seconds") => {
                let value = option_value(&args, &mut index, "--timeout-seconds")?;
                let seconds = value
                    .to_str()
                    .ok_or_else(|| anyhow!("--timeout-seconds must be UTF-8"))?
                    .parse::<u64>()?;
                if seconds == 0 {
                    bail!("--timeout-seconds must be positive");
                }
                if seconds > MAX_TIMEOUT_SECONDS {
                    bail!(
                        "--timeout-seconds cannot exceed {MAX_TIMEOUT_SECONDS}; this bound keeps the Codex Stop hook wait finite"
                    );
                }
                timeout = Duration::from_secs(seconds);
            }
            Some("--session-id") => {
                let value = option_value(&args, &mut index, "--session-id")?
                    .to_str()
                    .ok_or_else(|| anyhow!("--session-id must be UTF-8"))?;
                validate_session_id(value)?;
                session_id = Some(value.to_owned());
            }
            Some("--help" | "-h") => bail!("use benchmarkctl --help"),
            Some(option) if option.starts_with('-') => {
                bail!("unknown submit option {option}; put benchmark arguments after --");
            }
            _ => bail!("benchmark arguments must follow --"),
        }
        index += 1;
    }

    if benchmark_args.iter().any(|arg| {
        arg == "--json-output"
            || arg
                .to_str()
                .is_some_and(|value| value.starts_with("--json-output="))
    }) {
        bail!("benchmarkctl owns --json-output; do not pass it after --");
    }

    Ok(RunOptions {
        workspace,
        python,
        gpu_device,
        timeout,
        session_id,
        benchmark_args,
    })
}

fn parse_workspace_only(args: Vec<OsString>) -> Result<PathBuf> {
    let mut workspace = env::current_dir()?;
    let mut index = 0;
    while index < args.len() {
        match args[index].to_str() {
            Some("--workspace") => {
                workspace = PathBuf::from(option_value(&args, &mut index, "--workspace")?);
            }
            Some("--help" | "-h") => bail!("use benchmarkctl --help"),
            _ => bail!("unexpected argument {:?}", args[index]),
        }
        index += 1;
    }
    Ok(workspace)
}

fn parse_show(args: Vec<OsString>) -> Result<Cli> {
    let mut workspace = env::current_dir()?;
    let mut job_id = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].to_str() {
            Some("--workspace") => {
                workspace = PathBuf::from(option_value(&args, &mut index, "--workspace")?);
            }
            Some(value) if !value.starts_with('-') && job_id.is_none() => {
                job_id = Some(value.to_owned());
            }
            _ => bail!("unexpected show argument {:?}", args[index]),
        }
        index += 1;
    }
    let job_id = job_id.ok_or_else(|| anyhow!("show requires a job ID"))?;
    validate_job_id(&job_id)?;
    Ok(Cli::Show { workspace, job_id })
}

fn parse_cancel(args: Vec<OsString>) -> Result<Cli> {
    let mut workspace = env::current_dir()?;
    let mut job_id = None;
    let mut index = 0;
    while index < args.len() {
        match args[index].to_str() {
            Some("--workspace") => {
                workspace = PathBuf::from(option_value(&args, &mut index, "--workspace")?);
            }
            Some(value) if !value.starts_with('-') && job_id.is_none() => {
                job_id = Some(value.to_owned());
            }
            _ => bail!("unexpected cancel argument {:?}", args[index]),
        }
        index += 1;
    }
    let job_id = job_id.ok_or_else(|| anyhow!("cancel requires a job ID"))?;
    validate_job_id(&job_id)?;
    Ok(Cli::Cancel { workspace, job_id })
}

fn option_value<'a>(args: &'a [OsString], index: &mut usize, name: &str) -> Result<&'a OsStr> {
    *index += 1;
    args.get(*index)
        .map(OsString::as_os_str)
        .ok_or_else(|| anyhow!("{name} requires a value"))
}

fn print_help() {
    println!(
        r#"benchmarkctl - serialize local GPU benchmarks

Usage:
  benchmarkctl submit [--session-id ID] [--workspace PATH] [--python ABS_PATH]
                      [--gpu-device ID] [--timeout-seconds N]
                      [-- BENCHMARK_ARGS...]
  benchmarkctl list [--workspace PATH]
  benchmarkctl show JOB_ID [--workspace PATH]
  benchmarkctl cancel JOB_ID [--workspace PATH]

submit snapshots immediately; the Codex Stop hook runs the job asynchronously and
continues the same session with the result and log paths.
cancel only cancels an awaiting_hook job before the Stop hook claims it."#
    );
}

fn default_gpu_device() -> String {
    "0".to_owned()
}

fn default_timeout_seconds() -> u64 {
    DEFAULT_TIMEOUT_SECONDS
}

impl Repository {
    fn discover(start: &Path) -> Result<Self> {
        let output = Command::new("git")
            .args(["rev-parse", "--show-toplevel"])
            .current_dir(start)
            .output()
            .context("failed to start git")?;
        if !output.status.success() {
            bail!(
                "not inside a Git worktree: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        let root = fs::canonicalize(PathBuf::from(String::from_utf8(output.stdout)?.trim()))?;
        let state_root = match env::var_os("BENCHMARKCTL_STATE_DIR") {
            Some(value) => {
                let path = PathBuf::from(value);
                if !path.is_absolute() {
                    bail!("BENCHMARKCTL_STATE_DIR must be absolute");
                }
                path
            }
            None => root.join(".benchmarkctl"),
        };
        Ok(Self { root, state_root })
    }

    fn initialize(&self) -> Result<()> {
        fs::create_dir_all(self.state_root.join("jobs"))?;
        Ok(())
    }

    fn job_dir(&self, job_id: &str) -> PathBuf {
        self.state_root.join("jobs").join(job_id)
    }
}

#[derive(Debug)]
struct CompletionEvent {
    job_id: String,
    state: JobState,
    result_path: PathBuf,
    log_path: PathBuf,
}

#[derive(Debug, Deserialize)]
struct StopHookInput {
    session_id: String,
    cwd: PathBuf,
    hook_event_name: String,
    #[serde(default, rename = "stop_hook_active")]
    _stop_hook_active: bool,
}

fn submit_benchmark(mut options: RunOptions) -> Result<i32> {
    let session_id = options
        .session_id
        .take()
        .or_else(|| env::var("CODEX_SESSION_ID").ok())
        .ok_or_else(|| {
            anyhow!("submit requires --session-id or the CODEX_SESSION_ID environment variable")
        })?;
    validate_session_id(&session_id)?;
    let (repository, job_id, owner_lock) =
        prepare_job(options, JobState::AwaitingHook, Some(session_id.clone()))?;
    drop(owner_lock);
    let job = read_job(&repository, &job_id)?;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "schema_version": SCHEMA_VERSION,
            "event": "benchmark_submitted",
            "job_id": job.job_id,
            "state": job.state,
            "session_id": session_id,
            "snapshot_sha256": job.snapshot_sha256,
            "job_path": repository.job_dir(&job_id).join("job.json"),
        }))?
    );
    Ok(0)
}

fn prepare_job(
    options: RunOptions,
    ready_state: JobState,
    session_id: Option<String>,
) -> Result<(Repository, String, File)> {
    if !matches!(ready_state, JobState::Queued | JobState::AwaitingHook) {
        bail!("invalid prepared-job state");
    }
    let repository = Repository::discover(&options.workspace)?;
    repository.initialize()?;
    let python = resolve_python(&repository.root, options.python.as_deref())?;
    let benchmark_args = options
        .benchmark_args
        .iter()
        .map(|arg| {
            arg.to_str()
                .map(str::to_owned)
                .ok_or_else(|| anyhow!("benchmark arguments must be UTF-8"))
        })
        .collect::<Result<Vec<_>>>()?;
    let state_lock = open_lock_file(&repository.state_root.join("state.lock"))?;
    state_lock.lock()?;
    abandon_dead_jobs(&repository)?;
    let outstanding_jobs = read_all_jobs(&repository)?
        .into_iter()
        .filter(|job| {
            matches!(
                job.state,
                JobState::Preparing | JobState::AwaitingHook | JobState::Queued | JobState::Running
            )
        })
        .count();
    if outstanding_jobs >= MAX_OUTSTANDING_JOBS {
        bail!(
            "the local GPU queue already has {outstanding_jobs} unfinished jobs; the limit is {MAX_OUTSTANDING_JOBS}"
        );
    }
    let job_id = new_job_id()?;
    let job_dir = repository.job_dir(&job_id);
    let source_dir = job_dir.join("source");
    let result_path = job_dir.join("result.json");
    let log_path = job_dir.join("benchmark.log");
    fs::create_dir_all(&source_dir)?;
    let owner_lock = open_lock_file(&job_dir.join("owner.lock"))?;
    owner_lock.lock()?;

    let mut job = JobRecord {
        schema_version: SCHEMA_VERSION,
        job_id: job_id.clone(),
        state: JobState::Preparing,
        owner_pid: std::process::id(),
        benchmark_pid: None,
        created_at_unix_ms: unix_ms()?,
        queued_at_unix_ms: None,
        started_at_unix_ms: None,
        finished_at_unix_ms: None,
        workspace: repository.root.clone(),
        base_commit: git_head(&repository.root)?,
        snapshot_sha256: None,
        session_id,
        python: Some(python.clone()),
        environment: None,
        gpu_device: options.gpu_device,
        timeout_seconds: options.timeout.as_secs(),
        benchmark_args,
        result_path: result_path.clone(),
        log_path: log_path.clone(),
        exit_code: None,
        error: None,
    };
    write_job(&repository, &job)?;
    drop(state_lock);

    let snapshot_digest = match snapshot_worktree(&repository.root, &source_dir) {
        Ok(digest) => digest,
        Err(error) => {
            job.state = JobState::Failed;
            job.finished_at_unix_ms = Some(unix_ms()?);
            job.error = Some(format!("snapshot failed: {error:#}"));
            write_job(&repository, &job)?;
            return Err(error.context("failed to snapshot worktree"));
        }
    };
    let environment = match environment_identity(&python, &source_dir.join("uv.lock")) {
        Ok(environment) => environment,
        Err(error) => {
            job.state = JobState::Failed;
            job.finished_at_unix_ms = Some(unix_ms()?);
            job.error = Some(format!("environment capture failed: {error:#}"));
            write_job(&repository, &job)?;
            return Err(error.context("failed to capture benchmark environment"));
        }
    };
    job.snapshot_sha256 = Some(snapshot_digest);
    job.environment = Some(environment);
    job.state = ready_state;
    if job.state == JobState::Queued {
        job.queued_at_unix_ms = Some(unix_ms()?);
    }
    write_job(&repository, &job)?;
    eprintln!("[benchmarkctl] prepared {job_id} ({:?})", job.state);
    Ok((repository, job_id, owner_lock))
}

fn execute_queued_job(
    repository: &Repository,
    job_id: &str,
    owner_lock: File,
) -> Result<CompletionEvent> {
    let mut job = read_job(repository, job_id)?;
    if job.state != JobState::Queued {
        bail!("job {job_id} is not queued");
    }
    let gpu_lock = match wait_for_turn(repository, job_id) {
        Ok(lock) => lock,
        Err(error) => {
            job.state = JobState::Failed;
            job.finished_at_unix_ms = Some(unix_ms()?);
            job.error = Some(format!("queue failed: {error:#}"));
            write_job(repository, &job)?;
            return Err(error.context("failed while waiting for the GPU"));
        }
    };
    job = read_job(repository, job_id)?;
    job.state = JobState::Running;
    job.owner_pid = std::process::id();
    job.started_at_unix_ms = Some(unix_ms()?);
    write_job(repository, &job)?;
    eprintln!("[benchmarkctl] running {job_id} on GPU {}", job.gpu_device);

    let python = job
        .python
        .clone()
        .ok_or_else(|| anyhow!("job {job_id} does not record a Python executable"))?;
    let options = RunOptions {
        workspace: job.workspace.clone(),
        python: Some(python.clone()),
        gpu_device: job.gpu_device.clone(),
        timeout: Duration::from_secs(job.timeout_seconds),
        session_id: job.session_id.clone(),
        benchmark_args: job.benchmark_args.iter().map(OsString::from).collect(),
    };
    let job_dir = repository.job_dir(job_id);
    let source_dir = job_dir.join("source");
    let result_path = job.result_path.clone();
    let log_path = job.log_path.clone();
    let outcome = match verify_environment_identity(&job, &python, &source_dir.join("uv.lock")) {
        Ok(()) => execute_benchmark(Execution {
            repository,
            job: &mut job,
            gpu_lock,
            python: &python,
            source: &source_dir,
            job_dir: &job_dir,
            result_path: &result_path,
            log_path: &log_path,
            options: &options,
        }),
        Err(error) => Err(error.context("benchmark environment changed after submission")),
    };
    job = read_job(repository, job_id)?;
    job.finished_at_unix_ms = Some(unix_ms()?);
    let result_summary = read_result_summary(&result_path);
    match outcome {
        Ok(ProcessOutcome::Exited(status)) => {
            job.exit_code = status.code();
            let correctness_passed = result_summary
                .as_ref()
                .ok()
                .and_then(|result| result.get("correctness_passed"))
                .and_then(Value::as_bool)
                == Some(true);
            if status.success() && correctness_passed {
                job.state = JobState::Succeeded;
            } else {
                job.state = JobState::Failed;
                if let Err(error) = &result_summary {
                    job.error = Some(format!("invalid benchmark result: {error:#}"));
                } else if status.success() {
                    job.error = Some(
                        "benchmark exited successfully without correctness_passed=true".to_owned(),
                    );
                }
            }
        }
        Ok(ProcessOutcome::TimedOut) => {
            job.state = JobState::TimedOut;
            job.exit_code = Some(124);
            job.error = Some(format!(
                "benchmark exceeded {} seconds",
                options.timeout.as_secs()
            ));
        }
        Err(error) => {
            job.state = JobState::Failed;
            job.exit_code = Some(1);
            job.error = Some(format!("benchmark launch failed: {error:#}"));
        }
    }
    write_job(repository, &job)?;
    drop(owner_lock);
    Ok(CompletionEvent {
        job_id: job.job_id,
        state: job.state,
        result_path: job.result_path,
        log_path: job.log_path,
    })
}

fn run_stop_hook() -> Result<i32> {
    let mut input = String::new();
    std::io::stdin()
        .take(64 * 1024)
        .read_to_string(&mut input)?;
    let hook: StopHookInput =
        serde_json::from_str(&input).context("hook-stop expected Codex hook JSON on stdin")?;
    if hook.hook_event_name != "Stop" {
        bail!("hook-stop only accepts the Codex Stop event");
    }
    validate_session_id(&hook.session_id)?;
    let repository = Repository::discover(&hook.cwd)?;
    repository.initialize()?;
    let Some((job_id, owner_lock)) = claim_hook_job(&repository, &hook.session_id)? else {
        println!("{{}}");
        return Ok(0);
    };
    let reason = match execute_queued_job(&repository, &job_id, owner_lock) {
        Ok(completion) => format!(
            "Benchmark job {} finished with state {:?}. Inspect the deterministic result at {} \
and the full log at {}. Verify numerical correctness before interpreting performance. If the \
evidence is insufficient, submit at most one focused follow-up with benchmarkctl submit.",
            completion.job_id,
            completion.state,
            completion.result_path.display(),
            completion.log_path.display()
        ),
        Err(error) => {
            let mut job = read_job(&repository, &job_id)?;
            if matches!(job.state, JobState::Queued | JobState::Running) {
                job.state = JobState::Failed;
                job.finished_at_unix_ms = Some(unix_ms()?);
                job.error = Some(format!("hook execution failed: {error:#}"));
                write_job(&repository, &job)?;
            }
            format!(
                "Benchmark job {job_id} failed in the queue runner. Inspect {} and {} before \
deciding whether to retry. The runner error was: {}",
                repository.job_dir(&job_id).join("job.json").display(),
                job.log_path.display(),
                bounded_error(&error)
            )
        }
    };
    println!(
        "{}",
        serde_json::to_string(&json!({
            "decision": "block",
            "reason": reason,
        }))?
    );
    Ok(0)
}

fn bounded_error(error: &anyhow::Error) -> String {
    let rendered = format!("{error:#}");
    rendered.chars().take(500).collect()
}

fn claim_hook_job(repository: &Repository, session_id: &str) -> Result<Option<(String, File)>> {
    let state_lock = open_lock_file(&repository.state_root.join("state.lock"))?;
    state_lock.lock()?;
    let mut jobs = read_all_jobs(repository)?
        .into_iter()
        .filter(|job| {
            job.state == JobState::AwaitingHook && job.session_id.as_deref() == Some(session_id)
        })
        .collect::<Vec<_>>();
    jobs.sort_by_key(|job| job.created_at_unix_ms);
    let Some(mut job) = jobs.into_iter().next() else {
        return Ok(None);
    };
    let owner_lock = open_lock_file(&repository.job_dir(&job.job_id).join("owner.lock"))?;
    owner_lock.lock()?;
    job.state = JobState::Queued;
    job.owner_pid = std::process::id();
    job.queued_at_unix_ms = Some(unix_ms()?);
    write_job(repository, &job)?;
    Ok(Some((job.job_id, owner_lock)))
}

fn resolve_python(root: &Path, configured: Option<&Path>) -> Result<PathBuf> {
    let path = configured
        .map(Path::to_path_buf)
        .or_else(|| env::var_os("BENCHMARKCTL_PYTHON").map(PathBuf::from))
        .unwrap_or_else(|| root.join(".venv/bin/python"));
    if !path.is_absolute() {
        bail!("Python path must be absolute: {}", path.display());
    }
    if !path.is_file() {
        bail!("Python path is not a file: {}", path.display());
    }
    Ok(path)
}

fn environment_identity(python: &Path, uv_lock: &Path) -> Result<EnvironmentIdentity> {
    let resolved_python = fs::canonicalize(python)
        .with_context(|| format!("failed to resolve Python at {}", python.display()))?;
    let output = Command::new(python)
        .args(["-I", "-c", PYTHON_ENVIRONMENT_PROBE])
        .env_clear()
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("PYTHONNOUSERSITE", "1")
        .output()
        .with_context(|| {
            format!(
                "failed to inspect Python environment at {}",
                python.display()
            )
        })?;
    if !output.status.success() {
        bail!(
            "Python environment probe failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    if output.stdout.len() > ENVIRONMENT_PROBE_LIMIT_BYTES {
        bail!("Python environment inventory exceeds 1 MiB");
    }
    let inventory: Value = serde_json::from_slice(&output.stdout)
        .context("Python environment probe returned invalid JSON")?;
    let python_version = inventory
        .get("python_version")
        .and_then(Value::as_str)
        .filter(|version| !version.is_empty())
        .ok_or_else(|| anyhow!("Python environment probe omitted python_version"))?
        .to_owned();
    if !inventory.get("packages").is_some_and(Value::is_array) {
        bail!("Python environment probe omitted packages");
    }
    let canonical_inventory = serde_json::to_vec(&inventory)?;
    Ok(EnvironmentIdentity {
        uv_lock_sha256: sha256_file(uv_lock)?,
        python_executable_sha256: sha256_file(&resolved_python)?,
        python_inventory_sha256: sha256_bytes(&canonical_inventory),
        python_version,
    })
}

fn verify_environment_identity(job: &JobRecord, python: &Path, uv_lock: &Path) -> Result<()> {
    let expected = job
        .environment
        .as_ref()
        .ok_or_else(|| anyhow!("job does not record a benchmark environment identity"))?;
    let actual = environment_identity(python, uv_lock)?;
    if &actual != expected {
        bail!(
            "the recorded Python executable, package inventory, version, or uv.lock digest changed; resubmit after running uv sync --frozen"
        );
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)
        .with_context(|| format!("failed to open environment file {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn snapshot_worktree(root: &Path, destination: &Path) -> Result<String> {
    let output = Command::new("git")
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ])
        .current_dir(root)
        .output()
        .context("failed to list snapshot files")?;
    if !output.status.success() {
        bail!(
            "git ls-files failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    let mut hasher = Sha256::new();
    for raw_path in output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|p| !p.is_empty())
    {
        let relative = PathBuf::from(OsString::from_vec(raw_path.to_vec()));
        validate_relative_path(&relative)?;
        let source = root.join(&relative);
        let metadata = match fs::symlink_metadata(&source) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        let target = destination.join(&relative);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }

        hasher.update((raw_path.len() as u64).to_le_bytes());
        hasher.update(raw_path);
        hasher.update(metadata.permissions().mode().to_le_bytes());
        if metadata.file_type().is_symlink() {
            bail!(
                "symbolic links are not supported in immutable snapshots: {}",
                relative.display()
            );
        } else if metadata.is_file() {
            hasher.update(b"file\0");
            copy_and_hash(&source, &target, &mut hasher)?;
            fs::set_permissions(&target, metadata.permissions())?;
        } else if metadata.is_dir() {
            bail!(
                "Git submodules are not supported in snapshots: {}",
                relative.display()
            );
        } else {
            bail!("unsupported file type in snapshot: {}", relative.display());
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn copy_and_hash(source: &Path, target: &Path, hasher: &mut Sha256) -> Result<()> {
    let mut input = File::open(source)?;
    let mut output = File::create(target)?;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
        output.write_all(&buffer[..count])?;
    }
    output.sync_all()?;
    Ok(())
}

fn validate_relative_path(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("Git returned an unsafe snapshot path: {}", path.display());
    }
    Ok(())
}

fn wait_for_turn(repository: &Repository, job_id: &str) -> Result<File> {
    let state_lock_path = repository.state_root.join("state.lock");
    let gpu_lock_path = repository.state_root.join("gpu.lock");
    loop {
        let state_lock = open_lock_file(&state_lock_path)?;
        state_lock.lock()?;
        abandon_dead_jobs(repository)?;
        let first = queued_jobs(repository)?.into_iter().next();
        if first.as_ref().is_some_and(|job| job.job_id == job_id)
            && !has_active_running_job(repository)?
        {
            let gpu_lock = open_lock_file(&gpu_lock_path)?;
            match gpu_lock.try_lock() {
                Ok(()) => return Ok(gpu_lock),
                Err(TryLockError::WouldBlock) => {}
                Err(TryLockError::Error(error)) => return Err(error.into()),
            }
        }
        drop(state_lock);
        thread::sleep(QUEUE_POLL_INTERVAL);
    }
}

fn open_lock_file(path: &Path) -> Result<File> {
    OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path)
        .with_context(|| format!("failed to open lock file {}", path.display()))
}

fn abandon_dead_jobs(repository: &Repository) -> Result<()> {
    for mut job in read_all_jobs(repository)? {
        let owner_active = job_owner_is_active(repository, &job.job_id)?;
        let abandoned = match job.state {
            JobState::Preparing | JobState::Queued => !owner_active,
            JobState::Running => !owner_active && gpu_is_available(repository)?,
            _ => false,
        };
        if abandoned {
            job.state = JobState::Abandoned;
            job.finished_at_unix_ms = Some(unix_ms()?);
            job.error = Some("benchmarkctl owner and benchmark processes exited".to_owned());
            write_job(repository, &job)?;
        }
    }
    Ok(())
}

fn has_active_running_job(repository: &Repository) -> Result<bool> {
    Ok(read_all_jobs(repository)?
        .into_iter()
        .any(|job| job.state == JobState::Running))
}

fn job_owner_is_active(repository: &Repository, job_id: &str) -> Result<bool> {
    let lock = open_lock_file(&repository.job_dir(job_id).join("owner.lock"))?;
    match lock.try_lock() {
        Ok(()) => Ok(false),
        Err(TryLockError::WouldBlock) => Ok(true),
        Err(TryLockError::Error(error)) => Err(error.into()),
    }
}

fn gpu_is_available(repository: &Repository) -> Result<bool> {
    let lock = open_lock_file(&repository.state_root.join("gpu.lock"))?;
    match lock.try_lock() {
        Ok(()) => Ok(true),
        Err(TryLockError::WouldBlock) => Ok(false),
        Err(TryLockError::Error(error)) => Err(error.into()),
    }
}

fn queued_jobs(repository: &Repository) -> Result<Vec<JobRecord>> {
    let mut jobs = read_all_jobs(repository)?
        .into_iter()
        .filter(|job| job.state == JobState::Queued)
        .collect::<Vec<_>>();
    jobs.sort_by(|left, right| {
        (
            left.queued_at_unix_ms.unwrap_or(left.created_at_unix_ms),
            &left.job_id,
        )
            .cmp(&(
                right.queued_at_unix_ms.unwrap_or(right.created_at_unix_ms),
                &right.job_id,
            ))
    });
    Ok(jobs)
}

enum ProcessOutcome {
    Exited(ExitStatus),
    TimedOut,
}

struct Execution<'a> {
    repository: &'a Repository,
    job: &'a mut JobRecord,
    gpu_lock: File,
    python: &'a Path,
    source: &'a Path,
    job_dir: &'a Path,
    result_path: &'a Path,
    log_path: &'a Path,
    options: &'a RunOptions,
}

fn execute_benchmark(execution: Execution<'_>) -> Result<ProcessOutcome> {
    let Execution {
        repository,
        job,
        gpu_lock,
        python,
        source,
        job_dir,
        result_path,
        log_path,
        options,
    } = execution;
    if !source.join("torch_transformer_benchmark.py").is_file() {
        bail!("snapshot does not contain torch_transformer_benchmark.py");
    }
    let flock = resolve_flock()?;
    let log = File::create(log_path)?;
    let log_error = log.try_clone()?;
    let home = job_dir.join("home");
    let cache = job_dir.join("cache");
    fs::create_dir_all(&home)?;
    fs::create_dir_all(&cache)?;
    let python_bin = python
        .parent()
        .ok_or_else(|| anyhow!("Python executable has no parent directory"))?;
    let process_path = format!(
        "{}:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        python_bin.display()
    );

    let gpu_lock_path = repository.state_root.join("gpu.lock");
    let mut command = Command::new(flock);
    command
        .current_dir(source)
        .env_clear()
        .env("PATH", process_path)
        .env("HOME", &home)
        .env("XDG_CACHE_HOME", &cache)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("PYTHONNOUSERSITE", "1")
        .env("PYTHONUNBUFFERED", "1")
        .env("CUDA_VISIBLE_DEVICES", &options.gpu_device)
        .arg("--exclusive")
        .arg(&gpu_lock_path)
        .arg(python)
        .arg("torch_transformer_benchmark.py")
        .args(&options.benchmark_args)
        .arg("--json-output")
        .arg(result_path)
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(log_error))
        .process_group(0);
    let mut child = command.spawn().context("failed to start benchmark")?;
    job.benchmark_pid = Some(child.id());
    write_job(repository, job)?;
    drop(gpu_lock);
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(ProcessOutcome::Exited(status));
        }
        if started.elapsed() >= options.timeout {
            terminate_process_group(child.id());
            let _ = child.wait();
            return Ok(ProcessOutcome::TimedOut);
        }
        thread::sleep(QUEUE_POLL_INTERVAL);
    }
}

fn resolve_flock() -> Result<PathBuf> {
    for path in ["/usr/bin/flock", "/run/current-system/sw/bin/flock"] {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
    }
    if let Some(path) = env::var_os("PATH").and_then(|paths| {
        env::split_paths(&paths)
            .map(|directory| directory.join("flock"))
            .find(|candidate| candidate.is_file())
    }) {
        return Ok(path);
    }
    bail!("flock is required but was not found")
}

fn terminate_process_group(pid: u32) {
    let _ = Command::new("kill")
        .args(["-TERM", "--", &format!("-{pid}")])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    thread::sleep(Duration::from_secs(1));
    let _ = Command::new("kill")
        .args(["-KILL", "--", &format!("-{pid}")])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

fn read_result_summary(path: &Path) -> Result<Value> {
    let metadata = fs::metadata(path).with_context(|| format!("missing {}", path.display()))?;
    if metadata.len() > RESULT_LIMIT_BYTES {
        bail!("result JSON exceeds 8 MiB");
    }
    let value: Value = serde_json::from_slice(&fs::read(path)?)?;
    let object = value
        .as_object()
        .ok_or_else(|| anyhow!("result JSON must be an object"))?;
    let schema_version = object
        .get("schema_version")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("result JSON must contain an integer schema_version"))?;

    match schema_version {
        1 => {
            require_object_field(object, "config")?;
            require_object_field(object, "environment")?;
            require_object_field(object, "settings")?;
            require_object_field(object, "accuracy")?;
            require_null_or_object_field(object, "performance")?;
            let correctness_passed = require_bool_field(object, "correctness_passed")?;
            Ok(json!({
                "schema_version": schema_version,
                "mode": "single_case",
                "correctness_passed": correctness_passed,
                "failure_category": if correctness_passed { Value::Null } else { json!("correctness") },
            }))
        }
        2 => {
            if object.get("mode").and_then(Value::as_str) != Some("official_matrix") {
                bail!("schema 2 result mode must be official_matrix");
            }
            require_object_field(object, "environment")?;
            require_object_field(object, "settings")?;
            let requested = object
                .get("requested_case_ids")
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow!("schema 2 result must contain requested_case_ids"))?;
            if requested.is_empty() || requested.iter().any(|case| case.as_u64().is_none()) {
                bail!("requested_case_ids must be a non-empty integer array");
            }
            let cases = object
                .get("cases")
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow!("schema 2 result must contain cases"))?;
            let complete = require_bool_field(object, "complete")?;
            let all_cases_executed = require_bool_field(object, "all_cases_executed")?;
            if complete != (cases.len() == requested.len()) {
                bail!("schema 2 complete does not match the number of case results");
            }
            if all_cases_executed && !complete {
                bail!("schema 2 all_cases_executed requires complete=true");
            }
            let correctness = object
                .get("correctness_passed")
                .ok_or_else(|| anyhow!("schema 2 result omitted correctness_passed"))?;
            let correctness_passed = if correctness.is_null() {
                None
            } else {
                Some(
                    correctness
                        .as_bool()
                        .ok_or_else(|| anyhow!("correctness_passed must be boolean or null"))?,
                )
            };
            if correctness_passed == Some(true) && (!complete || !all_cases_executed) {
                bail!("a passing schema 2 result must be complete and fully executed");
            }
            let failure_category = object
                .get("failure_category")
                .ok_or_else(|| anyhow!("schema 2 result omitted failure_category"))?;
            if !failure_category.is_null() && !failure_category.is_string() {
                bail!("failure_category must be a string or null");
            }
            Ok(json!({
                "schema_version": schema_version,
                "mode": "official_matrix",
                "correctness_passed": correctness_passed,
                "failure_category": failure_category,
            }))
        }
        other => bail!("unsupported benchmark result schema version {other}"),
    }
}

fn require_object_field(object: &serde_json::Map<String, Value>, field: &str) -> Result<()> {
    if !object.get(field).is_some_and(Value::is_object) {
        bail!("result field {field} must be an object");
    }
    Ok(())
}

fn require_null_or_object_field(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<()> {
    let value = object
        .get(field)
        .ok_or_else(|| anyhow!("result omitted field {field}"))?;
    if !value.is_null() && !value.is_object() {
        bail!("result field {field} must be an object or null");
    }
    Ok(())
}

fn require_bool_field(object: &serde_json::Map<String, Value>, field: &str) -> Result<bool> {
    object
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| anyhow!("result field {field} must be boolean"))
}

fn list_jobs(repository: &Repository) -> Result<()> {
    repository.initialize()?;
    let mut jobs = read_all_jobs(repository)?;
    jobs.sort_by_key(|job| job.created_at_unix_ms);
    println!("{}", serde_json::to_string_pretty(&jobs)?);
    Ok(())
}

fn show_job(repository: &Repository, job_id: &str) -> Result<()> {
    repository.initialize()?;
    println!(
        "{}",
        serde_json::to_string_pretty(&read_job(repository, job_id)?)?
    );
    Ok(())
}

fn cancel_job(repository: &Repository, job_id: &str) -> Result<()> {
    repository.initialize()?;
    let state_lock = open_lock_file(&repository.state_root.join("state.lock"))?;
    state_lock.lock()?;
    let mut job = read_job(repository, job_id)?;
    if job.state != JobState::AwaitingHook {
        bail!(
            "job {job_id} cannot be cancelled from state {:?}; only awaiting_hook jobs can be cancelled",
            job.state
        );
    }
    job.state = JobState::Cancelled;
    job.finished_at_unix_ms = Some(unix_ms()?);
    job.error = Some("cancelled before the Stop hook claimed the job".to_owned());
    write_job(repository, &job)?;
    drop(state_lock);
    println!(
        "{}",
        serde_json::to_string(&json!({
            "schema_version": SCHEMA_VERSION,
            "event": "benchmark_cancelled",
            "job_id": job.job_id,
            "state": job.state,
            "job_path": repository.job_dir(job_id).join("job.json"),
        }))?
    );
    Ok(())
}

fn read_all_jobs(repository: &Repository) -> Result<Vec<JobRecord>> {
    let jobs_dir = repository.state_root.join("jobs");
    if !jobs_dir.exists() {
        return Ok(Vec::new());
    }
    let mut jobs = Vec::new();
    for entry in fs::read_dir(jobs_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let path = entry.path().join("job.json");
        if path.is_file() {
            jobs.push(read_json(&path)?);
        }
    }
    Ok(jobs)
}

fn read_job(repository: &Repository, job_id: &str) -> Result<JobRecord> {
    validate_job_id(job_id)?;
    let path = repository.job_dir(job_id).join("job.json");
    read_json(&path).with_context(|| format!("failed to read job {job_id}"))
}

fn write_job(repository: &Repository, job: &JobRecord) -> Result<()> {
    validate_job_id(&job.job_id)?;
    let directory = repository.job_dir(&job.job_id);
    fs::create_dir_all(&directory)?;
    atomic_write_json(&directory.join("job.json"), job)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    serde_json::from_slice(&fs::read(path)?).map_err(Into::into)
}

fn atomic_write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temp = path.with_extension(format!("tmp-{}-{sequence}", std::process::id()));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temp)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(&temp, path)?;
    Ok(())
}

fn git_head(root: &Path) -> Result<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()?;
    if !output.status.success() {
        bail!("benchmarkctl requires a repository with an existing HEAD commit");
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn new_job_id() -> Result<String> {
    let mut random = [0_u8; 8];
    let suffix = match File::open("/dev/urandom").and_then(|mut file| file.read_exact(&mut random))
    {
        Ok(()) => u64::from_le_bytes(random),
        Err(_) => {
            let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            SystemTime::now()
                .duration_since(UNIX_EPOCH)?
                .as_nanos()
                .wrapping_add(u128::from(sequence)) as u64
        }
    };
    Ok(format!("job-{}-{suffix:016x}", unix_ms()?))
}

fn unix_ms() -> Result<u64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before Unix epoch")?
        .as_millis()
        .try_into()?)
}

fn validate_job_id(job_id: &str) -> Result<()> {
    if job_id.is_empty()
        || job_id.len() > 80
        || !job_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
    {
        bail!("invalid job ID");
    }
    Ok(())
}

fn validate_session_id(session_id: &str) -> Result<()> {
    if session_id.is_empty()
        || session_id.len() > 128
        || !session_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        bail!("invalid Codex session ID");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submit_parser_accepts_and_validates_session_id() {
        let error = parse_cli(strings(&["submit", "--device", "cpu"])).unwrap_err();
        assert!(error.to_string().contains("unknown submit option"));

        let Cli::Submit(options) = parse_cli(strings(&[
            "submit",
            "--session-id",
            "01a04237-43ec-7d42-9524-ff23edd2151c",
            "--timeout-seconds",
            "10",
            "--",
            "--device",
            "cpu",
        ]))
        .unwrap() else {
            panic!("expected submit command");
        };
        assert_eq!(
            options.session_id.as_deref(),
            Some("01a04237-43ec-7d42-9524-ff23edd2151c")
        );
        assert_eq!(options.timeout, Duration::from_secs(10));
        assert_eq!(options.benchmark_args, strings(&["--device", "cpu"]));

        let error = parse_cli(strings(&["submit", "--session-id", "bad id"])).unwrap_err();
        assert!(error.to_string().contains("invalid Codex session ID"));

        let error = parse_cli(strings(&[
            "submit",
            "--timeout-seconds",
            "3601",
            "--",
            "--device",
            "cpu",
        ]))
        .unwrap_err();
        assert!(error.to_string().contains("cannot exceed 3600"));

        let error = parse_cli(strings(&["run"])).unwrap_err();
        assert!(error.to_string().contains("unknown command"));
    }

    #[test]
    fn cancel_parser_requires_one_valid_job_id() {
        let Cli::Cancel { workspace, job_id } = parse_cli(strings(&[
            "cancel",
            "job-123",
            "--workspace",
            "/tmp/worktree",
        ]))
        .unwrap() else {
            panic!("expected cancel command");
        };
        assert_eq!(workspace, PathBuf::from("/tmp/worktree"));
        assert_eq!(job_id, "job-123");

        let error = parse_cli(strings(&["cancel"])).unwrap_err();
        assert!(error.to_string().contains("requires a job ID"));

        let error = parse_cli(strings(&["cancel", "bad/id"])).unwrap_err();
        assert!(error.to_string().contains("invalid job ID"));
    }

    #[test]
    fn snapshot_captures_tracked_changes_untracked_files_and_deletions() -> Result<()> {
        let temp = TestDir::new("snapshot")?;
        let output = TestDir::new("snapshot-output")?;
        run_git(temp.path(), &["init", "--quiet"])?;
        fs::write(temp.path().join("tracked.txt"), "before")?;
        fs::write(temp.path().join("deleted.txt"), "remove me")?;
        run_git(temp.path(), &["add", "."])?;
        run_git_with_identity(temp.path(), &["commit", "--quiet", "-m", "base"])?;

        fs::write(temp.path().join("tracked.txt"), "after")?;
        fs::remove_file(temp.path().join("deleted.txt"))?;
        fs::write(temp.path().join("untracked.txt"), "new")?;
        fs::write(temp.path().join(".gitignore"), "ignored.txt\n")?;
        fs::write(temp.path().join("ignored.txt"), "ignored")?;
        let before_status = git_output(temp.path(), &["status", "--porcelain=v1"])?;

        let destination = output.path().join("snapshot");
        fs::create_dir(&destination)?;
        let digest = snapshot_worktree(temp.path(), &destination)?;

        assert_eq!(
            fs::read_to_string(destination.join("tracked.txt"))?,
            "after"
        );
        assert_eq!(
            fs::read_to_string(destination.join("untracked.txt"))?,
            "new"
        );
        assert!(!destination.join("deleted.txt").exists());
        assert!(!destination.join("ignored.txt").exists());
        assert_eq!(digest.len(), 64);
        assert_eq!(
            before_status,
            git_output(temp.path(), &["status", "--porcelain=v1"])?
        );
        Ok(())
    }

    #[test]
    fn snapshot_rejects_symbolic_links() -> Result<()> {
        let temp = TestDir::new("snapshot-symlink")?;
        let output = TestDir::new("snapshot-symlink-output")?;
        run_git(temp.path(), &["init", "--quiet"])?;
        std::os::unix::fs::symlink(
            "/tmp/mutable-benchmark-source",
            temp.path().join("model.py"),
        )?;
        run_git(temp.path(), &["add", "model.py"])?;

        let destination = output.path().join("snapshot");
        fs::create_dir(&destination)?;
        let error = snapshot_worktree(temp.path(), &destination).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("symbolic links are not supported")
        );
        Ok(())
    }

    #[test]
    fn environment_identity_changes_with_the_lockfile() -> Result<()> {
        let temp = TestDir::new("environment")?;
        let python = temp.path().join("python");
        fs::write(
            &python,
            "#!/bin/sh\nprintf '%s\\n' '{\"packages\":[[\"torch\",\"1\"]],\"python_version\":\"3.12.0\"}'\n",
        )?;
        fs::set_permissions(&python, fs::Permissions::from_mode(0o755))?;
        let uv_lock = temp.path().join("uv.lock");
        fs::write(&uv_lock, "first")?;
        let first = environment_identity(&python, &uv_lock)?;

        fs::write(&uv_lock, "second")?;
        let second = environment_identity(&python, &uv_lock)?;
        assert_ne!(first.uv_lock_sha256, second.uv_lock_sha256);
        assert_eq!(
            first.python_inventory_sha256,
            second.python_inventory_sha256
        );
        Ok(())
    }

    #[test]
    fn queue_applies_backpressure_after_four_unfinished_jobs() -> Result<()> {
        let temp = TestDir::new("queue-capacity")?;
        run_git(temp.path(), &["init", "--quiet"])?;
        fs::write(temp.path().join(".gitignore"), ".benchmarkctl/\n")?;
        fs::write(temp.path().join("uv.lock"), "lock")?;
        fs::write(temp.path().join("torch_transformer_benchmark.py"), "")?;
        let python = temp.path().join("python");
        fs::write(
            &python,
            "#!/bin/sh\nprintf '%s\\n' '{\"packages\":[],\"python_version\":\"3.12.0\"}'\n",
        )?;
        fs::set_permissions(&python, fs::Permissions::from_mode(0o755))?;
        run_git(temp.path(), &["add", "."])?;
        run_git_with_identity(temp.path(), &["commit", "--quiet", "-m", "base"])?;

        let mut job_ids = Vec::new();
        for index in 0..MAX_OUTSTANDING_JOBS {
            let (_, job_id, owner_lock) = prepare_job(
                test_run_options(temp.path(), &python),
                JobState::AwaitingHook,
                Some(format!("session-{index}")),
            )?;
            drop(owner_lock);
            job_ids.push(job_id);
        }
        let error = prepare_job(
            test_run_options(temp.path(), &python),
            JobState::AwaitingHook,
            Some("session-overflow".to_owned()),
        )
        .unwrap_err();
        assert!(error.to_string().contains("limit is 4"));

        let repository = Repository::discover(temp.path())?;
        cancel_job(&repository, &job_ids[0])?;
        let cancelled = read_job(&repository, &job_ids[0])?;
        assert_eq!(cancelled.state, JobState::Cancelled);
        assert!(cancelled.finished_at_unix_ms.is_some());
        assert!(claim_hook_job(&repository, "session-0")?.is_none());

        let (_, replacement_id, owner_lock) = prepare_job(
            test_run_options(temp.path(), &python),
            JobState::AwaitingHook,
            Some("session-replacement".to_owned()),
        )?;
        drop(owner_lock);
        let Some((claimed_id, claimed_owner_lock)) =
            claim_hook_job(&repository, "session-replacement")?
        else {
            panic!("expected the replacement job to be claimed");
        };
        assert_eq!(claimed_id, replacement_id);
        let error = cancel_job(&repository, &replacement_id).unwrap_err();
        assert!(error.to_string().contains("only awaiting_hook jobs"));
        drop(claimed_owner_lock);
        Ok(())
    }

    #[test]
    fn stop_hook_timeout_covers_the_bounded_queue() -> Result<()> {
        let hooks: Value = serde_json::from_str(include_str!("../../../../.codex/hooks.json"))?;
        let timeout = hooks["hooks"]["Stop"][0]["hooks"][0]["timeout"]
            .as_u64()
            .ok_or_else(|| anyhow!("Stop hook timeout is missing"))?;
        let maximum_benchmark_wait = MAX_TIMEOUT_SECONDS * MAX_OUTSTANDING_JOBS as u64;
        assert!(timeout >= maximum_benchmark_wait + 300);
        Ok(())
    }

    #[test]
    fn result_summary_requires_a_semantic_benchmark_result() -> Result<()> {
        let temp = TestDir::new("result-schema")?;
        let result_path = temp.path().join("result.json");
        fs::write(&result_path, "{}")?;
        assert!(
            read_result_summary(&result_path)
                .unwrap_err()
                .to_string()
                .contains("schema_version")
        );

        fs::write(
            &result_path,
            serde_json::to_vec(&json!({
                "schema_version": 1,
                "config": {},
                "environment": {},
                "settings": {},
                "correctness_passed": true,
                "accuracy": {},
                "performance": null,
            }))?,
        )?;
        let summary = read_result_summary(&result_path)?;
        assert_eq!(summary["mode"], "single_case");
        assert_eq!(summary["correctness_passed"], true);
        Ok(())
    }

    #[test]
    fn exclusive_file_lock_prevents_a_second_holder() -> Result<()> {
        let temp = TestDir::new("lock")?;
        let path = temp.path().join("gpu.lock");
        let first = open_lock_file(&path)?;
        let second = open_lock_file(&path)?;
        first.try_lock()?;
        assert!(matches!(second.try_lock(), Err(TryLockError::WouldBlock)));
        Ok(())
    }

    fn strings(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    fn test_run_options(workspace: &Path, python: &Path) -> RunOptions {
        RunOptions {
            workspace: workspace.to_owned(),
            python: Some(python.to_owned()),
            gpu_device: "0".to_owned(),
            timeout: Duration::from_secs(1),
            session_id: None,
            benchmark_args: Vec::new(),
        }
    }

    fn run_git(directory: &Path, args: &[&str]) -> Result<()> {
        let status = Command::new("git")
            .args(args)
            .current_dir(directory)
            .status()?;
        if !status.success() {
            bail!("git {args:?} failed with {status}");
        }
        Ok(())
    }

    fn run_git_with_identity(directory: &Path, args: &[&str]) -> Result<()> {
        let status = Command::new("git")
            .args(args)
            .current_dir(directory)
            .env("GIT_AUTHOR_NAME", "benchmarkctl test")
            .env("GIT_AUTHOR_EMAIL", "benchmarkctl@example.invalid")
            .env("GIT_COMMITTER_NAME", "benchmarkctl test")
            .env("GIT_COMMITTER_EMAIL", "benchmarkctl@example.invalid")
            .status()?;
        if !status.success() {
            bail!("git {args:?} failed with {status}");
        }
        Ok(())
    }

    fn git_output(directory: &Path, args: &[&str]) -> Result<Vec<u8>> {
        let output = Command::new("git")
            .args(args)
            .current_dir(directory)
            .output()?;
        if !output.status.success() {
            bail!("git {args:?} failed");
        }
        Ok(output.stdout)
    }

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> Result<Self> {
            let path = env::temp_dir().join(format!(
                "benchmarkctl-test-{name}-{}-{}",
                std::process::id(),
                TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path)?;
            Ok(Self(path))
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            if let Err(error) = fs::remove_dir_all(&self.0) {
                eprintln!("failed to remove {}: {error}", self.0.display());
            }
        }
    }
}
