use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use benchmark_contracts::{
    EnqueueJobRequest, EnqueueJobResponse, FailureCategory, JobKind, JobResultUpload, JobState,
    JobView, LeasedJob,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Deserialize, Serialize)]
struct PersistentState {
    schema_version: u32,
    next_job_id: u64,
    next_lease_id: u64,
    jobs: Vec<StoredJob>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Lease {
    lease_id: String,
    worker_id: String,
    expires_at_ms: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct StoredJob {
    view: JobView,
    logical_key: String,
    lease: Option<Lease>,
}

#[derive(Debug)]
pub struct Store {
    path: PathBuf,
    lease_duration_ms: u64,
    max_attempts: u32,
    inner: Mutex<PersistentState>,
}

impl Store {
    pub fn open(
        path: impl Into<PathBuf>,
        lease_duration_ms: u64,
        max_attempts: u32,
    ) -> Result<Self> {
        let path = path.into();
        let state = if path.exists() {
            let bytes = fs::read(&path)
                .with_context(|| format!("failed to read state from {}", path.display()))?;
            serde_json::from_slice(&bytes)
                .with_context(|| format!("failed to decode state from {}", path.display()))?
        } else {
            PersistentState {
                schema_version: 1,
                next_job_id: 1,
                next_lease_id: 1,
                jobs: Vec::new(),
            }
        };
        if state.schema_version != 1 {
            bail!("unsupported queue state schema {}", state.schema_version);
        }
        Ok(Self {
            path,
            lease_duration_ms,
            max_attempts,
            inner: Mutex::new(state),
        })
    }

    pub fn enqueue(&self, request: EnqueueJobRequest) -> Result<EnqueueJobResponse> {
        request.validate().map_err(|message| anyhow!(message))?;
        let logical_key = logical_key(&request)?;
        let mut state = self
            .inner
            .lock()
            .map_err(|_| anyhow!("queue lock poisoned"))?;
        expire_leases(&mut state, self.max_attempts);

        if let Some(existing) = state.jobs.iter().find(|job| job.logical_key == logical_key) {
            return Ok(EnqueueJobResponse {
                job_id: existing.view.job_id.clone(),
                state: existing.view.state.clone(),
                deduplicated: true,
            });
        }
        validate_lineage(&state, &request)?;

        let now = now_ms();
        let job_id = format!("job-{:016x}", state.next_job_id);
        state.next_job_id += 1;

        for job in &mut state.jobs {
            if job.view.request.repository == request.repository
                && job.view.request.pull_request == request.pull_request
                && job.view.request.head_sha != request.head_sha
                && !job.view.state.is_terminal()
            {
                match job.view.state {
                    JobState::Queued => {
                        job.view.state = JobState::Superseded;
                        job.view.updated_at_ms = now;
                    }
                    JobState::Leased | JobState::Running => {}
                    _ => continue,
                }
                job.view.superseded_by = Some(job_id.clone());
            }
        }

        let kind = request.kind();
        state.jobs.push(StoredJob {
            view: JobView {
                job_id: job_id.clone(),
                kind,
                state: JobState::Queued,
                request,
                attempts: 0,
                created_at_ms: now,
                updated_at_ms: now,
                superseded_by: None,
                result: None,
                analysis: None,
            },
            logical_key,
            lease: None,
        });
        persist(&self.path, &state)?;
        Ok(EnqueueJobResponse {
            job_id,
            state: JobState::Queued,
            deduplicated: false,
        })
    }

    pub fn lease(&self, worker_id: &str, hardware_profile: &str) -> Result<Option<LeasedJob>> {
        if worker_id.is_empty() || worker_id.len() > 128 {
            bail!("invalid worker_id");
        }
        if hardware_profile.is_empty() || hardware_profile.len() > 64 {
            bail!("invalid hardware_profile");
        }

        let mut state = self
            .inner
            .lock()
            .map_err(|_| anyhow!("queue lock poisoned"))?;
        let expired = expire_leases(&mut state, self.max_attempts);
        let already_active = state.jobs.iter().any(|job| {
            matches!(job.view.state, JobState::Leased | JobState::Running)
                && job
                    .lease
                    .as_ref()
                    .is_some_and(|lease| lease.worker_id == worker_id)
        });
        if already_active {
            if expired {
                persist(&self.path, &state)?;
            }
            bail!("worker already has an active lease");
        }

        let next_index = state.jobs.iter().position(|job| {
            job.view.state == JobState::Queued
                && job.view.request.hardware_profile == hardware_profile
        });
        let Some(index) = next_index else {
            if expired {
                persist(&self.path, &state)?;
            }
            return Ok(None);
        };

        let lease_id = format!("lease-{:016x}", state.next_lease_id);
        state.next_lease_id += 1;
        let expires_at_ms = now_ms().saturating_add(self.lease_duration_ms);
        let job = &mut state.jobs[index];
        job.view.state = JobState::Leased;
        job.view.attempts += 1;
        job.view.updated_at_ms = now_ms();
        job.lease = Some(Lease {
            lease_id: lease_id.clone(),
            worker_id: worker_id.to_owned(),
            expires_at_ms,
        });
        let response = LeasedJob {
            job_id: job.view.job_id.clone(),
            lease_id,
            lease_expires_at_ms: expires_at_ms,
            kind: job.view.kind.clone(),
            request: job.view.request.clone(),
        };
        persist(&self.path, &state)?;
        Ok(Some(response))
    }

    pub fn heartbeat(&self, job_id: &str, lease_id: &str) -> Result<JobView> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| anyhow!("queue lock poisoned"))?;
        expire_leases(&mut state, self.max_attempts);
        let now = now_ms();
        let job = find_job_mut(&mut state, job_id)?;
        let lease = require_lease(job, lease_id)?;
        lease.expires_at_ms = now.saturating_add(self.lease_duration_ms);
        job.view.state = JobState::Running;
        job.view.updated_at_ms = now;
        let view = job.view.clone();
        persist(&self.path, &state)?;
        Ok(view)
    }

    pub fn complete(&self, job_id: &str, upload: JobResultUpload) -> Result<(JobView, bool)> {
        upload.validate().map_err(|message| anyhow!(message))?;
        let mut state = self
            .inner
            .lock()
            .map_err(|_| anyhow!("queue lock poisoned"))?;
        let now = now_ms();
        let job = find_job_mut(&mut state, job_id)?;

        if job.view.state.is_terminal() {
            if job.view.result.as_ref() == Some(&upload) {
                return Ok((job.view.clone(), false));
            }
            bail!("job is already terminal");
        }
        let _ = require_lease(job, &upload.lease_id)?;
        job.view.state = if job.view.superseded_by.is_some() {
            JobState::Superseded
        } else if upload.success {
            JobState::Succeeded
        } else if upload.failure_category == Some(FailureCategory::Timeout) {
            JobState::TimedOut
        } else {
            JobState::Failed
        };
        job.view.updated_at_ms = now;
        job.view.result = Some(upload);
        job.lease = None;
        let view = job.view.clone();
        persist(&self.path, &state)?;
        Ok((view, true))
    }

    pub fn get(&self, job_id: &str) -> Result<JobView> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| anyhow!("queue lock poisoned"))?;
        if expire_leases(&mut state, self.max_attempts) {
            persist(&self.path, &state)?;
        }
        state
            .jobs
            .iter()
            .find(|job| job.view.job_id == job_id)
            .map(|job| job.view.clone())
            .ok_or_else(|| anyhow!("job not found"))
    }

    pub fn set_analysis(&self, job_id: &str, analysis: serde_json::Value) -> Result<()> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| anyhow!("queue lock poisoned"))?;
        let job = find_job_mut(&mut state, job_id)?;
        let existing_is_final = job
            .view
            .analysis
            .as_ref()
            .and_then(|value| value.get("verification_completed"))
            .and_then(serde_json::Value::as_bool)
            == Some(true);
        let incoming_is_final = analysis
            .get("verification_completed")
            .and_then(serde_json::Value::as_bool)
            == Some(true);
        if existing_is_final && !incoming_is_final {
            return Ok(());
        }
        job.view.analysis = Some(analysis);
        job.view.updated_at_ms = now_ms();
        persist(&self.path, &state)
    }
}

fn validate_lineage(state: &PersistentState, request: &EnqueueJobRequest) -> Result<()> {
    let Some(parent_id) = &request.parent_job_id else {
        return Ok(());
    };
    let parent = state
        .jobs
        .iter()
        .find(|job| &job.view.job_id == parent_id)
        .ok_or_else(|| anyhow!("parent job not found"))?;
    if parent.view.kind != JobKind::Benchmark {
        bail!("verification depth is limited to one");
    }
    if !parent.view.state.is_terminal() || parent.view.result.is_none() {
        bail!("verification requires a completed deterministic parent result");
    }
    if parent.view.request.repository != request.repository
        || parent.view.request.pull_request != request.pull_request
        || parent.view.request.head_sha != request.head_sha
        || parent.view.request.hardware_profile != request.hardware_profile
        || parent.view.request.benchmark_image != request.benchmark_image
        || parent.view.request.source != request.source
        || parent.view.request.pull_request_context != request.pull_request_context
    {
        bail!("verification job must inherit immutable parent inputs");
    }
    let child_count = state
        .jobs
        .iter()
        .filter(|job| job.view.request.parent_job_id.as_ref() == Some(parent_id))
        .count();
    if child_count >= 1 {
        bail!("verification child budget exhausted");
    }
    Ok(())
}

fn find_job_mut<'a>(state: &'a mut PersistentState, job_id: &str) -> Result<&'a mut StoredJob> {
    state
        .jobs
        .iter_mut()
        .find(|job| job.view.job_id == job_id)
        .ok_or_else(|| anyhow!("job not found"))
}

fn require_lease<'a>(job: &'a mut StoredJob, lease_id: &str) -> Result<&'a mut Lease> {
    if !matches!(job.view.state, JobState::Leased | JobState::Running) {
        bail!("job has no active lease");
    }
    let lease = job.lease.as_mut().ok_or_else(|| anyhow!("missing lease"))?;
    if lease.lease_id != lease_id {
        bail!("lease does not match");
    }
    if lease.expires_at_ms <= now_ms() {
        bail!("lease expired");
    }
    Ok(lease)
}

fn expire_leases(state: &mut PersistentState, max_attempts: u32) -> bool {
    let now = now_ms();
    let mut changed = false;
    for job in &mut state.jobs {
        let expired = job
            .lease
            .as_ref()
            .is_some_and(|lease| lease.expires_at_ms <= now);
        if expired && matches!(job.view.state, JobState::Leased | JobState::Running) {
            job.lease = None;
            job.view.updated_at_ms = now;
            if job.view.superseded_by.is_some() {
                job.view.state = JobState::Superseded;
            } else if job.view.attempts < max_attempts {
                job.view.state = JobState::Queued;
            } else {
                job.view.state = JobState::Failed;
            }
            changed = true;
        }
    }
    changed
}

fn logical_key(request: &EnqueueJobRequest) -> Result<String> {
    serde_json::to_string(request).context("failed to create logical job key")
}

fn persist(path: &Path, state: &PersistentState) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let temp = path.with_extension("json.tmp");
    let mut bytes = serde_json::to_vec_pretty(state).context("failed to encode queue state")?;
    bytes.push(b'\n');
    fs::write(&temp, bytes).with_context(|| format!("failed to write {}", temp.display()))?;
    fs::rename(&temp, path).with_context(|| format!("failed to replace {}", path.display()))?;
    Ok(())
}

fn now_ms() -> u64 {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    u64::try_from(millis).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;
    use benchmark_contracts::{
        API_SCHEMA_VERSION, BenchmarkOptions, EnqueueJobRequest, JobResultUpload, JobState,
        SourceBundle, VerificationSpec, VerificationTemplate,
    };
    use serde_json::json;

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(1);

    fn request(head: char) -> EnqueueJobRequest {
        EnqueueJobRequest {
            schema_version: API_SCHEMA_VERSION,
            repository: "owner/repo".to_owned(),
            pull_request: 7,
            head_sha: head.to_string().repeat(40),
            benchmark_version: "v1".to_owned(),
            hardware_profile: "rtx-4070".to_owned(),
            benchmark_image: format!("bench@sha256:{}", "f".repeat(64)),
            source: SourceBundle {
                url: "https://example.invalid/source.tar.gz".to_owned(),
                sha256: "e".repeat(64),
                strip_components: 0,
            },
            pull_request_context: None,
            options: BenchmarkOptions::default(),
            parent_job_id: None,
            verification: None,
        }
    }

    fn store() -> (PathBuf, Store) {
        let dir = std::env::temp_dir().join(format!(
            "techjam-store-test-{}-{}-{}",
            std::process::id(),
            now_ms(),
            TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).expect("create test directory");
        let store = Store::open(dir.join("state.json"), 60_000, 2).expect("store");
        (dir, store)
    }

    #[test]
    fn enqueue_is_idempotent() {
        let (dir, store) = store();
        let first = store.enqueue(request('a')).expect("first enqueue");
        let second = store.enqueue(request('a')).expect("second enqueue");
        assert_eq!(first.job_id, second.job_id);
        assert!(second.deduplicated);
        drop(store);
        fs::remove_dir_all(dir).expect("remove test directory");
    }

    #[test]
    fn worker_can_lease_only_one_job() {
        let (dir, store) = store();
        store.enqueue(request('a')).expect("enqueue first");
        let mut second = request('b');
        second.pull_request = 8;
        store.enqueue(second).expect("enqueue second");
        assert!(store.lease("worker", "rtx-4070").expect("lease").is_some());
        assert!(store.lease("worker", "rtx-4070").is_err());
        drop(store);
        fs::remove_dir_all(dir).expect("remove test directory");
    }

    #[test]
    fn new_sha_supersedes_queued_old_sha() {
        let (dir, store) = store();
        let old = store.enqueue(request('a')).expect("enqueue old");
        let new = store.enqueue(request('b')).expect("enqueue new");
        let old_view = store.get(&old.job_id).expect("old view");
        assert_eq!(old_view.state, JobState::Superseded);
        assert_eq!(old_view.superseded_by.as_deref(), Some(new.job_id.as_str()));
        drop(store);
        fs::remove_dir_all(dir).expect("remove test directory");
    }

    #[test]
    fn verification_is_allowlisted_and_bounded_to_one_child() {
        let (dir, store) = store();
        let parent = store.enqueue(request('a')).expect("enqueue parent");
        let lease = store
            .lease("worker", "rtx-4070")
            .expect("lease parent")
            .expect("parent job");
        store
            .complete(
                &parent.job_id,
                JobResultUpload {
                    lease_id: lease.lease_id,
                    success: true,
                    failure_category: None,
                    benchmark_result: json!({"schema_version": 1}),
                    log_excerpt: String::new(),
                },
            )
            .expect("complete parent");

        let mut child = request('a');
        child.parent_job_id = Some(parent.job_id.clone());
        child.verification = Some(VerificationSpec {
            template: VerificationTemplate::RepeatNoisyCase,
            reason: "timing variance".to_owned(),
            expected_evidence: "additional samples".to_owned(),
        });
        store.enqueue(child.clone()).expect("enqueue child");
        child.verification = Some(VerificationSpec {
            template: VerificationTemplate::StricterAccuracy,
            reason: "accuracy edge".to_owned(),
            expected_evidence: "stricter comparison".to_owned(),
        });
        assert!(store.enqueue(child).is_err());
        drop(store);
        fs::remove_dir_all(dir).expect("remove test directory");
    }

    #[test]
    fn duplicate_completion_is_identified_as_idempotent() {
        let (dir, store) = store();
        let enqueued = store.enqueue(request('a')).expect("enqueue");
        let lease = store
            .lease("worker", "rtx-4070")
            .expect("lease")
            .expect("job");
        let upload = JobResultUpload {
            lease_id: lease.lease_id,
            success: true,
            failure_category: None,
            benchmark_result: json!({"schema_version": 1}),
            log_excerpt: String::new(),
        };
        let (_, first) = store
            .complete(&enqueued.job_id, upload.clone())
            .expect("first completion");
        let (_, second) = store
            .complete(&enqueued.job_id, upload)
            .expect("duplicate completion");
        assert!(first);
        assert!(!second);
        drop(store);
        fs::remove_dir_all(dir).expect("remove test directory");
    }

    #[test]
    fn initial_analysis_cannot_overwrite_final_verification_review() {
        let (dir, store) = store();
        let enqueued = store.enqueue(request('a')).expect("enqueue");
        store
            .set_analysis(
                &enqueued.job_id,
                serde_json::json!({"summary": "final", "verification_completed": true}),
            )
            .expect("store final analysis");
        store
            .set_analysis(
                &enqueued.job_id,
                serde_json::json!({"summary": "stale initial"}),
            )
            .expect("ignore stale initial analysis");
        let job = store.get(&enqueued.job_id).expect("get job");
        assert_eq!(
            job.analysis
                .as_ref()
                .and_then(|value| value.get("summary"))
                .and_then(serde_json::Value::as_str),
            Some("final")
        );
        drop(store);
        fs::remove_dir_all(dir).expect("remove test directory");
    }
}
