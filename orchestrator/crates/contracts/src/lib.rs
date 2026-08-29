use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const API_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    Benchmark,
    Verification,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Leased,
    Running,
    Succeeded,
    Failed,
    Cancelled,
    TimedOut,
    Superseded,
}

impl JobState {
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::Cancelled | Self::TimedOut | Self::Superseded
        )
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceBundle {
    pub url: String,
    pub sha256: String,
    #[serde(default)]
    pub strip_components: u8,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactUploadResponse {
    pub url: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PullRequestContext {
    pub title: String,
    #[serde(default)]
    pub body: String,
    pub author: String,
    pub html_url: String,
    pub base_ref: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkCase {
    pub batch_size: u32,
    pub seq_len: u32,
    pub d_model: u32,
    pub heads: u32,
    pub ffn_dim: u32,
    pub layers: u32,
    #[serde(default)]
    pub causal: bool,
    #[serde(default)]
    pub padding_ratio: f64,
}

impl Default for BenchmarkCase {
    fn default() -> Self {
        Self {
            batch_size: 8,
            seq_len: 128,
            d_model: 512,
            heads: 8,
            ffn_dim: 2048,
            layers: 6,
            causal: false,
            padding_ratio: 0.0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkOptions {
    #[serde(default)]
    pub case: BenchmarkCase,
    #[serde(default = "default_dtype")]
    pub dtype: String,
    #[serde(default = "default_accuracy_trials")]
    pub accuracy_trials: u32,
    #[serde(default = "default_rtol")]
    pub rtol: f64,
    #[serde(default = "default_atol")]
    pub atol: f64,
    #[serde(default = "default_warmup")]
    pub warmup: u32,
    #[serde(default = "default_repeats")]
    pub repeats: u32,
    #[serde(default = "default_rounds")]
    pub benchmark_rounds: u32,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

impl Default for BenchmarkOptions {
    fn default() -> Self {
        Self {
            case: BenchmarkCase::default(),
            dtype: default_dtype(),
            accuracy_trials: default_accuracy_trials(),
            rtol: default_rtol(),
            atol: default_atol(),
            warmup: default_warmup(),
            repeats: default_repeats(),
            benchmark_rounds: default_rounds(),
            seed: default_seed(),
        }
    }
}

fn default_dtype() -> String {
    "bfloat16".to_owned()
}
const fn default_accuracy_trials() -> u32 {
    5
}
const fn default_rtol() -> f64 {
    0.02
}
const fn default_atol() -> f64 {
    0.002
}
const fn default_warmup() -> u32 {
    20
}
const fn default_repeats() -> u32 {
    100
}
const fn default_rounds() -> u32 {
    3
}
const fn default_seed() -> u64 {
    1234
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationTemplate {
    RepeatNoisyCase,
    StricterAccuracy,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationSpec {
    pub template: VerificationTemplate,
    pub reason: String,
    pub expected_evidence: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EnqueueJobRequest {
    #[serde(default = "schema_version")]
    pub schema_version: u32,
    pub repository: String,
    pub pull_request: u64,
    pub head_sha: String,
    pub benchmark_version: String,
    pub hardware_profile: String,
    pub benchmark_image: String,
    pub source: SourceBundle,
    #[serde(default)]
    pub pull_request_context: Option<PullRequestContext>,
    #[serde(default)]
    pub options: BenchmarkOptions,
    #[serde(default)]
    pub parent_job_id: Option<String>,
    #[serde(default)]
    pub verification: Option<VerificationSpec>,
}

impl EnqueueJobRequest {
    #[must_use]
    pub const fn kind(&self) -> JobKind {
        if self.verification.is_some() {
            JobKind::Verification
        } else {
            JobKind::Benchmark
        }
    }

    /// Validates all untrusted fields and execution bounds.
    ///
    /// # Errors
    ///
    /// Returns a human-readable contract violation.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != API_SCHEMA_VERSION {
            return Err("unsupported schema_version".to_owned());
        }
        if !valid_repository(&self.repository) {
            return Err("repository must have owner/name form".to_owned());
        }
        if self.pull_request == 0 {
            return Err("pull_request must be positive".to_owned());
        }
        if !valid_hex(&self.head_sha, 40) && !valid_hex(&self.head_sha, 64) {
            return Err("head_sha must be a full 40 or 64 character hex digest".to_owned());
        }
        if self.benchmark_version.is_empty() || self.benchmark_version.len() > 128 {
            return Err("invalid benchmark_version".to_owned());
        }
        if self.hardware_profile.is_empty() || self.hardware_profile.len() > 64 {
            return Err("invalid hardware_profile".to_owned());
        }
        if !valid_image(&self.benchmark_image) {
            return Err("benchmark_image must be pinned with @sha256:<64 hex>".to_owned());
        }
        if !valid_hex(&self.source.sha256, 64) {
            return Err("source.sha256 must be 64 hexadecimal characters".to_owned());
        }
        if !valid_source_url(&self.source.url) {
            return Err("source.url must use HTTPS, file://, or loopback HTTP".to_owned());
        }
        if self.source.strip_components > 1 {
            return Err("source.strip_components must be 0 or 1".to_owned());
        }
        if let Some(context) = &self.pull_request_context {
            validate_pull_request_context(context)?;
        }
        validate_options(&self.options)?;
        match (&self.parent_job_id, &self.verification) {
            (None, None) | (Some(_), Some(_)) => {}
            _ => {
                return Err(
                    "parent_job_id and verification must either both be set or both be absent"
                        .to_owned(),
                );
            }
        }
        if let Some(spec) = &self.verification {
            if spec.reason.is_empty() || spec.reason.len() > 1_000 {
                return Err("verification reason must contain 1..=1000 characters".to_owned());
            }
            if spec.expected_evidence.is_empty() || spec.expected_evidence.len() > 1_000 {
                return Err("expected_evidence must contain 1..=1000 characters".to_owned());
            }
        }
        Ok(())
    }
}

fn validate_pull_request_context(context: &PullRequestContext) -> Result<(), String> {
    if context.title.is_empty() || context.title.len() > 512 {
        return Err("pull_request_context.title must contain 1..=512 bytes".to_owned());
    }
    if context.body.len() > 16 * 1024 {
        return Err("pull_request_context.body exceeds 16 KiB".to_owned());
    }
    if context.author.is_empty() || context.author.len() > 128 {
        return Err("invalid pull_request_context.author".to_owned());
    }
    if context.base_ref.is_empty() || context.base_ref.len() > 256 {
        return Err("invalid pull_request_context.base_ref".to_owned());
    }
    if context.html_url.len() > 2_048 || !context.html_url.starts_with("https://") {
        return Err("pull_request_context.html_url must use HTTPS".to_owned());
    }
    Ok(())
}

fn valid_source_url(value: &str) -> bool {
    value.starts_with("https://")
        || value.starts_with("file://")
        || value.starts_with("http://127.0.0.1:")
        || value.starts_with("http://localhost:")
}

fn validate_options(options: &BenchmarkOptions) -> Result<(), String> {
    let case = &options.case;
    if case.batch_size == 0
        || case.seq_len == 0
        || case.d_model == 0
        || case.heads == 0
        || case.ffn_dim == 0
        || case.layers == 0
    {
        return Err("all benchmark dimensions must be positive".to_owned());
    }
    if !case.d_model.is_multiple_of(case.heads) {
        return Err("d_model must be divisible by heads".to_owned());
    }
    if !(0.0..1.0).contains(&case.padding_ratio) {
        return Err("padding_ratio must be in [0, 1)".to_owned());
    }
    if !matches!(options.dtype.as_str(), "float32" | "float16" | "bfloat16") {
        return Err("dtype is not allowlisted".to_owned());
    }
    if options.accuracy_trials == 0
        || options.repeats == 0
        || options.benchmark_rounds == 0
        || options.warmup > 10_000
        || options.repeats > 10_000
        || options.benchmark_rounds > 100
    {
        return Err("benchmark iteration counts are out of bounds".to_owned());
    }
    if !options.rtol.is_finite()
        || !options.atol.is_finite()
        || options.rtol < 0.0
        || options.atol < 0.0
    {
        return Err("accuracy tolerances must be finite and non-negative".to_owned());
    }
    Ok(())
}

fn valid_repository(value: &str) -> bool {
    value.len() <= 200
        && value.split_once('/').is_some_and(|(owner, name)| {
            !owner.is_empty()
                && !name.is_empty()
                && !name.contains('/')
                && value.bytes().all(|byte| {
                    byte.is_ascii_alphanumeric() || b"-_.".contains(&byte) || byte == b'/'
                })
        })
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn valid_image(value: &str) -> bool {
    value
        .rsplit_once("@sha256:")
        .is_some_and(|(name, digest)| !name.is_empty() && valid_hex(digest, 64))
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LeaseRequest {
    pub worker_id: String,
    pub hardware_profile: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct LeasedJob {
    pub job_id: String,
    pub lease_id: String,
    pub lease_expires_at_ms: u64,
    pub kind: JobKind,
    pub request: EnqueueJobRequest,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LeaseActionRequest {
    pub lease_id: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureCategory {
    Correctness,
    Build,
    Timeout,
    Infrastructure,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JobResultUpload {
    pub lease_id: String,
    pub success: bool,
    #[serde(default)]
    pub failure_category: Option<FailureCategory>,
    pub benchmark_result: Value,
    #[serde(default)]
    pub log_excerpt: String,
}

impl JobResultUpload {
    /// Validates result consistency and upload size bounds.
    ///
    /// # Errors
    ///
    /// Returns a human-readable contract violation.
    pub fn validate(&self) -> Result<(), String> {
        if self.lease_id.is_empty() || self.lease_id.len() > 128 {
            return Err("invalid lease_id".to_owned());
        }
        if self.success == self.failure_category.is_some() {
            return Err(
                "failure_category must be present exactly when success is false".to_owned(),
            );
        }
        if self.log_excerpt.len() > 64 * 1024 {
            return Err("log_excerpt exceeds 64 KiB".to_owned());
        }
        if !self.benchmark_result.is_object() {
            return Err("benchmark_result must be a JSON object".to_owned());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EnqueueJobResponse {
    pub job_id: String,
    pub state: JobState,
    pub deduplicated: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct JobView {
    pub job_id: String,
    pub kind: JobKind,
    pub state: JobState,
    pub request: EnqueueJobRequest,
    pub attempts: u32,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    pub superseded_by: Option<String>,
    pub result: Option<JobResultUpload>,
    pub analysis: Option<Value>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ApiError {
    pub error: String,
}

#[must_use]
pub const fn schema_version() -> u32 {
    API_SCHEMA_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_request() -> EnqueueJobRequest {
        EnqueueJobRequest {
            schema_version: API_SCHEMA_VERSION,
            repository: "owner/repo".to_owned(),
            pull_request: 1,
            head_sha: "a".repeat(40),
            benchmark_version: "v1".to_owned(),
            hardware_profile: "rtx-4070".to_owned(),
            benchmark_image: format!("benchmark@sha256:{}", "b".repeat(64)),
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

    #[test]
    fn accepts_a_pinned_valid_job() {
        assert_eq!(valid_request().validate(), Ok(()));
    }

    #[test]
    fn benchmark_defaults_match_current_policy() {
        let options = BenchmarkOptions::default();
        assert_eq!(options.dtype, "bfloat16");
        assert_eq!(options.rtol, 0.02);
        assert_eq!(options.atol, 0.002);
    }

    #[test]
    fn rejects_unpinned_image() {
        let mut request = valid_request();
        request.benchmark_image = "benchmark:latest".to_owned();
        assert!(request.validate().is_err());
    }

    #[test]
    fn verification_requires_parent() {
        let mut request = valid_request();
        request.verification = Some(VerificationSpec {
            template: VerificationTemplate::RepeatNoisyCase,
            reason: "variance".to_owned(),
            expected_evidence: "lower variance".to_owned(),
        });
        assert!(request.validate().is_err());
    }
}
