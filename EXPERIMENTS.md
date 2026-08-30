# Integration experiments

## E00 - d66 contract repair

- Status: promoted as the compliant parent for winner integration.
- Source: verified d66 snapshot
  `d66bc39ef07f5aaf99e4a4648d6feb899cbcaee51189badf390c822e0a567c8f`.
- Hypothesis: isolating candidate-only block dispatch from the measured baseline
  and cloning CUDA Graph outputs restores benchmark/API fairness. This is a
  compliance repair, not an optimization claim.
- Required patch: preserve every d66 optimization except (1) introduce a
  separate `UserOptimizedTransformerBlock`, leaving
  `BaselineTransformerBlock` pristine, and (2) return independent cloned graph
  outputs after capture and replay.
- Source changes:
  - restored a pristine `BaselineTransformerBlock` and moved the existing d66
    cuBLASLt candidate dispatch into `UserOptimizedTransformerBlock`;
  - constructed candidate layers from the candidate-only block without changing
    parameter names, weight-copy compatibility, attention paths, or CUDA Graph
    eligibility;
  - cloned both CUDA Graph capture and replay return values so consecutive
    forwards do not alias;
  - added focused tests for baseline isolation and independent graph outputs.
- Static validation: `git diff --check`, Python compilation, and all 11 tests
  passed; the prescribed CPU smoke was strict bitwise exact (`0 / 128`).
- Unified result: `job-1788120150037-a85f1703de65273a`, snapshot
  `039a3fa58f3e41839e913c83905cb4024c901b503dbc2f2930cf6f6d54b7500b`,
  succeeded for cases 1-13 with bitwise-exact strict correctness
  (`0 / 933,232,640` failures). It establishes compliance and does not promote
  a new optimization.

## I01 - FFN/GELU plus case-6 PV

- Status: integrated; local and unified cases 1-13 validation passed.
- Canonical source snapshot:
  `039a3fa58f3e41839e913c83905cb4024c901b503dbc2f2930cf6f6d54b7500b`.
- Winner A: fused BF16 first-FFN projection plus exact-erf GELU from
  `ffn-layout`, validated bitwise exact and faster across complete dispatch
  scope cases 1/5/9/10/11.
- Winner B: native FP32 softmax plus explicit FP32-to-BF16 Triton PV boundary
  from `attention-pv`, validated bitwise exact and 15.8361% faster on case 6.
- Excluded: both case-8 experiments. E01 was incorrect; E02 missed its
  preregistered performance threshold. Canonical case-8 code remains unchanged.
- Required validation: source-diff audit, static checks, unit tests, CPU smoke,
  then one unified cases 1-13 CUDA BF16 job through shared `benchmarkctl`.

### Integration audit

- `fused_ffn.py` is byte-identical to focused winner snapshot
  `17b32c95ece5c95331b02e99bd235877920885602b885c53fbbbb74d676dc815`.
- `pv_context.py` is byte-identical to focused winner snapshot
  `92edbeba0dd4b3f8ae89758157ed1dab311c0388798764ae8d35d135525e6650`.
- Relative to canonical, `models.py` contains exactly the two expected hunks:
  case-6 PV dispatch and exact-erf first-FFN fusion dispatch.
- `BaselineSelfAttention`, `BaselineTransformerBlock`, `BaselineTransformer`,
  weight copying, CUDA Graph eligibility/capture/replay, and both replay-output
  clones are AST/source-identical to canonical.
- The case-8 candidate cuBLASLt FFN-out dispatch remains source-identical to
  canonical. No beta-residual or packed-QKV case-8 experiment was merged.

### Local validation

- `git diff --check`: passed.
- Python compile check for facade, all package modules, and all tests: passed.
- `python -m unittest discover -s tests -v`: 11/11 passed.
- Required CPU smoke: strict correctness passed bitwise exact, `0 / 128`
  failures. This is execution/correctness evidence only, not GPU performance.

### Unified GPU submission

- Job: `job-1788121832512-dc0a634f40e6600f`.
- Immutable snapshot:
  `25b437a04e4328edfb8166c6ab63eacbd5866f49f35ea0ab061f34369e4a57d7`.
- Arguments: CUDA BF16 official cases 1 through 13, excluding case 14 per the
  supervisor's current campaign scope.
- Result: succeeded on RTX 4070 with PyTorch 2.13.0+cu130/CUDA 13.0; all 13
  requested cases passed strict correctness bitwise exact. Performance is
  interpreted per case only after that verdict.

### Case-11 timing audit

- The focused FFN scope job `job-1788121411455-34668919a74601c8`
  (`1237ca37b67d703aa1b6e7f21d0d3c8b569f09e9e69f56b7a63e59b55df6d2f7`)
  reported baseline/optimized medians `7.281296 / 1.308672 ms`; optimized
  per-round medians were `1.305600, 1.308672, 1.313792 ms`.
- Unified job `job-1788121832512-dc0a634f40e6600f` reported
  `7.286784 / 1.341440 ms`; optimized per-round medians were
  `1.340416, 1.342464, 1.341440 ms`. Relative to canonical optimized
  `1.334272 ms`, this is a `+0.537%` regression, while the same unified run's
  baseline moved only `+0.075%` relative to the focused baseline.
- `fused_ffn.py` is byte-identical between focused and unified snapshots, and
  `UserOptimizedTransformerBlock` is AST/source-identical. The facade, runner,
  triangular-score and cuBLASLt files are also byte-identical. The sole package
  differences are the case-6 PV hunk and `pv_context.py`; its predicate is
  false for case 11 and its captured operation sequence remains the canonical
  fallback.
- Conclusion: no FFN integration-source discrepancy was found. This is
  cross-job/runtime timing drift (or case-order/runtime-state interaction), not
  a source merge regression. The focused promote remains valid evidence, but
  its case-11 gain is not a guaranteed unified-matrix gain; the unified result
  is the authoritative integration measurement and the conflict remains
  explicitly recorded.
