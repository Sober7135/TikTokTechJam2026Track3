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
## FFN-out E01 - BF16 output projection plus residual fusion

- Hypothesis: official BF16 cases 1/3/4/5/9/10/11/12 all use `D=FFN=128`
  and materialize the second FFN projection before a separate residual add in
  each of four layers. A candidate-only Triton epilogue can eliminate that
  projection tensor's global-memory round trip and the add kernel while
  preserving the native numerical boundaries. The preregistered promotion
  gate was strict correctness on every dispatched case, candidate-latency
  geomean improvement of at least 1%, and no case regression over 0.5%.
- Diff: `fused_ffn_out.py` performs the 128-wide BF16 projection, explicitly
  rounds `FP32 dot + bias` to BF16, then adds the BF16 residual in FP32 and
  stores BF16. Exact-shape dispatch is limited to `(4,128,128)`,
  `(16,128,128)`, `(64,32,128)`, `(64,128,128)`, and `(128,128,128)`.
  Case 2 is excluded because its 128 rows do not amortize this custom GEMM;
  cases 6/8/13/14 and every non-CUDA, non-BF16, masked, training, grad,
  non-contiguous, and undeclared shape retain the native fallback. Baseline,
  weights, interface, attention, LayerNorm, masks, and CUDA Graph output clone
  are unchanged.
- Pre-GPU checks: `git diff --check`; `py_compile` for `fused_ffn_out.py`,
  `models.py`, and `test_models.py`; all six model unit tests; and the required
  CPU BF16 smoke all passed. The smoke was bitwise exact (`0 / 128` failures);
  its timing is not GPU evidence.
- Initial job `job-1788123610238-d745d13638fbf43c` used snapshot
  `f6983b8e54d866d795611139e20e13c60774adc14b985fddb8eaf4e27d8c40e5`
  but selected the newly created, empty worktree `.venv`. It failed before
  benchmark import with `ModuleNotFoundError: No module named 'torch'`, wrote
  no `result.json`, and provides no code correctness or performance evidence.
- Focused follow-up job `job-1788123704402-1f8147875a951f05` reused the exact
  same snapshot with the pinned root Python 3.12.14 environment. It ran on an
  RTX 4070 with PyTorch 2.13.0+cu130, CUDA 13.0, and BF16; requested and
  completed cases 1/3/4/5/9/10/11/12 with 5 accuracy trials, 20 warmups, 100
  repeats, and 3 alternating rounds.
- Correctness: PASS and bitwise exact for all 40 case trials. Across
  `34,406,400` validated output elements there were zero failures and zero
  maximum absolute or relative error. This total is the sum of the eight
  per-case `accuracy.total_elements` fields in the structured result.
- Current candidate medians and same-job speedups:
  - case 1: `0.732160 ms`, `1.944056x`;
  - case 3: `0.128000 ms`, `7.456000x`;
  - case 4: `0.274432 ms`, `3.469799x`;
  - case 5: `1.469440 ms`, `2.195819x`;
  - case 9: `0.525312 ms`, `1.653021x`;
  - case 10: `0.638976 ms`, `1.732372x`;
  - case 11: `1.306624 ms`, `5.575235x`;
  - case 12: `0.210944 ms`, `4.478762x`.
- Candidate round medians, derived in recorded order from each set of 300 raw
  samples (three contiguous 100-sample rounds), were: case 1
  `0.732160 / 0.732160 / 0.733184`; case 3
  `0.128000 / 0.128000 / 0.128000`; case 4
  `0.274432 / 0.274432 / 0.274432`; case 5
  `1.472512 / 1.468416 / 1.461760`; case 9
  `0.524288 / 0.526336 / 0.524288`; case 10
  `0.637952 / 0.638976 / 0.638976`; case 11
  `1.304576 / 1.306624 / 1.308672`; and case 12
  `0.210944 / 0.210944 / 0.210944 ms`. Full raw samples remain in the job's
  structured `result.json`.
- Against canonical shared-winner job
  `job-1788121832512-dc0a634f40e6600f`, candidate median improvements were
  3.07693%, 4.00000%, 3.35821%, 0.55750%, 4.67836%, 4.16667%, 2.66457%, and
  4.36893% for cases 1/3/4/5/9/10/11/12. The equal-case latency geomean
  improvement was 3.35143%; no case regressed.
- Decision: `promote`. Correctness passed exactly and both preregistered
  performance gates were satisfied. This is focused evidence for the complete
  dispatch scope, not a full official-matrix claim.

## I02 - unified FFN-out/residual winner

- Integrated commit: `68d4f503d5cd994faf5393ed842e9c5bfed9593f`, preserving the
  independent FFN-out optimization body and implementation from promoted
  focused commit `c47546d53eddf853ad35f7065a602f164cf4ed50`.
- Static integration checks: clean worktree, `git diff --check`, full Python
  compilation, 11/11 unit tests, and required CPU BF16 smoke passed; CPU smoke
  was bitwise exact (`0 / 128`) and is not GPU performance evidence.
- Unified GPU job/snapshot: `job-1788123970631-bca84587ca0e6e73` /
  `2d631300de60f00f7a583c85f6dbeb746b0546b2ec28021312d00bd62c341ed8`;
  CUDA BF16 official cases 1-13 on RTX 4070, PyTorch 2.13.0+cu130, CUDA 13.0.
- Correctness: all 13 requested cases executed and passed strict correctness
  bitwise exact over five trials; `0 / 933,232,640` failed elements, with zero
  maximum absolute and relative error.
- Per-case same-job speedups for cases 1-13:
  `1.943977x / 9.666341x / 7.442250x / 3.506063x / 2.195258x /
  2.390277x / 2.203285x / 1.070499x / 1.666667x / 1.734824x /
  5.540467x / 4.516078x / 3.114174x`.
- Relative to prior unified winner job
  `job-1788121832512-dc0a634f40e6600f`, candidate median changes for cases
  1-13 were `-3.2213% / 0.0000% / -4.0000% / -3.3582% / -0.6276% /
  +0.0326% / 0.0000% / +0.0375% / -4.6784% / -3.8339% / -1.9455% /
  -3.8647% / +0.0100%`; negative means faster. Non-dispatch movements are
  bounded cross-job timing noise.
- Aggregate result: equal-case speedup geomean `2.964738516x`, one-call-total
  speedup `2.462056400x` (`558.922880 / 227.014653 ms`), and supervisor-derived
  aggregate MFU `15.258284896%` using 58.25 TFLOP/s and
  `F=L*(8BSD^2 + 4BS^2D + 4BSDF)`.
- Decision: promote as the new shared winner. Versus the prior unified winner,
  equal-case geomean rose `1.9984%`, total candidate latency fell `0.0289%`,
  and aggregate MFU rose `0.004407` percentage points.

## E02 - Attention output projection plus residual

- Status: promoted as an independent winner; combination with the independent
  FFN-output winner still requires a unified integration job.
- Parent: auditable shared winner commit `7c6ab1c8371024c7c4743f9539221ee536464a04`.
- Hypothesis: official BF16 cases 1, 3, 4, 5, 9, 10, 11, and 12 each
  materialize a 128-wide attention output projection and then launch a separate
  residual add in every Transformer layer. Removing that launch and intermediate
  write should reduce latency across the shared shape family.
- Implementation: a candidate-only Triton 128-by-128 output projection consumes
  the BF16 residual in its epilogue. The internal optimized-attention call can
  accept the block residual, while the public
  `UserOptimizedTransformer.forward(x, valid_token_mask)` contract is unchanged.
- Numerical boundary: the dot product and bias accumulate in FP32, the
  projection is explicitly rounded to BF16 at the native `nn.Linear`
  materialization boundary, and only then is the BF16 residual added before the
  final BF16 store. This is not the rejected beta-times-C shortcut and does not
  replace softmax or LayerNorm reductions.
- Dispatch: exact context/residual shapes `(4,128,128)`, `(16,128,128)`,
  `(64,32,128)`, `(64,128,128)`, and `(128,128,128)` under CUDA BF16 inference,
  no grad, no padding mask, 128-by-128 output weights, and contiguous tensors.
  These are cases 3, 4, 12, 1/9/10/11, and 5 respectively.
- Fallback: cases 2, 6, 7, 8, 13, and 14, plus training, grad-enabled,
  non-inference, CPU, non-BF16, padded-mask, undeclared-shape, non-128, or
  non-contiguous calls retain native `out_proj` followed by residual addition.
  Baseline source, weight copying, masks, and CUDA Graph output cloning are
  unchanged.
- Local validation: `git diff --check` and Python compilation passed; all 7
  model unit tests passed; the required CPU BF16 smoke was bitwise exact with
  `0 / 128` failed elements. CPU timing is not GPU performance evidence.

### Focused GPU evidence

- Job: `job-1788124074308-3c724497113d7eb1`.
- Submitted source commit: `87d245933ab691fa1d10f5cea9ad509d65209716`.
- Immutable snapshot:
  `3720e7c4edb66df141a6c32ee9b86cb0d6a6f318e3a55d09a32c4233d67efbc3`.
- Arguments: CUDA BF16 official cases `1 3 4 5 9 10 11 12`; five accuracy
  trials, 20 warmups, 100 repeats, and three alternating benchmark rounds.
- Environment: NVIDIA GeForce RTX 4070, Python 3.12.14, PyTorch
  2.13.0+cu130, CUDA 13.0, matmul precision `high`, and TF32 enabled equally
  for baseline and candidate.
- Deterministic result: state `succeeded`, exit code 0, complete matrix subset,
  and `correctness_passed=true`. All 40 trials were bitwise exact: zero failures
  over 34,406,400 elements and maximum absolute/relative errors both zero.
- Raw structured result:
  `.benchmarkctl/jobs/job-1788124074308-3c724497113d7eb1/result.json`.

| Case | Baseline median (ms) | Candidate median (ms) | Same-job speedup | Improvement vs shared winner |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.420288 | 0.729088 | 1.948034x | 3.51124% |
| 3 | 0.935936 | 0.128000 | 7.311999x | 3.999995% |
| 4 | 0.947200 | 0.273712 | 3.460572x | 3.63009% |
| 5 | 3.225600 | 1.432576 | 2.251608x | 3.14511% |
| 9 | 0.867328 | 0.524288 | 1.654297x | 4.88282% |
| 10 | 1.103872 | 0.634880 | 1.738710x | 4.83871% |
| 11 | 7.285344 | 1.304576 | 5.584453x | 2.82574% |
| 12 | 0.938704 | 0.209920 | 4.471722x | 4.87804% |

- The preregistered comparison uses the common-parent shared-winner job
  `job-1788121832512-dc0a634f40e6600f`. Equal-case candidate latency geomean
  improved by 3.96113%, and every target improved, satisfying the required
  geomean-at-least-1% and no-regression-over-0.5% gates.
- For context only, this attention-only result is 0.7532% faster in equal-case
  geomean than the newer independent FFN-output-only unified job
  `job-1788123970631-bca84587ca0e6e73`. That cross-sibling comparison is not
  combination evidence; both commits must be stacked and validated together.
- Decision: `promote` as an independent optimization winner. No follow-up job
  was submitted from this worktree.
