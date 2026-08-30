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
  (`0 / 938,885,120` failures). It establishes compliance and does not promote
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
  bitwise exact over five trials; `0 / 938,885,120` failed elements, with zero
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

## QKV/layout E01 - consume packed V in cases 6 and 10

- Status: focused `promote`; integration and unified cases 1-13 validation are
  still required.
- Parent: audited winner `7c6ab1c8371024c7c4743f9539221ee536464a04`.
- Targets: official CUDA BF16 cases 6 and 10. The packed QKV projection already
  kept Q/K as strided views but copied V into a separate contiguous tensor.
  That copy moved about 312.5 MiB per layer in case 6 and 2 MiB per layer in
  case 10, or about 1.22 GiB and 8 MiB over four layers.
- Implementation: retain the projection-backed `[B,H,S,HD]` V view and extend
  the existing direct-write Triton PV path to consume its explicit strides for
  the matching 64-row case-10 chunks. The packed `nn.Linear` BF16 output, QK
  BF16 dot and scale materialization, native FP32 softmax, explicit FP32-to-BF16
  probability conversion, BF16 PV inputs, FP32 accumulation, and BF16 context
  output boundaries are unchanged.
- Dispatch: eval plus inference/no-grad CUDA BF16, causal, no effective token
  mask, and exact `(B,S,D,H)` shapes `(10000,128,128,4)` or
  `(64,128,128,2)`. Cases 11/13, training, CPU, other dtype/shape/mask paths,
  baseline, weights, interface, scoring, graph output ownership, softmax and
  LayerNorm reductions all retain the prior fallback.
- Static validation: `git diff --check`, full facade/package/test Python
  compilation, 13/13 unit tests, and the prescribed CPU BF16 smoke passed; the
  smoke was bitwise exact (`0 / 128`) and is not GPU performance evidence.
- Preregistered gate: both cases strict-correct; candidate latency geometric
  mean at least 1% faster than common-parent job
  `job-1788121832512-dc0a634f40e6600f`, with neither case more than 0.5%
  slower.
- Focused job/snapshot: `job-1788124124492-cc3fdb44ea8ae7dd` /
  `40dcb9893ed053a11d839fdcf31e67b601930a0fdfa5d34c63a25c7783ad346d`;
  state `succeeded`, exit 0, RTX 4070, PyTorch 2.13.0+cu130/CUDA 13.0.
- Correctness: both requested cases executed. Across 10 trials and
  `824,442,880` elements, strict correctness was bitwise exact: zero failures
  and zero maximum absolute or relative error.
- Raw-derived medians: case 6 baseline/candidate
  `416.684021 / 168.028671 ms` (`2.479838815x`); case 10
  `1.114112 / 0.547840 ms` (`2.033644898x`). Each side contains 300 timing
  samples from 3 rounds of 100 repeats after 20 warmups.
- Common-parent comparison: candidate incremental speedups are `3.708956%`
  for case 6 and `21.495328%` for case 10; their geometric mean is
  `12.250406%`, with no regression. Same-job baseline drift versus the parent
  was only `+0.004424% / +0.272169%`.
- Decision: `promote`. Relative to newer FFN-out unified winner job
  `job-1788123970631-bca84587ca0e6e73`, the QKV-only two-case geometric mean
  remains `10.1765%` faster. This cross-branch comparison is not combined
  evidence; layer QKV E01 with FFN-out and attention-out, then run one unified
  cases 1-13 validation before shared-source promotion.

## I03 - FFN-out, attention-out, and packed-V integration

- Status: promoted as the new shared cases 1-13 winner.
- Source head under test: `f157bcd0d1ba470664d1742f94eb5fb62bfc4fe6`.
  It layers independent FFN-output/residual, attention-output/residual, and
  packed-strided-V optimizations while retaining every experiment's exact-shape
  dispatch and native fallback.
- Static validation: clean diff, full facade/package/test compilation, 14/14
  unit tests, and the required CPU BF16 smoke bitwise exact (`0 / 128`).
- Unified ordinary job/snapshot:
  `job-1788124928473-8295c78ca0925273` /
  `63cd5873b3bf87a0e1cfd66a5e34c6e43a8bd384e54c3db0806bf88743263557`;
  state `succeeded`, exit zero, official CUDA BF16 cases 1-13 on the RTX 4070.
- Correctness: all requested cases executed. All five trials per case were
  bitwise exact: zero failures over `938,885,120` elements and maximum
  absolute/relative errors both zero under the strict OR rule.
- Same-job baseline/candidate medians and speedups for cases 1-13 (ms, x):
  `1.421312/0.706560/2.011594`, `0.956928/0.098304/9.734375`,
  `0.957440/0.130048/7.362204`, `0.962560/0.264192/3.643411`,
  `3.225600/1.415168/2.279305`, `416.669708/168.355843/2.474935`,
  `1.100800/0.498688/2.207392`, `11.701248/10.930688/1.070495`,
  `0.872448/0.500736/1.742331`, `1.110224/0.501008/2.215981`,
  `7.286272/1.288192/5.656200`, `0.958320/0.201728/4.750555`, and
  `111.706623/35.868671/3.114323`.
- Aggregate: equal-case speedup geomean `3.081836314x`; one-call-total speedup
  `2.531844194x` from `558.929484 / 220.759826 ms`; aggregate MFU
  `15.690600597%` using the recorded FLOP model and 58.25 TFLOP/s peak.
- Versus the preceding shared winner job
  `job-1788123970631-bca84587ca0e6e73`, candidate-latency equal-case geomean
  improved `3.868972%`, total candidate latency improved `2.833318%`, and MFU
  increased `0.432316` percentage points.
- Per-case candidate changes versus that preceding winner, positive when
  faster, were `+3.4783%, 0.0000%, -1.5748%, +3.8760%, +3.7627%, +3.5411%,
  0.0000%, +0.0234%, +4.9080%, +27.9469%, +2.1463%, +5.0761%, +0.0057%`.
  Case 3 exposes a small cross-optimization/runtime interaction, but remains
  `2.3622%` faster than the common implementation parent. This is retained as
  an explicit audit item while the exact, broad aggregate and total-call gains
  justify promotion.
- Decision: `promote` all three implementations as the shared winner.

## Cases-1/5 HD32 PV E01 - direct final-layout Triton prefix context

- Status: focused `promote`; ready for shared-winner integration and a new
  unified validation, with no follow-up job from this worktree.
- Parent: verified I03 implementation commit
  `f157bcd0d1ba470664d1742f94eb5fb62bfc4fe6`; comparable unified job
  `job-1788124928473-8295c78ca0925273`, snapshot
  `63cd5873b3bf87a0e1cfd66a5e34c6e43a8bd384e54c3db0806bf88743263557`.
  I03 candidate medians for cases 1/5 are `0.7065600157 / 1.4151680470 ms`.
  Their raw 100-sample round medians are respectively
  `0.706560/0.706560/0.706560 ms` and
  `1.417216/1.416192/1.413120 ms`.
- Exact targets: official CUDA BF16 cases 1 and 5, `(B,S,D,H,HD)` equal to
  `(64,128,128,4,32)` or `(128,128,128,4,32)`. I03 already uses four 32-row
  compact causal score/native FP32-softmax prefixes per layer. It then casts
  each probability tensor to BF16, calls native PV against a projection-backed
  strided V view, materializes four dense prefix contexts, and concatenates
  them. The hypothesis is that the already validated direct-write Triton PV
  boundary from cases 6/10 can avoid the native strided-V/output setup,
  per-prefix cast tensor, and final concatenation while retaining the faster
  I03 32-row schedule.
- Implementation: extend `bf16_probability_value` to the four exact
  `(M,K,N)` tiles `(32,32,32)`, `(32,64,32)`, `(32,96,32)`, and
  `(32,128,32)`. For only the two target query shapes, allocate the existing
  compact sequence-major context backing, write every disjoint prefix through
  the custom kernel, and reuse the already-contiguous `[B,S,H,HD]` transpose
  view for the unchanged attention output projection. QK score generation,
  score scaling, the four-prefix schedule, and native FP32 softmax remain
  unchanged; this is not an online or fused softmax experiment.
- Non-power-of-two K=96 boundary: the Triton tile uses compile-time
  `block_key_count=128`, masks probability and V loads at `key < 96`, and
  supplies exact zero for the 32 padded BF16 entries. Thus the only additional
  products are `+0 * +0`; no valid probability or V term is dropped or read
  out of bounds. All paths still materialize native FP32 softmax first, round
  loaded probabilities explicitly to BF16, multiply BF16 operands with
  `tl.dot(..., out_dtype=tl.float32)`, and round the context to BF16. Since the
  custom tensor-core lowering need not reproduce native cuBLAS's reduction tree
  bitwise, the official strict elementwise check—not algebra alone—is the
  promotion authority.
- Dispatch/fallback: eval plus inference/no-grad CUDA BF16, causal, no
  effective token mask, exact cases 1/5, and their exact four aligned prefix
  tiles only. Cases 6/10 retain their already-promoted 64-row custom PV path.
  Every other case, shape, dtype, device, training/grad mode, mask, or invalid
  tile retains I03's existing native/candidate fallback. Baseline source,
  weights, public `forward(x, valid_token_mask)` contract, output shape,
  masking, LayerNorm/FFN/projection behavior, CUDA Graph eligibility, and replay
  clone ownership are unchanged.
- Rejected adjacent retries: this does not use the measured-slower 64-row
  cases-1/5 schedule and does not use native `matmul(out=...)` into the strided
  final context, which regressed the prior direct-context experiment. The new
  evidence-bearing difference is the custom PV kernel's explicit-stride loads
  and stores.
- Local validation: `git diff --check`; full facade/package/test Python
  compilation; 15/15 unit tests including exact target and K=96 mask/padding
  predicates; and the prescribed CPU BF16 smoke all passed. The CPU smoke was
  bitwise exact (`0 / 128` failed elements). A CPU layout audit confirmed that
  target `empty_like(query)` contexts use strides `(16384,32,128,1)` and their
  `[B,S,H,HD]` transpose is contiguous. None of these checks is GPU performance
  evidence.
- Preregistered gate: both requested cases must pass strict correctness before
  latency is interpreted. Promote only if the equal-case geometric mean of
  candidate latency improves at least 1% against I03 and neither target
  regresses more than 0.5%. Any correctness/compile/execution failure, smaller
  improvement, or material per-case regression retains I03. At most this one
  focused ordinary job is allowed from the worktree.
- Focused ordinary job: `job-1788126919028-fbcd32a5f1925eae`, immutable
  snapshot `d629a27f154fcf49281fb4b2ef845e2e556b8a66b99bee0c3dcc9e57cd55a0a2`,
  base commit `dd657816355dc0ad595a9fec40ef5350104e23c0`. It requested exactly
  official cases 1/5 and completed with state `succeeded`, exit zero, complete
  structured output, and no failure category. The environment was RTX 4070,
  Python 3.12.14, PyTorch 2.13.0+cu130, CUDA 13.0, and BF16, with five accuracy
  trials, 20 warmups, 100 repeats, and three alternating rounds.
- Correctness: both requested cases executed and all 10 trials were bitwise
  exact. The strict `abs < 0.002 OR rel < 0.02` check found zero failures over
  `15,728,640` elements; maximum absolute and relative errors were both zero.
- Same-job medians and speedups: case 1 baseline/candidate
  `1.4223359823 / 0.5765119791 ms`, `2.467140378x`; case 5
  `3.2256000042 / 1.1704319715 ms`, `2.755905582x`.
- Raw-derived 100-sample round medians: case 1 baseline
  `1.420288 / 1.423360 / 1.422336 ms` and candidate
  `0.575488 / 0.577536 / 0.577536 ms`; case 5 baseline
  `3.223552 / 3.227648 / 3.225600 ms` and candidate
  `1.155072 / 1.172480 / 1.175552 ms`. All 300 samples per model/case remain
  in the structured result.
- I03 comparison: cases 1/5 candidate latency fell by
  `18.405802% / 17.293782%`; equal-case geometric-mean latency fell
  `17.851673%` (equivalently I03/new throughput-style gain `21.731025%`).
  Same-job baseline drift versus I03 was only `+0.072047% / 0.000000%`, so it
  does not explain the candidate-only gain. Both targets comfortably clear the
  preregistered 1% geometric-mean gate with no target regression.
- Decision: focused `promote` for integration. The measured result validates
  the padded K=96 mask path as well as the power-of-two prefix tiles under the
  official end-to-end strict contract. Do not submit a follow-up from this
  worktree; layer this independent commit onto the shared winner and use a new
  ordinary unified job before making a full cases 1-13 performance claim.

## Case-11 HD8 PV E01 - fuse probability rounding into direct context write

- Status: focused `promote` for integration; no follow-up job was submitted.
- Parent: verified I03 implementation head
  `f157bcd0d1ba470664d1742f94eb5fb62bfc4fe6`, whose unified job
  `job-1788124928473-8295c78ca0925273` measured case-11 candidate median
  `1.288192 ms` with bitwise-exact correctness.
- Exact target: official case 11 `(B,S,D,H,HD)=(64,128,128,16,8)` under eval,
  inference/no-grad, CUDA BF16, causal attention, and no effective token mask.
  Its unchanged I03 schedule computes eight 16-row prefixes with key counts
  `16,32,48,64,80,96,112,128` per layer.
- Hypothesis: fuse each native FP32-probability-to-BF16 materialization with its
  following BF16 PV dot, and write the BF16 output into a sequence-major final
  context backing. This replaces eight cast plus eight native matmul launches
  with eight Triton launches per layer and removes the layer's `torch.cat` and
  head-major-to-sequence-major copy. Across four layers, this removes 32 cast
  launches, four cats, four layout copies, and their transient traffic.
- Numerical boundary: triangular BF16 QK dot and BF16 scale, the 16-row I03
  schedule, and native ATen FP32 softmax remain unchanged. The new kernel
  explicitly rounds FP32 probabilities to BF16 before a BF16 tensor-core PV
  dot, accumulates in FP32, and rounds the result to BF16 before storing it.
  HD8 is padded only inside the dot tile to 16 columns; columns 8-15 are loaded
  as exact zero and never stored. Non-power-of-two key counts 48/80/96/112 are
  padded to the next power-of-two tile with masked exact-zero operands after
  the complete real K axis. Real K is already a multiple of the tensor-core
  K=16 step, and adding FP32 zero after it is exact. Whether Triton's reduction
  order matches native BF16 PV closely enough remains an empirical strict-
  correctness gate; no performance result is valid unless it passes.
- Layout: allocate contiguous `[B,S,H,HD]` context backing and expose its
  `[B,H,S,HD]` permuted view to the stride-aware kernel. The subsequent
  transpose recovers the contiguous backing, so no second materialization is
  needed. Prefix row ranges are disjoint and collectively cover all 128 rows.
- Isolation: the new kernel/wrapper is separate from the already-promoted
  cases-6/10 PV kernel, leaving that source path and dispatch unchanged.
  Training, gradients, CPU, non-BF16, masks, other shapes, all other official
  cases, baseline, weight copying, public interface, LayerNorm/softmax
  reductions, and CUDA Graph independent-output cloning retain I03 fallback.
- Preregistered decision gate: strict correctness must pass all five trials;
  then candidate median must improve at least 1% versus I03 `1.288192 ms`
  (`<=1.275310 ms`) with stable per-round medians. Otherwise retain I03; an
  execution or correctness failure rejects E01 without interpreting latency.
- Pre-GPU validation: `git diff --check`, full facade/package/test Python
  compilation, 15/15 unit tests, and the prescribed CPU BF16 smoke passed. The
  CPU smoke was bitwise exact (`0 / 128`) and is not GPU performance evidence.
- Focused GPU job/snapshot:
  `job-1788126945825-861eb1e88a6d6a03` /
  `27db96c45f9f54110cefd85bb1b0614aaeb476ab60c5971a88377a189715443c`.
  The ordinary job used the pinned root Python 3.12.14 environment on an RTX
  4070 with PyTorch 2.13.0+cu130, CUDA 13.0, BF16, five accuracy trials, 20
  warmups, 100 repeats, and three alternating benchmark rounds. It completed
  the requested case with state `succeeded`, exit code 0, and no failure
  category.
- Correctness: all five trials were bitwise exact under the strict OR rule;
  `0 / 5,242,880` elements failed, and maximum absolute and relative errors
  were both zero. This validates the HD8 and non-power-of-two K padding
  boundary for this exact declared workload.
- Same-job baseline/candidate medians were
  `7.275520 / 1.177600 ms`, for `6.178261x` speedup. Raw 100-sample round
  medians were baseline `7.273472 / 7.277056 / 7.275520 ms` and candidate
  `1.175552 / 1.180672 / 1.179648 ms`; candidate round spread was only
  `0.4341%` of the middle round median.
- Against I03 candidate `1.288192 ms`, candidate latency fell `8.585056%`
  (equivalently `9.391305%` old/new gain), well beyond the preregistered 1%
  gate. Baseline drift versus I03 was `-0.147568%`, so the improvement is not
  explained by whole-job timing movement.
- Decision: `promote` E01 for integration. This is focused case-11 evidence,
  not a unified cases-1-13 claim; shared-winner source remains unchanged until
  the supervisor layers the commit and validates the combined matrix.

## Cases-4/12 full HD32 PV E01 - explicit boundary and final-layout store

- Status: `promote` from focused evidence; ready for shared-winner integration
  and a later unified cases-1/13 validation.
- Parent: verified I03 implementation commit
  `f157bcd0d1ba470664d1742f94eb5fb62bfc4fe6`; comparable unified job
  `job-1788124928473-8295c78ca0925273`, snapshot
  `63cd5873b3bf87a0e1cfd66a5e34c6e43a8bd384e54c3db0806bf88743263557`.
  I03 candidate medians for cases 4/12 are `0.2641919851 / 0.2017280012 ms`.
  All three 100-sample candidate round medians equal those aggregate medians
  for both cases.
- Exact targets: official CUDA BF16 cases 4 and 12,
  `(B,S,D,H,HD)=(16,128,128,4,32)` and `(64,32,128,4,32)`. Their unchanged
  full-attention path materializes triangular scores, invokes native FP32
  softmax, rounds its output into a separate BF16 probability tensor, invokes
  native BF16 PV against a projection-backed strided V view, then copies BHSD
  context into sequence-major layout. The hypothesis is that one custom PV
  launch per layer can fuse only the explicit probability rounding boundary
  and write the BF16 context directly into final layout.
- Implementation: reuse the already verified cases-6/10 stride-aware Triton
  kernel for the exact full `(M,K,N)` tiles `(128,128,32)` and `(32,32,32)`.
  It loads native-softmax FP32 probabilities, explicitly rounds them to BF16,
  multiplies by BF16 V with `tl.dot(..., out_dtype=tl.float32)`, rounds the
  accumulator to BF16, and stores through a BHSD view of contiguous
  `[B,S,H,HD]` backing. The existing transpose then exposes that backing as a
  contiguous `[B,S,D]` tensor without a layout materialization.
- Numerical boundary: QKV projection, triangular BF16 QK dot, BF16 scaling and
  causal mask, complete native FP32 softmax reduction, full K=128 or K=32 PV
  reduction, and BF16 context materialization remain in the same order. There
  is no online/fused softmax, split-K, K padding, or reduction reassociation
  outside the existing Triton-versus-native tensor-core implementation choice.
  The official strict elementwise test remains authoritative because its
  reduction tree need not be bitwise identical to cuBLAS.
- Resource audit: case 4 launches one 128x128x32 tile per batch/head with eight
  warps and two stages. The 128x32 FP32 accumulator distributes to 16 values
  per thread across 256 threads; BF16 A/B staging is about 40 KiB per stage,
  or about 80 KiB at two stages, below the Ada per-block shared-memory limit.
  Case 12 retains four warps and two stages for its 32x32x32 tile. Both use one
  program per batch/head and keep the entire K axis in one program.
- Dispatch/fallback: only exact query shapes `(16,4,128,32)` and
  `(64,4,32,32)` after the existing eval/inference/no-grad CUDA BF16 causal
  no-mask full-attention dispatch reach the new path. All other cases, custom
  shapes, dtypes, devices, modes and masks retain I03 native PV fallback;
  cases 6/10 keep their promoted prefix kernel unchanged. Baseline source,
  weights, public interface/output, QK/softmax/LayerNorm behavior, CUDA Graph
  eligibility, and independent replay output cloning are unchanged.
- Local validation: `git diff --check`; full facade/package/test Python
  compilation; 16/16 unit tests including exact dispatch, tile and final-layout
  predicates; and the prescribed CPU BF16 smoke passed. The CPU smoke was
  bitwise exact (`0 / 128` failed elements). These are not GPU performance
  evidence.
- Preregistered gate: both requested cases must pass strict
  `abs<0.002 OR rel<0.02` correctness before latency is interpreted. Promote
  only if candidate-latency equal-case geometric mean improves at least 1%
  versus I03 and neither case regresses more than 0.5%. Otherwise retain I03;
  a compile/execution failure or incorrect output rejects E01. No follow-up GPU
  job is permitted from this experiment.
- Focused ordinary job: `job-1788127317209-26f7cc043362c708`, immutable
  snapshot `909b7e609bc5be9e1ee04099e4df2078fc7d53f60cc78b9eb149b4c3ce51c301`
  from pre-result commit `3fda366666933e76e0a7d9eae9263469ef776ce8`.
  The job used the pinned root Python/package identity, RTX 4070,
  PyTorch 2.13.0+cu130/CUDA 13.0, BF16, five accuracy trials, 20 warmups,
  100 repeats, and three rounds; it exited zero in state `succeeded` after
  executing both and only requested cases 4/12.
- Correctness: both cases passed all five trials bitwise exact. Aggregate
  failures were `0 / 2,621,440` elements, with maximum absolute and relative
  error both zero. Performance is therefore admissible under the strict rule.
- Same-job baseline/candidate medians and speedups: case 4
  `0.9656320214 / 0.2426880002 ms` (`3.978903039x`); case 12
  `0.9625599980 / 0.1751040071 ms` (`5.497075788x`). Each timing side has 300
  samples. Candidate round medians were
  `0.2426880002 / 0.2426880002 / 0.2426880002 ms` for case 4 and
  `0.1751040071 / 0.1751040071 / 0.1751040071 ms` for case 12. Baseline round
  medians were `0.9636480212 / 0.9666560292 / 0.9666560292 ms` and
  `0.9605920017 / 0.9640319943 / 0.9635840058 ms`, respectively.
- I03 comparison: candidate latency fell `8.139529637%` for case 4 and
  `13.197966543%` for case 12; equal-case geometric-mean latency fell
  `10.704559905%` (`0.2308569278 -> 0.2061447097 ms`). Same-job baseline drift
  versus I03 was only `+0.319151367% / +0.442441590%`, while neither candidate
  regressed. The preregistered `>=1%` geomean and `<=0.5%` per-case regression
  gates pass decisively.
- Decision: `promote`. The exact full-HD32 PV/final-layout specialization is a
  verified improvement for cases 4/12. No follow-up job was submitted and this
  branch does not modify the shared winner; integration still requires a
  separate commit plus unified correctness/performance evidence.

## I04 - integrated focused PV specializations

- Status: promoted as the new shared cases 1-13 winner.
- Source head under test: `2204240bebaebf26aef5be3f26f5a29c6d519dee`.
  It layers the independent cases-1/5 HD32 prefix PV, case-11 HD8 PV, and
  cases-4/12 full HD32 PV commits over I03 while retaining exact-shape dispatch
  and all declared fallbacks.
- Integrated static validation: `git diff --check`, full facade/package/test
  Python compilation, 18/18 standard-library `unittest` tests, and the required
  CPU BF16 smoke bitwise exact (`0 / 128`). `pytest` was not installed in the
  pinned environment; CPU timing is not GPU evidence.
- Unified ordinary job/snapshot:
  `job-1788127778535-ebd60f321c76182d` /
  `42556ecebb98432a351e0a1baa5c330b3733aca665483dcd4b01a0b8fd6dfd03`;
  state `succeeded`, exit zero, official CUDA BF16 cases 1-13 on RTX 4070,
  Python 3.12.14, PyTorch 2.13.0+cu130, CUDA 13.0, five accuracy trials,
  20 warmups, 100 repeats, and three rounds.
- Correctness: all 13 requested cases and all 65 trials executed and were
  bitwise exact under the strict OR rule: `0 / 938,885,120` failed elements,
  with zero maximum absolute and relative error.
- Same-job baseline/candidate medians and speedups for cases 1-13 (ms, x):
  `1.421504/0.577536/2.461325`, `0.964608/0.098304/9.812500`,
  `0.970704/0.126976/7.644783`, `0.969728/0.242688/3.995781`,
  `3.228672/1.172480/2.753712`, `414.119934/167.259651/2.475911`,
  `1.099776/0.498688/2.205339`, `11.686560/10.891264/1.073021`,
  `0.882688/0.505344/1.746707`, `1.110016/0.499712/2.221311`,
  `7.277568/1.188864/6.121447`, `0.967808/0.176128/5.494913`, and
  `110.904690/35.574783/3.117509`.
- Candidate round medians for cases 1-13 (three rounds each, ms):
  `0.575488/0.578560/0.576512`, `0.098304/0.098304/0.098304`,
  `0.126976/0.126976/0.126976`, `0.241664/0.241664/0.242688`,
  `1.170944/1.173504/1.173504`, `167.253647/167.263229/167.258629`,
  `0.498688/0.498688/0.498688`, `10.891264/10.892288/10.891264`,
  `0.500736/0.509952/0.503808`, `0.498688/0.499712/0.499712`,
  `1.187840/1.190912/1.189888`, `0.176128/0.176128/0.176128`, and
  `35.576832/35.572735/35.574272`.
- Speedup arithmetic uses only paired medians from this same job:
  per-case `baseline_median / candidate_median`; their equal-case geometric
  mean is `3.267674802x`. The one-call-total speedup is
  `555.604255 / 218.812418 = 2.539180634x`.
- Cross-job I03 comparison uses candidate latency only, not a substituted
  baseline. Per-case latency improvements are
  `+18.2609%, 0.0000%, +2.3622%, +8.1395%, +17.1491%, +0.6511%, 0.0000%,
  +0.3607%, -0.9202%, +0.2587%, +7.7107%, +12.6904%, +0.8193%`.
  Equal-case candidate-latency geomean improved `5.433887%`; total candidate
  latency improved `0.882139%` (`220.759826 -> 218.812418 ms`). Case 9 was
  unchanged in source and its three rounds were noisier; the `0.9202%`
  cross-job regression is retained as noise evidence rather than attributed to
  a PV dispatch that cannot reach case 9.
- Aggregate MFU is `15.830245%`, derived as
  `2,017,695,105,024 FLOPs / 0.218812418 s / 58.25e12`, using the recorded
  `L*(8BSD^2 + 4BS^2D + 4BSDF)` convention. This is `+0.139645` percentage
  points versus I03; MFU is supervisor-derived, not a harness field.
- Decision: `promote` I04. Unified correctness is exact, every targeted case
  retains a material candidate improvement versus I03, aggregate geomean,
  total-call latency, and MFU all improve, and no unsupported-path regression
  is evidenced by this job.
