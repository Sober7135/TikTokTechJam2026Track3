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

## Small-shape fusion E01 - extend exact FFN GELU fusion to cases 4/12

- Status: focused `promote`; no follow-up job was submitted.
- Historical best: I04 implementation head
  `2204240bebaebf26aef5be3f26f5a29c6d519dee`, unified ordinary job
  `job-1788127778535-ebd60f321c76182d`, snapshot
  `42556ecebb98432a351e0a1baa5c330b3733aca665483dcd4b01a0b8fd6dfd03`.
  Cases 4/12 are bitwise exact and have candidate medians
  `0.242688 / 0.176128 ms`; their equal-case candidate-latency geometric mean
  is `0.206746589 ms`.
- Exact targets: official CUDA BF16 cases 4/12,
  `(B,S,D,H,FFN)=(16,128,128,4,128)/(64,32,128,4,128)`. Both flatten to the
  same 2048-by-128 first FFN projection in each of four layers. Their attention
  PV path remains the I04 full-HD32 winner and is not retuned by this experiment.
- Bottleneck hypothesis: I04 runs native `nn.Linear` into a materialized BF16
  hidden tensor and then launches exact `F.gelu` separately for these shapes.
  Extending the already-proven candidate Triton linear-plus-exact-erf-GELU
  kernel removes four intermediate BF16 global write/read pairs and four GELU
  launches per complete model call. At 2048 rows the existing autotune key is
  new, so its launch selection cannot perturb the proven 8192/16384-row cases.
- Numerical boundary: the kernel accumulates the 128-wide BF16 dot and bias in
  FP32, explicitly rounds the post-bias result to BF16 exactly where native
  `nn.Linear` materializes it, converts that rounded value to FP32 for the same
  exact-erf GELU formula, then stores BF16. It does not alter LayerNorm,
  attention QK/softmax/PV, residual ordering, later FFN projection, graph
  output cloning, weights, inputs, or the strict correctness rule. The
  existing cases 1/5/9/10/11 dispatch remains byte-for-byte unchanged apart
  from adding two exact members to its shape set.
- Dispatch and fallback: require eval, inference mode, gradients disabled,
  CUDA BF16, no effective token mask, exact `x.shape` `(16,128,128)` or
  `(64,32,128)`, and 128-by-128 FFN dimensions. CPU, other dtypes/shapes,
  masks, training/grad, unsupported widths, and all other official/custom
  cases retain native `nn.Linear` plus `F.gelu(approximate="none")`.
- Scope and decision gate: the complete new dispatch scope is exactly cases
  4/12. Both must pass five strict trials under
  `abs_error < 0.002 OR abs_error < 0.02 * abs(reference)` before timing is
  interpreted. Promote only if their equal-case candidate-latency geometric
  mean is at most `0.204679123 ms` (at least 1% below I04) and neither target
  exceeds its I04 median by more than 0.5% (`0.243901440 / 0.177008640 ms`).
  Otherwise retain I04; any compile/execution/correctness failure rejects E01.
  Only one focused ordinary GPU job is authorized for this worktree.
- Pre-GPU validation: `git diff --check`; full facade/package/test Python
  compilation; 19/19 standard-library unit tests including exact-shape and
  fallback gate predicates; and the prescribed CPU BF16 smoke all passed. The
  CPU smoke was bitwise exact (`0 / 128` failed elements) and exercised the
  unchanged fallback only; it is not GPU performance evidence. No dependency
  or benchmark-control code changed.
- Focused ordinary job: `job-1788128412458-0e5ce318734f1236`, immutable
  snapshot `94f5fa4a8b6617978608c8ecd6d027d4bcb0f8766c0a137b62dfe33693cf8aa0`
  from pre-result commit `13fa3ae83c12697f2ec54f3e7430d1a55fd29256`.
  It requested and completed exactly official cases 4/12, exited zero in state
  `succeeded`, and recorded complete structured output with no failure
  category. The environment was RTX 4070, Python 3.12.14, PyTorch
  2.13.0+cu130, CUDA 13.0, BF16, five accuracy trials, 20 warmups, 100 repeats,
  and three alternating rounds.
- Correctness: both cases passed all ten trials bitwise exact under the strict
  OR rule. Aggregate failures were `0 / 2,621,440`; maximum absolute and
  relative errors were both zero. Timing is therefore admissible.
- Same-job baseline/candidate medians and speedups: case 4
  `0.960512 / 0.235520 ms` (`4.078261x`); case 12
  `0.951248 / 0.167936 ms` (`5.664349x`). Raw-derived 100-sample round medians
  were baseline `0.991008 / 0.959392 / 0.959472 ms` and candidate
  `0.235520 / 0.235520 / 0.235520 ms` for case 4; baseline
  `0.947552 / 0.952320 / 0.951344 ms` and candidate
  `0.167936 / 0.167936 / 0.167936 ms` for case 12. All 300 samples per model
  and case remain in the structured result.
- I04 comparison: candidate latency fell `2.953584% / 4.651164%`; equal-case
  candidate-latency geometric mean fell `3.806119%`, from `0.206746589` to
  `0.198877568 ms`. Same-job baseline medians drifted
  `-0.950371% / -1.711081%` versus I04, but the paired candidate reductions
  remain larger by `2.003213 / 2.940083` percentage points and both candidate
  round triplets are invariant. Neither target regressed, and the preregistered
  1% geomean and 0.5% per-case gates pass.
- Decision: `promote` E01 for shared-winner integration. This focused result
  validates only the complete new cases-4/12 dispatch scope; it is not a
  unified cases-1/13 claim. Retain the exact fallbacks and require the
  supervisor's ordinary unified job after layering this commit. Do not submit
  a follow-up from this worktree.

## Winner integration I05 - cases 4/12 exact GELU extension

- Source under test: unsigned implementation commit
  `eb1a1bcd804b59480d2ac75fa3e8a769a57df73a`, layered on I04 without the
  rejected case-13 or retained case-6 scheduling experiments.
- Unified ordinary job: `job-1788128838367-68f541c7b93cb504`, immutable
  snapshot `739fea5b14f1dffa09bc861777dc5873dbb346612abd40eb00f56e8862d3d29e`.
  It ran official CUDA BF16 cases 1-13 on the RTX 4070 with Python 3.12.14,
  PyTorch 2.13.0+cu130, CUDA 13.0, five accuracy trials, 20 warmups, 100
  repeats, and three alternating benchmark rounds.
- Correctness is admissible and exact: all 65 trials passed bitwise, with
  `0 / 938,885,120` failed output elements and zero maximum absolute and
  relative error under the strict elementwise OR rule.
- Same-job baseline/candidate medians in milliseconds and paired speedups were:
  case 1 `1.421440 / 0.577536` (`2.461215x`); case 2
  `0.953344 / 0.097280` (`9.800000x`); case 3
  `0.953344 / 0.126976` (`7.508065x`); case 4
  `0.958688 / 0.235520` (`4.070516x`); case 5
  `3.224576 / 1.173504` (`2.747819x`); case 6
  `414.053787 / 167.206093` (`2.476308x`); case 7
  `1.100000 / 0.498688` (`2.205788x`); case 8
  `11.689984 / 10.917888` (`1.070718x`); case 9
  `0.870400 / 0.505856` (`1.720648x`); case 10
  `1.110016 / 0.499712` (`2.221311x`); case 11
  `7.277568 / 1.187840` (`6.126724x`); case 12
  `0.950400 / 0.169984` (`5.591114x`); and case 13
  `110.906364 / 35.576321` (`3.117421x`).
- Speedup arithmetic uses only paired medians from this job. The equal-case
  geometric mean is `3.267271794x`; the one-call-total speedup is
  `555.469912 / 218.773197 = 2.539021778x`.
- Cross-job comparison versus I04 uses candidate latency only. The intended
  targets improved by `2.953584%` for case 4 and `3.488373%` for case 12.
  Across all 13 cases, equal-case candidate-latency geometric mean improved
  `0.558209%`, while total candidate latency improved `0.017925%`
  (`218.812418 -> 218.773197 ms`). Small untargeted movements are retained as
  run-to-run noise rather than attributed to unreachable dispatch code.
- Aggregate MFU is `15.833083%`, derived as
  `2,017,695,105,024 FLOPs / 0.218773197 s / 58.25e12`; this is `+0.002838`
  percentage points versus I04. MFU is supervisor-derived rather than a
  harness field.
- Decision: `promote` I05. The entire cases-1/13 matrix is bitwise exact, both
  intended targets materially improve, and aggregate candidate-geomean,
  total-call latency, and MFU improve. The slightly lower paired speedup
  geomean versus I04 is baseline drift and is not substituted for the
  candidate-only cross-job comparison.

## Cross-case exact attention E01 - consolidate S=128 QK key tiles

- Status: `retain-best` for the broad six-case dispatch; the one authorized
  follow-up prunes the same exact tile to the evidenced cases 6/11 only.
- Historical best: clean I05 implementation/docs head
  `5079e32ee15af761b17f0b9472616fa72425f65b`, unified ordinary job
  `job-1788128838367-68f541c7b93cb504`, snapshot
  `739fea5b14f1dffa09bc861777dc5873dbb346612abd40eb00f56e8862d3d29e`.
- Exact targets: official CUDA BF16 cases 1/5/6/7/10/11. All six use the
  candidate's compact S=128 causal-prefix attention with native FP32 softmax
  and query chunks of 32, 32, 64, 64, 64, and 16 rows respectively.
- Bottleneck hypothesis: the existing triangular QK kernel ties the query-row
  and output-key tile widths together. As a prefix grows, each query tile is
  therefore reloaded once per 16/32/64-key block. Giving S=128 score chunks an
  independent 128-column key tile reduces score programs per batch/head from
  `1+2+3+4` to `4` for cases 1/5, `1+2` to `2` for cases 6/7/10, and
  `1+...+8` to `8` for case 11: about 60%, 33%, and 78% fewer programs. The
  expected gain comes from fewer Q reloads and less program scheduling; kernel
  launch count, softmax, and PV work are deliberately unchanged.
- Numerical-equivalence boundary: each score scalar still performs one BF16
  `Q @ K^T` dot over the complete, unchanged `head_dim` reduction, rounds that
  dot to BF16, applies the same scale and BF16 rounding, and stores through the
  same causal predicate. Only the independent output-column extent changes;
  there is no split-K, padding on the dot reduction axis, or reassociation of
  partial sums. Each prefix retains its exact compact tensor shape before the
  unchanged native ATen FP32 softmax, so neither softmax shape nor reduction
  changes. PV, output projection, residuals, LayerNorm, and FFN are untouched.
  Triton lowering for the wider N tile remains an empirical strict-correctness
  risk even though the algebraic reduction axis is unchanged.
- Dispatch/fallback: only the already-dispatched eval, inference/no-grad,
  causal, no-effective-mask, CUDA BF16 S=128 score chunks reach the 128-key
  tile. S=1024 case 13 retains its existing 128-by-128 score tile; full-score
  cases, masks, CPU, other dtypes/shapes/modes, baseline code, public forward
  contract, weights, case 14, and every existing fallback remain unchanged.
- Historical I05 candidate medians for targets 1/5/6/7/10/11 are
  `0.577536 / 1.173504 / 167.206093 / 0.498688 / 0.499712 / 1.187840 ms`;
  their equal-case geometric mean is `1.795851394 ms`.
- Preregistered gate: all six cases and all five trials must pass strict
  `abs_error < 0.002 OR abs_error < 0.02 * abs(reference)` correctness before
  latency is interpreted. Promote only if target candidate-latency geometric
  mean is at most `1.777892880 ms` (at least 1% better than I05), no target
  exceeds its I05 candidate median by more than 3%, and round medians do not
  expose an unstable apparent gain. Otherwise retain I05; any correctness,
  compile, or execution failure rejects E01. One focused multi-case job is
  planned, with at most one follow-up only for a newly evidenced concrete bug.
- Pre-GPU validation: `git diff --check`; Python compilation of the facade,
  package modules, and tests; 20/20 standard-library unit tests including the
  independent key-tile dispatch predicate; and the prescribed CPU BF16 smoke
  all passed. The smoke was bitwise exact (`0 / 128` failed elements) and used
  the unchanged CPU fallback. Its latency is not GPU evidence.
- Focused ordinary job: `job-1788130791019-9b6ac79507a0cf2c`, immutable
  snapshot `19a1319275de6e9658e26922cf6ce31e6cbed7f486235851650b1ba318b53d67`,
  base commit `64014f36a147bae646f46456288855f4100aa256`. The job requested and
  completed exactly cases 1/5/6/7/10/11 with the pinned Python 3.12.14
  environment, RTX 4070, PyTorch 2.13.0+cu130/CUDA 13.0, BF16, five accuracy
  trials, 20 warmups, 100 repeats, and three alternating rounds. `job.json`
  records state `succeeded`, exit zero, matching snapshot and arguments.
- Correctness passed before timing was interpreted. All 30 trials were bitwise
  exact under the strict OR rule: `0 / 846,725,120` failed elements and zero
  maximum absolute/relative error. Shape mismatch would have failed execution;
  zero failed elements also proves every compared value finite, and the bounded
  log contains no dtype-mismatch warning.
- Same-job baseline/candidate medians and paired speedups were: case 1
  `1.421311975 / 0.586751997 ms` (`2.422338536x`); case 5
  `3.223551989 / 1.199103951 ms` (`2.688300697x`); case 6
  `414.094329834 / 165.875717163 ms` (`2.496413200x`); case 7
  `1.102848053 / 0.500735998 ms` (`2.202454102x`); case 10
  `1.110015988 / 0.509952009 ms` (`2.176706767x`); and case 11
  `7.278592110 / 1.160192013 ms` (`6.273609911x`).
- Raw-derived baseline round medians were case 1
  `1.419263959/1.422496021/1.422335982`, case 5
  `3.220479965/3.225600004/3.224575996`, case 6
  `413.955078125/414.105590820/414.102081299`, case 7
  `1.101824045/1.102848053/1.102848053`, case 10
  `1.104895949/1.111104012/1.111039996`, and case 11
  `7.276544094/7.280640125/7.280240059 ms`. Candidate round medians were
  respectively `0.584703982/0.586751997/0.586751997`,
  `1.195520043/1.199103951/1.200127959`,
  `165.875717163/165.875717163/165.875381470`,
  `0.500320002/0.500735998/0.499711990`,
  `0.508928001/0.512000024/0.508928001`, and
  `1.130496025/1.162240028/1.159168005 ms`. All 300 samples per side/case
  remain in the structured result.
- Candidate-only improvements versus I05 were `-1.595744% / -2.181497% /
  +0.795650% / -0.410677% / -2.049182% / +2.327585%` for cases
  1/5/6/7/10/11, positive when faster. The six-case candidate-latency
  geometric mean regressed `0.505534%`, from `1.795851394` to
  `1.804930025 ms`; the preregistered broad `>=1%` gate therefore fails.
- Decision: `retain-best` for E01's broad dispatch despite exact correctness.
  The result gives precise shape-specific evidence: cases 6/11 both improve,
  their two-case latency geomean improves `1.564598%`, case-6 rounds are
  invariant, and every case-11 round remains below I05. Use the sole follow-up
  to restore the square tile for cases 1/5/7/10 and validate only cases 6/11;
  do not retune launch parameters or reinterpret the four regressions as noise.

## Cross-case exact attention E02 - prune key consolidation to cases 6/11

- Status: focused `promote`; ready for shared-winner integration and a unified
  cases-1/13 job, with no further job from this worktree.
- Change: retain E01's independently tiled 128-column key prefix only for exact
  query shapes `(10000,4,128,32)` and `(64,16,128,8)`, corresponding to cases
  6/11. Every sibling S=128 shape returns to `block_key_size =
  block_query_size`; kernel arithmetic, warp/stage settings, native FP32
  softmax, PV, projections, residuals, LayerNorm, FFN, baseline, and harness are
  unchanged. This is evidence-driven dispatch pruning, not a new retune.
- Numerical and fallback boundary: the two selected shapes retain the E01
  bitwise-exact complete key tile. Cases 1/5/7/10 and every nonselected shape,
  mask, dtype, device, mode, S=1024 path, and case 14 regain or retain the I05
  square-tile fallback.
- Preregistered follow-up gate: cases 6/11 must both pass all five strict trials
  before latency is interpreted. Promote only if their candidate-latency
  geometric mean is at most `13.952120456 ms` (at least 1% below I05
  `14.093050965 ms`), neither exceeds I05 by over 3%, and round medians remain
  consistent with a gain. Otherwise retain I05. This is the final GPU job from
  this worktree.
- Pre-GPU validation: `git diff --check`, full facade/package/test Python
  compilation, 20/20 unit tests, and the prescribed CPU BF16 smoke passed. The
  smoke was bitwise exact (`0 / 128`) on the unchanged CPU fallback and is not
  GPU performance evidence.
- Final focused ordinary job: `job-1788131342767-499e46c95514aba1`, immutable
  snapshot `6dd2cacc9030b96178133f7896d4404809134e44432dcc6ef20f3fe58734aef8`,
  base commit `f622b61b99a58d5e2ddd02f4a63f8c56511c7eb8`. `job.json` records the
  exact requested cases 6/11, pinned Python/environment identity, state
  `succeeded`, exit zero, and no error. The structured result used RTX 4070,
  Python 3.12.14, PyTorch 2.13.0+cu130/CUDA 13.0, BF16, five trials, 20
  warmups, 100 repeats, and three alternating rounds.
- Correctness passed before timing was interpreted. Both cases and all ten
  trials were bitwise exact under the strict OR rule: `0 / 824,442,880`
  failed elements and zero maximum absolute/relative error. Shape mismatch
  would have failed execution; zero failed elements proves compared values
  finite, and the bounded log contains no dtype-mismatch warning.
- Same-job baseline/candidate medians and paired speedups: case 6
  `414.151672363 / 165.683197021 ms` (`2.499660073x`); case 11
  `7.268352032 / 1.158143997 ms` (`6.275862112x`).
- Raw-derived baseline round medians were case 6
  `414.148101807 / 414.151672363 / 414.155258179 ms` and case 11
  `7.268352032 / 7.268352032 / 7.268352032 ms`. Candidate round medians were
  case 6 `165.682174683 / 165.683776855 / 165.684219360 ms` and case 11
  `1.129472017 / 1.160192013 / 1.156095982 ms`. Case 6 is invariant at the
  displayed precision; case 11's first round is faster, but every round remains
  below I05 and the aggregate median reproduces E01 within 0.18%.
- Candidate-only improvements versus I05 were `0.910790%` for case 6 and
  `2.500000%` for case 11. Their two-case latency geomean fell from
  `14.093050965` to `13.852256136 ms`, a `1.708607%` improvement that clears
  the preregistered 1% gate with no target regression. Versus broad E01, case
  medians improved another `0.116063% / 0.176524%` and their geomean improved
  `0.146298%`, consistent with ordinary run variation rather than a pruning
  penalty on the still-selected shapes.
- Decision: focused `promote` E02. Integrate the net I05-to-E02
  `triangular_scores.py` and dispatch test diff only: independent query/key
  tile parameters plus the exact consolidated set `{(10000,4,128,32),
  (64,16,128,8)}`. Both implementation commits are needed in history because
  E02 prunes E01; do not integrate the broad six-shape dispatch by itself.
  Shared-winner source still requires an ordinary unified cases-1/13 job before
  any matrix-wide performance claim.

## Winner integration I06 - exact QK key tiles for cases 6/11

- Source under test: shared-winner implementation head
  `22e686b809a533293e80868a7b41356b03e9888b`, containing unsigned integration
  commits `c5b5d43` and `22e686b`. The net runtime dispatch differs from I05
  only for exact case-6/11 query shapes; all other official/custom paths retain
  I05 behavior.
- Unified ordinary job: `job-1788131811786-b0caf8beace8d3d3`, immutable
  snapshot `8592e0d4e5085a5afa83f71cb49852b3009333d15b029c17e6014201033036fd`.
  It ran official CUDA BF16 cases 1-13 on RTX 4070 with Python 3.12.14,
  PyTorch 2.13.0+cu130, CUDA 13.0, five trials, 20 warmups, 100 repeats, and
  three alternating rounds.
- Correctness is exact and matrix-complete: all 13 cases executed; all 65
  trials passed bitwise with `0 / 938,885,120` failed elements and zero maximum
  absolute/relative error under the unchanged strict elementwise OR rule.
- Same-job baseline/candidate medians in milliseconds and paired speedups:
  case 1 `1.421312 / 0.575488` (`2.469751x`); case 2
  `0.954608 / 0.097280` (`9.812993x`); case 3
  `0.947600 / 0.126976` (`7.462828x`); case 4
  `0.960000 / 0.235520` (`4.076087x`); case 5
  `3.225600 / 1.170432` (`2.755906x`); case 6
  `414.039032 / 165.830658` (`2.496758x`); case 7
  `1.101824 / 0.498688` (`2.209446x`); case 8
  `11.687536 / 10.892288` (`1.073010x`); case 9
  `0.870400 / 0.506880` (`1.717172x`); case 10
  `1.110160 / 0.499792` (`2.221244x`); case 11
  `7.278592 / 1.164288` (`6.251539x`); case 12
  `0.954256 / 0.169984` (`5.613799x`); and case 13
  `110.914558 / 35.577854` (`3.117517x`).
- Paired same-job equal-case speedup geomean is `3.276650791x`. One-call-total
  speedup is `555.465478 / 217.346128 = 2.555672294x`.
- Cross-job comparison uses candidate latency only. Versus I05, case 6/11
  improve `0.822599% / 1.982755%` in the unified run. Across all 13 cases,
  candidate-latency geomean improves `0.265666%` and total candidate latency
  improves `0.652306%` (`218.773198 -> 217.346128 ms`). Untargeted movements
  range from `-0.202426%` to `+0.354615%` and are retained as run-to-run noise;
  they are not attributed to unreachable exact-shape dispatches.
- Aggregate MFU is `15.937041%`, derived as
  `2,017,695,105,024 FLOPs / 0.217346128 s / 58.25e12`, up `0.103958`
  percentage points from I05 under the same supervisor convention.
- Decision: `promote` I06 as the shared winner. Unified exactness, targeted
  candidate improvements, all-case candidate geomean, total-call latency, and
  MFU all improve. Cases 2/3 remain at or above 7x; the wider 7x-10x objective
  remains incomplete for most cases and requires further independent rounds.

## Cross-case direct-layout QKV E01 - write projection into contiguous BHSD

- Status: pre-GPU candidate. Historical best is the clean I06 source/docs head
  `39e9629de19180f63b847397afe0668a6a924207`; its unified ordinary job is
  `job-1788131811786-b0caf8beace8d3d3`, immutable snapshot
  `8592e0d4e5085a5afa83f71cb49852b3009333d15b029c17e6014201033036fd`.
- Exact targets: official CUDA BF16 cases 1/4/5/7/9/10/11/12/13. Their I06
  same-job speedups remain below 7x, while cases 2/3 already meet the requested
  range. Cases 6/8 are deliberately excluded because their much larger row or
  feature dimensions require different GEMM tiling; case 14 remains outside
  this campaign as previously directed.
- Bottleneck hypothesis: the current packed `nn.Linear` writes row-major
  `[B,S,3D]` and exposes Q/K/V as `[B,H,S,HD]` views with sequence stride `3D`.
  Cases 11/13 additionally materialize a contiguous V copy. A candidate-only
  QKV kernel can preserve one packed projection launch while writing directly
  to a single contiguous `[3,B,H,S,HD]` backing. This removes the packed
  intermediate layout, makes all three attention operands contiguous, and
  removes the cases-11/13 V copy. Expected benefit is lower QK/PV operand
  traffic and layout overhead across all four layers, not fewer attention
  reductions.
- Numerical boundary: native `norm1` remains fully materialized and unchanged.
  The kernel loads the same BF16 normalized activation and packed
  `nn.Linear.weight[out,in]`, performs complete D=32 or D=128 BF16 tensor-core
  dots with FP32 accumulation, adds the same packed BF16 bias in FP32, and
  explicitly rounds to BF16 before storing Q/K/V. Its K=32 reduction tiles and
  weight orientation copy the already strict-correct fused FFN linear
  structure. Only output address calculation changes: packed feature
  `p*D+h*HD+d` and row `b*S+s` map to `[p,b,h,s,d]`. QK scaling/masking,
  native FP32 softmax, probability rounding, PV, projections, residual order,
  later LayerNorm/FFN work, weights, baseline, harness, and thresholds remain
  unchanged. Triton versus cuBLAS reduction lowering remains an empirical
  strict-correctness risk.
- Dispatch/fallback: require eval plus inference/no-grad, CUDA BF16, contiguous
  normalized input, causal attention with no effective token mask, packed QKV
  bias, and one of exact `(B,S,D,H)` shapes `(16,128,128,4)`,
  `(64,32,128,4)`, `(64,128,32,4)`, `(64,128,128,1)`,
  `(64,128,128,2)`, `(64,128,128,4)`, `(64,128,128,16)`,
  `(64,1024,128,4)`, or `(128,128,128,4)`. Training, gradients, CPU,
  non-BF16, masks, noncausal/custom shapes, cases 2/3/6/8/14, missing bias,
  baseline, public interface, and all existing candidate routes retain I06.
- I06 target candidate medians for cases 1/4/5/7/9/10/11/12/13 are
  `0.575488 / 0.235520 / 1.170432 / 0.498688 / 0.506880 / 0.499792 /
  1.164288 / 0.169984 / 35.577854 ms`. Their equal-case candidate-latency
  geometric mean is `0.804468593 ms`.
- Preregistered broad gate: all nine cases and all five trials must pass strict
  `abs_error < 0.002 OR abs_error < 0.02 * abs(reference)` correctness before
  performance is interpreted. Promote broadly only if candidate-latency
  geometric mean is at most `0.796423907 ms` (at least 1% below I06), no target
  exceeds its I06 median by more than 3%, and round medians do not expose an
  unstable apparent gain. Otherwise retain I06. The sole follow-up, if used,
  may only fix one concrete compile/address bug or prune to an evidenced
  multi-case winning subset; it may not retune unrelated launch parameters.
- Pre-GPU validation: `git diff --check`; full facade/package/test Python
  compilation; 23/23 standard-library unit tests including exact dispatch,
  `[3,B,H,S,HD]` storage layout, weight orientation, and BF16 boundary checks;
  and the prescribed CPU BF16 smoke passed. The smoke was bitwise exact with
  `0 / 128` failures on the unchanged CPU fallback and is not GPU performance
  evidence.
- Focused ordinary GPU job, snapshot, deterministic correctness, raw timings,
  and final decision are pending.

### E01 deterministic review

- Focused ordinary job/snapshot:
  `job-1788132477041-5eab7fb62fad7ec5` /
  `98f64f040a08baf3304ce3d8ffdcd44e1907e82394b246f0824dfcfedcc75e98`;
  base commit `e6a177a6a9d9643c75be04f2cc247520c84c60bb`. `job.json`
  records the exact requested cases 1/4/5/7/9/10/11/12/13, pinned Python
  environment, terminal state `failed`, exit code 2, and no execution error.
  The structured result is complete and matrix-subset-complete with top-level
  `failure_category=correctness` and `correctness_passed=false`.
- Correctness was isolated to D=32 case 7. Cases 1/4/5/9/10/11/12/13 and all
  40 of their trials were bitwise exact: `0 / 76,021,760` failed elements and
  zero maximum absolute/relative error. Case 7 failed four of five trials with
  per-trial failures `19 / 0 / 3 / 51 / 2`; in aggregate `75 / 1,310,720`
  elements failed, maximum absolute error was `0.03125`, and maximum relative
  error was `1.0`. Its trial 2 was bitwise exact. Performance was skipped for
  case 7, and the failed job supplies no valid broad performance conclusion.
- Bounded log inspection shows no compile, address, OOM, or runtime exception.
  The same output-address mapping and D=128 reduction are bitwise exact on all
  eight sibling cases, while D=32 produces sparse seed-dependent deviations.
  Evidence therefore points to a shape-specific D=32 Triton-versus-cuBLAS dot
  reduction mismatch, not a global weight-orientation or output-index bug.
- Decision: `reject-incorrect` for E01's nine-shape dispatch. Use the sole
  authorized follow-up only to prune exact tuple `(64,128,32,4)` so case 7
  regains I06 packed-QKV fallback. Do not change kernel math, output mapping,
  autotune configurations, or the eight bitwise-exact D=128 routes.

## Cross-case direct-layout QKV E02 - prune D=32 case 7

- Status: final focused follow-up candidate. E02 removes only
  `(B,S,D,H)=(64,128,32,4)` from the direct-layout allowlist. Case 7 therefore
  executes the verified I06 `nn.Linear` packed-QKV path; the exact same E01
  kernel and `[3,B,H,S,HD]` output layout remain active for D=128 cases
  1/4/5/9/10/11/12/13.
- Numerical/fallback boundary: the eight selected paths retain E01's measured
  bitwise-exact FP32-dot+bias then BF16 boundary. The only incorrect D=32 path
  is removed rather than approximated or retuned. Training, gradients, masks,
  CPU, non-BF16, noncausal/custom shapes, cases 2/3/6/7/8/14, baseline,
  interface, weights, QK/softmax/PV, LayerNorm, residuals, FFN, and harness all
  retain I06 fallback or behavior.
- Final preregistered job repeats the same nine cases and settings. All nine
  cases and all 45 trials must pass the strict OR rule before timing is read.
  Promote the eight-case direct-layout route only if the full nine-case
  candidate-latency geometric mean is at most `0.796423907 ms` (at least 1%
  below I06), no case exceeds its I06 median by more than 3%, and round medians
  are stable. Otherwise retain I06. No further follow-up is permitted.
- Pre-GPU validation: `git diff --check`; full facade/package/test Python
  compilation; 23/23 unit tests including the explicit case-7 exclusion; and
  the prescribed CPU BF16 smoke passed. The smoke was bitwise exact with
  `0 / 128` failures on the unchanged CPU fallback and is not GPU performance
  evidence. The final ordinary GPU job is pending.
