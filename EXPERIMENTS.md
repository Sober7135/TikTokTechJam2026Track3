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
  evidence.
- Final ordinary job/snapshot: `job-1788133262714-f0936aef42476d7c` /
  `4710dc2a27e69c62c2f122f0cee4ffd44c15870e1ae3c49df6e9ec61775187ba`;
  base commit `60ca0da3ddecb85110ca4ad5bd3f7b5dbfe4eacc`. `job.json`
  records exactly cases 1/4/5/7/9/10/11/12/13 with CUDA BF16, the pinned
  Python 3.12.14 executable and inventory, state `succeeded`, exit zero, and
  no error. The structured result is complete and subset-complete on RTX 4070
  with PyTorch 2.13.0+cu130/CUDA 13.0, five accuracy trials, 20 warmups, 100
  repeats, and three alternating benchmark rounds.
- Correctness passed before timing was interpreted. All nine cases and all 45
  trials were bitwise exact under the strict OR rule: `0 / 77,332,480` failed
  elements and zero maximum absolute/relative error. This validates the eight
  selected D=128 direct-layout routes and confirms that pruned D=32 case 7
  exactly regains I06 behavior.

| Case | Baseline median (ms) | Candidate median (ms) | Same-job speedup | Gain vs I06 candidate |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.422335982 | 0.584703982 | 2.432574476x | -1.576184% |
| 4 | 0.945183992 | 0.220159993 | 4.293168715x | +6.976750% |
| 5 | 3.225600004 | 1.178624034 | 2.736750576x | -0.695053% |
| 7 | 1.098752022 | 0.496639997 | 2.212371192x | +0.412374% |
| 9 | 0.868351996 | 0.488447994 | 1.777777791x | +3.773583% |
| 10 | 1.104895949 | 0.508928001 | 2.171026053x | -1.795147% |
| 11 | 7.272448063 | 1.068032026 | 6.809204110x | +9.012465% |
| 12 | 0.944127977 | 0.152768001 | 6.180142248x | +11.269374% |
| 13 | 110.900222778 | 34.620414734 | 3.203318725x | +2.765534% |

- The last column is the preregistered throughput-style candidate-only
  comparison `I06_median / E02_median - 1`, computed from both structured
  result files rather than rounded prose. I06 and E02 nine-case candidate
  latency geometric means are `0.804468589 / 0.779135125 ms`: latency falls
  `3.149093%`, equivalently old/new gain is `3.251485%`. The same-job paired
  speedup geometric mean is `3.176203437x`; focused baseline/candidate median
  sums are `127.781918764 / 39.318718761 ms` (`3.249900373x`).
- Raw-derived baseline round medians, cases 1/4/5/7/9/10/11/12/13 in order
  (ms): `1.419968009/1.423359990/1.422335982`,
  `0.943967998/0.946671993/0.945951998`,
  `3.222527981/3.227648020/3.226624012`,
  `1.097104013/1.099776030/1.098991990`,
  `0.865279973/0.869040012/0.868351996`,
  `1.100800037/1.107967973/1.104895949`,
  `7.270400047/7.273551941/7.273471832`,
  `0.946175992/0.943104029/0.944575995`, and
  `110.884193420/110.903392792/110.906364441`.
- Corresponding candidate round medians (ms):
  `0.583679974/0.584703982/0.582656026`,
  `0.219136000/0.219215997/0.220159993`,
  `1.179136038/1.180672050/1.171967983`,
  `0.495615989/0.496639997/0.496639997`,
  `0.486975998/0.489472002/0.487423986`,
  `0.506879985/0.512000024/0.509952009`,
  `1.045503974/1.070592046/1.045503974`,
  `0.152800001/0.152608000/0.152751997`, and
  `34.619392395/34.614273071/34.629631042`. Every apparent winning case keeps
  all three rounds below its I06 aggregate median; the three regressions are
  bounded below the preregistered limit.
- Decision: focused `promote` E02 for shared-winner integration. The nine-case
  geometric-mean gain exceeds 1%, and the worst target regression is case 10
  at `1.795147%`, below the allowed 3%; cases 1/5 regress only
  `1.576184% / 0.695053%`. Integrate both implementation commits so E02's
  case-7 pruning accompanies E01, then run a new ordinary unified cases-1/13
  job before making a matrix-wide claim. No further job is permitted from this
  worktree.

## Winner integration I07 - direct-layout QKV with D=32 fallback

- Starting shared winner: I06 implementation/docs head `22e686b` / `39e9629`,
  unified job `job-1788131811786-b0caf8beace8d3d3`.
- Layered the direct-layout QKV E01, Case-7 dispatch prune E02, and focused
  result record as unsigned integration commits `efbd73c` / `72d3c1b` /
  `a0dc957`. The selected CUDA BF16 D=128 shapes project packed QKV directly
  into one contiguous `[3,B,H,S,HD]` backing; D=32 Case 7 and every unsupported
  configuration retain I06's native packed-linear fallback.
- Integrated static validation passed `git diff --check`, full Python
  compilation, 23/23 unit tests, and the prescribed CPU BF16 smoke bitwise
  exactly at `0 / 128`. The CPU smoke exercises fallback behavior and is not
  GPU performance evidence.
- Unified ordinary job/snapshot:
  `job-1788133519165-d75dd352756ff0bf` /
  `f33228bfb8b8340bae67c1eb2b806bbc8b5af8165364710a1cec9d6f425f5324`;
  base commit `a0dc957411fbab051a796d9112d7d048e9d63f81`, official CUDA BF16
  cases 1-13, five trials, 20 warmups, 100 repeats, three rounds, state
  `succeeded`, exit zero.
- Unified correctness passed before performance interpretation: all 13 cases
  and all 65 trials were bitwise exact, `0 / 938,885,120` failed elements,
  with zero maximum absolute and relative error.
- Same-job baseline/candidate medians and paired speedups for cases 1-13 are:
  `1.421312/0.582656/2.439367x`, `0.947200/0.098304/9.635416x`,
  `0.947200/0.126976/7.459678x`, `0.956416/0.220160/4.344186x`,
  `3.223552/1.179648/2.732639x`, `414.117371/165.891068/2.496321x`,
  `1.100800/0.498688/2.207392x`, `11.687984/10.892288/1.073051x`,
  `0.872448/0.496640/1.756701x`, `1.110016/0.512000/2.168000x`,
  `7.276544/1.077248/6.754753x`, `0.954848/0.153600/6.216458x`, and
  `110.904320/34.628609/3.202679x`.
- Paired same-job equal-case speedup geometric mean is `3.334607761x`.
  One-call-total speedup is
  `555.520010471 / 216.357884496 = 2.567597718x`.
- Versus I06 candidate medians, cases 1-13 changed
  `-1.230238% / -1.041667% / 0.000000% / +6.976750% / -0.781256% /
  -0.036415% / 0.000000% / 0.000000% / +2.061853% / -2.384381% /
  +8.079854% / +10.666660% / +2.741217%`. Untargeted Cases 2/3/6/8 and
  fallback Case 7 retain I06 source behavior; their small cross-job movement is
  not attributed to direct QKV.
- All-case candidate-latency geometric mean falls from `1.092235353` to
  `1.072358624 ms`, an old/new gain of `1.853552%`. Total candidate latency
  falls from `217.346128307` to `216.357884496 ms`, improving `0.456763%`.
- Aggregate MFU is `16.009836%`, using `2,017,695,105,024 FLOPs`, the unified
  candidate median sum, and attached `58.25 TFLOP/s` peak. This remains a
  supervisor-derived metric under the ledger convention, not a harness field.
- Decision: `promote` I07 as the shared winner. Full-matrix strict correctness,
  targeted QKV gains, all-case candidate geomean, total-call latency, paired
  speedups, and MFU improve. Cases 2/3 remain at or above 7x; most cases remain
  below the requested 7x-10x objective, so the optimization campaign continues.

## I07 direct-layout QKV E03 - prune three regressing shapes

- Status: preregistered candidate. Focused E02 and unified I07 independently
  showed candidate-latency regressions for Cases 1/5/10: focused
  `-1.576184% / -0.695053% / -1.795147%`, unified
  `-1.230238% / -0.781256% / -2.384381%`. E03 removes only their exact
  `(B,S,D,H)` tuples `(64,128,128,4)`, `(128,128,128,4)`, and
  `(64,128,128,2)` from direct-layout dispatch.
- Hypothesis and attribution: those three shapes should recover the I06/I07
  native packed `nn.Linear` QKV route, eliminating a repeatable direct-layout
  overhead. The direct kernel remains selected only for the five shapes that
  improved in both jobs: Cases 4/9/11/12/13, respectively
  `(16,128,128,4)`, `(64,128,128,1)`, `(64,128,128,16)`,
  `(64,32,128,4)`, and `(64,1024,128,4)`.
- Numerical and fallback boundary: no kernel math, accumulation, BF16 rounding,
  output address mapping, attention, FFN, weights, interface, or harness changes.
  The five selected direct paths retain I07's bitwise-exact evidence. Cases
  1/5/10 join Case 7 and all unsupported training, gradient, mask, CPU,
  non-BF16, noncontiguous, and custom-shape configurations on the already
  verified native packed-linear fallback. Strict correctness remains elementwise
  `abs_error < 0.002 OR abs_error < 0.02 * abs(reference)`.
- Preregistered ordinary GPU gate: repeat official CUDA BF16 Cases
  1/4/5/7/9/10/11/12/13 with five accuracy trials, 20 warmups, 100 repeats,
  and three alternating rounds. Require all 45/45 trials correct before reading
  timing. Promote only if the nine-case candidate-latency geometric mean versus
  unified I07 improves at least `0.5%`, no case regresses more than `1.5%`, and
  round medians are stable. One concrete bug follow-up is permitted; otherwise
  retain I07.
- Pre-GPU validation passed `git diff --check`, full facade/package/test Python
  compilation, 23/23 unit tests including explicit inclusion/exclusion dispatch
  assertions, and the prescribed CPU BF16 smoke. The smoke was bitwise exact at
  `0 / 128` failures on fallback and is not GPU performance evidence. Ordinary
  GPU job/snapshot: pending.
## I07 direct-QKV autotune E01 - sequence-aware launch search

- Status: pre-GPU candidate based on shared winner commit
  `a4c2d81a6ea5d02d30220c622a5d10c5d14fd0da`. Exact target cases are official
  CUDA BF16 cases 4/9/11/12/13, the direct-QKV shapes with stable I07 gains.
- Historical-best evidence: I07 unified job
  `job-1788133519165-d75dd352756ff0bf`, snapshot
  `f33228bfb8b8340bae67c1eb2b806bbc8b5af8165364710a1cec9d6f425f5324`,
  passed all 65 trials bitwise exactly on RTX 4070. Target candidate medians for
  cases 4/9/11/12/13 are `0.220160 / 0.496640 / 1.077248 / 0.153600 /
  34.628609 ms`; their equal-case latency geometric mean is
  `0.910719035 ms`.
- Bottleneck hypothesis: the existing three-config search lacks a 128-row tile,
  a 32x128 tile, and lower-warp/deeper-stage alternatives for the fixed
  `M x 384 x 128` GEMMs. It also keys only on row count, width, and head count,
  so cases 4 `(B,S)=(16,128)` and 12 `(64,32)` reuse one selected config even
  though their direct `[3,B,H,S,HD]` store address shapes differ. Add five
  launch-only candidates covering 32x128, lower-warp 64-wide/128-wide tiles,
  the established 128x64 shape, and 128x128; include sequence length in the
  autotune key so those two store geometries are measured independently.
- Attribution boundary: output width 384 is exactly divisible by both 64 and
  128 and every target row count is divisible by 32/64/128. Every candidate
  keeps `block_reduction=32`, the same four increasing K tiles for D=128, the
  same `tl.dot(..., out_dtype=tl.float32)`, bias addition, explicit BF16
  rounding, and exact `[P,B,H,S,HD]` output address mapping. Only tile ownership,
  warp count, software-pipeline depth, and cache-key separation change; QKV
  math, rounding order, inputs, weights, dispatch allowlist, fallback,
  attention, FFN, baseline, interface, harness, and thresholds do not.
- Dispatch/fallback: unchanged from I07. The exact selected D=128 inference
  shapes use direct QKV; D=32 case 7 and all unsupported training, gradient,
  mask, device, dtype, layout, or shape conditions retain the established I07
  fallback.
- Preregistered focused gate: run exactly cases 4/9/11/12/13 with five accuracy
  trials, 20 warmups, 100 repeats, and three alternating rounds. Interpret
  timing only after all 25 trials pass strict
  `abs_error < 0.002 OR abs_error < 0.02 * abs(reference)` correctness. Promote
  only if candidate-latency geometric mean is at most `0.903939489 ms`
  (old/new improvement at least 0.75%) and no case exceeds its I07 median by
  more than 2% (`0.224563200 / 0.506572800 / 1.098792960 / 0.156672000 /
  35.321181180 ms`). Otherwise retain I07. At most one follow-up may prune one
  concretely failing compile/config; it may not alter math or add another
  optimization category.
- Pre-GPU validation passed `git diff --check`, full facade/package/test Python
  compilation, and 24/24 unit tests including exact key/config coverage. The
  prescribed CPU BF16 smoke was bitwise exact with `0 / 128` failures on the
  unchanged CPU fallback; it is not GPU performance evidence. Ordinary GPU
  job/snapshot, correctness, raw timings, and decision are pending.
## I07 direct-layout QKV E03 deterministic review

- Ordinary job/snapshot/base commit:
  `job-1788134032351-e0096467490e18c4` /
  `89995518531311186a244ba59c346a16e32fb4fc06fcff76a96a049006cc96dd` /
  `9e4db9f0063495b40e0f3873f43144241048b2d0`. `job.json` records the exact
  preregistered Cases 1/4/5/7/9/10/11/12/13 arguments, pinned Python 3.12.14
  executable and inventory, state `succeeded`, exit zero, and no error. The
  structured result is complete on RTX 4070 with PyTorch 2.13.0+cu130, CUDA
  13.0, BF16, five accuracy trials, 20 warmups, 100 repeats, and three rounds.
- Correctness passed before timing interpretation. All nine cases and all 45
  trials were bitwise exact under the strict OR rule: `0 / 77,332,480` failed
  elements with zero maximum absolute and relative error.

| Case | I07 candidate (ms) | E03 baseline (ms) | E03 candidate (ms) | I07/E03 gain | Same-job speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.582656026 | 1.422335982 | 0.577535987 | +0.886532% | 2.462765983x |
| 4 | 0.220159993 | 0.946175992 | 0.220159993 | +0.000000% | 4.297674529x |
| 5 | 1.179648042 | 3.225600004 | 1.170431972 | +0.787408% | 2.755905582x |
| 7 | 0.498688012 | 1.101824045 | 0.496639997 | +0.412374% | 2.218556806x |
| 9 | 0.496639997 | 0.867327988 | 0.486400008 | +2.105261% | 1.783157840x |
| 10 | 0.512000024 | 1.104911983 | 0.497664005 | +2.880662% | 2.220196705x |
| 11 | 1.077247977 | 7.274496078 | 1.074175954 | +0.285989% | 6.772164330x |
| 12 | 0.153600007 | 0.939007998 | 0.152575999 | +0.671146% | 6.154362420x |
| 13 | 34.628608704 | 110.898178101 | 34.626560211 | +0.005916% | 3.202691155x |

- The I07 reference is unified job `job-1788133519165-d75dd352756ff0bf`.
  The last-but-one column is the candidate-only throughput-style comparison
  `I07_median / E03_median - 1`, calculated from both structured results. The
  nine-case candidate-latency geometric mean falls from `0.782465276319` to
  `0.775573395184 ms`: latency is `0.880790668%` lower, equivalently old/new
  gain is `0.888617528%`. Same-job paired speedup geomean is `3.189903914x`;
  focused baseline/candidate median sums are
  `127.779858172 / 39.302144125 ms` (`3.251218503x`).
- Raw-derived baseline round medians, Cases 1/4/5/7/9/10/11/12/13 in order
  (ms): `1.420287967/1.422447979/1.422335982`,
  `0.946079999/0.947183996/0.945487976`,
  `3.223551989/3.226624012/3.225600004`,
  `1.099471986/1.102303982/1.102560043`,
  `0.864256024/0.869376004/0.868351996`,
  `1.103871942/1.105919957/1.105919957`,
  `7.274496078/7.273983955/7.274496078`,
  `0.938816011/0.939007998/0.939536005`, and
  `110.897155762/110.899200439/110.899200439`.
- Corresponding candidate round medians (ms):
  `0.576511979/0.579584002/0.576511979`,
  `0.220159993/0.220159993/0.221184000`,
  `1.163264036/1.173503995/1.171967983`,
  `0.496639997/0.496639997/0.496639997`,
  `0.483328015/0.494592011/0.487423986`,
  `0.495615989/0.498688012/0.498688012`,
  `1.078271985/1.076223969/1.069056034`,
  `0.152575999/0.152575999/0.152575999`, and
  `34.629631042/34.626560211/34.625537872`. Candidate round spread is at most
  `2.310924%` (Case 9); all other cases are below `0.874%`, with no aggregate
  regression or unstable sign that changes the gate decision.
- Decision: focused `promote`. Strict 45/45 correctness passes, the nine-case
  old/new geomean gain `0.888618%` exceeds the preregistered `0.5%` threshold,
  and every case is non-regressing versus I07, comfortably inside the `1.5%`
  limit. No bug follow-up is needed. Integrate the implementation commit into
  the shared winner and require a new ordinary unified Cases 1-13 job before a
  matrix-wide promotion claim.

### E01 deterministic review

- Identity and scope: ordinary job `job-1788134123272-8f4753aca47ede5c`
  evaluated immutable snapshot
  `213f0e091b0e9780cbe6cb8b8604fccc1686dcb022ef6657797cb50baa51d96d`
  from optimization commit `14f79942735c3e6316e9ddaf81e7522a960baa92`.
  `job.json` records exactly official cases 4/9/11/12/13, CUDA BF16, pinned
  Python 3.12.14 and package identity, terminal state `succeeded`, exit zero,
  and no execution error. The structured result is complete for all five
  requested cases on RTX 4070 with PyTorch 2.13.0+cu130/CUDA 13.0, five
  accuracy trials, 20 warmups, 100 repeats, and three alternating rounds.
- Correctness passed before timing was interpreted. All 25 trials were bitwise
  exact under the strict OR rule: `0 / 55,050,240` failed elements with zero
  maximum absolute and relative error. Every requested case reports the
  expected shape/dtype, finite values, `status=succeeded`, and no failure
  category.

| Case | I07 candidate (ms) | E01 baseline (ms) | E01 candidate (ms) | Same-job speedup | I07 / E01 - 1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.220159993 | 0.958752006 | 0.219136000 | 4.375146063x | +0.467286% |
| 9 | 0.496639997 | 0.877568007 | 0.489472002 | 1.792887036x | +1.464434% |
| 11 | 1.077247977 | 7.273471832 | 1.039360046 | 6.998029083x | +3.645313% |
| 12 | 0.153600007 | 0.957039982 | 0.153600007 | 6.230728754x | +0.000000% |
| 13 | 34.628608704 | 110.881790161 | 34.844673157 | 3.182173346x | -0.620079% |

- I07 and E01 five-case candidate-latency geometric means are
  `0.910719031 / 0.901876450 ms`. Latency falls `0.970945%`, equivalently the
  preregistered old/new metric improves `0.980465%`, exceeding the `0.75%`
  gate. Case 13 is the only regression: the table's old/new delta is
  `-0.620079%`, equivalently its latency rises `0.623948%`, below the 2% gate.
- The same-job equal-case paired-speedup geometric mean is `4.049083950x`.
  Focused baseline/candidate median sums are
  `120.948621988 / 36.746241212 ms`, a sum-ratio speedup of `3.291455616x`.
- Raw-derived baseline round medians for cases 4/9/11/12/13 (ms) are,
  respectively:
  `0.957520008/0.961519986/0.958432019`,
  `0.880639970/0.881663978/0.872448027`,
  `7.270400047/7.275519848/7.275519848`,
  `0.956463993/0.957872003/0.956271976`, and
  `110.880767822/110.882812500/110.882335663`.
- Corresponding candidate round medians (ms) are:
  `0.219136000/0.218768001/0.219136000`,
  `0.484351993/0.490496010/0.483328015`,
  `1.036288023/1.067200005/1.037312031`,
  `0.153600007/0.153600007/0.153600007`, and
  `34.844673157/34.843647003/34.844673157`. All winning-case rounds remain
  below their I07 aggregate medians; Case 12 is unchanged and Case 13's stable
  bounded regression remains inside the declared gate.
- Decision: focused `promote`. E01 passes 25/25 strict trials, exceeds the
  preregistered five-case geomean threshold, and has no case regression above
  2%. No config-prune follow-up is needed. The benchmark harness does not
  serialize Triton's selected config per key, so the end-to-end gain is
  attributable to this one launch-search category but cannot be split between
  the five added configs and sequence-aware key without a separate experiment.
  Integrate only together with the supervisor's intended direct-QKV allowlist
  pruning, then require a new ordinary unified cases 1-13 job before a
  matrix-wide promotion claim.

## I08 combined QKV prune plus autotune - unified retain-best

- Integrated the independently promoted QKV allowlist prune and sequence-aware
  autotune as unsigned commits `8d53d5d` and `79600b2`, retaining both result
  records as `94a3bb9` and `93b1f4d`. Static validation passed full Python
  compilation, 24/24 tests, and CPU BF16 smoke bitwise exactly at `0 / 128`.
- Unified ordinary job/snapshot:
  `job-1788134737177-f4ec2ab0e450c14a` /
  `0c0e40cbe6c7adaa838659fe625629773df3ef76b54d19fe6b338b6c8a9399c6`;
  base `93b1f4de5504aadf0b519d3866d32668a98ddfe0`, official CUDA BF16 Cases
  1-13, five trials, 20 warmups, 100 repeats, three rounds, state succeeded.
- Correctness passed before timing interpretation: all 65 trials were bitwise
  exact, `0 / 938,885,120` failed elements, with zero maximum absolute and
  relative error.
- Same-job paired speedup geomean was `3.343484256x`; total-call speedup was
  `555.468385458 / 216.508106381 = 2.565577773x`. Against I07, all-case
  candidate-latency geomean improved `0.301226%`, but total candidate latency
  regressed `0.069384%` and aggregate MFU fell from `16.009836%` to
  `15.998728%`.
- Versus I07 candidate medians, Cases 1-13 changed
  `+1.245561% / 0 / 0 / 0 / +0.523564% / +0.034146% / 0 /
  -0.046977% / +0.206613% / +2.459023% / +0.190478% / 0 /
  -0.660948%`. Case 11's focused autotune gain did not survive the unified
  run: speedup was `6.766399x`, not 7x. Case 13's stable regression dominates
  the total-latency and MFU result.
- Decision: combined I08 is `retain-best`; I07 remains the shared winner.
  Unified evidence separates the routes: the allowlist prune had no focused
  regressions, whereas the autotune's Case-13 regression persists and its
  Case-11 focused gain does not. Use the integration's single focused follow-up
  to revert only autotune source/tests while preserving its commits and result
  record, then run one unified prune-only Cases 1-13 job. No further follow-up
  is permitted.

## I08b prune-only integration - final follow-up candidate

- This follow-up commit removes only the launch-config and sequence-key changes from
  integrated commit `79600b2`; it retains the five-shape direct-QKV allowlist
  from `8d53d5d`, all I07 kernel math, BF16 rounding, dispatch fallbacks, and
  the complete audit history.
- Pre-GPU gate: full Cases 1-13 strict correctness must pass. Promote only if
  prune-only candidate geomean and total candidate latency both improve over
  I07 without a material per-case regression; otherwise retain I07. This is
  the final integration follow-up for the round.
- Final ordinary job/snapshot:
  `job-1788135136705-0ad3283d01e0b2e4` /
  `93c6cebc1cd3d33c1a4219ab19947663cf5ecc174e4cfbb0e59c0aaf1ba65258`;
  base `089d2b04d5bb893a44b93ea62136cc51d322dca4`, official CUDA BF16 Cases
  1-13, state succeeded, exit zero.
- Correctness passed first: all 65 trials were bitwise exact under the strict
  OR rule, `0 / 938,885,120` failed elements, with zero maximum absolute and
  relative error.
- Same-job baseline/candidate medians and paired speedups for Cases 1-13 are:
  `1.421664/0.575504/2.470294x`, `0.955376/0.097280/9.820888x`,
  `0.959488/0.126976/7.556452x`, `0.951984/0.220160/4.324055x`,
  `3.225600/1.174528/2.746295x`, `414.049286/165.839043/2.496694x`,
  `1.101200/0.498688/2.208194x`, `11.687936/10.892288/1.073047x`,
  `0.872448/0.487424/1.789916x`, `1.110016/0.499712/2.221311x`,
  `7.278592/1.077248/6.756654x`, `0.954368/0.154624/6.172185x`, and
  `110.905342/34.629631/3.202614x`.
- Paired speedup geomean is `3.355589228x`; one-call-total speedup is
  `555.473299921 / 216.273105852 = 2.568388232x`.
- Versus I07 candidate medians, Cases 1-13 changed
  `+1.242746% / +1.052632% / 0 / 0 / +0.435923% / +0.031371% / 0 /
  0 / +1.890759% / +2.459023% / 0 / -0.662247% / -0.002952%`.
  Cases 2/3/6/7/8 are source-identical cross-job observations rather than
  attributed wins; the changed allowlist routes explain Cases 1/5/10 fallback
  recovery and retained direct-QKV behavior for Cases 4/9/11/12/13.
- All-case candidate-latency geomean falls from `1.072358624` to
  `1.067105806 ms`, an old/new gain of `0.492249%`. Total candidate latency
  falls from `216.357884496` to `216.273105852 ms`, improving `0.039200%`.
  Aggregate MFU rises from `16.009836%` to `16.016112%` under the same attached
  FLOP/peak convention.
- Decision: promote prune-only I08 as the shared winner. Full-matrix strict
  correctness, candidate geomean, total candidate latency, paired speedups,
  and MFU all improve over I07. Cases 2/3 remain at or above 7x; Case 11 stays
  below at `6.756654x`, so the wider target remains incomplete.

## Case-11 HD8 PV E02 - two-warp launch

- Status: focused `promote` from shared I08 winner `a43ec01`; integration and
  unified Cases 1-13 validation remain the supervisor's responsibility.
- Target and reference: official Case 11
  `(B,S,D,H,HD)=(64,128,128,16,8)`. I08 unified job
  `job-1788135136705-0ad3283d01e0b2e4` measured candidate median
  `1.077247977 ms`, same-job speedup `6.756654x`, and bitwise-exact strict
  correctness over all five trials.
- Hypothesis: each HD8 PV launch uses 1,024 independent batch/head programs,
  but each program owns only a 16x16 accumulator with eight live output
  columns. Reducing the launch from four warps to two should lower per-program
  scheduling and register pressure while retaining ample grid-level
  parallelism. Pipeline depth remains exactly two stages, so the result is
  attributable to warp count rather than a compound launch search.
- Numerical-equivalence boundary: unchanged. Native ATen FP32 softmax remains
  outside the kernel; probabilities are explicitly rounded to BF16 before
  `tl.dot`, the dot accumulates in FP32, and context rounds to BF16 before the
  same masked eight-column store. Tile dimensions, K padding, inputs, weights,
  reduction order, output layout, eight prefix launches, and dispatch/fallback
  predicates are unchanged.
- Dispatch/fallback: unchanged exact Case-11 CUDA BF16 inference path. Case 12
  and every non-HD8 or unsupported shape continue through the established I08
  implementation.
- Preregistered gate: Cases 11 and 12 must both pass five strict correctness
  trials. Promote only if Case-11 candidate median improves by at least 1.5%
  versus `1.077247977 ms` (at most `1.061089257 ms`) and guard Case 12 does not
  regress by more than 1% versus `0.154624000 ms`. At most one follow-up may
  prune a concrete launch configuration; it may not change kernel math, tiles,
  dispatch, or fallback.
- Pre-GPU validation passed `git diff --check`, full facade/package/test Python
  compilation, and 23/23 unit tests. The prescribed CPU BF16 smoke used the
  unchanged fallback and passed bitwise exactly with `0 / 128` failures; it is
  execution/correctness evidence only, not GPU performance evidence. Ordinary
  focused GPU job identity and result are pending.
## Cases-4/12 full HD32 PV launch E01 - row split and warp specialization

- Status: focused `promote`; no shape-prune follow-up is needed. The candidate
  starts from I08 implementation `089d2b0`; the I08 audit
  head is `a43ec01`. The comparable unified job is
  `job-1788135136705-0ad3283d01e0b2e4`, snapshot
  `93c6cebc1cd3d33c1a4219ab19947663cf5ecc174e4cfbb0e59c0aaf1ba65258`.
  Its candidate medians are `0.220159993 / 0.154624000 ms` for Cases 4/12.
- Hypothesis: Case 4 launches only 64 batch/head programs and its monolithic
  128-row program carries a 128x32 FP32 accumulator across eight warps. Since
  output rows are independent, two 64-row/four-warp programs can expose 128
  blocks and reduce per-block accumulator/shared-memory pressure without
  splitting the K=128 reduction. Case 12 already has 256 batch/head programs;
  its 32x32 accumulator should need only two warps instead of four.
- Implementation and attribution: only the exact full-HD32 call is recognized
  inside `pv_context.py`. Case 4 splits M=128 into two disjoint M=64 programs,
  each with four warps and two stages. Case 12 retains one M=32 program but
  uses two warps and two stages. Prefix row-32/64 calls, HD8 PV, and every other
  tile keep the historical launch configuration.
- Numerical-equivalence boundary: each Case-4 program still loads all K=128
  FP32 probabilities, explicitly rounds them to BF16, performs one unsplit
  BF16 dot with FP32 accumulation, rounds context to BF16, and writes a unique
  64-row region. Case 12 changes only warp distribution. Native FP32 softmax,
  BF16 probability materialization, FP32 dot accumulation, BF16 context,
  layout, dispatch, fallback, baseline, weights, and public interface are
  unchanged.
- Preregistered gate: both focused Cases 4/12 must complete all 10 strict
  trials under `abs<0.002 OR rel<0.02`. Against I08 candidate medians, the
  equal-case candidate-latency geomean must improve at least `0.75%`, and no
  case may regress more than `1.5%`. At most one concrete shape-prune follow-up
  may remove one regressing shape; no other configuration search is allowed.
- Local checks: `git diff --check`, full facade/package/test Python compilation,
  and 23/23 unit tests passed. The prescribed CPU BF16 smoke passed bitwise
  exactly with `0 / 128` failures. Submit one ordinary shared-queue CUDA BF16
  job with five trials, 20 warmups, 100 repeats, and three rounds. CPU timing
  is not GPU evidence.
## I08 Case 11 direct-QKV fixed launch E01

- Status: preregistered candidate from shared winner `a43ec01`. The earlier
  expanded-autotune focused job `job-1788134123272-8f4753aca47ede5c`
  measured Case 11 at `1.039360046 ms`, a `3.645313%` gain over I07, but the
  combined unified job retained only `0.190478%` and introduced a stable
  Case-13 regression. Because the harness does not serialize Triton's selected
  config, that result was not auditable enough to retain globally.
- Hypothesis: for the exact Case 11 projection `(B,S,D,H)=(64,128,128,16)`,
  the GEMM is `M=8192, N=384, K=128`. A fixed `64x128x32`, four-warp,
  three-stage launch uses exactly three column tiles, exposes 384 CTAs across
  the large M dimension on the RTX 4070, and avoids the eight-warp scheduling
  overhead of the original `64x128` config. It is one of the previously
  correctness-proven expanded-autotune configs, selected here without
  extending the global search or its cache key.
- Equivalence boundary: launch shape and scheduling change only. The raw I08
  Triton kernel retains its four increasing K=32 reductions, FP32 dot
  accumulation, bias addition, explicit BF16 rounding, and direct contiguous
  `[3,B,H,S,HD]` address mapping. The direct route's causal/inference/CUDA/
  BF16/contiguity guards and exact-shape dispatch remain unchanged.
- Dispatch and fallback: only exact `(64,128,128,16)` bypasses the three-config
  autotuner and launches the raw kernel with the fixed config. Case 13 and all
  other supported direct-QKV shapes keep the I08 autotuner; unsupported shapes,
  masks, training, gradient, dtype, device, and layout conditions retain their
  existing native packed-linear fallback.
- Preregistered GPU gate: Cases 11 and 13 must pass all 10 strict trials.
  Promote only if Case 11 candidate latency improves at least `2%` versus the
  prune-only I08 reference `1.077248 ms`, Case 13 regression is at most `0.5%`
  versus `34.629631 ms`, and the three Case-11 round medians preserve the
  improvement sign. At most one follow-up may replace this fixed config with a
  different already-tested config if the measured launch misses the gate.
- Static validation passed `git diff --check`, full Python compilation, 24/24
  unit tests, and the prescribed CPU BF16 smoke bitwise exactly at `0 / 128`.
  The CPU smoke exercises the fallback and is not GPU performance evidence.
  Ordinary job/snapshot, raw timings, and the deterministic decision are
  pending.

### E01 deterministic review

- Identity and scope: ordinary job
  `job-1788135766924-5c720d633274e237` evaluated immutable snapshot
  `3360c3ea95c2ee0fc062374b6cfcc44d88271e00985c8f4fba55bb22dda9983f`
  from implementation commit `f5ebf2ee28e16b9e62e6a98ae58ecf864743e3d8`.
  `job.json` records exactly official Cases 11 and 13, CUDA BF16, pinned
  Python 3.12.14 and package identity, terminal state `succeeded`, exit zero,
  and no execution error. The structured result is complete on RTX 4070 with
  PyTorch 2.13.0+cu130/CUDA 13.0, five accuracy trials, 20 warmups, 100
  repeats, and three alternating rounds.
- Correctness passed before timing interpretation. All 10 trials were bitwise
  exact under the strict OR rule: `0 / 47,185,920` failed elements with zero
  maximum absolute and relative error. Both cases report the expected shape,
  dtype, finite values, and no failure category.
- Same-job baseline/candidate medians and speedups were
  `7.263232231 / 1.065984011 ms / 6.813640879x` for Case 11 and
  `110.875648499 / 34.634767532 ms / 3.201281729x` for Case 13. Their
  equal-case speedup geometric mean was `4.670373010x`; focused median sums
  were `118.138880730 / 35.700751543 ms` (`3.309142683x`).
- Against prune-only I08 job `job-1788135136705-0ad3283d01e0b2e4`, Case 11
  fell from `1.077247977` to `1.065984011 ms`, an old/new improvement of
  `1.056673%`; this misses the preregistered `2%` gate. Case 13 rose only from
  `34.629631042` to `34.634767532 ms`, a `0.014833%` regression, passing its
  `0.5%` guard.
- Raw-derived baseline round medians were
  `7.261184216/7.263232231/7.263232231 ms` for Case 11 and
  `110.876304626/110.876670837/110.874752045 ms` for Case 13. Candidate
  round medians were `1.063935995/1.068032026/1.065984011 ms` and
  `34.634239197/34.636287689/34.634750366 ms`, respectively. Every Case-11
  round improves over the I08 aggregate by `1.251201%/0.862891%/1.056673%`,
  with only `0.384988%` max/min spread, so the sign is stable but its magnitude
  remains below the gate.
- Decision: `retain-best`. E01 is correct, stable, and protects Case 13, but
  does not recover enough of the prior focused autotune result. Use the single
  authorized replacement follow-up to test the already-searched
  `128x128x32`, eight-warp, three-stage config; if that misses, retain I08 and
  close this route without another follow-up.

## I08 Case 11 direct-QKV fixed launch E02 - final replacement

- Status: preregistered final candidate. E01's fixed `64x128x32`, four-warp,
  three-stage launch was bitwise exact and improved every Case-11 round, but
  its aggregate gain was only `1.056673%`, below the `2%` gate. The single
  authorized replacement selects another config from the prior focused
  autotune search rather than widening the search again.
- Hypothesis: exact Case 11 remains `M=8192, N=384, K=128`. Fixing
  `128x128x32`, eight warps, and three stages halves E01's launch from 384 to
  192 CTAs while preserving exactly three N tiles. The launch still exposes
  more than four CTAs per RTX 4070 SM, and its larger M tile can amortize
  scheduling and direct-layout store setup enough to reach the missing `2%`
  end-to-end threshold.
- Attribution and equivalence: only `block_rows` and `num_warps` change from
  E01. The raw I08 kernel, K=32 reduction order, FP32 accumulation, bias,
  explicit BF16 rounding, direct output addresses, exact Case-11 dispatch,
  Case-13 autotuner, and all other fallbacks are identical.
- Final gate: Cases 11 and 13 must again pass all 10 strict trials. Promote only
  if Case 11 improves at least `2%` versus I08 `1.077247977 ms`, Case 13
  regression remains at most `0.5%` versus `34.629631042 ms`, and all three
  Case-11 rounds keep the improvement sign. There is no further follow-up;
  failure retains I08 and closes this route.
- Static validation passed `git diff --check`, full Python compilation, 24/24
  unit tests, and the prescribed CPU BF16 fallback smoke bitwise exactly at
  `0 / 128`. This is not GPU performance evidence. Ordinary benchmark
  evidence is pending.
## Case-11 HD8 PV E02 deterministic review

- Deterministic identity and scope: ordinary job
  `job-1788135643098-82e65f3390f8207e`, immutable snapshot
  `f24b727a3f7511eba9e59ab11d59acdd5382ac152d4a3faa83e3b9048897c4bc`,
  base optimization commit `b87e6e69f83d5e675428a66b2d06e98f5c8be84a`.
  `job.json` records exactly official Cases 11/12, CUDA BF16, five accuracy
  trials, 20 warmups, 100 repeats, and three alternating rounds; it completed
  with state `succeeded`, exit zero, and no error using pinned Python 3.12.14.
  The structured result identifies RTX 4070, PyTorch 2.13.0+cu130, and CUDA
  13.0.
- Correctness passed before timing interpretation. Both requested cases and all
  10 trials were bitwise exact under strict
  `abs_error < 0.002 OR abs_error < 0.02 * abs(reference)` correctness:
  `0 / 6,553,600` failed elements, with zero maximum absolute and relative
  error. Both cases report complete 300-sample baseline and candidate timing.
- Case 11 same-job baseline/candidate medians were
  `7.262207985 / 1.047551990 ms`, a `6.932551374x` speedup. Against I08
  candidate `1.077247977 ms`, the old/new gain is `2.834798%` (latency
  reduction `2.756653%`), exceeding the preregistered 1.5% gate. Baseline
  round medians were `7.260159969 / 7.262207985 / 7.263232231 ms`; candidate
  round medians were `1.029631972 / 1.049600005 / 1.029119968 ms`.
- Guard Case 12 same-job baseline/candidate medians were
  `0.947200000 / 0.153600007 ms`, a `6.166666376x` speedup. Against I08
  candidate `0.154624000 ms`, the old/new gain is `0.666662%`, so the guard
  does not regress and remains within the 1% limit. Baseline round medians were
  `0.944127977 / 0.948640019 / 0.948591977 ms`; all three candidate round
  medians were `0.153600007 ms`.
- Decision: focused `promote`. Strict correctness passed, Case 11 cleared the
  1.5% improvement gate, and the unaffected Case-12 guard did not regress.
  The one-parameter result supports the two-warp small-accumulator hypothesis;
  no config-prune follow-up is needed. This partial-case result is not a full
  matrix performance claim.
## Cases-4/12 full HD32 PV launch E01 deterministic review

- Focused ordinary job/snapshot:
  `job-1788135727706-d92485b8ecce4900` /
  `59184d0f31621734ead631a2fcf72bb611623c3e0909ec9be13f1e3ba5867b3d`;
  submitted implementation commit
  `cdb5ccb3dbd3f19eee73af8436a87eaec746bb3b`. Job identity matches this
  worktree and snapshot. It ran only requested Cases 4/12 on RTX 4070 with
  Python 3.12.14, PyTorch 2.13.0+cu130/CUDA 13.0, CUDA BF16, five accuracy
  trials, 20 warmups, 100 repeats, and three alternating rounds. The job
  completed both requested cases, exited zero, and has no failure category.
- Correctness passed before timing interpretation. All 10 trials were bitwise
  exact under the strict OR rule: `0 / 2,621,440` failed elements, with zero
  maximum absolute and relative error. Both structured case records report
  `status=succeeded` and `correctness_passed=true`.
- Same-job medians and paired speedups: Case 4 baseline/candidate
  `0.954512000 / 0.211968005 ms` (`4.503094707x`); Case 12
  `0.956384003 / 0.153600007 ms` (`6.226458058x`). Each side contains 300 raw
  samples. Baseline round medians are
  `0.952319980 / 0.953855991 / 0.956496000 ms` for Case 4 and
  `0.955712020 / 0.956863999 / 0.955888003 ms` for Case 12. Candidate round
  medians are `0.211071998 / 0.210975997 / 0.212384000 ms` and
  `0.153600007 / 0.153600007 / 0.153600007 ms`, respectively.
- Against I08 candidate medians, Case 4 changes
  `0.220159993 -> 0.211968005 ms` (`+3.864728%` old/new), and Case 12 changes
  `0.154624000 -> 0.153600007 ms` (`+0.666662%` old/new). Neither case
  regresses, so the `<=1.5%` per-case gate passes.
- The equal-case candidate-latency geometric mean changes
  `0.184504793 -> 0.180439151 ms`, an old/new improvement of `2.253193%`.
  This exceeds the preregistered `0.75%` gate. Decision: `promote` E01 as a
  focused winner for later integration and unified validation. The Case-4
  row split and Case-12 warp specialization both improve their independently
  dispatched shapes, so the single permitted shape-prune follow-up is not
  submitted.

### E02 deterministic review

- Identity and scope: ordinary job
  `job-1788136170239-bb023eed0ea6b239` evaluated immutable snapshot
  `69a922f83b3ac61a74cfaa34b1c48cac9f96381e6d680db757304940f800ffb4`
  from implementation commit `b964646623e6489ff64ce1f028ca4fc5023141c2`.
  `job.json` records exactly official Cases 11 and 13, CUDA BF16, pinned
  Python 3.12.14 and package identity, state `succeeded`, exit zero, and no
  error. The structured result is complete on RTX 4070 with PyTorch
  2.13.0+cu130/CUDA 13.0, five accuracy trials, 20 warmups, 100 repeats, and
  three alternating rounds.
- Correctness passed before timing interpretation. All 10 trials were bitwise
  exact under the strict OR rule: `0 / 47,185,920` failed elements with zero
  maximum absolute and relative error. Both cases report the expected shape,
  dtype, finite outputs, and no failure category.
- Same-job baseline/candidate medians and speedups were
  `7.260159969 / 1.053696036 ms / 6.890184379x` for Case 11 and
  `110.874626160 / 34.634750366 ms / 3.201253798x` for Case 13. Their
  equal-case speedup geometric mean was `4.696512420x`; focused median sums
  were `118.134786129 / 35.688446403 ms` (`3.310168921x`).
- Against prune-only I08 job `job-1788135136705-0ad3283d01e0b2e4`, Case 11
  fell from `1.077247977` to `1.053696036 ms`, an old/new improvement of
  `2.235174%`, passing the `2%` gate. Case 13 rose from `34.629631042` to
  `34.634750366 ms`, only a `0.014783%` regression, passing the `0.5%` guard.
- Raw-derived baseline round medians were
  `7.257423878/7.261200190/7.261184216 ms` for Case 11 and
  `110.873596191/110.873596191/110.875648499 ms` for Case 13. Candidate
  round medians were `1.047551990/1.073151946/1.051648021 ms` and
  `34.634750366/34.635776520/34.634750366 ms`, respectively. Every Case-11
  round remains faster than the I08 aggregate by
  `2.834798%/0.381682%/2.434270%`; the `2.443789%` max/min spread makes the
  magnitude noisy, but the preregistered improvement sign is stable.
- Decision: focused `promote`. E02 passes strict correctness, the aggregate
  Case-11 threshold, the Case-13 guard, and the per-round sign requirement.
  This is the final fixed-config attempt and there is no further follow-up.
  Integration still requires a new ordinary unified Cases 1-13 job before any
  matrix-wide promotion claim; focused same-job Case-11 speedup remains
  `6.890184x`, below the wider 7x target.

## Winner integration I09 - combined PV launch and Case-11 QKV specializations

- Integrated scope: layer the independently attributable HD8 PV two-warp
  launch, full HD32 PV Case-4 row split / Case-12 two-warp launch, and final
  fixed Case-11 direct-QKV tile on prune-only I08. The source changes remain
  separate unsigned commits (`07132c9`, `6e2f66c`, `8bb07bd`, and `2415330`)
  with their original dispatch, equivalence, fallback, validation, and rollback
  explanations; the associated focused-result commits are also preserved.
- Static integration validation passed `git diff --check`, full Python
  compilation, 24/24 unit tests, and the prescribed CPU BF16 fallback smoke
  bitwise exactly at `0 / 128`. CPU execution is not GPU performance evidence.
- Unified deterministic identity: ordinary job
  `job-1788136508167-1b1b4acd2b1d637d` evaluated immutable snapshot
  `067aafc231ae874976af42e0aa243f6cea70a0ea585a166d56e8e95a47c07bdf`
  from the winner integration worktree. `job.json` records exactly official
  Cases 1-13, CUDA BF16, pinned Python 3.12.14, terminal state `succeeded`,
  exit zero, and no error. The structured result is complete on RTX 4070 with
  PyTorch 2.13.0+cu130 and CUDA 13.0.
- Correctness passed before performance interpretation. All requested cases
  and all 65 trials were bitwise exact under the strict OR rule:
  `0 / 938,885,120` failed elements, zero maximum absolute and relative error,
  expected shape/dtype, finite outputs, and no failure category. Every case has
  300 baseline and 300 candidate timing samples (20 warmups, 100 repeats, three
  alternating rounds).
- Same-job per-case baseline/candidate medians and paired speedups were:

  | Case | Baseline ms | Candidate ms | Speedup |
  |---:|---:|---:|---:|
  | 1 | 1.421312 | 0.576608 | 2.464954x |
  | 2 | 0.959216 | 0.098304 | 9.757649x |
  | 3 | 0.954448 | 0.126976 | 7.516759x |
  | 4 | 0.959104 | 0.211968 | 4.524758x |
  | 5 | 3.222800 | 1.174528 | 2.743911x |
  | 6 | 414.120972 | 165.890045 | 2.496358x |
  | 7 | 1.099776 | 0.498688 | 2.205339x |
  | 8 | 11.689216 | 10.892288 | 1.073164x |
  | 9 | 0.871760 | 0.487424 | 1.788505x |
  | 10 | 1.111040 | 0.500736 | 2.218814x |
  | 11 | 7.277568 | 1.061888 | 6.853423x |
  | 12 | 0.956688 | 0.153600 | 6.228437x |
  | 13 | 110.907394 | 34.630657 | 3.202578x |

- Aggregate metrics keep distinct denominators. The equal-case geometric mean
  of the 13 paired speedups is `3.368691310x`. Summing the 13 medians gives
  `555.551294148 / 216.303710565 ms`, or a one-call-total-latency speedup of
  `2.568385409x`. Aggregate MFU is `16.013845757%`, using
  `sum(FLOPs) / sum(candidate median latency) / 58.25e12` and
  `F=L*(8BSD^2 + 4BS^2D + 4BSDF)`; this is a supervisor-derived convention,
  not a harness field.
- Against I08 candidate medians, Cases 4/11/12 improve
  `3.864728% / 1.446480% / 0.666662%`. Candidate-latency geometric mean falls
  from `1.067105806` to `1.063487668 ms`, an old/new gain of `0.340214%`.
  Candidate median sum rises only from `216.273105852` to `216.303710565 ms`
  (`0.014149%` regression), dominated by a `0.030745%` Case-6 timing shift;
  this is below a material-regression threshold and the same-job total-speedup
  change is only `-0.000110%` relative.
- Decision: promote I09 as the shared winner. The three targeted dispatches
  retain strict matrix-wide correctness and improve their intended cases with
  no material cross-case regression. Cases 2/3 remain above the requested 7x
  level; Case 11 reaches `6.853423x` and Case 12 `6.228437x`, so the wider
  multi-case 7-10x objective remains open.

## I09 small projection/layout E01 - Case-7 HD8 PV into final layout

- Status: preregistered focused candidate from shared I09 commit
  `15196cbaa9b6139bae1a6b134969a66351d8ef19`.
- Targets and anchors: exact official CUDA BF16 Cases 7/9/10. I09 candidate
  medians are `0.498688012 / 0.487423986 / 0.500735998 ms`; their geometric
  mean is `0.495574782 ms`. Only Case 7 changes in E01. Cases 9/10 remain
  byte-identical regression anchors so a Case-7 gain must survive the shared
  runtime and three-case promotion rule.
- Bottleneck and hypothesis: Case 7 executes two 64-row causal prefixes in
  each of four layers. Each prefix separately rounds native-softmax FP32
  probabilities to BF16 and invokes native BF16 PV, after which each layer
  copies the complete head-major HD8 context into sequence-major projection
  layout. An exact-shape HD8 PV kernel can fuse only the probability rounding
  with PV and write both disjoint prefixes directly into the final backing,
  removing eight cast/native-PV launch pairs and four layout copies per call.
- Numerical boundary: triangular QK, scale, prefix split, and native FP32
  softmax are unchanged. E01 explicitly rounds probabilities to BF16 before
  `tl.dot`, loads BF16 V, accumulates the PV dot in FP32, and rounds context to
  BF16 before the unchanged output projection. HD8 is padded to 16 output
  columns with exact zeros; columns 8-15 are masked on load and store, so no
  live term changes. This follows the separately validated Case-11 HD8 PV
  boundary without changing that kernel or its dispatch.
- Dispatch and fallback: only eval plus inference/no-grad CUDA BF16, causal,
  no effective token mask, exact query/value/context shape `(64,4,128,8)`,
  exact 64-row prefixes ending at keys 64 and 128, contiguous FP32
  probabilities, and unit-stride V/context columns use the new kernel. It
  allocates contiguous `[64,128,4,8]` backing and exposes a strided BHSD view;
  the following transpose is already contiguous. Cases 9/10, Cases 11/12,
  training, gradients, masks, CPU, other dtype/shape/layout, and custom calls
  retain I09 unchanged.
- Static and CPU validation: `git diff --check`, full facade/package/test
  compilation, and 25/25 unit tests passed, including exact dispatch,
  unchanged Case-11 isolation, explicit BF16/dot/store boundary, and
  sequence-major alias invariants. The prescribed CPU BF16 smoke was bitwise
  exact at `0 / 128`; it validates fallback only, not CUDA correctness or
  performance.
- Preregistered ordinary GPU gate: run official Cases 7/9/10 with five
  accuracy trials, 20 warmups, 100 repeats, and three alternating rounds.
  Read timing only if all 15 trials pass strict
  `abs_error < 0.002 OR abs_error < 0.02 * abs(reference)` correctness. Promote
  only if the three-case candidate-latency geometric mean improves by at least
  `0.75%` versus I09 and no case regresses by more than `1.5%`; otherwise
  retain I09. At most one evidence-specific follow-up is permitted.
- Ordinary focused job/snapshot:
  `job-1788138042767-8089c8a3186f6ce5` /
  `1efdf0152dd9cc2bf5d6b1e4a6686d8a1205d15837a7c694901e281c5b87c7d9`;
  submitted base commit `42700cf8774f9a148657d3e2e85c1948e278ea86`.
  `job.json` records exactly Cases 7/9/10, CUDA BF16, the pinned Python
  executable, terminal state `succeeded`, exit zero, and no error. The
  structured result is complete and subset-complete on NVIDIA GeForce RTX
  4070 with Python 3.12.14, PyTorch 2.13.0+cu130, and CUDA 13.0.
- Correctness passed before timing was interpreted. All requested cases and
  all 15 trials were bitwise exact under the strict elementwise OR rule:
  `0 / 11,796,480` failed elements, zero maximum absolute and relative error,
  expected output shapes/dtypes, finite values, and no failure category.
- Same-job medians and paired speedups were Case 7
  `1.096704006 / 0.456703991 ms` (`2.401345354x`), Case 9
  `0.868351996 / 0.484351993 ms` (`1.792811858x`), and Case 10
  `1.103871942 / 0.495615989 ms` (`2.227272659x`). Candidate mean/p90/min
  values were respectively `0.471226030/0.497664005/0.454656005`,
  `0.486308803/0.493568003/0.480255991`, and
  `0.496941330/0.496655998/0.492543995 ms`.
- Raw-derived baseline round medians, each from 100 structured samples, were
  Case 7 `1.168383956/1.095679998/1.095679998`, Case 9
  `0.864256024/0.869376004/0.868351996`, and Case 10
  `1.100800037/1.104895949/1.104895949 ms`. Candidate round medians were Case 7
  `0.496639997/0.455680013/0.456703991`, Case 9
  `0.482304007/0.484351993/0.485376000`, and Case 10
  `0.495615989/0.495615989/0.496639997 ms`.
- Versus I09 candidate medians, Case 7 improves `9.192830%`; unchanged anchors
  Cases 9/10 appear `0.634248%/1.033060%` faster but those deltas are runtime
  observations, not attributable to the Case-7-only dispatch. The three-case
  candidate geomean falls from `0.495581265` to `0.478608494 ms`, an old/new
  gain of `3.546274%`, exceeding the preregistered `0.75%` gate with no case
  regression.
- Timing caveat: Case 7 is bimodal across rounds. Its first candidate round
  (`0.496639997 ms`) is only `0.412417%` below the I09 aggregate, while rounds
  two and three are clearly faster by `9.438202%/9.192830%`. The first
  baseline round is also elevated (`1.168383956 ms` versus
  `1.095679998/1.095679998`), which supports a run-regime effect rather than a
  correctness issue, but the focused aggregate alone cannot prove that the
  faster regime will persist in a unified matrix run.
- Decision: focused `promote`; no follow-up is submitted. Strict correctness,
  aggregate gain, and per-case regression gates pass. Shared-winner promotion
  still requires integrating only implementation commit `42700cf` and running
  a new ordinary unified Cases 1-13 job; that job must reproduce Case-7 benefit
  without a material matrix regression and is the final authority on the
  bimodal timing caveat.

## Winner integration I10 - Case-7 final-layout HD8 PV

- Integrated source identity: I09 plus the independently attributable Case-7
  implementation/result commits `cbd17ed` / `f9d9444`. The integration changes
  no other declared case dispatch. `git diff --check`, complete Python
  compilation, 25/25 unit tests, and the prescribed CPU BF16 fallback smoke
  (`0 / 128`) passed before GPU submission; CPU timing is not GPU evidence.
- Unified deterministic identity: ordinary job
  `job-1788138587212-0aef822a88ddb071` evaluated immutable snapshot
  `6d6ffc8a7479137e4d463853472cb31b1704968d63bc8241c427b075ae90f0f4`
  from the winner worktree. It records exactly official Cases 1-13, CUDA BF16,
  pinned Python 3.12.14, RTX 4070, PyTorch 2.13.0+cu130, CUDA 13.0, state
  `succeeded`, exit zero, and no execution error.
- Correctness passed before timing interpretation. All 13 cases and all 65
  trials were bitwise exact under the strict OR rule: `0 / 938,885,120` failed
  elements, zero maximum absolute and relative error, expected shape/dtype,
  finite outputs, and no failure category. Every case contains 300 baseline
  and 300 candidate samples from 20 warmups, 100 repeats, and three alternating
  rounds.
- Same-job baseline/candidate medians and paired speedups were:

  | Case | Baseline ms | Candidate ms | Speedup |
  |---:|---:|---:|---:|
  | 1 | 1.422336 | 0.578560 | 2.458407x |
  | 2 | 0.956768 | 0.098304 | 9.732747x |
  | 3 | 0.958368 | 0.126976 | 7.547631x |
  | 4 | 0.963536 | 0.211968 | 4.545667x |
  | 5 | 3.223552 | 1.176576 | 2.739774x |
  | 6 | 414.055420 | 165.841408 | 2.496695x |
  | 7 | 1.100800 | 0.458752 | 2.399554x |
  | 8 | 11.687936 | 10.892288 | 1.073047x |
  | 9 | 0.872448 | 0.495616 | 1.760331x |
  | 10 | 1.110016 | 0.500736 | 2.216769x |
  | 11 | 7.276544 | 1.063936 | 6.839269x |
  | 12 | 0.957440 | 0.153600 | 6.233333x |
  | 13 | 110.907394 | 34.631680 | 3.202484x |

- The modified Case 7 reproduces the focused benefit in every unified round:
  candidate round medians are `0.457727998 / 0.459776014 / 0.458752006 ms`
  versus I09 aggregate `0.498688012 ms`. Its unified median improves
  `8.705358%` old/new and same-job paired speedup reaches `2.399554x`. This
  resolves the focused job's slow-first-round caveat.
- Aggregate metrics keep their denominators separate. Equal-case paired
  speedup geometric mean is `3.386436689x`. Summed medians are
  `555.492558271 / 216.230399534 ms`, a one-call-total-latency speedup of
  `2.568984562x`. Aggregate MFU is `16.019275111%`, using
  `sum(FLOPs) / sum(candidate median latency) / 58.25e12` and
  `F=L*(8BSD^2 + 4BS^2D + 4BSDF)`; this is a supervisor-derived convention,
  not a harness field.
- Against I09 candidate medians, all-case candidate geomean falls from
  `1.063487668` to `1.058589026 ms`, an old/new gain of `0.462752%`.
  Candidate median sum falls from `216.303710565` to `216.230399534 ms`, a
  gain of `0.033904%`. Unmodified Cases 1/5/9/11/13 show ordinary run-to-run
  deltas (`-0.337388% / -0.174066% / -1.652893% / -0.192494% /
  -0.002952%`); these are not attributed to Case 7 and do not offset either
  aggregate improvement.
- Decision: promote I10 as the shared winner. Matrix-wide strict correctness
  passes, the only changed case shows a large stable benefit in all unified
  rounds, and both candidate-geomean and candidate-sum objectives improve.
  Cases 2/3 remain above 7x; the broader 7-10x multi-case target remains open.

## I10 Cases-6/13 reduced masked-future chunk geometry E01

- Targets: compute-heavy CUDA BF16 Cases 6/13. I10 unified candidate medians
  are `165.841408 / 34.631680 ms`, together accounting for about 92.7% of the
  Cases 1-13 candidate median sum.
- Hypothesis: the compact causal path computes every key through the end of a
  chunk, then masks future positions inside that chunk. Case 6's two 64-row
  chunks materialize 12,288 score/probability positions per batch/head; four
  32-row chunks materialize 10,240, a 16.7% reduction. Case 13's four 256-row
  chunks materialize 655,360 positions; eight 128-row chunks materialize
  589,824, a 10% reduction. These large grids should amortize the extra
  launches while reducing QK, native softmax, probability traffic, and PV work.
- Change and scope: select 32 rows only for exact Case 6
  `(10000,128,128,H4)` and 128 rows only for exact Case 13
  `(64,1024,128,H4)`. Every other declared/custom shape retains I10 chunking.
- Numerical boundary: QKV, triangular QK math, BF16 score/scale boundaries,
  causal predicate, native FP32 softmax, BF16 probability boundary, BF16 V,
  FP32 PV accumulation, BF16 context, output layout, projections, FFN, weights,
  and public interface are unchanged. Removing masked `-inf` columns and zero
  PV terms changes native reduction widths, so strict correctness remains the
  first empirical gate despite algebraic equivalence.
- GPU plan: after static and CPU fallback validation, submit one ordinary CUDA
  BF16 Cases 6/13 job with five trials, 20 warmups, 100 repeats, and three
  rounds. Interpret performance only if all ten trials pass strict
  `abs_error < 0.002 OR relative_error < 0.02`. Promote focused E01 if the
  two-case candidate-latency geomean improves at least `1.0%` versus I10,
  neither case regresses above `2.0%`, and round medians are stable. One
  evidence-specific follow-up may restore I10 chunking for one incorrect or
  regressing shape while preserving a proven winner in the other.
- E01 deterministic result: ordinary job
  `job-1788146611282-a91535044c5d94ba` evaluated implementation
  `269d4c836dad9d6e997014178d04314518e0aeca` in immutable snapshot
  `115f8acceb3b41d6e9feb1e3536f8ed639cd3a96eecf1e25ecc7716f7178cba9`.
  It requested and executed exactly Cases 6/13 with pinned Python and CUDA
  BF16 on RTX 4070. `job.json` records terminal failed, exit 2, and no
  execution error; the structured result is complete.
- Case 6 passed all five trials bitwise exactly with `0 / 819,200,000`
  failures. Same-job baseline/candidate medians were
  `414.206970215 / 164.261886597 ms` (`2.521625550x`). Against I10 candidate
  `165.841408 ms`, the 32-row chunk improves `0.961587%` old/new. Its three
  candidate round medians are stable at
  `164.258308411 / 164.261886597 / 164.263420105 ms`.
- Case 13 failed all five trials with `37,614 / 41,943,040` strict-rule
  failures, maximum absolute error `0.046875`, and maximum relative error
  `3,509,521,408`; no Case-13 performance result exists. Reducing its native
  softmax and matmul widths does not preserve the strict four-layer output.
- E01 decision: `reject-incorrect` as a two-shape candidate, with precise
  shape-specific evidence for the one authorized E02. Restore only Case 13 to
  I10's 256-row chunk, retain Case 6's correct/stable 32-row chunk, and rerun
  the unchanged Cases 6/13 scope. E02 must pass all ten trials; promote only if
  Case 6 retains at least a `0.75%` gain versus I10 and Case 13 does not regress
  above `1.0%` from `34.631680 ms`. This exhausts the route's follow-up.
- E02 implementation is the evidence-specific one-line dispatch prune: exact
  Case 13 again selects I10's 256-row chunk, while exact Case 6 remains at 32
  rows. Kernel math, all launch configurations, native operations, and every
  other dispatch are byte-identical to E01/I10 as applicable. Diff-check,
  complete compilation, 26/26 unit tests, and CPU BF16 fallback at `0 / 128`
  failures pass before the final GPU submission.
- E02 deterministic result: final ordinary job
  `job-1788146941086-39fff544029cd5dc` evaluated implementation
  `9badbbb768106c30d16a5f641fd3f147c51543ae` in immutable snapshot
  `618b329215f1be1306de9dacbd2246d58f78d7204c2bff12d46542aba7cb54e5`.
  It executed exactly Cases 6/13 with pinned Python and CUDA BF16 on RTX 4070;
  state succeeded, exit zero, no error, complete result, and 300 samples per
  implementation/case.
- Correctness passed before timing interpretation. All ten trials were bitwise
  exact under the strict OR rule: `0 / 861,143,040` failures and zero maximum
  absolute/relative error. This both reproduces Case 6's E01 equivalence and
  confirms the Case-13 prune restored I10 behavior.
- Same-job baseline/candidate medians and speedups were Case 6
  `414.221313477 / 164.272125244 ms / 2.521555698x` and Case 13
  `110.914237976 / 34.675712585 ms / 3.198614526x`. Against I10, Case 6
  improves `0.955295%`, clearing the E02 `0.75%` gate; Case 13 regresses only
  `0.126984%`, inside the `1.0%` guard.
- Candidate round medians were Case 6
  `164.270080566 / 164.273147583 / 164.276741028 ms` and Case 13
  `34.672641754 / 34.678783417 / 34.675712585 ms`, with stable narrow spreads.
  The two-case candidate geomean improves `0.412697%`; their summed medians
  improve `200.473088000 -> 198.947837830 ms` (`0.766658%`).
- Decision: focused `promote` E02. It passes all final correctness, Case-6
  improvement, Case-13 guard, and stability gates. The route has exhausted its
  follow-up; integrate its auditable E01/result/E02/result commits into I10 and
  require a new ordinary unified Cases 1-13 job before a shared-winner claim.

## Winner integration I11 - Case-6 reduced chunk geometry

- Integrated history: I10 plus auditable E01/result/E02/result commits
  `46021db / 8809977 / 1903e4a / 0ac31c2`. The net runtime difference from
  I10 is only exact Case 6 `(10000,128,128,H4)` chunk size `64 -> 32`; exact
  Case 13 is restored to I10's 256 rows. Diff-check, complete compilation,
  26/26 unit tests, and CPU BF16 fallback (`0 / 128`) passed before GPU use.
- Unified deterministic identity: ordinary job
  `job-1788147316206-f1b67fdd1f5bb752` evaluated immutable snapshot
  `56aa1ab1aee54880b462b82d1ea08f93130610f304f075281d9496aff45ab9b1`
  from base `0ac31c2364e5f11f09c3d3fe9315ef281860df14`. It records exactly official
  Cases 1-13, CUDA BF16, pinned Python 3.12.14, RTX 4070, PyTorch
  2.13.0+cu130/CUDA 13.0, state succeeded, exit zero, and no error.
- Correctness passed before performance interpretation. All 13 cases and all
  65 trials were bitwise exact under the strict OR rule:
  `0 / 938,885,120` failures, zero maximum absolute/relative error, complete
  output, and 300 baseline plus 300 candidate samples per case.
- Same-job baseline/candidate medians and paired speedups were:

  | Case | Baseline ms | Candidate ms | Speedup |
  |---:|---:|---:|---:|
  | 1 | 1.422336 | 0.577536 | 2.462766x |
  | 2 | 0.961344 | 0.098304 | 9.779297x |
  | 3 | 0.956768 | 0.126976 | 7.535030x |
  | 4 | 0.960144 | 0.211968 | 4.529665x |
  | 5 | 3.223552 | 1.171456 | 2.751748x |
  | 6 | 414.051056 | 164.407303 | 2.518447x |
  | 7 | 1.103072 | 0.459776 | 2.399151x |
  | 8 | 11.695104 | 10.897408 | 1.073201x |
  | 9 | 0.872448 | 0.487424 | 1.789916x |
  | 10 | 1.110720 | 0.500736 | 2.218175x |
  | 11 | 7.278592 | 1.061888 | 6.854388x |
  | 12 | 0.956496 | 0.154624 | 6.185948x |
  | 13 | 110.906364 | 34.628609 | 3.202738x |

- The changed Case 6 reproduces the focused win in all unified rounds:
  `164.404220581 / 164.406265259 / 164.410369873 ms`. Its aggregate improves
  `0.872288%` from I10 `165.841408 ms`. Unchanged per-case movements are
  ordinary cross-job observations and are not attributed to the dispatch.
- Aggregate denominators remain separate. Equal-case paired-speedup geomean is
  `3.393298332x`. Baseline/candidate median sums are
  `555.497996122 / 214.784007043 ms`, a same-job total speedup of
  `2.586309864x`. Aggregate MFU is `16.127151669%` using the established
  `F=L*(8BSD^2 + 4BS^2D + 4BSDF)` convention and 58.25e12 peak.
- Versus I10, candidate median sum improves
  `216.230399534 -> 214.784007043 ms` (`0.673417%` old/new), and candidate
  latency geomean improves `1.058589026 -> 1.056623818 ms`
  (`0.185989%`). Both shared-winner objectives improve despite normal
  unmodified-case noise.
- Decision: promote I11 as the shared winner. Matrix-wide strict correctness,
  the changed Case-6 stable gain, candidate geomean, candidate sum, total
  speedup, and MFU all improve. Only Cases 2/3 remain at or above 7x; the
  broader multi-case 7-10x objective remains open.

## O90 - Case-13 fully-future QK fragment skip

- Parent/current unified winner: `f1d7140566fbdbb1976ad777c81c4eac5716cac3`.
  The focused candidate interleaves M16xN8 score-fragment ownership across
  eight warps and directly writes the established FP16 `-infinity` value for
  complete fragments strictly above the causal diagonal. Live QK chains,
  softmax, PV, dispatch, and fallbacks are unchanged.
- Static/local evidence: the executable coordinate oracle proves disjoint and
  complete writes, causal safety, and balanced ownership. All 46 tests, the
  exact CPU smoke, the sm89 build, PTX/SASS dependency-chain checks, and
  spill/resource checks passed before GPU evaluation.
- Deterministic GPU identity: ordinary job
  `job-1788219125730-051ece633911d073` evaluated immutable snapshot
  `ad7ecf43cc5061e1e26516deeeecef1c5273373a0961bb498b4fb9dabd3eab76`.
  It requested and completed exactly Cases 6/12/13 on an NVIDIA GeForce RTX
  4070 using CUDA BF16, Python 3.12.14, PyTorch 2.13.0+cu130, and CUDA 13.0;
  state succeeded, exit zero, complete result, and no execution failure.
- Correctness was checked before performance. All 15 trials passed bitwise
  exactly under the strict OR rule: `0 / 862,453,760` failed elements, zero
  maximum absolute error, and zero maximum relative error.
- Same-job medians and paired speedups were Case 6
  `414.213134766 / 93.310974121 ms / 4.439061307x`, Case 12
  `0.938383996 / 0.153663993 ms / 6.106726624x`, and Case 13
  `110.882812500 / 12.170240402 ms / 9.110979638x`. Case-13 candidate round
  medians were `12.172287941 / 12.168191910 / 12.213247776 ms`.
- Against the current winner's comparable focused Case-13 candidate median
  `12.617728233 ms`, O90 improves candidate latency by `3.546501%`. Baseline
  drift is `-0.001851%`, and the paired-speedup ratio is `1.036750`, supporting
  attribution to this change. Negative controls remain stable: Case 6 improves
  `0.002199%`, while Case 12 regresses `0.041657%`.
- Decision: `validated-building-block`, retain the current unified winner.
  The result clears the 3% focused floor but affects only Case 13, so it does
  not satisfy the 3-5% multi-case continuation/promote gate by itself. Preserve
  the commit for combination with a later Case-6 route; require a new ordinary
  focused and then unified job for any combined promotion claim.
