# Fair integration experiments

## E00 - d66 contract repair

- Status: static validation passed; unified GPU validation pending
- Source: verified d66 snapshot
  `d66bc39ef07f5aaf99e4a4648d6feb899cbcaee51189badf390c822e0a567c8f`.
- Hypothesis: isolating candidate-only block dispatch from the measured baseline
  and cloning CUDA Graph outputs restores benchmark/API fairness. This is a
  compliance repair, not an optimization claim.
- Required patch: preserve every d66 optimization except (1) introduce a
  separate `UserOptimizedTransformerBlock`, leaving
  `BaselineTransformerBlock` pristine, and (2) return independent cloned graph
  outputs after capture and replay.
- Validation: static checks, unit coverage for both invariants, CPU smoke, then
  one cases 1-13 GPU job through the shared `benchmarkctl` queue.
- Source changes:
  - restored a pristine `BaselineTransformerBlock` and moved the existing d66
    cuBLASLt candidate dispatch into `UserOptimizedTransformerBlock`;
  - constructed candidate layers from the candidate-only block without changing
    parameter names, weight-copy compatibility, attention paths, or CUDA Graph
    eligibility;
  - cloned both CUDA Graph capture and replay return values so consecutive
    forwards do not alias;
  - added focused tests for baseline isolation and independent graph outputs.
- Static validation:
  - `git diff --check`: passed;
  - `python -m py_compile transformer_benchmark/models.py tests/test_models.py`:
    passed;
  - `python -m unittest discover -s tests -v`: 11 tests passed;
  - prescribed CPU smoke: strict correctness passed bitwise exact
    (`0 / 128` failures). CPU timing is not GPU performance evidence.
- Result: pending one unified cases 1-13 GPU job.
