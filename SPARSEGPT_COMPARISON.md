# Matched SparseGPT comparison: additive patch for v0.9.0

This patch adds a separate SparseGPT implementation and outputs. It does not
replace original v0.9.0 files or write into the completed OBC output directory.
It runs all three vision networks against the existing **outputs-v4** references;
SQuaRE and OBC do not need to be rerun. It is not a v5 comparison.

## Install and check

Copy `sparsegpt-v0.9.0-patch-v1.zip` and its `.zip.sha256` file into the existing
server `pretrained-isolation-framework-v0.9.0` directory. From that directory:

```bash
sha256sum -c sparsegpt-v0.9.0-patch-v1.zip.sha256
unzip sparsegpt-v0.9.0-patch-v1.zip
bash scripts/run_sparsegpt_v4_2h.sh --check-only
```

The launcher reads the original Python executable and dataset root from
`outputs-obc/v4-block256/study.json`. There is no placeholder dataset path and
no package/dependency upgrade. It automatically uses the original environment,
even if the prompt currently says `(base)`. If that executable was moved, set
`PYTHON` to the original environment's Python executable explicitly.

The full OBC folder must still contain all model statistics, activation ranges,
base models, predictions, manifests, verification files and completed results.
The earlier row-sampled pilot folder `outputs-obc/v4` is not a valid source.
For a different location, append `--obc-root` with the actual completed study
directory to every launch command. Do not move source files during the study.
Allow approximately 15 GB of additional disk space for the independent artifacts.

`--check-only` checks package integrity and required reference/data/artifact paths;
it does not run the CUDA numerical or accuracy tests. Those run during profiling.

## Start and monitor

Arrange exclusive access to the original GPU for up to two hours. The launcher
refuses to start a child while another compute process occupies the GPU, or if
fewer than 4096 MiB are free. This point-in-time check is not a reservation and
cannot stop another user's job from starting later. It never kills other jobs.
On a multi-GPU host, keep the original device selection; ambiguous numeric
CUDA ordering is rejected rather than guessing which GPU is free.

```bash
mkdir -p logs
nohup bash scripts/run_sparsegpt_v4_2h.sh > logs/sparsegpt-v4-2h.log 2>&1 &
tail -f logs/sparsegpt-v4-2h.log
```

The default `--phase all` does the following, serially:

1. Checks original package/patch hashes, completed OBC sources and dataset paths.
2. Runs a small CUDA numerical check against pinned upstream SparseGPT and an
   independent dense-quantization oracle (maximum one minute).
3. For each network, freshly verifies dense top-1/top-5 on both reference sets,
   then profiles three representative layers with **all output channels** and
   all 16 configurations (maximum three minutes per network). Candidate files
   from successful profiles are reused in the full run.
4. Writes `outputs-sparsegpt/v4/profile_plan.json`, using measured full-bank and
   evaluation costs, layer shapes and a 1.5 safety margin. If the remaining
   projected work exceeds the remaining allowance, it stops **before full runs**.
   Send this file and the logs for review; do not reset the ledger or reduce
   calibration/evaluation counts to force a result.
5. If the projection fits, runs each model with an allocation proportional to
   its projected cost, preserving search/final-evaluation reserves.
6. Audits all results and produces the three-method comparison.

For a manual pause after profiling, use `--phase profile`. Inspect the plan,
then run the identical command with `--phase run`. These phases use the **same
two-hour ledger**; profiling does not grant a fresh two hours for the run.

Detailed logs are under `outputs-sparsegpt/v4/logs/`:

```bash
tail -f outputs-sparsegpt/v4/logs/deit_tiny-run.log
tail -f outputs-sparsegpt/v4/logs/swin_tiny-run.log
tail -f outputs-sparsegpt/v4/logs/resnet18-run.log
```

Only follow the file for the phase/model announced in the top-level log.
`Ctrl-C` exits `tail`; it does not terminate a separately launched `nohup` job.
The success marker is:

```text
[STUDY] completed; models ['deit_tiny', 'swin_tiny', 'resnet18']; charged ... GPU-job hours
```

## Cap, interruption and errors

The fixed total is **7200 seconds of newly executed child-process wall time**,
covering numerical checks, profiling, fresh dense checks, setup, compression,
search, final evaluation and child artifact audits. It is not GPU utilization
time. Parent-only path checks, final CPU audits and comparison export can finish
later. Reused historical OBC calibration costs are disclosed separately.

The parent enforces process-group timeouts; the ten-second termination grace is
inside each allowance. Operating-system scheduling can slightly delay cleanup.
The root study lock is inherited by children, preventing a restarted controller
from overlapping an orphaned child. An interrupted interval whose end time is
unknown is charged conservatively through the restart.

After resolving an OOM or interruption, run the same command again. Completed
models/candidates are reused, failed work is retried explicitly on that launch,
and elapsed time is retained. A failed child stops the study with its log path;
it is not silently skipped while other models are presented as a full study.
There is no automatic time-budget extension. Two hours is a cap, not a guarantee
that the entire comparison finishes. Partial banks never count as final results.

Candidate files are atomically saved with checksums. A file left before its
checksum was committed is regenerated; corrupted committed data fail closed.
After assignments have been frozen, resume can finish missing evaluations but
cannot reopen search using final-set results.

## Exact matched settings

| Setting | All three models |
|---|---|
| Networks | Original timm DeiT-Tiny, Swin-Tiny and ResNet18 checkpoints |
| Selected layers | Same 48 / 48 / 19 layers as v4 |
| Refinement images | Original duplicate-excluded Threshold-0.7: 1024 / 1024 / 256 |
| Final images | Original 10000 matched-frequency images per model |
| Budgets | 0.1 / 0.5 / 1.0 percentage-point top-1 drops |
| Candidate grid | FP32 / W8A8 / W6A6 / W4A4 crossed with dense / 2:4 / 4:8 / 3:8 |
| Hessian | Full sampled-input Gram, FP32 factorization; 1% damping after upstream dead-input handling |
| Processing blocks | 128 columns, including cross-block compensation |
| Statistics | Same 32 positions per example and up to 2048 score rows as OBC |
| Quantization | Same signed symmetric full-output-channel minmax weights; fixed per-layer dense activation ranges |
| Allocation | Same bounded reconstruction-score DP and fixed resource grid as the OBC adapter; at most 32 distinct refinement evaluations |
| Selection | All three lowest-cost refinement-feasible winners frozen before any compressed final evaluation |
| Cost | Same unweighted selected-layer average `(bits/32) * retained_fraction` |

Each sparse/quantized candidate starts from the original dense weights. Pruning
and weight quantization share one compensation pass. Hessian factorization is
reused across candidates, but masks/weights are newly computed by SparseGPT.
N denotes retained slots: 3:8 removes five slots, including when quantization
creates additional zeros. Convolution coordinates use `(C, Kh, Kw)` order.
The dense FP32 control is exactly unchanged.

The 128-column processing blocks are an efficiency mechanism, **not** a
block-diagonal Hessian approximation. This differs from OBC-block256.

Source/reference/checkpoint/data/order/environment/quantizer identities must
match before statistic reuse. Copied artifacts acquire SHA256 receipts and are
verified on resume and audit. The original OBC statistics had no per-file signed
receipt at creation; initial reuse relies on their recorded run identity and
structural checks, then pins the imported bytes. Fresh dense evaluation also
checks the reconstructed protocol (0.02 pp tolerance, unchanged from OBC).
No baseline tolerance is automatically widened.

## Method attribution and output interpretation

Use the label **SparseGPT-adapted**, expanded as:
"SparseGPT with full sampled-input Hessian, matched vision weight/activation
quantization and bounded DP allocation."

The SparseGPT core is based on
https://github.com/IST-DASLab/sparsegpt at commit
`147d2159dc4f3e9f73e47b32c04d7b3708f44436` (Apache-2.0). Unmodified reference
files, the license and attribution are bundled for testing. Adaptations include
vision/convolution support, signed matched quantization, dense-input candidate
statistics rather than sequential compressed-model calibration, the expanded
grid and our outer mixed-precision allocation. This is not an unchanged
reproduction of the paper's LLM experiments.

The final folder is `outputs-sparsegpt/v4/`. Keep `study.json`, `selfcheck.json`,
`profile_plan.json`, logs and all model directories. Each model contains source
and data manifests, dense checks, `reuse.json`, `pilot.json`, candidate caches,
search history, frozen selections, per-image predictions, selected checkpoints,
`results.json`, `verification.json` and `status.json`.

`comparison/` contains:

- `comparison_summary.csv`: 27 rows, including final/refinement feasibility.
- `layer_configurations.csv`: exact assignments for every method/model/budget.
- `runtime_by_model.csv`: newly executed work, shared search and inherited
  statistics; total job time includes profiling, retries and failed attempts.
- `comparison_table.tex`: a paper table with final-budget failures visible.
- `comparison_metadata.json`, `budget_outcomes.json`, `report.md`.

The energy column is a proxy, not measured hardware energy or latency. Historical
SQuaRE v4 candidate selection used final images, so this is not an untouched-test
generalization study. Historical timing boundaries differ and the OBC job had
GPU contention. Do not claim an unqualified runtime speedup, or hide inherited
preprocessing because only two new GPU hours were charged. SQuaRE is not assumed
to win; lowest-cost winners are reported only among final-feasible methods.

To repeat artifact verification without inference:

```bash
PYTHONPATH=src python -m pretrained_isolation.sparsegpt.audit outputs-sparsegpt/v4/swin_tiny
```

To regenerate the comparison without GPU experiments (use the original Python
environment for these direct module commands):

```bash
PYTHONPATH=src python -m pretrained_isolation.sparsegpt.compare
```

Copy the complete `outputs-sparsegpt/v4/` folder back for analysis, together with
the completed `outputs-obc/v4-block256/` folder. JSONs/CSVs/logs suffice for an
initial reading; predictions and selected checkpoints support the full audit.
