# Matched-setting OBC baseline (v0.8.0)

This package adds a **profile-first** baseline for DeiT-Tiny, Swin-Tiny and
ResNet-18, at 0.1 / 0.5 / 1.0 percentage-point accuracy-drop budgets. No new
ImageNetV2 results are bundled: those require running on the server.

## Method and limits of the comparison

Reference: Frantar, Singh and Alistarh, [Optimal Brain Compression, NeurIPS
2022](https://arxiv.org/abs/2208.11580), [official repository](https://github.com/IST-DASLab/OBC),
reviewed revision `9b7979bfc9ee20d87db553823a32ee9890beaa99`.

The compression core is an independent implementation of sequential ExactOBS
updates, using the full damped input Gram matrix, greedy removal/quantization,
and compensation of the remaining coordinates. It preserves the quantizer's
outlier-priority rule. It is **not** magnitude pruning, a diagonal-Hessian
approximation, or SparseGPT. It is not a verbatim reproduction of every OBC
experimental setting, nor does greedy ExactOBS guarantee a globally optimal mask.

Call the method **OBC (matched ExactOBS + bounded DP, adapted)**. Disclose these
changes when reporting results:

* The same 16 candidates as SQuaRE: FP32 / W8A8 / W6A6 / W4A4 crossed with
  dense / 2:4 / 3:8 / 4:8. Sparse FP32 parents are reused for quantization.
* Signed symmetric per-output-channel min/max weight scales and per-layer
  max-absolute activation ranges, collected in one dense calibration pass.
  These match SQuaRE's representation; they replace OBC's native scale-search
  recipe. Compensated weights are loaded directly, never re-quantized by the
  SQuaRE wrapper. Biases and unselected modules remain FP32.
* N means **retained slots**, including 3 retained out of 8. CNN grouping is
  the SQuaRE flattened `(input channel, kernel height, kernel width)` order,
  not OBC's alternate convolution permutation. A quantized retained value can
  itself become zero; mask-slot density and actual nonzero count are separate.
* By default, at most 32 evenly spaced input positions per example are used
  for the Gram matrix (every calibration batch is visited). Swin's attention
  windows are represented according to their module input batch dimension.
  `--positions-per-example 0` uses all positions and can be much slower.
  Hessian accumulation and inverse updates use float64 for numerical stability;
  candidate weights return to the original float32 representation. Damping is
  1% of the mean Gram diagonal by default. These choices can cost more time
  than the original float32 inverse implementation.
* A bounded, deterministic sample (up to 2,048 input rows, spread over batches)
  measures normalized layer-output squared error, including activation
  quantization, rather than evaluating all isolated candidates on final data.
  Input-position sampling and bounded scoring are approximations; the OBS
  updates are exact for their selected, damped Hessian.
* DP minimizes summed reconstruction scores at a fixed grid of resource
  targets, using the shared SQuaRE cost proxy. The 16 costs are exact integer
  multiples of 1/256, so the allocator needs no cost rounding. Up to 32 unique
  network proposals are evaluated on refinement data, shared across all three
  budgets. Duplicate proposals are skipped. Accuracy is not assumed monotonic.
  The cheapest observed refinement-feasible assignment wins, with dense FP32
  as an anchor. This bounded search need not find the best possible allocation.
* No retraining, BN tuning, bias correction, or normalization-statistics
  correction. The omission of OBC's optional corrections is a declared
  restriction of this matched comparison, not evidence against the complete
  published recipe. A native-recipe baseline would be a separate experiment.

The energy value is the legacy proxy
`mean_layers((weight_bits / 32) * retained_fraction)`, over **selected layers**.
It is not measured energy, latency, BOPs, or a whole-network cost estimate. A
comparison may establish a difference in this proxy, not a hardware speedup.

## Consistency checks and v4/v5 handling

`--reference` names a completed SQuaRE `*_global_refinement.json`. It determines
the model, seed, layer selection, calibration/refinement size and split. The
runner checks preprocessing, layer order/count, duplicate exclusion metadata,
sample counts, saved v5 split hashes, and dense top-1/top-5 accuracy on **both**
refinement and final data. A mismatch fails before generating candidates.
Default dense tolerance is 0.02 pp; investigate mismatches instead of widening
it to accommodate a different checkpoint or dataset.

For v4: 1,024 Threshold-0.7 refinement images for each Transformer, 256 for
ResNet-18, and all 10,000 Matched Frequency final images. Current v5 defaults
must not silently replace that protocol. v5's actual split is read from its
completed result when available. A v4 OBC result cannot be paired with v5:
the exporter checks the SHA-256 of the exact reference file.

The older references do not contain original checkpoint or content hashes.
We reconstruct their deterministic sample order and verify metadata plus dense
accuracy, but cannot retroactively prove bitwise checkpoint/data identity.
New OBC runs save full content manifests, checkpoint and code fingerprints.

The legacy SQuaRE layerwise sweep used the final-evaluation images for candidate
selection. Neither the existing v4 run nor a v5 run initialized from that same
sweep has a fully untouched test set. The reports explicitly state this
historical limitation; matching evaluation does not erase the exposure.

OBC freezes **all three** winning assignments before evaluating compressed
models on final data. Test-budget failure is reported, never repaired using
test labels. No energy winner is declared when both methods fail the final
accuracy budget. Fixed search settings are required on resume.

## Transfer and run the pilot (do this first)

Use a **new sibling directory** on the server. Do not overwrite the directory
or editable installation used by the currently running SQuaRE experiment.
The zip has a top-level `pretrained-isolation-framework-v0.8.0/` folder, code,
tests, configs, documentation, and the original tracked v2/v4 reference JSONs.
It excludes data, model checkpoints, papers, figure drafts, and OBC outputs.
`PACKAGE_MANIFEST.json` records SHA-256 hashes for the packaged files.

On your Mac (replace the SSH destination and directory):

```bash
scp pretrained-isolation-framework-v0.8.0.zip USER@SERVER:/YOUR/WORK/DIRECTORY/
```

On the server:

```bash
cd /YOUR/WORK/DIRECTORY
unzip pretrained-isolation-framework-v0.8.0.zip
cd pretrained-isolation-framework-v0.8.0

# Activate the existing working environment with torch/timm installed.
# Use your actual activation command/path; no package upgrade is required.
source /PATH/TO/EXISTING/ENV/bin/activate

python -c "import torch,timm; print(torch.__version__, timm.__version__); print(torch.cuda.get_device_name(0))"
```

v4 used torch `2.6.0+cu124`, timm `1.0.29`, Python `3.13.5`, RTX 3090.
The scripts set PYTHONPATH to this package's source **for their process only**;
they do not reinstall or modify the active SQuaRE environment.

Run when the GPU is idle so the pilot timings are meaningful. The data root is
the folder containing `imagenetv2-matched-frequency-format-val` and
`imagenetv2-threshold0.7-format-val`:

```bash
mkdir -p logs
PYTHON=python bash scripts/profile_obc_imagenetv2.sh \
  --data-root /PATH/TO/EXISTING/PROJECT/data/imagenetv2 \
  > logs/obc-pilot.log 2>&1
```

For a persistent run, use `nohup env PYTHON=python bash ... > logs/obc-pilot.log
2>&1 &`, or run inside `tmux`. Follow progress with:

```bash
tail -f logs/obc-pilot.log
```

Each model profiles the smallest, middle, and largest input-width layers, using
two evenly spaced output rows. The candidate-generation time cap is 120 seconds
per layer. Dataset checks, dense evaluations, and statistics collection are
additional work. The cap is cooperative between numerical operations: a single
GPU kernel or matrix factorization can overrun it. This is not a hard kill timer.

Results are under:

```text
outputs-obc/v4/deit_tiny/pilot.json
outputs-obc/v4/swin_tiny/pilot.json
outputs-obc/v4/resnet18/pilot.json
logs/obc-pilot.log
```

Send back those three JSONs and the log. A pilot timeout is itself a useful
measurement, not an accuracy result. Linear time extrapolations are emitted
only for completed sample banks and remain preliminary. Inverses, GPU
utilization, and output-row count can change scaling. Review individual layer
estimates and widths rather than treating them as a guaranteed whole-run ETA.

Calibration statistics and activation ranges are reusable. Partial-output-row
candidates are **not** reused as full-layer candidates. Cache identity includes
code, checkpoint, data contents, model, grid, numerical settings, batch size,
device, and environment. Changed settings require a new output directory.
The process uses an OS file lock to prevent concurrent writes to one directory.

Optional pilot controls (choose before starting a new directory):

```bash
OBC_ROOT=outputs-obc/v4-pilot-alt PYTHON=python \
  bash scripts/profile_obc_imagenetv2.sh \
  --data-root /PATH/TO/DATA/imagenetv2 \
  --row-batch 2 --pilot-rows 4 --pilot-layer-seconds 180
```

## Full run (only after choosing a cap from the pilot)

Set the agreed number explicitly. The example uses a shell placeholder, not a
recommended number of hours:

```bash
OBC_MAX_HOURS=AGREED_HOURS_PER_MODEL PYTHON=python \
  bash scripts/run_obc_imagenetv2.sh \
  --data-root /PATH/TO/EXISTING/PROJECT/data/imagenetv2 \
  > logs/obc-run.log 2>&1
```

The cap applies **per model, per invocation**. All three sequential model runs
can therefore use roughly three times that allocation, plus final evaluation
and indivisible-operation overhead. Time is checked during compression and
between evaluations. On a preparation/search timeout, completed candidates and
probes are retained; inspect `status.json`, then resume with the same command
when more time is agreed. A timeout does not emit a falsely complete comparison.
Final evaluations complete after the bounded search. An OS kill during a
candidate discards only that unfinished candidate. Interrupted operations can
make recorded stage times a lower bound; avoid deriving precise speedups from
incomplete historical timers.

The full ExactOBS pass may be expensive, particularly for wide ResNet
convolutions. No approximate-Hessian fallback is silently enabled. If the pilot
shows it is unsuitable, choose and validate a separately labeled approximation
before the full run.

Full outputs, for each model:

```text
outputs-obc/v4/MODEL/
  manifest.json                 # protocol, environment, identity, adaptations
  data_manifest.json            # ordered content/label manifests
  dense_checks.json             # measured dense accuracies
  activation_ranges.pt
  pilot.json
  preparation_attempts.json     # includes timed-out candidate attempts
  cache/layer_NNN/statistics.pt
  cache/layer_NNN/CONFIG.pt      # effective weights, mask, scale, score, timing
  search.json                   # refinement-only proposals and measurements
  frozen_selection.json         # all budgets fixed before final evaluation
  selected/budget_0.1.pt         # selected compressed layers + activation ranges
  selected/budget_0.5.pt
  selected/budget_1.pt
  results.json                  # final accuracy/cost/configuration outcomes
  status.json
```

Selected checkpoints contain compressed selected layers, not a self-contained
whole model: reconstruct the verified base checkpoint and apply those tensors
through the adapter. Candidate caches can require several GB; keep them for
resume and reproducibility.

Once all three `results.json` files are complete:

```bash
PYTHON=python bash scripts/compare_obc_imagenetv2.sh
```

This produces `comparisons/v4/{comparison_summary.csv,layer_configurations.csv,
runtime_by_model.csv,comparison_table.tex,report.md}`. With Matplotlib available
it also writes SVG accuracy-cost plots and layer precision/sparsity maps.
The table needs LaTeX `booktabs`; place it in a suitable wide table environment.

Runtime is reported once per model for OBC's shared preparation/search, plus
final evaluations. SQuaRE's recorded v2 candidate-evaluation time is shown
separately from v4 refinement-plus-final time. The historical timers omit some
setup and do not establish a clean overall speedup. The report never divides
OBC's full preparation by SQuaRE's refinement-only timer. Pilot-only compressed
rows have a separate timing field. Final evaluation reuse across identical
budget assignments is explicit.

## When v5 arrives

Copy the completed v5 outputs into this new package directory (or use an
absolute REFERENCE_ROOT). Leave the old running project unchanged:

```bash
REFERENCE_ROOT=outputs-v5/imagenetv2 OBC_ROOT=outputs-obc/v5 PYTHON=python \
  bash scripts/profile_obc_imagenetv2.sh --data-root /PATH/TO/DATA/imagenetv2
```

Use those same roots for `run_obc_imagenetv2.sh`, setting the agreed cap, and:

```bash
REFERENCE_ROOT=outputs-v5/imagenetv2 OBC_ROOT=outputs-obc/v5 \
  COMPARISON_ROOT=comparisons/v5 PYTHON=python \
  bash scripts/compare_obc_imagenetv2.sh
```

If v5 uses another layerwise selection source, set `LAYERWISE_ROOT` accordingly.
The exporter verifies its hash instead of attributing unrelated preparation
time. The old bank is not reused across changed splits/calibration.

## Local verification

```bash
PYTHONPATH=src python -m pytest -q
```

Tests cover constrained least-squares agreement, an independently recomputed
inverse quantization oracle, 3:8 semantics, sparse support preservation,
representable grids, batched-row agreement, DP versus exhaustive enumeration,
convolution layout, candidate scoring with activation quantization, v4/v5
protocol rejection, and a complete synthetic CPU run through resume/export.
With timm installed, all three actual architectures are checked for the exact
48/48/19 selected-layer scopes and unchanged dense outputs after wrapping.
These checks do not replace pretrained CUDA/ImageNetV2 validation on the server.
