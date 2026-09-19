# OBC-inspired comparison package v0.9.0

This package runs the three OBC-inspired baselines against the existing
`outputs-v4` SQuaRE results. It does not run SQuaRE and does not wait for v5.
The ZIP includes the original v4 reference JSONs and their v2 layerwise timing
sources. No datasets, pretrained weights or new experimental results are bundled.

The standard recipe is **OBC-inspired blockwise OBS (B=256, adapted)**. The CSV
and LaTeX tables use the short label **OBC-block256**. This is an explicit
block-diagonal approximation, not a reproduction of published full-Hessian OBC.
Do not remove that distinction in the paper.

## Transfer and start

Copy `pretrained-isolation-framework-v0.9.0.zip` and its `.zip.sha256` file to
`~/Mehrshad/Transformer/OBC-compare/` on the same server. Keep the existing
v0.8.0 environment activated; its installed Torch/timm versions are the ones
that produced the matching baselines. The scripts load this package's `src`
directly, so installing this package or upgrading Torch is unnecessary.

```bash
cd ~/Mehrshad/Transformer/OBC-compare
sha256sum -c pretrained-isolation-framework-v0.9.0.zip.sha256
unzip pretrained-isolation-framework-v0.9.0.zip
cd pretrained-isolation-framework-v0.9.0
mkdir -p logs

nohup env PYTHON=python bash scripts/run_obc_v4_15h.sh \
  --data-root /storage/users/pdarbani/Mehrshad/Transformer/v4.0/pretrained-isolation-framework/data/imagenetv2 \
  > logs/obc-v4-15h.log 2>&1 &
```

The data parent must contain both `imagenetv2-threshold0.7-format-val` and
`imagenetv2-matched-frequency-format-val`. The saved reference controls the
split even though the general project YAMLs contain newer v5 defaults.
Use a fresh extracted folder. Do not overlay the old package or copy old pilot
caches into the new output directory. Allow approximately 15 GB of free disk
space for statistics, all candidates, base and selected checkpoints.

```bash
tail -f logs/obc-v4-15h.log
# Detailed current model progress, for example:
tail -f outputs-obc/v4-block256/logs/deit_tiny-run.log
```

The top-level log announces each phase and its detailed log path. The script
first verifies package hashes, runs a small CUDA numerical self-check, and
checks all three dense baselines before starting expensive compression.
Torch and timm versions must match the recorded reference environment
(Torch 2.6.0+cu124, timm 1.0.29). Missing data, wrong preprocessing, mismatched
baselines or numerical failures stop the relevant stage with diagnostics.
No tolerance is increased automatically. The baseline tolerance is 0.02 pp.
The default numerical mode matches the reconstructed SQuaRE Torch 2.6 defaults:
CUDA matmul TF32 disabled, cuDNN TF32 enabled, benchmark/deterministic flags off.

## Total time budget and restart

The default is **15 hours across the entire study**, not 15 hours per network.
A persistent `study.json` ledger charges elapsed child-process time, including
baseline checks, numerical self-checks, setup, compression, search, evaluation,
and in-process artifact verification. It is GPU-job wall time, not a GPU
utilization measurement. Final parent-process CSV/figure generation and artifact
checks run on the CPU after the GPU jobs and may finish later.

Initial full-run allowances are 3 h for DeiT and 5 h each for Swin and ResNet.
Each full run reserves 5 minutes for search and 5 minutes for final evaluation.
Unused time is available for resumable incomplete runs after the first pass.
The controller terminates its child process group before its assigned allowance
is exhausted, including a termination grace period. As with any OS deadline,
process scheduling or an uninterruptible operation can delay cleanup.

Re-run the identical launch command after a recoverable interruption. The study
retains the remaining allowance, candidate caches, completed evaluations and
frozen selections. It does not silently grant another 15 hours. A child inherits
the study lock, preventing overlapping controllers after a parent crash. If an
interrupted interval cannot be measured, it is conservatively charged through
restart time and marked as uncertain in the ledger.

If the entire allowance is exhausted, completed work remains reusable. Only an
explicit later authorization such as `--extend-total-hours 18` raises the total
ceiling to 18 hours while retaining all consumed time and caches. Repeating that
flag does not grant more time. It is not used by the default 15-hour command.

A time cap cannot guarantee completion of all candidates. An incomplete bank
is never padded with another algorithm or reported as a completed comparison.
If all banks exist but search hits its reserved deadline, the completed probes
are frozen and evaluated; `search_finished_resource_grid=false` discloses that
not all planned search points were explored. A dense safety anchor is always
first. Actual search counts must accompany runtime claims.

Do not change block size, row batching, references or code in an existing study.
Those settings are part of its identity. No approximation is selected using
final evaluation accuracy.

## Matched v4 protocol

| Model | Selected layers | Threshold-0.7 refinement images | Matched-frequency final images |
|---|---:|---:|---:|
| DeiT-Tiny | 48 | 1024 | 10000 |
| Swin-Tiny | 48 | 1024 | 10000 |
| ResNet-18 | 19 | 256 | 10000 |

Exact model names, selection rules, preprocessing, seed, counts and duplicate
exclusion come from each v4 reference. Ordered image-content manifests and the
pretrained state hash are saved. Refinement/final content overlap is rejected.
The v4 reference lacks original checkpoint/image hashes, so historical identity
is supported by metadata and matching dense accuracy, not proven bitwise.

The grid remains FP32/W8A8/W6A6/W4A4 crossed with dense/2:4/4:8/3:8. N counts
retained slots; quantization may create additional zeros. The same flattened
(C,Kh,Kw) convolution grouping and full-output-channel symmetric minmax weight
quantizer are used. Activation ranges are calibrated once from the dense model.
No BatchNorm tuning, fine-tuning, or normalization correction is performed.

The numerical procedure is:

1. Visit every calibration example, selecting up to 32 evenly spaced positions
   per example. Swin attention treats windows as examples, covering every window.
2. Accumulate a full FP64 input Gram. Use its diagonal blocks of at most 256
   contiguous coordinates for OBS; N:M groups never cross block boundaries.
3. Add damping equal to 1% of the **full-layer** mean Gram diagonal to each block.
4. Prune using greedy OBS with compensation. Quantize the resulting sparse
   parents using OBS compensation and the active principal Hessian. The original
   OBC outlier-priority rule is used within each block. Quantization scales are
   computed over the **full output channel**, not independently per block.
5. Batch up to 32 independent output channels. Reuse the three FP32 sparse
   parents across bitwidths. Keep the dense FP32 control unchanged.
6. Score each full candidate's relative output reconstruction error on 2048
   saved input vectors, including activation quantization. Score is not accuracy.
7. Run exact DP for the additive reconstruction objective under the existing
   selected-layer proxy at a fixed resource grid. Evaluate at most 32 distinct
   network assignments on refinement images, shared across all three budgets.
8. Select the lowest-cost refinement-feasible assignment for each budget and
   irrevocably save all three winners before evaluating any compressed model on
   final images. Resume never reopens search after this point.

The energy proxy remains `(bits/32) * retained_fraction`, averaged without layer
size/MAC weighting over selected layers. It is not physical energy, latency or
whole-network cost. Fixed-bit fake-quantized inference is not a hardware speed
benchmark. Full-Hessian mode remains available through the lower-level CLI with
`--hessian-block-size 0`; it is not the standard 15-hour recipe.

## Outputs to return

The default root is `outputs-obc/v4-block256/`:

```text
study.json                         # total time ledger, commands, completion status
selfcheck.json                     # numerical self-check on the server GPU
logs/                              # baseline and run logs per model
MODEL/
  manifest.json                    # exact identities/settings and method label
  data_manifest.json               # ordered image paths, hashes, labels
  dense_checks.json                # both dense accuracy gates and numerical flags
  base_model.pt                    # original verified state for replay
  activation_ranges.pt             # fixed activation calibration
  setup_attempts.json              # setup timing, including resume work
  preparation_attempts.json        # full candidate timing and partial attempts
  cache/layer_XXX/statistics.pt     # full input Gram and score inputs
  cache/layer_XXX/CONFIG.pt         # candidate weights, masks, scales, score
  search.json                      # assignments, refinement scores and time
  frozen_selection.json            # irrevocable winners for all three budgets
  predictions/*.pt                 # ordered labels and top-5 predictions
  selected/budget_0.1.pt            # selected weights/masks/scales
  selected/budget_0.5.pt
  selected/budget_1.pt
  results.json                     # final metrics, proxy, layer choices, times
  verification.json                # independent artifact consistency checks
  status.json
comparison/                        # generated automatically after all 3 finish
  comparison_summary.csv           # 18 rows: 2 methods x 3 models x 3 budgets
  layer_configurations.csv         # every selected layer for both methods
  runtime_by_model.csv             # preparation, search, final evaluation
  comparison_table.tex             # booktabs table for the paper
  comparison_metadata.json         # exact baseline recipe and reference hashes
  methodology.md                   # description and limitations for reporting
  report.md
  *.svg                            # optional when Matplotlib is installed
```

If some models are incomplete, completed models are exported under
`comparison_partial/`, and `study.json` says `incomplete`. The process exits
nonzero in that case. Only `status=completed` with three verified model results
is the complete nine-budget comparison.

Every completed model automatically checks saved predictions against reported
accuracy and manifest label order; selected weights against recorded masks,
N:M counts and quantizer grids; checkpoints and datasets against hashes; final
assignments against the frozen selection; and proxy/feasibility against those
artifacts. This does not run inference a second time. Verification can also be
repeated independently:

```bash
PYTHONPATH=src python -m pretrained_isolation.obc.audit \
  outputs-obc/v4-block256/deit_tiny
```

To regenerate the comparison files without GPU experiments:

```bash
PYTHON=python OBC_ROOT=outputs-obc/v4-block256 \
  COMPARISON_ROOT=outputs-obc/v4-block256/comparison \
  bash scripts/compare_obc_imagenetv2.sh
```

Send back the complete `outputs-obc/v4-block256/` folder if possible. Keep the
large checkpoints/caches on the server if transfer is expensive; the JSONs,
comparison folder and logs suffice for initial analysis. Per-image prediction
files support later accuracy audits. Do not delete the source artifacts merely
because the CSV has been generated.

## Reporting accurately

Use the explicit OBC-block256 label and approximation description. Report final
top-1/top-5, final drop, requested budget, final feasibility and proxy cost.
Refinement feasibility is separately exported. Existing v4 rows are preserved,
including rows that exceed final budgets; `status=succeeded` in the old file is
not interpreted as accuracy feasibility.

Report OBC preparation and search once per network, shared across budgets.
SQuaRE's original v2 candidate-generation timers are included along with v4
refinement/final timers. Historical timers omit some setup; they do not have
identical boundaries to the new study ledger. Do not derive an unqualified
wall-clock speedup from those numbers.

Historical SQuaRE v4 candidate selection used final images. This matched legacy
comparison does not establish untouched-test generalization. No SQuaRE rerun is
required to compare the existing results, but the paper must describe the
protocol honestly. Future v5 comparisons need separate reference identities and
outputs, not relabeled v4 results.

The original method reference is Frantar et al., Optimal Brain Compression,
NeurIPS 2022; upstream reference implementation:
https://github.com/IST-DASLab/OBC at
`9b7979bfc9ee20d87db553823a32ee9890beaa99`. This package is an independent adapted
implementation. Its tests and artifact checks reduce implementation mistakes;
they cannot guarantee a particular accuracy outcome or server runtime.
