# SQuaRE: joint sparsity and quantization experiments

This repository implements post-training quantization and sparsity experiments
for pretrained vision models, isolated layer configuration sweeps, simultaneous
layer assignment, and interaction-aware global refinement. It also includes
matched OBC-inspired and SparseGPT-adapted comparison pipelines. No training or
fine-tuning is required.

Supported pretrained `timm` models:

| Family | Model identifier |
|---|---|
| Transformer | `deit_tiny_patch16_224.fb_in1k` |
| Hierarchical transformer | `swin_tiny_patch4_window7_224.ms_in1k` |
| CNN | `resnet18.a1_in1k` |

The framework uses the public ImageNetV2 matched-frequency and threshold-0.7
variants; the original ImageNet-1k dataset is not required. Model weights are
loaded through `timm`. Experimental outcomes belong in the paper and archived
result files, rather than this usage guide.

## Workflow

- **Isolation:** evaluate weight quantization, activation quantization, and
  pruning independently or in fixed combinations.
- **Layer-wise selection:** evaluate a 16-choice sparsity/precision grid with
  one selected layer changed at a time, then select per-layer configurations.
- **Joint evaluation:** apply all independently selected configurations together.
- **Global refinement:** use contextual accuracy probes for single-, pair-, and
  block-wise repair, followed by energy-proxy reclamation.
- **Prior-work comparisons:** generate compensated candidates with OBC-inspired
  blockwise OBS or adapted SparseGPT and allocate them under matched budgets.

The implemented SQuaRE selector uses measured isolated accuracy and contextual
accuracy probes. Baseline adapters use reconstruction scores for their bounded
allocation search. Resource values are analytical proxies, not measurements from
hardware energy or latency models.

## Isolation guarantees

Quantization-only inference keeps every pruning mask equal to one:

```text
Y = Q_A(X) op Q_W(W) + b
```

Sparsity-only inference hard-bypasses both quantizers at FP32:

```text
Y = X op (M * W) + b
```

`op` is a linear projection for the transformers and convolution for ResNet-18. There
is no retraining, fine-tuning, batch-normalization adaptation, or bias correction.

## Install

```bash
# From the repository root:
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[test,analysis]'
pytest -q
```

Install the PyTorch build appropriate for the server's CUDA version first if needed.

## Download the public benchmark

```bash
ptq-download-imagenetv2 --root data/imagenetv2
```

This downloads and validates:

- MatchedFrequency: all 10,000 images are used for evaluation in the isolation,
  layer-wise and legacy v4 suites. Current v5 refinement reserves a disjoint
  subset for final evaluation, as described below.
- Threshold0.7: a deterministic subset supplies quantizer/pruning calibration.
  The legacy v4 protocol also uses a labeled subset for refinement.

Both contain 1,000 classes with ten images per class. The download is resumable. Archives
are deleted after successful extraction, leaving about 2.4 GiB of images. The loader maps
numeric directories directly to canonical ImageNet class indices; it does not use unsafe
lexicographic class ordering.

The framework hashes both variants once and removes byte-identical overlaps from the
calibration pool. Hash manifests are reused on later runs. Quantizer calibration
does not use labels; accuracy evaluation and refinement do.

## Verify all dense baselines first

```bash
ptq-isolate --config configs/deit_tiny_imagenetv2.yaml --suite baseline
ptq-isolate --config configs/swin_tiny_imagenetv2.yaml --suite baseline
ptq-isolate --config configs/resnet18_imagenetv2.yaml --suite baseline
```

Each command prints and saves the measured top-1/top-5, reference values, differences,
and a `within_tolerance` check. Do not start paper sweeps until these full 10,000-image
checks pass. The configured tolerance is 0.15 percentage points. Full sweeps stop before
their first compression experiment if this check fails, preserving the diagnostic JSON.

## Optional pipeline smoke tests

These partial runs validate wrapping, calibration, masking, quantization, and JSON output.
They are not reference-accuracy or paper results.

```bash
ptq-isolate --config configs/deit_tiny_imagenetv2.yaml --suite all \
  --max-eval-samples 128 --calibration-samples 64

ptq-isolate --config configs/swin_tiny_imagenetv2.yaml --suite all \
  --max-eval-samples 128 --calibration-samples 64

ptq-isolate --config configs/resnet18_imagenetv2.yaml --suite all \
  --max-eval-samples 128 --calibration-samples 64
```

Partial smoke tests deliberately mark the dense reference as non-comparable.

## Full experiments

Run the six paper-result commands:

```bash
ptq-isolate --config configs/deit_tiny_imagenetv2.yaml --suite quantization
ptq-isolate --config configs/deit_tiny_imagenetv2.yaml --suite sparsity

ptq-isolate --config configs/swin_tiny_imagenetv2.yaml --suite quantization
ptq-isolate --config configs/swin_tiny_imagenetv2.yaml --suite sparsity

ptq-isolate --config configs/resnet18_imagenetv2.yaml --suite quantization
ptq-isolate --config configs/resnet18_imagenetv2.yaml --suite sparsity
```

Or run the supplied sequential script:

```bash
scripts/run_full_imagenetv2.sh
```

## Layer-by-layer 16-configuration experiment

The `layerwise` suite evaluates a 4 x 4 grid for every layer in the existing model
selection scope:

- Weight and input-activation precision together: FP32, INT8, INT6, or INT4.
- Weight sparsity: dense, 2:4, 4:8, or 3:8.
- N:M masks retain the N largest-magnitude weights in each consecutive group of M
  weights along the flattened input dimension. The retained weights multiply their
  corresponding activations normally.

Only the named target layer is changed in each evaluation. Every other selected module,
as well as every unselected part of the network, remains dense FP32. `FP32__dense` is an
intentional no-op control, so it should reproduce the measured dense baseline apart from
any numerical nondeterminism.

The selection scope is unchanged from the earlier study: 48 QKV/attention-output/MLP
linear modules in DeiT-Tiny, 48 such modules in Swin-Tiny, and 19 residual-stage
convolutions in ResNet-18. Thus, the full study contains 768 + 768 + 304 = 1,840 isolated
evaluations, plus one dense baseline per model.

Run one model at a time:

```bash
ptq-isolate --config configs/deit_tiny_imagenetv2.yaml --suite layerwise --resume
ptq-isolate --config configs/swin_tiny_imagenetv2.yaml --suite layerwise --resume
ptq-isolate --config configs/resnet18_imagenetv2.yaml --suite layerwise --resume
```

Or run the complete sequence and analysis export:

```bash
scripts/run_layerwise_imagenetv2.sh
```

For a persistent job:

```bash
mkdir -p logs
nohup scripts/run_layerwise_imagenetv2.sh > logs/layerwise_imagenetv2.log 2>&1 &
echo $! > logs/layerwise_imagenetv2.pid
tail -f logs/layerwise_imagenetv2.log
```

`--resume` reads the existing output and skips every successful experiment. JSON is
atomically rewritten after each evaluation, so stopping the process does not lose earlier
rows. Do not launch two processes that write the same JSON file.

Before the full run, a one-layer partial-data smoke test is useful:

```bash
ptq-isolate --config configs/deit_tiny_imagenetv2.yaml --suite layerwise \
  --layer-start 0 --layer-end 1 --max-eval-samples 128 \
  --calibration-samples 64 --output outputs/smoke/deit_layer0.json
```

Layer ranges use zero-based indices with an inclusive start and exclusive end. They can
also split a model across GPUs, but each process must use a distinct `--output` file.
The analysis command accepts full result files and/or shards and de-duplicates identical
model/experiment IDs.

```bash
ptq-analyze-layerwise \
  outputs/imagenetv2/deit_tiny/deit_tiny_imagenetv2_layerwise.json \
  outputs/imagenetv2/swin_tiny/swin_tiny_imagenetv2_layerwise.json \
  outputs/imagenetv2/resnet18/resnet18_imagenetv2_layerwise.json
```

This writes `all_layerwise_results.csv`, `best_configuration_per_layer.csv`,
`best_compressed_configuration_per_layer.csv`, `analysis_summary.json`, and one top-1
drop heatmap per model under `outputs/imagenetv2/layerwise_analysis/`. “Best” means the
highest measured top-1 accuracy; exact ties favor fewer bits and then higher sparsity.
CSV export works without Matplotlib; plots are skipped if it is unavailable.

## Apply all energy-optimal layer choices simultaneously

The `joint` suite consumes a completed layer-wise JSON. For each requested top-1
accuracy-drop threshold, it independently selects the minimum-energy feasible
configuration for every layer using:

```text
normalized energy = (k / 32) * (a / b)
```

Here `a:b` retains `a` weights in each group of `b`, and dense uses a retained
fraction of one. Energy ties prefer the configuration with the smaller isolated
top-1 drop. It then applies all selected configurations to their layers at the
same time and measures full-network accuracy over all 10,000 images. The original
threshold constrains each isolated layer result; it does not guarantee that the
combined network drop remains below that threshold.

Run all three networks and all thresholds:

```bash
scripts/run_joint_thresholds_imagenetv2.sh
```

The script uses the standard layer-wise result locations under `outputs/imagenetv2`,
runs thresholds 0.1, 0.5, and 1.0 percentage points, permits the already-recorded
dense-reference mismatches, resumes completed joint evaluations, and exports CSVs.

An individual command is:

```bash
ptq-isolate \
  --config configs/deit_tiny_imagenetv2.yaml \
  --suite joint \
  --layerwise-results outputs/imagenetv2/deit_tiny/deit_tiny_imagenetv2_layerwise.json \
  --accuracy-thresholds 0.1 0.5 1.0 \
  --ignore-reference-tolerance \
  --resume
```

Persistent server execution:

```bash
mkdir -p logs
nohup scripts/run_joint_thresholds_imagenetv2.sh \
  > logs/joint_thresholds_imagenetv2.log 2>&1 &
echo $! > logs/joint_thresholds_imagenetv2.pid
tail -f logs/joint_thresholds_imagenetv2.log
```

Joint JSON files are written atomically after each threshold:

```text
outputs/imagenetv2/deit_tiny/deit_tiny_imagenetv2_joint_thresholds.json
outputs/imagenetv2/swin_tiny/swin_tiny_imagenetv2_joint_thresholds.json
outputs/imagenetv2/resnet18/resnet18_imagenetv2_joint_thresholds.json
```

The final CSV files are:

```text
outputs/imagenetv2/joint_analysis/joint_accuracy_summary.csv
outputs/imagenetv2/joint_analysis/joint_selected_layers.csv
outputs/imagenetv2/joint_analysis/analysis_summary.json
```

## Interaction-aware global configuration refinement

The `refinement` suite completes the network-wide search after the independent
layer choices have been initialized. For each accuracy budget it:

1. Builds each layer's isolated accuracy-energy Pareto frontier and retains
   `FP32__dense` as a guaranteed safety anchor.
2. Starts from the same minimum-energy per-layer assignment used by `joint`.
3. Splits ImageNetV2 matched-frequency deterministically and by class into a
   4,000-image refinement subset (four images per class) and a disjoint
   6,000-image final subset (six images per class), then measures every search
   probe only on the refinement subset.
4. If the global budget is violated, probes every layer's next safer frontier
   point and accepts the largest measured accuracy recovery per added energy.
5. If all single moves stall, probes pairs among the six best singles and then
   one block-wise next-safer move.
6. Once feasible, probes one-step aggressive moves and accepts the best feasible
   energy saving per measured accuracy cost.
7. Evaluates the final assignment once on the full final evaluation set.

Every probe and accepted move is atomically saved. `--resume` therefore reuses
the persistent assignment cache instead of repeating completed model evaluations.
The global constraint is optimized on the matched-frequency refinement subset;
the disjoint matched-frequency final accuracy is reported, not used by the
refinement search. Split indices and their SHA-256 fingerprints are stored in
the JSON output. The existing layer-wise JSON can still initialize the search,
but because that legacy sweep evaluated all 10,000 matched-frequency images,
its isolated lookup measurements are not a fully untouched selection source.

Install the new editable entry points in the existing virtual environment:

```bash
source .venv/bin/activate
python -m pip install --no-deps --no-build-isolation -e .
```

Run all three models using the existing `outputs-v2` layer-wise results and write
new results under `outputs-v5`:

```bash
mkdir -p logs
nohup scripts/run_global_refinement_imagenetv2.sh \
  > logs/global_refinement_imagenetv2.log 2>&1 &
echo $! > logs/global_refinement_imagenetv2.pid
tail -f logs/global_refinement_imagenetv2.log
```

If the layer-wise or destination roots differ, override them without editing the
script:

```bash
LAYERWISE_ROOT=outputs/imagenetv2 \
REFINEMENT_ROOT=outputs/refinement/imagenetv2 \
nohup scripts/run_global_refinement_imagenetv2.sh \
  > logs/global_refinement_imagenetv2.log 2>&1 &
```

An individual resumable command is:

```bash
ptq-isolate \
  --config configs/deit_tiny_imagenetv2.yaml \
  --suite refinement \
  --layerwise-results outputs-v2/imagenetv2/deit_tiny/deit_tiny_imagenetv2_layerwise.json \
  --output outputs-v5/imagenetv2/deit_tiny/deit_tiny_imagenetv2_global_refinement.json \
  --accuracy-thresholds 0.1 0.5 1.0 \
  --refinement-samples 4000 \
  --refinement-seed 42 \
  --refinement-pairwise-top-k 6 \
  --ignore-reference-tolerance \
  --resume
```

Useful controls are:

- `--no-energy-reclamation`: stop as soon as the global budget is repaired.
- `--no-refinement-block-fallback`: disable the block move after single/pair stalls.
- `--refinement-pairwise-top-k K`: control pairwise fallback cost; `0` disables it.
- `--refinement-max-accepted-moves N`: cap accepted search iterations.
- `--refinement-samples N`: change the labeled matched-frequency refinement
  split size. The remainder is the disjoint final set. Resume requires the same
  size and seed used to create the output file.
- `--refinement-seed N`: change the deterministic stratified split seed.
- `--calibration-samples N`: change calibration size for non-refinement suites.

The analysis command writes `refinement_summary.csv`, `refinement_moves.csv`,
`refinement_selected_layers.csv`, and `analysis_summary.json`:

```bash
ptq-analyze-refinement \
  outputs-v5/imagenetv2/deit_tiny/deit_tiny_imagenetv2_global_refinement.json \
  outputs-v5/imagenetv2/swin_tiny/swin_tiny_imagenetv2_global_refinement.json \
  outputs-v5/imagenetv2/resnet18/resnet18_imagenetv2_global_refinement.json \
  --output-dir outputs-v5/imagenetv2/analysis
```

For a persistent background job:

```bash
mkdir -p logs
nohup scripts/run_full_imagenetv2.sh > logs/full_imagenetv2.log 2>&1 &
echo $! > logs/full_imagenetv2.pid
```

Monitor it with:

```bash
tail -f logs/full_imagenetv2.log
ps -p "$(cat logs/full_imagenetv2.pid)" -o pid,etime,cmd
```

## Output files

```text
outputs/imagenetv2/deit_tiny/deit_tiny_imagenetv2_{baseline,quantization,sparsity}.json
outputs/imagenetv2/swin_tiny/swin_tiny_imagenetv2_{baseline,quantization,sparsity}.json
outputs/imagenetv2/resnet18/resnet18_imagenetv2_{baseline,quantization,sparsity}.json
outputs/imagenetv2/{deit_tiny,swin_tiny,resnet18}/*_layerwise.json
```

JSON is written atomically after every experiment, so completed results survive a stopped
job. Each file includes the dense baseline/reference check, exact model and preprocessing,
environment versions, calibration/evaluation metadata, layer scope, per-layer error and
sparsity, accuracy drops, runtime, and isolation checks.

## Quantization sweep (17 experiments)

- Weight-only: W8/W6/W4/W3/W2 with FP32 activations.
- Activation-only: A8/A6/A4/A3/A2 with FP32 weights.
- Joint: W8A8, W6A6, W4A8, W4A6, W4A4, W3A4, W2A4.

Weights use signed symmetric per-output-channel fake quantization. Activations use signed
symmetric per-layer fake quantization with a fixed calibration range. Fake quantization
measures numerical accuracy and does not imply integer-kernel speedup.

## Sparsity sweep (15 experiments)

- Random, magnitude, and Wanda policies.
- Per-layer unstructured 25%, 50%, and 75% sparsity.
- Structured 2:4 and 1:4 sparsity.

Transformer runs select QKV, attention output, and both MLP projections in every block.
The CNN run selects residual-stage convolutions while leaving the stem and classifier
dense. Sparsity experiments remain FP32 throughout.

## Matched OBC and SparseGPT comparisons

The comparison controllers reconstruct the **legacy v4** protocol from saved
references, including preprocessing, layer scope, calibration/refinement splits,
and dense checks. They do not rerun SQuaRE. The current general refinement
script instead uses the v5 split described above; results from those protocols
must not be mixed.

- **OBC-block256:** OBC-inspired block-diagonal OBS with 256-column Hessian
  blocks, matched quantization and bounded reconstruction-score DP allocation.
  This is an approximation to published OBC, not a full-Hessian reproduction.
- **SparseGPT-adapted:** full sampled-input Gram matrix, 128-column processing
  blocks with cross-block compensation, matched vision weight/activation
  quantization and bounded DP allocation. It reuses verified OBC input statistics;
  it is not an unchanged reproduction of the published LLM experiments.

Run OBC with the actual dataset parent directory (containing both ImageNetV2
variants):

```bash
bash scripts/run_obc_v4_15h.sh --data-root data/imagenetv2
```

After a complete OBC study, launch SparseGPT using the original server artifacts
and environment:

```bash
bash scripts/run_sparsegpt_v4_2h.sh --check-only
bash scripts/run_sparsegpt_v4_2h.sh
```

Defaults are `outputs-obc/v4-block256/` and `outputs-sparsegpt/v4/`. The SparseGPT
launcher accepts `--obc-root` and `--output-root` overrides and recovers the
original interpreter from the OBC ledger. The two-hour controller profiles first
and stops if projected work cannot fit; the cap does not guarantee completion.
Copied comparison CSVs alone cannot substitute for the complete OBC artifacts.
Both controllers retain elapsed work and resumable caches; GPU experiments run
sequentially. Keep the original Torch/timm environment for historical matching.

Each completed comparison exports accuracy/feasibility summaries, layer
configurations, runtime accounting and protocol metadata. Full model directories
also retain checkpoints, predictions, manifests and verification records.
See [OBC instructions](docs/OBC_COMPARISON.md),
[OBC recovery](OBC_RECOVERY.md), and
[SparseGPT instructions and attribution](SPARSEGPT_COMPARISON.md).

Search uses the unweighted selected-layer average
`mean(retained_fraction * weight_bits / 32)`. A separate post-hoc
parameter-weighted proxy, `sum(P * retained_fraction * weight_bits) / (32 * sum(P))`,
estimates ideal selected-weight payload. It excludes packing/metadata overhead
and must not be labeled measured energy. Weighting existing configurations does
not rerun or change the search objective. Refinement feasibility and final-set
feasibility are reported separately; the former does not guarantee the latter.

Compact paper-comparison exports and their schema are in
[outputs-comparison/v4](outputs-comparison/v4/README.md).
Large model caches and local analysis reports are not part of that archive.

## Practical notes

- Baseline, quantization, and sparsity use different filenames, so smoke tests cannot
  overwrite a baseline file. A later smoke quantization run can overwrite an earlier full
  quantization result; use `--output` for custom smoke-test filenames if needed.
- ResNet Wanda calibration uses exact unfolded convolution statistics and therefore uses
  batch size 16 and 256 calibration images to control GPU memory.
- Swin-Tiny is the slowest model. Run models sequentially unless the server has multiple
  GPUs and you explicitly assign a different GPU to each process.
- Do not report fake-quantization runtime as hardware latency or sparse speedup.
- Report absolute accuracy loss in percentage points relative to the dense result measured
  in the same JSON.
- Layer-wise fake quantization uses one fixed activation range per layer, collected in a
  single dense calibration pass. This is valid because all layers preceding the isolated
  target remain dense FP32 in every experiment.
