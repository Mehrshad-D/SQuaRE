# Compact comparison archive for Experimental Results, Section D

This archive contains the legacy v4 comparison of SQuaRE, OBC-block256 and
SparseGPT-adapted. It supplies the SQuaRE/OBC table and retains the SparseGPT
comparison for the accompanying text. It does not contain v5 experiments.

## Files

- `comparison_summary.csv`: 27 method/model/budget rows with refinement and
  final-evaluation accuracy, separate feasibility flags, original unweighted
  cost, and post-hoc parameter-weighted cost. SQuaRE rows also contain initial
  refinement drop, initial weighted cost, repair/reclaim batches and changed
  layer counts for the paper table. Blank baseline cells mean not applicable.
- `layer_configurations.csv`: all 1,035 final layer assignments, augmented with
  original dense weight counts and nominal retained fractions. This is enough
  to independently recompute both cost columns without model checkpoints.
- `provenance.json`: method settings/adaptations, original input hashes, v4
  reference hashes and checksums for both archived CSV files. Source paths are
  recorded as provenance; the local `outsputs-sparsegpt` upload is not required
  to read this archive. Original SQuaRE references remain in `outputs-v4`.

## Interpreting the columns

Accuracy and accuracy drops are in percent and percentage points, respectively.
`final_*` denotes evaluation on the matched-frequency final set;
`refinement_*` denotes the threshold-0.7 refinement set used for selection.
Negative drops indicate improvement over the corresponding dense baseline.
Passing a refinement constraint does not imply passing the final constraint.

The original search objective is the equal-layer average of `d * w / 32`.
The paper's weighted column is `sum(P * d * w) / (32 * sum(P))`, where `P` is
original layer weight count, `d` is retained fraction and `w` is weight bits.
The scope includes only selected layers. Dense FP32 is 1; lower is better.
This estimates ideal weight payload, excludes metadata and packing overhead,
and is not measured hardware energy. Weighting is post hoc and does not change
search assignments. Initial and final SQuaRE costs use the same weighting.
`normalized_energy_proxy` in the layer file is the individual layer's `d*w/32`.

For aggregate comparisons, take the arithmetic mean of the nine unrounded
`parameter_weighted_cost` values for each method. Compute a relative reduction
as `100 * (1 - mean_SQuaRE / mean_baseline)`. A negative reduction means an
increase. Do not substitute a sparsity-only metric for this quantity.

Repair columns count accepted single-, pair-, and block-wise repair batches;
reclaim counts accepted energy-reclamation batches. Changed layers are counted
relative to the initial SQuaRE assignment. Initial assignments and complete
search histories are in the tracked v4 reference JSON files.

## Scope and limitations

OBC-block256 is a block-diagonal OBC-inspired approximation. SparseGPT-adapted
uses a full sampled-input Gram matrix and matched vision quantization, with a
bounded outer allocation search. Neither is an unchanged published baseline.
The metadata describes their settings and upstream revisions.

The baseline export reports matched protocols, but the uploaded comparison
files do not include model-specific checkpoints, predictions or data manifests.
This archive supports numerical reconstruction, not an independent checkpoint
or image-identity audit. Keep the full server artifacts separately. Legacy
SQuaRE layer-wise selection also used final-set images; v4 is not an
untouched-test study. Preserve separate refinement/final feasibility when
reporting cost trade-offs.

Only these compact comparison files are published here. Candidate caches,
checkpoints, raw logs, figures, runtime reports and local analysis packages are
intentionally outside this archive.
