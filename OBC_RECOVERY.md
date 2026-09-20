# Resume the interrupted v0.9.0 comparison

This add-on retries Swin using the original v0.9.0 model runner. It adds files
without replacing any original package files, numerical code, or results.
Keep the original package and output directories in their current locations.

The reported study completed DeiT-Tiny and ResNet18 and charged 8.142 of its
15 GPU-job hours. Approximately 6.858 hours remain. The original launcher
deliberately skips failed jobs such as Swin's CUDA out-of-memory failure;
simply restarting that launcher will not retry Swin.

## Install and run

Copy `obc-v0.9.0-recovery.zip` and `obc-v0.9.0-recovery.zip.sha256` into the
existing `pretrained-isolation-framework-v0.9.0` directory on the server.
From that directory, with the same Python environment active:

```bash
sha256sum -c obc-v0.9.0-recovery.zip.sha256
unzip obc-v0.9.0-recovery.zip
mkdir -p logs
nohup python -u scripts/resume_obc_v090.py --model swin_tiny > logs/obc-swin-recovery.log 2>&1 &
```

The saved study supplies the dataset path and all model settings. No new
dataset argument is needed. The default output root is
`outputs-obc/v4-block256`; use `--output-root` only if that was different in
the original run.

Arrange access to an idle GPU before launching. The helper checks `nvidia-smi`
and refuses to launch if another compute process occupies the selected GPU
or fewer than 4096 MiB are free. This check is a snapshot, not a reservation:
another job can still start later. If the helper reports a busy GPU, rerun
the same command when the GPU is available. It does not kill other jobs or
spend the allowance waiting for them. Keep the original GPU selection; on a
multi-GPU machine an ambiguous numeric CUDA device mapping is rejected.

## Monitor and verify completion

```bash
tail -f logs/obc-swin-recovery.log
```

After the launcher reports that it has started Swin, its detailed output is:

```bash
tail -f outputs-obc/v4-block256/logs/swin_tiny-recovery.log
```

The old `swin_tiny-run.log` remains the record of the failed attempt and will
not show recovery progress. A successful controller log ends with:

```text
[RECOVERY] completed; completed ['deit_tiny', 'swin_tiny', 'resnet18']; remaining ... h
```

The helper validates the original package, reference files, numerical source
fingerprint, and saved command before retrying. It invokes the original
runner with `--resume`, preserving compatible Swin caches and completed
DeiT/ResNet results. The failed attempt remains in `study.json`, and the new
attempt's elapsed time is added to the same ledger. The helper neither resets
the clock nor extends the total allowance.

On completion it audits all three saved runs and exports the full comparison
to `outputs-obc/v4-block256/comparison/`, including:

- `report.md` and `methodology.md`
- `comparison_summary.csv` and `comparison_table.tex`
- `layer_configurations.csv` and `runtime_by_model.csv`
- `comparison_metadata.json`

Per-model results and `verification.json` remain inside each model's existing
directory. If Swin fails again or reaches the cap before completion, the
controller exits nonzero, retains partial work, and reports `incomplete`.
There is no guarantee that the remaining allowance is sufficient. Do not
delete the ledger, caches, or completed results to restart the clock.

The comparison still uses the original v4 protocol and the label
**OBC-inspired blockwise OBS (B=256, adapted)**. Recovery does not make the
earlier contended timings suitable for an unqualified hardware speed
comparison. Report the shared-GPU conditions and preserve the study ledger
alongside the per-model runtime exports.
