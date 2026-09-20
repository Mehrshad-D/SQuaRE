"""Explicitly retry one failed v0.9.0 model without changing numerical code/caches.

Install this file under scripts/ in the existing v0.9.0 package. It deliberately
lives outside src/pretrained_isolation/obc so the numerical source fingerprint
and package's existing file checksums remain unchanged.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pretrained_isolation.obc.audit import verify_run
from pretrained_isolation.obc.cli import exclusive_output, now, parser as model_parser
from pretrained_isolation.obc.compare import compare
from pretrained_isolation.obc.evaluation import file_sha256
from pretrained_isolation.obc.protocol import digest, source_fingerprint
from pretrained_isolation.obc.study import (
    MODELS, recover_unfinished_attempt, remaining_seconds, run_child, verify_package,
)
from pretrained_isolation.runner import _atomic_json


def available_gpu():
    """A point-in-time safety check, not a GPU reservation or a wait loop."""
    def query(fields, kind):
        raw = subprocess.check_output(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            text=True, timeout=15)
        return [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(raw)) if row]
    devices = query("index,uuid,memory.total,memory.free", "gpu")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    if len(devices) > 1 and not visible.startswith("GPU-") and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("Multiple GPUs with ambiguous CUDA ordering; identify the original device by UUID before retrying")
    matches = [r for r in devices if r[0] == visible or (visible.startswith("GPU-") and r[1].startswith(visible))]
    if len(matches) != 1:
        raise RuntimeError("Cannot identify CUDA device 0; check CUDA_VISIBLE_DEVICES before retrying")
    index, uuid, total, free = matches[0]
    processes = query("gpu_uuid,pid", "compute-apps")
    busy = [pid for gpu, pid in processes if gpu == uuid and pid != str(os.getpid())]
    if busy:
        raise RuntimeError(f"GPU {index} is still occupied by compute PIDs {', '.join(busy)}. Wait for those jobs to finish; no retry was launched.")
    if float(free) < 4096:
        raise RuntimeError(f"GPU {index} has only {free} MiB free; require at least 4096 MiB before retrying")
    return {"gpu_index": index, "gpu_uuid": uuid, "total_mib": float(total), "free_mib": float(free),
            "other_compute_pids": busy, "checked_at": now(), "is_reservation": False}


def replay_command(ledger, model, directory, allowance):
    attempts = [a for a in ledger["attempts"] if a["model"] == model and a["stage"] in ("run", "resume")]
    if not attempts:
        raise ValueError("No original model command exists to resume")
    command = list(attempts[-1]["command"])
    if command[1:4] != ["-m", "pretrained_isolation.obc.cli", "run"]:
        raise ValueError("Recorded command is not the expected OBC model runner")
    args = model_parser().parse_args(command[3:])
    identity = ledger["identity"]
    if (Path(args.output_dir).resolve() != directory.resolve() or not args.resume
        or Path(args.data_root).resolve() != Path(identity["data_root"]).resolve()
        or args.hessian_block_size != identity["hessian_block_size"]
        or args.row_batch != identity["row_batch"] or args.numerics != identity["numerics"]
        or args.max_evaluations != identity["max_evaluations"] or args.budgets != identity["budgets"]
        or args.damping != identity["damping"] or args.score_rows != identity["score_rows"]
        or args.positions_per_example != identity["positions_per_example"] or args.device != identity["device"]):
        raise ValueError("Recorded retry command differs from the original study protocol")
    if file_sha256(args.reference) != identity["references"][MODELS.index(model)]:
        raise ValueError("Reference file changed")
    command[0] = sys.executable
    command[command.index("--max-hours") + 1] = str((allowance - 20.) / 3600.)
    return command


def recover(output_root, model="swin_tiny", *, root=ROOT):
    verify_package(root)
    output = Path(output_root).resolve()
    path = output / "study.json"
    if not path.exists():
        raise FileNotFoundError(f"Original study ledger not found: {path}")
    # Ensure the child uses the original package, not an installed or newer copy.
    os.environ["PYTHONPATH"] = str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONUNBUFFERED"] = "1"
    with exclusive_output(output) as lock_fd:
        ledger = json.loads(path.read_text())
        if ledger["fingerprint"] != digest(ledger["identity"]) or ledger["identity"]["source_sha256"] != source_fingerprint():
            raise ValueError("Numerical source/study identity differs; do not change the v0.9.0 code")
        if ledger["identity"]["device"] not in ("cuda", "cuda:0"):
            raise ValueError("This recovery launcher expects the original CUDA device 0 study")
        recover_unfinished_attempt(ledger)
        _atomic_json(path, ledger)

        def complete(name):
            f = output / name / "results.json"
            return f.exists() and bool(json.loads(f.read_text()).get("completed_at"))

        if not complete(model):
            allowance = remaining_seconds(ledger)
            if allowance <= 660:
                raise RuntimeError(f"Only {allowance/3600:.3f} h remains; this launcher does not extend the total budget")
            command = replay_command(ledger, model, output / model, allowance)
            snapshot = available_gpu()  # Refuse a busy GPU without burning hours waiting.
            attempt = {"stage": "resume", "model": model, "started_at": now(),
                       "allowance_seconds": allowance, "command": command,
                       "explicit_recovery": True, "gpu_before_retry": snapshot,
                       "recovery_launcher_sha256": file_sha256(__file__)}
            ledger["attempts"].append(attempt)
            ledger["status"] = "recovering"
            _atomic_json(path, ledger)
            log = output / "logs" / f"{model}-recovery.log"
            print(f"[RECOVERY] Resuming only {model}; remaining allowance {allowance/3600:.3f} h; log {log}", flush=True)
            started = time.monotonic()
            try:
                code, status = run_child(command, allowance, log, lock_fd)
            except Exception:
                attempt.update(charged_seconds=time.monotonic() - started, status="launch_error", finished_at=now())
                ledger["status"] = "incomplete"
                _atomic_json(path, ledger)
                raise
            attempt.update(charged_seconds=time.monotonic() - started, status=status, returncode=code, finished_at=now())
            status_path = output / model / "status.json"
            if status_path.exists():
                attempt["model_status_at_exit"] = json.loads(status_path.read_text()).get("status")
            ledger["status"] = "incomplete"
            ledger["remaining_seconds"] = remaining_seconds(ledger)
            _atomic_json(path, ledger)

        finished = [name for name in MODELS if complete(name)]
        for name in finished:
            verify_run(output / name)
        ledger["completed_models"] = finished
        ledger["remaining_seconds"] = remaining_seconds(ledger)
        _atomic_json(path, ledger)
        if finished:
            references = []
            for name in finished:
                manifest = json.loads((output / name / "manifest.json").read_text())
                reference = Path(manifest["reference_file"])
                if file_sha256(reference) != ledger["identity"]["references"][MODELS.index(name)]:
                    raise ValueError("Comparison reference changed")
                references.append(reference)
            comparison = output / ("comparison" if len(finished) == len(MODELS) else "comparison_partial")
            compare(references, [output / name / "results.json" for name in finished], comparison,
                    [root / "outputs-v2/imagenetv2" / name / f"{name}_imagenetv2_layerwise.json" for name in finished])
        ledger["status"] = "completed" if len(finished) == len(MODELS) else "incomplete"
        ledger["updated_at"] = now()
        _atomic_json(path, ledger)
        print(f"[RECOVERY] {ledger['status']}; completed {finished}; remaining {remaining_seconds(ledger)/3600:.3f} h", flush=True)
        return ledger


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", default="outputs-obc/v4-block256")
    p.add_argument("--model", choices=MODELS, default="swin_tiny")
    args = p.parse_args()
    try:
        result = recover(args.output_root, args.model)
    except (RuntimeError, ValueError, FileNotFoundError, subprocess.SubprocessError) as error:
        raise SystemExit(f"[RECOVERY] {error}") from error
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
