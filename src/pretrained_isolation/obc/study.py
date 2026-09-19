"""Run only OBC against saved references, with one persistent total time budget."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .cli import exclusive_output, now
from .evaluation import file_sha256
from .protocol import digest, source_fingerprint
from ..runner import _atomic_json

MODELS = ("deit_tiny", "swin_tiny", "resnet18")
ALLOWANCES = (3 * 3600, 5 * 3600, 5 * 3600)


def remaining_seconds(ledger):
    return max(0., ledger["total_seconds"] - sum(a.get("charged_seconds", 0.) for a in ledger["attempts"]))


def recover_unfinished_attempt(ledger):
    # The child inherits the study lock, so a new owner cannot overlap it.
    # If the parent died, conservatively charge the unknown interval rather
    # than allowing a restarted job to silently reset its 15-hour allowance.
    for attempt in ledger["attempts"]:
        if "charged_seconds" not in attempt:
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(attempt["started_at"])).total_seconds()
            attempt.update(charged_seconds=max(0., elapsed), status="interrupted_interval_charged", finished_at=now())


def run_child(command, timeout, log_path, lock_fd):
    """Own process group; SIGTERM grace is included inside the assigned allowance."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("a") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, pass_fds=(lock_fd,))
        try:
            process.wait(timeout=max(.01, timeout - 10.))
            status = "exited"
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            status = "time_cap"
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=max(.01, timeout - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    return process.returncode, status


def verify_package(root):
    path = root / "PACKAGE_MANIFEST.json"
    if path.exists():
        for name, expected in json.loads(path.read_text())["files"].items():
            if file_sha256(root / name) != expected:
                raise ValueError(f"Package file changed: {name}. Use a clean extracted package.")


def study(args):
    root = Path(__file__).resolve().parents[3]
    verify_package(root)
    output = Path(args.output_root).resolve()
    refs = Path(args.reference_root).resolve()
    data_root = Path(args.data_root).resolve()
    reference_paths = [refs / model / f"{model}_imagenetv2_global_refinement.json" for model in MODELS]
    for path in reference_paths:
        ref = json.loads(path.read_text())
        if not ref.get("completed_at_utc") or len(ref.get("experiments", [])) != 3:
            raise ValueError(f"Incomplete reference: {path}")
        for field in ("eval_dir", "calib_dir"):
            folder = data_root / Path(ref["data"][field]).name
            if not folder.is_dir():
                raise FileNotFoundError(f"Required data directory not found: {folder}")
    identity = {"source_sha256": source_fingerprint(), "references": [file_sha256(p) for p in reference_paths],
                "data_root": str(data_root), "hessian_block_size": args.block_size, "row_batch": args.row_batch,
                "device": args.device, "total_hours": float(args.total_hours), "max_evaluations": 32,
                "budgets": [.1, .5, 1.], "positions_per_example": 32, "score_rows": 2048,
                "damping": .01, "numerics": "square-default"}
    with exclusive_output(output) as lock_fd:
        path = output / "study.json"
        ledger = json.loads(path.read_text()) if path.exists() else {
            "schema": 1, "identity": identity, "fingerprint": digest(identity),
            "total_seconds": args.total_hours * 3600, "attempts": [], "created_at": now()}
        if ledger["fingerprint"] != digest(identity):
            raise ValueError("Study identity/budget changed; resume uses the original total allowance")
        recover_unfinished_attempt(ledger)
        extension = getattr(args, "extend_total_hours", None)
        if extension is not None:
            target = extension * 3600
            if target < ledger["total_seconds"]:
                raise ValueError("Explicit extension cannot reduce the existing total allowance")
            if target > ledger["total_seconds"]:
                ledger.setdefault("explicit_budget_extensions", []).append({
                    "previous_total_seconds": ledger["total_seconds"], "total_seconds": target, "at": now()})
                ledger["total_seconds"] = target
        _atomic_json(path, ledger)

        def launch(stage, model, command, allowance):
            allowance = min(allowance, remaining_seconds(ledger))
            if allowance <= 10:
                return False
            attempt = {"stage": stage, "model": model, "started_at": now(), "allowance_seconds": allowance,
                       "command": command}
            ledger["attempts"].append(attempt)
            _atomic_json(path, ledger)
            started = time.monotonic()
            print(f"[STUDY] {stage} {model}: allowance {allowance/3600:.2f} h; log {output / 'logs' / (model + '-' + stage + '.log')}", flush=True)
            try:
                code, status = run_child(command, allowance, output / "logs" / f"{model}-{stage}.log", lock_fd)
            except Exception:
                attempt.update(charged_seconds=time.monotonic() - started, status="launch_error", finished_at=now())
                _atomic_json(path, ledger)
                raise
            attempt.update(charged_seconds=time.monotonic() - started, status=status, returncode=code, finished_at=now())
            model_status = output / model / "status.json"
            if model_status.exists():
                attempt["model_status_at_exit"] = json.loads(model_status.read_text()).get("status")
            _atomic_json(path, ledger)
            return code == 0

        def command(model, mode, allowance):
            index = MODELS.index(model)
            return [sys.executable, "-m", "pretrained_isolation.obc.cli", mode,
                "--config", str(root / "configs" / f"{model}_imagenetv2.yaml"),
                "--reference", str(reference_paths[index]), "--output-dir", str(output / model),
                "--data-root", str(data_root), "--device", args.device, "--resume",
                "--hessian-block-size", str(args.block_size), "--row-batch", str(args.row_batch),
                "--numerics", "square-default", "--max-hours", str(max(1., allowance - 20.) / 3600),
                "--search-reserve-seconds", "300" if mode == "run" else "0",
                "--final-reserve-seconds", "300" if mode == "run" else "0"]

        def complete(model):
            result_path = output / model / "results.json"
            return result_path.exists() and bool(json.loads(result_path.read_text()).get("completed_at"))

        selfcheck_path = output / "selfcheck.json"
        if selfcheck_path.exists() and not json.loads(selfcheck_path.read_text()).get("passed"):
            raise ValueError("Existing server numerical self-check did not pass")
        if not selfcheck_path.exists():
            ok = launch("selfcheck", "all", [sys.executable, "-m", "pretrained_isolation.obc.selfcheck",
                "--device", args.device, "--output", str(output / "selfcheck.json")], 300.)
            if not ok:
                ledger["status"] = "selfcheck_failed_or_time_cap"
                _atomic_json(path, ledger)
                return ledger
        # Gate every network before spending hours compressing any network.
        for model in MODELS:
            checks = output / model / "dense_checks.json"
            if checks.exists() and json.loads(checks.read_text()).get("passed"):
                continue
            if not launch("baseline", model, command(model, "baseline", 500.), 500.):
                ledger["status"] = "baseline_failed_or_time_cap"
                _atomic_json(path, ledger)
                return ledger

        for model, target in zip(MODELS, ALLOWANCES):
            if complete(model):
                continue
            # Do not repeat nominal allocations on restart. Remaining reserve
            # is consumed only in the explicit second pass below.
            if any(a["stage"] == "run" and a["model"] == model for a in ledger["attempts"]):
                continue
            allowance = min(target, remaining_seconds(ledger))
            if allowance > 660:
                launch("run", model, command(model, "run", allowance), allowance)
        for model in MODELS:
            if complete(model):
                continue
            failures = [a for a in ledger["attempts"] if a["model"] == model and a["stage"] in ("run", "resume")]
            if any(a.get("returncode", 0) != 0 and a.get("status") != "time_cap" for a in failures):
                continue  # Never repeatedly retry a correctness/error failure.
            allowance = remaining_seconds(ledger)
            if allowance > 660:
                launch("resume", model, command(model, "run", allowance), allowance)

        from .audit import verify_run
        from .compare import compare
        finished = [m for m in MODELS if complete(m)]
        for model in finished:
            verify_run(output / model)
        ledger["completed_models"] = finished
        ledger["remaining_seconds"] = remaining_seconds(ledger)
        ledger["status"] = "completed" if len(finished) == len(MODELS) else "incomplete"
        ledger["updated_at"] = now()
        _atomic_json(path, ledger)
        if finished:
            comparison = output / ("comparison" if len(finished) == len(MODELS) else "comparison_partial")
            compare([reference_paths[MODELS.index(m)] for m in finished],
                    [output / m / "results.json" for m in finished], comparison,
                    [root / "outputs-v2/imagenetv2" / m / f"{m}_imagenetv2_layerwise.json" for m in finished])
        print(f"[STUDY] {ledger['status']}; completed {finished}; charged {sum(a['charged_seconds'] for a in ledger['attempts'])/3600:.3f} GPU-job hours", flush=True)
        return ledger


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--reference-root", default="outputs-v4/imagenetv2")
    p.add_argument("--output-root", default="outputs-obc/v4-block256")
    p.add_argument("--total-hours", type=float, default=15.)
    p.add_argument("--extend-total-hours", type=float,
                   help="Explicitly authorize a larger total ceiling for the same cached study; never resets consumed time")
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--row-batch", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if not math.isfinite(args.total_hours) or args.total_hours <= 0 or args.block_size < 8 or args.block_size % 8 or args.row_batch < 1:
        p.error("Require positive total hours, row batch, and block size divisible by 8")
    if args.extend_total_hours is not None and (not math.isfinite(args.extend_total_hours) or args.extend_total_hours < args.total_hours):
        p.error("Explicit extension must be finite and at least the original total allowance")
    result = study(args)
    if result["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
