"""Profile, budget and run three matched SparseGPT models within two GPU-job hours."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

from ..obc.cli import exclusive_output, now
from ..obc.study import MODELS, remaining_seconds, recover_unfinished_attempt, run_child, verify_package
from ..obc.evaluation import file_sha256
from ..obc.protocol import digest
from ..runner import _atomic_json
from .protocol import source_fingerprint
from .gpu import available_gpu
from .cache import candidate_keys


def verify_patch(root):
    verify_package(root)
    manifest = root / "SPARSEGPT_PATCH_MANIFEST.json"
    if manifest.exists():
        for name, expected in json.loads(manifest.read_text())["files"].items():
            if file_sha256(root / name) != expected:
                raise ValueError(f"SparseGPT patch file changed: {name}")


def preflight(root, source, references):
    study = json.loads((source / "study.json").read_text())
    if study.get("status") != "completed" or set(study.get("completed_models", [])) != set(MODELS):
        raise ValueError("OBC study must have all three completed models before calibration reuse")
    if study["fingerprint"] != digest(study["identity"]):
        raise ValueError("OBC study identity is inconsistent")
    data_root = Path(study["identity"]["data_root"])
    for model, reference in zip(MODELS, references):
        directory = source / model
        metadata = json.loads((directory / "manifest.json").read_text())
        result = json.loads((directory / "results.json").read_text())
        if not result.get("completed_at") or result["reference_sha256"] != file_sha256(reference):
            raise ValueError(f"Incomplete or different reference for {model}")
        for name in ("base_model.pt", "dense_checks.json", "data_manifest.json", "activation_ranges.pt", "verification.json"):
            if not (directory / name).is_file():
                raise FileNotFoundError(f"Required reusable OBC artifact missing: {directory / name}")
        for index in range(len(metadata["layer_shapes"])):
            stats = directory / "cache" / f"layer_{index:03d}" / "statistics.pt"
            if not stats.is_file():
                raise FileNotFoundError(f"Full OBC statistics required: {stats}")
        ref = json.loads(reference.read_text())
        if ref["refinement_set"].get("split", {}).get("request", {}).get("source") == "eval_dir":
            raise ValueError("This release is the matched v4 comparison, not a v5 experiment")
        for field in ("eval_dir", "calib_dir"):
            if not (data_root / Path(ref["data"][field]).name).is_dir():
                raise FileNotFoundError(f"Dataset directory missing under recorded root: {data_root}")
    return study, data_root


def plan_from_profiles(output, source):
    """Conservative measured projection, explicitly an estimate rather than a promise."""
    models = {}
    for name in MODELS:
        directory = output / name
        result = directory / "results.json"
        if result.exists() and json.loads(result.read_text()).get("completed_at"):
            models[name] = {"missing_layer_banks": 0, "candidate_estimate_seconds": 0.,
                            "search_reserve_seconds": 0., "final_reserve_seconds": 0., "projected_seconds": 0.}
            continue
        pilot = json.loads((directory / "pilot.json").read_text())
        metadata = json.loads((directory / "manifest.json").read_text())
        rows = pilot["layers"]
        if not pilot.get("completed_at") or len(rows) < min(3, len(metadata["layer_shapes"])) or any(r["status"] != "completed" or not r["sampled_candidates_reusable"] for r in rows):
            raise ValueError(f"Full-row pilot incomplete for {name}; no automatic full launch")
        preparation = 0.
        missing = 0
        for index, shape in enumerate(metadata["layer_shapes"].values()):
            bank = directory / "cache" / f"layer_{index:03d}"
            if all(s["id"] in candidate_keys(bank) for s in metadata["identity"]["grid"]):
                continue
            missing += 1
            columns, weights = math.prod(shape[1:]), math.prod(shape)
            nearest = min(rows, key=lambda r: abs(math.log(columns / math.prod(r["shape"][1:]))) + abs(math.log(weights / math.prod(r["shape"]))))
            c = columns / math.prod(nearest["shape"][1:])
            w = weights / math.prod(nearest["shape"])
            preparation += nearest["candidate_seconds"] * max(c, w, c ** 3)
        # Per-network reservations include checkpoint loads/exports and auditing.
        search = max(90., 1.5 * 32 * pilot["dense_refinement_evaluation_seconds"] + 30.)
        final = max(90., 1.5 * 3 * pilot["dense_final_evaluation_seconds"] + 45.)
        if (directory / "frozen_selection.json").exists():
            search = 0.
        models[name] = {"missing_layer_banks": missing,
                        "candidate_estimate_seconds": 1.5 * preparation,
                        "search_reserve_seconds": search, "final_reserve_seconds": final,
                        "projected_seconds": 1.5 * preparation + search + final + 60.}
    return {"models": models, "projected_remaining_seconds": sum(r["projected_seconds"] for r in models.values()),
            "caveat": "Measured shape-based projection with 1.5 margin; contention, I/O and kernels can differ. All settings stay fixed.",
            "created_at": now()}


def study(args, *, root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    verify_patch(root)
    source, output = Path(args.obc_root).resolve(), Path(args.output_root).resolve()
    if source == output or output.is_relative_to(source):
        raise ValueError("SparseGPT output must be separate from OBC")
    references = [root / "outputs-v4/imagenetv2" / m / f"{m}_imagenetv2_global_refinement.json" for m in MODELS]
    old_study, data_root = preflight(root, source, references)
    if getattr(args, "check_only", False):
        print(f"[STUDY] Preflight ready: three completed OBC sources, full statistics, v4 references and dataset directories found. Data root: {data_root}. No GPU experiment started.", flush=True)
        return {"status": "preflight_ready"}
    device = old_study["identity"]["device"]
    if device not in ("cuda", "cuda:0"):
        raise ValueError("Expected the original CUDA device 0 study")
    os.environ["PYTHONPATH"] = str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONUNBUFFERED"] = "1"
    identity = {"source_sha256": source_fingerprint(), "obc_study_fingerprint": old_study["fingerprint"],
                "obc_root": str(source), "references": [file_sha256(r) for r in references],
                "data_root": str(data_root), "total_seconds": 7200., "device": device,
                "processing_block_size": 128, "max_evaluations": 32, "budgets": [.1, .5, 1.]}
    with exclusive_output(output) as lock_fd:
        path = output / "study.json"
        ledger = json.loads(path.read_text()) if path.exists() else {
            "schema": 1, "identity": identity, "fingerprint": digest(identity),
            "total_seconds": 7200., "attempts": [], "created_at": now()}
        if ledger["fingerprint"] != digest(identity) or ledger["total_seconds"] != 7200.:
            raise ValueError("Study identity/time cap changed; resume cannot reset the two-hour budget")
        recover_unfinished_attempt(ledger)
        _atomic_json(path, ledger)

        def launch(stage, model, command, allowance):
            allowance = min(allowance, remaining_seconds(ledger))
            if allowance < 15:
                raise RuntimeError("Two-hour budget exhausted; saved work remains, no automatic extension")
            snapshot = available_gpu()
            attempt = {"stage": stage, "model": model, "command": command, "started_at": now(),
                       "allowance_seconds": allowance, "gpu_before_launch": snapshot}
            ledger["attempts"].append(attempt)
            ledger["status"] = "running"
            _atomic_json(path, ledger)
            log = output / "logs" / f"{model}-{stage}.log"
            print(f"[STUDY] {stage} {model}: allowance {allowance/60:.1f} min; log {log}", flush=True)
            started = time.monotonic()
            try:
                code, status = run_child(command, allowance, log, lock_fd)
            except Exception:
                attempt.update(charged_seconds=time.monotonic() - started, status="launch_error", finished_at=now())
                ledger["status"] = "incomplete"
                _atomic_json(path, ledger)
                raise
            attempt.update(charged_seconds=time.monotonic() - started, status=status, returncode=code, finished_at=now())
            ledger["remaining_seconds"] = remaining_seconds(ledger)
            ledger["status"] = "incomplete"
            _atomic_json(path, ledger)
            if code != 0:
                raise RuntimeError(f"{stage} {model} stopped ({status}, exit {code}); inspect {log}. No other model was silently substituted. Re-run after resolving the cause; remaining time is preserved.")

        def command(model, mode, allowance, search=0., final=0.):
            return [sys.executable, "-m", "pretrained_isolation.sparsegpt.cli", mode,
                    "--config", str(root / "configs" / f"{model}_imagenetv2.yaml"),
                    "--reference", str(references[MODELS.index(model)]), "--data-root", str(data_root),
                    "--obc-source", str(source / model), "--output-dir", str(output / model),
                    "--device", device, "--resume", "--max-hours", str((allowance - 15.) / 3600.),
                    "--search-reserve-seconds", str(search), "--final-reserve-seconds", str(final)]

        def complete(model):
            p = output / model / "results.json"
            return p.exists() and bool(json.loads(p.read_text()).get("completed_at"))

        selfcheck = output / "selfcheck.json"
        if not selfcheck.exists():
            launch("selfcheck", "all", [sys.executable, "-m", "pretrained_isolation.sparsegpt.selfcheck",
                "--device", device, "--output", str(selfcheck)], 60.)
        if not json.loads(selfcheck.read_text()).get("passed"):
            raise ValueError("CUDA numerical self-check failed")
        for name in MODELS:
            pilot = output / name / "pilot.json"
            if pilot.exists():
                saved = json.loads(pilot.read_text())
                if saved.get("completed_at") and all(r["status"] == "completed" for r in saved["layers"]):
                    continue
            allowance = min(180., remaining_seconds(ledger))
            launch("profile", name, command(name, "profile", allowance), allowance)
        plan = plan_from_profiles(output, source)
        plan["remaining_seconds"] = remaining_seconds(ledger)
        plan["fits_remaining_allowance"] = plan["projected_remaining_seconds"] <= remaining_seconds(ledger)
        _atomic_json(output / "profile_plan.json", plan)
        print(f"[STUDY] estimated remaining {plan['projected_remaining_seconds']/60:.1f} min; available {remaining_seconds(ledger)/60:.1f} min", flush=True)
        if args.phase == "profile":
            ledger["status"] = "profile_complete"
            _atomic_json(path, ledger)
            return ledger
        if not plan["fits_remaining_allowance"] and not all(complete(n) for n in MODELS):
            ledger["status"] = "profile_exceeds_remaining_budget"
            _atomic_json(path, ledger)
            raise RuntimeError("Profile projection exceeds the remaining two-hour allowance. Send profile_plan.json for review; no full run started and settings were not weakened.")
        for index, name in enumerate(MODELS):
            if complete(name):
                continue
            unfinished = [n for n in MODELS[index:] if not complete(n)]
            estimate = plan["models"][name]
            weights = sum(plan["models"][n]["projected_seconds"] for n in unfinished)
            allowance = remaining_seconds(ledger) * estimate["projected_seconds"] / weights
            if allowance <= estimate["search_reserve_seconds"] + estimate["final_reserve_seconds"] + 30:
                raise RuntimeError("Insufficient remaining time for full search/final reserves")
            launch("run", name, command(name, "run", allowance, estimate["search_reserve_seconds"], estimate["final_reserve_seconds"]), allowance)
            if not complete(name):
                raise RuntimeError(f"{name} reached its allocated time without complete results. Preserve outputs; no partial bank counts as a result.")
        from .audit import verify_run
        from .compare import compare_study
        for name in MODELS:
            verify_run(output / name)
        ledger["completed_models"] = list(MODELS)
        # CPU export is outside the GPU-job ledger, as in the OBC study.
        compare_study(root, source, output, output / "comparison")
        ledger.update(status="completed", updated_at=now(), remaining_seconds=remaining_seconds(ledger))
        _atomic_json(path, ledger)
        print(f"[STUDY] completed; models {list(MODELS)}; charged {(7200-remaining_seconds(ledger))/3600:.3f} GPU-job hours", flush=True)
        return ledger


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--obc-root", default="outputs-obc/v4-block256")
    p.add_argument("--output-root", default="outputs-sparsegpt/v4")
    p.add_argument("--phase", choices=("all", "profile", "run"), default="all")
    p.add_argument("--check-only", action="store_true", help="Check package, references and required paths without starting GPU jobs")
    args = p.parse_args()
    try:
        study(args)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        raise SystemExit(f"[STUDY] {error}") from error


if __name__ == "__main__":
    main()
