"""SparseGPT vision adapter with read-only reuse of completed OBC calibration."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import torch

from ..config import load_config, layerwise_configurations
from ..data import make_refinement_loaders
from ..engine import calibrate, environment, instrument, seed_all
from ..runner import _atomic_json, create_pretrained
from . import ALGORITHM_VERSION, UPSTREAM_REVISION, METHOD
from ..obc.core import CompressionTimeout, allocate_dp
from .core import build_candidates
from .cache import candidate_keys, load_candidate as checked_candidate, save_candidate as commit_candidate
from ..obc.evaluation import evaluate, file_sha256
from ..obc.numerics import configure_execution
from ..obc.protocol import (check_accuracy, check_protocol, dataset_manifest, digest,
                       matched_config, tensor_fingerprint)
from .protocol import (source_fingerprint, validate_source, copy_verified, validate_statistics, verify_reuse)
from ..obc.runtime import (apply_assignment, atomic_torch, collect_statistics,
                      load_tensors, synchronize)


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def exclusive_output(directory: Path):
    """OS releases the advisory lock even after a crash; no stale-lock guessing."""
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".obc.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another comparison process owns {directory}") from error
        try:
            yield stream.fileno()
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def representative_layers(modules):
    ordered = sorted(modules, key=lambda name: (modules[name].weight[0].numel(),
                                               modules[name].weight.numel(), name))
    return list(dict.fromkeys([ordered[0], ordered[len(ordered) // 2], ordered[-1]]))


def proposed_assignments(candidates, max_evaluations, deadline=None):
    """A fixed grid with dense anchor; no false monotonicity assumption or test feedback."""
    names = list(candidates)
    dense = {n: next(c["configuration"] for c in candidates[n]
                     if c["configuration"] == "FP32__dense") for n in names}
    yield dense
    seen = {tuple(dense.values())}
    minimum = sum(min(c["normalized_energy"] for c in candidates[n]) for n in names)
    maximum = float(len(names))
    # Bias resolution toward cheaper points; all budgets share this fixed pool.
    for i in range(max_evaluations - 1):
        fraction = i / max(1, max_evaluations - 2)
        cap = minimum + (maximum - minimum) * fraction ** 2
        indices = allocate_dp([candidates[n] for n in names], cap, deadline=deadline)
        assignment = {n: candidates[n][idx]["configuration"] for n, idx in zip(names, indices)}
        key = tuple(assignment.values())
        if key not in seen:
            yield assignment
            seen.add(key)


def best_feasible(probes, threshold, dense_top1):
    feasible = [p for p in probes if dense_top1 - p["accuracy"]["top1"] <= threshold + 1e-9]
    if not feasible:
        raise ValueError("Dense safety anchor missing or inconsistent")
    return min(feasible, key=lambda p: (p["normalized_energy"], -p["accuracy"]["top1"], p["key"]))


def save_dense_checks(path, fingerprint, dense, dense_refinement, reference, tolerance, execution):
    """Keep measured evidence even when a gate fails; never cache it as passed."""
    report = {"fingerprint": fingerprint, "dense_final": dense,
              "dense_refinement": dense_refinement, "execution_settings": execution,
              "tolerance_pp": tolerance, "checks": {}, "passed": True, "checked_at": now()}
    errors = []
    for label, actual, expected in (
        ("final", dense, reference["dense_baseline"]),
        ("refinement", dense_refinement, reference["refinement_set"]["dense_accuracy"]),
    ):
        row = {"actual": actual, "expected": expected,
               "actual_minus_expected_pp": {k: actual[k] - expected[k] for k in ("top1", "top5")}}
        try:
            check_accuracy(actual, expected, tolerance)
            row["passed"] = True
        except ValueError as error:
            row["passed"] = False
            row["error"] = str(error)
            errors.append(f"{label}: {error}")
        report["checks"][label] = row
    report["passed"] = not errors
    _atomic_json(path, report)
    if errors:
        raise ValueError("; ".join(errors) + f". Diagnostics saved to {path}")
    return report


def run(args):
    directory = Path(args.output_dir)
    with exclusive_output(directory):
        return _run(args, directory)


@contextmanager
def preparation_timer(directory, mode):
    started = time.perf_counter()
    try:
        yield
    finally:
        path = directory / "preparation_wall_attempts.json"
        rows = json.loads(path.read_text()) if path.exists() else []
        rows.append({"mode": mode, "seconds": time.perf_counter() - started, "finished_at": now()})
        _atomic_json(path, rows)


def _run(args, directory):
    invocation_start = time.monotonic()
    final_deadline = invocation_start + args.max_hours * 3600 if args.max_hours else None
    search_deadline = final_deadline - args.final_reserve_seconds if final_deadline else None
    operation_deadline = search_deadline - args.search_reserve_seconds if search_deadline else None
    reference_path = Path(args.reference)
    reference_bytes = reference_path.read_bytes()
    reference = json.loads(reference_bytes)
    if not reference.get("completed_at_utc") or any(e.get("status") != "succeeded" for e in reference.get("experiments", [])):
        raise ValueError("Reference run must be complete before a SparseGPT comparison")
    cfg = matched_config(load_config(args.config), reference)
    if args.data_root:
        for key in ("eval_dir", "calib_dir"):
            cfg["data"][key] = str(Path(args.data_root) / Path(reference["data"][key]).name)
    if args.batch_size:
        cfg["data"]["batch_size"] = args.batch_size
    if args.workers is not None:
        cfg["data"]["num_workers"] = args.workers
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")
    execution = configure_execution(args.numerics)
    for key in ("torch", "timm"):
        expected_version = reference.get("environment", {}).get(key)
        if expected_version and expected_version != environment().get(key):
            raise ValueError(f"Reference {key}={expected_version}, current={environment().get(key)}; use the original server environment")
    seed_all(cfg["seed"])
    setup_start = time.perf_counter()
    print(f"[SparseGPT] {args.mode}: {cfg['model']['name']} | {device}", flush=True)
    print(f"[SparseGPT] numerics={execution['mode']} | matmul_tf32={execution['cuda_matmul_allow_tf32']} "
          f"| cudnn_tf32={execution['cudnn_allow_tf32']}", flush=True)
    model = create_pretrained(cfg).eval().to(device)
    checkpoint_sha = tensor_fingerprint(model.state_dict())
    calibration, evaluation, data_cfg, nc, ne, output_indices, labels, split = make_refinement_loaders(model, cfg["data"])
    check_protocol(reference, data_cfg, split, nc, ne, labels)
    manifests = {"refinement": dataset_manifest(calibration.dataset),
                 "final": dataset_manifest(evaluation.dataset)}
    overlap = {x["sha256"] for x in manifests["refinement"]} & {x["sha256"] for x in manifests["final"]}
    if overlap:
        raise ValueError("Calibration/refinement and final sets contain duplicate image content")
    settings = {"algorithm": ALGORITHM_VERSION, "upstream_revision": UPSTREAM_REVISION,
                "damping": args.damping, "processing_block_size": args.processing_block_size,
                "positions_per_example": args.positions_per_example, "score_rows": args.score_rows,
                "quantizer": "SQuaRE symmetric per-output-channel minmax W / per-layer maxabs A",
                "conv_layout": "flattened input dimension (C,Kh,Kw)",
                "normalization_correction": False,
                "hessian": "full sampled-input Gram; cross-block compensation retained",
                "hessian_dtype": "float32", "damping_scope": "full-layer mean after upstream dead-input treatment",
                "quantization_scale_scope": "full output channel, not per block",
                "execution_settings": execution}
    identity = {"reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
                "checkpoint_sha256": checkpoint_sha, "source_sha256": source_fingerprint(),
                "dataset_sha256": {k: digest(v) for k, v in manifests.items()},
                "settings": settings, "model": cfg["model"]["name"],
                "selection": cfg["selection"], "data_config": data_cfg,
                "seed": cfg["seed"], "environment": environment(),
                "device": str(device), "batch_size": cfg["data"]["batch_size"],
                "grid": layerwise_configurations(cfg)}
    reused_metadata, reused_checks = validate_source(args.obc_source, identity, manifests)
    identity["reused_obc_fingerprint"] = reused_metadata["fingerprint"]
    fingerprint = digest(identity)
    manifest_path = directory / "manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if existing and existing["fingerprint"] != fingerprint:
        raise ValueError("Cache identity changed (code/data/checkpoint/settings/environment). Use a new output directory.")
    if existing and not args.resume:
        raise ValueError("Output exists; use --resume or a fresh directory")
    existing_search = directory / "search.json"
    if existing_search.exists():
        prior_search = json.loads(existing_search.read_text())
        if prior_search["settings"] != {"max_evaluations": args.max_evaluations, "budgets": args.budgets}:
            raise ValueError("Search settings changed; use a fresh directory")
    metadata = {"fingerprint": fingerprint, "identity": identity,
                "method": METHOD,
                "reference_file": str(reference_path.resolve()), "model": cfg["model"],
                "protocol": {"split": split, "refinement_samples": nc, "final_samples": ne},
                "adaptations": ["expanded 16-point grid", "signed symmetric quantizer and shared activation calibration",
                                "SQuaRE convolution grouping", "bounded reconstruction scoring",
                                "DP uses SQuaRE proxy; fixed resource grid with refinement accuracy selection",
                                "no BN tuning/statistics correction"],
                "limitations": ["Reference lacks original checkpoint/content hashes; agreement is checked via metadata and dense accuracy.",
                                "SQuaRE legacy layerwise selection used final images; historical comparison is not an untouched-test study.",
                                "Energy is an unweighted selected-layer proxy, not measured hardware energy."],
                "created_at": existing["created_at"] if existing else now()}
    metadata["adaptations"].extend(["dense-input candidate bank instead of sequential compressed-model calibration",
                                  "vision and convolution adapters; retained-N convention including 3:8",
                                  "joint SparseGPT pruning/weight quantization; matched activation quantization"])
    metadata["reused_obc_fingerprint"] = reused_metadata["fingerprint"]
    _atomic_json(manifest_path, metadata)
    _atomic_json(directory / "data_manifest.json", manifests)
    base_path = directory / "base_model.pt"
    if base_path.exists():
        if tensor_fingerprint(load_tensors(base_path)) != checkpoint_sha:
            raise ValueError("Saved base checkpoint does not match verified pretrained model")
    else:
        atomic_torch(base_path, {k: v.detach().cpu() for k, v in model.state_dict().items()})
    checks_path = directory / "dense_checks.json"
    if args.resume and checks_path.exists():
        checks = json.loads(checks_path.read_text())
        if checks["fingerprint"] != fingerprint:
            raise ValueError("Dense checks cache mismatch")
        dense, dense_refinement = checks["dense_final"], checks["dense_refinement"]
    else:
        # A fresh dense gate also measures current evaluation throughput for
        # the two-hour plan; no reliance on old contended timing estimates.
        dense = evaluate(model, evaluation, device, None, output_indices,
                         prediction_path=directory / "predictions/dense_final.pt", deadline=final_deadline)
        dense_refinement = evaluate(model, calibration, device, None, output_indices,
                         prediction_path=directory / "predictions/dense_refinement.pt", deadline=final_deadline)
    checks = save_dense_checks(checks_path, fingerprint, dense, dense_refinement, reference,
                               args.baseline_tolerance, execution)
    finished_path = directory / "results.json"
    if args.mode == "run" and finished_path.exists():
        finished = json.loads(finished_path.read_text())
        if finished.get("completed_at"):
            from .audit import verify_run
            verify_run(directory)
            return finished
    modules = instrument(model, cfg, 32, 32)
    expected = list(reference.get("pareto_frontiers", {}))
    if list(modules) != expected:
        raise ValueError("Selected layer names/order differ from reference")
    if len(modules) != reference["selected_layer_count"]:
        raise ValueError("Selected layer count differs from reference")
    def record_setup():
        path = directory / "setup_attempts.json"
        attempts = json.loads(path.read_text()) if path.exists() else []
        attempts.append({"mode": args.mode, "seconds": time.perf_counter() - setup_start, "completed_at": now()})
        _atomic_json(path, attempts)
    if args.mode == "baseline":
        record_setup()
        metadata["first_setup_seconds"] = existing.get("first_setup_seconds", time.perf_counter() - setup_start) if existing else time.perf_counter() - setup_start
        _atomic_json(manifest_path, metadata)
        print(f"[SparseGPT] Dense baseline checks passed: {checks_path}", flush=True)
        return checks
    if list(modules) != list(reused_metadata["layer_shapes"]) or {n: list(m.weight.shape) for n, m in modules.items()} != reused_metadata["layer_shapes"]:
        raise ValueError("Reused OBC layer shapes/order do not match")
    ranges_path = copy_verified(args.obc_source, directory, "activation_ranges.pt")
    ranges = load_tensors(ranges_path)
    if set(ranges) != set(modules):
        raise ValueError("Missing activation ranges")
    for name, module in modules.items():
        if ranges[name].numel() != 1 or not torch.isfinite(ranges[name]).all() or (ranges[name] < 0).any():
            raise ValueError("Invalid fixed activation range")
        module.act_amax.copy_(ranges[name])
    for module in modules.values():
        module.finalize()
    metadata["layer_shapes"] = {name: list(m.weight.shape) for name, m in modules.items()}
    # Record setup only once; later resumes are overhead, not fresh preparation.
    metadata["first_setup_seconds"] = existing.get("first_setup_seconds", time.perf_counter() - setup_start) if existing else time.perf_counter() - setup_start
    _atomic_json(manifest_path, metadata)
    record_setup()
    with preparation_timer(directory, args.mode):
        specs = identity["grid"]
        layer_names = representative_layers(modules) if args.mode == "profile" else list(modules)
        pilot_path = directory / "pilot.json"
        pilot = json.loads(pilot_path.read_text()) if args.resume and pilot_path.exists() else {
            "suite": "sparsegpt_pilot", "fingerprint": fingerprint, "model": cfg["model"],
            "environment": environment(), "layers": [], "settings": settings,
            "warning": "Full-row representative-layer timing estimates are preliminary; no compressed paper accuracy results."}
        pilot_settings = {"rows": args.pilot_rows, "layer_seconds": args.pilot_layer_seconds}
        if args.mode == "profile" and pilot.get("pilot_settings", pilot_settings) != pilot_settings:
            raise ValueError("Pilot sampling/time limits changed; use a fresh output directory")
        pilot["pilot_settings"] = pilot_settings
        attempts_path = directory / "preparation_attempts.json"
        attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
        for name in layer_names:
            index = list(modules).index(name)
            module = modules[name]
            layer_dir = directory / "cache" / f"layer_{index:03d}"
            stats_path = layer_dir / "statistics.pt"
            keys = candidate_keys(layer_dir)
            if args.mode == "run" and all(s["id"] in keys for s in specs):
                for spec in specs:
                    checked_candidate(layer_dir, spec["id"], fingerprint)
                print(f"[SparseGPT] cached {name}", flush=True)
                continue
            if args.mode == "profile" and any(row["layer"] == name and row["status"] == "completed" for row in pilot["layers"]):
                continue
            if operation_deadline and time.monotonic() >= operation_deadline:
                _atomic_json(directory / "status.json", {"status": "preparation_time_cap", "next_layer": name,
                                                         "message": "Resume to continue; no partial result is a completed comparison."})
                return
            print(f"[SparseGPT] statistics/candidates {name} {list(module.weight.shape)}", flush=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            copy_verified(args.obc_source, directory, str(stats_path.relative_to(directory)))
            stats = load_tensors(stats_path)
            validate_statistics(stats, module.weight[0].numel(), args.score_rows)
            stats["gram"] = stats["gram"].to(device)
            completed = {}
            for spec in specs:
                path = layer_dir / f"{spec['id']}.pt"
                if spec["id"] in keys:
                    completed[spec["id"]] = checked_candidate(layer_dir, spec["id"], fingerprint)
            weight = module.weight.detach()
            pilot_rows = min(args.pilot_rows, len(weight)) if args.mode == "profile" else len(weight)
            pilot_is_full_layer = pilot_rows == len(weight)
            if not pilot_is_full_layer:
                row_indices = torch.linspace(0, len(weight) - 1, pilot_rows, device=device).long()
                weight = weight[row_indices]
                completed = {}  # Partial rows must never masquerade as full-layer cache.
            deadline = min(operation_deadline or float("inf"), time.monotonic() + args.pilot_layer_seconds) if args.mode == "profile" else operation_deadline
            recorded = []
            def save_candidate(key, entry):
                recorded.append({k: v for k, v in entry.items() if k not in ("weight", "mask", "scale")})
                if args.mode == "run" or pilot_is_full_layer:
                    commit_candidate(layer_dir, key, entry, fingerprint)
            start = time.perf_counter()
            status = "completed"
            try:
                bank = build_candidates(weight, stats, specs, module.act_amax, damping=args.damping, deadline=deadline, completed=completed,
                                        on_candidate=save_candidate, block_size=args.processing_block_size)
            except CompressionTimeout:
                status = "time_cap"
                bank = None
            synchronize(device)
            elapsed = time.perf_counter() - start
            attempts.append({"layer": name, "mode": args.mode, "status": status,
                             "seconds": elapsed, "full_output_rows": pilot_is_full_layer,
                             "completed_at": now()})
            _atomic_json(attempts_path, attempts)
            if args.mode == "profile":
                sample_seconds = sum(r["seconds"] for r in recorded)
                # A resumed pilot must project the complete bank cost, not only
                # its last few candidates. Include prior full-row attempts.
                bank_seconds = sum(a["seconds"] for a in attempts if a["layer"] == name and a["full_output_rows"])
                pilot["layers"] = [r for r in pilot["layers"] if r["layer"] != name]
                pilot["layers"].append({"layer": name, "shape": list(module.weight.shape),
                    "sampled_output_rows": pilot_rows, "total_output_rows": len(module.weight),
                    "status": status, "statistics_seconds": stats["seconds"],
                    "statistics_input_rows": stats["input_rows"], "scoring_rows": stats["score_rows"],
                    "candidate_seconds": bank_seconds, "completed_candidates": recorded,
                    "full_bank_seconds_linear_estimate": sample_seconds * len(module.weight) / pilot_rows if bank is not None else None,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                    "sampled_candidates_reusable": pilot_is_full_layer})
                _atomic_json(pilot_path, pilot)
            elif status != "completed":
                _atomic_json(directory / "status.json", {"status": "preparation_time_cap", "next_layer": name})
                return
            del stats, completed, bank
        if args.mode == "profile":
            pilot["dense_final_evaluation_seconds"] = dense["seconds"]
            pilot["dense_refinement_evaluation_seconds"] = dense_refinement["seconds"]
            pilot["completed_at"] = now()
            _atomic_json(pilot_path, pilot)
            print(f"[SparseGPT] Pilot saved: {pilot_path}", flush=True)
            return pilot

        def load_candidate(name, key):
            idx = list(modules).index(name)
            return checked_candidate(directory / "cache" / f"layer_{idx:03d}", key, fingerprint)
        candidates = {}
        reused_statistics_seconds = 0.
        for name in modules:
            idx = list(modules).index(name)
            stats = load_tensors(directory / "cache" / f"layer_{idx:03d}" / "statistics.pt")
            reused_statistics_seconds += stats["seconds"]
            del stats
            candidates[name] = []
            for spec in specs:
                entry = load_candidate(name, spec["id"])
                candidates[name].append({k: v for k, v in entry.items() if k not in ("weight", "mask", "scale")})
    preparation_seconds = sum(r["seconds"] for r in json.loads((directory / "setup_attempts.json").read_text())) + sum(r["seconds"] for r in json.loads((directory / "preparation_wall_attempts.json").read_text()))
    search_settings = {"max_evaluations": args.max_evaluations, "budgets": args.budgets}
    search_path = directory / "search.json"
    search = json.loads(search_path.read_text()) if search_path.exists() else {
        "fingerprint": fingerprint, "settings": search_settings, "probes": [], "allocation_seconds": 0.0}
    if search["settings"] != search_settings or search["fingerprint"] != fingerprint:
        raise ValueError("Search settings changed; use a fresh directory")
    selection_path = directory / "frozen_selection.json"
    if selection_path.exists():
        frozen = json.loads(selection_path.read_text())
        if frozen["fingerprint"] != fingerprint or frozen["search_settings"] != search_settings:
            raise ValueError("Frozen selection identity/settings mismatch")
        winners = frozen["winners"]
        expected_winners = {str(b): best_feasible(search["probes"], b, dense_refinement["top1"]) for b in args.budgets}
        if winners != expected_winners:
            raise ValueError("Search cache differs from irrevocably frozen selection")
        # Crucial on resume: never generate/evaluate new proposals after any
        # compressed final-set evaluation could have occurred.
    else:
        proposals = iter(proposed_assignments(candidates, args.max_evaluations, search_deadline))
        finished_grid = False
        while True:
            started_allocation = time.perf_counter()
            try:
                assignment = next(proposals)
            except StopIteration:
                finished_grid = True
                break
            except CompressionTimeout:
                search["allocation_seconds"] += time.perf_counter() - started_allocation
                break
            search["allocation_seconds"] += time.perf_counter() - started_allocation
            key = digest(assignment)
            if any(p["key"] == key for p in search["probes"]):
                continue
            if search_deadline and time.monotonic() >= search_deadline:
                break
            start = time.perf_counter()
            apply_assignment(modules, assignment, load_candidate, ranges)
            try:
                accuracy = evaluate(model, calibration, device, None, output_indices,
                    prediction_path=directory / "predictions" / f"refinement_{key}.pt", deadline=search_deadline)
            except CompressionTimeout:
                search["incomplete_evaluation_seconds"] = search.get("incomplete_evaluation_seconds", 0.) + time.perf_counter() - start
                break
            energy = sum(next(c["normalized_energy"] for c in candidates[n] if c["configuration"] == k)
                         for n, k in assignment.items()) / len(modules)
            search["probes"].append({"key": key, "assignment": assignment, "accuracy": accuracy,
                                    "normalized_energy": energy, "seconds": time.perf_counter() - start})
            _atomic_json(search_path, search)
            print(f"[SparseGPT] search {len(search['probes'])}/{args.max_evaluations}: top1={accuracy['top1']:.5f}, proxy={energy:.6f}", flush=True)
        search["finished_resource_grid"] = finished_grid
        _atomic_json(search_path, search)
        if not search["probes"]:
            _atomic_json(directory / "status.json", {"status": "search_time_cap", "message": "No completed dense anchor; no comparison emitted"})
            return
        winners = {str(b): best_feasible(search["probes"], b, dense_refinement["top1"]) for b in args.budgets}
        frozen = {"fingerprint": fingerprint, "winners": winners, "search_settings": search_settings,
                  "selection_uses_final_labels": False, "frozen_at": now(),
                  "finished_resource_grid": finished_grid}
        _atomic_json(selection_path, frozen)
    result_path = directory / "results.json"
    results = json.loads(result_path.read_text()) if result_path.exists() else {
        "suite": "sparsegpt_comparison", "method": METHOD,
        "artifact_schema": 2, "algorithm_settings": settings, "frozen_selection_sha256": digest(frozen),
        "fingerprint": fingerprint, "reference_sha256": identity["reference_sha256"],
        "model": cfg["model"], "environment": environment(), "protocol": metadata["protocol"],
        "execution_settings": execution,
        "dense_baseline": dense, "dense_refinement": dense_refinement,
        "preparation_seconds": preparation_seconds,
        "reused_statistics_seconds": reused_statistics_seconds,
        "new_dense_evaluation_seconds": dense["seconds"] + dense_refinement["seconds"],
        "reuse_timing_note": "Preparation is newly executed work including the fresh dense gate. Reused statistics are separate; original activation calibration has no isolated timer.",
        "pilot_candidate_seconds_excluded": sum(a["seconds"] for a in attempts if a["mode"] == "profile" and not a["full_output_rows"]),
        "search_seconds": search["allocation_seconds"] + sum(p["seconds"] for p in search["probes"]) + search.get("incomplete_evaluation_seconds", 0.),
        "search_finished_resource_grid": search["finished_resource_grid"],
        "search_evaluations": len(search["probes"]), "experiments": [],
        "adaptations": metadata["adaptations"], "limitations": metadata["limitations"]}
    if results["fingerprint"] != fingerprint or results["frozen_selection_sha256"] != digest(frozen):
        raise ValueError("Result identity/frozen selection changed")
    results["preparation_seconds"] = preparation_seconds
    for budget in args.budgets:
        if any(e["budget_pp"] == budget for e in results["experiments"]):
            continue
        winner = winners[str(budget)]
        old = next((e for e in results["experiments"] if e["assignment_key"] == winner["key"]), None)
        apply_assignment(modules, winner["assignment"], load_candidate, ranges)
        try:
            accuracy = deepcopy(old["accuracy"]) if old else evaluate(model, evaluation, device, None, output_indices,
                prediction_path=directory / "predictions" / f"final_{winner['key']}.pt", deadline=final_deadline)
        except CompressionTimeout:
            _atomic_json(directory / "status.json", {"status": "final_evaluation_time_cap", "message": "Frozen selections retained; resume only missing final evaluations"})
            return
        selected_layers = []
        for name, key in winner["assignment"].items():
            candidate = next(c for c in candidates[name] if c["configuration"] == key)
            selected_layers.append({"target_layer": name, "shape": list(modules[name].weight.shape),
                                     "num_weights": modules[name].weight.numel(), **candidate})
        drop = dense["top1"] - accuracy["top1"]
        experiment = {"budget_pp": budget, "assignment_key": winner["key"], "accuracy": accuracy,
                      "final_evaluation_reused": old is not None,
                      "accuracy_drop_top1_pp": drop,
                      "refinement_drop_top1_pp": dense_refinement["top1"] - winner["accuracy"]["top1"],
                      "refinement_feasible": True, "final_feasible": drop <= budget + 1e-9,
                      "normalized_energy": winner["normalized_energy"], "selected_layers": selected_layers}
        # Durable final selected weights, masks and activation ranges for replay.
        checkpoint = {"fingerprint": fingerprint, "assignment": winner["assignment"],
                      "layers": {name: load_candidate(name, key) for name, key in winner["assignment"].items()},
                      "activation_ranges": ranges}
        checkpoint_path = directory / "selected" / f"budget_{budget:g}.pt"
        atomic_torch(checkpoint_path, checkpoint)
        experiment["checkpoint_file"] = str(checkpoint_path.relative_to(directory))
        experiment["checkpoint_sha256"] = file_sha256(checkpoint_path)
        results["experiments"].append(experiment)
        _atomic_json(result_path, results)
    results["completed_at"] = now()
    _atomic_json(result_path, results)
    from .audit import verify_run
    verify_run(directory)
    _atomic_json(directory / "status.json", {"status": "completed", "verification_passed": True})
    print(f"[SparseGPT] Results saved: {result_path}", flush=True)
    return results


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["baseline", "profile", "run"])
    p.add_argument("--config", required=True)
    p.add_argument("--reference", required=True, help="Completed SQuaRE global_refinement JSON")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data-root", help="Directory containing both ImageNetV2 variants")
    p.add_argument("--device", default="cuda")
    p.add_argument("--numerics", choices=["square-default", "strict-fp32"], default="square-default",
                   help="Match SQuaRE's PyTorch 2.6 backend defaults; strict-fp32 is an explicit alternative")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--obc-source", required=True, help="Completed v4-block256 model directory with full statistics")
    p.add_argument("--processing-block-size", type=int, default=128)
    p.add_argument("--search-reserve-seconds", type=float, default=300.)
    p.add_argument("--final-reserve-seconds", type=float, default=300.)
    p.add_argument("--damping", type=float, default=0.01)
    p.add_argument("--positions-per-example", type=int, default=32, help="0 uses every token/spatial position")
    p.add_argument("--score-rows", type=int, default=2048)
    p.add_argument("--pilot-rows", type=int, default=1000000)
    p.add_argument("--pilot-layer-seconds", type=float, default=120)
    p.add_argument("--max-evaluations", type=int, default=32)
    p.add_argument("--max-hours", type=float, help="Required for run; per invocation, covers setup/preparation/search/final evaluation, cooperatively")
    p.add_argument("--budgets", nargs="+", type=float, default=[0.1, 0.5, 1.0])
    p.add_argument("--baseline-tolerance", type=float, default=0.02)
    return p


def main():
    import math
    p = parser()
    args = p.parse_args()
    if args.mode == "run" and (args.max_hours is None or args.max_hours <= 0):
        p.error("run requires an explicitly agreed positive --max-hours; run profile first")
    numbers = [args.damping, args.baseline_tolerance, args.pilot_layer_seconds, args.search_reserve_seconds, args.final_reserve_seconds, *args.budgets]
    if args.max_hours is not None:
        numbers.append(args.max_hours)
    if not all(math.isfinite(n) for n in numbers) or args.damping <= 0 or args.baseline_tolerance < 0:
        p.error("Require finite positive damping and nonnegative baseline tolerance")
    if args.batch_size is not None and args.batch_size < 1 or args.workers is not None and args.workers < 0:
        p.error("Invalid data-loader settings")
    if min(args.score_rows, args.pilot_rows, args.pilot_layer_seconds) <= 0 or args.positions_per_example < 0:
        p.error("Invalid sampling/batching limits")
    if args.max_evaluations < 2 or not args.budgets or any(b < 0 for b in args.budgets) or len(set(args.budgets)) != len(args.budgets):
        p.error("Require >=2 evaluations and distinct nonnegative accuracy budgets")
    if args.processing_block_size < 8 or args.processing_block_size % 8:
        p.error("Processing block size must be a positive multiple of 8")
    if min(args.search_reserve_seconds, args.final_reserve_seconds) < 0:
        p.error("Time reserves must be nonnegative")
    if args.mode == "run" and args.max_hours * 3600 <= args.search_reserve_seconds + args.final_reserve_seconds:
        p.error("Run allowance must exceed reserved search/final time")
    run(args)


if __name__ == "__main__":
    main()
