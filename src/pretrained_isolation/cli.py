from __future__ import annotations

import argparse
from copy import deepcopy
import json

from .config import load_config
from .runner import run, run_joint, run_layerwise, run_refinement


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolate quantization and sparsity on pretrained models")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--suite",
        choices=["baseline", "quantization", "sparsity", "layerwise", "joint", "refinement", "all"],
        required=True,
    )
    parser.add_argument("--eval-dir", help="Override ImageNet validation ImageFolder path")
    parser.add_argument("--calib-dir", help="Override ImageNet training/subset ImageFolder path")
    parser.add_argument("--output", help="Output JSON path; only valid for a single suite")
    parser.add_argument("--max-eval-samples", type=int, help="Temporary smoke-test override")
    parser.add_argument("--calibration-samples", type=int, help="Temporary calibration-size override")
    parser.add_argument(
        "--refinement-samples", type=int,
        help="Override the labeled matched-frequency refinement split size",
    )
    parser.add_argument(
        "--refinement-seed", type=int,
        help="Override the deterministic matched-frequency refinement split seed",
    )
    parser.add_argument("--resume", action="store_true", help="Resume a layerwise, joint, or refinement output file")
    parser.add_argument("--layer-start", type=int, help="First selected layer index (inclusive)")
    parser.add_argument("--layer-end", type=int, help="Last selected layer index (exclusive)")
    parser.add_argument("--layerwise-results", help="Completed layerwise JSON used by joint/refinement suites")
    parser.add_argument(
        "--accuracy-thresholds", nargs="+", type=float, default=[0.1, 0.5, 1.0],
        help="Top-1 percentage-point thresholds for joint/refinement suites",
    )
    parser.add_argument(
        "--ignore-reference-tolerance", action="store_true",
        help="Record a dense reference mismatch but do not stop the sweep",
    )
    parser.add_argument(
        "--refinement-pairwise-top-k", type=int, default=6,
        help="When single repair moves stall, probe pairs among this many best singles (default: 6)",
    )
    parser.add_argument(
        "--refinement-max-accepted-moves", type=int,
        help="Optional cap on accepted repair/reclamation iterations",
    )
    parser.add_argument(
        "--no-refinement-block-fallback", action="store_true",
        help="Disable the simultaneous next-safer block probe used after single/pair stalls",
    )
    parser.add_argument(
        "--no-energy-reclamation", action="store_true",
        help="Stop after global feasibility repair instead of probing aggressive energy-saving moves",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.eval_dir:
        cfg["data"]["eval_dir"] = args.eval_dir
    if args.calib_dir:
        cfg["data"]["calib_dir"] = args.calib_dir
    if args.max_eval_samples is not None:
        cfg["data"]["max_eval_samples"] = args.max_eval_samples
    if args.calibration_samples is not None:
        cfg["data"]["calibration_samples"] = args.calibration_samples
    if args.refinement_samples is not None:
        cfg["data"].setdefault("refinement_split", {})["samples"] = args.refinement_samples
    if args.refinement_seed is not None:
        cfg["data"].setdefault("refinement_split", {})["seed"] = args.refinement_seed
    if args.ignore_reference_tolerance and cfg.get("reference"):
        cfg["reference"]["enforce_for_full_sweeps"] = False
    if args.suite in {"joint", "refinement"}:
        if not args.layerwise_results:
            parser.error(f"--layerwise-results is required with --suite {args.suite}")
        if args.suite == "joint":
            results = run_joint(
                cfg,
                args.layerwise_results,
                args.output,
                thresholds=args.accuracy_thresholds,
                resume=args.resume,
            )
        else:
            results = run_refinement(
                cfg,
                args.layerwise_results,
                args.output,
                thresholds=args.accuracy_thresholds,
                resume=args.resume,
                pairwise_top_k=args.refinement_pairwise_top_k,
                enable_block_fallback=not args.no_refinement_block_fallback,
                enable_energy_reclamation=not args.no_energy_reclamation,
                max_accepted_moves=args.refinement_max_accepted_moves,
            )
    elif args.suite == "layerwise":
        results = run_layerwise(
            cfg,
            args.output,
            resume=args.resume,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
        )
    elif args.suite == "all":
        if args.output:
            parser.error("--output cannot be used with --suite all")
        results = {
            "quantization": run(deepcopy(cfg), "quantization"),
            "sparsity": run(deepcopy(cfg), "sparsity"),
        }
    else:
        results = run(cfg, args.suite, args.output)
    summary = {"status": "completed", "suite": args.suite, "experiments": (
        {key: len(value["experiments"]) for key, value in results.items()}
        if args.suite == "all" else len(results["experiments"])
    )}
    if args.suite == "baseline":
        summary["dense_baseline"] = results["dense_baseline"]
        summary["dense_reference_check"] = results["dense_reference_check"]
        summary["output_file"] = results["output_file"]
    elif args.suite == "layerwise":
        summary["output_file"] = results["output_file"]
        summary["layers"] = len(results["layer_scope"]["layers_in_this_file"])
    elif args.suite in {"joint", "refinement"}:
        summary["output_file"] = results["output_file"]
        summary["threshold_results"] = [
            {
                "threshold_top1_pp": item["accuracy_threshold_top1_pp"],
                "top1": item.get("accuracy", {}).get("top1"),
                "top1_drop_pp": item.get("accuracy_drop_top1_pp"),
                "refinement_top1_drop_pp": item.get("refinement_accuracy_drop_top1_pp"),
                "refinement_constraint_satisfied": item.get("constraint_satisfied_on_refinement_set"),
                "search_evaluations": item.get("search_evaluation_count"),
                "status": item["status"],
            }
            for item in results["experiments"]
        ]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
