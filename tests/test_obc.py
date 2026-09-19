from copy import deepcopy
import itertools
import json
from pathlib import Path
import time

import pytest
import torch
from torch import nn

from pretrained_isolation.obc.core import (
    CompressionTimeout, allocate_dp, damp_hessian, obs_prune, obs_quantize,
)
from pretrained_isolation.obc.protocol import matched_config, check_protocol, check_accuracy
from pretrained_isolation.obc.runtime import input_rows, build_candidates, apply_assignment
from pretrained_isolation.config import load_config, layerwise_configurations
from pretrained_isolation.modules import IsolatedLinear, IsolatedConv2d

ROOT = Path(__file__).resolve().parents[1]


def spd(n):
    torch.manual_seed(91)
    x = torch.randn(n * 3, n, dtype=torch.double)
    return damp_hessian(x.T @ x, 0.03)


@pytest.mark.parametrize("n,m", [(2, 4), (3, 8), (4, 8)])
def test_prune_matches_independent_constrained_least_squares(n, m):
    torch.manual_seed(4)
    w = torch.randn(3, 16, dtype=torch.double)
    h = spd(16)
    result, mask = obs_prune(w, h, n, m, row_batch=2)
    assert torch.all(mask.reshape(3, -1, m).sum(-1) == n)
    assert torch.all(result[~mask] == 0)
    # Independent KKT solution after the greedy algorithm chooses its support.
    for row in range(len(w)):
        active, fixed = torch.where(mask[row])[0], torch.where(~mask[row])[0]
        expected = w[row, active] + torch.linalg.solve(h[active[:, None], active], h[active[:, None], fixed] @ w[row, fixed])
        torch.testing.assert_close(result[row, active], expected, atol=1e-9, rtol=1e-9)
        original_error = (w[row] * mask[row] - w[row])
        actual_error = result[row] - w[row]
        assert actual_error @ h @ actual_error <= original_error @ h @ original_error + 1e-9


def test_diagonal_hessian_pruning_scores_and_n_semantics():
    w = torch.tensor([[1., 2., 3., 4., 5., 6., 7., 8.]], dtype=torch.double)
    h = torch.diag(torch.tensor([100., 1., 1., 1., 1., 1., 1., 1.], dtype=torch.double))
    result, mask = obs_prune(w, h, 3, 8)
    assert mask.tolist() == [[True, False, False, False, False, False, True, True]]
    torch.testing.assert_close(result, w * mask)


@pytest.mark.parametrize("bits", [4, 6, 8])
def test_quantization_with_identity_hessian_matches_grid(bits):
    torch.manual_seed(5)
    w = torch.randn(4, 16)
    q, scales = obs_quantize(w, torch.eye(16, dtype=torch.double), bits, row_batch=3)
    qmax = 2 ** (bits - 1) - 1
    expected = (w / scales).round().clamp(-qmax, qmax) * scales
    torch.testing.assert_close(q, expected)


def test_sparse_quantization_preserves_mask_and_batching():
    torch.manual_seed(7)
    w = torch.randn(4, 16, dtype=torch.double)
    h = spd(16)
    pruned, mask = obs_prune(w, h, 3, 8)
    first, scale = obs_quantize(pruned, h, 4, mask=mask, row_batch=1)
    batched, _ = obs_quantize(pruned, h, 4, mask=mask, row_batch=3)
    torch.testing.assert_close(first, batched, atol=1e-10, rtol=1e-10)
    assert torch.all(first[~mask] == 0)
    torch.testing.assert_close(first / scale, (first / scale).round())
    assert (first / scale).abs().max() <= 7 + 1e-10


def test_quantization_matches_scalar_recomputed_inverse_oracle():
    w = torch.tensor([[.2, -.51, .83, -.76, .12, 1.03, -.33, .6]], dtype=torch.double)
    h = spd(8)
    actual, _ = obs_quantize(w, h, 4)
    original = w[0].clone()
    current = original.clone()
    scale = original.abs().max() / 7
    fixed, values = [], {}
    for _ in range(8):
        active = [i for i in range(8) if i not in fixed]
        if fixed:
            ix = torch.tensor(active); fx = torch.tensor(fixed)
            displacement = torch.tensor([values[i] for i in fixed]) - original[fx]
            current[ix] = original[ix] - torch.linalg.solve(h[ix[:, None], ix], h[ix[:, None], fx] @ displacement)
        ix = torch.tensor(active)
        inv = torch.linalg.inv(h[ix[:, None], ix])
        q = (current[ix] / scale).round().clamp(-7, 7) * scale
        errors = (current[ix] - q).square()
        j = int((errors / inv.diag()).argmin())
        if (errors > .25 * scale.square()).any() and current[ix[j]] != 0:
            j = int(errors.argmax())
        coordinate = active[j]
        fixed.append(coordinate)
        values[coordinate] = q[j].item()
    expected = torch.tensor([[values[i] for i in range(8)]], dtype=torch.double)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_deadline_and_invalid_inputs_fail_closed():
    with pytest.raises(CompressionTimeout):
        obs_prune(torch.ones(1, 8), torch.eye(8), 3, 8, deadline=time.monotonic() - 1)
    with pytest.raises(ValueError):
        obs_prune(torch.ones(1, 7), torch.eye(7), 3, 8)
    with pytest.raises(ValueError):
        damp_hessian(torch.zeros(8, 8))


def test_dp_matches_exhaustive_enumeration():
    candidates = [[{"normalized_energy": c, "score": s} for c, s in row] for row in
                  [[(1., 0), (.25, .8), (.09375, 1.7)], [(1., 0), (.1875, .4), (.046875, 2.)],
                   [(1., 0), (.5, .01), (.125, .9)]]]
    for cap in [.4, .8, 1.2, 2., 3.]:
        options = [p for p in itertools.product(range(3), repeat=3)
                   if sum(candidates[i][j]["normalized_energy"] for i, j in enumerate(p)) <= cap]
        winner = allocate_dp(candidates, cap)
        actual = sum(candidates[i][j]["score"] for i, j in enumerate(winner))
        assert actual == pytest.approx(min(sum(candidates[i][j]["score"] for i, j in enumerate(p)) for p in options))


def test_conv_flattening_agrees_with_pytorch_convolution():
    torch.manual_seed(10)
    conv = nn.Conv2d(2, 3, 3, padding=1, bias=False)
    module = IsolatedConv2d(conv, 32, 32)
    inputs = torch.randn(2, 2, 4, 4)
    rows = list(input_rows(module, inputs, 0))
    outputs = torch.cat([torch.nn.functional.linear(r, conv.weight.flatten(1)) for r in rows])
    expected = conv(inputs).permute(0, 2, 3, 1).reshape(-1, 3)
    torch.testing.assert_close(outputs, expected)


def test_candidate_bank_matches_joint_weight_activation_reconstruction():
    torch.manual_seed(21)
    layer = nn.Linear(8, 2)
    x = torch.randn(32, 8)
    specs = layerwise_configurations(load_config(ROOT / "configs/deit_tiny_imagenetv2.yaml"))
    stats = {"gram": x.double().T @ x.double() / len(x), "score_inputs": x}
    bank = build_candidates(layer.weight, stats, specs, x.abs().max())
    assert len(bank) == 16
    assert bank["FP32__dense"]["score"] == 0
    module = IsolatedLinear(layer, 32, 32)
    entry = bank["INT4__3to8"]
    apply_assignment({"proj": module}, {"proj": "INT4__3to8"}, lambda n, k: bank[k], {"proj": x.abs().max()})
    error = (module(x) - layer(x)).double().square().sum()
    signal = torch.nn.functional.linear(x, layer.weight).double().square().sum()
    assert entry["score"] == pytest.approx((error / signal).item(), abs=1e-7)
    torch.testing.assert_close(module.effective_weight.cpu(), entry["weight"])


def legacy_reference():
    return {"suite": "global_refinement", "model": {"name": "toy"}, "selection": {"include": ["proj"]},
        "seed": 42, "data": {"max_eval_samples": None, "resolved_model_data_config": {},
        "label_space": {"dataset": "ImageNetV2"}}, "refinement_set": {"samples": 256},
        "dense_baseline": {"evaluated_samples": 10000}}


def test_protocol_reconstructs_v4_instead_of_using_current_v5_config():
    ref = legacy_reference()
    cfg = {"model": ref["model"], "selection": ref["selection"], "data": {"refinement_split": {"source": "eval_dir"}}}
    result = matched_config(cfg, ref)
    assert "refinement_split" not in result["data"]
    assert result["data"]["calibration_samples"] == 256
    assert "refinement_split" in cfg["data"]  # no in-place mutations


def test_v5_split_hash_and_dense_mismatches_rejected():
    ref = legacy_reference()
    ref["refinement_set"]["split"] = {"refinement_indices_sha256": "abc"}
    with pytest.raises(ValueError, match="fingerprint"):
        check_protocol(ref, {}, {"refinement_indices_sha256": "wrong"}, 256, 10000, {})
    with pytest.raises(ValueError, match="top1"):
        check_accuracy({"evaluated_samples": 100, "top1": 65, "top5": 90},
                       {"evaluated_samples": 100, "top1": 66, "top5": 90}, .02)


def test_preprocessing_accepts_json_list_equivalent_to_live_tuple():
    actual = {"input_size": (3, 224, 224), "mean": (.485, .456, .406),
              "std": (.229, .224, .225), "interpolation": "bicubic",
              "crop_pct": .9, "crop_mode": "center"}
    reference = legacy_reference()
    reference["data"]["resolved_model_data_config"] = json.loads(json.dumps(actual))
    assert actual != reference["data"]["resolved_model_data_config"]
    before = deepcopy(actual)
    check_protocol(reference, actual, {}, 256, 10000, {})
    assert actual == before


@pytest.mark.parametrize("field,value", [
    ("input_size", (3, 256, 256)), ("mean", (.5, .456, .406)),
    ("std", (.2, .224, .225)), ("interpolation", "bilinear"),
    ("crop_pct", .95), ("crop_mode", "squash"),
])
def test_preprocessing_rejects_real_value_changes_with_diagnostics(field, value):
    expected = {"input_size": [3, 224, 224], "mean": [.485, .456, .406],
                "std": [.229, .224, .225], "interpolation": "bicubic",
                "crop_pct": .9, "crop_mode": "center"}
    reference = legacy_reference()
    reference["data"]["resolved_model_data_config"] = expected
    actual = {**expected, field: value}
    with pytest.raises(ValueError, match=field) as error:
        check_protocol(reference, actual, {}, 256, 10000, {})
    assert '"actual"' in str(error.value) and '"expected"' in str(error.value)


@pytest.mark.parametrize("extra", [False, True])
def test_preprocessing_rejects_missing_or_added_fields(extra):
    reference = legacy_reference()
    reference["data"]["resolved_model_data_config"] = {"crop_pct": .9}
    actual = {"crop_pct": .9, "crop_mode": "center"} if extra else {}
    with pytest.raises(ValueError, match="<missing>"):
        check_protocol(reference, actual, {}, 256, 10000, {})


def test_numerical_modes_set_explicit_flags_and_change_identity():
    from pretrained_isolation.obc.numerics import configure_execution
    from pretrained_isolation.obc.protocol import digest
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
           torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
           torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        strict = configure_execution("strict-fp32")
        assert not strict["cuda_matmul_allow_tf32"] and not strict["cudnn_allow_tf32"]
        matched = configure_execution("square-default")
        assert not matched["cuda_matmul_allow_tf32"] and matched["cudnn_allow_tf32"]
        assert not matched["cudnn_benchmark"] and not matched["cudnn_deterministic"]
        assert digest(strict) != digest(matched)
        with pytest.raises(ValueError, match="Unknown"):
            configure_execution("choose-the-best-accuracy")
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[:2]
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = old[2:4]
        torch.use_deterministic_algorithms(old[4], warn_only=old[5])


def test_failed_dense_check_preserves_both_metrics_and_both_splits(tmp_path):
    pytest.importorskip("timm")
    from pretrained_isolation.obc.cli import save_dense_checks
    expected = {"top1": 59.38, "top5": 80.93, "evaluated_samples": 10000, "seconds": 1.}
    refinement = {"top1": 64.84375, "top5": 88.28125, "evaluated_samples": 256, "seconds": .1}
    reference = {"dense_baseline": expected, "refinement_set": {"dense_accuracy": refinement}}
    actual = {**expected, "top1": 59.35}
    path = tmp_path / "dense_checks.json"
    with pytest.raises(ValueError, match="Diagnostics saved"):
        save_dense_checks(path, "fingerprint", actual, refinement, reference, .02, {"cudnn_allow_tf32": True})
    report = json.loads(path.read_text())
    assert not report["passed"] and report["checks"]["refinement"]["passed"]
    assert report["checks"]["final"]["actual_minus_expected_pp"]["top1"] == pytest.approx(-.03)
    assert report["checks"]["final"]["actual_minus_expected_pp"]["top5"] == 0
    assert save_dense_checks(path, "fingerprint", expected, refinement, reference, .02, {})["passed"]


@pytest.mark.parametrize("model,expected", [("deit_tiny", 48), ("swin_tiny", 48), ("resnet18", 19)])
def test_real_architecture_adapters_preserve_dense_predictions(model, expected):
    timm = pytest.importorskip("timm")
    from pretrained_isolation.engine import instrument
    cfg = load_config(ROOT / f"configs/{model}_imagenetv2.yaml")
    torch.manual_seed(8)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        net = timm.create_model(cfg["model"]["name"], pretrained=False).eval()
        # Exercise the real timm -> saved-JSON protocol boundary. The original
        # toy workflow's empty data config did not cover tuple/list serialization.
        actual_config = timm.data.resolve_model_data_config(net)
        path = ROOT / f"outputs-v4/imagenetv2/{model}/{model}_imagenetv2_global_refinement.json"
        reference = json.loads(path.read_text())
        check_protocol(reference, actual_config, {}, reference["refinement_set"]["samples"],
                       reference["dense_baseline"]["evaluated_samples"], reference["data"]["label_space"])
        images = torch.randn(1, 3, 224, 224)
        with torch.inference_mode():
            original = net(images)
            modules = instrument(net, cfg, 32, 32)
            assert len(modules) == expected
            for module in modules.values():
                module.finalize()
            actual = net(images)
        torch.testing.assert_close(actual, original, atol=1e-6, rtol=1e-6)
    finally:
        torch.set_num_threads(old_threads)


def test_full_cpu_workflow_freeze_resume_and_compare(tmp_path, monkeypatch):
    pytest.importorskip("timm")
    from torch.utils.data import DataLoader, TensorDataset
    from pretrained_isolation.obc import cli
    from pretrained_isolation.obc.compare import compare
    from pretrained_isolation.engine import evaluate
    torch.manual_seed(22)
    model = nn.Sequential()
    model.add_module("proj", nn.Linear(8, 6))
    refinement = DataLoader(TensorDataset(torch.randn(12, 8), torch.randint(0, 6, (12,))), batch_size=4)
    final = DataLoader(TensorDataset(torch.randn(12, 8), torch.randint(0, 6, (12,))), batch_size=4)
    device = torch.device("cpu")
    dense = evaluate(model, final, device)
    dense_ref = evaluate(model, refinement, device)
    cfg = load_config(ROOT / "configs/deit_tiny_imagenetv2.yaml")
    cfg["model"] = {"name": "toy", "short_name": "toy"}
    cfg["selection"] = {"module_types": ["linear"], "include": ["proj"]}
    cfg["data"]["batch_size"] = 4
    reference = legacy_reference()
    reference.update({"selection": cfg["selection"], "dense_baseline": dense, "selected_layer_count": 1,
        "pareto_frontiers": {"proj": []}, "environment": {"gpu": None}, "selection_source": {"sha256": "unknown"},
        "completed_at_utc": "test-fixture"})
    reference["refinement_set"] = {"samples": 12, "dense_accuracy": dense_ref}
    dense_spec = next(s for s in layerwise_configurations(cfg) if s["id"] == "FP32__dense")
    reference["experiments"] = [{"status": "succeeded", "accuracy_threshold_top1_pp": b,
        "accuracy": dense, "accuracy_drop_top1_pp": 0., "refinement_accuracy_drop_top1_pp": 0.,
        "energy_proxy": {"normalized_energy_vs_selected_layer_dense_baseline": 1.},
        "selected_layers": [{"target_layer": "proj", "configuration": "FP32__dense", "specification": dense_spec,
                             "normalized_energy": 1.}], "total_seconds": 1., "search_evaluation_count": 1} for b in [.1, .5, 1.]]
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference))
    monkeypatch.setattr(cli, "load_config", lambda _: deepcopy(cfg))
    monkeypatch.setattr(cli, "create_pretrained", lambda _: deepcopy(model))
    monkeypatch.setattr(cli, "make_refinement_loaders", lambda *a: (refinement, final, {}, 12, 12, None,
        {"dataset": "ImageNetV2"}, {"request": {"source": "calib_dir"}}))
    monkeypatch.setattr(cli, "dataset_manifest", lambda ds: [{"sha256": "ref" if ds is refinement.dataset else "final"}])
    events = []
    real_eval = cli.evaluate
    def checked_eval(net, loader, *a):
        if loader is final and events:
            assert (tmp_path / "obc" / "frozen_selection.json").exists()
        events.append(loader)
        return real_eval(net, loader, *a)
    monkeypatch.setattr(cli, "evaluate", checked_eval)
    args = cli.parser().parse_args(["run", "--config", "unused", "--reference", str(reference_path),
        "--output-dir", str(tmp_path / "obc"), "--device", "cpu", "--max-hours", "1", "--max-evaluations", "6"])
    args.mode = "baseline"
    checks = cli.run(args)
    assert checks["passed"]
    assert not (tmp_path / "obc/cache").exists()
    args.mode = "profile"
    args.resume = True
    profile = cli.run(args)
    assert profile["layers"][0]["sampled_output_rows"] == 2
    assert not (tmp_path / "obc/cache/layer_000/FP32__dense.pt").exists()
    args.mode = "run"
    args.resume = True
    result = cli.run(args)
    assert len(result["experiments"]) == 3
    assert result["search_evaluations"] <= 6
    args.resume = True
    events.clear()
    resumed = cli.run(args)
    assert len(resumed["experiments"]) == 3
    assert events == []
    args.numerics = "strict-fp32"
    with pytest.raises(ValueError, match="Cache identity changed"):
        cli.run(args)
    args.numerics = "square-default"
    rows = compare([reference_path], [tmp_path / "obc/results.json"], tmp_path / "comparison")
    assert len(rows) == 6
    assert (tmp_path / "comparison/comparison_table.tex").exists()
    reference["seed"] = 17
    reference_path.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="different reference"):
        compare([reference_path], [tmp_path / "obc/results.json"], tmp_path / "wrong")
