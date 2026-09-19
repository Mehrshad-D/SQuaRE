import json
from datetime import datetime, timedelta, timezone
import sys
import time

import pytest
import torch
from pretrained_isolation.obc.blockwise import prepare_blocks, blockwise_prune, blockwise_quantize
from pretrained_isolation.obc.core import damp_hessian, obs_prune, obs_quantize, CompressionTimeout
from pretrained_isolation.obc.runtime import build_candidates
from pretrained_isolation.config import load_config, layerwise_configurations
from pathlib import Path


@pytest.mark.parametrize("n,m", [(2, 4), (4, 8), (3, 8)])
def test_block_pruning_matches_independent_kkt_and_explicit_diagonal(n, m):
    torch.manual_seed(311)
    x = torch.randn(100, 40, dtype=torch.float64)
    gram = x.T @ x / len(x)
    weight = torch.randn(5, 40, dtype=torch.float64)
    blocks = prepare_blocks(gram, 16, .01)  # Includes a partial final block.
    result, mask = blockwise_prune(weight, blocks, n, m, row_batch=3)
    expected, expected_mask = obs_prune(weight, torch.block_diag(*(b[2] for b in blocks)), n, m)
    assert torch.equal(mask, expected_mask)
    torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)
    for start, stop, h in blocks:
        # Check compensation using a direct solve, independently of inverse updates.
        for row in range(len(weight)):
            active = torch.where(mask[row, start:stop])[0]
            removed = torch.where(~mask[row, start:stop])[0]
            w = weight[row, start:stop]
            optimum = w[active] + torch.linalg.solve(h[active[:, None], active], h[active[:, None], removed] @ w[removed])
            torch.testing.assert_close(result[row, start:stop][active], optimum, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("bits", [4, 6, 8])
def test_block_quantization_uses_global_channel_scales_and_batch_invariance(bits):
    torch.manual_seed(200)
    w = torch.randn(5, 24, dtype=torch.float64)
    w[:, :8] *= .001
    w[:, 16:] *= 20
    blocks = prepare_blocks(torch.eye(24, dtype=torch.float64), 8, .01)
    a, scale = blockwise_quantize(w, blocks, bits, row_batch=1)
    b, _ = blockwise_quantize(w, blocks, bits, row_batch=4)
    expected_scale = w.abs().amax(1, keepdim=True) / (2 ** (bits - 1) - 1)
    torch.testing.assert_close(scale, expected_scale)
    torch.testing.assert_close(a, (w / expected_scale).round() * expected_scale)
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def test_damping_is_full_layer_not_per_block():
    gram = torch.diag(torch.cat((torch.ones(8), torch.ones(8) * 100))).double()
    blocks = prepare_blocks(gram, 8, .01)
    for start, stop, h in blocks:
        torch.testing.assert_close(h, gram[start:stop, start:stop] + torch.eye(8) * .505, atol=1e-7, rtol=1e-7)


def test_full_size_block_reproduces_all_exact_candidates():
    torch.manual_seed(22)
    w = torch.randn(4, 16)
    x = torch.randn(32, 16)
    stats = {"gram": x.double().T @ x.double() / len(x), "score_inputs": x}
    root = Path(__file__).resolve().parents[1]
    specs = layerwise_configurations(load_config(root / "configs/deit_tiny_imagenetv2.yaml"))
    exact = build_candidates(w, stats, specs, x.abs().max(), row_batch=1)
    blocked = build_candidates(w, stats, specs, x.abs().max(), row_batch=4, block_size=32)
    for key in exact:
        assert torch.equal(exact[key]["mask"], blocked[key]["mask"])
        torch.testing.assert_close(exact[key]["weight"], blocked[key]["weight"], atol=1e-6, rtol=1e-6)
        assert exact[key]["score"] == pytest.approx(blocked[key]["score"], abs=1e-7)


def test_blocks_reject_misaligned_groups_and_expired_work():
    with pytest.raises(ValueError, match="multiple of 8"):
        prepare_blocks(torch.eye(16), 12, .01)
    blocks = prepare_blocks(torch.eye(20), 16, .01)
    with pytest.raises(ValueError, match="N:M"):
        blockwise_prune(torch.ones(2, 20), blocks, 3, 8)
    with pytest.raises(CompressionTimeout):
        blockwise_prune(torch.ones(2, 16), prepare_blocks(torch.eye(16), 8, .01), 2, 4, deadline=time.monotonic() - 1)


def test_prediction_evaluator_agrees_with_square_and_saved_labels(tmp_path):
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from pretrained_isolation.engine import evaluate as square_evaluate
    from pretrained_isolation.obc.evaluation import evaluate, prediction_metrics
    torch.manual_seed(9)
    model = nn.Linear(16, 8)
    loader = DataLoader(TensorDataset(torch.randn(15, 16), torch.randint(0, 6, (15,))), batch_size=4)
    indices = [7, 6, 5, 4, 3, 2]
    actual = evaluate(model, loader, torch.device("cpu"), 13, indices, prediction_path=tmp_path / "predictions/test.pt")
    expected = square_evaluate(model, loader, torch.device("cpu"), 13, indices)
    for k in ("top1", "top5", "evaluated_samples"):
        assert actual[k] == expected[k]
    saved = torch.load(tmp_path / "predictions/test.pt", weights_only=True)
    assert saved["labels"].tolist() == loader.dataset.tensors[1][:13].tolist()
    assert prediction_metrics(saved)["top1"] == actual["top1"]


def test_study_resume_charges_previous_and_interrupted_time():
    from pretrained_isolation.obc.study import remaining_seconds, recover_unfinished_attempt
    ledger = {"total_seconds": 15 * 3600, "attempts": [
        {"charged_seconds": 4 * 3600},
        {"started_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()}]}
    recover_unfinished_attempt(ledger)
    assert remaining_seconds(ledger) == pytest.approx(9 * 3600, abs=2)
    before = json.dumps(ledger)
    recover_unfinished_attempt(ledger)
    assert json.dumps(ledger) == before


def test_subprocess_time_limit_terminates_own_process_group(tmp_path):
    from pretrained_isolation.obc.study import run_child
    from pretrained_isolation.obc.cli import exclusive_output
    start = time.monotonic()
    with exclusive_output(tmp_path) as fd:
        code, status = run_child([sys.executable, "-c", "import time; time.sleep(30)"], .3, tmp_path / "job.log", fd)
    assert code != 0 and status == "time_cap"
    assert time.monotonic() - start < 3


def test_selfcheck_cpu():
    from pretrained_isolation.obc.selfcheck import selfcheck
    assert selfcheck()["passed"]


def test_study_budget_covers_all_models_and_does_not_reset(tmp_path, monkeypatch):
    from argparse import Namespace
    from types import SimpleNamespace
    from pretrained_isolation.obc import study as module
    data = tmp_path / "data"
    for folder in ("matched", "threshold"):
        (data / folder).mkdir(parents=True)
    refs = tmp_path / "refs"
    for model in module.MODELS:
        folder = refs / model
        folder.mkdir(parents=True)
        (folder / f"{model}_imagenetv2_global_refinement.json").write_text(json.dumps({
            "completed_at_utc": "fixture", "experiments": [{}, {}, {}],
            "data": {"eval_dir": "matched", "calib_dir": "threshold"}}))
    clock = [0.]
    launches = []
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    def child(command, timeout, log_path, lock_fd):
        launches.append(command)
        if "pretrained_isolation.obc.selfcheck" in command:
            Path(command[command.index("--output") + 1]).write_text('{"passed": true}')
            clock[0] += 1
        elif "baseline" in command:
            out = Path(command[command.index("--output-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "dense_checks.json").write_text('{"passed": true}')
            clock[0] += 10
        else:
            clock[0] += timeout  # A full allowance is consumed without a completed bank.
        return 0, "exited"
    monkeypatch.setattr(module, "run_child", child)
    args = Namespace(data_root=str(data), reference_root=str(refs), output_root=str(tmp_path / "outputs"),
                     total_hours=15., block_size=256, row_batch=32, device="cpu")
    first = module.study(args)
    assert first["status"] == "incomplete"
    assert sum(a["charged_seconds"] for a in first["attempts"]) == 15 * 3600
    primary = [a["model"] for a in first["attempts"] if a["stage"] == "run"]
    assert primary == list(module.MODELS)
    count = len(launches)
    second = module.study(args)
    assert len(launches) == count and second["remaining_seconds"] == 0
    args.total_hours = 30
    with pytest.raises(ValueError, match="identity/budget changed"):
        module.study(args)
    args.total_hours = 15
    args.extend_total_hours = 16
    extended = module.study(args)
    assert len(extended["explicit_budget_extensions"]) == 1
    assert extended["total_seconds"] == 16 * 3600
    assert extended["remaining_seconds"] == 0
    count = len(launches)
    assert module.study(args)["remaining_seconds"] == 0
    assert len(launches) == count
