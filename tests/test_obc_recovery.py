"""Recovery orchestration only; compression mathematics remain the tested v0.9.0 code."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def recovery():
    path = Path(__file__).resolve().parents[1] / "scripts/resume_obc_v090.py"
    spec = importlib.util.spec_from_file_location("obc_recovery_test_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def study_fixture(tmp_path, monkeypatch, recovery):
    h = recovery
    root = tmp_path / "package"
    output = root / "outputs-obc/v4-block256"
    output.mkdir(parents=True)
    refs = []
    for name in h.MODELS:
        ref = root / f"{name}-reference.json"
        ref.write_text(json.dumps({"name": name}))
        refs.append(ref)
        directory = output / name
        directory.mkdir()
        (directory / "manifest.json").write_text(json.dumps({"reference_file": str(ref)}))
        if name != "swin_tiny":
            (directory / "results.json").write_text('{"completed_at": "already complete"}')
    cache = output / "swin_tiny/cache"
    cache.mkdir()
    (cache / "existing-candidate.pt").write_bytes(b"unchanged cached work")
    command = ["original-python", "-m", "pretrained_isolation.obc.cli", "run",
        "--config", str(root / "swin.yaml"), "--reference", str(refs[1]),
        "--output-dir", str(output / "swin_tiny"), "--data-root", str(root / "data"),
        "--device", "cuda", "--resume", "--hessian-block-size", "256", "--row-batch", "32",
        "--numerics", "square-default", "--max-hours", "5"]
    identity = {"source_sha256": h.source_fingerprint(), "device": "cuda", "data_root": str(root / "data"),
                "hessian_block_size": 256, "row_batch": 32, "numerics": "square-default",
                "max_evaluations": 32, "budgets": [.1, .5, 1.], "damping": .01,
                "score_rows": 2048, "positions_per_example": 32,
                "references": [h.file_sha256(p) for p in refs]}
    ledger = {"identity": identity, "fingerprint": h.digest(identity), "total_seconds": 15 * 3600,
              "status": "incomplete", "completed_models": ["deit_tiny", "resnet18"],
              "attempts": [{"model": "swin_tiny", "stage": "run", "charged_seconds": 8.142 * 3600,
                            "returncode": 1, "status": "exited", "command": command}]}
    (output / "study.json").write_text(json.dumps(ledger))
    clock = [100.]
    launches, comparisons = [], []
    monkeypatch.setenv("PYTHONPATH", "")
    monkeypatch.setattr(h, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(h, "verify_package", lambda root: None)
    monkeypatch.setattr(h, "verify_run", lambda directory: {"passed": True})
    monkeypatch.setattr(h, "available_gpu", lambda: {"free_mib": 23000, "other_compute_pids": []})
    monkeypatch.setattr(h, "compare", lambda *args: comparisons.append(args))
    def child(command, timeout, log_path, lock_fd):
        launches.append((command, timeout, log_path))
        clock[0] += 900.
        (output / "swin_tiny/results.json").write_text('{"completed_at": "recovered"}')
        return 0, "exited"
    monkeypatch.setattr(h, "run_child", child)
    return SimpleNamespace(root=root, output=output, ledger=ledger, launches=launches,
                           comparisons=comparisons, clock=clock)


def test_only_swin_retried_with_original_settings_and_remaining_budget(recovery, study_fixture):
    h, f = recovery, study_fixture
    before = {name: (f.output / name / "results.json").read_bytes() for name in ("deit_tiny", "resnet18")}
    result = h.recover(f.output, root=f.root)
    assert result["status"] == "completed" and result["completed_models"] == list(h.MODELS)
    assert len(f.launches) == 1
    command, allowance, log = f.launches[0]
    assert allowance == pytest.approx((15 - 8.142) * 3600)
    assert float(command[command.index("--max-hours") + 1]) == pytest.approx((allowance - 20) / 3600)
    assert command[command.index("--hessian-block-size") + 1] == "256"
    assert command[command.index("--row-batch") + 1] == "32"
    assert command[command.index("--output-dir") + 1] == str(f.output / "swin_tiny")
    assert result["attempts"][0] == f.ledger["attempts"][0]  # Preserve failed history.
    assert result["attempts"][-1]["charged_seconds"] == 900
    assert result["attempts"][-1]["explicit_recovery"]
    assert result["remaining_seconds"] == pytest.approx((15 - 8.142) * 3600 - 900)
    assert len(f.comparisons[0][0]) == 3 and f.comparisons[0][2].name == "comparison"
    for name, content in before.items():
        assert (f.output / name / "results.json").read_bytes() == content
    assert (f.output / "swin_tiny/cache/existing-candidate.pt").read_bytes() == b"unchanged cached work"
    h.recover(f.output, root=f.root)
    assert len(f.launches) == 1  # A second invocation does not repeat completed GPU work.


def test_busy_gpu_does_not_launch_or_charge_retry(recovery, study_fixture, monkeypatch):
    def busy():
        raise RuntimeError("GPU still occupied")
    monkeypatch.setattr(recovery, "available_gpu", busy)
    with pytest.raises(RuntimeError, match="occupied"):
        recovery.recover(study_fixture.output, root=study_fixture.root)
    saved = json.loads((study_fixture.output / "study.json").read_text())
    assert len(saved["attempts"]) == 1 and not study_fixture.launches


def test_retry_failure_preserves_partial_results_and_records_cost(recovery, study_fixture, monkeypatch):
    def fail(*args):
        study_fixture.clock[0] += 30
        return 1, "exited"
    monkeypatch.setattr(recovery, "run_child", fail)
    saved = recovery.recover(study_fixture.output, root=study_fixture.root)
    assert saved["status"] == "incomplete" and saved["completed_models"] == ["deit_tiny", "resnet18"]
    assert saved["attempts"][-1]["charged_seconds"] == 30
    assert saved["attempts"][-1]["returncode"] == 1
    assert study_fixture.comparisons[0][2].name == "comparison_partial"


def test_exhausted_budget_does_not_reset_or_launch(recovery, study_fixture):
    path = study_fixture.output / "study.json"
    saved = json.loads(path.read_text())
    saved["attempts"][0]["charged_seconds"] = saved["total_seconds"]
    path.write_text(json.dumps(saved))
    with pytest.raises(RuntimeError, match="does not extend"):
        recovery.recover(study_fixture.output, root=study_fixture.root)
    assert not study_fixture.launches


@pytest.mark.parametrize("change", ["source", "reference", "command"])
def test_changed_identity_or_command_rejected(recovery, study_fixture, change):
    f = study_fixture
    saved = json.loads((f.output / "study.json").read_text())
    if change == "source":
        saved["identity"]["source_sha256"] = "changed"
    elif change == "reference":
        (f.root / "swin_tiny-reference.json").write_text("changed")
    else:
        command = saved["attempts"][0]["command"]
        command[command.index("--row-batch") + 1] = "1"
    (f.output / "study.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        recovery.recover(f.output, root=f.root)
    assert not f.launches


@pytest.mark.parametrize("busy", [False, True])
def test_gpu_snapshot_rejects_competing_compute_processes(recovery, monkeypatch, busy):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    def output(command, **kwargs):
        if any("query-gpu=" in v for v in command):
            return "0, GPU-example, 24248, 22000\n"
        return "GPU-example, 222281\n" if busy else ""
    monkeypatch.setattr(recovery.subprocess, "check_output", output)
    if busy:
        with pytest.raises(RuntimeError, match="222281"):
            recovery.available_gpu()
    else:
        assert recovery.available_gpu()["free_mib"] == 22000
