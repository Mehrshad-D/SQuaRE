from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import time

import pytest
import torch
from torch import nn
from pretrained_isolation.config import load_config, layerwise_configurations
from pretrained_isolation.sparsegpt.core import prepare_hessian, compress, build_candidates
from pretrained_isolation.sparsegpt.reference import reference_class, MatchedQuantizer
from pretrained_isolation.sparsegpt.selfcheck import dense_quantization_oracle, selfcheck
from pretrained_isolation.obc.core import CompressionTimeout
from pretrained_isolation.obc.protocol import digest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("bits", [32, 8, 6, 4])
@pytest.mark.parametrize("n,m", [(2, 4), (4, 8), (3, 8)])
@pytest.mark.parametrize("block_size", [8, 32])
def test_sparse_kernel_matches_pinned_upstream(bits, n, m, block_size):
    torch.manual_seed(301)
    w = torch.randn(9, 32)
    x = torch.randn(90, 32)
    x[:, 7] = 0  # Exercise upstream dead-coordinate treatment too.
    gram = x.T @ x / len(x)
    factor, dead = prepare_hessian(gram)
    actual, mask, _ = compress(w, factor, dead, bits=bits, n=n, m=m, block_size=block_size)
    layer = nn.Linear(32, 9, bias=False)
    layer.weight.data.copy_(w)
    reference = reference_class()(layer)
    reference.H = gram.clone()
    if bits != 32:
        reference.quantizer = MatchedQuantizer(w, bits)
    with contextlib.redirect_stdout(io.StringIO()):
        reference.fasterprune(0., prunen=m-n, prunem=m, blocksize=block_size, percdamp=.01)
    torch.testing.assert_close(actual, layer.weight, atol=2e-5, rtol=2e-5)
    assert torch.all(mask.reshape(9, -1, m).sum(-1) == n)
    assert torch.all(actual[~mask] == 0)


@pytest.mark.parametrize("bits", [4, 6, 8])
def test_dense_quantization_matches_independent_recomputed_inverse(bits):
    torch.manual_seed(76)
    x, w = torch.randn(128, 32), torch.randn(5, 32)
    gram = x.T @ x / len(x)
    factor, dead = prepare_hessian(gram)
    result, mask, scale = compress(w, factor, dead, bits=bits, block_size=8)
    torch.testing.assert_close(result, dense_quantization_oracle(w, gram, bits), atol=2e-5, rtol=2e-5)
    assert mask.all()
    torch.testing.assert_close(scale, w.abs().amax(1, keepdim=True) / (2**(bits-1)-1))


def test_dense_control_exact_and_cross_block_correlations_retained():
    torch.manual_seed(29)
    x, w = torch.randn(64, 32), torch.randn(4, 32)
    x[:, 16:] += .7*x[:, :16]
    gram = x.T @ x / len(x)
    factor, dead = prepare_hessian(gram)
    exact, _, _ = compress(w, factor, dead)
    assert torch.equal(exact, w)
    a = compress(w, factor, dead, bits=4, n=3, m=8, block_size=8)[0]
    b = compress(w, factor, dead, bits=4, n=3, m=8, block_size=32)[0]
    torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    diagonal = torch.block_diag(gram[:16, :16], gram[16:, 16:])
    h, d = prepare_hessian(diagonal)
    c = compress(w, h, d, bits=4, n=3, m=8, block_size=8)[0]
    assert not torch.allclose(a, c)


def test_conv_matrix_adapter_and_all_candidates_reconstruction():
    from pretrained_isolation.quantization import fake_quant_symmetric
    torch.manual_seed(43)
    conv = nn.Conv2d(8, 5, 3, bias=False)
    images = torch.randn(3, 8, 6, 6)
    x = torch.nn.functional.unfold(images, 3).transpose(1, 2).reshape(-1, 72)
    stats = {"gram": x.double().T @ x.double()/len(x), "score_inputs": x}
    specs = layerwise_configurations(load_config(ROOT / "configs/resnet18_imagenetv2.yaml"))
    bank = build_candidates(conv.weight, stats, specs, x.abs().max(), block_size=16)
    assert len(bank) == 16
    for spec in specs:
        entry = bank[spec["id"]]
        quantized = fake_quant_symmetric(x, spec["activation_bits"], fixed_amax=x.abs().max())
        target = torch.nn.functional.linear(x, conv.weight.flatten(1))
        actual = torch.nn.functional.linear(quantized, entry["weight"].flatten(1))
        expected = (actual-target).double().square().sum()/target.double().square().sum()
        assert entry["score"] == pytest.approx(expected.item(), abs=1e-9)
        assert entry["weight"].shape == conv.weight.shape


def test_deadline_invalid_groups_and_selfcheck():
    h, dead = prepare_hessian(torch.eye(16))
    with pytest.raises(CompressionTimeout):
        compress(torch.randn(3, 16), h, dead, deadline=time.monotonic()-1)
    with pytest.raises(ValueError):
        compress(torch.randn(3, 16), h, dead, n=8, m=8)
    assert selfcheck()["passed"]


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    from torch.utils.data import DataLoader, TensorDataset
    from pretrained_isolation.obc import cli as obc
    from pretrained_isolation.sparsegpt import cli as sparse
    from pretrained_isolation.engine import evaluate
    torch.manual_seed(22)
    model = nn.Sequential()
    model.add_module("proj", nn.Linear(16, 6))
    refinement = DataLoader(TensorDataset(torch.randn(12, 16), torch.randint(0, 6, (12,))), batch_size=4)
    final = DataLoader(TensorDataset(torch.randn(12, 16), torch.randint(0, 6, (12,))), batch_size=4)
    cfg = load_config(ROOT / "configs/deit_tiny_imagenetv2.yaml")
    cfg["model"] = {"name": "toy", "short_name": "toy"}
    cfg["selection"] = {"module_types": ["linear"], "include": ["proj"]}
    cfg["data"]["batch_size"] = 4
    dense, dense_ref = evaluate(model, final, torch.device("cpu")), evaluate(model, refinement, torch.device("cpu"))
    reference = {"suite": "global_refinement", "model": cfg["model"], "selection": cfg["selection"], "seed": cfg["seed"],
        "data": {"label_space": {"dataset": "ImageNetV2"}, "resolved_model_data_config": {}},
        "refinement_set": {"samples": 12, "dense_accuracy": dense_ref}, "dense_baseline": dense,
        "selected_layer_count": 1, "pareto_frontiers": {"proj": []}, "environment": {"gpu": None}, "completed_at_utc": "fixture"}
    spec = next(s for s in layerwise_configurations(cfg) if s["id"] == "FP32__dense")
    reference["experiments"] = [{"status": "succeeded", "accuracy_threshold_top1_pp": b,
        "accuracy": dense, "accuracy_drop_top1_pp": 0., "refinement_accuracy_drop_top1_pp": 0.,
        "energy_proxy": {"normalized_energy_vs_selected_layer_dense_baseline": 1.},
        "selected_layers": [{"target_layer": "proj", "configuration": spec["id"], "specification": spec, "normalized_energy": 1.}],
        "total_seconds": 1., "search_evaluation_count": 1} for b in [.1, .5, 1.]]
    # Use three model directory names so the exact three-way exporter is tested.
    models = ("deit_tiny", "swin_tiny", "resnet18")
    ref_paths = []
    for name in models:
        layerwise = tmp_path / "outputs-v2/imagenetv2" / name / f"{name}_imagenetv2_layerwise.json"
        layerwise.parent.mkdir(parents=True)
        layerwise.write_text(json.dumps({"experiments": [{"status": "succeeded", "total_seconds": 2.}]}))
        from pretrained_isolation.obc.evaluation import file_sha256
        reference["selection_source"] = {"sha256": file_sha256(layerwise)}
        ref_path = tmp_path / "outputs-v4/imagenetv2" / name / f"{name}_imagenetv2_global_refinement.json"
        ref_path.parent.mkdir(parents=True)
        ref_path.write_text(json.dumps(reference))
        ref_paths.append(ref_path)
    for cli in (obc, sparse):
        monkeypatch.setattr(cli, "load_config", lambda _: deepcopy(cfg))
        monkeypatch.setattr(cli, "create_pretrained", lambda _: deepcopy(model))
        monkeypatch.setattr(cli, "make_refinement_loaders", lambda *a: (refinement, final, {}, 12, 12, None, {"dataset": "ImageNetV2"}, {"request": {"source": "calib_dir"}}))
        monkeypatch.setattr(cli, "dataset_manifest", lambda ds: [{"file": str(i), "label": int(ds[i][1]), "sha256": ("ref" if ds is refinement.dataset else "final")+str(i)} for i in range(len(ds))])
    source, output = tmp_path / "obc", tmp_path / "sparse"
    args = obc.parser().parse_args(["run", "--config", "unused", "--reference", str(ref_paths[0]), "--output-dir", str(source / models[0]), "--device", "cpu", "--max-hours", "1", "--max-evaluations", "6", "--hessian-block-size", "256"])
    obc.run(args)
    args = sparse.parser().parse_args(["run", "--config", "unused", "--reference", str(ref_paths[0]), "--output-dir", str(output / models[0]), "--obc-source", str(source / models[0]), "--device", "cpu", "--max-hours", "1", "--max-evaluations", "6", "--resume"])
    return tmp_path, source, output, sparse, args, refinement, final, ref_paths


def test_end_to_end_reuse_frozen_resume_and_three_method_export(workflow, monkeypatch):
    import shutil
    root, source, output, cli, args, refinement, final, refs = workflow
    from pretrained_isolation.obc.evaluation import file_sha256
    from pretrained_isolation.sparsegpt.audit import verify_run
    before = {str(p.relative_to(source)): file_sha256(p) for p in source.rglob('*') if p.is_file()}
    events = []
    evaluate = cli.evaluate
    def checked(model, loader, *a, **kw):
        if loader is final and 'dense_final' not in str(kw['prediction_path']):
            assert (output / 'deit_tiny/frozen_selection.json').exists()
        events.append(loader)
        return evaluate(model, loader, *a, **kw)
    monkeypatch.setattr(cli, 'evaluate', checked)
    args.mode = "profile"
    pilot = cli.run(args)
    assert all(r["sampled_candidates_reusable"] for r in pilot["layers"])
    args.mode = "run"
    result = cli.run(args)
    assert len(result["experiments"]) == 3 and result['suite'] == 'sparsegpt_comparison'
    assert result['reused_statistics_seconds'] > 0
    assert verify_run(output / 'deit_tiny')['passed']
    frozen = (output / 'deit_tiny/frozen_selection.json').read_bytes()
    partial = deepcopy(result)
    partial.pop('completed_at')
    partial['experiments'] = partial['experiments'][:1]
    (output / 'deit_tiny/results.json').write_text(json.dumps(partial))
    monkeypatch.setattr(cli, 'proposed_assignments', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('search reopened')))
    events.clear()
    assert len(cli.run(args)['experiments']) == 3
    assert all(loader is final for loader in events)
    assert (output / 'deit_tiny/frozen_selection.json').read_bytes() == frozen
    assert before == {str(p.relative_to(source)): file_sha256(p) for p in source.rglob('*') if p.is_file()}
    # Fixture models deliberately share toy tensors; validate the real export
    # paths, 27 distinct method/model/budget rows, and unchanged source artifacts.
    for name in ('swin_tiny', 'resnet18'):
        shutil.copytree(source / 'deit_tiny', source / name)
        shutil.copytree(output / 'deit_tiny', output / name)
    for folder in (source, output):
        (folder / 'study.json').write_text(json.dumps({'attempts': []}))
    from pretrained_isolation.sparsegpt.compare import compare_study
    rows = compare_study(root, source, output, root / 'comparison')
    assert len(rows) == 27
    assert {r['method'] for r in rows} == {'SQuaRE', 'OBC-block256', 'SparseGPT-adapted'}
    assert (root / 'comparison/comparison_table.tex').exists()
    # Content corruption must fail the independent audit.
    prediction = output / 'deit_tiny' / result['experiments'][0]['accuracy']['predictions_file']
    prediction.write_bytes(prediction.read_bytes() + b'bad')
    with pytest.raises(ValueError, match='Prediction checksum'):
        verify_run(output / 'deit_tiny')


@pytest.mark.parametrize('change', ['reference', 'positions', 'partial_source'])
def test_reuse_rejects_mismatch_before_compression(workflow, change):
    root, source, output, cli, args, *_ = workflow
    if change == 'reference':
        p = Path(args.reference)
        ref = json.loads(p.read_text()); ref['seed'] += 1; p.write_text(json.dumps(ref))
    elif change == 'positions':
        args.positions_per_example = 16
    else:
        p = source / 'deit_tiny/results.json'
        result = json.loads(p.read_text()); result.pop('completed_at'); p.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        cli.run(args)
    assert not (output / 'deit_tiny/cache').exists()


def test_candidate_partial_commit_and_corruption_rejected(tmp_path):
    from pretrained_isolation.sparsegpt.cache import candidate_keys, load_candidate, save_candidate
    entry = {'configuration': 'test', 'weight': torch.ones(2, 8)}
    torch.save(entry, tmp_path / 'test.pt')
    assert not candidate_keys(tmp_path)
    save_candidate(tmp_path, 'test', entry, 'identity')
    assert torch.equal(load_candidate(tmp_path, 'test', 'identity')['weight'], entry['weight'])
    (tmp_path / 'test.pt').write_bytes((tmp_path / 'test.pt').read_bytes() + b'bad')
    with pytest.raises(ValueError, match='checksum'):
        load_candidate(tmp_path, 'test', 'identity')
