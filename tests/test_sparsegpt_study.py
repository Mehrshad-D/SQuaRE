"""Exercise the persistent cap, error recovery and launch gates without a GPU."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from pretrained_isolation.sparsegpt import study as module
from pretrained_isolation.obc.protocol import digest


@pytest.fixture
def controller(tmp_path, monkeypatch):
    root, output, source = tmp_path, tmp_path / 'sparse', tmp_path / 'obc'
    for name in module.MODELS:
        p = root / 'outputs-v4/imagenetv2' / name / f'{name}_imagenetv2_global_refinement.json'
        p.parent.mkdir(parents=True)
        p.write_text('{}')
    clock = [10.]
    launches = []
    old_identity = {'device': 'cuda', 'data_root': str(root/'data')}
    monkeypatch.setattr(module, 'preflight', lambda *a: ({'identity': old_identity, 'fingerprint': digest(old_identity)}, root/'data'))
    monkeypatch.setattr(module, 'verify_patch', lambda *a: None)
    monkeypatch.setattr(module, 'available_gpu', lambda: {'other_compute_pids': [], 'free_mib': 24000})
    monkeypatch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setenv('PYTHONPATH', '')
    def plans(*args):
        models = {n: {'projected_seconds': 0. if (output/n/'results.json').exists() else 600.,
                      'search_reserve_seconds': 90., 'final_reserve_seconds': 90.} for n in module.MODELS}
        return {'models': models, 'projected_remaining_seconds': sum(r['projected_seconds'] for r in models.values())}
    monkeypatch.setattr(module, 'plan_from_profiles', plans)
    def child(command, timeout, log, lock):
        launches.append((command, timeout))
        clock[0] += 10.
        if command[2].endswith('selfcheck'):
            p = Path(command[command.index('--output')+1]); p.write_text('{"passed":true}')
        else:
            directory = Path(command[command.index('--output-dir')+1]); directory.mkdir(parents=True, exist_ok=True)
            if command[3] == 'profile':
                (directory/'pilot.json').write_text('{"completed_at":"done","layers":[{"status":"completed"}]}')
            else:
                (directory/'results.json').write_text('{"completed_at":"done"}')
        return 0, 'exited'
    monkeypatch.setattr(module, 'run_child', child)
    from pretrained_isolation.sparsegpt import audit, compare
    monkeypatch.setattr(audit, 'verify_run', lambda *a: {'passed': True})
    monkeypatch.setattr(compare, 'compare_study', lambda *a: None)
    args = SimpleNamespace(obc_root=str(source), output_root=str(output), phase='all', check_only=False)
    return SimpleNamespace(root=root, output=output, source=source, args=args, clock=clock, launches=launches, child=child)


def test_two_hour_ledger_survives_profile_then_run_and_completed_restart(controller):
    f = controller
    f.args.phase = 'profile'
    first = module.study(f.args, root=f.root)
    assert first['status'] == 'profile_complete' and len(f.launches) == 4
    assert sum(a['charged_seconds'] for a in first['attempts']) == 40
    f.args.phase = 'run'
    saved = module.study(f.args, root=f.root)
    assert saved['status'] == 'completed' and len(f.launches) == 7
    assert saved['total_seconds'] == 7200 and saved['remaining_seconds'] == 7130
    module.study(f.args, root=f.root)
    assert len(f.launches) == 7  # Export only; no repeated completed GPU work.


def test_busy_gpu_does_not_launch_or_charge(controller, monkeypatch):
    def busy():
        raise RuntimeError('occupied')
    monkeypatch.setattr(module, 'available_gpu', busy)
    with pytest.raises(RuntimeError, match='occupied'):
        module.study(controller.args, root=controller.root)
    saved = json.loads((controller.output/'study.json').read_text())
    assert saved['attempts'] == [] and not controller.launches


def test_error_stops_and_next_invocation_retries_instead_of_silently_skipping(controller, monkeypatch):
    f = controller
    def fail(command, *args):
        if command[2].endswith('cli') and command[3] == 'run':
            f.clock[0] += 30.
            return 1, 'exited'
        return f.child(command, *args)
    monkeypatch.setattr(module, 'run_child', fail)
    with pytest.raises(RuntimeError, match='exit 1'):
        module.study(f.args, root=f.root)
    before = json.loads((f.output/'study.json').read_text())
    assert before['attempts'][-1]['charged_seconds'] == 30
    assert before['attempts'][-1]['model'] == 'deit_tiny'
    monkeypatch.setattr(module, 'run_child', f.child)
    after = module.study(f.args, root=f.root)
    assert after['status'] == 'completed'
    assert after['attempts'][:len(before['attempts'])] == before['attempts']
    assert sum(a['charged_seconds'] for a in after['attempts']) == 100


def test_profile_projection_stops_before_full_launch(controller, monkeypatch):
    f = controller
    monkeypatch.setattr(module, 'plan_from_profiles', lambda *a: {'models': {}, 'projected_remaining_seconds': 20000.})
    with pytest.raises(RuntimeError, match='projection exceeds'):
        module.study(f.args, root=f.root)
    assert len(f.launches) == 4
    assert json.loads((f.output/'study.json').read_text())['status'] == 'profile_exceeds_remaining_budget'


def test_consumed_budget_cannot_reset(controller):
    f = controller
    f.args.phase = 'profile'
    module.study(f.args, root=f.root)
    path = f.output/'study.json'
    saved = json.loads(path.read_text())
    saved['attempts'][0]['charged_seconds'] = 7200
    path.write_text(json.dumps(saved))
    f.args.phase = 'run'
    with pytest.raises(RuntimeError, match='projection exceeds'):
        module.study(f.args, root=f.root)
    assert len(f.launches) == 4


def test_check_only_launches_nothing_and_creates_no_study(controller):
    f = controller
    f.args.check_only = True
    assert module.study(f.args, root=f.root)['status'] == 'preflight_ready'
    assert not f.launches and not f.output.exists()


def test_profile_rejects_partial_rows_and_handles_already_completed(tmp_path):
    for name in module.MODELS:
        d = tmp_path/name; d.mkdir()
        (d/'manifest.json').write_text(json.dumps({'layer_shapes': {'p':[8,16]}, 'identity':{'grid':[{'id':'dense'}]}}))
        (d/'pilot.json').write_text(json.dumps({'completed_at':'done','layers':[{'status':'completed','sampled_candidates_reusable':False}]}))
    with pytest.raises(ValueError, match='Full-row pilot incomplete'):
        module.plan_from_profiles(tmp_path, tmp_path/'source')
    for name in module.MODELS:
        (tmp_path/name/'results.json').write_text('{"completed_at":"done"}')
    assert module.plan_from_profiles(tmp_path, tmp_path/'source')['projected_remaining_seconds'] == 0
