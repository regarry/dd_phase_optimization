"""CPU checks for pilot gates, immutable artifacts and continuation decisions."""
import importlib.util
from pathlib import Path
import json
import subprocess

import pytest
import torch

from training.pilot import continuation_state

spec = importlib.util.spec_from_file_location('pilot_control', 'scripts/slurm/pilot_control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def smoke_report(train=10, valid=3, save=2):
    return dict(success=True, particles=1200, timing=[dict(train_seconds=train,
                validation_seconds=valid, checkpoint_seconds=save)] * 2)


def test_plan_uses_measured_full_cycle():
    p = control.choose_plan(smoke_report())
    assert p['estimated_cycle_seconds'] == 48
    assert p['epochs'] == 3 and p['ntrain'] == 8 and p['nvalid'] == 4
    assert p['margin_seconds'] == 900
    assert 1800 <= p['wall_seconds'] <= 21600


@pytest.mark.parametrize('report', [dict(success=False,timing=[]), dict(success=True,timing=[])])
def test_smoke_gate_refuses_missing_timing(report):
    with pytest.raises(RuntimeError):
        control.choose_plan(report)


def test_unbounded_cycle_rejected():
    with pytest.raises(RuntimeError):
        control.choose_plan(smoke_report(train=10000))


def test_render_and_immutable_verification(tmp_path,monkeypatch):
    monkeypatch.setenv('DD_SLURM_GRES','gpu:l40s:2')
    monkeypatch.setenv('DD_SLURM_SMOKE_QOS','short_gpu')
    (tmp_path / 'A').mkdir()
    (tmp_path / 'A/input.yaml').write_text('ntrain: 8\n')
    (tmp_path / 'source_manifest.json').write_text('{}')
    control.render(tmp_path, 'A', 'p_bme_gpu', 90, 'arm')
    control.verify(tmp_path, 'A')
    script = (tmp_path / 'A/submitted.sbatch').read_text()
    assert '--gres=gpu:l40s:2' in script and '--nodes=1' in script
    assert '--qos=p_bme_gpu' in script and '--time=01:30:00' in script
    assert 'export PYTHONPATH=' + str(tmp_path) + '/code' in script
    assert 'sbatch ' not in script
    continuation = (tmp_path / 'A/continuation.sbatch').read_text()
    assert '--qos=short_gpu' in continuation and '--time=02:00:00' in continuation
    assert ' continue --root ' in continuation
    path = tmp_path / 'A/input.yaml'
    path.chmod(0o644); path.write_text('ntrain: 9\n')
    with pytest.raises(RuntimeError, match='artifact changed'):
        control.verify(tmp_path, 'A')


def test_continuation_dependency_has_explicit_bound():
    with pytest.raises(ValueError):
        control.continuation_command('continuation.sbatch', 123, None, 1)
    with pytest.raises(ValueError):
        control.continuation_command('continuation.sbatch', 123, 2, 3)
    command = control.continuation_command('continuation.sbatch', 123, 2, 1)
    assert '--dependency=afterok:123' in command
    assert '--kill-on-invalid-dep=yes' in command


def test_continuation_checks_checkpoint_before_decision(tmp_path):
    path = tmp_path / 'latest.pt'
    with pytest.raises(FileNotFoundError):
        continuation_state(path, 2, {})
    state = dict(format_version=2, raw_config={'wavelength':561e-9}, epoch=0,
                 early_stopping={'early_stop':False})
    torch.save(state, path)
    assert continuation_state(path, 2, state['raw_config'])[1] is False
    assert continuation_state(path, 1, state['raw_config'])[1] is True
    with pytest.raises(ValueError, match='wavelength'):
        continuation_state(path, 2, {'wavelength':532e-9})
    state['early_stopping']['early_stop'] = True
    torch.save(state, path)
    assert continuation_state(path, 2, state['raw_config'])[1] is True


def test_failed_predecessor_blocks_b_before_model_creation(tmp_path):
    (tmp_path / 'source_manifest.json').write_text('{}')
    for name in ('A', 'B', 'smoke'):
        (tmp_path / name).mkdir()
    (tmp_path / 'B/artifacts.json').write_text('{}')
    report = smoke_report()
    (tmp_path / 'smoke/success.json').write_text(json.dumps(report))
    (tmp_path / 'plan.json').write_text(json.dumps(control.choose_plan(report)))
    (tmp_path / 'A/success.json').write_text('{"success": false}')
    with pytest.raises(RuntimeError, match='A success gate'):
        control.run_arm(tmp_path, 'B')
