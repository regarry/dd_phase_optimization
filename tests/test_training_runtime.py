import json
import math

import pytest
import torch

from training.runtime import TrainingRuntime, epoch_fits


def test_epoch_budget_uses_slowest_cycle_and_margin():
    assert epoch_fits(1025,[80,100],900)
    assert not epoch_fits(1024,[80,100],900)
    assert epoch_fits(900,[],900)


def test_progress_rejects_bad_loss_and_phase_gradient(tmp_path):
    runtime=TrainingRuntime(tmp_path,progress_every=1,require_phase_gradient=True)
    with pytest.raises(RuntimeError,match='Nonfinite'):
        runtime.progress('train',0,1,float('nan'),torch.ones(2))
    with pytest.raises(RuntimeError,match='gradient'):
        runtime.progress('train',0,1,1.0,torch.zeros(2))
    runtime.progress('train',0,1,1.0,torch.ones(2))
    assert json.loads((tmp_path/'live_progress.json').read_text())['batch']==1
    assert json.loads((tmp_path/'first_phase_gradient.json').read_text())['norm']>0


def test_verified_cycle_persists_for_continuation(tmp_path):
    runtime=TrainingRuntime(tmp_path)
    runtime.record_epoch(0,2,1,.5,4,2,1,.9,tmp_path/'latest.pt')
    loaded=TrainingRuntime(tmp_path)
    assert loaded.timings[0]['cycle_seconds']==3.5
    assert json.loads((tmp_path/'last_checkpoint.json').read_text())['epoch']==1
