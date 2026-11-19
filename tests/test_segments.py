import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from training.model_archive import sha256
from training.runtime import TrainingRuntime
from training.seeds import isolated_seed
from training.segments import inspect_predecessor,terminal_reason
import training.segments as segments
from training.model_archive import archive_model


def test_progress_counts_actual_scenes_emitters_updates_and_restores(tmp_path):
    runtime=TrainingRuntime(tmp_path)
    runtime.begin_epoch(0)
    runtime.observe_batch('train',2,13,{'ce_loss':torch.tensor(.5)},True)
    runtime.observe_batch('train',1,7,{'ce_loss':torch.tensor(.8)},True)
    runtime.observe_batch('validation',1,6,{'ce_loss':torch.tensor(.9)})
    progress=runtime.completed_progress(4,2)
    assert progress['train_scenes']==3 and progress['train_emitter_exposures']==20
    assert progress['optimizer_updates']==2 and progress['validation_scenes']==1
    runtime.record_epoch(0,4,2,1,2,1,.6,.9,tmp_path/'latest.pt')
    other=TrainingRuntime(tmp_path);other.restore_progress(progress,1);other.begin_epoch(1)
    other.observe_batch('train',1,9,{'ce_loss':1.})
    next_progress=other.completed_progress(3,1)
    assert next_progress['train_seconds']==7 and next_progress['train_scenes']==4
    assert next_progress['train_emitter_exposures']==29


def test_terminal_guard_enforces_minimum_actual_training_time():
    state=dict(raw_config=dict(minimum_training_seconds=10,max_epochs=100),
               progress=dict(train_seconds=9),early_stopping=dict(early_stop=True),epoch=2)
    with pytest.raises(RuntimeError,match='minimum'):terminal_reason(state)
    state['progress']['train_seconds']=10
    assert terminal_reason(state)=='converged'
    state['early_stopping']['early_stop']=False
    assert terminal_reason(state) is None


def test_continuation_missing_or_changed_checkpoint_never_restarts(tmp_path):
    raw=dict(max_epochs=100,minimum_training_seconds=10)
    with pytest.raises(RuntimeError,match='receipt'):inspect_predecessor(tmp_path,2,10,raw)
    (tmp_path/'segments').mkdir();(tmp_path/'checkpoints').mkdir()
    path=tmp_path/'checkpoints/latest.pt'
    state=dict(format_version=2,raw_config=raw,config={'training_results_dir':str(tmp_path)},epoch=2)
    torch.save(state,path)
    receipt=dict(success=True,segment_index=1,segment_limit=10,
                 checkpoint_sha256=sha256(path),checkpoint_epoch_index=2)
    (tmp_path/'segments/001.json').write_text(json.dumps(receipt))
    assert inspect_predecessor(tmp_path,2,10,raw)[0]['epoch']==2
    with path.open('ab') as f:f.write(b'changed')
    with pytest.raises(RuntimeError,match='changed'):inspect_predecessor(tmp_path,2,10,raw)


def test_optional_validation_seed_is_repeatable_without_disturbing_training_rng():
    random.seed(20);np.random.seed(20);torch.manual_seed(20)
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    with isolated_seed(100):a=(random.random(),np.random.rand(),torch.rand(1))
    with isolated_seed(100):b=(random.random(),np.random.rand(),torch.rand(1))
    assert a[:2]==b[:2];torch.testing.assert_close(a[2],b[2])
    assert random.getstate()==before[0]
    assert np.array_equal(np.random.get_state()[1],before[1][1])
    torch.testing.assert_close(torch.get_rng_state(),before[2])


def test_segment_chain_advances_then_noops_after_convergence(tmp_path,monkeypatch):
    import yaml
    raw=dict(max_epochs=100,minimum_training_seconds=1,resume_checkpoint=None)
    (tmp_path/'input.yaml').write_text(yaml.safe_dump(raw))
    (tmp_path/'segment_plan.json').write_text(json.dumps(dict(segment_limit=3,
        config_sha256=sha256(tmp_path/'input.yaml'),expected_gpu_count=2,source_snapshot=str(tmp_path))))
    monkeypatch.setenv('SLURM_JOB_ID','100')
    monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'example_gpu')
    monkeypatch.setattr(segments,'archive_model',lambda *a:dict(verified=True))
    calls=[]
    def child(*a,**kw):
        calls.append(a)
        path=tmp_path/'checkpoints/latest.pt';path.parent.mkdir(exist_ok=True)
        n=len(calls)
        state=dict(format_version=2,raw_config=raw,config=dict(training_results_dir=str(tmp_path)),
            epoch=n-1,early_stopping=dict(early_stop=n==2),raw_phase=torch.ones(1),
            progress=dict(train_seconds=n),cuda_rng=[torch.zeros(1),torch.zeros(1)])
        torch.save(state,path)
    monkeypatch.setattr(segments.subprocess,'run',child)
    one=segments.run_segment(tmp_path,1,3,wall_seconds=20,margin=1)
    two=segments.run_segment(tmp_path,2,3,wall_seconds=20,margin=1)
    three=segments.run_segment(tmp_path,3,3,wall_seconds=20,margin=1)
    assert one['checkpoint_epoch_index']==0 and two['checkpoint_epoch_index']==1
    assert two['terminal_reason']=='converged' and three['no_op']
    assert len(calls)==2


def test_model_bundle_preserves_source_state_and_standalone_detector(tmp_path):
    import subprocess,sys
    from models.unet import OpticsDesignUnet
    config=dict(num_classes=1,Nimgs=1,conv3d=False,image_volume=[16,16,16],
                defocused_beads_filename='beads.mat',training_results_dir=str(tmp_path))
    detector=OpticsDesignUnet(config).eval()
    (tmp_path/'beads.mat').write_bytes(b'fixture template bytes; not an optical simulation')
    (tmp_path/'checkpoints').mkdir()
    state=dict(format_version=2,epoch=1,config=config,raw_config=config,raw_phase=torch.zeros(2,2),
               model={'unet.'+k:v for k,v in detector.state_dict().items()},progress={'train_seconds':10})
    torch.save(state,tmp_path/'checkpoints/latest.pt')
    manifest=archive_model(tmp_path,Path.cwd(),'fixture_complete')
    bundle=tmp_path/'model_bundle'
    assert (bundle/'code/models/unet.py').exists() and (bundle/'full_state.pt').exists()
    assert manifest['detector']['available'],manifest['detector']
    traced=torch.jit.load(str(bundle/'detector.ts'))
    with torch.no_grad():torch.testing.assert_close(traced(torch.zeros(1,1,16,16)),detector(torch.zeros(1,1,16,16)))
    with torch.no_grad():torch.save(detector(torch.zeros(1,1,16,16)),tmp_path/'expected.pt')
    code="""import sys,torch
assert 'models.unet' not in sys.modules
model=torch.jit.load(sys.argv[1])
torch.testing.assert_close(model(torch.zeros(1,1,16,16)),torch.load(sys.argv[2],weights_only=True))
assert 'models.unet' not in sys.modules
"""
    subprocess.run([sys.executable,'-c',code,str(bundle/'detector.ts'),str(tmp_path/'expected.pt')],check=True,timeout=60)
    for name,digest in manifest['files'].items():assert sha256(bundle/name)==digest


def test_preempted_job_resumes_own_checkpoint_and_never_restarts(tmp_path,monkeypatch):
    import yaml
    raw=dict(max_epochs=100,minimum_training_seconds=2,maximum_training_seconds=2,resume_checkpoint=None)
    (tmp_path/'input.yaml').write_text(yaml.safe_dump(raw))
    plan=dict(segment_limit=1,config_sha256=sha256(tmp_path/'input.yaml'),expected_gpu_count=1,source_snapshot=str(tmp_path))
    (tmp_path/'segment_plan.json').write_text(json.dumps(plan))
    monkeypatch.setenv('SLURM_JOB_ID','111');monkeypatch.delenv('SLURM_RESTART_COUNT',raising=False)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'example_gpu')
    monkeypatch.setattr(segments,'archive_model',lambda *a:{})
    calls=[]
    def child(args,**kwargs):
        requested=yaml.safe_load(Path(args[args.index('--config')+1]).read_text());calls.append(requested)
        path=tmp_path/'checkpoints/latest.pt';path.parent.mkdir(exist_ok=True)
        n=len(calls)
        state=dict(format_version=2,raw_config=raw,config=dict(training_results_dir=str(tmp_path)),epoch=n-1,
            early_stopping=dict(early_stop=False),raw_phase=torch.ones(1),progress=dict(train_seconds=n),cuda_rng=[torch.zeros(1)])
        torch.save(state,path)
        if n==1:raise RuntimeError('simulated preemption after saved epoch')
    monkeypatch.setattr(segments.subprocess,'run',child)
    with pytest.raises(RuntimeError,match='preemption'):segments.run_segment(tmp_path,1,1,20,1)
    monkeypatch.setenv('SLURM_RESTART_COUNT','1')
    result=segments.run_segment(tmp_path,1,1,20,1)
    assert result['checkpoint_epoch_index']==1 and result['terminal_reason']=='training_budget_reached'
    assert calls[0]['resume_checkpoint'] is None
    assert calls[1]['resume_checkpoint']==str(tmp_path/'checkpoints/latest.pt')
    # A late scheduler requeue after the completed receipt is an idempotent no-op.
    monkeypatch.setenv('SLURM_RESTART_COUNT','2')
    assert segments.run_segment(tmp_path,1,1,20,1)['success'] and len(calls)==2
    monkeypatch.setenv('SLURM_JOB_ID','222')
    with pytest.raises(RuntimeError,match='proof mismatch'):segments.run_segment(tmp_path,1,1,20,1)


def test_preemption_before_first_checkpoint_fails_closed(tmp_path,monkeypatch):
    import yaml
    raw=dict(resume_checkpoint=None)
    (tmp_path/'input.yaml').write_text(yaml.safe_dump(raw));(tmp_path/'segments').mkdir()
    digest=sha256(tmp_path/'input.yaml')
    (tmp_path/'segment_plan.json').write_text(json.dumps(dict(segment_limit=1,config_sha256=digest,expected_gpu_count=1)))
    (tmp_path/'segments/001.started.json').write_text(json.dumps(dict(job_id='111',segment_index=1,segment_limit=1,config_sha256=digest,gpu_names=['example_gpu'],before_epoch=-1)))
    monkeypatch.setenv('SLURM_JOB_ID','111');monkeypatch.setenv('SLURM_RESTART_COUNT','1')
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1);monkeypatch.setattr(torch.cuda,'get_device_name',lambda i:'example_gpu')
    with pytest.raises(RuntimeError,match='never restart'):segments.run_segment(tmp_path,1,1,20,1)
