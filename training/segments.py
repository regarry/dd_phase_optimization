"""One fail-closed allocated segment; all submissions are prepared on login nodes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import yaml

from training.checkpoint import read_checkpoint,resume_config
from training.model_archive import archive_model,sha256
from training.runtime import atomic_json


def terminal_reason(state):
    minimum=float(state['raw_config'].get('minimum_training_seconds',0))
    seconds=float(state.get('progress',{}).get('train_seconds',0))
    maximum=float(state['raw_config'].get('maximum_training_seconds',float('inf')))
    if maximum<minimum:raise ValueError('Maximum training budget is below the required minimum')
    if seconds>=maximum:return 'training_budget_reached'
    if state['early_stopping']['early_stop']:
        if seconds<minimum:raise RuntimeError('Checkpoint early-stopped before its minimum training budget')
        return 'converged'
    if state['epoch']+1>=int(state['raw_config']['max_epochs']):
        return 'epoch_cap_reached' if seconds>=minimum else 'epoch_cap_before_minimum'
    return None


def inspect_predecessor(run,index,limit,raw):
    run=Path(run);checkpoint=run/'checkpoints/latest.pt'
    if not isinstance(limit,int) or not 1<=limit<=32 or not 1<=index<=limit:
        raise ValueError('Explicit bounded segment index/limit required')
    if index==1:
        if checkpoint.exists():raise RuntimeError('First segment refuses an existing checkpoint')
        return None,None
    receipt=run/'segments'/('%03d.json'%(index-1))
    if not receipt.exists():raise RuntimeError('Missing predecessor receipt; never restart from scratch')
    previous=json.loads(receipt.read_text())
    if previous['segment_index']!=index-1 or previous['segment_limit']!=limit:
        raise RuntimeError('Predecessor index/segment limit mismatch')
    if previous.get('success') is not True:raise RuntimeError('Predecessor did not complete successfully')
    if not checkpoint.is_file() or sha256(checkpoint)!=previous['checkpoint_sha256']:
        raise RuntimeError('Predecessor checkpoint missing or changed')
    state=read_checkpoint(checkpoint)
    if state['epoch']!=previous['checkpoint_epoch_index']:
        raise RuntimeError('Predecessor checkpoint epoch mismatch')
    resume_config(raw,state)
    if Path(state['config']['training_results_dir']).resolve()!=run.resolve():
        raise RuntimeError('Continuation run directory mismatch')
    return state,previous


def run_segment(run,index,limit,wall_seconds=6900,margin=900):
    run=Path(run).resolve();started=time.monotonic()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Segments execute only in a Slurm allocation')
    plan=json.loads((run/'segment_plan.json').read_text())
    if plan['segment_limit']!=limit:raise RuntimeError('Submitted segment bound differs from plan')
    config_path=run/'input.yaml'
    if sha256(config_path)!=plan['config_sha256']:raise RuntimeError('Immutable segment configuration changed')
    raw=yaml.safe_load(config_path.read_text())
    if raw.get('resume_checkpoint'):raise ValueError('Segment input must be the immutable new-run configuration')
    gpu_count=torch.cuda.device_count()
    if gpu_count!=int(plan['expected_gpu_count']):raise RuntimeError('Visible logical GPU count differs from plan')
    checkpoint=run/'checkpoints/latest.pt'
    gpu_names=[torch.cuda.get_device_name(i) for i in range(gpu_count)]
    (run/'segments').mkdir(exist_ok=True)
    own_receipt=run/'segments'/('%03d.json'%index)
    lease_path=run/'segments'/('%03d.started.json'%index)
    restart_count=int(os.environ.get('SLURM_RESTART_COUNT','0'))
    if restart_count:
        if restart_count>int(plan.get('maximum_preemptions',3)):
            raise RuntimeError('Bounded preemption restart limit exceeded')
        if not lease_path.is_file():raise RuntimeError('Requeued job has no original execution proof')
        lease=json.loads(lease_path.read_text())
        if lease['job_id']!=os.environ['SLURM_JOB_ID'] or lease['segment_index']!=index or lease['segment_limit']!=limit or lease['config_sha256']!=plan['config_sha256'] or lease['gpu_names']!=gpu_names:
            raise RuntimeError('Requeue identity/configuration/GPU proof mismatch')
        if not checkpoint.is_file():raise RuntimeError('Preempted before checkpoint; never restart from scratch')
        state=read_checkpoint(checkpoint);resume_config(raw,state)
        if Path(state['config']['training_results_dir']).resolve()!=run or state['epoch']<=lease['before_epoch']:
            raise RuntimeError('Requeue checkpoint does not advance original boundary')
        previous=None
        if own_receipt.exists():
            completed=json.loads(own_receipt.read_text())
            if completed.get('success') is not True or completed['job_id']!=os.environ['SLURM_JOB_ID'] or completed['checkpoint_sha256']!=sha256(checkpoint):
                raise RuntimeError('Requeued completion receipt failed verification')
            return completed
    else:
        state,previous=inspect_predecessor(run,index,limit,raw)
        if lease_path.exists():raise RuntimeError('Segment already started under a different execution')
        atomic_json(lease_path,dict(job_id=os.environ['SLURM_JOB_ID'],segment_index=index,segment_limit=limit,
            config_sha256=plan['config_sha256'],gpu_names=gpu_names,before_epoch=state['epoch'] if state else -1))
    if state is not None and len(state['cuda_rng'])!=gpu_count:
        raise RuntimeError('Resume requires original logical CUDA device count/order')
    if previous and previous['gpu_names']!=gpu_names:
        raise RuntimeError('GPU models/order changed across segments; explicit migration required')
    checkpoint=run/'checkpoints/latest.pt'
    reason=terminal_reason(state) if state is not None else None
    if reason=='epoch_cap_before_minimum':raise RuntimeError('Epoch cap reached before required actual training time')
    before_epoch=state['epoch'] if state is not None else -1
    (run/'segments').mkdir(exist_ok=True)
    own_receipt=run/'segments'/('%03d.json'%index)
    if own_receipt.exists():raise RuntimeError('Segment receipt already exists; refusing duplicate execution')
    if not reason:
        runtime_config=dict(raw)
        if state is not None:runtime_config['resume_checkpoint']=str(checkpoint)
        runtime_config_path=run/'segments'/('%03d.restart%02d.input.yaml'%(index,restart_count))
        runtime_config_path.write_text(yaml.safe_dump(runtime_config,sort_keys=False))
        runtime_config_path.chmod(0o444)
        remaining=wall_seconds-(time.monotonic()-started)
        if remaining<=margin:raise RuntimeError('No training budget remains after segment verification/setup')
        subprocess.run([sys.executable,'-u','-m','training.training','--config',str(runtime_config_path),
                        '--run-dir',str(run),'--worker-timeout','60','--progress-every','32',
                        '--require-phase-gradient','--wall-seconds',str(remaining),
                        '--stop-margin-seconds',str(margin)],check=True)
        state=read_checkpoint(checkpoint)
        if state['epoch']<=before_epoch:raise RuntimeError('Continuation did not advance a verified epoch')
        if not torch.isfinite(state['raw_phase']).all():raise RuntimeError('Nonfinite saved phase')
        reason=terminal_reason(state)
    actual=float(state.get('progress',{}).get('train_seconds',0))
    minimum=float(raw.get('minimum_training_seconds',0))
    if index==limit and not reason:
        reason='segment_cap_reached' if actual>=minimum else 'segment_cap_before_minimum'
    receipt=dict(success=reason not in ['epoch_cap_before_minimum','segment_cap_before_minimum'],
        segment_index=index,segment_limit=limit,job_id=os.environ['SLURM_JOB_ID'],
        no_op=before_epoch==state['epoch'],checkpoint_epoch_index=state['epoch'],
        checkpoint_sha256=sha256(checkpoint),progress=state.get('progress',{}),
        minimum_training_seconds=minimum,minimum_training_met=actual>=minimum,
        terminal_reason=reason,gpu_names=gpu_names,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        restart_count=restart_count,elapsed_seconds=time.monotonic()-started)
    if reason:
        receipt['model_bundle']=archive_model(run,plan['source_snapshot'],reason)
        atomic_json(run/'terminal.json',dict(reason=reason,minimum_training_met=actual>=minimum,
                                           converged=reason=='converged',progress=state.get('progress',{})))
    atomic_json(own_receipt,receipt)
    print('SEGMENT_RECEIPT '+json.dumps({k:v for k,v in receipt.items() if k!='model_bundle'}),flush=True)
    if not receipt['success']:raise RuntimeError('Bounded allocation chain ended before required training minimum')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['run']);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--segment-index',type=int,required=True);p.add_argument('--segment-limit',type=int,required=True)
    p.add_argument('--wall-seconds',type=float,default=6900);p.add_argument('--margin-seconds',type=float,default=900)
    a=p.parse_args();run_segment(a.run_dir,a.segment_index,a.segment_limit,a.wall_seconds,a.margin_seconds)
