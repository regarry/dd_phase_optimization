"""Verify an interrupted Slurm segment's last complete checkpoint for continuation.

Run on login node: no training or GPU operations. Preserves the failed scheduler
state in an explicit recovery receipt, not a claim of successful training.
"""
import argparse,hashlib,json,subprocess
from pathlib import Path
import torch,yaml
from training.checkpoint import read_checkpoint,resume_config


def verify_boundary(run,index,failed_job):
    run=Path(run);receipt=run/'segments'/('%03d.json'%index)
    if receipt.exists():raise RuntimeError('Refusing to overwrite segment receipt')
    previous=json.loads((run/'segments'/('%03d.json'%(index-1))).read_text())
    checkpoint=run/'checkpoints/latest.pt';state=read_checkpoint(checkpoint)
    resume_config(yaml.safe_load((run/'input.yaml').read_text()),state)
    if state['epoch']<previous['checkpoint_epoch_index']:raise RuntimeError('Checkpoint regressed')
    if len(state['train_losses'])!=state['epoch']+1 or len(state['val_losses'])!=state['epoch']+1:raise RuntimeError('Incomplete histories')
    if len(state['cuda_rng'])!=len(previous['gpu_names']):raise RuntimeError('CUDA state changed')
    if not torch.isfinite(state['raw_phase']).all():raise RuntimeError('Nonfinite phase')
    if not all(torch.isfinite(v).all() for v in state['model'].values() if v.is_floating_point()):raise RuntimeError('Nonfinite model')
    if state['progress']['completed_epochs']!=state['epoch']+1:raise RuntimeError('Progress and epoch disagree')
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return dict(success=True,recovery=True,checkpoint_boundary_verified=True,
        original_slurm_state='FAILED',original_failed_job=failed_job,job_id=str(failed_job),
        note='Success applies to verification of the saved complete epoch boundary only. Original job FAILED on gpu19 with CUDA misaligned address. Incomplete epoch work is not counted; original state is never restarted from scratch.',
        segment_index=index,segment_limit=previous['segment_limit'],checkpoint_epoch_index=state['epoch'],
        checkpoint_sha256=digest,progress=state['progress'],gpu_names=previous['gpu_names'],visible_devices=previous.get('visible_devices'),
        minimum_training_seconds=state['raw_config'].get('minimum_training_seconds',0),
        minimum_training_met=state['progress']['train_seconds']>=state['raw_config'].get('minimum_training_seconds',0),
        terminal_reason=None,no_op=state['epoch']==previous['checkpoint_epoch_index'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--failed-job',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    data=verify_boundary(a.run,a.index,a.failed_job);Path(a.output).write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({k:data[k] for k in ('checkpoint_epoch_index','checkpoint_sha256','minimum_training_met')}))
