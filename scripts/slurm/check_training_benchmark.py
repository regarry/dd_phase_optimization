"""Fail-closed GPU-job gate: prove a real benchmark trained and saved v2 state."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from training.checkpoint import read_checkpoint
from training.runtime import atomic_json


def verify_benchmark(root,run):
    root=Path(root);run=Path(run)
    path=run/'checkpoints/latest.pt'
    state=read_checkpoint(path)
    if state['epoch']!=0 or len(state['train_losses'])!=1 or len(state['val_losses'])!=1:
        raise RuntimeError('Benchmark did not save exactly one complete epoch')
    if (state['raw_config']['ntrain'],state['raw_config']['nvalid'])!=(32,8):
        raise RuntimeError('Unexpected benchmark sample counts')
    if len(state['cuda_rng'])!=2 or Path(state['config']['training_results_dir']).resolve()!=run.resolve():
        raise RuntimeError('Benchmark checkpoint device count or run directory differs')
    if not all(math.isfinite(v) for v in state['train_losses']+state['val_losses']):
        raise RuntimeError('Benchmark losses are nonfinite')
    gradient=json.loads((run/'first_phase_gradient.json').read_text())
    if not math.isfinite(gradient['norm']) or gradient['norm']<=0:
        raise RuntimeError('Benchmark has no finite nonzero raw-phase gradient')
    steps=[float(v['step']) for v in state['optimizer']['state'].values() if 'step' in v]
    if not steps or max(steps)<1:raise RuntimeError('Benchmark has no optimizer update')
    timing=json.loads((run/'last_checkpoint.json').read_text())
    if timing['epoch']!=1 or timing['train_batches']!=32 or timing['validation_batches']!=8:
        raise RuntimeError('Benchmark timing/sample-count proof failed')
    proof=dict(success=True,checkpoint=str(path.resolve()),
               checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               completed_epochs=1,optimizer_steps=max(steps),gradient=gradient,timing=timing)
    atomic_json(root/'benchmark_success.json',proof)
    print('Benchmark trained and saved a verified v2 checkpoint.',flush=True)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',required=True);p.add_argument('--run',required=True)
    args=p.parse_args();verify_benchmark(args.root,args.run)
