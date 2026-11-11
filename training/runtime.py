"""Runtime observability and safe epoch-boundary stopping; no science settings."""
import json
import math
from pathlib import Path
import time

import torch


def atomic_json(path, value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def epoch_fits(remaining, cycles, margin):
    return remaining >= margin + 1.25 * max(cycles,default=0)


class TrainingRuntime:
    def __init__(self, directory, progress_every=0, wall_seconds=None, margin=900,
                 require_phase_gradient=False, started=None):
        self.directory=Path(directory)
        self.progress_every=progress_every
        self.wall_seconds=wall_seconds
        self.margin=margin
        self.require_phase_gradient=require_phase_gradient
        self.started=time.monotonic() if started is None else started
        self.epoch=0
        self.gradient_checked=False
        path=self.directory/'epoch_timings.json'
        self.timings=json.loads(path.read_text()) if path.exists() else []

    def can_start_epoch(self):
        if self.wall_seconds is None:return True
        remaining=self.wall_seconds-(time.monotonic()-self.started)
        return epoch_fits(remaining,[t['cycle_seconds'] for t in self.timings],self.margin)

    def progress(self,stage,index,total,loss,phase_gradient=None):
        if not math.isfinite(loss):raise RuntimeError('Nonfinite '+stage+' loss')
        if stage=='train' and not self.gradient_checked and self.require_phase_gradient:
            if phase_gradient is None or not torch.isfinite(phase_gradient).all() or not phase_gradient.abs().max()>0:
                raise RuntimeError('First raw-phase gradient is missing, zero or nonfinite')
            atomic_json(self.directory/'first_phase_gradient.json',dict(
                norm=float(phase_gradient.norm()),maximum=float(phase_gradient.abs().max()),
                epoch=self.epoch,batch=index))
            self.gradient_checked=True
        if not self.progress_every:return
        if index==0 or (index+1)%self.progress_every==0 or index+1==total:
            record=dict(epoch=self.epoch+1,stage=stage,batch=index+1,batches=total,
                        loss=loss,elapsed_seconds=time.monotonic()-self.started)
            atomic_json(self.directory/'live_progress.json',record)
            print('PROGRESS '+json.dumps(record),flush=True)

    def record_epoch(self,epoch,train_seconds,validation_seconds,checkpoint_seconds,
                     train_batches,validation_batches,loss,val_loss,checkpoint):
        cycle=train_seconds+validation_seconds+checkpoint_seconds
        if not all(math.isfinite(v) for v in (cycle,loss,val_loss)):
            raise RuntimeError('Nonfinite epoch measurements')
        record=dict(epoch=epoch+1,train_seconds=train_seconds,
                    validation_seconds=validation_seconds,checkpoint_seconds=checkpoint_seconds,
                    cycle_seconds=cycle,train_batches=train_batches,validation_batches=validation_batches,
                    seconds_per_train_batch=train_seconds/train_batches,
                    train_loss=loss,validation_loss=val_loss,checkpoint=str(checkpoint),
                    peak_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])
        self.timings.append(record)
        atomic_json(self.directory/'epoch_timings.json',self.timings)
        atomic_json(self.directory/'last_checkpoint.json',record)
        print('CHECKPOINT_VERIFIED '+json.dumps(record),flush=True)
