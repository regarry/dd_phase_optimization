"""Complete, atomic checkpoints at completed epoch boundaries."""
import os
import random
from pathlib import Path
import numpy as np
import torch

RESUME_CONTROLS = {'resume_checkpoint', 'max_epochs', 'save_epoch_interval',
                   'resume_reset_early_stopping'}

def read_checkpoint(path):
    # Only load your own trusted training checkpoints (contains Python/NumPy RNG state).
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('format_version') != 2:
        raise ValueError('Checkpoint predates averaged-gradient resume format v2; exact continuation requires a new v2 checkpoint')
    return state

def resume_config(requested, checkpoint):
    original = checkpoint['raw_config']
    for key, value in requested.items():
        if key not in RESUME_CONTROLS and original.get(key) != value:
            raise ValueError(f'Resume changes {key!r}; exact continuation requires the saved configuration')
    return {**original, **{k:v for k,v in requested.items() if k in RESUME_CONTROLS}}

def save_checkpoint(path, model, phase, optimizer, scheduler, stopper, epoch,
                    raw_config, config, train_losses, val_losses, val_samples):
    state = dict(format_version=2, model=model.state_dict(), raw_phase=phase.detach().cpu(),
                 optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                 early_stopping=vars(stopper).copy(), epoch=epoch, raw_config=raw_config,
                 config=config, train_losses=train_losses, val_losses=val_losses,
                 val_samples=val_samples, python_rng=random.getstate(),
                 numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)

def restore_checkpoint(state, model, phase, optimizer, scheduler, stopper, reset_early=False):
    if tuple(phase.shape) != tuple(state['raw_phase'].shape):
        raise ValueError('Checkpoint phase shape differs from model')
    if len(state['cuda_rng']) != torch.cuda.device_count():
        raise ValueError('Resume requires the same visible CUDA device count/order')
    model.load_state_dict(state['model'])
    with torch.no_grad(): phase.copy_(state['raw_phase'].to(phase.device))
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    if not reset_early:
        vars(stopper).update(state['early_stopping'])
        if stopper.early_stop:
            raise ValueError('Run already early-stopped; explicitly set resume_reset_early_stopping: true to continue')
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']: torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['epoch'] + 1
