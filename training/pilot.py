"""Bounded matched-control pilot using the existing optical model, loss and v2 checkpoints.

Run only inside a two-L40S Slurm allocation. No scheduler submissions here.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.datasets import SyntheticMicroscopeData, ValidationDataset
from data.io import expand_config
from data.preprocessing import generate_bead_templates
from models.wrappers import ParallelEndToEndModel
from physics.masks import get_initial_phase_mask
from physics.phase_io import quantize_phase, save_codes
from training.checkpoint import (read_checkpoint, resume_config, restore_checkpoint,
                                 save_checkpoint)
from training.training import (BinaryDiceLoss, EarlyStopping, require_accumulation_steps,
                               train_one_epoch, validate_one_epoch)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def tensor_hash(items):
    h = hashlib.sha256()
    for tensor in items:
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def environment():
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Pilot requires exactly two visible GPUs')
    names = [torch.cuda.get_device_name(i) for i in range(2)]
    if not all('L40S' in name for name in names):
        raise RuntimeError('Pilot requires two L40S GPUs: ' + repr(names))
    return dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                numpy=np.__version__, gpu_names=names,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                job_id=os.environ.get('SLURM_JOB_ID'), host=os.uname().nodename)


def sync():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


class FiniteDice(BinaryDiceLoss):
    def forward(self, outputs, targets):
        if not torch.isfinite(outputs).all():
            raise RuntimeError('Nonfinite optical/network output')
        value = super().forward(outputs, targets)
        if not torch.isfinite(value):
            raise RuntimeError('Nonfinite Dice loss')
        return value


def continuation_state(path, target, requested):
    """Missing/invalid checkpoint fails; finished/early-stopped is a clean no-op."""
    state = read_checkpoint(path)
    resume_config(requested, state)
    finished = state['epoch'] + 1 >= target or state['early_stopping']['early_stop']
    return state, finished


def beam_profiles(sim, phase, dest, check_gaussian=False):
    """Renderer's axes: propagation x; collapse scan rows; remaining columns=z.

    Raw profiles preserve power. A separate peak-normalized fit checks Gaussian
    shape in the inner 80% of the finite computational pupil, not fixed dose.
    """
    dest = Path(dest)
    with torch.no_grad():
        field = sim.sample_4f(quantize_phase(phase, sim.config['slm_bit_depth']))
        xs = np.linspace(-sim.image_volume_size_px[1] // 2,
                         sim.image_volume_size_px[1] // 2, 9).astype(int)
        profiles = []
        for x in xs:
            propagated = sim.angular_spectrum_propagation(field, int(x), pad=True)
            profiles.append(propagated.abs().square().sum(dim=-2).squeeze().cpu().numpy())
    profiles = np.asarray(profiles)
    z_um = (np.arange(sim.N) - sim.N // 2) * sim.px * 1e6
    np.savez(dest.with_suffix('.npz'), propagation_x_um=xs * sim.px * 1e6,
             thickness_z_um=z_um, scan_collapsed_intensity=profiles)
    fit_reports = []
    inner = np.abs(z_um) < .4 * sim.N * sim.px * 1e6
    for profile in profiles:
        normalized = profile / profile.max()
        coeff = np.polyfit(z_um[inner], np.log(np.maximum(normalized[inner], 1e-12)), 2)
        fitted = np.exp(np.polyval(coeff, z_um[inner]))
        residual = float(np.linalg.norm(normalized[inner] - fitted) /
                         np.linalg.norm(normalized[inner]))
        fit_reports.append(dict(quadratic=float(coeff[0]), relative_l2=residual,
                                fitted_fwhm_um=float(2 * math.sqrt(math.log(2) / -coeff[0]))
                                if coeff[0] < 0 else None))
    report = dict(axes='rows: illumination propagation x; columns: sheet thickness z; scan y summed',
                  finite_pupil_width_um=sim.N * sim.px * 1e6,
                  input_amplitude_sigma_um=sim.laser_beam_FWHM * 1e6,
                  input_intensity_fwhm_um=2 * math.sqrt(math.log(2)) * sim.laser_beam_FWHM * 1e6,
                  gaussian_fit_region='inner 80 percent of finite grid', fits=fit_reports,
                  edge_to_peak=float(profiles[:, [0, -1]].max() / profiles.max()))
    report['gaussian_pass'] = all(f['quadratic'] < 0 and f['relative_l2'] < .05 for f in fit_reports)
    write_json(dest.with_suffix('.json'), report)
    if check_gaussian and not report['gaussian_pass']:
        raise RuntimeError('Flat-phase field failed Gaussian profile check; see ' + str(dest))
    return report


def segment(config_path, run_dir, target, seconds, margin=900, resume=False,
            verify_gaussian=False, evaluate=False):
    started = time.monotonic()
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    env = environment()
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    require_accumulation_steps(raw)
    if raw['batch_size'] != 1 or raw['num_classes'] != 1:
        raise ValueError('Pilot uses batch size one and binary detection')
    if raw['initial_phase_mask'] != 'empty' or raw['lens_approach'] != 'sample_4f':
        raise ValueError('Pilot requires matched flat-phase sample_4f geometry')
    weights = {k: v for k, v in raw.items() if k.endswith('_loss_weight')}
    if weights.get('dice_loss_weight') != 1 or any(v for k, v in weights.items() if k != 'dice_loss_weight'):
        raise ValueError('Pilot objective must be Dice only')
    checkpoint_path = run_dir / 'checkpoints/latest.pt'
    state = None
    if resume:
        state, finished = continuation_state(checkpoint_path, target, raw)
        if Path(state['config']['training_results_dir']).resolve() != run_dir:
            raise RuntimeError('Continuation changed persistent run directory')
        old_env = json.loads((run_dir / 'environment.json').read_text())
        if old_env['gpu_names'] != env['gpu_names']:
            raise RuntimeError('Continuation GPU count/order/types differ')
        if finished:
            return dict(noop=True, epoch=state['epoch'], reason='finished or early-stopped')
    elif checkpoint_path.exists():
        raise RuntimeError('Refusing to restart an existing run without --resume')
    write_json(run_dir / ('resume_environment.json' if resume else 'environment.json'), env)
    cfg = expand_config(copy.deepcopy(raw), str(run_dir))
    (run_dir / 'expanded.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    seed = raw['random_seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    if not (run_dir / raw['defocused_beads_filename']).exists():
        generate_bead_templates(cfg, str(run_dir))
    model = ParallelEndToEndModel(cfg)
    phase = torch.nn.Parameter(torch.from_numpy(get_initial_phase_mask(cfg)).to('cuda:0'),
                               requires_grad=not raw['freeze_slm'])
    initial_detector_hash = tensor_hash(model.unet.state_dict().values())
    optimizer = torch.optim.Adam(list(model.parameters()) + ([] if raw['freeze_slm'] else [phase]),
                                 lr=raw['initial_learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=raw['learning_rate_scheduler_factor'],
        patience=raw['learning_rate_scheduler_patience'])
    stopper = EarlyStopping(raw['early_stopping_patience'], raw['early_stopping_min_delta'])
    # Independent deterministic coordinate streams; A/B also reset validation noise
    # per epoch. This does not claim bitwise CUDA determinism.
    train_ds = SyntheticMicroscopeData(raw['ntrain'], cfg)
    if state is None:
        random.seed(seed + 10000); np.random.seed(seed + 10000)
        val_samples = [train_ds[i] for i in range(raw['nvalid'])]
    else:
        val_samples = state['val_samples']
    val_hash = tensor_hash(t for pair in val_samples for t in pair)
    write_json(run_dir / 'matching.json', dict(initial_detector_sha256=initial_detector_hash,
               validation_sha256=val_hash, ntrain=len(train_ds), nvalid=len(val_samples),
               particles=raw['num_particles_range'], camera='per-image percentile auto-exposure',
               percentile=raw['robust_max_percentile'], target_ratio=raw['target_saturation_ratio']))
    val_loader = DataLoader(ValidationDataset(val_samples), batch_size=1, num_workers=0)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)
    losses = {'dice_loss': FiniteDice()}
    train_losses, val_losses, start_epoch = [], [], 0
    if state is not None:
        train_losses, val_losses = state['train_losses'], state['val_losses']
        start_epoch = restore_checkpoint(state, model, phase, optimizer, scheduler, stopper)
    timing_path = run_dir / 'timings.json'
    timings = json.loads(timing_path.read_text()) if timing_path.exists() else []
    phase_gradient = {}

    def check_gradient(gradient):
        if not torch.isfinite(gradient).all():
            raise RuntimeError('Raw-phase gradient is nonfinite')
        if not phase_gradient:
            if not gradient.abs().max() > 0:
                raise RuntimeError('First raw-phase gradient is zero')
            phase_gradient.update(norm=float(gradient.norm()), maximum=float(gradient.abs().max()))
            write_json(run_dir / 'phase_gradient.json', phase_gradient)
        return gradient

    if phase.requires_grad:
        phase.register_hook(check_gradient)
    for i in range(2):
        torch.cuda.reset_peak_memory_stats(i)
    if verify_gaussian:
        beam_profiles(model.physics.replicas[0], phase, run_dir / 'initial_beam', True)
    for epoch in range(start_epoch, target):
        remaining = seconds - (time.monotonic() - started)
        measured_cycle = max((t['cycle_seconds'] for t in timings), default=0)
        # No signal handler saves; all stops are after verified atomic checkpoints.
        if remaining < measured_cycle * 1.25 + margin:
            if not timings:
                raise RuntimeError('No full epoch fits the available allocation margin')
            break
        # Explicit epoch streams make training scenes and camera draws matched.
        random.seed(seed + epoch); np.random.seed(seed + epoch); torch.manual_seed(seed + epoch)
        sync(); t0 = time.monotonic()
        train_loss = train_one_epoch(model, train_loader, optimizer, losses, phase, cfg, epoch)
        sync(); t1 = time.monotonic()
        torch.manual_seed(seed + 20000 + epoch)
        val_loss = validate_one_epoch(model, val_loader, losses, phase, cfg, epoch)
        sync(); t2 = time.monotonic()
        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            raise RuntimeError('Nonfinite epoch loss')
        train_losses.append(train_loss); val_losses.append(val_loss)
        scheduler.step(val_loss); stopper(val_loss)
        save_checkpoint(checkpoint_path, model, phase, optimizer, scheduler, stopper, epoch,
                        raw, cfg, train_losses, val_losses, val_samples)
        verified = read_checkpoint(checkpoint_path)
        if verified['epoch'] != epoch or len(verified['cuda_rng']) != 2:
            raise RuntimeError('Saved v2 checkpoint failed epoch/device verification')
        if epoch % raw['save_epoch_interval'] == 0 or epoch == target - 1:
            out = run_dir / 'learned_phase_masks'
            out.mkdir(exist_ok=True)
            np.save(out / ('raw_phase_%d.npy' % epoch), phase.detach().cpu().numpy())
            save_codes(phase.detach().cpu().numpy(), out / ('codes_%d.tif' % epoch), raw['slm_bit_depth'])
        sync(); t3 = time.monotonic()
        timings.append(dict(epoch=epoch, train_seconds=t1-t0, validation_seconds=t2-t1,
                     checkpoint_seconds=t3-t2, cycle_seconds=t3-t0,
                     seconds_per_train_step=(t1-t0)/len(train_loader),
                     peak_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(2)],
                     peak_reserved_bytes=[torch.cuda.max_memory_reserved(i) for i in range(2)],
                     train_loss=train_loss, validation_loss=val_loss))
        write_json(timing_path, timings)
        print('PILOT_EPOCH ' + json.dumps(timings[-1]), flush=True)
        if stopper.early_stop:
            break
    if not checkpoint_path.exists():
        raise RuntimeError('Segment did not produce a complete checkpoint')
    if evaluate:
        evaluate_model(model, phase, cfg, val_samples, run_dir, seed)
    final = read_checkpoint(checkpoint_path)
    result = dict(epoch=final['epoch'], run_dir=str(run_dir),
                  checkpoint=str(checkpoint_path), phase_gradient=phase_gradient,
                  seconds=time.monotonic()-started)
    write_json(run_dir / 'segment_result.json', result)
    return result


def evaluate_model(model, phase, cfg, samples, run_dir, seed):
    model.eval()
    counts = np.zeros(3, dtype=np.int64)
    field_counts = np.zeros((4, 3), dtype=np.int64)
    per_scene = []
    with torch.no_grad():
        torch.manual_seed(seed + 30000)
        for xyz, target in samples:
            logits = model(quantize_phase(phase, cfg['slm_bit_depth']), xyz[None].to('cuda:0'))
            pred = logits.sigmoid().cpu() >= .5
            truth = target[None].bool()
            tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
            counts += [tp, fp, fn]
            for b, (p, t) in enumerate(zip(torch.tensor_split(pred, 4, dim=-1),
                                         torch.tensor_split(truth, 4, dim=-1))):
                field_counts[b] += [int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())]
            per_scene.append(dict(tp=tp, fp=fp, fn=fn))
        beam_profiles(model.physics.replicas[0], phase, run_dir / 'final_beam')
    tp, fp, fn = map(int, counts)
    write_json(run_dir / 'evaluation.json', dict(tp=tp, fp=fp, fn=fn,
               pixel_precision=tp/max(1,tp+fp), pixel_recall=tp/max(1,tp+fn),
               pixel_dice=2*tp/max(1,2*tp+fp+fn), scenes=per_scene,
               field_bins_tp_fp_fn=field_counts.tolist(),
               field_bin_axis='four equal camera-column / propagation-x bins, left to right',
               threshold=.5, caveat='Validation scenes; pixel bead-center detection, not held-out fixed-dose performance'))


def smoke(root):
    """One job; only a CUDA OOM permits one documented particle-count reduction."""
    root = Path(root).resolve()
    base = yaml.safe_load((root / 'smoke/input.yaml').read_text())
    for count in (1200, 600):
        run = root / 'smoke' / ('particles_%d' % count)
        run.mkdir()
        raw = {**base, 'num_particles_range': [count, count + 1]}
        config = run / 'input.yaml'
        config.write_text(yaml.safe_dump(raw, sort_keys=False)); config.chmod(0o444)
        command = [sys.executable, '-u', '-m', 'training.pilot', 'segment', '--config', str(config),
                   '--run-dir', str(run), '--target', '1', '--seconds', '2700', '--gaussian']
        with (run / 'first.log').open('w') as log:
            attempt = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if attempt.returncode:
            message = (run / 'first.log').read_text()
            if count == 1200 and ('CUDA out of memory' in message or 'torch.OutOfMemoryError' in message):
                write_json(root / 'smoke/memory_reduction.json', dict(from_particles=1200, to_particles=600,
                           reason='CUDA OOM in 1200-particle smoke', log=str(run / 'first.log')))
                continue
            raise RuntimeError('Smoke failed; see ' + str(run / 'first.log'))
        first = read_checkpoint(run / 'checkpoints/latest.pt')
        if first['epoch'] != 0:
            raise RuntimeError('Smoke did not finish exactly its first epoch')
        command = [sys.executable, '-u', '-m', 'training.pilot', 'segment', '--config', str(config),
                   '--run-dir', str(run), '--target', '2', '--seconds', '2700', '--resume']
        with (run / 'continuation.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        final = read_checkpoint(run / 'checkpoints/latest.pt')
        if final['epoch'] != 1 or not np.array_equal(first['val_samples'][0][0], final['val_samples'][0][0]):
            raise RuntimeError('Smoke continuation failed epoch or scene consistency')
        if Path(final['config']['training_results_dir']) != run:
            raise RuntimeError('Smoke continuation changed run directory')
        _, done = continuation_state(run / 'checkpoints/latest.pt', 2, raw)
        if not done:
            raise RuntimeError('Finished continuation guard failed')
        try:
            continuation_state(run / 'missing.pt', 3, raw)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError('Missing checkpoint guard failed')
        timing = json.loads((run / 'timings.json').read_text())
        if len(timing) != 2 or not (run / 'phase_gradient.json').exists():
            raise RuntimeError('Smoke missing full cycle timing or phase gradient')
        write_json(root / 'smoke/success.json', dict(success=True, particles=count,
                   run_dir=str(run), epochs=[0,1], timing=timing,
                   checkpoint=str(run / 'checkpoints/latest.pt'), environment=environment(),
                   config_sha256=hashlib.sha256(config.read_bytes()).hexdigest()))
        return


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['smoke', 'segment'])
    parser.add_argument('--root')
    parser.add_argument('--config')
    parser.add_argument('--run-dir')
    parser.add_argument('--target', type=int)
    parser.add_argument('--seconds', type=int, default=6300)
    parser.add_argument('--margin', type=int, default=900)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gaussian', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.action == 'smoke':
        smoke(args.root)
    else:
        result = segment(args.config, args.run_dir, args.target, args.seconds,
                         args.margin, args.resume, args.gaussian, args.evaluate)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
