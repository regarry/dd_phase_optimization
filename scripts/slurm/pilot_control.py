"""Prepare/verify a bounded pilot on the login node; run it only under Slurm.

This program never submits jobs. Submission is explicit from the login terminal.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from datetime import datetime

import yaml

from scripts.slurm.site import directives, module_setup, settings

PYTHON = settings()['python']


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root, run=None):
    root = Path(root)
    for name, sha in json.loads((root / 'source_manifest.json').read_text()).items():
        if digest(root / 'code' / name) != sha:
            raise RuntimeError('Code snapshot changed: ' + name)
    if run:
        for name, sha in json.loads((root / run / 'artifacts.json').read_text()).items():
            if digest(root / run / name) != sha:
                raise RuntimeError('Immutable run artifact changed: ' + name)


def render(root, name, qos, minutes, action):
    run = root / name
    script = run / 'submitted.sbatch'
    q = shlex.quote
    command = ('-m training.pilot smoke --root ' + q(str(root))) if action == 'smoke' else (
        q(str(root / 'code/scripts/slurm/pilot_control.py')) + ' run --root ' + q(str(root)) + ' --arm ' + name)
    script.write_text(f'''#!/bin/bash
#SBATCH --job-name=ddpilot_{name}
{directives(qos)}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time={minutes // 60:02d}:{minutes % 60:02d}:00
#SBATCH --output={run}/slurm.%j.out
#SBATCH --error={run}/slurm.%j.err
set -euo pipefail
{module_setup()}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH={q(str(root / 'code'))}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR={q(str(run / 'mpl_cache'))}
cd {q(str(root / 'code'))}
{q(PYTHON)} {q(str(root / 'code/scripts/slurm/pilot_control.py'))} verify --root {q(str(root))} --arm {name}
hostname
module list 2>&1
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
source {q(str(root / 'code/scripts/slurm/worker_runtime.sh'))}
{q(PYTHON)} -m utils.worker_preflight
{q(PYTHON)} -u {command}
''')
    subprocess.run(['bash', '-n', str(script)], check=True)
    artifacts = ['input.yaml', 'submitted.sbatch']
    if name in ('A', 'B'):
        continuation = script.read_text()
        if qos:
            continuation = continuation.replace('#SBATCH --qos='+qos+'\n','')
        if settings()['smoke_qos']:
            continuation = continuation.replace('#!/bin/bash\n', '#!/bin/bash\n#SBATCH --qos='+settings()['smoke_qos']+'\n',1)
        continuation = continuation.replace(f'--time={minutes // 60:02d}:{minutes % 60:02d}:00', '--time=02:00:00')
        continuation = continuation.replace(f'--job-name=ddpilot_{name}', f'--job-name=ddpilot_{name}_continuation')
        continuation = continuation.replace(' run --root ', ' continue --root ')
        (run / 'continuation.sbatch').write_text(continuation)
        subprocess.run(['bash','-n',str(run / 'continuation.sbatch')], check=True)
        artifacts.append('continuation.sbatch')
    save_json(run / 'artifacts.json', {f: digest(run / f) for f in artifacts})
    for f in artifacts + ['artifacts.json']:
        (run / f).chmod(0o444)


def prepare(root, checkout):
    root.mkdir(parents=True, exist_ok=False)
    code = root / 'code'
    code.mkdir()
    files = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=checkout).decode().split('\0')
    for name in files:
        if not name or not (checkout / name).is_file():
            continue
        dest = code / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkout / name, dest)
        dest.chmod(0o444)
    save_json(root / 'source_manifest.json', {str(p.relative_to(code)): digest(p) for p in code.rglob('*') if p.is_file()})
    for command, filename in ((['git', 'diff', '--binary', 'HEAD'], 'local.diff'),
                              (['git', 'status', '--short'], 'git_status.txt'),
                              (['git', 'rev-parse', 'HEAD'], 'git_commit.txt'),
                              (['git', 'branch', '--show-current'], 'git_branch.txt')):
        (root / filename).write_bytes(subprocess.check_output(command, cwd=checkout))
    raw = yaml.safe_load((checkout / 'config.yaml').read_text())
    raw.update(batch_size=1, num_particles_range=[1200,1201], ntrain=2, nvalid=2,
               max_epochs=2, initial_phase_mask='empty', freeze_slm=False, freeze_unet=False,
               z_coupled_ratio=0.0, num_classes=1, convolve_psf_ground_truth=False,
               resume_checkpoint=None, save_epoch_interval=10, cnn_device='cuda:0')
    for key in raw:
        if key.endswith('_loss_weight'):
            raw[key] = 1 if key == 'dice_loss_weight' else 0
    raw['accumulation_steps'] = int(raw['accumulation_steps'])
    if raw['accumulation_steps'] < 1:
        raise ValueError('accumulation_steps must be positive')
    (root / 'base_input.yaml').write_text(yaml.safe_dump(raw, sort_keys=False))
    (root / 'base_input.yaml').chmod(0o444)
    (root / 'smoke').mkdir()
    (root / 'smoke/input.yaml').write_text(yaml.safe_dump(raw, sort_keys=False))
    render(root, 'smoke', settings()['smoke_qos'], 120, 'smoke')
    save_json(root / 'manifest.json', dict(created=datetime.now().isoformat(), checkout=str(checkout),
              persistent_root=str(root), source_commit=(root/'git_commit.txt').read_text().strip(),
              scientific_notes=['561 nm excitation, existing AF555 emission-weighted detection PSF (one fluorophore/color)',
              'Existing sample_4f geometry, finite 1024 um grid, aberrations disabled; no calibrated chamber model',
              'laser_beam_FWHC is used by existing code as amplitude Gaussian sigma, not actual FWHM',
              'Camera per-image auto-exposure: percentile .999, target saturation ratio .01; not fixed dose',
              'Dice-only binary bead-center in-focus z target; no axial pairs or spatial-profile objective',
              'Matched seed/model init, deterministic epoch scene/noise streams; CUDA bitwise determinism not claimed'],
              requested_resources=dict(nodes=1,gpus=settings()['gres'],cpus=4,ram='32G',account=settings()['account'],partition=settings()['partition']),
              jobs={}))
    print(root)


def choose_plan(smoke):
    if not smoke.get('success') or len(smoke['timing']) != 2:
        raise RuntimeError('Smoke success and two full timed epochs are required')
    timing = smoke['timing']
    # Two smoke train/validation scenes, eight/four in each training arm.
    cycle = max(t['train_seconds'] * 4 + t['validation_seconds'] * 2 + t['checkpoint_seconds'] for t in timing)
    epochs = 3
    # Include setup, full validation/profile evaluation, and 15 minute boundary margin.
    evaluation = max(t['validation_seconds'] for t in timing) * 2 + 180
    requested = math.ceil((1.5 * epochs * cycle + evaluation + 900 + 300) / 900) * 900
    if requested > 6 * 3600:
        epochs = min(3, int((6*3600-evaluation-1200)/(1.5*cycle)))
        if epochs < 1:
            raise RuntimeError('No bounded matched pilot fits six hours per arm')
        requested = math.ceil((1.5*epochs*cycle+evaluation+1200)/900)*900
    return dict(epochs=epochs,ntrain=8,nvalid=4,particles=smoke['particles'],
                estimated_cycle_seconds=cycle,wall_seconds=max(1800,requested),margin_seconds=900,
                evaluation_reserve_seconds=evaluation)


def finalize(root):
    verify(root, 'smoke')
    smoke = json.loads((root / 'smoke/success.json').read_text())
    plan = choose_plan(smoke)
    save_json(root / 'plan.json', plan)
    (root / 'plan.json').chmod(0o444)
    base = yaml.safe_load((root / 'base_input.yaml').read_text())
    for name in ('A', 'B'):
        run = root / name
        run.mkdir()
        raw = {**base, 'freeze_slm': name == 'A', 'ntrain': plan['ntrain'], 'nvalid': plan['nvalid'],
               'max_epochs': plan['epochs'], 'num_particles_range': [plan['particles'],plan['particles']+1]}
        (run / 'input.yaml').write_text(yaml.safe_dump(raw, sort_keys=False))
        render(root, name, settings()['qos'], plan['wall_seconds']//60, 'arm')
    print(json.dumps(plan, indent=2))


def run_arm(root, name):
    from training.pilot import segment
    verify(root, name)
    smoke = json.loads((root / 'smoke/success.json').read_text())
    plan = json.loads((root / 'plan.json').read_text())
    if not smoke['success'] or plan != choose_plan(smoke):
        raise RuntimeError('Smoke/timing gate failed')
    if name == 'B' and not json.loads((root / 'A/success.json').read_text())['success']:
        raise RuntimeError('A success gate failed')
    result = segment(root / name / 'input.yaml', root / name, plan['epochs'],
                     plan['wall_seconds'] - int(plan['evaluation_reserve_seconds']),
                     margin=plan['margin_seconds'], evaluate=True)
    if result['epoch'] + 1 != plan['epochs']:
        raise RuntimeError('Matched epoch budget not completed; B must not start')
    match = json.loads((root / name / 'matching.json').read_text())
    if name == 'B':
        other = json.loads((root / 'A/matching.json').read_text())
        if other != match:
            raise RuntimeError('A/B initialization or validation scenes differ')
    save_json(root / name / 'success.json', dict(success=True,result=result,matching=match))


def continuation_command(script, predecessor, segment_limit, segment_index):
    if not isinstance(segment_limit, int) or segment_limit < 1:
        raise ValueError('Explicit positive user-authorized segment limit is required')
    if not 1 <= segment_index <= segment_limit or not str(predecessor).isdigit():
        raise ValueError('Invalid predecessor or continuation segment bound')
    return ['sbatch', '--parsable', '--kill-on-invalid-dep=yes',
            '--dependency=afterok:' + str(predecessor), str(script)]


def continue_arm(root, name):
    from training.pilot import segment
    verify(root, name)
    plan = json.loads((root / 'plan.json').read_text())
    # read_checkpoint fails if missing; strict config check and completed-run
    # no-op are performed before any model construction. Never start fresh.
    result = segment(root / name / 'input.yaml', root / name, plan['epochs'],
                     seconds=7200, margin=900, resume=True)
    save_json(root / name / 'continuation_result.json', result)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('action', choices=['prepare','verify','finalize','run','continue'])
    p.add_argument('--root', required=True)
    p.add_argument('--arm', choices=['smoke','A','B'])
    args = p.parse_args()
    root = Path(args.root).resolve()
    if args.action == 'prepare':
        prepare(root, Path.cwd())
    elif args.action == 'verify':
        verify(root, args.arm)
    elif args.action == 'finalize':
        finalize(root)
    elif args.action == 'run':
        run_arm(root, args.arm)
    else:
        continue_arm(root, args.arm)


if __name__ == '__main__':
    main()
