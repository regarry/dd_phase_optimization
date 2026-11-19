"""Prepare a finite parallel study; submission is an explicit login-node action."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root,base_config,account,gate_job,python=None,segments=10):
    root=Path(root).resolve();python=python or sys.executable
    if not 1<=segments<=10:raise ValueError('Study permits at most ten segments per run')
    if not str(gate_job).isdigit():raise ValueError('Explicit production gate job is required')
    code=root/'code'
    if not (code/'training/segments.py').is_file():raise ValueError('Freeze the tested segment implementation first')
    base=yaml.safe_load(Path(base_config).read_text());runs={};q=shlex.quote
    for approach in ['two_sided','one_sided','between_beads']:
        for initialization in ['gaussian','random','axicon','airy']:
            name=approach+'_'+initialization;run=root/'runs'/name
            run.mkdir(parents=True,exist_ok=False)
            raw=dict(base)
            raw.update(num_particles_range=[1200,1201],image_volume=[128,1000,256],
                ntrain=128,nvalid=32,batch_size=1,accumulation_steps=32,
                max_epochs=100000,save_epoch_interval=10,minimum_training_seconds=43200,
                early_stopping_patience=20,early_stopping_min_delta=.001,
                random_seed=21,training_seed=211,validation_seed=212,
                initial_phase_mask=initialization,gaussian_phase_mode='focused',airy_alpha=1e10,
                freeze_unet=False,freeze_slm=False,resume_checkpoint=None,
                convolve_psf_ground_truth=False,ce_loss_weight=1,dice_loss_weight=1,
                mae_loss_weight=0,mse_loss_weight=0,tv_loss_weight=0,
                spatial_multi_well_loss_weight=0,z_coupled_lateral_jitter_px=3)
            if approach=='between_beads':
                raw.update(num_classes=3,tissue_spatial_loss_weight=0)
            else:
                raw.update(num_classes=1,tissue_spatial_loss_weight=.01,
                    tissue_profile_x_um=[-400,-200,0,200,400],
                    tissue_profile_z_bounds_um=[-150,150],
                    tissue_z_bounds_um=[0,100] if approach=='one_sided' else [-100,100],
                    tissue_sample_z_bounds_um=[0,100] if approach=='one_sided' else [-50,50],
                    tissue_core_halfwidth_um=5,tissue_core_min_fraction=.8,
                    tissue_core_floor_weight=10,tissue_core_uniformity_weight=0)
            input_path=run/'input.yaml';input_path.write_text(yaml.safe_dump(raw,sort_keys=False))
            plan=dict(segment_limit=segments,config_sha256=sha(input_path),
                      expected_gpu_count=2,source_snapshot=str(code),
                      requested_gpu_type='l40s',minimum_actual_training_seconds=43200)
            (run/'segment_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
            scripts=[]
            for index in range(1,segments+1):
                path=run/('segment_%02d.sbatch'%index)
                path.write_text(f'''#!/bin/bash
#SBATCH --job-name=dd_{name[:20]}_{index:02d}
#SBATCH --account={account}
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output={run}/segment_{index:02d}.%j.out
#SBATCH --error={run}/segment_{index:02d}.%j.err
set -euo pipefail
export PYTHONPATH={q(str(code))}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR={q(str(run/'cache/mpl'))}
export XDG_CACHE_HOME={q(str(run/'cache/xdg'))}
export TORCH_HOME={q(str(run/'cache/torch'))}
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME/torch/kernels" "$TORCH_HOME"
cd {q(str(code))}
source scripts/slurm/worker_runtime.sh
{q(python)} {q(str(root/'verify_source.py'))}
{q(python)} -m utils.worker_preflight --report {q(str(run/('preflight_%02d.json'%index)))}
{q(python)} -u -m training.segments run --run-dir {q(str(run))} --segment-index {index} --segment-limit {segments} --wall-seconds 6900 --margin-seconds 900
''')
                subprocess.run(['bash','-n',str(path)],check=True);path.chmod(0o444);scripts.append(str(path))
            for p in [input_path,run/'segment_plan.json']:p.chmod(0o444)
            runs[name]=dict(run_dir=str(run),scripts=scripts,job_ids=[],
                gate_job=int(gate_job),config_sha256=sha(input_path),
                minimum_actual_training_hours=12,maximum_allocation_hours=2*segments)
    manifest=dict(runs=runs,source_snapshot=str(code),gate_job=int(gate_job),
        runtime_training_counts=dict(train=128,validation=32,emitters=1200),
        progression='actual train seconds, scenes, emitter presentations and optimizer updates; validation separate',
        stopping='no convergence stop before 12 actual training hours; ten allocations maximum; unmet minimum is an error',
        initials='Gaussian with explicit configured quadratic focus, random, axicon, nondegenerate cubic-phase Airy alpha1e10 rad/m^3',
        geometry='centered nominal100um with200um translation-safe exclusion; one-sided100um; between-bead three-class baseline')
    (root/'study_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def submit(root):
    """Finite afterok chains; persist each ID before the next submission."""
    root=Path(root).resolve();path=root/'study_manifest.json';manifest=json.loads(path.read_text())
    proof=json.loads((root/'production_gate_success.json').read_text())
    if proof.get('success') is not True or proof.get('particles')!=1200:raise RuntimeError('Verified production checkpoint gate required')
    # The checkpoint gate is authoritative even if a post-training receipt wrapper
    # needed repair. Every first script independently verifies its artifact proof.
    # Any interruption leaves an idempotent ledger; never submit an existing slot.
    for name,run in manifest['runs'].items():
        predecessor=run['job_ids'][-1] if run['job_ids'] else manifest['gate_job']
        for script in run['scripts'][len(run['job_ids']):]:
            dependency=('afterany:' if not run['job_ids'] else 'afterok:')+str(predecessor)
            response=subprocess.check_output(['sbatch','--parsable','--kill-on-invalid-dep=yes',
                    '--dependency='+dependency,script],text=True).strip()
            job=int(response.split(';')[0]);run['job_ids'].append(job);predecessor=job
            temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(manifest,indent=2)+'\n');temporary.replace(path)
            print(name,job,flush=True)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['prepare','submit'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--base-config',type=Path)
    p.add_argument('--account');p.add_argument('--gate-job');p.add_argument('--segments',type=int,default=10)
    a=p.parse_args()
    if a.action=='prepare':
        if not all([a.base_config,a.account,a.gate_job]):p.error('Preparation requires base config, account and gate job')
        prepare(a.root,a.base_config,a.account,a.gate_job,segments=a.segments)
    else:submit(a.root)
