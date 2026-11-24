"""Prepare an explicitly bounded frozen-Gaussian control/class-weight study."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import yaml


def configurations(base,hours=8):
    if not 0<hours<=24:raise ValueError('Explicit per-run training hours must be in (0,24]')
    common=copy.deepcopy(base)
    common.update(initial_phase_mask='gaussian',gaussian_phase_mode='focused',phase_mask_file=None,
        phase_mask_input_units=None,freeze_slm=True,freeze_unet=False,resume_checkpoint=None,
        resume_reset_early_stopping=False,num_particles_range=[1200,1201],batch_size=1,
        accumulation_steps=32,ntrain=128,nvalid=32,z_depth_list=[0],
        emission_strength_range=[.5,1.5],tissue_sample_z_bounds_um=[-100,100],
        ce_loss_weight=1,dice_loss_weight=1,mae_loss_weight=0,mse_loss_weight=0,
        tv_loss_weight=0,spatial_multi_well_loss_weight=0,tissue_spatial_loss_weight=0,
        minimum_training_seconds=hours*3600,maximum_training_seconds=hours*3600,
        max_epochs=100000,random_seed=21,training_seed=211,validation_seed=212)
    for k in list(common):
        if k.startswith('tissue_') and k not in ['tissue_sample_z_bounds_um','tissue_spatial_loss_weight']:common.pop(k)
    runs={}
    for name in ['binary_bce_dice','one_sided','two_sided']:
        raw=copy.deepcopy(common);raw['num_classes']=1
        if name!='binary_bce_dice':
            raw.update(tissue_spatial_loss_weight=.01,tissue_z_bounds_um=[0,100] if name=='one_sided' else [-100,100],
                tissue_sample_z_bounds_um=[0,100] if name=='one_sided' else [-100,100],
                tissue_profile_x_um=[-400,-200,0,200,400],tissue_profile_z_bounds_um=[-150,150],
                tissue_core_halfwidth_um=5,tissue_core_min_fraction=.8,tissue_core_floor_weight=10,
                tissue_core_uniformity_weight=0)
        runs[name]=raw
    for bg in [.05,.1,.5]:
        for connection in [.5,2,4]:
            name='between_bg%03d_conn%03d'%(round(bg*100),round(connection*100))
            raw=copy.deepcopy(common);raw.update(num_classes=3,ce_class_weights=[bg,1.,connection]);runs[name]=raw
    return runs


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root,base_config,schedule,account,code,python=sys.executable,hours=8):
    root=Path(root).resolve();code=Path(code).resolve();schedule=dict(schedule)
    configs=configurations(yaml.safe_load(Path(base_config).read_text()),hours)
    if set(schedule)!=set(configs):raise ValueError('Each of the twelve runs needs an explicit resource assignment')
    if sum(int(r['count']) for r in schedule.values() if r['qos']=='p_bme_gpu')>2:
        raise ValueError('Submitted BME concurrency must not exceed two GPUs')
    runs={};q=shlex.quote
    for name,raw in configs.items():
        run=root/'runs'/name;run.mkdir(parents=True,exist_ok=False);request=schedule[name]
        if request['qos'] not in ['short_gpu','p_bme_gpu','scavenger_gpu'] or request['gpu'] not in ['l40s','h100','h200'] or request['count']!=1:
            raise ValueError('Frozen study uses one explicitly validated GPU per run')
        path=run/'input.yaml';path.write_text(yaml.safe_dump(raw,sort_keys=False));path.chmod(0o444)
        short=request['qos']=='short_gpu';limit=7 if short else 1;wall=7200 if short else int(hours*3600*1.35+1800)
        plan=dict(segment_limit=limit,config_sha256=digest(path),expected_gpu_count=1,source_snapshot=str(code),requested_gpu_type=request['gpu'],maximum_preemptions=3,minimum_actual_training_seconds=hours*3600)
        (run/'segment_plan.json').write_text(json.dumps(plan,indent=2)+'\n');(run/'segment_plan.json').chmod(0o444);scripts=[]
        for index in range(1,limit+1):
            script=run/('segment_%02d.sbatch'%index);clock='%02d:%02d:%02d'%(wall//3600,wall%3600//60,wall%60)
            script.write_text(f'''#!/bin/bash
#SBATCH --job-name=dd_fg_{name}_{index:02d}
#SBATCH --account={account}
#SBATCH --partition=gpu_partners
#SBATCH --qos={request['qos']}
#SBATCH --gres=gpu:{request['gpu']}:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time={clock}
#SBATCH --exclude=gpu19
#SBATCH --requeue
#SBATCH --output={run}/segment_{index:02d}.%j.out
#SBATCH --error={run}/segment_{index:02d}.%j.err
set -euo pipefail
export PYTHONPATH={q(str(code))} PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR={q(str(run/'cache/mpl'))} XDG_CACHE_HOME={q(str(run/'cache/xdg'))} TORCH_HOME={q(str(run/'cache/torch'))}
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME/torch/kernels" "$TORCH_HOME"
cd {q(str(code))}
source scripts/slurm/worker_runtime.sh
{q(python)} {q(str(root/'verify_source.py'))}
{q(python)} -c {q("import json; assert json.load(open("+repr(str(root/'smoke_success.json'))+"))['success'] is True")}
{q(python)} -m utils.worker_preflight --report {q(str(run/('preflight_%02d.json'%index)))}
{q(python)} -u -m training.segments run --run-dir {q(str(run))} --segment-index {index} --segment-limit {limit} --wall-seconds {wall-300} --margin-seconds 900
''')
            subprocess.run(['bash','-n',str(script)],check=True);script.chmod(0o444);scripts.append(str(script))
        runs[name]=dict(run_dir=str(run),config_sha256=digest(path),resources=request,job_ids=[],scripts=scripts,target_actual_train_hours=hours,max_allocation_hours_per_attempt=wall*limit/3600,class_weights=raw.get('ce_class_weights') if raw['num_classes']==3 else None)
    manifest=dict(runs=runs,source_snapshot=str(code),gaussian='Focused quadratic Gaussian, not empty/flat or learned phase',
        shared_settings='1200 emitters; batch1; accumulation32; train128/valid32; one view; uniform strength[0.5,1.5]; detector trainable; phase frozen',
        grid='CE weights [background,bead,connection]: {0.05,0.1,0.5} × {1} × {0.5,2,4}; multiclass Dice weighting unchanged',
        spatial_interpretation='Frozen spatial penalties are constant offsets with no direct detector gradient; total validation loss still drives the existing relative-threshold LR scheduler',
        training_budget='8 actual train-loop hours per run; validation/setup/save/queue waits separate; safe complete-epoch overshoot allowed')
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ['root','base-config','schedule','account','code']:p.add_argument('--'+name,required=True)
    p.add_argument('--hours',type=float,required=True);a=p.parse_args()
    prepare(a.root,a.base_config,json.loads(Path(a.schedule).read_text()),a.account,a.code,hours=a.hours)
