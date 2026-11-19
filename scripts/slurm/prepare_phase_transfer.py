"""Prepare five bounded phase-transfer/multiview runs; never submits jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import numpy as np
import yaml
from physics.phase_io import save_radians
from training.checkpoint import read_checkpoint


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root,previous_study,code,account,schedule,gate,python=sys.executable):
    root=Path(root).resolve();code=Path(code).resolve();previous_study=Path(previous_study).resolve()
    runs={}
    for initialization in ['gaussian','random','axicon','airy','between_random']:
        between=initialization=='between_random'
        name='between_random_3view' if between else 'phase_'+initialization
        run=root/'runs'/name;run.mkdir(parents=True,exist_ok=False)
        source=previous_study/'runs'/('between_beads_random' if between else 'one_sided_'+initialization)
        raw=yaml.safe_load((source/'input.yaml').read_text())
        hours=6 if between else 4
        raw.update(emission_strength_range=[.5,1.5],tissue_sample_z_bounds_um=[-100,100],
            tissue_spatial_loss_weight=0,spatial_multi_well_loss_weight=0,num_classes=3 if between else 1,
            ce_loss_weight=1,dice_loss_weight=1,mae_loss_weight=0,mse_loss_weight=0,tv_loss_weight=0,
            minimum_training_seconds=hours*3600,maximum_training_seconds=hours*3600,
            max_epochs=100000,resume_checkpoint=None,resume_reset_early_stopping=False,
            freeze_unet=False,freeze_slm=False)
        # Keep sample support explicitly separate from optional optical penalty geometry.
        if not between:
            for key in list(raw):
                if key.startswith('tissue_') and key not in ['tissue_sample_z_bounds_um','tissue_spatial_loss_weight']:raw.pop(key)
        radius_px=raw['bead_radius']/(raw['slm_px']/raw['phase_mask_upsample_factor']*1e6)
        raw['z_depth_list']=[-radius_px,0,radius_px] if between else [0]
        provenance=dict(source_input=str(source/'input.yaml'),source_input_sha256=digest(source/'input.yaml'),
            phase_only_transfer=not between,fresh_detector=True,fresh_optimizer=True)
        if between:raw['initial_phase_mask']='random'
        else:
            checkpoint=source/'checkpoints/latest.pt';state=read_checkpoint(checkpoint)
            phase=state['raw_phase'].numpy().astype(np.float32)
            if phase.shape!=(raw['phase_mask_pixel_size'],)*2:raise ValueError('Phase dimensions differ')
            phase_path=run/'initial_phase_radians.tif';save_radians(phase,phase_path)
            raw.update(initial_phase_mask='file',phase_mask_file=str(phase_path),phase_mask_input_units='radians')
            provenance.update(source_checkpoint=str(checkpoint),source_checkpoint_sha256=digest(checkpoint),
                source_completed_epochs=state['epoch']+1,phase_sha256=digest(phase_path))
        config=run/'input.yaml';config.write_text(yaml.safe_dump(raw,sort_keys=False));config.chmod(0o444)
        assignment=schedule[name];qos=assignment['qos'];gpu=assignment['gpu'];count=int(assignment['count'])
        if qos not in ['short_gpu','scavenger_gpu','p_bme_gpu']:raise ValueError('Unsupported explicit queue')
        if gpu not in ['l40s','h100','h200'] or count not in [1,2]:raise ValueError('Unvalidated GPU request')
        if qos=='p_bme_gpu' and count>2:raise ValueError('BME aggregate limit is two GPUs')
        # The short queue gets finite two-hour afterok segments. Longer scavenger
        # allocations include validation/save overhead and resume only on Slurm requeue.
        limit=(5 if between else 3) if qos=='short_gpu' else 1
        wall=7200 if qos=='short_gpu' else int(hours*3600*1.35+1800)
        plan=dict(segment_limit=limit,config_sha256=digest(config),expected_gpu_count=count,
            source_snapshot=str(code),requested_gpu_type=gpu,minimum_actual_training_seconds=hours*3600,
            maximum_preemptions=3)
        (run/'segment_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        (run/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        scripts=[]
        for index in range(1,limit+1):
            script=run/('segment_%02d.sbatch'%index);q=shlex.quote
            clock='%02d:%02d:%02d'%(wall//3600,wall%3600//60,wall%60)
            script.write_text(f'''#!/bin/bash
#SBATCH --job-name=dd_xfer_{name}_{index:02d}
#SBATCH --account={account}
#SBATCH --partition=gpu_partners
#SBATCH --qos={qos}
#SBATCH --gres=gpu:{gpu}:{count}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time={clock}
{('#SBATCH --exclude='+assignment['exclude_nodes']) if assignment.get('exclude_nodes') else ''}
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
{q(python)} -c {q("import json; assert json.load(open("+repr(str(gate))+"))['success'] is True")}
{q(python)} -m utils.worker_preflight --report {q(str(run/('preflight_%02d.json'%index)))}
{q(python)} -u -m training.segments run --run-dir {q(str(run))} --segment-index {index} --segment-limit {limit} --wall-seconds {wall-300} --margin-seconds 900
''')
            subprocess.run(['bash','-n',str(script)],check=True);script.chmod(0o444);scripts.append(str(script))
        runs[name]=dict(run_dir=str(run),scripts=scripts,job_ids=[],resources=assignment,
            target_training_hours=hours,max_allocation_hours=limit*wall/3600,config_sha256=digest(config),provenance=provenance)
    manifest=dict(runs=runs,code=str(code),gate=str(gate),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        scientific_changes='Relative per-emitter uniform[0.5,1.5]; actual support[-100,+100]um; BCE+Dice-only phase transfer; fresh three-class three-view random control',
        time_accounting='Actual train-loop hours; validation/save/setup and queue waits reported separately; final epoch may slightly exceed target')
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ['root','previous-study','code','account','schedule','gate']:p.add_argument('--'+key,required=True)
    a=p.parse_args();prepare(a.root,a.previous_study,a.code,a.account,json.loads(Path(a.schedule).read_text()),a.gate)
