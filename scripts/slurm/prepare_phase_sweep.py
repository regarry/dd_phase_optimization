"""Prepare one benchmark and three serial phase runs; never submit jobs here."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

import yaml

from scripts.slurm.site import directives, module_setup, settings

PYTHON=settings()['python']


def prepare(checkout, config_path):
    checkout=Path(checkout).resolve()
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    root=checkout/'training_results'/('relaunch_'+stamp)
    root.mkdir(parents=True,exist_ok=False)
    base_bytes=Path(config_path).read_bytes()
    base=yaml.safe_load(base_bytes)
    if base.get('resume_checkpoint'):
        raise ValueError('A new initialization sweep requires resume_checkpoint: null')
    if base.get('accumulation_steps',0)<1 or not any(v for k,v in base.items() if k.endswith('_loss_weight')):
        raise ValueError('Explicit positive accumulation_steps and a nonzero loss are required')
    (root/'base_config.yaml').write_bytes(base_bytes)
    code=root/'code';code.mkdir()
    files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=checkout).decode().split('\0')
    for name in files:
        source=checkout/name
        if not name or not source.is_file():continue
        target=code/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);target.chmod(0o444)
    for command,name in ((['git','diff','--binary','HEAD'],'local.diff'),
                         (['git','status','--short'],'git_status.txt'),
                         (['git','rev-parse','HEAD'],'git_commit.txt'),
                         (['git','branch','--show-current'],'git_branch.txt')):
        (root/name).write_bytes(subprocess.check_output(command,cwd=checkout))
    (root/'verify.py').write_text('''import hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
for name,sha in json.loads((root/'checksums.json').read_text()).items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha:
        raise RuntimeError('Code/config/script changed: '+name)
if '--benchmark' in sys.argv:
    proof=json.loads((root/'benchmark_success.json').read_text())
    if not proof['success'] or not Path(proof['checkpoint']).is_file():
        raise RuntimeError('Benchmark success/checkpoint gate failed')
    if hashlib.sha256(Path(proof['checkpoint']).read_bytes()).hexdigest()!=proof['checkpoint_sha256']:
        raise RuntimeError('Benchmark checkpoint changed')
print('Immutable snapshot and requested benchmark gate verified.',flush=True)
''')
    records={}
    for name in ('benchmark','random','gaussian','axicon'):
        run=checkout/'training_results'/(stamp+'_'+name)
        run.mkdir(exist_ok=False)
        (run/'mpl_cache').mkdir()
        raw=dict(base)
        raw['initial_phase_mask']='random' if name=='benchmark' else name
        if name=='benchmark':raw.update(ntrain=32,nvalid=8,max_epochs=1)
        input_path=run/'input.yaml'
        input_path.write_text(yaml.safe_dump(raw,sort_keys=False))
        q=shlex.quote
        minutes=20 if name=='benchmark' else 1440
        seconds=1140 if name=='benchmark' else 85500
        margin=120 if name=='benchmark' else 900
        gate='' if name=='benchmark' else ' --benchmark'
        benchmark_run=checkout/'training_results'/(stamp+'_benchmark')
        proof_command='' if name=='benchmark' else (
            q(PYTHON)+' -m scripts.slurm.check_training_benchmark --root '+q(str(root))+
            ' --run '+q(str(benchmark_run))+'\n')
        script=f'''#!/bin/bash
#SBATCH --job-name=dd_{name}_verified
{directives()}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time={minutes//60:02d}:{minutes%60:02d}:00
#SBATCH --output={run}/stdout.%j.log
#SBATCH --error={run}/stderr.%j.log
set -euo pipefail
{module_setup()}
export PYTHONPATH={q(str(code))}
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLCONFIGDIR={q(str(run/'mpl_cache'))}
cd {q(str(code))}
source ./scripts/slurm/worker_runtime.sh
{proof_command.rstrip()}
{q(PYTHON)} {q(str(root/'verify.py'))}{gate}
{q(PYTHON)} -m utils.worker_preflight
hostname
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv
{q(PYTHON)} -u -m training.training --config {q(str(input_path))} --run-dir {q(str(run))} \\
    --worker-timeout 60 --progress-every 32 --require-phase-gradient \\
    --wall-seconds {seconds} --stop-margin-seconds {margin}
'''
        script_path=run/'submitted.sbatch';script_path.write_text(script)
        subprocess.run(['bash','-n',str(script_path)],check=True)
        records[name]=dict(run_dir=str(run),input=str(input_path),script=str(script_path),
                           job_id=None,checkpoint=str(run/'checkpoints/latest.pt'),
                           wall_time_minutes=minutes,scientific_overrides={k:v for k,v in raw.items() if base.get(k)!=v})
    checks={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code.rglob('*') if p.is_file()}
    for path in [root/'base_config.yaml',root/'verify.py']+[Path(d[k]) for d in records.values() for k in ('input','script')]:
        # Run folders are siblings of the control root; resolve ../ paths here.
        import os
        checks[os.path.relpath(path,root)]=hashlib.sha256(path.read_bytes()).hexdigest()
        path.chmod(0o444)
    (root/'checksums.json').write_text(json.dumps(checks,indent=2)+'\n')
    (root/'checksums.json').chmod(0o444)
    manifest=dict(root=str(root),created=datetime.now().isoformat(),runs=records,
                  base_config_sha256=hashlib.sha256(base_bytes).hexdigest(),
                  resources=dict(nodes=1,gpus=settings()['gres'],cpus=4,host_ram='32G',qos=settings()['qos'],account=settings()['account'],partition=settings()['partition']),
                  benchmark='32 train / 8 validation; all optical/loss/particle/optimizer settings unchanged',
                  production='Full input counts and settings; only initialization varies; serial one-day jobs')
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return root


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',default='config.yaml')
    args=p.parse_args();print(prepare(Path.cwd(),args.config))
