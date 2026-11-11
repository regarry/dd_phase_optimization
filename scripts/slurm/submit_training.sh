#!/bin/bash
# Run from the repository root in the configured Python environment.
set -euo pipefail
DD_DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DD_DRY_RUN=1
    shift
fi
DD_CONFIG="${1:-config.yaml}"
DD_PYTHON="${DD_PYTHON:-python}"
export DD_PYTHON
mkdir -p temp_configs logs
DD_INPUT=$(mktemp "$PWD/temp_configs/config.XXXXXXXXXX.yaml")
DD_SCRIPT="${DD_INPUT%.yaml}.sbatch"
cp -- "$DD_CONFIG" "$DD_INPUT"
"$DD_PYTHON" - "$DD_INPUT" "$DD_SCRIPT" <<'PY'
import os,shlex,sys
from pathlib import Path
from scripts.slurm.site import directives
wall=os.environ.get('DD_SLURM_TIME','24:00:00')
days,clock=wall.split('-',1) if '-' in wall else ('0',wall)
parts=list(map(int,clock.split(':')))
if len(parts)!=3 or any(v<0 for v in parts) or parts[1]>=60 or parts[2]>=60:
    raise ValueError('DD_SLURM_TIME must be [D-]HH:MM:SS')
seconds=int(days)*86400+parts[0]*3600+parts[1]*60+parts[2]
if seconds<=1800:raise ValueError('Training request must exceed 30 minutes to include setup and stop margins')
text=Path('scripts/slurm/train_job.sbatch').read_text()
text=text.replace('#SBATCH --time=24:00:00','#SBATCH --time='+wall).replace('#SBATCH --gres=gpu:2\n','')
text=text.replace('#!/bin/bash\n','#!/bin/bash\n'+directives()+'\n',1)
text=text.replace('__CONFIG_FILE__',shlex.quote(sys.argv[1])).replace('__WALL_SECONDS__',str(seconds-900))
Path(sys.argv[2]).write_text(text)
PY
chmod a-w "$DD_INPUT" "$DD_SCRIPT"
bash -n "$DD_SCRIPT"
if [[ "$DD_DRY_RUN" == 1 ]]; then
    cat "$DD_SCRIPT"
    exit 0
fi
DD_SBATCH_OPTIONS=()
for DD_SETTING in ACCOUNT PARTITION QOS GRES TIME; do
    DD_KEY="DD_SLURM_${DD_SETTING}"
    if [[ -n "${!DD_KEY:-}" ]]; then
        DD_SBATCH_OPTIONS+=("--${DD_SETTING,,}=${!DD_KEY}")
    fi
done
sbatch "${DD_SBATCH_OPTIONS[@]}" "$DD_SCRIPT"
