#!/bin/bash
#BSUB -J gpu_diag
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=2GB]"
#BSUB -q bme_gpu
#BSUB -gpu "num=1:mode=exclusive_process:mps=no"
#BSUB -o ./logs/diag.%J.out
#BSUB -e ./logs/diag.%J.err
#BSUB -cwd /rsstu/users/a/agrinba/DeepDesign/deepdesign/dd_phase_optimization

set -euo pipefail

TS=$(date +%Y%m%d-%H%M%S)
OUTDIR=./logs/diagnostics_$TS
mkdir -p "$OUTDIR"

echo "JOBID: $LSB_JOBID" > "$OUTDIR/summary.txt"
echo "HOST: $(hostname)" >> "$OUTDIR/summary.txt"
echo "USER: $(whoami)" >> "$OUTDIR/summary.txt"
echo "PWD: $(pwd)" >> "$OUTDIR/summary.txt"
date >> "$OUTDIR/summary.txt"

# environment and modules
echo "=== environment variables (CUDA/LSF related) ===" > "$OUTDIR/env.txt"
env | egrep -i 'cuda|nvidia|lsf|ld_library_path|XDG_RUNTIME_DIR|CUDA_VISIBLE_DEVICES' >> "$OUTDIR/env.txt" || true
echo "=== module list ===" > "$OUTDIR/modules.txt"
module list 2>&1 | tee -a "$OUTDIR/modules.txt" || true

# lsbatch file check
echo "=== .lsbatch files ===" > "$OUTDIR/lsbatch.txt"
ls -l /home/$USER/.lsbatch* 2>/dev/null | sed -n '1,200p' >> "$OUTDIR/lsbatch.txt" || true
stat /home/$USER/.lsbatch/* 2>/dev/null || true

# nvidia-smi outputs
nvidia-smi -L > "$OUTDIR/nvidia_smi_list.txt" 2>&1 || true
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total --format=csv,noheader > "$OUTDIR/nvidia_smi_gpus.csv" 2>&1 || true
nvidia-smi -q > "$OUTDIR/nvidia_smi_q.txt" 2>&1 || true
nvidia-smi -q -d ECC > "$OUTDIR/nvidia_smi_ecc.txt" 2>&1 || true
nvidia-smi --query-compute-apps=pid,gpu_index,process_name,used_gpu_memory --format=csv,noheader > "$OUTDIR/nvidia_smi_compute_apps.csv" 2>&1 || true

# kernel messages (NVRM / ECC)
dmesg | egrep -i 'NVRM|ecc|GPU|nvidia' | sed -n '1,200p' > "$OUTDIR/dmesg_nvidia.txt" || true

# small python diagnostic using your conda env (will capture torch errors)
PYFILE="$OUTDIR/torch_diag.py"
cat > "$PYFILE" <<'PY'
import os, subprocess, traceback
try:
    import torch
    print("torch imported:", torch.__version__)
except Exception as e:
    print("torch import failed:", e)
print("PID", os.getpid())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.is_available():", getattr(__import__('torch'), 'cuda').is_available() if 'torch' in globals() else "torch missing")
if 'torch' in globals():
    try:
        cur = torch.cuda.current_device()
        print("current_device:", cur)
        print("device_count:", torch.cuda.device_count())
        print("device_name:", torch.cuda.get_device_name(cur))
        try:
            print("device_properties:", torch.cuda.get_device_properties(cur))
        except Exception as e:
            print("get_device_properties failed:", e)
    except Exception:
        traceback.print_exc()
# try a synchronous allocation test
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
try:
    x = torch.randn(1024,1024, device='cuda')
    print("allocation OK")
except Exception as e:
    print("allocation error:", repr(e))
# nvidia-smi compute apps
try:
    out = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_index,process_name,used_gpu_memory","--format=csv,noheader"])
    print("nvidia-smi compute apps:\\n", out.decode())
except Exception as e:
    print("nvidia-smi compute apps failed:", e)
PY

# run the python diagnostic inside your conda env (use conda run)
# change the -p path below if your env path differs
CONDAPATH=/rsstu/users/a/agrinba/DeepDesign/deepdesign
echo "Running Python diagnostic (conda env at $CONDAPATH)..."
conda run -p "$CONDAPATH" --no-capture-output python -u "$PYFILE" > "$OUTDIR/torch_diag.txt" 2>&1 || true

# final tarball for easy attachment to admin ticket
TARFILE=./logs/diagnostics_${TS}_${LSB_JOBID:-nojob}.tar.gz
tar -czf "$TARFILE" -C "$OUTDIR" . || true

echo "Diagnostics collected in $OUTDIR and tarball $TARFILE"