# Continue on another computer

This change set contains reusable source, tests and technical documentation.
Private run inputs, conversations, local allocation details, generated datasets,
checkpoints, reports and machine-specific environment dumps are not part of it.
The existing repository history is not rewritten by this change set. Legacy local IDE settings, research notes, machine environment exports, an optimized-mask data file and the bundled figure image are removed from this review branch only. The active workspace keeps them.

## Python environment

Use a dedicated environment with compatible NumPy, SciPy, PyTorch, torchvision,
scikit-image, tifffile, matplotlib and PyYAML. Install pytest for CPU checks.
A tested compatibility baseline is Python 3.8 and PyTorch 2.4 with CUDA 12.1;
choose a PyTorch build compatible with the target GPU/driver rather than assuming
all hardware supports that build. The explicit requirements list records tested package versions without local paths; it is not a hardware/driver lockfile.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q tests
```

The editable install includes `utils` and the script packages, which are imported
by training and operational entry points. The historical README/environment
exports are not dependency locks. GPU/optical correctness requires a small
allocation-side smoke test after moving machines.

## Config and outputs

Keep site credentials and local environment variables out of Git. Use
`configs/example_training.yaml` as an explicitly synthetic small-workload example,
then choose optical settings, nonzero loss weights and resource bounds for the
actual task. The historical root configuration is preserved except for the new
optional pair-jitter key; it is not an approved experiment specification.

Outputs belong in Git-ignored `training_results/` or another caller-selected
persistent run directory. On a new computer, a source checkout has no generated
PSF data, trained weights or captured evaluations. Generate PSF/templates through
the training entry point; only evaluate a checkpoint you deliberately transfer
separately. Resume must preserve checkpoint scientific settings and the expected
run directory; moving complete existing runs needs an explicit path migration
plan, not a fresh-start fallback.

## Local Slurm configuration

Activate the project environment before preparing scripts. Configure these
variables locally, using the values authorized by the target cluster:

```bash
export DD_PYTHON="$(command -v python)"
export DD_SLURM_ACCOUNT="YOUR_ACCOUNT"
export DD_SLURM_PARTITION="YOUR_GPU_PARTITION"
export DD_SLURM_QOS="YOUR_TRAINING_QOS"
export DD_SLURM_SMOKE_QOS="YOUR_SHORT_QOS"
export DD_SLURM_GRES="gpu:2"
export DD_SLURM_TIME="24:00:00"
# Optional site module names; omit when the activated environment is sufficient.
export DD_MODULES=""
bash scripts/slurm/submit_training.sh --dry-run configs/example_training.yaml
```

Omitted account/partition/QoS use Slurm's local defaults; verify them before
submitting. Typed GPUs may require a site-specific `DD_SLURM_GRES`. The rendered
Python preparation tools store the resolved interpreter/modules/site directives
in ignored run artifacts; these local values are not committed source.
The shell launcher's optional resource overrides are applied at submission.
`DD_SLURM_TIME` accepts `[D-]HH:MM:SS`, must exceed 30 minutes, and determines the
runtime wall budget minus a setup reserve. Preserve enough time for a full cycle
and stop margin. `--dry-run` creates config/script snapshots but submits nothing.
Remove `--dry-run` only when job submission and resources are authorized.

The launcher references `scripts/slurm/train_job.sbatch`, runs training as a
module, preserves the copied config, and uses short node-local worker IPC paths.
The tiny CPU-only transfer check may need to run outside an execution sandbox
that isolates IPC; a host pass does not validate another compute node.

## Review scope

Core additions: explicit Gaussian initialization; optional paired-bead lateral
jitter; runtime progress/gradient/cycle/checkpoint checks; bounded worker startup;
portable Slurm preparation; matched pilot and continuation gates; trained
checkpoint evaluation; reusable saved-artifact analysis. No changes to the
optical forward equations, camera exposure model, historical checkpoint format
or unrelated experiment settings are part of this preparation.

CPU tests establish software contracts, not full CUDA or instrument correctness.
No scripts in this package schedule training automatically on import.

The optional benchmark monitor is read-only unless DD_GPU_CAPACITY_HELPER is explicitly configured for a site-specific free-capacity table. Its typed GPU choices and capacity parsing must be validated at the destination; it is not a portable allocation discovery service.

The legacy run-comparison helper now lives at scripts/parse_runs.py instead of inside the ignored results directory.
