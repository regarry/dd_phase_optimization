# Phase transfer, heterogeneous beads and three focal views

Use a previous complete `checkpoints/latest.pt` as a trusted source of **raw phase
only**, not as a resume checkpoint when changing the scientific experiment.
`scripts/slurm/prepare_phase_transfer.py` exports the phase through
`physics.phase_io.save_radians`, copies the new immutable YAML and records the
source checkpoint/input hashes and epoch. The detector, optimizer and schedules
start fresh. The four phase-transfer arms share seeds, detector initialization,
1200 emitters, binary BCE weight 1 plus Dice weight 1, and zero spatial penalties.
The script takes explicit paths, account, schedule JSON and a CUDA success gate;
it prepares files without submitting them.

`tissue_sample_z_bounds_um: [-100,100]` restricts bead **centres** in thickness z,
not propagation x. This is a 200 µm fluorescent support and intentionally differs
from historical symmetric runs whose actual bead support was only ±50 µm.
The camera geometry, noise, exposure, bead radius and other controls are inherited
from each source input. Robust percentile auto-exposure remains enabled; these
are not fixed-dose comparisons.

`emission_strength_range: [0.5,1.5]` independently samples uniform relative photon
multipliers (mean 1) per bead after target construction. A scene is `[E,4]` with
columns `[x_pixel,y_pixel,z_pixel,strength]`; legacy `[E,3]` means strength 1.
The optional default `[1,1]` performs no random draw and preserves old scene RNG.
Strength multiplies illumination intensity before PSF accumulation/camera noise.
All focal views use the same coordinates and strengths; cached validation and
complete checkpoints persist both. Binary/multiclass labels ignore brightness.

The fresh between-bead control uses three detector input channels corresponding
to `z_depth_list: [-r/px,0,+r/px]`, where r is bead radius in µm and px is sample
pitch in µm. This changes **detection focal planes**, preserving the same sample
and illumination beam. Targets remain the central z=0 in-focus classification
with the historical third between-bead class and weighted CE + multiclass Dice.
It is not a simulation of three independently illuminated translated samples.
At radius 1.5 µm and pitch 1 µm these are [-1.5,0,+1.5] pixels. Fractional defocus
linearly interpolates neighboring intensity PSFs; exact integer lookups retain
legacy behavior. PSF depth and canvas include the furthest focal offset. Each
emitter's illumination graph is shared across channels, avoiding three copies.

`minimum_training_seconds` delays convergence stopping; optional
`maximum_training_seconds` stops at a verified epoch boundary after reaching the
requested measured train-loop time. A complete final epoch can overshoot slightly.
Validation, saves, setup and queue wait are counted separately. Short-QoS plans
are finite afterok chains with enough allocation overhead; later segments exit
without training when the target is already reached. Longer preemptible plans
use scheduler requeue with a recorded job ID, config hash, GPU model/order and
original epoch. At most three preemptions are accepted by default. A valid
advancing complete checkpoint is mandatory; missing checkpoints never trigger
fresh training. A preemption can lose work since the last completed epoch.
No Python checkpoint is written from a signal handler.

Validate through `python -m pytest -q tests`, then run
`python -m scripts.transfer_smoke --config INPUT --output OUT` in an allocated GPU
job. It performs one full-density forward/backward step for both channel modes,
checking finite output/loss and nonzero finite phase/detector gradients and
recording memory and timing. It does not prove statistical model performance.
