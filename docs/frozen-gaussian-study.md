# Frozen-Gaussian controls and class-weight grid

`scripts/slurm/prepare_frozen_gaussian_study.py` prepares twelve bounded runs and
never submits them. Supply an explicit account, resource schedule, source snapshot,
base YAML and per-run training hours. The study has three binary controls
(BCE+Dice, one-sided tissue, symmetric tissue) and nine three-class between-bead
runs. All use the focused conventional Gaussian initializer, frozen phase,
trainable fresh detector, one camera view, 1200 beads, batch 1, accumulation 32,
128 train/32 cached validation scenes per epoch, and brightness uniform [0.5,1.5].
The original optical/noise/exposure settings, learning rate and seeds are retained.
The Gaussian includes the configured quadratic focusing phase; it is not an
empty/flat mask or a previously learned phase.

Cross-entropy weights are ordered [background, bead, connection]. Background
varies over {0.05,0.1,0.5}, bead stays 1, connection varies over {0.5,2,4}.
Only relative ratios matter for weighted mean cross-entropy: scaling every class
weight equally cancels. Dice remains the existing equally weighted multiclass
Dice term. Binary CE uses BCE, so the three-element CE weights are inactive there.

Symmetric runs generate bead centres within [-100,100] µm; the one-sided control
uses [0,100] µm. Optical spatial penalties have their historical 0.01 weight,
core halfwidth 5 µm and fixed-reference throughput floor. A frozen phase makes
these scene-independent constant offsets: they cannot optimize the beam or add
detector gradients. The existing relative-threshold LR scheduler still sees total
validation loss, so these offsets can indirectly change its plateau decisions.
Do not interpret them as a newly optimized spatial objective or attribute all
control differences to an optical penalty.

`freeze_slm` now disables phase autograd as well as excluding it from Adam.
The detector still receives gradients. Each verified checkpoint must contain the
exact unchanged raw phase from that segment's start. Trainable-phase behavior and
strict complete-state continuation remain intact. Frozen segments omit the
nonzero-phase-gradient check, which is incompatible with an actually frozen phase.

The requested eight hours are cumulative train-loop time. Validation, setup,
checkpointing, partial-epoch work lost to preemption and queue waits are separate.
Stop only after a verified epoch boundary; a small overshoot is expected. Short
QoS plans have at most seven two-hour allocations per run; longer allocations
include overhead. All continuation jobs are pre-submitted with afterok dependencies
and retain one GPU of the same model. Missing checkpoints never restart from
scratch. Scavenger requeues use the bounded existing identity/config checks.
The preparation helper rejects schedules whose submitted BME total exceeds two
GPUs. Resource availability and duplicate jobs must also be checked at submission.

Each ignored run directory contains immutable input, source/asset hashes, scripts,
Slurm IDs, logs and complete latest.pt state; completion creates a frozen full
model bundle. Equal hours on different GPU types do not imply equal sample/update
counts, which must be reported when comparing class weights.
