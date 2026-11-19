# Code architecture

The training driver jointly optimizes a raw radian illumination phase and a
bead-detection network through differentiable synthetic image formation. This
map describes code contracts, not physical calibration or experimental results.

## Main path

`training/training.py` loads and expands YAML through `data/io.py`, generates
PSF/bead templates, constructs `models/wrappers.ParallelEndToEndModel`, owns the
raw phase and optimizer, caches validation scenes, and trains at epoch boundaries.
The wrapper partitions emitters through `physics/wrappers.MultiGpuSimulation`,
sums ideal images on cuda:0, adds camera noise once, and applies the U-Net.
`physics/phase_io.py` wraps/quantizes phase with a straight-through gradient and
owns hardware drive-code export and explicit bit-depth metadata.

| Module | Contract |
|---|---|
| `data/generators.py` | Integer-valued x/y/z positions; optional bounded lateral jitter of depth-coupled pairs. |
| `data/datasets.py`, `data/transforms.py` | Fresh synthetic training scenes; cached validation; binary or three-class projected targets. |
| `data/psf_gen.py` | Scalar homogeneous detection PSF and finite 2D disk templates. |
| `physics/simulation.py` | Gaussian input, optical routes, angular-spectrum propagation, emitter rendering and camera auto-exposure/noise. |
| `models/unet.py` | Detector logits; losses own sigmoid/softmax. |
| `training/checkpoint.py` | Atomic complete v2 checkpoints; strict scientific resume configuration. |
| `training/runtime.py` | Progress, nonfinite checks, optional nonzero phase-gradient check, cycle timings and safe epoch-boundary stopping. |
| `utils/worker_preflight.py` | Bounded real four-worker tensor-transfer startup check with optional persistent diagnostics. |
| `training/pilot.py`, `scripts/slurm/pilot_control.py` | Bounded matched-arm pilot preparation, smoke gates, continuation checks; preparation never submits jobs. |
| `scripts/slurm/prepare_phase_sweep.py` | Frozen benchmark/random/Gaussian/axicon configurations and scripts; no automatic submission. |
| `scripts/evaluate_checkpoint.py` | Strict trained-checkpoint inference on an allocation; separate fresh/reused scenes and CPU rendering. |
| `scripts/*archive*.py` | Read-only inventories and figures from caller-selected saved artifacts. |

## Coordinates and scientific boundaries

Emitter arrays are `[B,E,3]`: x is camera column / illumination propagation,
y is camera row / scan direction, z is detection depth / sheet thickness.
`image_volume=[H,W,D]` is in sample-grid pixels; propagation accepts distance in
pixels and converts through `px`. Wavelength and optical lengths generally use
meters; profiler keys ending `_mm` use millimeters. Raw phase is radians.

The differentiable path must remain loss → logits → camera simulation → field
intensity → complex field → quantized phase → raw phase. Detaching/caching across
that path requires explicit gradient equivalence checks. Illumination and
detection aberrations are separate models.

`initial_phase_mask: gaussian` means zero phase on the simulator's existing
Gaussian amplitude. It does not set a Gaussian-shaped phase or guarantee an
unmodified Gaussian after the complete optical route. Historical `empty` remains
an alias for flat initialization. Lateral pair jitter defaults to zero when
omitted; see [bead-generation.md](bead-generation.md).

The camera rescales intensity toward a percentile saturation target. Existing
losses and normalized profiles do not establish equal-dose performance. Profile
arrays are `[propagation,thickness]`; the current spatial-well penalty collapses
the thickness dimension. Verify axes before introducing sectioning objectives.
The projected target cannot resolve two beads sharing x,y at different depths.
The nominal 3D detector path is not established by CPU tests.

## Runtime and continuation

Use `python -m training.training --config CONFIG --run-dir RUN` inside an allocation.
`--worker-timeout`, `--progress-every`, `--require-phase-gradient`, `--wall-seconds`
and `--stop-margin-seconds` control observability and bounds, not the objective.
The runtime starts an epoch only when 1.25 times the slowest measured full cycle
plus the margin fits; a first unexpectedly long epoch is not rescued by this rule.
Losses and the first requested phase gradient must be finite. Every completed
checkpoint is read back before its timing record is accepted.

`checkpoints/latest.pt` contains raw/expanded config, model, raw phase, optimizer,
scheduler, early stopping, Python/NumPy/Torch/CUDA RNG, histories and validation
scenes. Resume keeps scientific settings and GPU count/order and never falls back
to fresh initialization. Physical CUDA determinism is not proved by CPU tests.

The worker preflight allows 60 seconds for tensor transfer and an additional
120 seconds for interpreter/import/cleanup. Errors are fatal and recorded when
`--report` is provided. Short node-local temporary paths prevent AF_UNIX path
length failures. Scientific outputs stay in ignored persistent run directories.

## Execution and documentation

See [portability.md](portability.md) for environment and scheduler setup,
[bounded-pilot.md](bounded-pilot.md) for matched controls and continuation limits,
[checkpoint-evaluation.md](checkpoint-evaluation.md) for trained evaluation, and
[research-archive-review.md](research-archive-review.md) for artifact review.
The CPU regression command is `python -m pytest -q tests`. Full optical execution
requires a validated GPU allocation; the incomplete CPU simulation fallback does
not make the full driver a supported CPU workload.

## Joint optical objectives and continuation

Tissue-aware losses and streamed differentiable profiles are in
`training/tissue_loss.py` and `physics/wrappers.py`; see
[tissue-aware-objective.md](tissue-aware-objective.md). Complete v2 checkpoints,
seed isolation, cumulative work counters, bounded continuation and frozen full
model bundles are implemented by `training/checkpoint.py`, `training/seeds.py`,
`training/runtime.py`, `training/segments.py` and `training/model_archive.py`.
See [segmented-training.md](segmented-training.md).

`data/emission.py` optionally extends emitter rows from xyz to xyz+relative
brightness without changing target construction. The default [1,1] leaves RNG
and legacy scenes unchanged. `physics/simulation.py` shares illumination across
focal views and interpolates fractional detection PSFs. `data/io.py` and
`data/psf_gen.py` include all focal offsets in PSF support.
[phase-transfer.md](phase-transfer.md) documents units, camera auto-exposure,
phase-only transfer and measured training-time caps.

`physics/masks.py` adds an opt-in focused/rotated Airy initializer with explicit
alpha and sampling guards. `utils/beam_metrics.py` and the optical reporting
scripts produce linear-scale beam comparisons, phase maps, camera inputs,
confusion matrices and power/width metrics; see
[optical-evaluation.md](optical-evaluation.md). These source tools do not include
trained weights, datasets or experiment outputs.
