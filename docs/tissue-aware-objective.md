# Tissue-aware optical objective: opt-in MVP

This is a new objective, not a reinterpretation of historical spatial-loss runs.
It keeps the detector trainable and combines explicit detection losses with an
illumination penalty. It does not model clearing, tissue scattering, physical
ablation, or calibrated chamber aberrations. The tissue interval is an assumed
fluorescent occupancy mask in sample coordinates, not measured instrument data.

## Why the historical implementation is not the same experiment

Historical `DynamicSpatialWellLoss` receives `[propagation, thickness]`, sums the
thickness axis and penalizes the remaining propagation coordinate. That is not
a thickness exclusion interval. Its helper also retains a complete `[P,N,N]`
intensity stack and applies min/max normalization; increasing remote bright
features can reduce a normalized penalty without improving sectioning.

The archive includes frozen-detector, one-to-ten-emitter optical runs, as well as
unfrozen-detector runs with hundreds of emitters and poor saved detection scores.
Consequently, freezing and VRAM pressure are not the only supported explanation
for poor performance. Historical code snapshots are incomplete, so this does
not prove that every past result used the present implementation exactly.

## Explicit settings and geometry

```yaml
ce_loss_weight: 1
dice_loss_weight: 1
spatial_multi_well_loss_weight: 0
tissue_spatial_loss_weight: 0.01 # provisional experimental coefficient, not tuned
freeze_unet: false
freeze_slm: false
tissue_profile_x_um: [-400, -200, 0, 200, 400]
tissue_profile_z_bounds_um: [-150, 150]
tissue_z_bounds_um: [-100, 100]
tissue_sample_z_bounds_um: [-50, 50]
tissue_core_halfwidth_um: 5
tissue_core_min_fraction: 0.8
tissue_core_floor_weight: 10
tissue_core_uniformity_weight: 0 # optional; defaults to zero
```

All geometry above is micrometers. x is illumination propagation/camera column;
y is the summed scan direction/camera row; z is sheet thickness/detection depth.
For the two-sided experiment, a nominal 100 µm sample has bead centres in
`[-50,50]`, while the optical penalty uses the conservative 200 µm envelope
`[-100,100]` to represent the union of fluorescent positions during translation.
The one-sided experiment uses `[0,100]` for both the nominal 100 µm sample and
its optical penalty interval. These are experimental assumptions, not a claim
that the two-sided specimen itself is 200 µm thick. The
profile interval must cover the entire tissue interval; the data depth must
cover the intended bead population. At 1 µm pixels, depth 256 covers `[-128,128)`.
The optical profile can extend beyond that specimen depth within its larger
computational grid. The existing expander auto-sizes camera height from PSF
width; setting a larger raw height alone does not override that behavior.

`tissue_sample_z_bounds_um` is optional. When present it restricts bead-centre z
support, including both paired endpoints and their links, using expanded `px`.
Bounds are inclusive for sampled centres. They do not constrain the finite
bead's entire physical extent. Omitting the key leaves legacy sampling/RNG
behavior unchanged. Changing it is a scientific configuration change.

All new objective settings except the uniformity coefficient are mandatory when
the new loss is enabled. A detection loss must remain nonzero. The legacy spatial
loss and new tissue loss cannot be active together. `expand_config` activates
the model's auxiliary-profile return path when the new weight is positive.

## Loss and memory contract

At each requested propagation coordinate, raw propagated intensity is summed
along scan y. Thickness z uses the exact emitter renderer's `N//2−1` origin.
`sample_4f` and padded angular-spectrum propagation match image rendering.
The new helper only supports that reviewed optical route. Requested x positions
are rounded to grid pixels; duplicate or out-of-field positions fail clearly.

For every x, let C be raw intensity integrated over the in-tissue core, B the
raw in-tissue intensity outside that core, and R the core power from the same
zero-phase Gaussian-illumination reference through the same optical model.
The optical loss is:

`mean(B/R) + floor_weight * mean(relu(min_fraction − C/R)^2)`

An optional uniformity term penalizes variation across x of `C/mean(R)`.
The detector objective is added using the independent YAML coefficients.
Reference R is fixed and gradient-free. No intensity outside tissue appears in
the denominator and it has no direct reward: adding a remote bright lobe cannot
dilute this loss. The core floor is a finite soft penalty, not a strict throughput
constraint; record its violations when choosing weights. Camera auto-exposure is
unchanged and still prevents fixed-dose claims.

The learned profile remains differentiable to raw phase. Each propagation is
activation-checkpointed and reduced to its 1D profile immediately, avoiding the
legacy 3D intensity stack and retained per-position activation volume. The
deterministic reference may be cached without detaching the learned field.
Checkpointed output/phase gradients match a dense toy implementation in CPU
tests; actual CUDA memory, image formation and gradients require allocation-side
measurement. This optimization does not change the emitter renderer's separate
memory cost.

## Allocation-side debugging

`python -m scripts.spatial_mvp_debug --config CONFIG --output NEW_DIR --particles 32`
runs exactly one joint step on one Slurm GPU, with both phase and U-Net unfrozen.
It records weighted loss components, separate detection/optical raw-phase
gradient norms, detector gradient, phase update, peak allocated memory and step
time. It deliberately selects the first nonempty-target scene from a bounded
search, so its result is a debugging check, not representative detection accuracy.
It writes its exact input and expanded YAML and raw profile arrays. It submits
no further jobs. Increase emitters only after a successful bounded test, and
measure a full cycle before selecting segmented production counts.
