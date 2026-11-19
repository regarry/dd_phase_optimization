# Latest-checkpoint optical and camera audit

This workflow evaluates atomic copies of complete v2 checkpoints; it does not
change training settings or resume from model-only weights. Use the run plan's
frozen source for GPU capture. Root entry points are Python modules:

```bash
python -m scripts.airy_parameter_debug --config CAPTURE.json --output AUDIT
python -m scripts.evaluate_tissue_study capture --plan PLAN.json --count 4
python -m scripts.evaluate_tissue_study references --plan PLAN.json
python -m scripts.evaluate_tissue_study render --plan PLAN.json
python -m scripts.matched_bead_evaluation all --plan PLAN.json
```

Capture/debug require a scheduler allocation. Rendering and metrics use saved
arrays on CPU. Inference may reduce two saved optical replicas to one only after
verifying identical buffers and loading required state strictly. This is not a
migration of training optimizer/CUDA RNG state. All captures retain checkpoint
hash, epoch, scientific settings, sample/noise seeds, GPU and environment data.
HTML reads/writes explicitly use UTF-8 on compute nodes with C locales.

## Axes and scan integration

x is illumination propagation and camera columns; y is the scan axis/camera rows;
z is detection depth/sheet thickness. Profiles use the same sample_4f field and
padded propagation as emitter rendering, then sum |U|² over y. The legacy renderer
uses `real(U*conj(U))[0,0,:,N//2-1+z].sum()`. The profiler's sum over the penultimate
axis is that same integral, not a central slice. Current dithering is uniform over
the complete simulated scan axis; no finite scan dwell window is modeled.

Phase arrays are radians, with explicit quantization through physics.phase_io.
Image coordinates use the configured px; the renderer origin is N//2−1. Camera
arrays are simulated ADU, not calibrated photon measurements. Beam maps and line
profiles use **linear, independent per-beam normalization** for visual inspection;
raw intensities remain available. This display cannot demonstrate equal dose.
Full-z overviews are needed because remote lobes may be outside the ±150 µm zoom.

## Width definitions

The definitions follow [Remacha et al. (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6968747/):
main-lobe thickness is the first-crossing width at 1/e intensity, optical sectioning
is the shortest contiguous interval containing 1−1/e of the integrated power,
and generalized propagation range is measured to doubling of that width. The
Gaussian √2 Rayleigh convention is different. We report left/right distances from
the minimum observed optical-sectioning waist, its x coordinate, and censoring
when the crossing is outside the simulated propagation field.

utils/beam_metrics.py computes interpolated widths at x=0, records the global
strongest lobe location, and separately measures the strongest lobe within
|z|≤20 µm. It never silently calls a remote lobe the useful focal-plane core.
Power is integrated only over the simulated z window; edge power and finite
windows are recorded. These illumination metrics are not a combined detection
PSF or a proof of two-bead resolution. A 1 µm grid limits precision of thin lobes.

A sample-restricted optical-sectioning width is additionally reported when
fluorescent support is known. It uses intensity masked to that support, and is
explicitly separate from the global definition. This matters for spaced-lobe
beams: their total illuminated thickness can be large even when the lobes fall
in clear, nonfluorescent regions and contribute little camera background.

## Airy phase estimates

The existing cubic-only mask is exp[-i α(u³+v³)], with α in rad/m³ and u,v in
metres. sample_4f currently performs free-space propagation rather than inserting
a focusing lens. A cubic mask alone at that plane is not automatically a focused
Airy sheet. The debug experiment adds an explicit quadratic focusing phase, plus
a linear carrier to align the predicted main lobe with the renderer's z origin.
It preserves old training masks and defaults; comparison metadata distinguishes
the new focused analytic reference from historical Airy initialization.

From the paraxial Fourier integral, characteristic Airy scale
`s0 = f*(3*abs(alpha))^(1/3)/k`, where k=2πn/λ. The ideal intensity 1/e lobe width
is approximately 1.94735*s0. Match this to a target width to obtain
`abs(alpha) = (k*target_width/(1.94735*f))^3/3`.
Finite pupil, scan integration, phase sampling, quantization and spectral clipping
require numerical verification. An ideal unscanned J0² Bessel core is not the same
width as a scanned Bessel sheet. The debug sweep reports that distinction instead
of claiming an exact match or an optimized alpha. The nominal numerical_aperture
also controls the PSF model; illumination in this route is separately constrained
by its actual frequency cutoff and finite Gaussian/SLM support.

## Camera interpretation

The actual NoiseLayer first rescales each ideal image by its 99.9th percentile,
with target_saturation_ratio=.01, then applies QE=.6, dark mean=120, read σ=60
and multiplicative gain=.46 before clipping at 65535 ADU. Thus target .01 is not
an instruction to saturate the camera. With the current settings, dark offset is
55.2 ADU and dark+read RMS is approximately 28.06 ADU. The scaling is not a measured
exposure time or fixed incident dose. Noiseless expected ADU is reconstructed from
the exact pre-noise tensor captured by a hook; noise is applied only once.

Plots label percentile display stretches and add equal-pixel-aspect crops.
The isolated PSF templates can be audited using intensity second moments. Round
kernels do not imply a clean image: defocus lowers peaks while retaining nearly
the same integrated fluorescence, many beads overlap, and display stretching can
make read noise conspicuous. Binary/three-class point detection losses do not
require the raw camera image itself to have a dark background. Inference metrics
are bead-class pixel Dice/F1, precision and recall, not real-cell segmentation.

## Matched-volume comparison and training monitoring

scripts/matched_bead_evaluation.py cross-evaluates saved phase/detector pairs on
four identical confined 1200-emitter scenes with paired noise seeds and common
PSF, camera and image geometry. Original detector weights load strictly; no
retraining occurs. Historical detection-only baselines had different training
geometry, initialization conventions, sample streams and update budgets. This
comparison measures checkpoint performance under a common inference task; it
cannot isolate a causal benefit of one training objective.

scripts/slurm/study_status.py reads original and recovery job ledgers plus sacct,
progress and terminal receipts. It performs no submissions. A failed allocation
is not convergence. scripts/recover_training_boundary.py verifies an interrupted
run's persistent complete checkpoint, histories, finite model/phase, strict raw
config and CUDA-state count. Its recovery receipt explicitly preserves FAILED
Slurm status; success denotes checkpoint-boundary verification only. Replacement
successors may consume only original cancelled/unexecuted slots, never extend the
ten-allocation bound, and never restart a run from scratch. Preserve scientific
config, logical GPU count/order and GPU model through those continuations.

## Reproducing the refined Airy reference in future runs

The tested cubic orientation is an additional phase parameter. At45°, cubic
phase coordinates a=(y+z)/sqrt(2), b=(-y+z)/sqrt(2) give
`-alpha*(a^3+b^3) = -alpha/sqrt(2)*(z^3+3*y^2*z)`.
That scan/thickness cross term changes the dithered profile at fixed incident
field, pupil and propagation distance. With the current grid/optics, alpha1.1e12
and45° yields43.95µm1/e thickness versus43.59µm for the4° axicon. This is a numerical
main-lobe-width match, not an assertion of equal optical sectioning, FWHM, retained
power, dose, or full propagation length. The figures retain independent linear
normalization; raw finite-plane power is recorded separately.

A new **opt-in** initializer uses the exact tested phase construction:

```yaml
initial_phase_mask: airy
airy_phase_mode: focused
airy_alpha: 1.1e+12  # rad/m^3
airy_rotation_degrees: 45
fresnel_lens_focal_length: 0.00375  # m
lensless_prop_distance: 0.00375    # m
lens_approach: sample_4f
phase_mask_upsample_factor: 1
```

physics.masks.focused_airy_phase_parameters is shared by the factory and GPU debug
script. It returns float32 unwrapped radians and a conservative maximum analytic
phase step. The focused factory requires explicit alpha, positive matching focal
and propagation distances, the reviewed route/upsample factor, and a phase step
below pi per pixel. Physical quantization remains exclusively in phase_io.
The origin carrier uses the renderer's one-pixel offset. Default rotation is0°
within focused mode; absence of airy_phase_mode preserves historical cubic_only.
Existing scientific input YAMLs and frozen queued jobs are not modified.
Factory raw phases are bitwise equal to the submitted GPU-tested construction;
CPU/CUDA quantization differs by one drive level in1/1048576 pixels in this check,
so cross-device bitwise equivalence is not claimed.

Power-interval inverse CDFs retain zero-intensity plateaus. Lower endpoints use
the right edge of a plateau, upper endpoints its left edge. This prevents dark
space from being treated as carrying interpolated power. Regression checks cover
zero-padding invariance and power windows separated by a dark gap. This correction
updates optical metrics, including sample-restricted widths; camera and detection
arrays remain unchanged.

## Final report presentation and PDF

`python -m scripts.update_final_report --report-dir REPORT_DIR` updates saved
final evaluations without GPU inference. It plots complete train/validation
histories directly from each immutable v2 checkpoint, replaces expected-signal
panels with camera-only views, and records 1/e strongest-lobe thickness and
63% power-interval width at the exact left, central and right sampled x positions.
These are FOV boundary measurements, not the OS-doubling locations that define
propagation length. Values and displaced peak positions are saved in per-run
fov_widths.json and the combined fov_widths_all_runs.json.

Missing analytic Airy arrays may be reused only after every relevant illumination
setting and x/z grid matches an existing captured reference; source and checked
keys are logged. Otherwise additional scheduler-side capture is required.

The refreshed HTML uses eager image loading and print CSS, with landscape A3
pages, preserved image aspect ratios, repeated table headers, and protected image/
row page breaks. Export this actual HTML using headless Chromium with print
backgrounds, rather than assembling a separate text-only PDF. Browser profiles,
PDF inspection dependencies and generated exports remain in ignored report paths.
