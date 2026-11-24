# Relative fixed-incident-dose performance

`NoiseLayer` supports an optional explicit policy:

```yaml
camera_exposure_mode: fixed
fixed_exposure_scale: 123.0  # example only; replace with a recorded calibration
```

The positive scale is mandatory in fixed mode and never depends on a scene,
beam maximum or detected signal. The default `auto_percentile` retains the
historical behavior and random draw sequence. Camera QE, shot/read noise,
dark offset, gain and clipping are unchanged. A scene-independent multiplier
preserves optical-to-raw-phase gradients. Camera mode/scale are scientific
settings protected by strict checkpoint resume checks; new fixed-dose training
requires a new run, not an altered continuation of auto-exposure training.

`scripts/evaluate_fixed_dose.py` evaluates immutable trusted checkpoint snapshots.
It defines **relative incident illumination energy before the phase/pupil/filter**,
using identical incident Gaussian amplitude, pupil, wavelength, pitch and uniform
discrete scan integration. It verifies matching input arrays and common PSF
kernels, and never renormalizes each mask to equal transmitted/sample power.
Sample geometry, scan dwell convention, noise settings and scenes stay fixed.
One Gaussian reference scale is calibrated from four separate synthetic scenes;
1x is that recorded operating level, not a measured physical dose. Independent
sixteen-scene tests use 0.25x,0.5x,1x,2x,4x, plus the legacy auto-exposure comparison.
All scene and noise seeds are shared across models/dose levels.

The reported total-dose convention treats focal views as sequential acquisitions
with fixed per-frame integration and camera noise. A V-view model receives the
signal coefficient K*dose/V per view, representing reduced illumination intensity
per acquisition; it is not allowed V times the incident dose. Acquisition time is
not held equal. This is an explicit evaluation assumption, not an assertion about
an unmeasured multi-camera instrument. Dark/read parameters remain per frame;
changing physical exposure time would need a different dark-current model.

Dose is not calibrated in J/cm². Absolute calibration needs incident optical
power, illumination/scan dwell times, illuminated area and calibrated conversion
from fluorescence to detected photoelectrons. Existing detectors were trained
with auto-exposure: posthoc fixed-dose performance includes exposure distribution
shift and does not establish the best achievable fixed-dose-trained performance.
The current scalar simulation assumes linear fluorescence and no bleaching.

Optional `return_power_audit=True` on ASM returns the unchanged differentiable
field plus detached scalar-field L2 integrals before propagation, on the full
padded output and after cropping, and a Parseval check on the filtered spectrum.
The audit separates configured spectral-filter loss from numerical crop loss.
It does not restore discarded energy or prove grid-converged physical throughput.
Current evaluated sampling supports only propagating frequencies; finer grids
that include evanescent modes require additional validation (the legacy gamma
calculation clamps negative arguments rather than modeling evanescent decay).
All power integrals are relative arbitrary-amplitude²·m², not calibrated watts.

Reports distinguish final learned-phase checkpoints from interim frozen-Gaussian
snapshots. Binary threshold0.5 and multiclass argmax remain unchanged. Bead hard
Dice/F1, precision, recall, clipping and scene-bootstrap intervals are reported;
class objectives and training budgets differ and must not be conflated.
No ongoing training or its queued evaluation is changed by this workflow.
