# Lateral jitter of z-coupled bead pairs

```yaml
z_coupled_ratio: 0.5
z_coupled_spacing_range: [1, 10]
z_coupled_lateral_jitter_px: 3
```

For each z-coupled pair, sample the first bead normally and offset the second
bead independently along x and y by integer grid pixels. `3` means an inclusive
displacement of -3 through +3 pixels on **each** axis relative to the first bead.
This is a square per-axis bound, not a Euclidean disk radius. The first bead is
the anchor; the relative displacement cannot grow to twice the configured radius.

The generator samples uniformly among the allowed coordinates within that
window, separately per axis. Near boundaries the allowed offsets become
asymmetric so beads remain inside the existing PSF-safe placement ranges;
coordinates are not clipped after sampling. A zero offset remains possible.

Coordinates follow the existing `[x,y,z]` contract: x is camera column /
illumination propagation, y is camera row / scan direction, z is detection depth.
The jitter applies to the x/y projection and does not change z-spacing selection.
One grid pixel is `slm_px / phase_mask_upsample_factor` meters.
Random, unpaired beads keep their original sampling rules.

`0` disables jitter. Omitting the key also means 0, with no extra random draws:
historical seeded scenes remain reproducible. Negative values, fractions,
booleans and strings fail with an explicit configuration error. The example templates set 3; frozen run inputs must retain their original value.

Where three-class connection labels are requested, intermediate z slices follow
the straight line between the actual jittered endpoints, rounded to integer x/y.
This does not add depth-resolved output targets or make the 2D detector able to
distinguish beads sharing a projection. Existing binary targets remain unchanged
except for the intentionally jittered bead locations.

Changing jitter is a scientific input change: resume from the saved run YAML
to retain the original setting. Strict resume checks reject introducing a changed
jitter value into a historical checkpoint.
