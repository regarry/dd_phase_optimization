# Trained binary-checkpoint evaluation

Run through Slurm with exactly two allocated GPUs:

```bash
python scripts/evaluate_checkpoint.py all --checkpoint RUN/checkpoints/latest.pt \
  --output training_results/evaluation --count 16 --seed 1729
```

The trusted complete v2 checkpoint supplies model weights (strict loading), raw
radian phase, bit depth and expanded scientific config. Count is bounded to 1–32
scenes per group. The supported path is binary targets, sample_4f optics and phase
upsample factor one. No training or continuation takes place.

Fresh scenes use the requested seed and unchanged saved generator/configuration.
The same number of first cached validation scenes is evaluated separately.
Explicit per-scene noise seeds are recorded. Each NPZ stores coordinates, target,
probabilities and the exact camera tensor supplied to the detector, in ADU.
Saved-validation scenes are reused model-selection data, not independent tests.
These are synthetic bead results, not cell or instrument validation.

Pixel confusion uses probability >=0.5. Undefined denominators remain undefined.
Supplementary point detection uses 8-connected predicted components, their
maximum-probability pixel and one-to-one matching within two sample pixels of
unique projected in-focus target points. It does not resolve beads sharing x,y
at different depths. Threshold sweeps are exploratory; the reported threshold is
not optimized afterward.

Beam NPZs preserve raw scan-summed intensities through the same sample_4f field
and pad=True angular-spectrum propagation used in image rendering. The horizontal
axis is propagation x; thickness z uses the renderer's N//2−1 origin. The flat
reference has zero phase with the same input amplitude/pupil, not a second
trained detector. Individually peak-normalized display and camera auto-exposure
do not establish fixed-dose benefit.

Capture records checkpoint hash, configuration, epoch, seeds and GPU/runtime
metadata. CPU rendering writes metrics, per-scene CSVs, threshold curves, PNG
figures, index.html and REPORT.md. SUCCESS.json is written only after complete
capture/rendering. `render --output DIR` can regenerate plots from captured arrays
without optical inference. The source checkpoint is never modified.
