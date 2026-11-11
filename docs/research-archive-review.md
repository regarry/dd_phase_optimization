# Review saved training artifacts

These tools analyze separately provided experiment artifacts; none are bundled
with the source. Run from the project environment:

```bash
python scripts/review_training_archive.py --root training_results --output training_results/review
python scripts/visualize_archive_samples.py --review training_results/review --runs RUN_ID
python scripts/visualize_training_archive.py --review training_results/review --runs RUN_ID
```

Use optional `--since YYYYMMDD` and `--until YYYYMMDD` on the inventory scanner.
The gallery accepts repeated `--cohort ID_A ID_B` options for explicitly selected
comparisons; check configuration and target hashes yourself before calling a
comparison matched. Select run IDs from the inventory, not from embedded defaults.
Run the sample renderer before the gallery renderer to include the 3D figures.
The sample renderer loads trusted project-generated pickles only.

The inventory prunes model/checkpoint/large propagation folders, summarizes saved
losses and recomputes metrics from paired TIFF predictions/targets. It never
runs missing inference or submits jobs. Scores are pooled pixel confusion at
threshold 0.5 or RGB bead-class argmax. Fractional targets form a separate overlap
task. Best saved evaluation selection is explicit and introduces selection bias.
Historical validation reuse is not independent testing.

Profile axes are reconstructed from saved configuration and checked against shape.
Normalized archived profiles are not calibrated photon budgets. The 3D renderer
verifies target identity before assigning projected-pixel status to coordinates;
out-of-focus points are outside the task and false-positive pixels are not
localized 3D beads. Source artifact trees remain unchanged; generated review
files go only to the requested output directory.
