# Training and inference handoff

Start with [architecture](architecture.md), [segmented training](segmented-training.md), [checkpoint evaluation](checkpoint-evaluation.md) and [fixed dose](fixed-dose.md). Run jobs from an immutable source snapshot through the portable Slurm launcher. Set site resources using the launcher environment variables documented in the README and keep outputs in ignored `training_results/`. Use the same device requirements throughout a continuation chain.

For matched training budgets, set identical `max_epochs`, `ntrain`, batch size and accumulation. Set time/convergence controls so they allow the target to finish, then verify completed epochs, train scenes and optimizer updates in checkpoint progress. `models/net_310.pt` is the zero-based epoch 310 weights snapshot. It represents 311 epochs. Use `checkpoints/latest.pt` for resume and the model bundle for portable inference.

Fixed exposure holds the incident amplitude and camera signal multiplier constant. Train an exposure ablation with identical initialization, scenes, optimizer updates and geometry. At evaluation, render both models under the same fixed scale and optionally under auto exposure. Beam panels normalized to each peak reveal shape. Shared-scale intensity and power audits quantify throughput.

The experiment replay scripts below expect the saved artifacts from the corresponding experiments and a single allocated CUDA GPU. They write to an explicitly supplied output directory. Their checkpoint and scene counts are specific to these studies.

- `python -m scripts.evaluate_matched_grid_snapshots --study STUDY_DIR --output OUT_DIR` runs the nine frozen Gaussian grid arms at epoch index 310 on 128 paired fresh scenes. Results contain pooled metrics and exploratory scene-bootstrap intervals.
- `python -m scripts.evaluate_dose_scene_match --fixed-run FIXED_RUN --auto-run AUTO_RUN --output OUT_DIR` compares the fixed-dose final checkpoint at 34,048 scenes with the historical auto exposure epoch-34 snapshot and phase TIFF at 33,600 scenes. Shared inference settings support a historical comparison. Training geometry, brightness and validation cadence differ.
