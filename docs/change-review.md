# Prepared change review

Suggested commit title: **Make training workflows portable and add bounded diagnostics**

## Included changes

- Explicit Gaussian illumination initialization without changing the underlying input amplitude.
- Optional per-axis lateral jitter for depth-coupled beads, including PSF-safe bounds and legacy zero-jitter reproducibility.
- Training progress, finite-loss/nonzero-phase-gradient checks, timing measurements, complete checkpoint read-back and safe epoch-boundary stopping.
- Bounded four-worker startup preflight with timed diagnostics and persistent results.
- Correct Slurm launcher path and module execution; persistent copied input/scripts; configurable interpreter, modules, account, partition, QoS, GPU request and wall time.
- Frozen benchmark/sweep preparation and matched-control pilot/continuation gates; no implicit job submission.
- Trained-checkpoint capture of exact detector-input camera tensors, fresh/reused-scene metrics, confusion matrices and raw optical profiles.
- Parameterized archive inventories, selected-run galleries and verified 3D sample plots.
- Packaging of imported utilities/script namespaces, an explicit tested dependency list, generic technical documentation and a synthetic small-workload configuration.

The root scientific configuration retains its previous settings except for the
new optional jitter setting. Active experiment-specific settings were not copied
into this branch. No optical propagation equation or exposure model was changed.

## Exclusions and privacy

No conversation files, private instrument notes, run configurations, checkpoints,
datasets, inference arrays, generated figures, logs or captured evaluation results
are included. Local IDE settings, dated research notes, machine-specific environment
exports and bundled legacy generated mask/figure assets are removed from this
review tree. Legacy scripts no longer embed local interpreter/account paths or
an email address. Active experiment artifacts and original local files remain in
the separate working checkout.

A normal branch retains its existing parent history. The separately prepared
source archive contains the staged source tree only, without any Git history.
The privacy checks concern current file contents, not historical Git objects.

## Validation

- 74 CPU tests passed with Python 3.8.13 and Torch 2.4.1+cu121.
- Shell syntax and scheduler dry-run rendering passed with fictitious site options.
- The generic inventory, 3D sample renderer and gallery passed an end-to-end
  synthetic fixture; no fixture outputs are included.
- All 60 Python source files parsed; package discovery includes utils and scripts.
- Current text tree scanned for local accounts, home directories and scratch paths, email addresses,
  recognizable private-key/token signatures and binary files; no matches found.
- No allocation or GPU jobs were launched for this portability preparation.

CPU checks do not establish CUDA or physical-device correctness on a new computer.
Validate a small optical smoke run in the destination environment before training.
The worker-startup mitigation does not claim a proved cause for a previous node
failure. Resume must preserve the complete checkpoint scientific settings, device
count/order and expected run directory.

## Local review

From this review checkout:

```bash
git status --short
git diff --cached --stat
git diff --cached
```

The changes are staged on `codex-deep-design`, with no commit or push. Review the
index before committing. See portability.md for destination setup and local site
settings. No credentials belong in the repository or source archive.
