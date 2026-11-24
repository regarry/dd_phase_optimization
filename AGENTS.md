# Working on Deep Design

Read [the architecture map](docs/architecture.md) and [the workflow handoff](docs/workflow-handoff.md) before changing training, optics, or evaluation. Verify the current implementation and preserve unrelated changes.

- Treat phase arrays as radians and use `physics.phase_io` for hardware codes.
- Preserve the differentiable path to raw phase. Validate optical forward and phase gradients on an allocated GPU when changing propagation or quantization.
- Match completed epochs, scenes per epoch, batch size, accumulation and seeds in scientific comparisons. Report GPU time separately.
- Keep configs, source provenance and complete checkpoints for each run. Resume from `checkpoints/latest.pt`. Periodic model weights are for inference.
- Run GPU workloads through the configured scheduler. Keep generated results in ignored run directories and configure site paths locally.
- Run `python -m pytest -q tests` in the configured environment and state checks that require a GPU.
