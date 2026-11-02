# Training, phase export and resume fixes

Based on feature/multi-gpu-opt commit ac941e394cce5c6f0bb988d30c780da7a58a5604. Changes are local; no remote push was performed.

## Changes

- Dice dispatch passes logits exactly once. Multiclass Dice handles both 2D and 3D targets.
- Forward preserves parent evaluation mode; frozen networks remain in eval mode.
- `spatial_well_steepness` is a positive fixed Python scalar from YAML, with no tensor/device or optimizer state.
- `slm_bit_depth` controls straight-through phase quantization and export. Default 8; set 10 for 1024 codes. Both assume a linear full-2pi phase stroke. The device driver still needs the correct measured wavelength/polarization LUT and vendor data format.
- `physics/phase_io.py` owns quantization, typed loading, export and circular resize. Input fields inside physics always use radians. File loaders no longer guess units from max()==255.
- Training saves separate raw-radian float TIFFs and drive-code TIFFs with `.json` sidecars. Drive BMPs are emitted only for <=8 bits. For >8 bits, TIFF/PNG use uint16 storage without stretching 10-bit values to 65535.
- Exporter entry points are retained as thin adapters: array input (`process_phase_mask`) and file input (`normalize_phase_image`). The batch launcher used the file version; `physics/axicon_mask.py` had a stale import for the array version. FullAxiconGenerator and RotateImage also contained duplicated conversion code; both now use shared export logic.
- Circular interpolation avoids scalar interpolation through pi across a 0/2pi wrap. Ambiguous cancelling phasors fail explicitly. Padding is available. Carrier periods are defined in final hardware pixels and are applied after resizing/padding. Rotation uses nearest-neighbor phase sampling and preserves metadata.
- Complete `checkpoints/latest.pt` is atomically replaced each completed epoch. Legacy network-only `models/net_*.pt` files are retained for current inference workflows.

## YAML

New-run settings, added to config.yaml and batch_config.yaml:

```yaml
accumulation_steps: 32       # required; microbatches per optimizer step
slm_bit_depth: 8
phase_mask_input_units: null  # only needed for old files: radians or drive_codes
spatial_well_steepness: 1.0   # inverse pixels
resume_checkpoint: null
resume_reset_early_stopping: false
```

For a legacy integer TIFF/BMP with 8-bit drive codes, explicitly set `phase_mask_input_units: drive_codes` and `slm_bit_depth: 8`. New sidecar-tagged files are self-describing. Floating legacy TIFFs default to radians. Integer files without metadata or explicit units are rejected, including images whose maximum happens to be below 255. Keep each image with its sidecar.

The supplied config.yaml still has all loss weights zero. Set the desired nonzero loss weights before training; it now fails with a clear error early rather than after expensive initialization. This change does not pick a scientific objective for you.

## Resume

Use a separate minimal YAML to inherit the saved training configuration:

```yaml
resume_checkpoint: /absolute/path/to/run/checkpoints/latest.pt
max_epochs: 1500
resume_reset_early_stopping: false
```

```sh
python -m training.training -c resume.yaml
```

`max_epochs` is the total desired epoch count, not the number of additional epochs. Resume uses the saved output directory and preserves original config.yaml, writing resume_config.yaml. A full YAML is also accepted but changes to saved scientific/model settings are rejected. Allowed controls are resume_checkpoint, max_epochs, save_epoch_interval and resume_reset_early_stopping. To continue a run that already early-stopped, explicitly set the reset flag true.

Saved state: model weights/buffers; raw phase parameter; Adam moments and step counts; learning-rate scheduler; early-stopping state; epoch; Python, NumPy, Torch CPU and CUDA RNG; fixed validation samples; loss histories; original and expanded config. Initialization happens before restoring RNG state. Multiworker loaders use nonpersistent workers and their next-epoch seeds derive from restored Torch state.

Resume is at an **epoch boundary**. A mid-epoch crash replays that epoch from the latest completed checkpoint. Old net_*.pt files do not contain enough information for exact continuation. Load only your own trusted checkpoints because resume includes Python/NumPy state. Same code, dependencies, data, and visible GPU count/order are required; nondeterministic CUDA kernels can still prevent bitwise identity.

## Export CLI

```sh
python -m physics.phase_mask_norm raw_radians.tif display.bmp --bit-depth 8 --width 1920 --height 1152 --pad
python -m physics.phase_mask_norm raw_radians.tif display.tif --bit-depth 10 --width 1920 --height 1152 --pad
python -m physics.phase_mask_norm old_codes.tif display.tif --input-units drive_codes --input-bit-depth 8 --bit-depth 10
```

`bash RunPhaseMaskNorm.sh INPUT OUTPUT [options]` now forwards arguments. Cluster module/environment lines remain site-specific. For the axicon helper, run `python -m physics.axicon_mask` from the repository root. Legacy standalone inference scripts were migrated at their mask-loading boundaries; their pre-existing obsolete imports/API differences were not otherwise repaired or smoke-tested.

## Gradient sums versus averages (finding 8)

For K equal-size microbatches, backpropagating each mean loss yields a summed gradient Σg_i. The gradient of the effective-batch mean is Σg_i/K. Plain SGD with an unchanged learning rate therefore makes a K-times larger update with the sum. With Adam, uniform constant scaling often mostly cancels between first and second moments; it is not generally a K-times update. Epsilon, clipping, changing window sizes, optimizer state and relative regularization matter. Averaging gives stable semantics as accumulation changes. A partial last window should use its actual sample count, especially with unequal microbatch sizes.

Implemented: `accumulation_steps` is mandatory and validated before training setup. Missing, misspelled/legacy `gradient_accumulation_steps`, boolean, noninteger, zero and negative values raise explicit errors; there is no fallback. Each microbatch mean loss is multiplied by its sample count before backward. Immediately before each optimizer step, every accumulated parameter gradient (including phase) is divided by the actual number of samples in that window. This handles a smaller final microbatch and a partial final window. Epoch training loss is sample-weighted for reporting; that reporting division does not replace gradient normalization.

For sample-mean objectives and models without batch-dependent operations, this matches training on the concatenated effective batch. BatchNorm, stochastic layers, global-batch objectives and weighted cross-entropy with PyTorch's class-weight denominator can differ from concatenated-batch semantics; the implementation consistently averages microbatch objective gradients by sample count.

Checkpoint format is now v2. Old v1 checkpoints used summed gradients and are rejected rather than silently changing optimizer continuation semantics. New runs save v2, which supports exact state restoration under the same assumptions described above. Legacy network-only files remain inference weights.

## Propagation reuse versus shared CPU storage (finding 21)

No cache rewrite was made without a benchmark. Preferred first experiment: cache the scan-collapsed intensity profile for each unique propagation coordinate within a forward pass, retaining its autograd graph. A 1024x1024 complex64 field takes 8 MiB; a length-1024 float32 collapsed profile takes 4 KiB. Those figures exclude FFT workspaces and tensors retained for backward, which can still dominate memory.

Compare: original per-bead propagation; per-device unique-coordinate reuse; unique-coordinate reuse with activation checkpointing; and PyTorch saved-tensor CPU offload. Record forward+backward wall time and peak memory, and compare output and raw-phase gradients. GPU interconnect, PCIe bandwidth, repeat rate and retained graph size determine the winner; CPU shared memory is not automatically faster.

`.to('cpu')` and `.to('cuda')` on floating/complex tensors preserve graph edges within a process. However, merely moving the final field to CPU does not necessarily free all GPU intermediates saved for backward. `torch.autograd.graph.save_on_cpu(pin_memory=True)` targets those saved intermediates. `.detach()`, NumPy conversion and ordinary multiprocessing shared-array caches do not preserve the original gradient connection. Setting requires_grad on a detached copy does not reconnect it. A cache must not survive a phase update; cross-forward reuse during accumulation also requires careful graph-lifetime handling. Cache within one forward first.

Official references:
- https://docs.pytorch.org/docs/2.15/autograd.html (saved tensors and save_on_cpu)
- https://docs.pytorch.org/docs/2.15/notes/amp_examples.html (gradient accumulation)

## Validation

CPU tests exercise binary/multiclass Dice dispatch and gradients in 2D/3D; 8-/10-bit exporter round trips and straight-through gradients; legacy units rejection; circular resizing; final-grid carriers; fixed steepness; evaluation-mode preservation; exact optimizer/RNG resume; a mocked-physics training-main continuation versus uninterrupted run; and 10-bit rotation metadata.

The training-main integration test uses a small synthetic model, not the full optical simulation. No CUDA throughput, full optical training, inter-GPU gradients or physical SLM display were tested. Tests run with Python 3.10 and PyTorch 2.14.1 on CPU.

```sh
python -m pytest -q tests
```
