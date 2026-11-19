# Bounded training segments and durable inference bundles

`training.segments` runs one already-submitted allocation and never calls sbatch.
Prepare an explicit finite chain on a login node, with `afterok` dependencies.
Every segment uses the same persistent run directory and immutable input YAML.
Reduced train/validation counts belong in a **new** run's input; an existing
checkpoint's scientific settings are not silently changed to fit short segments.

The frozen `RUN/segment_plan.json` requires:

```json
{
  "segment_limit": 10,
  "config_sha256": "SHA256_OF_INPUT_YAML",
  "expected_gpu_count": 2,
  "source_snapshot": "/path/to/frozen/code"
}
```

From that snapshot, an allocated segment executes:

```bash
python -m training.segments run --run-dir RUN --segment-index 1 \
  --segment-limit 10 --wall-seconds 6900 --margin-seconds 900
```

Use index 2, 3, etc. in the pre-submitted continuation scripts. Segment 1 refuses
an existing checkpoint. Every later segment requires the immediate predecessor's
successful receipt, matching checkpoint hash/epoch and strict resume config. The
visible CUDA count and logical GPU model/order must match. GPU UUIDs can differ
between allocations; CUDA bitwise equivalence across hardware is not guaranteed.
No missing or invalid state falls back to fresh training. Each successful active
segment must advance at least one verified epoch. A converged/finished run is a
clean no-op in the remaining already-queued jobs.

## Progress and stopping

The optional immutable YAML `minimum_training_seconds: 43200` requires twelve
hours in the training loop before validation early stopping is permitted. This
time includes training data-loading, forward, losses, backward and optimizer
work; it excludes setup, validation, checkpointing and queue time. The ordinary
early-stopping best/counter is tracked before the minimum, but cannot terminate
training early. The finite allocation/epoch cap remains a hard upper limit: if
it prevents meeting the minimum, the terminal receipt explicitly fails instead
of claiming twelve hours or convergence.

Use actual work counters instead of epochs alone. Complete v2 checkpoints now
contain an optional `progress` dictionary with cumulative training seconds,
training/validation scenes, emitter exposures, microbatches and optimizer updates.
An emitter exposure is one emitter present in one generated scene, not an
independent physical bead or detected object. Reused validation scenes and their
emitter exposures remain separate. `progress.json` and epoch timing records expose
these quantities, plus weighted mean loss components. Counts are observed from
the real batches rather than inferred from nominal YAML sizes. Historical
checkpoints without counters recover recorded timing only; scene counts are not
invented. A safe stop occurs only after a complete checkpointed epoch.

Optional `validation_seed` builds cached validation scenes in an isolated RNG
context, so initialization-specific random draws do not change the evaluation
scenes. Optional `training_seed` sets the Python/NumPy/Torch streams to seed+epoch
at the start of each training epoch, including worker seeding. Both are strict
scientific configuration on resume; omitting them preserves historical behavior.
These controls establish matched pseudorandom scene streams, not CUDA determinism.

## Final artifact

At convergence or the last allocation, `training.model_archive` creates
`RUN/model_bundle/` atomically. It includes complete state, raw phase, frozen
source modules, scientific config, PSF/bead templates, environment versions,
checksums and a full-model loader. A compatible environment can therefore use
the old implementation even when the main checkout changes.

The helper also attempts a standalone TorchScript detector and records whether
tracing/read-back comparison succeeded. That artifact accepts the recorded camera
shape and does **not** contain the optical simulation. If export is unsupported,
the failure is recorded and the frozen full eager pipeline is still preserved.
Saving a Python model pickle alone would not make the optical model immune to
future framework/driver changes. Load trusted bundles only. Use a fresh Python
process with the recorded dependencies and original logical GPU count/order.

Receipts are `segments/001.json`, etc. `terminal.json` distinguishes convergence,
an explicit cap and a cap that prevented the training minimum. Unexpected errors
remain errors; they are not disguised as early stopping or successful continuation.
