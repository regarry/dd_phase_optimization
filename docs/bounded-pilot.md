# Bounded matched-control pilot

The pilot prepares a smoke run and two matched training arms: A freezes the flat
phase while training the detector; B trains both phase and detector from the same
initialization. This is an explicit optional example workflow, not a standing
instruction to submit jobs or a calibrated microscope specification.

Configure local site variables as described in [portability.md](portability.md).
Then prepare persistent, immutable code/config/script artifacts without submitting:

```bash
python -m scripts.slurm.pilot_control prepare --root "$PWD/training_results/pilots/example"
```

The smoke runner checks the actual forward path, finite loss, nonzero raw-phase
gradient, complete v2 checkpoint and epoch advancement on continuation. It records
complete training/validation/save cycles and a finite-window Gaussian shape check.
The existing optical model's pupil/phase and exposure conventions still apply.
A failed smoke blocks the matched arms rather than silently changing objectives.

After a successful smoke, `finalize` selects a small equal epoch budget from
measured timing, with an upper bound of six hours per arm and an epoch-boundary
margin. Both arms use the successful smoke particle count, common detector
initialization and validation scenes. Check their recorded hashes and actual
sample counts before interpreting differences. Existing camera auto-exposure is
labeled and is not a fixed-dose comparison.

Submit only within an explicit resource authorization, from a login terminal.
Gate A with `afterok:SMOKE_JOB`, B with `afterok:A_JOB`, and keep the two-GPU runs
serial. Scripts never submit further jobs from a compute node. Check the queue
for duplicates before any submission.

Continuation templates require a separate explicit positive segment limit.
`continuation_command` builds an afterok dependency for an already prepared
segment; a missing/invalid checkpoint is fatal, and a finished or early-stopped
run is a clean no-op. Preserve scientific config, visible GPU count/order and
persistent run directory. The helper never restarts from scratch as a fallback.
