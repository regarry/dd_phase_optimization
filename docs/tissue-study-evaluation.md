# Interim evaluation of tissue studies

`scripts/evaluate_tissue_study.py` evaluates saved complete v2 checkpoint snapshots
without training or modifying the source run. It supports both binary detection
and three-class background/bead/connection models. It loads the frozen backend
through PYTHONPATH, requires one or two Slurm GPUs for capture, loads network state
strictly, and checks the immutable checkpoint SHA256 before use.

Prepare a JSON plan with `output` and a `runs` list. Each entry names `name`,
`checkpoint` (an independently copied complete checkpoint), `checkpoint_sha256`,
`output` (an existing output directory), and `input_yaml` (a preserved input copy).
Atomic replacement of the source latest.pt allows an opened source descriptor to
be copied consistently while training continues. Preserve the plan, evaluation
script, resource request and effective scheduler submission in that directory.

```bash
python scripts/evaluate_tissue_study.py phases --plan PLAN.json
# In a one- or two-GPU Slurm allocation, using the frozen backend:
python EVALUATION/evaluate_tissue_study.py all --plan PLAN.json --count 2
# Render existing capture arrays without GPU inference:
python scripts/evaluate_tissue_study.py render --plan PLAN.json
```

The bounded count is 1–4 **per group**: fresh synthetic scenes and cached training
validation scenes. Seed 20261112 is reset for every run; samples match across
initializations of the same approach. Geometry differs between approaches.
Noise seeds are also recorded. Camera auto-exposure and all physical/scientific
settings remain unchanged. Only the auxiliary training-loss profile return is
suppressed during camera inference, avoiding an unused loss evaluation.

The exact camera tensor supplied to the detector is saved as ADU, alongside
coordinates, targets, all class probabilities and decisions. Binary decisions use
threshold 0.5; three-class decisions use argmax. Confusion matrices are rows=true,
columns=predicted, with per-class precision/recall/F1. Connection predictions are
not counted as bead detections. Metrics are pixel metrics, not depth-resolved
bead matching, real-cell segmentation, or a threshold-tuned evaluation. Two fresh
scenes are a diagnostic, not a precise generalization estimate.

Phase arrays are raw and quantized radians; phase PNGs show quantized wrapped
phase with a 0–2π color scale. They are not hardware drive-code exports.
Beam arrays use the renderer's sample_4f and padded angular-spectrum propagation,
with scan y integrated. x is illumination propagation/camera columns; z is
sheet thickness/detection depth at the renderer's N//2−1 origin. Learned,
focused-quadratic Gaussian, analytic axicon, and one-sided analytic Airy profiles use the same configured
incident amplitude, pupil and independent per-beam intensity normalization.
Reference detector inference is not performed. None of these scans verifies
instrument calibration, fixed-dose performance or grid convergence.

The spatial-cost plot shows the exact off-core indicator and the soft throughput
floor. Two-sided optical exclusion is [-100,100] µm, while bead support is
[-50,50] µm. One-sided optical exclusion and bead support are [0,100] µm. Core
halfwidth is 5 µm; actual one-sided useful core is its intersection with tissue,
[0,5] µm. The full optical penalty is `mean(B/R) + 10 mean(relu(.8-C/R)^2)`,
weighted by .01. R is the deterministic zero-phase Gaussian-illumination
normalizer used by the current objective, not the focused Gaussian comparison
plotted as a baseline. It is fixed and cannot be diluted by bright remote lobes.
Outside-tissue intensity has no direct penalty/reward; the floor is soft and
cannot guarantee preserved central flux.

`index.html` includes phases immediately and explicitly labels pending inference.
Run SUCCESS.json is emitted only after capture and rendering succeed. Global
SUCCESS.json requires all listed runs. Partial failures retain ERROR.json and
cause a nonzero GPU job exit, without retraining or automatic job retry.

## Single-GPU inference

Inference may use one L40S, H100, H200, or a smaller GPU after a bounded memory
check. The evaluator verifies that every saved optical replica buffer equals
replica zero exactly before removing redundant replica keys, then loads the
one-replica state strictly. No detector state is omitted. It does not migrate
training optimizer/RNG state or relax continuation checks. All emitters are still
rendered; partitioning and summation grouping may change floating-point rounding.
Recorded metadata includes inference/training GPU counts, GPU name, peak allocated
memory and runtime. A low-memory failure must be diagnosed before another job.

For new training plans, check L40S, H100 and H200 resource and QoS availability
before choosing a GPU type/count. Preserve that run's chosen device requirements
through continuation; an H200 allocation may have a one-device per-job limit.
Keep aggregate BME use at two GPUs; use available short_gpu quotas across validated
GPU types. Quotas are time-dependent and must be queried before submissions.

CUDA capture explicitly initializes the CUDA runtime/allocator before resetting
peak-memory statistics. In the deployed Torch version, device_count can use NVML
without initializing the allocator, and reset_peak_memory_stats does not perform
lazy initialization itself. Capture failures retain a complete traceback in the
run ERROR.json and Slurm stdout; a failed job is not an inference result.

## Linear profiles and Dice + BCE baselines

Beam heatmaps and line cuts use linear intensity scales starting at zero. Each beam is independently normalized by its own peak; raw arrays remain available. Confusion-matrix colors
may remain logarithmic because they represent highly imbalanced counts rather
than beam intensity.

One-sided comparisons require `airy_reference.npz` with matched x/z axes. Generate
it without rerunning saved scene inference, on one allocated GPU:

```bash
python EVALUATION/evaluate_linear_airy.py references --plan PLAN.json
```

This uses the captured configuration's `airy_alpha` in rad/m³, the same incident
amplitude, quantization, sample_4f field and padded propagation. Current study
alpha is 1e10; it is a documented starting point, not an optimized reference.
Existing raw beam arrays are preserved, with the Airy scan stored separately.
Rendering fails clearly if its required reference is missing or axes differ.

A combined plan can append completed historical Dice + BCE checkpoint snapshots.
They receive the same bounded diagnostic capture and rendering but retain their
original generator, noise and physical settings. They must be labeled historical
baselines: specimen depth, lateral safe region, jitter, validation seed and
training budget can differ from the tissue study. The historical Gaussian
initialization used its older zero-phase convention, whereas new analytic
Gaussian comparisons explicitly use the configured quadratic focus. A failed
run without a complete checkpoint is not replaced by initialized weights.
The gallery reports a missing historical training-time counter as unavailable,
not zero hours. All source checkpoint/config files remain immutable.

The newer audit and reporting contracts are in [optical-evaluation.md](optical-evaluation.md), including Remacha widths, camera diagnostics, matched-volume comparisons, and recovery monitoring.
