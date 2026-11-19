# Intelligent Beam Optimization for Light-Sheet Fluorescence Microscopy through Deep Learning
Here we report that a joint-optimization continuously updates the phase mask and resulting in improved image quality for better cell detection. Our method's efficacy is demonstrated through both simulations and experiments, revealing substantial enhancements in imaging quality compared to traditional Gaussian light sheet. We discuss how designing microscopy systems through a computational approach provides novel insights for advancing optical design that relies on deep learning models for the analysis of imaging datasets.  
Paper reference (doi: https://doi.org/10.1101/2023.11.29.569329)
## Autofocus pipeline
## Required packages
pytorch (>= 1.12), torchvision, matplotlib, numpy, random, skimage, scipy  

## Function of scripts

**mask_learning.py**
 - main file for training model and mask optimization
 - model and training parameters setting
   
**cnn_utils.py**
 - design the structure of CNN
   
**loss_utils.py**
 - loss function design, we deprecate it since we only use cross entropy for simplicity. this part could be future work.
   
**physics_utils.py**
 - design the physical simulated layer (part of CNN)
   
**data_utils.py**
 - define the dataset (simulated random beads)
 - generate the random 3D locations of arbitrary density of beads
   
**Generate_data.py**
- generate the training examples using data_utils.py, output: labels.pickle
- define the size of training
  
**psf_gen.py**
 - generate point spread function by given certain defocus distance
   
**test_localization.py**
 - input: 3D beads batch, trained model and phase mask
 - predition map indicating infocus beads position

**beam_profile_gen.py**
 - generate the 3D profile sections of beam gave the phase mask and propagation range

**FWHM_cal.py**
 - calcuate the full width half maximum by the given beam profile section folder

**submit.sh**
 - bash file for the cluster training

## References
- Li, Chen, et al. "Deep learning-based autofocus method enhances image quality in light-sheet fluorescence microscopy." Biomedical Optics Express 12.8 (2021): 5214-5226.
- Royer, Loïc A., et al. "Adaptive light-sheet microscopy for long-term, high-resolution imaging in living organisms." Nature biotechnology 34.12 (2016): 1267-1278.
- 


## Current training and portable workflows

See [portability](docs/portability.md), the [architecture map](docs/architecture.md),
[bounded pilot](docs/bounded-pilot.md), and [checkpoint evaluation](docs/checkpoint-evaluation.md).
The historical entry-point list above predates the current package structure.

## Tissue objectives, phase transfer and multi-view detection

Use the configured environment in `requirements.txt`, then run
`python -m pytest -q tests`. Scientific GPU execution requires a suitable allocation;
see [portability](docs/portability.md) for site environment variables.

Example scientific configs are `configs/tissue_study.example.yaml`,
`configs/phase_transfer.example.yaml` and `configs/between_three_view.example.yaml`.
They retain explicit seeds, geometry, losses, noise and exposure settings. To use
a learned phase, generate the source study locally or transfer its trusted
checkpoint separately, then use the phase-transfer preparer described in
[phase-transfer](docs/phase-transfer.md). No trained phases or data are distributed
with this source-only checkout. Set your own Slurm account/partition/QoS and GPU
request. The generic single-run launcher accepts `DD_SLURM_*` variables.

Bounded full-density CUDA validation:
`python -m scripts.transfer_smoke --config configs/phase_transfer.example.yaml --output training_results/smoke`.
Run this inside a GPU allocation, not on a shared login node.
See [segmented training](docs/segmented-training.md) for complete-state continuation
and frozen full-model bundles, and [optical evaluation](docs/optical-evaluation.md)
for inference and beam/performance reporting contracts.
