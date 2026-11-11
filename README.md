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
