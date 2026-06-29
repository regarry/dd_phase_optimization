import os
from datetime import datetime
from pathlib import Path

# Native programmatic function tracking imports
from beam_profiler import run_beam_profiler
from inference import run_inference

if __name__ == "__main__":
    training_folder = "./training_results/20260623-145614"
    epoch = 6
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    inference_results = os.path.join(training_folder, timestamp)
    beam_profiles = inference_results

    # Match original path evaluation logic exactly
    mask_path_1 = Path(os.path.join(training_folder, f"mask_phase_epoch_{epoch}.tiff"))
    mask_path_2 = Path(os.path.join(training_folder, f"mask_phase_epoch_{epoch}.tif"))
    mask_path_3 = Path(os.path.join(training_folder, "learned_phase_masks", "bmp", f"mask_phase_epoch_{epoch}.bmp"))
    
    if mask_path_1.exists():
        mask_path = mask_path_1
    elif mask_path_2.exists():
        mask_path = mask_path_2
    elif mask_path_3.exists():
        mask_path = mask_path_3
    else:
        raise FileNotFoundError(f"Mask file not found for epoch {epoch} in {training_folder}")
    
    print("RunProfilerInference targeting mask: ", mask_path)
    config_path = os.path.join(training_folder, "config.yaml")
    
    # 1. Primary Mask Beam Profiling Call
    print("Running beam_profiler.py logic natively...")
    run_beam_profiler(
        config_path=config_path,
        mask_path=str(mask_path),
        output_dir=beam_profiles
    )
    print("Beam profiling completed successfully.")
    
    # 2. Main Validation Loop/Inference Processing Call
    print("Running mask_inference.py logic natively...")
    run_inference(
        input_dir=training_folder,
        epoch=epoch,
        res_dir=inference_results,
        num_inferences=5,
        plot_loss=True
    )
    print("Inference completed successfully.")
    
    # 3. Auxiliary Baseline Target Controls Comparisons (Axicon and Fresnel Lens)
    print("Running comparison profiles natively...")
    axicon_beam_profiles = os.path.join(inference_results, "axicon_beam_profile")
    run_beam_profiler(
        config_path=config_path,
        output_dir=axicon_beam_profiles,
        gen_phase_mask="axicon"
    )
    
    fresnel_lens_beam_profiles = os.path.join(inference_results, "fresnel_lens_beam_profile")
    run_beam_profiler(
        config_path=config_path,
        output_dir=fresnel_lens_beam_profiles,
        gen_phase_mask="fresnel_lens"
    )
    
    print("All inference pipeline configurations executed flawlessly without spawning subprocesses!")