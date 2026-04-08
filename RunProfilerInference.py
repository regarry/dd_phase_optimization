import subprocess
import os
from datetime import datetime
from pathlib import Path
# bsub -n 8 -R "rusage[mem=16GB]" -W 6:00 -q bme_gpu -gpu "num=1:mode=exclusive_process:mps=no" -Is bash
# conda activate /rsstu/users/a/agrinba/DeepDesign/deepdesign
# cd
# python RunProfilerInference.py
if __name__ == "__main__":
    # Set your arguments here
    #training_folder = "./training_results/800_beads_phase_model_20251021-111735"
    #training_folder = "./training_results/20260211-162226"
    
    training_folder = "./training_results/20260408-122219"
    epoch = 0
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    inference_results = os.path.join(training_folder, timestamp)
    beam_profiles = inference_results

    # Load mask from tiff file (for both models)
    mask_path_1 = Path(os.path.join(training_folder, f"mask_phase_epoch_{epoch}.tiff"))
    mask_path_2 = Path(os.path.join(training_folder, f"mask_phase_epoch_{epoch}.tif"))
    mask_path_3 = Path(os.path.join(training_folder, "learned_phase_masks","bmp",f"mask_phase_epoch_{epoch}.bmp"))
    
    if mask_path_1.exists():
        mask_path = mask_path_1
    elif mask_path_2.exists():
        mask_path = mask_path_2
    elif mask_path_3.exists():
        mask_path = mask_path_3
    else:
        raise FileNotFoundError(f"Mask file not found for epoch {epoch} in {training_folder}")
    
    print("runprofilerinference: ", mask_path)
    #mask_path = os.path.join(training_folder, f"mask_phase_epoch_{epoch}.tiff")

    # Run mask_inference.py
    inference_cmd = [
        "python", "inference.py",
        "--input_dir", training_folder,
        "--epoch", str(epoch),
        "--res_dir", inference_results,
        "--num_inferences", "5",
        "--plot_loss"
        #"--device", "cuda"
    ]
    
    print("Running mask_inference.py...")
    subprocess.run(inference_cmd, check=True)
    print("Inference completed.")
    
    # Run beam_profiler.py
    config_path = os.path.join(training_folder, "config.yaml")
    profiler_cmd = [
        "python", "beam_profiler.py",
        "--output_dir", beam_profiles,
        "--config", config_path,
        "--mask", mask_path
    ]
    
    print("Running beam_profiler.py...")
    subprocess.run(profiler_cmd, check=True)
    print("Beam profiling completed.")
    
    
    
    # comparison to bessel
    # Run beam_profiler.py
    axicon_beam_profiles = os.path.join(inference_results, "axicon_beam_profile")
    axicon_profiler_cmd = [
        "python", "beam_profiler.py",
        "--output_dir", axicon_beam_profiles,
        "--config", config_path,
        "--gen_phase_mask", "axicon",
        "--bessel_angle", "4.0" # 0.4 deg x 4
    ]
    subprocess.run(axicon_profiler_cmd, check=True)
    
    fresnel_lens_beam_profiles = os.path.join(inference_results, "fresnel_lens_beam_profile")
    fresnel_profiler_cmd = [
        "python", "beam_profiler.py",
        "--output_dir", fresnel_lens_beam_profiles,
        "--config", config_path,
        "--gen_phase_mask", "fresnel_lens"
    ]
    subprocess.run(fresnel_profiler_cmd, check=True)
    
    """
    fresnel_lens_beam_profiles = os.path.join(inference_results, "fresnel_lens_beam_profile")
    fresnel_profiler_cmd = [
        "python", "beam_profiler.py",
        "--output_dir", fresnel_lens_beam_profiles,
        "--config", config_path,
        "--gen_phase_mask", "fresnel_lens"
    ]
    subprocess.run(fresnel_profiler_cmd, check=True)
    """