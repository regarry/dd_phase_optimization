import subprocess
import os

if __name__ == "__main__":
    # Set your arguments here
    training_folder = "training_results/phase_model_20251021-165104"
    epoch = 199
    inference_results = "inference_results"
    beam_profiles = "beam_profiles"

    mask_file = os.path.join(training_folder, f"mask_phase_epoch_{epoch}.tiff")

    # Run mask_inference.py
    inference_cmd = [
        "python", "mask_inference.py",
        "--input_dir", training_folder,
        "--epoch", str(epoch),
        "--res_dir", inference_results,
        "--num_inferences", "1",
        "--plot_train_loss",
        "--device", "cpu"
    ]
    print("Running mask_inference.py...")
    subprocess.run(inference_cmd, check=True)

    # Run beam_profiler.py
    profiler_cmd = [
        "python", "beam_profiler.py",
        "--output_dir", beam_profiles,
        "--mask", mask_file
    ]
    
    print("Running beam_profiler.py...")
    subprocess.run(profiler_cmd, check=True)