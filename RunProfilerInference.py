import subprocess
import os

# bsub -n 8 -R "rusage[mem=16GB]" -W 6:00 -q bme_gpu -gpu "num=1:mode=exclusive_process:mps=no" -Is bash
# conda activate /rsstu/users/a/agrinba/DeepDesign/deepdesign
# cd
# python RunProfilerInference.py
if __name__ == "__main__":
    # Set your arguments here
    training_folder = "./training_results/20251211-151155/"
    epoch = 999
    inference_results = "inference_results/"
    beam_profiles = inference_results

    mask_file = os.path.join(training_folder, f"mask_phase_epoch_{epoch}.tiff")

    # Run mask_inference.py
    inference_cmd = [
        "python", "mask_inference.py",
        "--input_dir", training_folder,
        "--epoch", str(epoch),
        "--res_dir", inference_results,
        "--num_inferences", "1",
        "--plot_train_loss",
        "--device", "cuda"
    ]
    # print("Running mask_inference.py...")
    subprocess.run(inference_cmd, check=True)

    # Run beam_profiler.py
    profiler_cmd = [
        "python", "beam_profiler.py",
        "--output_dir", beam_profiles,
        "--mask", mask_file
    ]
    
    print("Running beam_profiler.py...")
    subprocess.run(profiler_cmd, check=True)