import os
import re
import subprocess
import sys
import argparse
from pathlib import Path


def queue_jobs(base_dir, start_dir, single_job=False):
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: Base directory '{base_dir}' does not exist.")
        return

    # Create logs folder to prevent LSF writing errors
    os.makedirs("./logs", exist_ok=True)

    # Matches standard YYYYMMDD-HHMMSS format
    dir_pattern = re.compile(r"^\d{8}-\d{6}$")

    # Filter directories based on the 'single' flag selection strategy
    subdirs = []
    for d in base_path.iterdir():
        if d.is_dir() and dir_pattern.match(d.name):
            if single_job:
                # Strictly look for an exact match
                if d.name == start_dir:
                    subdirs.append(d)
            else:
                # Filter directories that are chronologically >= start_dir
                if d.name >= start_dir:
                    subdirs.append(d)

    # Sort directories chronologically
    subdirs.sort(key=lambda x: x.name)

    if not subdirs:
        if single_job:
            print(f"No subdirectory found exactly matching: {start_dir}")
        else:
            print(f"No subdirectories found matching or newer than: {start_dir}")
        return

    if single_job:
        print(f"Queueing single target directory: {start_dir}\n")
    else:
        print(
            f"Found {len(subdirs)} target directory(ies) starting from {start_dir}\n"
        )

    for subdir in subdirs:
        models_dir = subdir / "models"
        if not models_dir.exists():
            print(f"Skipping {subdir.name}: 'models' folder missing.")
            continue

        # Look for net_X.pt files and extract the maximum epoch number
        epochs = []
        for file in models_dir.glob("net_*.pt"):
            match = re.match(r"net_(\d+)\.pt", file.name)
            if match:
                epochs.append(int(match.group(1)))

        if not epochs:
            print(f"Skipping {subdir.name}: No net_X.pt files found.")
            continue

        latest_epoch = max(epochs)
        print(
            f"Target Discovered -> Folder: {subdir.name} | Max Epoch: {latest_epoch}"
        )

        # Build runtime environment dictionary to feed into bsub
        env = os.environ.copy()
        env["TRAINING_FOLDER"] = str(subdir)
        env["EPOCH"] = str(latest_epoch)

        try:
            # Read the base .sh script contents
            with open("RunProfilerInference.sh", "r") as f:
                script_content = f.read()

            # Pipe the script contents directly to bsub via stdin while injecting env variables
            process = subprocess.run(
                ["bsub"],
                input=script_content,
                text=True,
                capture_output=True,
                env=env,
                check=True,
            )
            print(f" -> {process.stdout.strip()}\n")

        except subprocess.CalledProcessError as e:
            print(
                f" -> Failed to submit job for {subdir.name}. Error: {e.stderr.strip()}\n"
            )
        except FileNotFoundError:
            print("Error: RunProfilerInference.sh could not be found.")
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Queue up inference jobs using bsub based on chronological or single directory targets."
    )
    
    # Arguments
    parser.add_argument(
        "target_dir", 
        type=str, 
        help="The target subdirectory name (e.g., 20260626-054358)."
    )
    parser.add_argument(
        "base_dir", 
        type=str, 
        nargs="?", 
        default="./training_results", 
        help="Base directory containing the results (default: ./training_results)."
    )
    
    # Flags
    parser.add_argument(
        "-s", "--single", 
        action="store_true", 
        help="Queue ONLY the exact folder matching target_dir, instead of processing sequentially onward."
    )

    args = parser.parse_args()

    queue_jobs(args.base_dir, args.target_dir, single_job=args.single)