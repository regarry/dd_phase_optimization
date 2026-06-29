import os
import re
import subprocess
import sys
from pathlib import Path


def queue_jobs(base_dir, start_dir):
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: Base directory '{base_dir}' does not exist.")
        return

    # Create logs folder to prevent LSF writing errors
    os.makedirs("./logs", exist_ok=True)

    # Matches standard YYYYMMDD-HHMMSS format
    dir_pattern = re.compile(r"^\d{8}-\d{6}$")

    # Filter directories that are chronologically >= start_dir
    subdirs = []
    for d in base_path.iterdir():
        if d.is_dir() and dir_pattern.match(d.name):
            if d.name >= start_dir:
                subdirs.append(d)

    # Sort directories chronologically
    subdirs.sort(key=lambda x: x.name)

    if not subdirs:
        print(f"No subdirectories found matching or newer than: {start_dir}")
        return

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
    if len(sys.argv) < 2:
        print("Usage: python queue_jobs.py <oldest_subdir_name> [base_dir]")
        print("Example: python queue_jobs.py 20260626-054358")
        sys.exit(1)

    start_directory = sys.argv[1]
    # Defaults to ./training_results if not specified
    base_directory = (
        sys.argv[2] if len(sys.argv) > 2 else "./training_results"
    )

    queue_jobs(base_directory, start_directory)