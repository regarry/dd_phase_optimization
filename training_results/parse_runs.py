#!/usr/bin/env python3
import os
import argparse
import re
import yaml
import csv
from datetime import datetime

# Regex to match yyyymmdd-hhmmss pattern
DIR_PATTERN = re.compile(r'^\d{8}-\d{6}$')

# Regex to extract the epoch number from 'net_xxx.pt' (with no leading zero constraint)
MODEL_PATTERN = re.compile(r'^net_(\d+)\.pt$')

# Parameters to always ignore by default (including nested matches)
DEFAULT_IGNORED_PARAMS = [
    "training_results_dir",
]

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare config.yaml files across date-time directories, count trained epochs, and differentiate jobs."
    )
    parser.add_argument(
        "datetime",
        type=str,
        help="The target datetime in 'yyyymmdd-hhmmss' format."
    )
    parser.add_argument(
        "-s", "--single",
        action="store_true",
        help="Process only the exact datetime specified. If omitted, processes this datetime and all newer ones."
    )
    parser.add_argument(
        "-d", "--dir",
        type=str,
        default=".",
        help="The parent directory containing the timestamped subfolders (default: current directory)."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="run_comparison_summary.csv",
        help="The name of the output summary file. Use '.csv' for spreadsheet compatibility or '.txt' for aligned text."
    )
    parser.add_argument(
        "-i", "--ignore",
        type=str,
        nargs="+",
        default=[],
        help="Space-separated list of parameter keys to ignore (e.g., -i learning_rate batch_size)."
    )
    parser.add_argument(
        "-m", "--min-epochs",
        type=int,
        default=None,
        help="Filter out any run directories that have trained for fewer than this many epochs."
    )
    return parser.parse_args()

def get_valid_directories(parent_dir):
    """Finds all subdirectories that match the yyyymmdd-hhmmss pattern."""
    dirs = []
    for item in os.listdir(parent_dir):
        if os.path.isdir(os.path.join(parent_dir, item)) and DIR_PATTERN.match(item):
            dirs.append(item)
    return sorted(dirs)

def flatten_dict(d, parent_key='', sep='.'):
    """Flattens nested dictionaries so we can easily compare deep nested configurations."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_max_epochs(run_dir):
    """Parses the 'models' subdirectory to find the highest epoch number in net_xxx.pt."""
    models_dir = os.path.join(run_dir, "models")
    if not os.path.isdir(models_dir):
        return 0
    
    epochs = []
    for file in os.listdir(models_dir):
        match = MODEL_PATTERN.match(file)
        if match:
            epochs.append(int(match.group(1)))
            
    return max(epochs) if epochs else 0

def main():
    args = parse_arguments()
    
    ignored_keys = set(DEFAULT_IGNORED_PARAMS + args.ignore)

    # 1. Parse and validate the input datetime
    try:
        target_dt = datetime.strptime(args.datetime, "%Y%m%d-%H%M%S")
    except ValueError:
        print(f"Error: Provided datetime '{args.datetime}' does not match format 'yyyymmdd-hhmmss'")
        return

    # 2. Collect all valid timestamp directories
    all_dirs = get_valid_directories(args.dir)
    if not all_dirs:
        print(f"No directories matching 'yyyymmdd-hhmmss' found in '{args.dir}'.")
        return

    # 3. Filter directories based on user criteria (Datetime checks)
    selected_dirs = []
    for d in all_dirs:
        dir_dt = datetime.strptime(d, "%Y%m%d-%H%M%S")
        if args.single:
            if dir_dt == target_dt:
                selected_dirs.append(d)
        else:
            if dir_dt >= target_dt:
                selected_dirs.append(d)

    if not selected_dirs:
        print("No matching directories found with the specified criteria.")
        return

    # 4. Load YAML configs and compute epochs (applying min-epochs filtering)
    configs = {}
    epoch_counts = {}
    filtered_dirs = []
    all_keys = set()
    
    for d in selected_dirs:
        run_path = os.path.join(args.dir, d)
        
        # Calculate epochs first
        epochs = get_max_epochs(run_path)
        
        # Apply min-epochs filter if provided
        if args.min_epochs is not None and epochs < args.min_epochs:
            continue
            
        epoch_counts[d] = epochs
        filtered_dirs.append(d)
        
        config_path = os.path.join(run_path, "config.yaml")
        if not os.path.exists(config_path):
            print(f"Warning: {config_path} not found. Skipping config values.")
            continue
            
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f) or {}
                flat_data = flatten_dict(data)
                configs[d] = flat_data
                all_keys.update(flat_data.keys())
        except Exception as e:
            print(f"Error reading {config_path}: {e}")

    if not filtered_dirs:
        print(f"No runs met the minimum epoch threshold of {args.min_epochs}.")
        return

    print(f"Processing {len(filtered_dirs)} directories (after filtering out runs below {args.min_epochs or 0} epochs)...")

    # 5. Identify variables that actually differ across the remaining runs
    differing_keys = []
    for key in sorted(all_keys):
        if any(key == ig or key.startswith(ig + ".") for ig in ignored_keys):
            continue

        values = set()
        for d in filtered_dirs:
            if d not in configs:
                continue
            val = configs[d].get(key, None)
            if isinstance(val, list):
                val = tuple(val)
            values.add(val)
        
        if len(values) > 1:
            differing_keys.append(key)

    # 6. Pair/Group Duplicates among remaining configurations
    signature_groups = {}
    for d in filtered_dirs:
        if d not in configs:
            continue
        signature = tuple((k, str(configs[d].get(k, ""))) for k in differing_keys)
        signature_groups.setdefault(signature, []).append(d)

    # Assign duplicate group labels and build sorted order
    duplicate_labels = {}
    sorted_display_dirs = []
    group_counter = 1

    for signature, dirs in signature_groups.items():
        if len(dirs) > 1:
            label = f"Group {group_counter}"
            group_counter += 1
        else:
            label = "Unique"
            
        for d in dirs:
            duplicate_labels[d] = label
            sorted_display_dirs.append(d)

    # 7. Write out the results
    is_csv = args.output.lower().endswith('.csv')

    if is_csv:
        # --- WRITE CSV ---
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Datetime", "Duplicate_Group", "Epochs_Trained"] + differing_keys)
            
            for d in sorted_display_dirs:
                row = [d, duplicate_labels[d], epoch_counts.get(d, 0)] + [str(configs[d].get(k, "")) for k in differing_keys]
                writer.writerow(row)
        print(f"Successfully exported CSV summary to: {args.output}")

    else:
        # --- WRITE ALIGNED TXT (Clean Table Format) ---
        col_widths = {
            "Datetime": 19,
            "Duplicate_Group": 15,
            "Epochs_Trained": 14
        }
        for key in differing_keys:
            max_val_len = max([len(str(configs[d].get(key, ""))) for d in filtered_dirs if d in configs] + [len(key)])
            col_widths[key] = max_val_len

        with open(args.output, "w", encoding="utf-8") as out:
            if not differing_keys:
                out.write("Datetime             | Epochs_Trained | Differentiators\n")
                out.write("-" * 65 + "\n")
                for d in filtered_dirs:
                    out.write(f"{d:19} | {epoch_counts.get(d, 0):<14} | [All non-ignored parameters are identical]\n")
            else:
                # Header
                header_parts = [
                    f"{'Datetime':<{col_widths['Datetime']}}",
                    f"{'Duplicate_Group':<{col_widths['Duplicate_Group']}}",
                    f"{'Epochs_Trained':<{col_widths['Epochs_Trained']}}"
                ]
                header_parts.extend([f"{k:<{col_widths[k]}}" for k in differing_keys])
                header = " | ".join(header_parts)
                out.write(header + "\n")
                out.write("-" * len(header) + "\n")
                
                # Rows
                for d in sorted_display_dirs:
                    row_parts = [
                        f"{d:<{col_widths['Datetime']}}",
                        f"{duplicate_labels[d]:<{col_widths['Duplicate_Group']}}",
                        f"{str(epoch_counts.get(d, 0)):<{col_widths['Epochs_Trained']}}"
                    ]
                    row_parts.extend([f"{str(configs[d].get(k, '')):<{col_widths[k]}}" for k in differing_keys])
                    out.write(" | ".join(row_parts) + "\n")
        print(f"Successfully exported aligned text summary to: {args.output}")

if __name__ == "__main__":
    main()
    
    # python parse_runs.py 20260710-164915 -o summary.txt -m 16