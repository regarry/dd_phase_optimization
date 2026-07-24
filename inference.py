import os
import sys
import time
import argparse
import numpy as np
import torch
import random
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from datetime import datetime
import skimage
from skimage import io, img_as_ubyte
import matplotlib.pyplot as plt
from pathlib import Path

# --- Local Imports ---
from data.io import expand_config, load_config, makedirs, save_png, savePhaseMask
from data.datasets import SyntheticMicroscopeData, ValidationDataset
from models.wrappers import ParallelEndToEndModel
from data.preprocessing import generate_bead_templates
from utils.debug import MemorySnapshot, print_all_gpu_stats
from physics.masks import get_initial_phase_mask
from physics.simulation import TotalVariationLoss
from data.transforms import batch_xyz_to_boolean_grid
from utils.metrics import compute_and_log_metrics

def inference_one_epoch(model, dataloader, mask_param, config, out_dir):
    model.eval() 
    main_device = torch.device(config.get('cnn_device', 'cuda:0'))
    
    with torch.no_grad():
        for batch_idx, (bead_xyz_list, targets) in enumerate(dataloader):
            # Move inputs to device
            bead_xyz_list = bead_xyz_list.to(main_device)
            targets = targets.to(main_device)
            
            # Target preprocessing for multi-class
            if config['num_classes'] > 1:
                if targets.dim() == 4 and targets.shape[1] == 1:
                    targets = targets.squeeze(1).long()

            # Model Forward Pass
            if config.get('SpatialMulitWellLoss', False):
                logits, column_sums = model(mask_param, bead_xyz_list)
            else:
                logits = model(mask_param, bead_xyz_list)
                
            # --- PROCESS PREDICTIONS & PROBABILITIES ---
            if config['num_classes'] == 3:
                probs = torch.softmax(logits, dim=1)  # Shape: (B, 3, H, W)
                cnn_img = torch.argmax(probs, dim=1)  # Shape: (B, H, W)
                
                # Take first batch item, transpose from (3, H, W) -> (H, W, 3) for RGB
                prob_map = probs[0].detach().cpu().numpy()
                rgb_prob_image = np.transpose(prob_map, (1, 2, 0))
                
                # Scale float probabilities [0, 1] to [0, 255] uint8 for standard image viewing
                gray_data = (rgb_prob_image * 255).astype(np.uint8)
                
            elif config['num_classes'] == 1:
                probs = torch.sigmoid(logits)
                cnn_img = (probs > 0.5).long() # threshold for metrics if needed
                gray_data = probs.squeeze().detach().cpu().numpy()
            else:
                raise ValueError(f"Unsupported num_classes: {config['num_classes']}")
            
            # Save the Prediction/Probability Image
            out_path = os.path.join(out_dir, f"inference_{batch_idx}.tif")
            io.imsave(out_path, gray_data)
            print(f"Saved inference for key {batch_idx} to {out_path}")
            
            # --- PROCESS GROUND TRUTH & METRICS ---
            print(f"Ground truth shape: {targets.shape}, dtype: {targets.dtype}")
            
            # Safe conversion of the entire batch to numpy for metric tracking
            batch_targets_np = targets.detach().cpu().numpy()
            
            if config['num_classes'] == 3:
                palette = np.array([
                    [255,   0,   0], # Class 0 -> Red
                    [  0, 255,   0], # Class 1 -> Green
                    [  0,   0, 255]  # Class 2 -> Blue
                ], dtype=np.uint8)
                
                # Explicitly isolate the first sample's 2D slice for visual palette mapping
                gt_img_2d = targets[0].squeeze().detach().cpu().numpy().astype(np.int64)
                rgb_gt_image = palette[gt_img_2d]
                
                # Log metrics using clean numpy arrays across the whole batch
                compute_and_log_metrics(
                    batch_targets_np, 
                    cnn_img.cpu().numpy(), 
                    out_dir, 
                    f"batch_{batch_idx}", 
                    num_classes=3
                )
            elif config['num_classes'] == 1:
                gt_img = targets.squeeze().detach().cpu().numpy()
                if gt_img.dtype == np.bool_:
                    gt_img = gt_img.astype(np.uint8)
                rgb_gt_image = gt_img
                
                compute_and_log_metrics(
                    batch_targets_np, 
                    gray_data, 
                    out_dir, 
                    f"batch_{batch_idx}", 
                    num_classes=1
                )
            else:
                raise ValueError(f"Unsupported num_classes: {config['num_classes']}")
                
            # Save Ground Truth Image
            gt_path = os.path.join(out_dir, f"ground_truth_{batch_idx}.tif")
            io.imsave(gt_path, rgb_gt_image)
            print(f"Saved ground truth for key {batch_idx} to {gt_path}")
            
            # --- OPTICAL/CAMERA SIMULATION LOGIC ---
            if config.get('SpatialMulitWellLoss', False):
                camera, column_sums = model.physics(mask_param, bead_xyz_list)
            else:
                camera = model.physics(mask_param, bead_xyz_list)
                
            camera_path = os.path.join(out_dir, f"camera_image_{batch_idx}.tif")
            camera_img = (camera.squeeze().detach().cpu().numpy() * config.get('camera_max_adu', 65535))
            
            # Ensure proper casting if saving as high-depth integer TIFF
            if config.get('camera_max_adu', 65535) > 255:
                camera_img = camera_img.astype(np.uint16)
            else:
                camera_img = camera_img.astype(np.uint8)
                
            io.imsave(camera_path, camera_img)
            print(f"Saved camera image for {batch_idx} to {camera_path}")

def run_inference(input_dir, epoch, res_dir="inference_results", num_inferences=1, plot_loss=False, model_path=""):
    """
    Programmatic module handling data loading, plotting diagnostics and evaluation loops natively.
    """
    config = load_config(os.path.join(input_dir, 'config.yaml'))
    config['px'] = float(config['px'])
    config['inference_epoch'] = epoch
    
    out_dir = os.path.join(res_dir, "inference")
    makedirs(out_dir)
    
    if plot_loss:
        train_loss_file = os.path.join(input_dir, "train_losses.txt")
        val_loss_file = os.path.join(input_dir, "val_losses.txt")
        if not os.path.exists(train_loss_file):
            print(f"train_losses.txt not found in {input_dir}")
        else:
            # 1. Reset the font size to default so it isn't massive and cramped
            plt.rcdefaults()
            plt.figure(figsize=(8, 5)) # Standard, comfortable figure size
            
            with open(train_loss_file, "r") as f:
                train_losses = [float(line.strip()) for line in f if line.strip()]
            plt.plot(train_losses, label="Training Loss")
            
            if os.path.exists(val_loss_file):
                with open(val_loss_file, "r") as f:
                    val_losses = [float(line.strip()) for line in f if line.strip()]
                plt.plot(val_losses, label="Validation Loss")
            else:
                print(f"val_losses.txt not found in {input_dir}. Plotting only training loss.")
            
            plt.xlabel("Epoch or Iteration")
            plt.ylabel("Loss")
            plt.title("Log Loss Over Time")
            plt.yscale("log")
            plt.legend()
            
            # 2. Fix the padding so labels don't get cut off at the borders
            plt.tight_layout()
            
            save_path = os.path.join(out_dir, "loss_plot.png")
            plt.savefig(save_path)
            plt.close() # Free memory and clear the active figure
            print(f"Loss plot saved to {save_path}")
            
    if not model_path:
        chosen_epoch = epoch
        model_file = f"net_{chosen_epoch}.pt"
        model_path_1 = Path(os.path.join(input_dir, "models", model_file))
        model_path_2 = Path(os.path.join(input_dir, model_file))
        if model_path_1.exists():
            model_path = str(model_path_1)
        elif model_path_2.exists():
            model_path = str(model_path_2)
        print(f"Automatically using CNN model: {model_path}")
        
    config["model_path"] = model_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mask_path_1 = Path(os.path.join(input_dir, f"mask_phase_epoch_{epoch}.tiff"))
    mask_path_2 = Path(os.path.join(input_dir, f"mask_phase_epoch_{epoch}.tif"))
    mask_path_3 = Path(os.path.join(input_dir, "learned_phase_masks", "tif", f"mask_phase_epoch_{epoch}.tif"))
    
    if mask_path_1.exists():
        mask_path = mask_path_1
    elif mask_path_2.exists():
        mask_path = mask_path_2
    elif mask_path_3.exists():
        mask_path = mask_path_3
    else:
        raise FileNotFoundError(f"Mask file not found for epoch {epoch} in {input_dir}")
    
    mask_np = skimage.io.imread(mask_path)
    mask_tensor = torch.from_numpy(mask_np).type(torch.FloatTensor).to(device)
    
    if mask_tensor.max() == 255:
        print("Converting mask from 8-bit to radians")
        mask_tensor = (mask_tensor / 255.0) * 2 * np.pi
    
    mask_param = torch.nn.Parameter(mask_tensor, requires_grad=False)
    
    config_output_path = os.path.join(out_dir, "config.yaml")
    with open(config_output_path, "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    labels_path = os.path.join(input_dir, "labels.pickle")
    with open(labels_path, 'rb') as f:
        labels_dict = pickle.load(f)
    
    val_samples = []
    for i in sorted(labels_dict.keys())[:num_inferences]:
        item = labels_dict[i]
        sample_pair = (np.array(item['xyz']), np.array(item['target']))
        val_samples.append(sample_pair)
        
    val_ds = ValidationDataset(val_samples)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    cnn_config = config.copy()
    cnn_config['skip_noise'] = False
    if config.get("use_unet", False):
        cnn_model = ParallelEndToEndModel(cnn_config)
        print("Instantiated Model")
    else:
        return
        
    cnn_model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    cnn_model.eval()

    inference_one_epoch(cnn_model, val_loader, mask_param, config, out_dir)

def main():
    parser = argparse.ArgumentParser(description="Combined inference for physical and CNN mask models.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--res_dir", type=str, default="inference_results")
    parser.add_argument("--num_inferences", type=int, default=1)
    parser.add_argument("--plot_loss", action="store_true")
    args = parser.parse_args()
    
    run_inference(
        input_dir=args.input_dir,
        epoch=args.epoch,
        res_dir=args.res_dir,
        num_inferences=args.num_inferences,
        plot_loss=args.plot_loss
    )

if __name__ == "__main__":
    main()