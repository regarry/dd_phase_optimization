import os
import sys
import time
# Add the parent directory to sys.path so we can import 'data', 'models', etc.
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
    """
    Runs one epoch of validation.
    """
    model.eval() # Set model to evaluation mode
    
    main_device = torch.device(config.get('cnn_device', 'cuda:0'))
    
    # Disable gradient calculation for validation
    with torch.no_grad():
        for batch_idx, (bead_xyz_list, targets) in enumerate(dataloader):
            # ---------------------------------------------------------
            # 1. MOVE DATA TO MAIN GPU
            # ---------------------------------------------------------
            bead_xyz_list = bead_xyz_list.to(main_device)
            targets = targets.to(main_device)
            
            # Handle target dimensions (Mirroring training logic)
            if config['num_classes'] > 1:
                if targets.dim() == 4 and targets.shape[1] == 1:
                    targets = targets.squeeze(1).long()
            else:
                pass

            # ---------------------------------------------------------
            # 2. FORWARD PASS
            # ---------------------------------------------------------
            logits = model(mask_param, bead_xyz_list)
            if config['num_classes'] == 3:
                probs = torch.softmax(logits, dim=1)
                cnn_img = torch.argmax(probs,dim=1)
                class_data = probs[0, :3, :, :].detach().cpu().numpy()
                rgb_image = np.transpose(class_data, (1, 2, 0))
            elif config['num_classes'] == 1:
                probs = torch.sigmoid(logits)
                gray_data = probs.squeeze().detach().cpu().numpy()
                rgb_image = gray_data * 255
                #cnn_img = (probs > 0.5).long()
            else:
                raise ValueError(f"Unsupported num_classes: {config['num_classes']}")
            
            # visualize the outputs and targets
            #out_img = out_img.detach().cpu().squeeze().numpy()
            out_path = os.path.join(out_dir, f"inference_{batch_idx}.tif")
            io.imsave(out_path, gray_data)
            print(f"Saved inference for key {batch_idx} to {out_path}")
            
            # Save ground truth label from boolean grid
            gt_img = targets
            print(f"Ground truth shape: {gt_img.shape}, dtype: {gt_img.dtype}")
            if torch.is_tensor(gt_img):
                gt_img = gt_img.squeeze().detach().cpu().numpy()
            if gt_img.dtype == np.bool_:
                gt_img = (gt_img.astype(np.uint8))
            if config['num_classes'] == 3:
                palette = np.array([
                    [255,   0,   0], # 0: Bright Red
                    [  0, 255,   0], # 1: Bright Green
                    [  0,   0, 255]  # 2: Bright Blue
                ], dtype=np.uint8)
                
                rgb_gt_image = palette[gt_img]
                compute_and_log_metrics(targets.cpu().numpy(), cnn_img.cpu().numpy(), out_dir, f"batch_{batch_idx}", num_classes=3)
            elif config['num_classes'] == 1:
                rgb_gt_image = gt_img
                compute_and_log_metrics(gt_img, gray_data, out_dir, f"batch_{batch_idx}", num_classes=1)
            else:
                raise ValueError(f"Unsupported num_classes: {config['num_classes']}")
            gt_path = os.path.join(out_dir, f"ground_truth_{batch_idx}.tif")
            io.imsave(gt_path, rgb_gt_image)
            print(f"Saved ground truth for key {batch_idx} to {gt_path}")
            
            # camera image
            camera = model.physics(mask_param, bead_xyz_list)
            camera_path = os.path.join(out_dir, f"camera_image_{batch_idx}.tif")
            camera_img = (camera.squeeze().detach().cpu().numpy() * config.get('camera_max_adu', 65535)).astype(np.uint16)
            io.imsave(camera_path, camera_img)
            print(f"Saved camera image for {batch_idx} to {camera_path}")
            

def main():
    """
    Main function to run combined inference for physical and CNN mask models.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Combined inference for physical and CNN mask models.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing config.yaml and labels.pickle")
    parser.add_argument("--epoch", type=int, required=True, help="Desired epoch for the mask (closest available at x*10+1)")
    parser.add_argument("--res_dir", type=str, default="inference_results", help="Directory to save inference outputs")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")
    parser.add_argument("--num_inferences", type=int, default=1, help="Number of samples for inference (if 0, use all keys)")
    parser.add_argument("--lens_approach", type=str, default="", help="Override lens_approach in config.yaml for PhysicalLayer")
    parser.add_argument("--empty_mask", action="store_true", help="Run inference with an empty mask")
    parser.add_argument("--paper_mask", type=str, default="", help="File path to paper phase mask for inference")
    parser.add_argument("--no_noise", action="store_true", help="Disable noise in PhysicalLayer inference")
    parser.add_argument("--model_path", type=str, default="", help="Optional: Path to the CNN pretrained model checkpoint")
    parser.add_argument("--beam_3d_sections", type=str, default="beam_3d_sections", help="Optional: Path to the beam 3d sections file")
    parser.add_argument("--generate_beam_profile", action="store_true", help="Generate beam profile for the input mask (default: off)")
    parser.add_argument("--x_min", type=int, default=-100, help="Minimum z value for beam section generation")
    parser.add_argument("--x_max", type=int, default=100, help="Maximum z value for beam section generation")
    parser.add_argument("--y_min", type=int, default=-50, help="Minimum y value for beam section generation")
    parser.add_argument("--y_max", type=int, default=50, help="Maximum y value for beam section generation")
    parser.add_argument("--max_intensity", type=float, help="Maximum intensity for the mask")
    parser.add_argument("--bead_volume", action="store_true", help="Save bead volume as tiff files")
    parser.add_argument("--plot_loss", action="store_true", help="Plot the training loss over time from train_losses.txt in input_dir")
    args = parser.parse_args()
    
    
    config = load_config(os.path.join(args.input_dir, 'config.yaml'))
    config['px'] = float(config['px'])
    config['inference_epoch'] = args.epoch
    
    # Create output directory for inference results using current datetime.
    dt_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.res_dir, "inference", dt_str)
    makedirs(out_dir)
    
    if args.plot_loss:
        train_loss_file = os.path.join(args.input_dir, "train_losses.txt")
        val_loss_file = os.path.join(args.input_dir, "val_losses.txt")
        if not os.path.exists(train_loss_file):
            print(f"train_losses.txt not found in {args.input_dir}")
        elif not os.path.exists(val_loss_file) and os.path.exists(train_loss_file):
            print(f"val_losses.txt not found in {args.input_dir}")
            print("Plotting only training loss.")
            with open(train_loss_file, "r") as f:
                train_losses = [float(line.strip()) for line in f if line.strip()]
            plt.figure()
            plt.plot(train_losses, label="Training Loss")
            plt.xlabel("Epoch or Iteration")
            plt.ylabel("Loss")
            plt.title("Log Loss Over Time")
            plt.yscale("log")
            plt.legend()
            save_path = os.path.join(out_dir, "loss_plot.png")
            plt.savefig(save_path)
            print(f"Loss plot saved to {save_path}")
            
        else:
            with open(train_loss_file, "r") as f:
                train_losses = [float(line.strip()) for line in f if line.strip()]
                
            with open(val_loss_file, "r") as f:
                val_losses = [float(line.strip()) for line in f if line.strip()]
            plt.figure()
            plt.plot(train_losses, label="Training Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.xlabel("Epoch or Iteration")
            plt.ylabel("Loss")
            plt.title("Log Loss Over Time")
            plt.yscale("log")
            plt.legend()
            save_path = os.path.join(out_dir, "loss_plot.png")
            plt.savefig(save_path)
            print(f"Loss plot saved to {save_path}")
    
    # Automatically determine CNN model path if not provided
    if not args.model_path:
        # x is an integer that begins at 0 and increases by 1.
        
        # x0 = int((args.epoch - 1) / 10)
        # candidate_low = x0 * 10 + 1
        # candidate_high = (x0 + 1) * 10 
        # if abs(args.epoch - candidate_low) <= abs(candidate_high - args.epoch):
        #     chosen_epoch = candidate_low
        # else:
        #     chosen_epoch = candidate_high
        chosen_epoch = args.epoch
        model_file = f"net_{chosen_epoch}.pt"
        args.model_path = os.path.join(args.input_dir, model_file)
        print(f"Automatically using CNN model: {args.model_path}")
        
    config["model_path"] = args.model_path
    learned_lens_approach = config['lens_approach']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load mask from tiff file (for both models)
    mask_path_1 = Path(os.path.join(args.input_dir, f"mask_phase_epoch_{args.epoch}.tiff"))
    mask_path_2 = Path(os.path.join(args.input_dir, f"mask_phase_epoch_{args.epoch}.tif"))
    mask_path_3 = Path(os.path.join(args.input_dir, "learned_phase_masks","tif",f"mask_phase_epoch_{args.epoch}.tif"))
    
    if mask_path_1.exists():
        mask_path = mask_path_1
    elif mask_path_2.exists():
        mask_path = mask_path_2
    elif mask_path_3.exists():
        mask_path = mask_path_3
    else:
        raise FileNotFoundError(f"Mask file not found for epoch {args.epoch} in {args.input_dir}")
    
    #mask_path = os.path.join(args.input_dir, f"mask_phase_epoch_{args.epoch}.tiff")
    mask_np = skimage.io.imread(mask_path)
    mask_tensor = torch.from_numpy(mask_np).type(torch.FloatTensor).to(device)
    
    if mask_tensor.max() == 255:
        print("Converting mask from 8-bit to radians")
        mask_tensor = (mask_tensor / 255.0) * 2 * np.pi
    
    mask_param = torch.nn.Parameter(mask_tensor, requires_grad=False)
    # Save updated configuration in out_dir, including inference epoch.
    config_output_path = os.path.join(out_dir, "config.yaml")
    with open(config_output_path, "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    # Load labels
    labels_path = os.path.join(args.input_dir, "labels.pickle")
    with open(labels_path, 'rb') as f:
        labels_dict = pickle.load(f)
    
    #make a dataaloader
    # Reconstructing val_samples from labels_dict
    val_samples = []

    # Sorting by key ensures the order remains the same as the original loop
    # i want to contstrain to 5 samples max for inference
    for i in sorted(labels_dict.keys())[:5]:
        item = labels_dict[i]
        # Converting back to numpy arrays (standard for most ML data_pairs)
        sample_pair = (np.array(item['xyz']), np.array(item['target']))
        val_samples.append(sample_pair)
    val_ds = ValidationDataset(val_samples)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    # Instantiate CNN model based on config and load checkpoint
    cnn_config = config.copy()
    cnn_config['skip_noise'] = False
    if config.get("use_unet", False):
        cnn_model = ParallelEndToEndModel(cnn_config)
        print("Instantiated ")
    else:
        pass
    cnn_model
    cnn_model.load_state_dict(torch.load(args.model_path, map_location="cpu"), strict=False)
    cnn_model.eval()

    
    inference_one_epoch(cnn_model, val_loader, mask_param, config, out_dir)
    
    train_loss_file = os.path.join(args.input_dir, "train_losses.txt")
    if not os.path.exists(train_loss_file):
        print(f"train_losses.txt not found in {args.input_dir}")
    else:
        with open(train_loss_file, "r") as f:
            train_losses = [float(line.strip()) for line in f if line.strip()]
        plt.figure()
        plt.plot(train_losses, label="Training Loss")
        plt.xlabel("Epoch or Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.yscale("log")
        plt.legend()
        save_path = os.path.join(out_dir, "train_loss.png")
        plt.savefig(save_path)
        print(f"Training loss plot saved to {save_path}")
    
    
    
if __name__ == "__main__":
    main()