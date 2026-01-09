import os
import sys
# Add the parent directory to sys.path so we can import 'data', 'models', etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from datetime import datetime

# --- Local Imports ---
from data.io import load_config, makedirs, save_png, savePhaseMask
# NEW: Import the Live Dataset
from data.datasets import SyntheticMicroscopeData
# NEW: Import the Parallel Wrapper
from models.wrappers import ParallelEndToEndModel
# We still need templates for the physics engine to load, but NOT offline labels
#from data. import generate_bead_templates
from utils.debug import MemorySnapshot, print_all_gpu_stats
from physics.masks import get_initial_phase_mask
from physics.simulation import TotalVariationLoss

def train_one_epoch(model, dataloader, optimizer, criterion, tv_loss, mask_param, config, epoch):
    """
    Runs one epoch of training.
    model: The ParallelEndToEndModel (Wrapper on CPU)
    """
    model.train() # Sets sub-modules (UNet/Physics) to train mode
    
    total_loss = 0.0
    # The Wrapper expects inputs on the Main Device (usually cuda:0)
    # It will handle splitting to cuda:1, cuda:2 internally.
    main_device = torch.device(config.get('cnn_device', 'cuda:0'))
    grad_accum = config.get('gradient_accumulation_steps', 32)
    
    for batch_idx, (xyz, targets) in enumerate(dataloader):
        # ---------------------------------------------------------
        # 1. MOVE DATA TO MAIN GPU
        # ---------------------------------------------------------
        # xyz shape from loader: (Batch, N_emitters, 3)
        # targets shape from loader: (Batch, Channels, H, W)
        xyz = xyz.to(main_device)
        targets = targets.to(main_device)
        
        # Handle target dimensions (CrossEntropy expects specific shapes)
        if config['num_classes'] > 1:
            # If (Batch, 1, H, W), squeeze to (Batch, H, W) for CrossEntropy
            if targets.dim() == 4 and targets.shape[1] == 1:
                targets = targets.squeeze(1).long()
        else:
             # For BCE, we usually want (Batch, 1, H, W) or (Batch, H, W) depending on implementation
             pass

        # ---------------------------------------------------------
        # 2. FORWARD PASS (Through Parallel Wrapper)
        # ---------------------------------------------------------
        # Wrapper acts as traffic controller.
        outputs = model(mask_param, xyz)
        
        # ---------------------------------------------------------
        # 3. LOSS & OPTIMIZATION
        # ---------------------------------------------------------
        loss_main = criterion(outputs, targets)
        loss_reg = tv_loss(mask_param)
        loss = loss_main + loss_reg
        
        loss.backward()
        
        if (batch_idx + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}], Step [{batch_idx+1}], Loss: {loss.item():.4f}")
            
    return total_loss / len(dataloader)

def main():
    # 1. Setup & Config
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.yaml', help='Path to config')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Fix Pixel Size Scaling
    config['px'] = config['slm_px'] / config['phase_mask_upsample_factor']
    
    # Setup Directories
    model_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    res_dir = os.path.join(config.get('training_data_path', 'training_results'), model_name)
    makedirs(res_dir)
    print(f"🚀 Training started. Results: {res_dir}")

    # Save Config
    with open(os.path.join(res_dir, 'config.yaml'), 'w') as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

    # 2. Data Preparation
    # We NO LONGER generate offline labels. Data is live.
    # We DO generate bead templates (PSFs) because the physics engine needs to load them once.
    #generate_bead_templates(config)

    # 3. Model Initialization
    print("Initializing Parallel End-to-End Model...")
    # This wrapper stays on CPU. It orchestrates the Multi-GPU Physics and the GPU-0 UNet.
    model = ParallelEndToEndModel(config)
    
    # NOTE: Do NOT call model.to(device). 
    # The wrapper internally pinned the UNet to cuda:0 and Physics to cuda:0,1,2...

    # 4. Phase Mask Parameter
    # The mask usually lives on the Main Device (cuda:0). 
    # The wrapper copies it to other GPUs as needed.
    main_device = torch.device(config.get('cnn_device', 'cuda:0'))
    
    initial_mask = get_initial_phase_mask(config)
    save_png(initial_mask, res_dir, "initial_phase_mask", config)
    
    mask_tensor = torch.from_numpy(initial_mask).float().to(main_device)
    mask_param = nn.Parameter(mask_tensor)

    # 5. Optimizer
    # The optimizer finds parameters across all devices automatically.
    optimizer = Adam(list(model.parameters()) + [mask_param], lr=config['initial_learning_rate'])
    
    # 6. Loss Function
    if config['num_classes'] > 1:
        weights = torch.tensor(config.get('weights', [1,1,1])).float().to(main_device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    tv_loss = TotalVariationLoss(weight=config.get('tv_loss_weight', 0.0))

    # 7. Data Loaders (LIVE)
    print("Setting up Live Data Generation...")
    # Calculate how many batches make up one "epoch"
    steps_per_epoch = int(config['ntrain'] / config['batch_size'])
    
    # Initialize the Live Dataset
    train_ds = SyntheticMicroscopeData(epoch_length=steps_per_epoch, config=config)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        # CRITICAL: Use workers to generate math on CPU while GPU trains
        num_workers=config.get('num_workers', 4), 
        pin_memory=True
    )

    # 8. Training Loop
    print(f"Starting training for {config['max_epochs']} epochs...")
    train_losses = []
    
    # MemorySnapshot helps debug OOM errors
    with MemorySnapshot(os.path.join(res_dir, "crash_snapshot.pickle")):
        for epoch in range(config['max_epochs']):
            
            avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, 
                                   tv_loss, mask_param, config, epoch)
            
            train_losses.append(avg_loss)
            print(f"🟢 Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
            
            # Save Checkpoints & Viz
            np.savetxt(os.path.join(res_dir, 'train_losses.txt'), train_losses, delimiter=',')
            save_png(mask_param.detach(), res_dir, str(epoch).zfill(3), config)
            savePhaseMask(mask_param, epoch, res_dir)
            
            if epoch % 5 == 0:
                # Saves state dict of wrapper (which includes UNet + Physics)
                torch.save(model.state_dict(), os.path.join(res_dir, f'net_{epoch}.pt'))
                print_all_gpu_stats()

if __name__ == '__main__':
    main()