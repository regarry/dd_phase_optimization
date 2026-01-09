import os
import sys
# Add the parent directory to sys.path so we can import 'data', 'models', etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import numpy as np
import torch
import random
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from datetime import datetime

# --- Local Imports ---
from data.io import expand_config, load_config, makedirs, save_png, savePhaseMask
from data.datasets import SyntheticMicroscopeData
from models.wrappers import ParallelEndToEndModel
from data.preprocessing import generate_bead_templates
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
    # 1. Load & Expand Config
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.yaml', help='Path to config')
    args = parser.parse_args()
    
    raw_config = load_config(args.config)
    config = expand_config(raw_config) # <--- ONE CLEAN LINE

    # 2. Setup Seeding
    random_seed = config.get('random_seed', 42)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # 3. Directories
    model_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    res_dir = os.path.join(config.get('training_data_path', 'training_results'), model_name)
    makedirs(res_dir)
    
    # Save the FULL config (including the calculated values) for reproducibility
    with open(os.path.join(res_dir, 'config.yaml'), 'w') as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

    print(f"🚀 Training started: {model_name}")
    print(f"   Image Volume: {config['image_volume']}")
    print(f"   Safe Bead Volume: {config['bead_volume']}")

    # 4. Pre-processing & Model
    generate_bead_templates(config)
    
    model = ParallelEndToEndModel(config)
    
    main_device = torch.device(config.get('cnn_device', 'cuda:0'))
    initial_mask = get_initial_phase_mask(config)
    mask_param = nn.Parameter(torch.from_numpy(initial_mask).float().to(main_device))
    
    optimizer = Adam(list(model.parameters()) + [mask_param], lr=config['initial_learning_rate'])
    
    # Loss setup
    if config['num_classes'] > 1:
        weights = torch.tensor(config.get('weights', [1,1,1])).float().to(main_device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    tv_loss = TotalVariationLoss(weight=config.get('tv_loss_weight', 0.0))

    # 5. Dataloader
    steps_per_epoch = int(config['ntrain'] / config['batch_size'])
    train_ds = SyntheticMicroscopeData(epoch_length=steps_per_epoch, config=config)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    # 6. Loop
    train_losses = []
    with MemorySnapshot(os.path.join(res_dir, "crash_snapshot.pickle")):
        for epoch in range(config['max_epochs']):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, 
                                   tv_loss, mask_param, config, epoch)
            train_losses.append(loss)
            print(f"🟢 Epoch {epoch+1} Loss: {loss:.4f}")
            
            # Save artifacts
            np.savetxt(os.path.join(res_dir, 'train_losses.txt'), train_losses, delimiter=',')
            save_png(mask_param.detach(), res_dir, str(epoch).zfill(3), config)
            savePhaseMask(mask_param, epoch, res_dir)
            
            if epoch % 5 == 0:
                torch.save(model.state_dict(), os.path.join(res_dir, f'net_{epoch}.pt'))

if __name__ == '__main__':
    main()