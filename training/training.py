import os
import sys
import time
# Add the parent directory to sys.path so we can import 'data', 'models', etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
from skimage import io
import torch.nn.functional as F

# --- Local Imports ---
from data.io import expand_config, load_config, makedirs, save_png, savePhaseMask, save_normalized_png
from data.datasets import SyntheticMicroscopeData, ValidationDataset
from models.wrappers import ParallelEndToEndModel
from data.preprocessing import generate_bead_templates
from utils.debug import MemorySnapshot, print_all_gpu_stats
from physics.masks import get_initial_phase_mask
from physics.simulation import TotalVariationLoss, apply_8bit_physics

def train_one_epoch(model, dataloader, optimizer, criterion_ce, criterion_dice, tv_loss, mask_param, config, epoch):
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
    
    for batch_idx, (bead_xyz_list, targets) in enumerate(dataloader):
        # ---------------------------------------------------------
        # 1. MOVE DATA TO MAIN GPU
        # ---------------------------------------------------------
        # xyz shape from loader: (Batch, N_emitters, 3)
        # targets shape from loader: (Batch, Channels, H, W)
        bead_xyz_list = bead_xyz_list.to(main_device)
        targets = targets.to(main_device)
        
        # Handle target dimensions (CrossEntropy expects specific shapes)
        if config['num_classes'] > 1:
            # If (Batch, 1, H, W), squeeze to (Batch, H, W) for CrossEntropy
            if targets.dim() == 4 and targets.shape[1] == 1:
                targets = targets.squeeze(1).long()
        else:
             # For BCE, we usually want (Batch, 1, H, W) or (Batch, H, W) depending on implementation
             pass
        # 1. APPLY HARDWARE PHYSICS
        # mask_param is continuous and unbounded.
        # physical_slm_phase is strictly 8-bit quantized.
        physical_slm_phase = apply_8bit_physics(mask_param)
        # ---------------------------------------------------------
        # 2. FORWARD PASS (Through Parallel Wrapper)
        # ---------------------------------------------------------
        # Wrapper acts as traffic controller.
        outputs = model(physical_slm_phase, bead_xyz_list)
        
        # ---------------------------------------------------------
        # 3. LOSS & OPTIMIZATION
        # ---------------------------------------------------------
        criterion_ce_loss = criterion_ce(outputs, targets)
        criterion_dice_loss = criterion_dice(outputs, targets)
        total_variation_loss = tv_loss(physical_slm_phase)
        oof_loss = 0.0
        if config.get('oof_loss_weight', 0.0) > 0.0:
            # Out-of-Focus (OOF) Light Penalty
            # Assume targets shape: (Batch, Channels, H, W) or (Batch, H, W)
            # outputs shape: (Batch, Channels, H, W) or (Batch, H, W)
            # Bead mask: 1 inside bead volume, 0 outside
            if targets.dim() == 4:
                # (Batch, 1, H, W) or (Batch, C, H, W)
                bead_mask = (targets > 0).float()  # 1 where bead exists
            else:
                bead_mask = (targets > 0).float().unsqueeze(1)  # (Batch, 1, H, W)

            # Sum intensity outside bead region
            oof_mask = 1.0 - bead_mask
            # outputs may be logits; use sigmoid if BCE, softmax if CE
            if config['num_classes'] > 1:
                # For multi-class, sum all non-background classes
                probs = torch.softmax(outputs, dim=1)
                # Assume class 0 is background, penalize all others outside bead
                oof_intensity = (probs[:, 1:, :, :] * oof_mask).sum()
            else:
                probs = torch.sigmoid(outputs)
                oof_intensity = (probs * oof_mask).sum()

            oof_weight = config.get('oof_loss_weight', 1.0)
            oof_loss = oof_weight * oof_intensity / outputs.numel()

        loss = criterion_ce_loss + total_variation_loss + oof_loss + criterion_dice_loss
        
        loss.backward()
        
        if (batch_idx + 1) % grad_accum == 0 or batch_idx == (len(dataloader) - 1):
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item()
        
        # if batch_idx % 10 == 0:
        #     pass
        #     #print(f"Epoch [{epoch+1}], Step [{batch_idx+1}], Loss: {loss.item():.4f}")
            
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion_ce, criterion_dice, tv_loss, mask_param, config, epoch):
    """
    Runs one epoch of validation.
    """
    model.eval() # Set model to evaluation mode
    
    total_loss = 0.0
    main_device = torch.device(config.get('cnn_device', 'cuda:0'))
    
    # Disable gradient calculation for validation
    with torch.no_grad():
        # 1. APPLY HARDWARE PHYSICS
        # mask_param is continuous and unbounded.
        # physical_slm_phase is strictly 8-bit quantized.
        physical_slm_phase = apply_8bit_physics(mask_param)
        
        # We include TV loss in validation so the number compares 1:1 with training loss
        total_variation_loss = tv_loss(physical_slm_phase)
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
            logits = model(physical_slm_phase, bead_xyz_list)
            
            # ---------------------------------------------------------
            # 3. LOSS CALCULATION
            # ---------------------------------------------------------
            criterion_ce_loss = criterion_ce(logits, targets)
            criterion_dice_loss = criterion_dice(logits, targets)
            
            
            
            oof_loss = 0.0
            if config.get('oof_loss_weight', 0.0) > 0.0:
                # Re-using your exact OOF logic
                if targets.dim() == 4:
                    bead_mask = (targets > 0).float()
                else:
                    bead_mask = (targets > 0).float().unsqueeze(1)

                oof_mask = 1.0 - bead_mask
                
                if config['num_classes'] > 1:
                    probs = torch.softmax(logits, dim=1)
                    oof_intensity = (probs[:, 1:, :, :] * oof_mask).sum()
                else:
                    probs = torch.sigmoid(logits)
                    oof_intensity = (probs * oof_mask).sum()

                oof_weight = config.get('oof_loss_weight', 0.0)
                oof_loss = oof_weight * oof_intensity / logits.numel()

            # Sum all losses
            loss = criterion_ce_loss + criterion_dice_loss + total_variation_loss + oof_loss
            
            total_loss += loss.item()
            
            # Optional: Print less frequently during validation
            if batch_idx % 10 == 0:
                pass
                #print(f"Val Epoch [{epoch+1}], Step [{batch_idx+1}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"==> Validation Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
    
    return avg_loss

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
                            Default: 7
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                               Default: 0
            path (str): Path for the checkpoint to be saved to.
                               Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            #self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            # Loss didn't improve enough
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Loss improved
            self.best_loss = val_loss
            #self.save_checkpoint(val_loss, model)
            self.counter = 0
class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [Batch, 1, H, W] (Raw scores from model)
        targets: [Batch, H, W] or [Batch, 1, H, W] (0 or 1)
        """
        # 1. Apply Sigmoid (Not Softmax!)
        probs = torch.sigmoid(logits)
        
        # 2. Flatten targets to match probs
        targets = targets.view_as(probs).float()
        
        # 3. Calculate Intersection (Element-wise multiplication)
        #    Sum over spatial dims (Batch, Channel, H, W) -> Sum over (2,3)
        dims = (2, 3) 
        if probs.dim() == 5: dims = (2, 3, 4) # Handle 3D volumes
            
        intersection = (probs * targets).sum(dim=dims)
        cardinality = (probs + targets).sum(dim=dims)
        
        # 4. Dice Score
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        return 1.0 - dice_score.mean()
    
class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [Batch, Classes, D, H, W] (Raw scores, BEFORE softmax)
        targets: [Batch, D, H, W] (Integer class indices 0, 1, 2...)
        """
        # 1. Apply Softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        
        # 2. One-Hot Encode the targets to match probs shape
        #    Output: [Batch, Classes, D, H, W]
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # 3. Calculate Intersection and Union (per class, per batch)
        #    Sum over spatial dimensions (D, H, W) -> (Batch, Classes)
        dims = (2, 3) 
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)
        
        # 4. Calculate Dice Score
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # 5. Loss is 1 - Dice (Averaged over classes and batch)
        return 1.0 - dice_score.mean()

def main():
    start_time = time.time()
    # 1. Load & Expand Config
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.yaml', help='Path to config')
    args = parser.parse_args()
    
    raw_config = load_config(args.config)
    
    # 3. Directories
    model_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    training_results_dir = os.path.join(raw_config.get('training_data_path', 'training_results'), model_name)
    makedirs(training_results_dir)
    
    config = expand_config(raw_config, training_results_dir)

    # 2. Setup Seeding
    random_seed = config.get('random_seed', 42)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Save the FULL config (including the calculated values) for reproducibility
    with open(os.path.join(training_results_dir, 'config.yaml'), 'w') as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

    print(f"🚀 Training started: {model_name}")
    print(f"   Image Volume: {config['image_volume']}")
    print(f"   Bead Volume: {config['bead_volume']}")

    # 4. Pre-processing & Model
    generate_bead_templates(config, training_results_dir)
    
    model = ParallelEndToEndModel(config)
    
    main_device = torch.device(config.get('cnn_device', 'cuda:0'))
    initial_mask = get_initial_phase_mask(config)
    # save initial mask as a png scaled between 0-255
    save_normalized_png(initial_mask, os.path.join(training_results_dir, 'initial_phase_mask.png'))
    mask_param = nn.Parameter(torch.from_numpy(initial_mask).float().to(main_device))
    
    optimizer = Adam(list(model.parameters()) + [mask_param], lr=config['initial_learning_rate'])
    
    # 2. Setup Scheduler
    # mode='min': We want to reduce LR when loss stops decreasing (minimizing)
    # factor=0.5: When triggered, cut LR in half.
    # patience=10: Wait 10 epochs with no improvement before cutting LR.
    # verbose=True: Print a message when the LR changes.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config['learning_rate_scheduler_factor'], 
        patience=config['learning_rate_scheduler_patience']
    )
    
    early_stopper = EarlyStopping(
        patience=config['early_stopping_patience'], 
        min_delta=config['early_stopping_min_delta']
    )
    
    # Loss setup
    if config['num_classes'] == 3:
        weights = torch.tensor(config.get('weights', [1,1,1])).float().to(main_device)
        criterion_ce = nn.CrossEntropyLoss(weight=weights)
        print("Using MultiClassDiceLoss for 3-class segmentation.")
        criterion_dice = MultiClassDiceLoss()
    elif config['num_classes'] == 1:
        criterion_ce = nn.BCEWithLogitsLoss()
        print("Using standard Dice Loss for binary segmentation.")
        criterion_dice = BinaryDiceLoss()  
    else:
        raise ValueError("Unsupported number of classes. Only 1 (binary) or 3 (multi-class) are supported.")
    
    tv_loss = TotalVariationLoss(weight=config.get('tv_loss_weight', 0.0))

    # 5. Training Dataloader
    train_steps_per_epoch = int(config['ntrain'] / config['batch_size'])
    train_ds = SyntheticMicroscopeData(epoch_length=train_steps_per_epoch, config=config)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    
    # Validation Dataloader
    val_samples = []
    labels_dict = {}
    val_steps_per_epoch = int(config['nvalid'] / config['batch_size'])
    print("Caching validation data...")
    for i in range(val_steps_per_epoch):
        # This calls your random generation logic
        data_pair = train_ds[i] 
        val_samples.append(data_pair)
        xyz = data_pair[0].tolist()
        target = data_pair[1].tolist()
        labels_dict[i] = {'xyz': xyz, 'target':target}
         
    path_labels = os.path.join(training_results_dir,'labels.pickle')
    with open(path_labels, 'wb') as handle:
        pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    val_ds = ValidationDataset(val_samples)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    
    # 6. Loop
    train_losses = []
    val_losses = []
    with MemorySnapshot(os.path.join(training_results_dir, "crash_snapshot.pickle")) as snapshot:
        for epoch in range(config['max_epochs']):
            loss = train_one_epoch(model, train_loader, optimizer, criterion_ce, criterion_dice, 
                                   tv_loss, mask_param, config, epoch)
            train_losses.append(loss)
            val_loss = validate_one_epoch(model, val_loader, criterion_ce, criterion_dice, tv_loss, mask_param, config, epoch)
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            early_stopper(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"🟢 Epoch {epoch+1} Loss: {loss:.4f}| Val Loss: {val_loss:.4f}| LR: {current_lr:.2e}")
            if epoch == 0:
                print_all_gpu_stats()
                # also save the gpu stats to txt file
                with open(os.path.join(training_results_dir, "gpu_stats.txt"), "w") as f:
                    # save gpu statistics to text
                    if torch.cuda.is_available():
                        num_devices = torch.cuda.device_count()
                        print("-" * 30,file=f)
                        for i in range(num_devices):
                            prop = torch.cuda.get_device_properties(i)
                            allocated = torch.cuda.memory_allocated(i) / 1024**2
                            reserved = torch.cuda.memory_reserved(i) / 1024**2
                            print(f"GPU {i}: {prop.name} | Alloc: {allocated:.2f} MB | Res: {reserved:.2f} MB | Cap: {prop.total_memory / 1024**2:.2f} MB", file=f)
                        print("-" * 30, file=f)
                snapshot.dump()
                snapshot.end()
                
            with torch.no_grad():
                max_stroke = 2 * np.pi # Or whatever your physical SLM's max stroke is
                
                # 2. Wrap the raw parameter just like in training
                wrapped_phase = torch.remainder(mask_param, max_stroke)
                
                # 3. Scale it to the 8-bit 0-255 range
                scaled_to_bits = (wrapped_phase / max_stroke) * 255.0
                
                # 4. Round to exact integer levels
                rounded_bits = torch.round(scaled_to_bits)
                
                # 5. Convert to standard 8-bit image format (uint8 array)
                # Move it off the GPU, convert to numpy, and change type to uint8
                slm_display_image = rounded_bits.cpu().numpy().astype(np.uint8)
            
            # Save artifacts
            np.savetxt(os.path.join(training_results_dir, 'train_losses.txt'), train_losses, delimiter=',')
            np.savetxt(os.path.join(training_results_dir, 'val_losses.txt'), val_losses, delimiter=',')
            
            epoch_png_path = os.path.join(training_results_dir,"learned_phase_masks","png",f"mask_phase_epoch_{epoch}.png")
            epoch_tif_path = os.path.join(training_results_dir,"learned_phase_masks","tif",f"mask_phase_epoch_{epoch}.tif")
            epoch_bmp_path = os.path.join(training_results_dir,"learned_phase_masks","bmp",f"mask_phase_epoch_{epoch}.bmp")
            
            # create directories if they don't exist
            os.makedirs(os.path.dirname(epoch_png_path), exist_ok=True)
            os.makedirs(os.path.dirname(epoch_tif_path), exist_ok=True)
            os.makedirs(os.path.dirname(epoch_bmp_path), exist_ok=True)
            
            #io.imsave(epoch_tif_path, mask_param.detach().cpu().numpy().astype(np.float32))
            io.imsave(epoch_tif_path, slm_display_image)
            io.imsave(epoch_bmp_path, slm_display_image)
            save_png(slm_display_image, epoch_png_path, config)
            #savePhaseMask(slm_display_image, epoch, training_results_dir)
            
            if epoch % config.get('save_epoch_interval', 5) == 0 or epoch == config['max_epochs'] - 1 or early_stopper.early_stop:
                epoch_model_path = os.path.join(training_results_dir, "models", f'net_{epoch}.pt')
                os.makedirs(os.path.dirname(epoch_model_path), exist_ok=True)
                torch.save(model.state_dict(), epoch_model_path)
                
            if early_stopper.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}. No improvement in validation loss for {early_stopper.patience} epochs.")
                break
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed/60/60:.2f} hours.")
if __name__ == '__main__':
    main()