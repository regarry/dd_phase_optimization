import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio
import os
from .generators import create_random_emitters
from .transforms import batch_xyz_to_boolean_grid, batch_xyz_to_3_class_grid, batch_xyz_to_ideal_image

class SyntheticMicroscopeData(Dataset):
    def __init__(self, epoch_length, config):
        self.length = epoch_length
        self.config = config

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. GENERATE FRESH COORDINATES
        bead_xyz_list, between_bead_xyz_list = create_random_emitters(self.config)
        
        # 2. GENERATE TARGET (Ground Truth Volume)
        xyz_batch = bead_xyz_list[np.newaxis, ...] # Shape: (1, N, 3)
        
        if self.config.get('num_classes', 1) == 3:
             # Clean guard to pass None if no connections were randomly generated
             between_batch = between_bead_xyz_list[np.newaxis, ...] if between_bead_xyz_list.size > 0 else None
             target = batch_xyz_to_3_class_grid(xyz_batch, between_batch, self.config)
        else:
             # Standard Binary Case
             if self.config.get('convolve_psf_ground_truth', False):
                defocused_bead_stack_path = os.path.join(self.config['training_results_dir'], self.config.get('defocused_beads_filename'))
                ideal_psf = sio.loadmat(defocused_bead_stack_path)['defocus_beads'][0]
                ideal_psf = torch.from_numpy(ideal_psf).float()
                target = batch_xyz_to_ideal_image(ideal_psf, xyz_batch, self.config)
             else:
                target = batch_xyz_to_boolean_grid(xyz_batch, self.config)
                
        # 3. CLEANUP
        # Target shape goes from (1, 1, H, W) -> (1, H, W)
        target = target.squeeze(0)
        
        # Convert inputs to float tensor
        xyz_tensor = torch.from_numpy(bead_xyz_list).float()
        
        return xyz_tensor, target
    
class ValidationDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]