import torch
import numpy as np
from torch.utils.data import Dataset
from .generators import create_random_emitters
from .transforms import batch_xyz_to_boolean_grid, batch_xyz_to_3_class_grid

class SyntheticMicroscopeData(Dataset):
    def __init__(self, epoch_length, config):
        """
        epoch_length: Arbitrary length (e.g., 1000) to define one 'epoch'.
        """
        self.length = epoch_length
        self.config = config

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. GENERATE FRESH COORDINATES
        # xyz shape: (N_emitters, 3)
        bead_xyz_list, between_bead_xyz_list = create_random_emitters(self.config)
        
        # 2. GENERATE TARGET (Ground Truth Volume)
        # The generator functions expect a Batch dimension (Batch, N, 3).
        # Since __getitem__ handles a single sample, we add a fake batch dim.
        xyz_batch = bead_xyz_list[np.newaxis, ...] # Shape becomes (1, N, 3)
        
        if self.config.get('num_classes', 1) > 1:
             # Handle 3-class case (Background, Bead, Connection)
             between_batch = between_bead_xyz_list[np.newaxis, ...] if between_bead_xyz_list is not None else None
             target = batch_xyz_to_3_class_grid(xyz_batch, between_batch, self.config)
        else:
             # Standard Binary Case
             target = batch_xyz_to_boolean_grid(xyz_batch, self.config)

        # 3. CLEANUP
        # Remove the fake batch dimension so the DataLoader can stack them properly later.
        # Target shape goes from (1, Channels, H, W) -> (Channels, H, W)
        target = target.squeeze(0)
        
        # Convert inputs to float tensor
        xyz_tensor = torch.from_numpy(bead_xyz_list).float()
        
        # 4. RETURN TUPLE (Matches your training loop structure)
        return xyz_tensor, target