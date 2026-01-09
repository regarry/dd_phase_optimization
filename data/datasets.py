import torch
from torch.utils.data import Dataset
import numpy as np
from .generators import create_random_emitters

class SyntheticMicroscopeData(Dataset):
    def __init__(self, epoch_length, config):
        """
        epoch_length: How many 'batches' to simulate per epoch. 
                      Since data is infinite, we just pick a number (e.g., 1000).
        """
        self.length = epoch_length
        self.config = config

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. GENERATE FRESH DATA INSTANTLY
        xyz, between_beads = create_random_emitters(self.config)
        
        # 2. Generate Photons (Simple logic from your snippet)
        n_beads = xyz.shape[0]
        # Example photon count logic
        photons = np.random.randint(10000, 10001, size=n_beads).astype(np.float32)

        # 3. Return Tensors
        # Note: We return 'between_beads' only if your model uses it, 
        # otherwise usually you just need xyz and photons.
        sample = {
            'xyz': torch.from_numpy(xyz).float(),      # Shape: (N, 3)
            'photons': torch.from_numpy(photons).float()
        }
        
        return sample