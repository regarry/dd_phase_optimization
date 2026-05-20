import torch
from torch import nn
from physics.wrappers import MultiGpuSimulation
from models.unet import OpticsDesignUnet

class ParallelEndToEndModel(nn.Module):
    """
    The CPU-based conductor.
    1. Orchestrates the Multi-GPU Physics Simulation.
    2. Passes the result to the UNet on the Main GPU.
    """
    def __init__(self, config):
        super().__init__()
        
        # A. Hardware Setup
        self.config = config
        self.px = float(config['px'])
        self.main_device = torch.device('cuda:0')
        
        # B. Initialize Sub-Modules
        # 1. Physics (Manages its own GPUs internally)
        self.physics = MultiGpuSimulation(config)
        
        # 2. UNet (Standard CNN)
        self.unet = OpticsDesignUnet(config)
        
        # C. Pin UNet to Main GPU
        # We explicitly move ONLY the UNet. 
        print(f"🧠 Initializing UNet on {self.main_device}...")
        self.unet.to(self.main_device)

    def forward(self, mask_param, emitters):
        """
        mask_param: The learnable Phase Mask (usually on cuda:0)
        emitters:   (Batch, N, 3) Coordinates beads
        """
        # --- Handle UNet State First ---
        if self.config.get('freeze_unet', False):
            for param in self.unet.parameters():
                param.requires_grad = False
            self.unet.eval()
        else:
            # Ensure it's in training mode if not frozen
            self.unet.train()

        # --- Step 1: Physical Simulation (Multi-GPU) ---
        # Run physics once and unpack based on your configuration rules
        if self.config.get('SpatialMulitWellLoss', False):
            sensor_image, column_sums = self.physics(mask_param, emitters)
        else:
            sensor_image = self.physics(mask_param, emitters)
            column_sums = None
        
        # --- Step 2: Bead Prediction (Single GPU) ---
        # sensor_image is now guaranteed to be a single Tensor on cuda:0
        bead_prediction = self.unet(sensor_image)
        
        # --- Step 3: Balanced Returns ---
        if self.config.get('SpatialMulitWellLoss', False):
            return bead_prediction, column_sums
        else:
            return bead_prediction