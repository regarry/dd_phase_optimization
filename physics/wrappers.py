import torch
import torch.nn as nn
import math

# Import your components
from physics.simulation import OpticsSimulation  # The heavy engine
from models.unet import OpticsDesignUnet         # The reconstruction network

class MultiGpuSimulation(nn.Module):
    """
    Scalable Physics Layer.
    Splits the 'emitters' batch across all available GPUs to save memory,
    then sums the resulting images back onto the main device.
    """
    def __init__(self, config):
        super().__init__()
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus < 1:
            raise ValueError("No GPUs detected for Multi-GPU Simulation.")
        
        print(f"⚡ Initializing Physics Engine across {self.num_gpus} GPUs...")
        self.replicas = nn.ModuleList()
        
        # Create a copy of the simulation engine on each GPU
        for i in range(self.num_gpus):
            device = f'cuda:{i}'
            # Initialize and immediately move to specific GPU
            sim = OpticsSimulation(config).to(device)
            self.replicas.append(sim)

    def forward(self, mask_param, xyz, photons=None):
        batch_size, n_emitters, _ = xyz.shape
        
        # 1. Determine Split Size
        chunk_size = math.ceil(n_emitters / self.num_gpus)
        outputs = []
        
        # 2. Distribute Work
        for i in range(self.num_gpus):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_emitters)
            
            if start >= n_emitters: break
            
            device = f'cuda:{i}'
            
            # Slice Data & Move to Worker GPU
            xyz_slice = xyz[:, start:end, :].to(device)
            
            # Handle photons if they exist
            if photons is not None:
                photons_slice = photons[:, start:end].to(device)
            else:
                photons_slice = None

            # Move Mask (Critical for Autograd to track gradients back to GPU 0)
            mask_slice = mask_param.to(device)
            
            # Run Simulation
            # The replica is already on the correct device
            chunk_output = self.replicas[i](mask_slice, xyz_slice, photons_slice)
            outputs.append(chunk_output)

        # 3. Aggregate Results (All-Reduce)
        # Sum everything onto the main device (cuda:0)
        final_image = outputs[0].to('cuda:0')
        for i in range(1, len(outputs)):
            final_image = final_image + outputs[i].to('cuda:0')
            
        return final_image
