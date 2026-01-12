import torch
import torch.nn as nn
import math
from physics.simulation import OpticsSimulation

class MultiGpuSimulation(nn.Module):
    """
    Scalable Physics Layer.
    Automatically handles Single-GPU, Multi-GPU, and CPU cases.
    """
    def __init__(self, config):
        super().__init__()
        self.num_gpus = torch.cuda.device_count()
        self.replicas = nn.ModuleList()
        
        # CASE A: Multi-GPU or Single-GPU
        if self.num_gpus > 0:
            print(f"⚡ Initializing Physics Engine across {self.num_gpus} GPUs...")
            for i in range(self.num_gpus):
                device = f'cuda:{i}'
                sim = OpticsSimulation(config).to(device)
                self.replicas.append(sim)
        # CASE B: CPU Only (Fallback)
        else:
            print("⚠️ No GPU detected. Running on CPU (this will be slow).")
            self.replicas.append(OpticsSimulation(config)) # Default device is CPU

    def forward(self, mask_param, xyz, photons=None):
        # 1. Bypass complex logic if we only have 1 device (1 GPU or CPU)
        if len(self.replicas) == 1:
            # Ensure inputs are on the correct device (cuda:0 or cpu)
            device = next(self.replicas[0].parameters()).device
            return self.replicas[0](mask_param.to(device), xyz.to(device))

        # 2. Multi-GPU Logic (Emitter Parallelism)
        batch_size, n_emitters, _ = xyz.shape
        chunk_size = math.ceil(n_emitters / self.num_gpus)
        outputs = []
        
        for i in range(self.num_gpus):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_emitters)
            
            if start >= n_emitters: break
            
            device = f'cuda:{i}'
            
            # Slice & Move
            # Note: We ignore 'photons' here as discussed
            xyz_slice = xyz[:, start:end, :].to(device)
            mask_slice = mask_param.to(device)
            
            # Run Replica
            chunk_output = self.replicas[i](mask_slice, xyz_slice)
            outputs.append(chunk_output)

        # 3. Aggregate Results (All-Reduce to GPU 0)
        final_image = outputs[0].to('cuda:0')
        for i in range(1, len(outputs)):
            final_image = final_image + outputs[i].to('cuda:0')
            
        return final_image