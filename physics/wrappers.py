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
        self.camera_max_adu = torch.tensor(config['camera_max_adu'], dtype=torch.float32)
        
        # CASE A: Multi-GPU or Single-GPU
        if self.num_gpus > 0:
            print(f"⚡ Initializing Physics Engine across {self.num_gpus} GPUs...")
            for i in range(self.num_gpus):
                device = f'cuda:{i}'
                sim = OpticsSimulation(config, device)
                params = list(sim.named_parameters())
                print(list(sim.named_parameters()))
                if len(params) == 0:
                    print("✅ No internal weights found. Your current manual loop is actually SAFE.")
                    print("   (However, DDP is still usually faster/more efficient).")
                else:
                    print(f"⚠️ Found {len(params)} internal weights! You MUST use DDP.")
                    for name, param in params:
                        print(f"   - {name}: {param.shape}")
                self.replicas.append(sim)
        # CASE B: CPU Only (Fallback)
        else:
            print("⚠️ No GPU detected. Running on CPU (this will be slow).")
            self.replicas.append(OpticsSimulation(config)) # Default device is CPU

    def forward(self, mask_param, xyz):
        # 2. Multi-GPU Logic (Emitter Parallelism)
        batch_size, n_emitters, _ = xyz.shape
        if self.num_gpus > 0:
            chunk_size = math.ceil(n_emitters / self.num_gpus)
            outputs = []
            
            for i in range(self.num_gpus):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, n_emitters)
                
                if start >= n_emitters: break
                
                device = f'cuda:{i}'
                
                # Slice & Move
                xyz_slice = xyz[:, start:end, :].to(device)
                mask_slice = mask_param.to(device)
                
                # Run Replica
                chunk_output = self.replicas[i](mask_slice, xyz_slice)
                outputs.append(chunk_output)

            # 3. Aggregate Results (All-Reduce to GPU 0)
            summed_image = outputs[0].to('cuda:0')
            self.camera_max_adu = self.camera_max_adu.to('cuda:0')
            for i in range(1, len(outputs)):
                summed_image += outputs[i].to('cuda:0')
        else:
            # Single CPU Replica
            summed_image = self.replicas[0](mask_param, xyz)
        # add noise and normalize
        #print("imgs3d max before noise and norm: ", torch.max(summed_image))
        noisy_imgs3D = self.replicas[0].noise(summed_image)
        noisy_clamped_imgs3d = torch.clamp(noisy_imgs3D, min=0.0)
        noisy_clamped_imgs3d = torch.clamp(noisy_clamped_imgs3d, max=self.camera_max_adu)
        final_image = noisy_imgs3D / self.camera_max_adu
        #print("final_image max after noise and norm: ", torch.max(final_image))
        return final_image