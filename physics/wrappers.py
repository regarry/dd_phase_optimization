import torch
import torch.nn as nn
import math
from physics.simulation import OpticsSimulation
import numpy as np

class MultiGpuSimulation(nn.Module):
    """
    Scalable Physics Layer.
    Automatically handles Single-GPU, Multi-GPU, and CPU cases.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_gpus = torch.cuda.device_count()
        self.replicas = nn.ModuleList()
        self.camera_max_adu = torch.tensor(config['camera_max_adu'], dtype=torch.float32)
        self.debug = False
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
        
        noisy_imgs3D = self.replicas[0].noise(summed_image)
        final_image = noisy_imgs3D / self.camera_max_adu
        
        if self.debug:
            self.debug = False  # Only print once
            print("imgs3d max before noise and norm: ", torch.max(summed_image))
            print("final_image max after noise and norm: ", torch.max(final_image))
            
        if self.config.get('SpatialMulitWellLoss', False):
            lens_approach = self.config['lens_approach']
            phase_mask_upsample_factor = self.config.get('phase_mask_upsample_factor', 1)
            # Get z_min, z_max, y_min, y_max, and num_z_steps from config
            px_mm = self.config.get('px') * 1e3 # px in mm
            z_min_mm = self.config.get('smw_z_min_mm') # Default to -10 mm if not in config
            z_max_mm = self.config.get('smw_z_max_mm')  # Default to 10 mm if not in config
            y_min_mm = self.config.get('smw_y_min_mm')  # Default to -1 mm if not in config
            y_max_mm = self.config.get('smw_y_max_mm')   # Default to 1 mm if not in config
            num_z_steps = self.config.get('smw_num_z_steps') # Default to 200 steps if not in config

            # Convert mm values to pixel values for internal calculations
            z_min_pixels = int(z_min_mm / px_mm)
            z_max_pixels = int(z_max_mm / px_mm)
            y_min_pixels = int(y_min_mm / px_mm)
            y_max_pixels = int(y_max_mm / px_mm)

            # Calculate z_step_pixels
            if num_z_steps > 1:
                z_step_pixels = int((z_max_pixels - z_min_pixels) / (num_z_steps - 1))
            else:
                z_step_pixels = 1 # Handle case of single step to avoid division by zero

            # Ensure z_step_pixels is at least 1
            z_step_pixels = max(1, z_step_pixels)
            if not lens_approach == 'lazy_4f' and phase_mask_upsample_factor > 1:
                mask_tensor = OpticsSimulation.expand_matrix_kron_torch(mask_param, phase_mask_upsample_factor)
            else:
                mask_tensor = mask_param
        

            if mask_tensor.max() == 255:
                print("Converting mask from 8-bit to radians")
                mask_tensor = (mask_tensor / 255.0) * 2 * np.pi
            if lens_approach == 'against_lens':
                print("are you sure you didnt mean fourier lens?")
                output_layer = self.replicas[0].against_lens(mask_tensor)
            elif lens_approach == 'fourier_lens' or lens_approach == 'convolution':
                output_layer = self.replicas[0].fourier_lens(mask_tensor)
            elif lens_approach == 'lensless':
                output_layer = self.replicas[0].lensless(mask_tensor)
            elif lens_approach == '4f':
                output_layer = self.replicas[0].fourf(mask_tensor, self.replicas[0].pad_4f)
            elif lens_approach == '9f':
                output_layer = self.replicas[0].ninef(mask_tensor)
            elif lens_approach == 'lazy_4f':
                output_layer = self.replicas[0].lazy_fourf(mask_tensor)
            elif lens_approach == 'sample_4f':
                output_layer = self.replicas[0].sample_4f(mask_tensor)
            else:
                raise ValueError('lens approach not supported')

            # squeeze the output layer to remove singleton dimensions
            output_layer = output_layer.squeeze()
            column_sums = self.replicas[0].get_column_sums_array( 
                                               output_layer, 
                                               (z_min_pixels, z_max_pixels, z_step_pixels),
                                               (y_min_pixels, y_max_pixels), 
                                               asm = self.config['angular_spectrum_method']
                                             )
            return final_image, column_sums
        
        else:
            return final_image