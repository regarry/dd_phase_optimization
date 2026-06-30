import numpy as np
import torch
import torch.nn.functional as F



def batch_xyz_to_ideal_image(psf_kernel, xyz_np, config):
    boolean_grid = batch_xyz_to_boolean_grid(xyz_np, config)
    # convolve boolean grid with ideal psf to get ideal image
    # 1. Reshape kernel to (1, 1, k, k) for conv2d compatibility
    kernel = psf_kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, k, k)
    ideal_image = F.conv2d(boolean_grid, kernel, padding='same')
    print("\n" + "="*50)
    print("SHAPE TRACE: batch_xyz_to_ideal_image")
    print(f"  1. PSF Kernel (original): {psf_kernel.shape}")
    print(f"  2. PSF Kernel (expanded): {kernel.shape}")
    print(f"  3. Boolean Grid:         {boolean_grid.shape} (Total: {boolean_grid.numel()})")
    print(f"  4. Ideal Image (output): {ideal_image.shape}  (Total: {ideal_image.numel()})")
    print("="*50 + "\n")
    scaled_ideal_image = ideal_image / (ideal_image.max() + 1e-8)
    return scaled_ideal_image

def batch_xyz_to_boolean_grid(xyz_np, config):

    image_volume = config["image_volume"]
    ratio_input_output_image_size = config["ratio_input_output_image_size"]
    z_range_cost_function = config["z_range_cost_function"]

    # number of particles
    batch_size, num_particles = xyz_np[:, :, 2].shape
    # set dimension
    H = image_volume[0]
    W = image_volume[1]
    D = 1

    boolean_grid = np.zeros((batch_size, D, H // int(ratio_input_output_image_size), \
                             W // int(ratio_input_output_image_size)))
    for i in range(batch_size):
        for j in range(num_particles):
            z = xyz_np[i, j, 2]
            if z_range_cost_function[0] <= z <= z_range_cost_function[1]:
                x = xyz_np[i, j, 1]
                y = xyz_np[i, j, 0]
                boolean_grid[i, 0, int(x // ratio_input_output_image_size), int(y // ratio_input_output_image_size)] = 1
    boolean_grid = torch.from_numpy(boolean_grid).type(torch.FloatTensor)
    return boolean_grid

def batch_xyz_to_3_class_grid(xyz, xyz_between_beads, config):
    """
    Converts batch xyz bead locations to a 1-channel categorical 2D grid.
    Pixel Values:
      0: Background
      1: Bead
      2: Between-bead connection
    Args:
        xyz: (batch_size, num_particles, 3) array of bead positions.
        xyz_between_beads: (batch_size, num_between_beads, 3) array of between-bead positions.
        config: dict with spatial constraints
    Returns:
        volume: (batch_size, 1, H, W) torch.LongTensor
    """
    z_range_cost_function = config["z_range_cost_function"]
    image_volume = config["image_volume"]  # [H, W, D]
    
    # Extract resolution scaling ratio to match the 1-class branch structure
    ratio = int(config.get("ratio_input_output_image_size", 1))
    H = image_volume[0] // ratio
    W = image_volume[1] // ratio
    
    batch_size, num_particles, _ = xyz.shape
    
    # Approach 1: Single-channel target filled with integer class IDs (Default: 0 for Background)
    volume = np.zeros((batch_size, 1, H, W), dtype=np.int64)
    
    for k in range(batch_size):
        # 1. Mark between-bead class FIRST (Class 2)
        if xyz_between_beads is not None and len(xyz_between_beads) > 0 and xyz_between_beads[k].size > 0:
            for m in range(len(xyz_between_beads[k])):
                raw_y = xyz_between_beads[k][m, 1]  # Index 1 is Y
                raw_x = xyz_between_beads[k][m, 0]  # Index 0 is X
                z = int(xyz_between_beads[k][m, 2])
                
                if z_range_cost_function[0] <= z <= z_range_cost_function[1]:
                    grid_y = int(raw_y // ratio)
                    grid_x = int(raw_x // ratio)
                    
                    if 0 <= grid_y < H and 0 <= grid_x < W:
                        volume[k, 0, grid_y, grid_x] = 2

        # 2. Mark beads SECOND (Class 1) - Overwrites connection if they share a pixel coordinate
        for j in range(num_particles):
            raw_y = xyz[k, j, 1]  # Index 1 is Y
            raw_x = xyz[k, j, 0]  # Index 0 is X
            z = int(xyz[k, j, 2])
            
            if z_range_cost_function[0] <= z <= z_range_cost_function[1]:
                grid_y = int(raw_y // ratio)
                grid_x = int(raw_x // ratio)
                
                if 0 <= grid_y < H and 0 <= grid_x < W:
                    volume[k, 0, grid_y, grid_x] = 1
                        
    # Return as LongTensor (required by nn.CrossEntropyLoss)
    return torch.from_numpy(volume).long()