import numpy as np
import torch

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
                x = xyz_np[i, j, 0]
                y = xyz_np[i, j, 1]
                boolean_grid[i, 0, int(x // ratio_input_output_image_size), int(y // ratio_input_output_image_size)] = 1
    boolean_grid = torch.from_numpy(boolean_grid).type(torch.FloatTensor)
    return boolean_grid

def batch_xyz_to_3_class_grid(xyz, xyz_between_beads, config):
    """
    Converts batch xyz bead locations to a 3-class 3D volume for each batch.
    Class 0: background
    Class 1: bead
    Class 2: between-bead (direct z-line between z-coupled pairs)
    Args:
        xyz: (batch_size, num_particles, 3) array of bead positions.
        xyz_between_beads: (batch_size, num_between_beads, 3) array of between-bead positions.
        config: dict with keys 'image_volume' (list/tuple of [H, W, D])
    Returns:
        volumes: (batch_size, D, H, W) numpy array, values 0, 1, 2
    """
    z_range_cost_function = config["z_range_cost_function"]
    image_volume = config["image_volume"]  # [H, W, D]
    H, W, _ = image_volume
    batch_size, num_particles, _ = xyz.shape
    D = z_range_cost_function[1] - z_range_cost_function[0] + 1
    volume = np.zeros((batch_size, D, H, W))
    for k in range(batch_size):
        # Mark beads
        for j in range(num_particles):
            x = int(xyz[k, j, 0])
            y = int(xyz[k, j, 1])
            z = int(xyz[k, j, 2])
            if 0 <= x < H and 0 <= y < W and z_range_cost_function[0] <= z <= z_range_cost_function[1]:
                volume[k, z, x, y] = 1
        # Mark between-bead class
        for m in range(len(xyz_between_beads[k])):
            x = int(xyz_between_beads[k][m, 0])
            y = int(xyz_between_beads[k][m, 1])
            z = int(xyz_between_beads[k][m, 2])
            if 0 <= x < H and 0 <= y < W and z_range_cost_function[0] <= z <= z_range_cost_function[1]:
                volume[k, z, x, y] = 2
        volume = torch.from_numpy(volume).type(torch.FloatTensor)
    return volume