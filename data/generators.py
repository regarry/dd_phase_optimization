import numpy as np
import random

import numpy as np
import random

def create_random_emitters(config):
    """
    Generates positions for ONE single image volume.
    Used by the Dataset class in __getitem__.
    """
    # Extract config
    num_particles_range = config['num_particles_range']
    
    # Handle spatial ranges (convert to list if they are [min, max] pairs)
    spatial_xy = config['particle_spatial_range_xy']
    spatial_z = config['particle_spatial_range_z']

    # Helper to convert [min, max] to range object if needed
    if isinstance(spatial_xy, (list, tuple)) and len(spatial_xy) == 2:
        spatial_xy_list = range(int(spatial_xy[0]), int(spatial_xy[1]))
    else:
        spatial_xy_list = spatial_xy

    if isinstance(spatial_z, (list, tuple)) and len(spatial_z) == 2:
         # Assuming z steps of 1. If step is in config, handle accordingly.
        spatial_z_list = range(int(spatial_z[0]), int(spatial_z[1]))
    else:
        spatial_z_list = spatial_z

    z_coupled_ratio = config.get('z_coupled_ratio', 0.0)
    z_coupled_spacing = config.get('z_coupled_spacing_range', None)

    # 1. Determine N particles (Random for EVERY image)
    num_particles = np.random.randint(num_particles_range[0], num_particles_range[1])
    
    # 2. Logic for Coupled vs Random
    n_z_coupled = int(num_particles * z_coupled_ratio // 2) * 2
    n_random = num_particles - n_z_coupled
    
    beads = []
    between_beads = [] 

    # A. Z-Coupled Pairs (Vertical "Filaments")
    for _ in range(n_z_coupled // 2):
        x = random.choice(spatial_xy_list)
        y = random.choice(spatial_xy_list)
        z1 = random.choice(spatial_z_list)
        
        # Determine Z2
        z2 = random.choice(spatial_z_list) # Fallback
        
        if z_coupled_spacing is not None:
            min_s, max_s = z_coupled_spacing
            # Find valid spacings that keep Z2 inside the volume
            valid_spacings = [s for s in range(min_s, max_s+1) if (z1 + s) in spatial_z_list]
            
            if valid_spacings:
                spacing = random.choice(valid_spacings)
                z2 = z1 + spacing
                
                # Add the "Connector" beads (points strictly between z1 and z2)
                for u in range(z1 + 1, z2):
                    between_beads.append([x, y, u])

        beads.append([x, y, z1])
        beads.append([x, y, z2])

    # B. Remaining Random Beads
    for _ in range(n_random):
        x = random.choice(spatial_xy_list)
        y = random.choice(spatial_xy_list)
        z = random.choice(spatial_z_list)
        beads.append([x, y, z])

    random.shuffle(beads)
    
    # Return as numpy arrays
    # Beads: (N, 3)
    # Between: (M, 3) or None
    beads_np = np.array(beads)
    
    between_np = None
    if len(between_beads) > 0:
        between_np = np.array(between_beads)
        
    return beads_np, between_np

'''
def generate_batch(batch_size, num_particles_range, particle_spatial_range_xy, particle_spatial_range_z, z_coupled_ratio, z_coupled_spacing_range):
    """
    Generate a batch of bead locations, with an option to include a mixture of random beads and z-coupled bead pairs.
    Args:
        batch_size: Number of images in the batch.
        num_particles_range: [min, max] number of beads per image.
        particle_spatial_range_xy: allowed x/y positions.
        particle_spatial_range_z: allowed z positions.
        seed: random seed.
        z_coupled_ratio: fraction of beads that are z-coupled pairs (0.0 = all random, 1.0 = all z-coupled).
        z_coupled_spacing_range: [min, max] allowed z spacing for z-coupled pairs (inclusive).
    Returns:
        xyz_grid: (batch_size, num_particles, 3) array of bead positions.
        Nphotons: (batch_size, num_particles) array of photon counts.
    """
    #if seed is not None:
    #    np.random.seed(seed)
    #    random.seed(seed)

    num_particles = np.random.randint(num_particles_range[0], num_particles_range[1], 1).item()
    #Nsig_range = [10000, 10001]  # in [counts]
    #Nphotons = np.random.randint(Nsig_range[0], Nsig_range[1], (batch_size, num_particles)).astype('float32')

    xyz_grid = np.zeros((batch_size, num_particles, 3)).astype('int')
    between_beads_grid = [] # this is a list of variable length
    for k in range(batch_size):
        n_z_coupled = int(num_particles * z_coupled_ratio // 2) * 2  # must be even, each pair is 2 beads
        n_random = num_particles - n_z_coupled
        beads = []
        between_beads = []
        # Generate z-coupled pairs
        for _ in range(n_z_coupled // 2):
            x = random.choice(particle_spatial_range_xy)
            y = random.choice(particle_spatial_range_xy)
            z1 = random.choice(particle_spatial_range_z)
            if z_coupled_spacing_range is not None:
                min_spacing, max_spacing = z_coupled_spacing_range
                possible_spacings = [s for s in range(min_spacing, max_spacing+1) if (z1 + s) in particle_spatial_range_z]
                if not possible_spacings:
                    # fallback: just pick another z
                    z2 = random.choice(particle_spatial_range_z)
                else:
                    spacing = random.choice(possible_spacings)
                    z2 = z1 + spacing
                    # here i want to append to a seperate list each between bead location
                    for u in range(z1+1,z2):
                        between_beads.append([x, y, u])
                    
            else:
                z2 = random.choice(particle_spatial_range_z)
            beads.append([x, y, z1])
            beads.append([x, y, z2])
        # Generate remaining random beads
        for _ in range(n_random):
            x = random.choice(particle_spatial_range_xy)
            y = random.choice(particle_spatial_range_xy)
            z = random.choice(particle_spatial_range_z)
            beads.append([x, y, z])
        # Shuffle to avoid ordering bias
        random.shuffle(beads)
        #random.shuffle(between_beads) 
        between_beads_grid.append(np.array(between_beads))
        xyz_grid[k, :, :] = np.array(beads)
    if n_z_coupled > 0:
        return xyz_grid, between_beads_grid
    else:
        return xyz_grid, None
        '''