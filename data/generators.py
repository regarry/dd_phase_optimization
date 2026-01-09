import numpy as np
import random

def create_random_emitters(config):
    """
    Generates positions for ONE single image volume.
    Refactored from your original 'generate_batch'
    """
    # Extract config
    num_particles_range = config['num_particles_range']
    spatial_xy = config['particle_spatial_range_xy']
    spatial_z = config['particle_spatial_range_z']
    z_coupled_ratio = config.get('z_coupled_ratio', 0.0)
    z_coupled_spacing = config.get('z_coupled_spacing_range', None)

    # 1. Determine N particles
    num_particles = np.random.randint(num_particles_range[0], num_particles_range[1])
    
    # 2. Logic for Coupled vs Random
    n_z_coupled = int(num_particles * z_coupled_ratio // 2) * 2
    n_random = num_particles - n_z_coupled
    
    beads = []
    between_beads = [] # Optional extra logic you had

    # A. Z-Coupled Pairs
    for _ in range(n_z_coupled // 2):
        x = random.choice(spatial_xy)
        y = random.choice(spatial_xy)
        z1 = random.choice(spatial_z)
        
        # Handle spacing logic
        z2 = random.choice(spatial_z) # Default fallback
        if z_coupled_spacing is not None:
            min_s, max_s = z_coupled_spacing
            valid_s = [s for s in range(min_s, max_s+1) if (z1 + s) in spatial_z]
            if valid_s:
                spacing = random.choice(valid_s)
                z2 = z1 + spacing
                # Your specific logic for "between beads"
                for u in range(z1+1, z2):
                    between_beads.append([x, y, u])

        beads.append([x, y, z1])
        beads.append([x, y, z2])

    # B. Remaining Random Beads
    for _ in range(n_random):
        x = random.choice(spatial_xy)
        y = random.choice(spatial_xy)
        z = random.choice(spatial_z)
        beads.append([x, y, z])

    random.shuffle(beads)
    
    # Return as numpy array (N, 3)
    return np.array(beads), np.array(between_beads) if between_beads else None