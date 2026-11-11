import random
import numpy as np
from scipy.stats import truncnorm


def _jitter_coordinate(anchor, allowed, radius):
    """Sample an allowed integer position within inclusive +/- radius."""
    if isinstance(allowed, range) and allowed.step == 1:
        return random.randint(max(allowed.start, anchor - radius),
                              min(allowed.stop - 1, anchor + radius))
    return random.choice([value for value in allowed if abs(value - anchor) <= radius])


def create_random_emitters(config):
    """
    Generates positions for ONE single image volume.
    Used by the Dataset class in __getitem__.
    Supports toggling between Uniform and true Truncated Gaussian distributions along the Z axis.
    """
    # Extract config
    num_particles_range = config['num_particles_range']
    lateral_jitter = config.get('z_coupled_lateral_jitter_px', 0)
    if isinstance(lateral_jitter, (bool, np.bool_)) or not isinstance(lateral_jitter, (int, np.integer)) or lateral_jitter < 0:
        raise ValueError('z_coupled_lateral_jitter_px must be a nonnegative integer')
    
    # Handle spatial ranges (convert to list if they are [min, max] pairs)
    spatial_x = config['particle_spatial_range_x']
    spatial_y = config['particle_spatial_range_y']
    spatial_z = config['particle_spatial_range_z']

    # Helper to convert [min, max] to range object if needed
    if isinstance(spatial_x, (list, tuple)) and len(spatial_x) == 2:
        spatial_x_list = range(int(spatial_x[0]), int(spatial_x[1]))
    else:
        spatial_x_list = spatial_x
        
    if isinstance(spatial_y, (list, tuple)) and len(spatial_y) == 2:
        spatial_y_list = range(int(spatial_y[0]), int(spatial_y[1]))
    else:
        spatial_y_list = spatial_y

    if isinstance(spatial_z, (list, tuple)) and len(spatial_z) == 2:
        spatial_z_list = range(int(spatial_z[0]), int(spatial_z[1]))
    else:
        spatial_z_list = spatial_z

    # --- Setup Uniform vs. Truncated Gaussian Z Sampling ---
    min_z = min(spatial_z_list)
    max_z = max(spatial_z_list)
    
    use_gaussian_z = config.get('gaussian_z_bead_dist', False)

    if use_gaussian_z:
        mu = config.get('z_gaussian_mu', (min_z + max_z) / 2.0) 
        
        # Guard against division by zero
        divisor = max(0.1, config.get('z_gaussian_sigma_divisor', 6.0))
        sigma = (max_z - min_z) / divisor
        if sigma == 0:
            sigma = 1e-6

        # Convert physical boundaries to standardized Z-scores for SciPy
        a_scaled = (min_z - mu) / sigma
        b_scaled = (max_z - mu) / sigma

        def get_z_coordinate():
            """Samples directly from a truncated normal distribution in O(1) time."""
            return int(round(truncnorm.rvs(a_scaled, b_scaled, loc=mu, scale=sigma)))
    else:
        def get_z_coordinate():
            """Samples a Z coordinate uniformly from the available spatial list."""
            return random.choice(spatial_z_list)
    # -------------------------------------------------------

    z_coupled_ratio = config.get('z_coupled_ratio', 0.0)
    z_coupled_spacing = config.get('z_coupled_spacing_range', None)

    # 1. Determine N particles (Random for EVERY image)
    num_particles = np.random.randint(num_particles_range[0], num_particles_range[1])
    
    # 2. Logic for Coupled vs Random
    n_z_coupled = int(num_particles * z_coupled_ratio // 2) * 2
    n_random = num_particles - n_z_coupled
    
    beads = []
    between_beads = [] 

    # A. Z-Coupled Pairs. Jitter is the second bead's displacement relative
    # to the first, independently in camera-column x and camera-row y.
    for _ in range(n_z_coupled // 2):
        x = random.choice(spatial_x_list)
        y = random.choice(spatial_y_list)
        
        z1 = get_z_coordinate()
        z2 = get_z_coordinate() 
        connect_pair = False
        
        if z_coupled_spacing is not None:
            min_s, max_s = z_coupled_spacing
            valid_spacings = [s for s in range(min_s, max_s+1) if (z1 + s) in spatial_z_list]
            
            if valid_spacings:
                spacing = random.choice(valid_spacings)
                z2 = z1 + spacing
                connect_pair = True

        # Sampling directly from valid coordinates avoids clipped-edge piles
        # and keeps both beads inside the PSF-safe placement region. At zero,
        # draw no additional random numbers, preserving legacy seeded scenes.
        x2, y2 = x, y
        if lateral_jitter:
            x2 = _jitter_coordinate(x, spatial_x_list, lateral_jitter)
            y2 = _jitter_coordinate(y, spatial_y_list, lateral_jitter)

        if connect_pair:
            step = 1 if z2 > z1 else -1
            for u in range(z1 + step, z2, step):
                if lateral_jitter:
                    fraction = (u - z1) / (z2 - z1)
                    between_beads.append([round(x + fraction * (x2 - x)),
                                          round(y + fraction * (y2 - y)), u])
                else:
                    between_beads.append([x, y, u])

        beads.append([x, y, z1])
        beads.append([x2, y2, z2])

    # B. Remaining Random Beads
    for _ in range(n_random):
        x = random.choice(spatial_x_list)
        y = random.choice(spatial_y_list)
        z = get_z_coordinate()
        beads.append([x, y, z])

    random.shuffle(beads)
    
    # Return as numpy arrays
    beads_np = np.array(beads) if len(beads) > 0 else np.array([])
    between_np = np.array(between_beads) if len(between_beads) > 0 else np.array([])
        
    return beads_np, between_np
