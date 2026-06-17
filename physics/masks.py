import numpy as np
import torch
import os

# Import specific physics math
# Assuming you moved bessel.py to physics/bessel.py in previous steps
from .bessel import generate_axicon_phase_mask
from data.io import load_tiff # Reuse your robust IO


def get_initial_phase_mask(config):
    """
    Factory function to generate the initial phase mask based on config.
    Returns a float32 numpy array of shape (H, W).
    """
    mode = config.get('initial_phase_mask', 'empty').lower()
    size = config['phase_mask_pixel_size']
    slm_px = config['slm_px'] # in meters
    wavelength = config['wavelength'] # already in meters
    # sometimes slm_px is written to config.yaml as 1e-6 not 1.0e-6 so it loads as a string not float
    if isinstance(slm_px, str):
        slm_px = float(slm_px)
    
    # 1. Physics-based initialization (Axicon)
    if mode == "axicon":
        # Ensure parameters exist
        angle = config.get('bessel_half_cone_angle_degrees', 1.0)
        
        print(f"Initializing with Axicon (Angle: {angle}°)...")
        return generate_axicon_phase_mask(
            (size, size), 
            slm_px * 1e6,       # bessel function expects microns
            wavelength * 1e9, # bessel function expects nm
            angle
        )
        
    # 2. Spherical Lens
    elif mode == "lens":
        print("Initializing with Lens Phase Mask...")
        # Simple quadratic lens phase profile
        x = np.linspace(-size//2, size//2 - 1, size) * slm_px # in m
        y = np.linspace(-size//2, size//2 - 1, size) * slm_px # in m
        X, Y = np.meshgrid(x, y)
    
        focal_length = config['fresnel_lens_focal_length'] #  focal length in meters
        k = 2 * np.pi / wavelength # wavenumber in m^-1
        lens_phase = (k / (2 * focal_length)) * (X**2 + Y**2) # Quadratic phase
        # 3. WRAP THE PHASE to [0, 2*pi] to make it a Fresnel lens
        fresnel_phase = np.mod(lens_phase, 2 * np.pi)
        fresnel_phase = 2 * np.pi - fresnel_phase # Invert phase for SLM compatibility
        return fresnel_phase.astype(np.float32)

    # 3. Cylindrical Lens
    elif mode == "cylinder":
        print("Initializing with Cylindrical Lens Phase Mask...")
        # Focuses light in only one dimension (x or y)
        x = np.linspace(-size//2, size//2 - 1, size) * slm_px # in m
        y = np.linspace(-size//2, size//2 - 1, size) * slm_px # in m
        X, Y = np.meshgrid(x, y)
        
        # Use a specific cylinder focal length, or fall back to lenless_prop_distance
        focal_length = config.get('cylinder_focal_length', config.get('lenless_prop_distance', )) 
        k = 2 * np.pi / wavelength # wavenumber in nm^-1
        
        axis = config.get('cylinder_axis', 'x').lower()
        if axis == 'x':
            cylinder_phase = (k / (2 * focal_length)) * (X**2)
        elif axis == 'y':
            cylinder_phase = (k / (2 * focal_length)) * (Y**2)
        else:
            raise ValueError("cylinder_axis in config must be 'x' or 'y'")
        
        wrapped_cylinder_phase = np.mod(cylinder_phase, 2 * np.pi)
        wrapped_cylinder_phase = 2 * np.pi - wrapped_cylinder_phase # Invert phase for SLM compatibility
        return wrapped_cylinder_phase.astype(np.float32)

    # 4. Airy Beam (Cubic Phase Mask)
    elif mode in ["air", "airy"]:
        print("Initializing with Airy (Cubic) Phase Mask...")
        # Generates an Airy beam using a 2D cubic phase profile: alpha * (x^3 + y^3)
        x = np.linspace(-size//2, size//2 - 1, size) * slm_px  # in m
        y = np.linspace(-size//2, size//2 - 1, size) * slm_px  # in m
        X, Y = np.meshgrid(x, y)
        
        # alpha controls the trajectory/bending rate of the beam
        alpha = config.get('airy_alpha', 1e-4)
        airy_phase = alpha * (X**3 + Y**3)
        wrapped_airy_phase = np.mod(airy_phase, 2 * np.pi)
        wrapped_airy_phase = 2 * np.pi - wrapped_airy_phase # Invert phase for SLM compatibility
        return wrapped_airy_phase.astype(np.float32)

    # 5. Flat / Empty initialization
    elif mode == "empty":
        print("Initializing with Flat (Zero) mask...")
        return np.zeros((size, size), dtype=np.float32)

    # 6. Random Noise initialization
    elif mode == "random":
        print("Initializing with Random Noise...")
        # Random phase between 0 and 2pi
        return np.random.rand(size, size).astype(np.float32) * 2 * np.pi

    # 7. Load from File
    elif mode == "file":
        path = config.get('phase_mask_file')
        if not path:
            raise ValueError("Config set to 'file' but 'phase_mask_file' path is missing.")
        
        print(f"Initializing from file: {path}...")
        
        # Use our robust loader (handles checks and errors)
        mask = load_tiff(path) # Ensure load_tiff is defined in your broader scope
        
        # Safety check for dimensions
        if mask.shape != (size, size):
            print(f"⚠️ Warning: Loaded mask shape {mask.shape} != config size ({size}, {size}). Resizing may occur in model.")
            
        return mask.astype(np.float32)

    else:
        raise ValueError(f"Unknown initial_phase_mask type: {mode}")