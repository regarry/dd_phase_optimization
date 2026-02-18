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
    mode = config.get('initial_phase_mask', 'empty')
    size = config['phase_mask_pixel_size']
    slm_px = config['slm_px'] # in meters
    
    # 1. Physics-based initialization (Axicon)
    if mode == "axicon":
        # Ensure parameters exist
        angle = config.get('bessel_half_cone_angle_degrees', 1.0)
        #px_m = slm_px # config['slm_px'] should already be in meters from main() correction
        wavelength_m = config['wavelength'] # already in meters
        
        print(f"Initializing with Axicon (Angle: {angle}°)...")
        return generate_axicon_phase_mask(
            (size, size), 
            slm_px * 1e6,       # bessel function expects microns
            wavelength_m * 1e9, # bessel function expects nm
            angle
        )
    elif mode == "lens":
        print("Initializing with Lens Phase Mask...")
        # Simple quadratic lens phase profile
        x = np.linspace(-size//2, size//2 - 1, size) * slm_px * 1e6 # in microns
        y = np.linspace(-size//2, size//2 - 1, size) * slm_px * 1e6 # in microns
        X, Y = np.meshgrid(x, y)
        focal_length_mm = config['lenless_prop_distance'] * 1e3 # Example focal length in mm
        wavelength_nm = config['wavelength'] * 1e9 # in nm
        k = 2 * np.pi / wavelength_nm # wavenumber in nm^-1
        lens_phase = (k / (2 * focal_length_mm * 1e3)) * (X**2 + Y**2) # Quadratic phase
        return lens_phase.astype(np.float32)
    # 2. Flat / Empty initialization
    elif mode == "empty":
        print("Initializing with Flat (Zero) mask...")
        return np.zeros((size, size), dtype=np.float32)

    # 3. Random Noise initialization
    elif mode == "random":
        print("Initializing with Random Noise...")
        # Random phase between 0 and 2pi
        return np.random.rand(size, size).astype(np.float32) * 2 * np.pi

    # 4. Load from File
    elif mode == "file":
        path = config.get('phase_mask_file')
        if not path:
            raise ValueError("Config set to 'file' but 'phase_mask_file' path is missing.")
        
        print(f"Initializing from file: {path}...")
        
        # Use our robust loader (handles checks and errors)
        mask = load_tiff(path)
        
        # Safety check for dimensions
        if mask.shape != (size, size):
            print(f"⚠️ Warning: Loaded mask shape {mask.shape} != config size ({size}, {size}). Resizing may occur in model.")
            # Optional: Add resize logic here if strictness is required
            
        return mask.astype(np.float32)

    else:
        raise ValueError(f"Unknown initial_phase_mask type: {mode}")