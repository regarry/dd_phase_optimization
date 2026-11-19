from physics.phase_io import load_phase_config
import numpy as np
import torch
import os

# Import specific physics math
# Assuming you moved bessel.py to physics/bessel.py in previous steps
from .bessel import generate_axicon_phase_mask
from data.io import load_tiff # Reuse your robust IO


def focused_airy_phase_parameters(size,pitch,k,f,alpha,angle,renderer_pixel):
    """Return unwrapped radians and maximum analytic phase step in rad/pixel.

    Coordinates are metres: rows y (scan), columns z (sheet thickness).
    Rotation is applied to cubic-phase coordinates, not to an exported bitmap.
    The carrier aligns the ideal Airy main lobe to the renderer's N//2-1 origin.
    """
    values=np.asarray([pitch,k,f,alpha,angle,renderer_pixel],dtype=float)
    if not np.isfinite(values).all() or min(pitch,k,f,alpha,renderer_pixel)<=0:
        raise ValueError('Focused Airy requires finite positive scales/alpha and a finite angle')
    if isinstance(size,bool) or not isinstance(size,(int,np.integer)) or size<=0:
        raise ValueError('Focused Airy mask size must be a positive integer')
    coordinate=(np.arange(size)-size//2)*pitch
    y,z=np.meshgrid(coordinate,coordinate,indexing='ij')
    theta=np.deg2rad(angle);c=np.cos(theta);s=np.sin(theta)
    a=c*y+s*z;b=-s*y+c*z
    scale=f*(3*alpha)**(1/3)/k
    dy=1.018792971647471*scale*(c-s)-renderer_pixel
    dz=1.018792971647471*scale*(s+c)-renderer_pixel
    phase=-k*(y*y+z*z)/(2*f)-alpha*(a**3+b**3)+k*(dy*y+dz*z)/f
    gy=-k*y/f-3*alpha*(c*a*a-s*b*b)+k*dy/f
    gz=-k*z/f-3*alpha*(s*a*a+c*b*b)+k*dz/f
    step=float(max(np.max(abs(gy)),np.max(abs(gz)))*pitch)
    return phase.astype(np.float32),step


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
        airy_mode=config.get('airy_phase_mode','cubic_only')
        if airy_mode=='focused':
            if config.get('lens_approach')!='sample_4f' or config.get('phase_mask_upsample_factor',1)!=1:
                raise ValueError('Focused Airy supports the reviewed sample_4f route with upsample factor1')
            if 'airy_alpha' not in config:
                raise ValueError('Focused Airy requires explicit airy_alpha in rad/m^3')
            focal=float(config['fresnel_lens_focal_length'])
            propagation=float(config['lensless_prop_distance'])
            if not np.isfinite(focal) or focal<=0 or not np.isclose(focal,propagation,rtol=1e-7,atol=0):
                raise ValueError('Focused Airy focal length must match the positive sample_4f propagation distance')
            k=2*np.pi*float(config['refractive_index'])/float(wavelength)
            phase,step=focused_airy_phase_parameters(size,slm_px,k,focal,float(config['airy_alpha']),
                        float(config.get('airy_rotation_degrees',0)),float(config.get('px',slm_px)))
            if step>=np.pi:
                raise ValueError('Focused Airy phase exceeds the conservative pi-radian-per-pixel sampling limit')
            print('Initializing focused Airy; alpha=%g rad/m^3, rotation=%g degrees, max step=%g rad/pixel'%
                  (float(config['airy_alpha']),float(config.get('airy_rotation_degrees',0)),step))
            return phase
        if airy_mode!='cubic_only':
            raise ValueError('airy_phase_mode must be cubic_only or focused')
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

    # Gaussian illumination: the simulator supplies the incident Gaussian
    # amplitude; zero phase leaves that beam unmodulated.
    elif mode == "gaussian":
        phase_mode = config.get('gaussian_phase_mode', 'flat')
        if phase_mode == 'focused':
            focal = float(config['fresnel_lens_focal_length'])
            if not np.isfinite(focal) or focal <= 0:
                raise ValueError('Focused Gaussian requires a positive configured focal length')
            # Explicit quadratic lens applied to the existing Gaussian amplitude.
            # The historical flat option remains unchanged for old checkpoints.
            return get_initial_phase_mask(dict(config, initial_phase_mask='lens'))
        if phase_mode != 'flat':
            raise ValueError('gaussian_phase_mode must be flat or focused')
        print("Initializing Gaussian illumination (zero phase modulation)...")
        return np.zeros((size, size), dtype=np.float32)

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
        mask = load_phase_config(path, config) # Ensure load_tiff is defined in your broader scope
        
        # Safety check for dimensions
        if mask.shape != (size, size):
            print(f"⚠️ Warning: Loaded mask shape {mask.shape} != config size ({size}, {size}). Resizing may occur in model.")
            
        return mask.astype(np.float32)

    else:
        raise ValueError(f"Unknown initial_phase_mask type: {mode}")
