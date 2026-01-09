import os
import numpy as np
import scipy.io as io
from scipy.special import j1
import matplotlib.pyplot as plt

def generate_bead_templates(config):
    """
    Master function to ensure PSF templates exist.
    Refactored from the old 'if not os.path.exists...' blocks.
    """
    # 1. Define paths
    psf_path = 'psf_z.mat'
    data_path = config.get('data_path', 'beads_img_defocus/')
    
    # 2. Check if PSFs need generation
    if not os.path.isfile(psf_path):
        print("⚠️  PSF templates not found. Generating 'psf_z.mat'...")
        generate_psf(config, save_path=psf_path)
    else:
        print(f"✅ Found existing PSF templates: {psf_path}")

    # 3. Check if Bead Images need generation (Optional, if you use them)
    # If your dataloader generates beads on the fly (which SyntheticMicroscopeData does),
    # you might not actually need this 'beads_img' folder anymore.
    # But if you rely on saved disk assets:
    if not os.path.isdir(data_path):
        print(f"⚠️  Data folder {data_path} not found. (Skipping generation as we use on-the-fly synthesis)")
        # os.makedirs(data_path)
        # beads_img(config)


def generate_psf(config, save_path='psf_z.mat'):
    """
    Simulates the Point Spread Function (PSF) at various defocus levels (Z).
    Calculates the 3D optical response of the system.
    """
    # Extract Physics Parameters
    N = config['N']               # Grid size (e.g., 500)
    px = config['px']             # Pixel size (m)
    wav = config['wavelength']    # Wavelength (m)
    NA = config['numerical_aperture']
    f = config['focal_length']    # Focal length (m)
    n = config['refractive_index']
    
    # Spatial Range
    L = N * px
    
    # 1. Coordinate System (Frequency Domain)
    # Create grid in frequency space (fx, fy)
    df = 1.0 / L
    fx = np.arange(-N/2, N/2) * df
    fy = np.arange(-N/2, N/2) * df
    FX, FY = np.meshgrid(fx, fy)
    
    # Rho is radial frequency coordinate
    RHO = np.sqrt(FX**2 + FY**2)
    
    # Aperture Limit (Cutoff Frequency)
    # Abbe diffraction limit: rho_max = NA / lambda
    rho_max = NA / wav
    
    # Pupil Function (Circular Aperture)
    pupil = RHO <= rho_max
    
    # 2. Defocus Phase Factor
    # We generate PSFs for a specific Z range (e.g., -41um to +41um)
    # Adjust this range based on your needs or config
    max_defocus_um = config.get('max_defocus', 40)
    z_step_um = 1 # 1 micron steps
    z_range = np.arange(-max_defocus_um, max_defocus_um + 1, z_step_um)
    
    # Storage for the stack
    psf_stack = []
    
    print(f"   Computing PSFs for Z range: {z_range[0]} to {z_range[-1]} um...")

    for z_um in z_range:
        z = z_um * 1e-6 # convert to meters
        
        # Defocus Phase Term: exp(i * k * z * sqrt(1 - (lambda*rho)^2))
        # Paraxial approximation often used: exp(-i * pi * lambda * z * rho^2)
        # Non-paraxial (Rigorous):
        # term = sqrt(1 - (wav * RHO)**2)
        # phase = 2 * np.pi * z * term / wav
        
        # Standard Paraxial Defocus for Microscopy:
        # W20 * r^2 ... here implemented directly in Fourier space
        k = 2 * np.pi / wav
        # Phase shift due to propagation by distance z
        # H(fx, fy) = exp(j * k * z * sqrt(1 - (lambda * fx)^2 - (lambda * fy)^2))
        
        # Angle term (using Fresnel approximation for simplicity or exact)
        # Fresnel: H = exp(-1j * pi * lambda * z * (fx^2 + fy^2))
        phase_defocus = -1j * np.pi * wav * z * (RHO**2)
        
        # Transfer Function
        H = pupil * np.exp(phase_defocus)
        
        # PSF is Inverse Fourier Transform of Transfer Function
        # We use iftshift to handle the center-zero correctly
        psf_complex = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(H)))
        psf_intensity = np.abs(psf_complex)**2
        
        # Normalize
        psf_intensity = psf_intensity / np.max(psf_intensity)
        
        # Crop to desired size (e.g., small window around center)
        # config['psf_width_pixels'] determines how much we keep
        width = config['psf_width_pixels']
        center = N // 2
        start = center - width // 2
        end = start + width
        
        psf_crop = psf_intensity[start:end, start:end]
        psf_stack.append(psf_crop)

    # 3. Save to .mat
    psf_stack = np.array(psf_stack) # Shape: (Depth, H, W)
    
    # Transpose to match typical MATLAB format if needed: (H, W, Depth)
    # The Physics engine usually expects (Depth, H, W) or (H, W, Depth). 
    # Let's stick to (H, W, Depth) as that is common in .mat files.
    psf_stack = np.transpose(psf_stack, (1, 2, 0))
    
    data = {'psf_z': psf_stack, 'z_values': z_range}
    io.savemat(save_path, data)
    print(f"   Saved PSFs to {save_path}. Shape: {psf_stack.shape}")