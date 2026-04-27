# -*- coding: utf-8 -*-
import os
import numpy as np
import imageio.v2 as imageio
import skimage.filters
import scipy.special
import scipy.io as sio
from scipy.signal import fftconvolve

def get_airy_psf_slice(r_grid, r_vec, z, wavelength, numerical_aperture, refractive_index):
    """Generates a single PSF slice using the non-paraxial scalar Debye integral."""
    k = 2 * np.pi / wavelength
    NA = numerical_aperture
    n = refractive_index
    
    rho = np.linspace(0, 1, 500) 
    d_rho = rho[1] - rho[0]
    
    arg_bessel = k * NA * r_vec[:, None] * rho[None, :]
    
    sin_theta = (rho * NA) / n
    phase_term = np.exp(1j * k * z * n * np.sqrt(1 - sin_theta**2))
    
    integrand = scipy.special.j0(arg_bessel) * phase_term * rho[None, :]
    integral_result = np.sum(integrand, axis=1) * d_rho
    
    intensity_1d = np.abs(integral_result)**2
    psf_slice = np.interp(r_grid.ravel(), r_vec, intensity_1d).reshape(r_grid.shape)
    
    return psf_slice

def generate_psf(config, results_dir):
    """Generates a 3D PSF stack with anti-aliasing and spectrum averaging."""
    # 1. Setup Base Parameters
    psf_width_pixels = int(config.get('psf_width_pixels', 31))
    pixel_size = config.get('px', config.get('pixel_size_meters'))
    NA = config['numerical_aperture']
    n = config['refractive_index']
    
    if NA >= n:
        print(f"Warning: NA ({NA}) >= Refractive Index ({n}). This breaks physical assumptions.")

    # --- UPGRADE 1: Polychromatic Emission (Alexa Fluor 555) ---
    # AF555 has an excitation peak at 555nm, but its emission peak is ~565nm, 
    # trailing off towards the red spectrum. We discretize this curve.
    af555_spectrum = {
        565e-9: 1.00,  # Peak emission
        580e-9: 0.60,  # ~60% intensity
        595e-9: 0.25,  # ~25% intensity
        610e-9: 0.05   # ~5% intensity
    }
    total_weight = sum(af555_spectrum.values())

    # --- UPGRADE 2: Supersampling (Anti-aliasing) ---
    # Generate the grid at 3x resolution, then pool it down later.
    supersample_factor = config.get('supersample_factor', 1) # 1 means no supersampling
    super_width = psf_width_pixels * supersample_factor
    super_pixel_size = pixel_size / supersample_factor

    # 2. Pre-calculate Super-Resolved Grid
    center = (super_width - 1.0) / 2.0
    y, x = np.ogrid[:super_width, :super_width]
    
    r_grid_pixels = np.hypot(x - center, y - center)
    r_grid_meters = r_grid_pixels * super_pixel_size
    
    max_radius = np.max(r_grid_meters)
    r_vec = np.linspace(0, max_radius, int(super_width * 1.5))
    
    # Helper function to simulate camera pixel area integration
    def downsample_to_camera(img, factor):
        new_shape = (img.shape[0] // factor, factor, img.shape[1] // factor, factor)
        return img.reshape(new_shape).mean(axis=(1, 3))

    # 3. Define Z-depth
    if 'bead_volume' in config:
        z_stop = config['bead_volume'][2] * pixel_size / 2
    else:
        z_stop = 2.0e-6 
    z_depth = np.arange(0, z_stop + pixel_size/1000.0, pixel_size) 
    
    print(f"Generating polychromatic, anti-aliased PSF stack: {len(z_depth)} slices...")
    
    # Get Z=0 baseline for normalization
    z0_super_slice = np.zeros_like(r_grid_meters)
    for wl, weight in af555_spectrum.items():
        z0_super_slice += weight * get_airy_psf_slice(r_grid_meters, r_vec, 0.0, wl, NA, n)
    
    z0_camera_pixels = downsample_to_camera(z0_super_slice, supersample_factor)
    normalization_factor = np.sum(z0_camera_pixels)

    psf_stack = []
    psf_images_path = os.path.join(results_dir, 'psf_imgs/')
    os.makedirs(psf_images_path, exist_ok=True)

    # 4. Generate Stack
    for i, z in enumerate(z_depth):
        super_slice = np.zeros_like(r_grid_meters)
        
        # Sum the weighted wavelengths
        for wl, weight in af555_spectrum.items():
            super_slice += weight * get_airy_psf_slice(r_grid_meters, r_vec, z, wl, NA, n)
            
        # Integrate light over the pixel areas
        camera_slice = downsample_to_camera(super_slice, supersample_factor)
        
        # Normalize
        camera_slice /= normalization_factor
        psf_stack.append(camera_slice)

    psf_stack = np.array(psf_stack)

    # [Saving logic remains exactly the same as your original code below...]
    for i in range(len(psf_stack)):
        slice_min = psf_stack[i].min()
        slice_max = psf_stack[i].max()

        if slice_max > slice_min:
            visual_slice = (psf_stack[i] - slice_min) / (slice_max - slice_min)
            psf_8bit = (visual_slice * 255).astype(np.uint8)
        else:
            psf_8bit = np.zeros_like(psf_stack[i], dtype=np.uint8)
        imageio.imwrite(os.path.join(psf_images_path, f"psf_z_{i:03d}.png"), psf_8bit)

    psf_stack_path = os.path.join(results_dir, 'psf_stack.mat')
    sio.savemat(psf_stack_path, {'psf': psf_stack})
    print(f"PSF stack saved to {psf_stack_path}")
    
    return psf_stack

def generate_bead_defocus_stack(config, results_dir, setup_defocus_psf=None):
    """Convolves a perfect bead image with the 3D PSF stack."""
    psf_width_pixels = int(config['psf_width_pixels'])
    bead_radius_px = config['bead_radius'] 
    bead_intensity = config['bead_intensity']
    
    if setup_defocus_psf is None:
        mat_path = os.path.join(results_dir, 'psf_stack.mat') # Fixed the file name to match save
        if not os.path.exists(mat_path):
            print("Error: psf_stack.mat not found. Run generate_psf first.")
            return
        setup_defocus_psf = sio.loadmat(mat_path)['psf']
    else:
        print("Using provided PSF for bead defocus stack generation.")

    y, x = np.ogrid[:psf_width_pixels, :psf_width_pixels]
    center = (psf_width_pixels - 1) / 2
    
    dist_sq_pixels = (x - center)**2 + (y - center)**2
    bead_ori_img = np.zeros((psf_width_pixels, psf_width_pixels))
    
    bead_ori_img[dist_sq_pixels <= bead_radius_px**2] = bead_intensity
    
    if bead_radius_px > 0:
        bead_ori_img = skimage.filters.gaussian(bead_ori_img, sigma=0.5)

    print(f"Convolving {len(setup_defocus_psf)} slices...")
    defocused_beads = []
    for i in range(len(setup_defocus_psf)):
        blurred_img = fftconvolve(bead_ori_img, setup_defocus_psf[i], mode='same')
        defocused_beads.append(blurred_img)
        
    defocused_beads_png_path = os.path.join(results_dir, 'bead_defocus_pngs/')
    os.makedirs(defocused_beads_png_path, exist_ok=True)
    
    for i in range(len(defocused_beads)):
        slice_min = defocused_beads[i].min()
        slice_max = defocused_beads[i].max()

        if slice_max > slice_min:
            visual_slice = (defocused_beads[i] - slice_min) / (slice_max - slice_min)
            defocused_beads_8bit = (visual_slice * 255).astype(np.uint8)
        else:
            defocused_beads_8bit = np.zeros_like(defocused_beads[i], dtype=np.uint8)
        imageio.imwrite(os.path.join(defocused_beads_png_path, f"defocused_bead_z_{i:03d}.png"), defocused_beads_8bit)    
    
    defocused_beads_np = np.array(defocused_beads)    
    print("Bead stack generation complete.")
    
    defocused_bead_stack_path = os.path.join(results_dir, config.get('defocused_beads_filename', 'defocused_beads.mat'))
    sio.savemat(defocused_bead_stack_path, {'defocus_beads': defocused_beads_np})
    print(f"Defocused bead stack saved to {defocused_bead_stack_path}")
    
    return defocused_beads_np