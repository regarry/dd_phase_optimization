# -*- coding: utf-8 -*-

import os
import numpy as np
import imageio.v2 as imageio
import skimage.filters
#import skimage.io
import scipy.special
import scipy.io as sio
from scipy.signal import fftconvolve
#import tifffile

def get_airy_psf_slice(r_grid, r_vec, z, wavelength, numerical_aperture, refractive_index):
    """
    Generates a single PSF slice using the non-paraxial scalar Debye integral.
    """
    k = 2 * np.pi / wavelength
    NA = numerical_aperture
    n = refractive_index
    
    # Integration variable rho (normalized pupil radius 0 to 1)
    # Using more points (500) for smoother high-frequency phase oscillations
    rho = np.linspace(0, 1, 500) 
    d_rho = rho[1] - rho[0]
    
    # 1. Transverse component: k * NA * r * rho
    arg_bessel = k * NA * r_vec[:, None] * rho[None, :]
    
    # 2. IMPROVED Phase term: Exact axial phase 
    # Instead of the 0.5 * (NA/n)**2 * rho**2 approximation, use the exact term:
    # This accounts for the actual spherical shape of the converging wavefront.
    sin_theta = (rho * NA) / n
    phase_term = np.exp(1j * k * z * n * np.sqrt(1 - sin_theta**2))
    
    # 3. Integrand
    # We include 'rho' as the area element for the circular aperture
    integrand = scipy.special.j0(arg_bessel) * phase_term * rho[None, :]
    
    # Integral over rho (Simpsons rule or simple sum)
    integral_result = np.sum(integrand, axis=1) * d_rho
    
    # Intensity (modulus squared)
    intensity_1d = np.abs(integral_result)**2
    
    # Interpolate 1D profile to 2D Grid
    psf_slice = np.interp(r_grid.ravel(), r_vec, intensity_1d).reshape(r_grid.shape)
    
    return psf_slice

def generate_psf(config, results_dir):
    """
    Generates a 3D PSF stack.
    """
    # 1. Setup Parameters
    if 'psf_width_pixels' in config:
        psf_width_pixels = int(config['psf_width_pixels'])
    else:
        psf_width_pixels = 2 * (config.get('psf_keep_radius', 50) * 2 + 1)
        
    pixel_size = config.get('px', config.get('pixel_size_meters'))
    ############### added this temporarily to replicate psfs that got the lenslet beam #####
    #pixel_size = 1.0e-6
    ######################################################################################
    wavelength = config['wavelength']
    NA = config['numerical_aperture']
    n = config['refractive_index']
    
    if NA >= n:
        print(f"Warning: NA ({NA}) >= Refractive Index ({n}). This breaks physical assumptions.")

    # 2. Pre-calculate Grid (Optimization)
    center = (psf_width_pixels - 1.0) / 2.0
    y, x = np.ogrid[:psf_width_pixels, :psf_width_pixels]
    
    # r_grid in meters for physics calculation
    r_grid_pixels = np.hypot(x - center, y - center)
    r_grid_meters = r_grid_pixels * pixel_size
    
    # 1D Vector for integration
    max_radius = np.max(r_grid_meters)
    r_vec = np.linspace(0, max_radius, int(psf_width_pixels * 1.5))

    # Mask for circular aperture
    #mask = r_grid_pixels <= center
    
    # 3. Define Z-depth
    # Ensure we cover the requested volume
    if 'bead_volume' in config:
        z_stop = config['bead_volume'][2] * pixel_size / 2
    else:
        z_stop = 2.0e-6 

    z_depth = np.arange(0, z_stop + pixel_size/1000.0, pixel_size) 
    
    psf_stack = []
    psf_images_path = os.path.join(results_dir, 'psf_imgs/')
    os.makedirs(psf_images_path, exist_ok=True)
    
    print(f"Generating PSF stack: {len(z_depth)} slices, width {psf_width_pixels}px...")
    # 1. First, generate the in-focus slice to get the baseline energy
    z0_slice = get_airy_psf_slice(r_grid_meters, r_vec, 0.0, wavelength, NA, n)
    normalization_factor = np.sum(z0_slice)
    
    # 4. Generate Slices
    for i, z in enumerate(z_depth):
        raw_slice = get_airy_psf_slice(r_grid_meters, r_vec, z, wavelength, NA, n)
        # normalize so it sums to 1
        raw_slice /= normalization_factor
        #raw_slice *= mask 
        psf_stack.append(raw_slice)

    # Convert to numpy array for 3D operations
    psf_stack = np.array(psf_stack)


    # global_max = np.max(psf_stack)
    # if global_max > 0:
    #     psf_stack /= global_max
    
    # Save slices
    for i in range(len(psf_stack)):
        slice_min = psf_stack[i].min()
        slice_max = psf_stack[i].max()

        if slice_max > slice_min:
            # Scale to 0-255 based on the slice's own range
            visual_slice = (psf_stack[i] - slice_min) / (slice_max - slice_min)
            psf_8bit = (visual_slice * 255).astype(np.uint8)
        else:
            psf_8bit = np.zeros_like(psf_stack[i], dtype=np.uint8)
        imageio.imwrite(os.path.join(psf_images_path, f"psf_z_{i:03d}.png"), psf_8bit)

    # Save 3D stack
    psf_stack_path = os.path.join(results_dir, 'psf_stack.mat')
    sio.savemat(psf_stack_path, {'psf': psf_stack})
    print(f"PSF stack saved to {psf_stack_path}")
    print("psf stack type", psf_stack.dtype )
    print("psf stack max", np.max(psf_stack))

    return psf_stack


def generate_bead_defocus_stack(config, results_dir, setup_defocus_psf=None):
    psf_width_pixels = int(config['psf_width_pixels'])
    
    # Updated: bead_radius is treated directly as pixels
    bead_radius_px = config['bead_radius'] 
    bead_intensity = config['bead_intensity']
    
    # Output path
    #bead_defocus_img_path = os.path.join(results_dir, 'bead_defocus_imgs/')
    #os.makedirs(bead_defocus_img_path, exist_ok=True)

    # Load PSF if not provided
    if setup_defocus_psf is None:
        mat_path = os.path.join(results_dir, 'psf_z.mat')
        if not os.path.exists(mat_path):
            print("Error: psf_z.mat not found. Run generate_psf first.")
            return
        setup_defocus_psf = sio.loadmat(mat_path)['psf']
    else:
        print("Using provided PSF for bead defocus stack generation.")

    # --- BEAD GENERATION ---
    y, x = np.ogrid[:psf_width_pixels, :psf_width_pixels]
    center = (psf_width_pixels - 1) / 2
    
    dist_sq_pixels = (x - center)**2 + (y - center)**2
    bead_ori_img = np.zeros((psf_width_pixels, psf_width_pixels))
    
    # Draw bead using pixel radius directly
    bead_ori_img[dist_sq_pixels <= bead_radius_px**2] = bead_intensity
    
    # Anti-aliasing (Gaussian smooth the edges of the perfect circle)
    if bead_radius_px > 0:
        bead_ori_img = skimage.filters.gaussian(bead_ori_img, sigma=0.5)

    print(f"Convolving {len(setup_defocus_psf)} slices...")
    defocused_beads = []
    for i in range(len(setup_defocus_psf)):
        blurred_img = fftconvolve(bead_ori_img, setup_defocus_psf[i], mode='same')
        defocused_beads.append(blurred_img)
        # Save as 16-bit TIFF
        #fname = os.path.join(bead_defocus_img_path, f'z{i:02d}.tiff')
        #tifffile.imwrite(fname, blurred_img)
        
    # Save slices
    defocused_beads_png_path = os.path.join(results_dir, 'bead_defocus_pngs/')
    os.makedirs(defocused_beads_png_path, exist_ok=True)
    for i in range(len(defocused_beads)):
        slice_min = defocused_beads[i].min()
        slice_max = defocused_beads[i].max()

        if slice_max > slice_min:
            # Scale to 0-255 based on the slice's own range
            visual_slice = (defocused_beads[i] - slice_min) / (slice_max - slice_min)
            defocused_beads_8bit = (visual_slice * 255).astype(np.uint8)
        else:
            defocused_beads_8bit = np.zeros_like(defocused_beads[i], dtype=np.uint8)
        imageio.imwrite(os.path.join(defocused_beads_png_path, f"defocused_bead_z_{i:03d}.png"), defocused_beads_8bit)    
    
    
    
    defocused_beads_np = np.array(defocused_beads)    
    print("Bead stack generation complete.")
    # Save 3D stack
    defocused_bead_stack_path = os.path.join(results_dir, 'defocused_beads.mat')
    sio.savemat(defocused_bead_stack_path, {'defocus_beads': defocused_beads_np})
    print(f"Defocused bead stack saved to {defocused_bead_stack_path}")
    return defocused_beads_np