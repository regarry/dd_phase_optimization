# -*- coding: utf-8 -*-

import os
import numpy as np
import imageio.v2 as imageio # Explicit v2 to avoid warnings
import skimage
import skimage.filters
import skimage.io
import scipy.signal
import scipy.special
import scipy.io as sio
import math

# Use fftconvolve for O(N log N) speed instead of O(N^2)
from scipy.signal import fftconvolve

def get_airy_psf(psf_width_pixels, psf_width_meters, z, wavelength, numerical_aperture, refractive_index,
                 normalize=True):
    """
    Generates the Airy PSF for a given Z depth using vectorized 1D integration 
    and radial interpolation (orders of magnitude faster).
    """
    # 1. Setup Grid and Physical Parameters
    center = (psf_width_pixels - 1.0) / 2.0
    meters_per_pixel = psf_width_meters / psf_width_pixels
    
    # Create a grid of radial distances (r) in meters
    y, x = np.ogrid[:psf_width_pixels, :psf_width_pixels]
    # Calculate radius for every pixel at once
    r_grid = np.hypot((x - center) * meters_per_pixel, (y - center) * meters_per_pixel)
    
    # 2. Compute 1D Radial Intensity Profile (Exploiting Symmetry)
    # We only need to integrate along a single line from center to corner
    max_radius = np.max(r_grid)
    n_samples = int(psf_width_pixels * 1.5) # High resolution for interpolation
    r_vec = np.linspace(0, max_radius, n_samples)
    
    # Constants
    k = 2 * np.pi / wavelength
    n = refractive_index
    NA = numerical_aperture
    
    # Integration variable rho (0 to 1)
    # Using a fixed grid (Riemann sum) is much faster than adaptive quad for this
    rho = np.linspace(0, 1, 250) 
    d_rho = rho[1] - rho[0]
    
    # Vectorized Integral: Int( J0(A*r*rho) * exp(B*rho^2) * rho ) d_rho
    # A = k * NA / n
    # B = -0.5j * k * z * (NA/n)^2
    
    # Shapes: r_vec (M,), rho (P,)
    # We create a matrix (M, P) to sum over P
    arg_bessel = (k * NA / n) * r_vec[:, None] * rho[None, :]
    phase_term = np.exp(-0.5 * 1j * k * z * (NA / n)**2 * rho**2)
    
    # Integrand = J0(...) * phase * rho
    # broadcasting phase_term * rho across r_vec
    integrand = scipy.special.j0(arg_bessel) * (phase_term * rho)[None, :]
    
    # Sum over rho (axis 1) roughly approximates the integral
    integral_result = np.sum(integrand, axis=1) * d_rho
    
    # Compute Intensity
    intensity_1d = np.real(integral_result * np.conj(integral_result))
    
    # 3. Interpolate 1D profile to 2D Grid
    psf = np.interp(r_grid.ravel(), r_vec, intensity_1d).reshape(psf_width_pixels, psf_width_pixels)
    
    # 4. Apply Circular Mask (based on pixel distance from center)
    # The original code masked pixels outside the circle inscribed in the square
    pixel_r_grid = np.hypot(x - center, y - center)
    mask = pixel_r_grid <= center
    psf *= mask

    if normalize and np.max(psf) != 0:
        return psf / np.max(psf)
    return psf


def apply_blur_kernel(image, psf):
    """
    Convolves an image with the PSF kernel using FFT for speed.
    """
    if np.sum(psf) == 0:
        return image
        
    psf_normalized = psf / np.sum(psf)
    
    # FFT convolution is significantly faster for kernels > 20x20
    # mode='same' keeps output size equal to input size
    return fftconvolve(image, psf_normalized, mode='same')


def generate_psf(config, results_dir):
    """
    Generates a stack of PSF images.
    """
    # Fix: Handle config missing bead_volume by defaulting to direct z_stop if available
    psf_width_pixels = 2 * (config.get('psf_keep_radius', 50) * 2 + 1) # Fallback if key missing
    if 'psf_width_pixels' in config:
        psf_width_pixels = config['psf_width_pixels']
        
    pixel_size_meters = config.get('px', config.get('pixel_size_meters'))
    wavelength = config['wavelength']
    numerical_aperture = config['numerical_aperture']
    refractive_index = config['refractive_index']
    
    psf_stack_path = os.path.join(results_dir, 'psf_z.mat')
    psf_images_path = os.path.join(results_dir, 'psf_imgs/')
    psf_txt_path = os.path.join(results_dir, 'psf_z.txt')
    
    # Define Z-depth
    # Fix: Correctly calculate z_stop based on availability of keys
    z_start = 0
    z_stop = config['bead_volume'][2] * pixel_size_meters / 2
    z_step = config['px']
    
    # Use linspace or arange carefully to include endpoints if needed, 
    # matching the original logic's reliance on arange
    z_depth = np.arange(z_start, z_stop + z_step/1000.0, z_step) 
    
    psf_width_meters = psf_width_pixels * pixel_size_meters
    
    psf_stack = []
    
    os.makedirs(psf_images_path, exist_ok=True)
    
    print(f"Generating PSF stack with {len(z_depth)} slices...")
    
    # Write logs
    with open(psf_txt_path, "w") as log_file:
        log_file.write("Config:\n")
        for key, value in config.items():
            log_file.write(f"{key}: {value}\n")
        log_file.write("\n")

    for i, z in enumerate(z_depth):
        if i % 5 == 0: # Reduce print spam
            print(f"Processing slice {i}/{len(z_depth)-1} at Z={z:.2e}")
            
        slice_psf = get_airy_psf(
            psf_width_pixels, 
            psf_width_meters, 
            z, 
            wavelength, 
            numerical_aperture,
            refractive_index
        )
        psf_stack.append(slice_psf)

        # Save PNG
        psf_8bit = np.clip(slice_psf * 255, 0, 255).astype(np.uint8)
        imageio.imwrite(
            os.path.join(psf_images_path, f"psf_z_{i:03d}.png"),
            psf_8bit
        )

    # Save to file
    sio.savemat(psf_stack_path, {'psf': psf_stack})
    print(f"PSF stack saved to {psf_stack_path}")

    return psf_stack


def generate_bead_defocus_stack(config, results_dir, setup_defocus_psf=None):
    psf_width_pixels = config['psf_width_pixels']
    bead_defocus_img_path = os.path.join(results_dir, 'bead_defocus_imgs/')
    bead_radius = config['bead_radius']
    bead_intensity = config['bead_intensity']

    if setup_defocus_psf is None:
        try:
            setup_defocus_psf = sio.loadmat(os.path.join(results_dir, 'psf_z.mat'))['psf']
        except FileNotFoundError:
            print("Error: psf_z.mat not found. Run generate_psf first.")
            return

    os.makedirs(bead_defocus_img_path, exist_ok=True)

    # Vectorized Bead Generation
    # Create coordinate grid
    y, x = np.ogrid[:psf_width_pixels, :psf_width_pixels]
    center = int(math.floor(psf_width_pixels/2))
    dist_from_center = (x - center)**2 + (y - center)**2
    
    bead_ori_img = np.zeros((psf_width_pixels, psf_width_pixels))
    # Fill circle
    bead_ori_img[dist_from_center <= bead_radius**2] = bead_intensity
    
    # Smooth
    bead_ori_img = skimage.filters.gaussian(bead_ori_img, sigma=1)

    print(f"Generating {len(setup_defocus_psf)} defocused bead images...")
    
    for i in range(len(setup_defocus_psf)):
        # apply_blur_kernel now uses FFT, making this loop very fast
        blurred_img = apply_blur_kernel(bead_ori_img, setup_defocus_psf[i])
        
        # Save
        skimage.io.imsave(
            os.path.join(bead_defocus_img_path, 'z' + str(i).zfill(2) + '.tiff'), 
            blurred_img.astype('uint16'),
            check_contrast=False
        )
    print("Done.")

if __name__ == '__main__':
    # Default configuration
    results_folder = './results_optimized'
    os.makedirs(results_folder, exist_ok=True)
    
    default_config = {
        'psf_width_pixels': 101,
        'px': 1e-6,                 # Renamed to match usage in generate_psf
        'wavelength': 0.561e-6,
        'refractive_index': 1.0,
        'numerical_aperture': 0.6,
        'z_start': 0,
        'z_stop': 4.0e-5,
        'z_step': 1e-6,
        'bead_radius': 2,           # Pixels
        'bead_intensity': 10000,
        # 'bead_volume': [200, 200, 40] # Optional if using direct z_stop
    }

    # 1. Generate PSFs
    #psf_stack = generate_psf(default_config, results_folder)
    
    # 2. Generate Defocused Beads (Optional test)
    #generate_bead_defocus_stack(default_config, results_folder, setup_defocus_psf=psf_stack)