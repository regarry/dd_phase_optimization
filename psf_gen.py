# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.integrate
import scipy.signal
import scipy.special
import scipy.io as sio

def get_airy_psf(psf_width_pixels, psf_width_meters, z, wavelength, numerical_aperture, refractive_index,
                 normalize=True):
    """
    Generates the Airy PSF for a given Z depth.
    """
    meters_per_pixel = psf_width_meters / psf_width_pixels
    psf = np.zeros((psf_width_pixels, psf_width_pixels), dtype=np.float64)
    
    # Center point calculation
    center = (psf_width_pixels - 1.0) / 2.0
    
    for i in range(psf_width_pixels):
        for j in range(psf_width_pixels):
            # Calculate physical coordinates relative to center
            x = (i - center) * meters_per_pixel
            y = (j - center) * meters_per_pixel
            
            # Circular mask check
            if (i - center) ** 2 + (j - center) ** 2 <= center ** 2:
                psf[i, j] = eval_airy_function(x, y, z, wavelength, numerical_aperture, refractive_index)

    # Normalize PSF to max value.
    if normalize and np.max(psf) != 0:
        return psf / np.max(psf)
    return psf


def eval_airy_function(x, y, z, wavelength, NA, refractive_index):
    """
    Evaluates the Airy function integral at specific coordinates.
    """
    k = 2 * np.pi / wavelength
    n = refractive_index
    r_coordinate = np.sqrt(x * x + y * y)

    def fun_integrate(rho):
        bessel_arg = k * NA / n * r_coordinate * rho
        # Using J0 bessel function
        return scipy.special.j0(bessel_arg) * np.exp(-0.5 * 1j * k * rho * rho * z * np.power(NA / n, 2)) * rho

    integral_result = integrate_numerical(fun_integrate, 0.0, 1.0)

    return float(np.real(integral_result * np.conj(integral_result)))


def integrate_numerical(function_to_integrate, start, end):
    """
    Helper to integrate complex functions by separating real and imaginary parts.
    """
    def real_function(x):
        return np.real(function_to_integrate(x))

    def imag_function(x):
        return np.imag(function_to_integrate(x))

    real_result = scipy.integrate.quad(real_function, start, end, limit=200)[0]
    imag_result = scipy.integrate.quad(imag_function, start, end, limit=200)[0]
    return real_result + 1j * imag_result


def apply_blur_kernel(image, psf):
    """
    Convolves an image with the PSF kernel.
    """
    psf_normalized = psf / np.sum(psf)
    return scipy.signal.convolve2d(image, psf_normalized, 'same', boundary='symm')


def generate_psf(config):
    """
    Generates a stack of PSF images based on the provided configuration dictionary.
    
    Expected config keys:
    - psf_width_pixels: int (odd number)
    - pixel_size_meters: float
    - wavelength: float
    - numerical_aperture: float
    - refractive_index: float
    - z_start: float (start of z-stack)
    - z_stop: float (end of z-stack)
    - z_step: float (step size for z-stack)
    - output_filename: str (optional, default 'psf_z.mat')
    """
    
    # Extract parameters from config
    psf_width_pixels = config['psf_width_pixels']
    pixel_size_meters = config['px']
    wavelength = config['wavelength']
    numerical_aperture = config['numerical_aperture']
    refractive_index = config['refractive_index']
    
    # Define Z-depth range based on config
    z_start = config.get('z_start', 0)
    z_stop = config.get('z_stop', 4.1e-5)
    z_step = config.get('px', 1e-6)
    
    # Note: adding z_step/100 to stop ensures the last value is included if it matches the step exactly
    z_depth = np.arange(z_start, z_stop + (z_step / 100.0), z_step)
    
    # Calculate physical width of PSF
    psf_width_meters = psf_width_pixels * pixel_size_meters
    
    psf_stack = []
    
    print(f"Generating PSF stack with {len(z_depth)} slices...")
    
    for i, z in enumerate(z_depth):
        if i % 5 == 0: # Print status every 5 slices
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

    # Save to file
    output_filename = config.get('output_filename', 'psf_z.mat')
    sio.savemat(output_filename, {'psf': psf_stack})
    print(f"PSF stack saved to {output_filename}")

    return psf_stack


if __name__ == '__main__':
    # Default configuration matching the original global variables
    default_config = {
        'psf_width_pixels': 101,
        'pixel_size_meters': 1e-6,
        'focal_length': 55e-3,      # Included for completeness, though unused in calc
        'wavelength': 0.561e-6,
        'refractive_index': 1.0,
        'numerical_aperture': 0.6,
        'z_start': 0,
        'z_stop': 4.0e-5,           # Adjusted to hit exactly 41 steps (0 to 40 inclusive)
        'z_step': 1e-6,
        'output_filename': 'psf_z.mat'
    }

    generate_psf(default_config)