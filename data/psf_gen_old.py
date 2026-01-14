# -*- coding: utf-8 -*-

import os
import numpy as np
import imageio
import skimage
import scipy.integrate
import scipy.signal
import scipy.special
import scipy.io as sio
import math

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


def generate_psf(config, results_dir):
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
    - psf_images_path: str (optional, directory to save PNG images)
    """

    # Extract parameters from config
    psf_width_pixels = 2 * (config['psf_keep_radius'] * 2 + 1)  # Ensure odd size
    pixel_size_meters = config['px']
    wavelength = config['wavelength']
    numerical_aperture = config['numerical_aperture']
    refractive_index = config['refractive_index']
    
    psf_stack_path = os.path.join(results_dir, 'psf_z.mat')
    psf_images_path = os.path.join(results_dir, 'psf_imgs/')
    psf_txt_path = os.path.join(results_dir, 'psf_z.txt')
    
    # Define Z-depth range based on config
    z_start = 0
    z_stop = config['bead_volume'][2] * config['px'] / 2 + config['psf'] # half bead volume depth in meters
    z_step = config['px']
    
    z_depth = np.arange(z_start, z_stop, z_step)
    
    # Calculate physical width of PSF
    psf_width_meters = psf_width_pixels * pixel_size_meters
    
    psf_stack = []
    
    os.makedirs(psf_images_path, exist_ok=True)
    
    print(f"Generating PSF stack with {len(z_depth)} slices...")
    if os.path.exists(psf_txt_path):
        os.remove(psf_txt_path)
        
    # Save config at the start of psf_z.txt
    with open(psf_txt_path, "w") as log_file:
        log_file.write("Config:\n")
        for key, value in config.items():
            log_file.write(f"{key}: {value}\n")
        log_file.write("\n")
    for i, z in enumerate(z_depth):
        log_msg = f"Processing slice {i}/{len(z_depth)-1} at Z={z:.2e}"
        print(log_msg)
        with open(psf_txt_path, "a") as log_file:
            log_file.write(log_msg + "\n")
            
        slice_psf = get_airy_psf(
            psf_width_pixels, 
            psf_width_meters, 
            z, 
            wavelength, 
            numerical_aperture,
            refractive_index
        )
        psf_stack.append(slice_psf)

        # Save as 8-bit PNG if path is provided
        psf_8bit = np.clip(slice_psf / np.max(slice_psf) * 255, 0, 255).astype(np.uint8)
        imageio.imwrite(
            os.path.join(psf_images_path, f"psf_z_{i:03d}.png"),
            psf_8bit
        )

    # Save to file
    sio.savemat(psf_stack_path, {'psf': psf_stack})
    print(f"PSF stack saved to {psf_stack_path}")

    return psf_stack

# AG this part is generating the images of the defocused images at different planes 
# the code will save these images as template for future use.
# The code takes into aacount large beads, as well as small
def generate_bead_defocus_stack(config, results_dir, setup_defocus_psf=None):
    #Unpack the parameters
    psf_width_pixels = config['psf_width_pixels']
    bead_defocus_img_path = os.path.join(results_dir, 'bead_defocus_imgs/')
    bead_radius = config['bead_radius']
    bead_ori_img = np.zeros((psf_width_pixels,psf_width_pixels))  
    if setup_defocus_psf is None:
        setup_defocus_psf = sio.loadmat('psf_z.mat')['psf'] #this is a matrix that Chen has generated before
    else:
        setup_defocus_psf = setup_defocus_psf
    bead_intensity = config['bead_intensity']
    #max_defocus = config['max_defocus']
    

    os.makedirs(bead_defocus_img_path, exist_ok=True)

    #generate the bead in the center based on the bead size in pixels
    for x in range(int(math.floor(psf_width_pixels/2)-bead_radius), int(math.floor(psf_width_pixels/2)+bead_radius+1)):
        for y in range(int(math.floor(psf_width_pixels/2)-bead_radius), int(math.floor(psf_width_pixels/2)+bead_radius+1)):
            if (x - math.floor(psf_width_pixels/2))**2 + (y - math.floor(psf_width_pixels/2))**2 <= bead_radius**2:
                bead_ori_img[x,y] = bead_intensity
    #Smooth every bead
    bead_ori_img = skimage.filters.gaussian(bead_ori_img, sigma=1)

    #Create the images that have the PSF, based on distance from the detection lens focal point
    #41 refers to 41 um maximum defocus in the simulation; the volume was 200x200x30um^3, 30 um in Z so max 15 um defocus
    for i in range(len(setup_defocus_psf)):
        blurred_img = apply_blur_kernel(bead_ori_img, setup_defocus_psf[i])
        skimage.io.imsave(os.path.join(bead_defocus_img_path, 'z' + str(i).zfill(2) + '.tiff'), blurred_img.astype('uint16'))
    return None

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

    #generate_psf(default_config)