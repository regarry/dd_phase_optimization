import os
import yaml
import glob
import re
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import torch
import tifffile
import imageio

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_config(config_file):
    with open(config_file, 'r') as stream:
        return yaml.load(stream, Loader=yaml.Loader)

def normalize_to_uint16(img: np.ndarray) -> np.ndarray:
    """Normalizes a numpy array to the uint16 range [0, 65535]."""
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    img_float = img.astype(np.float64)
    img_float -= img_float.min()
    max_val = img_float.max()
    if max_val > 0:
        img_float /= max_val
    return (img_float * 65535).astype(np.uint16)

def img_save_tiff(img, out_dir, name, key=None, as_rgb=False, as_rgb_channels=False):
    """Save 3D or 2D image arrays as TIFFs."""
    if key is None:
        base_filename = f"{name}.tiff"
        base_path = os.path.join(out_dir, base_filename)
    else:
        base_filename = f"{name}_{key}.tiff"
        base_path = os.path.join(out_dir, base_filename)

    def to_rgb(arr):
        rgb = np.zeros(arr.shape + (3,), dtype=np.uint8)
        rgb[arr == 1] = [255, 0, 0]
        rgb[arr == 2] = [0, 255, 0]
        return rgb

    def scale_custom_rgb_channels(arr):
        if arr.ndim == 3:
            if arr.shape[0] == 3 and arr.shape[2] != 3:
                arr = np.transpose(arr, (1, 2, 0))
            elif arr.shape[2] != 3:
                raise ValueError("Input must have 3 channels.")
            arr = np.clip(arr, 0, 1)
            h, w, _ = arr.shape
            rgb = np.zeros((h, w, 3), dtype=np.float32)
            rgb[..., 2] += arr[..., 0]  # B
            rgb[..., 0] += arr[..., 1]  # R
            rgb[..., 1] += arr[..., 2]  # G
            rgb = np.clip(rgb, 0, 1)
            return (rgb * 255).astype(np.uint8)
        else:
            raise ValueError("Input must be a 3-channel 2D image.")

    if as_rgb_channels:
        rgb_img = scale_custom_rgb_channels(img)
        skimage.io.imsave(base_path, rgb_img)
        print(f"Saved {name} as custom RGB channels to {base_path}")
        return

    if img.ndim == 3:
        for i in range(img.shape[0]):
            path = base_path if img.ndim == 2 else \
                   os.path.join(out_dir, f"{name}_{key if key else ''}_z_{i-(img.shape[0]//2)}.tiff")
            
            if as_rgb:
                skimage.io.imsave(path, to_rgb(img[i]))
            else:
                max_val = np.max(img[i])
                norm_img = (img[i] / max_val * 255).astype(np.uint8) if max_val > 0 else img[i].astype(np.uint8)
                skimage.io.imsave(path, norm_img)
    elif img.ndim == 2:
        if as_rgb:
            skimage.io.imsave(base_path, to_rgb(img))
        else:
            skimage.io.imsave(base_path, img)
            
def save_png(image, output_path, config):
    """Saves a given image as a PNG file with a formatted title and colorbar."""
    png_path = output_path
    plt.figure(figsize=(image.shape[1] / 30, image.shape[0] / 30))
    plt.imshow(normalize_to_uint16(image), cmap='hot', aspect='equal')
    plt.colorbar()
    plt.title(f"Pixel size: {config['px']*10**6:.2f} um")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close('all')


def save_normalized_png(data_array, output_path):
    """
    Normalizes a 2D numpy array to 0-255 and saves it as a PNG.
    
    Args:
        data_array (np.ndarray): The 2D array to save.
        output_path (str): The full path (including filename) to save the image.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Calculate min and max for the specific slice
    local_min = data_array.min()
    local_max = data_array.max()

    if local_max > local_min:
        # Scale to 0-1 float based on the array's own range
        normalized = (data_array - local_min) / (local_max - local_min)
        # Convert to 0-255 uint8
        img_8bit = (normalized * 255).astype(np.uint8)
    else:
        # Handle case where array is flat (all zeros or all same value)
        img_8bit = np.zeros_like(data_array, dtype=np.uint8)

    # Save the file
    imageio.imwrite(output_path, img_8bit)

def save_3d_volume_as_tiffs(volume, out_dir, base_name_prefix=""):
    """Save a 3D numpy array as a series of 2D tiff images."""
    D = volume.shape[1]
    # Assuming volume is (Batch, Depth, Height, Width), we take the first batch item
    vol_data = volume[0] 
    for z in range(D):
        base_name = f"{base_name_prefix}_{z:02d}"
        img_save_tiff(vol_data[z], out_dir, base_name, as_rgb=True)

def savePhaseMask(mask_param, epoch, res_dir):
    mask_numpy = mask_param.data.cpu().clone().numpy()
    skimage.io.imsave(os.path.join(res_dir, f'mask_phase_epoch_{epoch}.tiff'), mask_numpy.squeeze(), check_contrast=False)

def save_output_layer(output_layer, base_dir, lens_approach, counter, datetime_str, config):
    output_array = torch.square(torch.abs(output_layer)).cpu().detach().numpy()[0,0,:,:]
    filename = f"{counter}.tiff"
    dir_path = os.path.join(base_dir, lens_approach, datetime_str)
    makedirs(dir_path)
    skimage.io.imsave(os.path.join(dir_path, filename), output_array)
    
    if counter == 0:
        with open(os.path.join(dir_path, 'config.yaml'), 'w') as file:
            for key, value in config.items():
                file.write(f'{key}: {value}\n')

def save_output_layer_complex(output_layer, base_dir, lens_approach, counter, datetime_str, config):
    output_array = output_layer.cpu().detach().numpy()[0, 0, :, :]
    filename = f"{counter}_complex.npy"
    dir_path = os.path.join(base_dir, lens_approach, datetime_str)
    makedirs(dir_path)
    np.save(os.path.join(dir_path, filename), output_array)
    
    if counter == 0:
        with open(os.path.join(dir_path, 'config.yaml'), 'w') as file:
            for key, value in config.items():
                file.write(f'{key}: {value}\n')

def find_image_with_wildcard(directory, filename_prefix, file_type):
    search_pattern = os.path.join(directory, filename_prefix + "*." + file_type)
    matching_files = glob.glob(search_pattern)
    if not matching_files:
        return None
    return matching_files[0]

def extract_datetime_and_epoch(filepath):
    datetime_match = re.search(r'phase_model_(\d{8}-\d{6})', filepath)
    epoch_match = re.search(r'epoch_(\d+)_', filepath)
    datetime_str = datetime_match.group(1) if datetime_match else None
    epoch_num = int(epoch_match.group(1)) if epoch_match else None
    return f"{datetime_str}-{epoch_num}"

def load_tiff(path):
    """
    Reads a TIFF stack from disk.
    Args:
        path (str): Path to the .tif file.
    Returns:
        np.ndarray: The image stack.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"TIFF file not found: {path}")
    
    # skimage handles multipage tiffs automatically
    return skimage.io.imread(path)

def expand_config(config, training_results_dir):
    """
    Calculates derived parameters (pixel sizes, spatial ranges, volumes)
    based on the core settings in the config YAML.
    """
    # 1. Pixel Size Calculation
    # We calculate the effective pixel size at the sample plane
    config['training_results_dir'] = training_results_dir
    config['4f_magnification'] = config['focal_length_2'] / config['focal_length_1']
    config['scale_factor'] = config['phase_mask_upsample_factor'] * config['4f_magnification']
    config['px'] = config['slm_px'] / config['phase_mask_upsample_factor']
    pixel_size = config['px']
    if config['lens_approach'] == 'lazy_4f':
        config['N'] = int(config['phase_mask_pixel_size'] * config['scale_factor'])
    else:  
        config['N'] = config['phase_mask_upsample_factor'] * config['phase_mask_pixel_size']

    if config["spatial_multi_well_loss_weight"] > 0:
        config["SpatialMulitWellLoss"] = True
    
    if config["mse_loss_weight"] > 0:
        config["mse_loss"] = True
    # 2. Derive Bead Volume from Image Volume
    # The 'image_volume' is the total field of view. 
    # The 'bead_volume' is the safe zone where beads can exist (avoiding edges).
    image_volume = config['image_volume'] # the first two dimensions should be powers of 2 for the u-net [y, x, z] volume of imaging in px (camera coordinates)
    
     
    # 1. Calculate max defocus depth in meters
    z_max = image_volume[2] * pixel_size / 2

    # 2. Calculate the physical radius of the light cone at z_max
    theta = np.arcsin(config['numerical_aperture'] / config['refractive_index'])
    max_blur_radius_meters = z_max * np.tan(theta)

    # 3. Convert to pixels and pad it so the edges safely reach zero
    max_blur_radius_pixels = np.ceil(max_blur_radius_meters / pixel_size)
    padding = config.get('psf_padding', 10)# Extra pixels to ensure the diffraction rings trail off to 0

    # 4. Set the new dynamic width (Diameter = 2 * Radius + 1 for center pixel)
    psf_width_pixels = int(2 * (max_blur_radius_pixels + padding)) | 1

    config['psf_width_pixels'] = psf_width_pixels
    psf_keep_radius = psf_width_pixels // 2
    config['psf_keep_radius'] = psf_keep_radius
    
    print(f"Auto-calculated PSF canvas width: {psf_width_pixels} x {psf_width_pixels} pixels")

    # image volume the first two dimensions should be powers of 2 for the u-net [y, x, z] volume of imaging in px (camera coordinates)
    # round the psf width to the nearest power of 2 to ensure the PSF fits within the image y volume
    image_volume[0] = int(2 ** np.ceil(np.log2(psf_width_pixels)))
    config['image_volume'] = image_volume
    print(f"Adjusted image volume to ensure PSF fits: {image_volume} pixels (y, x, z)")
    print(f'PSF Width in pixels: {psf_width_pixels}, which corresponds to a max blur radius of {max_blur_radius_pixels:.1f} pixels at the maximum defocus depth.')
    bead_vol_y = image_volume[0] - psf_width_pixels # this one should be auto set to prevent errors with larger psfs
    bead_vol_x = image_volume[1] - psf_width_pixels
    bead_vol_z = image_volume[2] 

    config['bead_volume'] = [bead_vol_y, bead_vol_x, bead_vol_z]
    
    print(f"Derived bead volume (safe zone for emitters): {config['bead_volume']} pixels")

    # 3. Set Spatial Ranges (for the generator)
    # XY: Start at radius, end at image_size - radius
    config['particle_spatial_range_x'] = [psf_keep_radius, image_volume[1] - psf_keep_radius]
    config['particle_spatial_range_y'] = [psf_keep_radius, image_volume[0] - psf_keep_radius]

    # Z: Center around 0. If Depth is 30, range is -15 to +15.
    z_length = image_volume[2]
    z_start = -(z_length // 2)
    z_end = z_start + z_length
    config['particle_spatial_range_z'] = [z_start, z_end]
    
    z_depth_list = config['z_depth_list']
    Nimgs = len(z_depth_list)
    config['Nimgs'] = Nimgs

    # 5. Validity Checks
    if config['num_classes'] > 1:
        z_coupled_ratio = config.get('z_coupled_ratio', 0)
        z_coupled_spacing = config.get('z_coupled_spacing_range', (0,0))
        if z_coupled_ratio <= 0 or z_coupled_spacing[0] <= 0:
             raise ValueError("For multi-class, you must set z_coupled_ratio > 0 and z_coupled_spacing_range.")

    config['defocused_beads_filename'] = config.get('defocused_beads_filename', 'defocused_beads.mat')
    
    # this will be half of the bead volume z
    config['spatial_well_y'] = config['bead_volume'][2] // 2 + 1
    return config

def load_tiff_sequence(data_path):
    """
    Loads all TIFF files from a directory into a list, prints their shapes and total memory usage.
    Returns the list of images.
    """
    tiff_files = sorted([f for f in os.listdir(data_path) if f.lower().endswith('.tiff')])
    print(f"Found {len(tiff_files)} TIFF files in {data_path}:")
    imgs = []
    for idx, fname in enumerate(tiff_files):
        img_path = os.path.join(data_path, fname)
        img = tifffile.imread(img_path)
        imgs.append(img)
        print(f"Loaded [{idx}] {fname} with shape {img.shape}")
    print(f"Total images loaded: {len(imgs)}")
    total_bytes = sum(img.nbytes for img in imgs)
    print(f"Total memory size: {total_bytes / 1024**2:.2f} MB")
    return imgs