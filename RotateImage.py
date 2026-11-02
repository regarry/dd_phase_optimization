from pathlib import Path
import json
from physics.phase_io import load_phase, save_codes
from physics.phase_mask_norm import process_phase_mask
import numpy as np
from PIL import Image
import math
import os
from datetime import datetime

def generate_blazed_grating(width, height, period_x=0, period_y=0, angle_rad=0):
    """
    Generates a 2D blazed grating phase mask.

    Args:
        width (int): The width of the grating in pixels.
        height (int): The height of the grating in pixels.
        period_x (float): The period of the grating in the x-direction (pixels).
                          If 0, no grating in x.
        period_y (float): The period of the grating in the y-direction (pixels).
                          If 0, no grating in y.
        angle_rad (float): The rotation angle of the grating in radians.
                           A positive angle rotates the grating counter-clockwise.

    Returns:
        numpy.ndarray: A 2D array representing the blazed grating phase in [0, 2*pi).
    """
    grating = np.zeros((height, width), dtype=np.float64)
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    # Apply rotation
    # Rotate coordinates by -angle_rad to rotate the grating itself by angle_rad
    Xr = X * np.cos(-angle_rad) - Y * np.sin(-angle_rad)
    Yr = X * np.sin(-angle_rad) + Y * np.cos(-angle_rad)

    if period_x > 0:
        grating += 2 * math.pi * (Xr / period_x)
    if period_y > 0:
        grating += 2 * math.pi * (Yr / period_y)

    # Normalize to [0, 2*pi)
    grating = (np.fmod(grating, 2 * math.pi) + 2 * math.pi) % (2 * math.pi)
    return grating


def process_phase_mask_and_save(input_path, output_directory, base_filename, slm_width=None, slm_height=None,
                                padding_mode=False, add_grating=False, grating_period_x=0, grating_period_y=0,
                                grating_angle_deg=0, rotate_angles=None, bit_depth=8,
                                input_units=None, input_bit_depth=None):
    """Shared calibrated export contract; rotations use nearest-neighbor phase sampling."""
    phase = load_phase(input_path, input_units, input_bit_depth)
    suffix = '.bmp' if bit_depth <= 8 else '.tif'
    path = os.path.join(output_directory, f"{base_filename}_{datetime.now():%Y%m%d-%H%M%S}{suffix}")
    process_phase_mask(phase, path, slm_width, slm_height, padding_mode, add_grating,
                       grating_period_x, grating_period_y, grating_angle_deg, bit_depth)
    for angle in rotate_angles or []:
        rotate_bmp_image_and_rename(path, angle)
    return path


def rotate_bmp_image_and_rename(input_image_path, angle, input_units=None, bit_depth=None):
    """Preserve phase units/bit depth and image size; content outside the frame is cropped."""
    path = Path(input_image_path)
    metadata = Path(str(path) + '.json')
    if metadata.exists():
        bit_depth = json.loads(metadata.read_text())['bit_depth']
    phase = load_phase(path, input_units, bit_depth)
    if bit_depth is None:
        raise ValueError('Specify bit_depth for a legacy rotation input')
    rotated = np.array(Image.fromarray(phase).rotate(angle, resample=Image.Resampling.NEAREST,
                                                    expand=False, fillcolor=0.0))
    output = path.with_name(f'{path.stem}_{angle:g}degRotated{path.suffix}')
    save_codes(rotated, output, bit_depth)
    return str(output)


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Input phase mask file (TIFF or BMP)
    input_file = r"training_results/phase_model_20250724-141949/mask_phase_epoch_55_.tiff"
    
    # Output directory for all processed BMPs
    output_base_dir = r"processed_phase_masks"
    
    # SLM dimensions (desired output image size)
    slm_target_width = 1920
    slm_target_height = 1152

    # Resizing or Padding mode
    # Set to True for padding (adds black borders), False for resizing (stretches/shrinks)
    use_padding = True

    # Grating parameters
    add_grating = True
    grating_x_period = 8  # Period in pixels. Set to 0 for no X grating.
    grating_y_period = 0  # Period in pixels. Set to 0 for no Y grating.
    grating_angle_deg = 0 # Angle in degrees. 0 means grating lines are vertical (blaze horizontal).

    # Angles for post-processing rotation (applied to the final BMP).
    # Provide a list of angles. Set to [] or None if no rotation is desired.
    # Positive values rotate counter-clockwise, negative values clockwise.
    angles_to_rotate = [-105] # Example: Rotate by -90, 0, 90, and 180 degrees


    # --- SCRIPT EXECUTION ---

    # Ensure the input file exists for testing
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        print("Creating a dummy TIFF file for demonstration purposes...")
        os.makedirs(os.path.dirname(input_file), exist_ok=True)
        dummy_array = np.linspace(0, 2 * np.pi, 100 * 100).reshape((100, 100)) # A simple gradient phase
        dummy_img = Image.fromarray((dummy_array / (2 * np.pi) * 255).astype(np.uint8), mode='L')
        dummy_img.save(input_file)
        print(f"Dummy TIFF created at: {input_file}")
        print("Please replace this dummy file with your actual TIFF phase mask for real processing.")
        print("-" * 50)

    # Process the phase mask with grating
    print("\n--- Processing Phase Mask WITH Grating ---")
    processed_bmp_path_with_grating = process_phase_mask_and_save(
        input_path=input_file,
        output_directory=output_base_dir,
        base_filename="phase_mask_with_grating",
        slm_width=slm_target_width,
        slm_height=slm_target_height,
        padding_mode=use_padding,
        add_grating=add_grating,
        grating_period_x=grating_x_period,
        grating_period_y=grating_y_period,
        grating_angle_deg=grating_angle_deg,
        rotate_angles=angles_to_rotate
    )

    if processed_bmp_path_with_grating:
        print(f"\nFinal processed BMP (with grating) saved to: {processed_bmp_path_with_grating}")
        print(f"Check '{os.path.dirname(processed_bmp_path_with_grating)}' for the base image and any rotated versions.")
    else:
        print("\nPhase mask processing with grating failed.")

    # Process the phase mask without grating (optional, for comparison)
    print("\n--- Processing Phase Mask WITHOUT Grating (for comparison) ---")
    processed_bmp_path_no_grating = process_phase_mask_and_save(
        input_path=input_file,
        output_directory=output_base_dir,
        base_filename="phase_mask_no_grating",
        slm_width=slm_target_width,
        slm_height=slm_target_height,
        padding_mode=use_padding,
        add_grating=False, # Explicitly set to False for this run
        rotate_angles=angles_to_rotate # Still apply rotations if desired
    )

    if processed_bmp_path_no_grating:
        print(f"\nFinal processed BMP (no grating) saved to: {processed_bmp_path_no_grating}")
        print(f"Check '{os.path.dirname(processed_bmp_path_no_grating)}' for the base image and any rotated versions.")
    else:
        print("\nPhase mask processing without grating failed.")