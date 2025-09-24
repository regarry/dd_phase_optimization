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
                                grating_angle_deg=0, rotate_angles=None):
    """
    Loads an image (TIFF or BMP) representing a phase mask, normalizes its phase values,
    optionally adds a grating, resizes/pads to SLM dimensions, scales to 8-bit,
    saves as a base BMP, and then optionally rotates the base BMP by specified angles.

    Args:
        input_path (str): Path to the input phase mask image (TIFF or BMP).
        output_directory (str): Directory where the processed BMPs will be saved.
        base_filename (str): Base name for the output BMP file (e.g., "my_phase_mask").
                              Timestamps, grating info, and rotation info will be appended.
        slm_width (int, optional): Desired width of the output image in pixels. If None, original width is used.
        slm_height (int, optional): Desired height of the output image in pixels. If None, original height is used.
        padding_mode (bool): If True, image will be padded; if False, resized.
        add_grating (bool): If True, a blazed grating will be superimposed.
        grating_period_x (float): Period of the blazed grating in x-direction (pixels).
        grating_period_y (float): Period of the blazed grating in y-direction (pixels).
        grating_angle_deg (float): Rotation angle of the grating in degrees.
        rotate_angles (list of int or float, optional): List of angles (in degrees) to rotate
                                                        the final BMP by. If None or empty, no rotation is performed.
    Returns:
        str: Path to the primary (non-rotated) processed BMP, or None if processing fails.
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)

        # 1. Load the image
        print(f"Loading image from: {input_path}")
        img = Image.open(input_path)

        # Convert to a NumPy array for numerical operations
        initial_phase_array = np.array(img, dtype=np.float64)
        original_width, original_height = img.size
        print(f"Original image dimensions: {original_width}x{original_height} pixels")
        print(f"Original image data type: {initial_phase_array.dtype}")
        print(f"Original phase array min: {initial_phase_array.min():.4f}, max: {initial_phase_array.max():.4f}")

        P = 2 * math.pi

        # 2. Normalize phase
        print(f"Normalizing phase to the [0, {P:.4f}) range...")
        normalized_phase_array = (np.fmod(initial_phase_array, P) + P) % P
        print(f"Normalized phase array min: {normalized_phase_array.min():.4f}, max: {normalized_phase_array.max():.4f}")

        # 3. Superimpose Blazed Grating (if enabled)
        current_phase_array_unprocessed = normalized_phase_array
        if add_grating:
            print(f"Superimposing blazed grating with periods X={grating_period_x}, Y={grating_period_y} "
                  f"and angle {grating_angle_deg} degrees on original dimensions...")
            grating_angle_rad = math.radians(grating_angle_deg)
            blazed_grating = generate_blazed_grating(original_width,
                                                     original_height,
                                                     period_x=grating_period_x,
                                                     period_y=grating_period_y,
                                                     angle_rad=grating_angle_rad)

            current_phase_array_unprocessed = (current_phase_array_unprocessed + blazed_grating) % P
            print(f"Phase array after grating superposition min: {current_phase_array_unprocessed.min():.4f}, max: {current_phase_array_unprocessed.max():.4f}")

        target_width = slm_width if slm_width is not None else original_width
        target_height = slm_height if slm_height is not None else original_height

        # 4. Handle resizing or padding
        if target_width != original_width or target_height != original_height:
            if padding_mode:
                print(f"Padding phase mask (with grating) to SLM dimensions: {target_width}x{target_height} pixels...")
                padded_array = np.zeros((target_height, target_width), dtype=np.float64)
                start_x = (target_width - original_width) // 2
                start_y = (target_height - original_height) // 2

                if start_x < 0 or start_y < 0:
                    raise ValueError("Target SLM dimensions are smaller than the original image. Cannot pad, consider resizing.")

                padded_array[start_y:start_y + original_height, start_x:start_x + original_width] = current_phase_array_unprocessed
                current_phase_array = padded_array
                print(f"Padded phase array dimensions: {current_phase_array.shape[1]}x{current_phase_array.shape[0]} pixels")
            else:
                print(f"Resizing phase mask (with grating) to SLM dimensions: {target_width}x{target_height} pixels...")
                temp_img = Image.fromarray(current_phase_array_unprocessed, mode='F')
                resized_img = temp_img.resize((target_width, target_height), Image.Resampling.BICUBIC)
                current_phase_array = np.array(resized_img, dtype=np.float64)
                print(f"Resized phase array dimensions: {current_phase_array.shape[1]}x{current_phase_array.shape[0]} pixels")
        else:
            current_phase_array = current_phase_array_unprocessed

        # 5. Scale to 0-255 range for 8-bit BMP
        if P == 0:
            raise ValueError("Period P (2*pi) cannot be zero for scaling.")

        print("Scaling normalized phase to 0-255 range for 8-bit image...")
        scaled_image_array = (current_phase_array / P) * 255.0
        scaled_image_array = np.round(scaled_image_array).astype(np.uint8)
        print(f"Scaled image array min: {scaled_image_array.min()}, max: {scaled_image_array.max()}")

        output_img = Image.fromarray(scaled_image_array, mode='L') # 'L' for 8-bit grayscale

        # Construct base output filename with timestamp and grating info
        timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        grating_info = ""
        if add_grating:
            grating_info = f"_gratingX{grating_period_x}_Y{grating_period_y}_Ang{int(grating_angle_deg)}"
        
        primary_output_filename = f"{base_filename}_{timestamp_str}{grating_info}.bmp"
        primary_output_path = os.path.join(output_directory, primary_output_filename)

        # 6. Save the primary BMP
        print(f"Saving primary 8-bit BMP image to: {primary_output_path}")
        output_img.save(primary_output_path)
        print("Primary processing complete!")

        # 7. Optional Rotation
        if rotate_angles:
            print(f"\nPerforming rotations on: {primary_output_path}")
            for angle in rotate_angles:
                rotate_bmp_image_and_rename(primary_output_path, angle)
        else:
            print("\nNo rotation angles specified. Skipping image rotation.")

        return primary_output_path

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None
    except Exception as e:
        print(f"An error occurred during phase mask processing: {e}")
        return None


def rotate_bmp_image_and_rename(input_image_path, angle):
    """
    Rotates a BMP image by a specified angle and saves the result
    with "rotated" and the degrees of rotation in the filename.
    The output image will have the same dimensions as the input image,
    which may result in cropping of the rotated content if the rotated
    image extends beyond the original boundaries.

    Args:
        input_image_path (str): The path to the input BMP image.
        angle (int or float): The angle of rotation in degrees. Positive values
                              rotate counter-clockwise, negative values clockwise.
    """
    try:
        # Split the file path into directory, base name, and extension
        directory, filename = os.path.split(input_image_path)
        base_name, extension = os.path.splitext(filename)

        # Construct the new output filename
        angle_str = str(int(angle)) if angle == int(angle) else str(angle).replace('.', '_').replace('-', 'minus')
        output_filename = f"{base_name}_{angle_str}degRotated{extension}"
        output_image_path = os.path.join(directory, output_filename)

        # Open the image
        img = Image.open(input_image_path)

        # Rotate the image, keeping the output image the same size as the input
        rotated_img = img.rotate(angle, expand=False)

        # Save the rotated image
        rotated_img.save(output_image_path)

        print(f"  Successfully rotated by {angle} degrees and saved to {output_image_path}")

    except FileNotFoundError:
        print(f"  Error: Input file not found for rotation at {input_image_path}")
    except Exception as e:
        print(f"  An error occurred during rotation: {e}")


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