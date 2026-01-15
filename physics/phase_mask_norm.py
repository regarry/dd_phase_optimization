import numpy as np
from PIL import Image
import math
import os
from datetime import datetime
#from data_utils import extract_datetime_and_epoch

def process_phase_mask(input_phase_array, output_bmp_path, slm_width=None, slm_height=None, padding_mode=False,
                          add_grating=False, grating_period_x=0, grating_period_y=0, grating_angle_deg=0):
    """
    Processes a NumPy array representing a phase mask, normalizes its phase values
    to the [0, 2*pi) range, scales them to [0, 255], and saves as an 8-bit BMP.
    (Existing function)
    """
    try:
        original_height, original_width = input_phase_array.shape
        print(f"Initial phase mask dimensions: {original_width}x{original_height} pixels")
        print(f"Initial phase array min: {input_phase_array.min():.4f}, max: {input_phase_array.max():.4f}")

        P = 2 * math.pi

        print(f"Normalizing phase to the [0, {P:.4f}) range...")
        normalized_phase_array = (np.fmod(input_phase_array, P) + P) % P
        print(f"Normalized phase array min: {normalized_phase_array.min():.4f}, max: {normalized_phase_array.max():.4f}")

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
    
        if P == 0:
            raise ValueError("Period P (2*pi) cannot be zero for scaling.")

        print("Scaling normalized phase to 0-255 range for 8-bit image...")
        scaled_image_array = (current_phase_array / P) * 255.0
        
        scaled_image_array = np.round(scaled_image_array).astype(np.uint8)
        print(f"Scaled image array min: {scaled_image_array.min()}, max: {scaled_image_array.max()}")

        output_img = Image.fromarray(scaled_image_array, mode='L')
        
        print(f"Saving 8-bit BMP image to: {output_bmp_path}")
        output_img.save(output_bmp_path)
        print("Processing complete!")

    except Exception as e:
        print(f"An error occurred: {e}")

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


def normalize_phase_image(input_tiff_path, output_bmp_path, slm_width=None, slm_height=None, padding_mode=False,
                          add_grating=False, grating_period_x=0, grating_period_y=0, grating_angle_deg=0):
    """
    Loads a TIFF image representing a phase mask, normalizes its phase values
    to the [0, 2*pi) range, scales them to [0, 255], and saves as an 8-bit BMP.
    Applies blazed grating (if specified) BEFORE resizing/padding.
    Optionally resizes or pads the phase mask to match the SLM's dimensions.

    Args:
        input_tiff_path (str): Path to the input TIFF phase mask image.
        output_bmp_path (str): Path to save the output 8-bit BMP image.
        slm_width (int, optional): Desired width of the output image in pixels,
                                    matching the SLM's resolution. If None, original width is used.
        slm_height (int, optional): Desired height of the output image in pixels,
                                     matching the SLM's resolution. If None, original height is used.
        padding_mode (bool): If True and slm_width/height are provided, the image will be padded
                             to the target dimensions instead of resized. If False, it will be resized.
        add_grating (bool): If True, a blazed grating will be superimposed on the phase mask.
        grating_period_x (float): Period of the blazed grating in the x-direction (pixels).
                                  Only used if add_grating is True.
        grating_period_y (float): Period of the blazed grating in the y-direction (pixels).
                                  Only used if add_grating is True.
        grating_angle_deg (float): Rotation angle of the grating in degrees.
                                   Only used if add_grating is True.
    """
    try:
        # 1. Load the TIFF image
        print(f"Loading TIFF image from: {input_tiff_path}")
        img = Image.open(input_tiff_path)

        # Convert to a NumPy array for numerical operations
        initial_phase_array = np.array(img, dtype=np.float64)
        original_width, original_height = img.size
        print(f"Original image dimensions: {original_width}x{original_height} pixels")
        print(f"Original image data type: {initial_phase_array.dtype}")
        print(f"Original phase array min: {initial_phase_array.min()}, max: {initial_phase_array.max()}")

        # Define P (2*pi for phase normalization)
        P = 2 * math.pi

        # 2. Apply the normalization formula: (initial_phase % P + P) % P
        print(f"Normalizing phase to the [0, {P:.4f}) range...")
        normalized_phase_array = (np.fmod(initial_phase_array, P) + P) % P
        print(f"Normalized phase array min: {normalized_phase_array.min()}, max: {normalized_phase_array.max()}")

        # --- NEW POSITION FOR GRATING APPLICATION ---
        # 2.5. Superimpose Blazed Grating on the original dimensions
        current_phase_array_unprocessed = normalized_phase_array # Start with the normalized original phase
        if add_grating:
            print(f"Superimposing blazed grating with periods X={grating_period_x}, Y={grating_period_y} "
                  f"and angle {grating_angle_deg} degrees on original dimensions...")
            grating_angle_rad = math.radians(grating_angle_deg)
            blazed_grating = generate_blazed_grating(original_width,
                                                     original_height,
                                                     period_x=grating_period_x,
                                                     period_y=grating_period_y,
                                                     angle_rad=grating_angle_rad)
            
            # Add the grating phase to the current phase array
            # The result is then normalized back to [0, 2*pi)
            current_phase_array_unprocessed = (current_phase_array_unprocessed + blazed_grating) % P
            print(f"Phase array after grating superposition min: {current_phase_array_unprocessed.min()}, max: {current_phase_array_unprocessed.max()}")
        # --- END NEW POSITION ---

        target_width = slm_width if slm_width is not None else original_width
        target_height = slm_height if slm_height is not None else original_height

        # 2.75. Handle resizing or padding (now applied to the grating-added phase)
        if target_width != original_width or target_height != original_height:
            if padding_mode:
                print(f"Padding phase mask (with grating) to SLM dimensions: {target_width}x{target_height} pixels...")
                # Create a new array filled with 0.0 (representing 0 phase)
                padded_array = np.zeros((target_height, target_width), dtype=np.float64)

                # Calculate start coordinates for pasting the original image
                start_x = (target_width - original_width) // 2
                start_y = (target_height - original_height) // 2

                # Ensure dimensions fit
                if start_x < 0 or start_y < 0:
                    raise ValueError("Target SLM dimensions are smaller than the original image. Cannot pad, consider resizing.")

                # Place the phase array (which now includes grating) into the center of the padded array
                padded_array[start_y:start_y + original_height, start_x:start_x + original_width] = current_phase_array_unprocessed
                current_phase_array = padded_array
                print(f"Padded phase array dimensions: {current_phase_array.shape[1]}x{current_phase_array.shape[0]} pixels")

            else: # Resizing mode (default if padding_mode is False)
                print(f"Resizing phase mask (with grating) to SLM dimensions: {target_width}x{target_height} pixels...")
                temp_img = Image.fromarray(current_phase_array_unprocessed, mode='F') # Use the array with grating for resizing
                resized_img = temp_img.resize((target_width, target_height), Image.Resampling.BICUBIC)
                current_phase_array = np.array(resized_img, dtype=np.float64)
                print(f"Resized phase array dimensions: {current_phase_array.shape[1]}x{current_phase_array.shape[0]} pixels")
        else:
            current_phase_array = current_phase_array_unprocessed # No resizing/padding needed, use the grating-added array
    
        # 3. Scale the (potentially resized/padded/grating-added) normalized phase from [0, 2*pi) to [0, 255] for 8-bit BMP
        if P == 0:
            raise ValueError("Period P (2*pi) cannot be zero for scaling.")

        print("Scaling normalized phase to 0-255 range for 8-bit image...")
        scaled_image_array = (current_phase_array / P) * 255.0
        
        # Round to nearest integer and ensure values are within 0-255
        scaled_image_array = np.round(scaled_image_array).astype(np.uint8)
        print(f"Scaled image array min: {scaled_image_array.min()}, max: {scaled_image_array.max()}")

        # 4. Save the resulting image as an 8-bit BMP
        output_img = Image.fromarray(scaled_image_array, mode='L') # 'L' for 8-bit grayscale
        
        print(f"Saving 8-bit BMP image to: {output_bmp_path}")
        output_img.save(output_bmp_path)
        print("Processing complete!")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_tiff_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    # Define input and output file paths
    # IMPORTANT: Replace these paths with the actual paths to your TIFF file and desired output.
    
    # Updated file paths as per your request
    input_file = "training_results/20260115-140727/mask_phase_epoch_99.tiff"
    parent_dir = os.path.dirname(input_file)
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    #datetime_epoch = extract_datetime_and_epoch(input_file)
    output_file = os.path.join(parent_dir,"normalized_phase", f"{datetime_str}.bmp")
    #output_file_with_grating = os.path.join(parent_dir,"normalized_phase", f"g_{datetime_epoch}.bmp")

    # --- SET YOUR SLM DIMENSIONS HERE ---
    slm_target_width = 1920
    slm_target_height = 1152

    # --- CHOOSE PADDING MODE ---
    # Set padding_mode=True to pad the image.
    # Set padding_mode=False (or omit) to resize the image (previous behavior).
    use_padding = True 

    # --- GRATING PARAMETERS ---
    # Set add_grating=True to enable grating superposition
    add_grating_to_output = True
    grating_x_period = 4 # Period in pixels. Adjust this value to change the grating frequency.
    grating_y_period = 0 # Set to 0 if you only want a grating along X, or specify for Y.
    grating_angle = 0 # Angle in degrees. 0 degrees means grating lines are vertical (blaze horizontal).

    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    #os.makedirs(os.path.dirname(output_file_with_grating), exist_ok=True)

    print("--- Processing WITHOUT Grating ---")
    normalize_phase_image(input_file, output_file, 
                          slm_width=slm_target_width, 
                          slm_height=slm_target_height, 
                          padding_mode=use_padding,
                          add_grating=False) # Ensure grating is NOT added for this output

    print(f"\nCheck '{output_file}' for the normalized and {'padded' if use_padding else 'resized'} 8-bit phase mask (without grating).")

    print("\n--- Processing WITH Grating (applied BEFORE resize/padding) ---")
    # normalize_phase_image(input_file, output_file_with_grating, 
    #                       slm_width=slm_target_width, 
    #                       slm_height=slm_target_height, 
    #                       padding_mode=use_padding,
    #                       add_grating=add_grating_to_output,
    #                       grating_period_x=grating_x_period,
    #                       grating_period_y=grating_y_period,
    #                       grating_angle_deg=grating_angle)

    #print(f"\nCheck '{output_file_with_grating}' for the normalized and {'padded' if use_padding else 'resized'} 8-bit phase mask WITH blazed grating.")
    print("Remember to replace 'input_phase_mask.tiff' with your actual input file path.")