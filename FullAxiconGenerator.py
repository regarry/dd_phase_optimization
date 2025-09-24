import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
# Assuming bessel.py and PhaseMaskNorm.py are in the same directory or accessible in PYTHONPATH
# If you combined them, you might not need explicit imports here depending on how you structure
# For clarity, I'm keeping the original structure from your prompt.

# --- Re-including functions for a self-contained script ---
# You can remove these if you are importing them from separate files and just use the imports above
# However, if this is meant to be a single runnable file, keep them.
# For this example, I'll provide a single, combined file as requested.

def generate_blazed_grating(width, height, period_x=0, period_y=0, angle_rad=0):
    """
    Generates a 2D blazed grating phase mask.
    (Existing function)
    """
    grating = np.zeros((height, width), dtype=np.float64)
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    Xr = X * np.cos(-angle_rad) - Y * np.sin(-angle_rad)
    Yr = X * np.sin(-angle_rad) + Y * np.cos(-angle_rad)

    if period_x > 0:
        grating += 2 * math.pi * (Xr / period_x)
    if period_y > 0:
        grating += 2 * math.pi * (Yr / period_y)

    grating = (np.fmod(grating, 2 * math.pi) + 2 * math.pi) % (2 * math.pi)
    return grating

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

def generate_axicon_phase_mask(
    mask_resolution_pixels=(512, 512),
    pixel_pitch_um=10,
    wavelength_nm=632.8,
    bessel_cone_angle_degrees=1.0,
):
    """
    Generates a 2D axicon phase mask for a phase SLM to produce a Bessel beam.
    (Existing function)
    """
    height_pixels, width_pixels = mask_resolution_pixels

    pixel_pitch_m = pixel_pitch_um * 1e-6
    wavelength_m = wavelength_nm * 1e-9
    bessel_cone_angle_rad = np.deg2rad(bessel_cone_angle_degrees)

    x = np.arange(-width_pixels / 2, width_pixels / 2) * pixel_pitch_m
    y = np.arange(-height_pixels / 2, height_pixels / 2) * pixel_pitch_m
    X, Y = np.meshgrid(x, y)

    R = np.sqrt(X**2 + Y**2)

    axicon_constant = (2 * np.pi * np.sin(bessel_cone_angle_rad)) / wavelength_m

    phase_mask = (axicon_constant * R) % (2 * np.pi)
    phase_mask = 2 * np.pi - phase_mask

    return phase_mask

# --- NEW FUNCTION FOR THEORETICAL CALCULATIONS ---
def calculate_bessel_beam_properties(
    slm_width_pixels,
    slm_height_pixels,
    pixel_pitch_um,
    wavelength_nm,
    bessel_cone_angle_degrees
):
    """
    Calculates the theoretical Depth of Field (DOF) and FWHM of the central lobe
    for a Bessel beam generated by an axicon on an SLM.

    Args:
        slm_width_pixels (int): The width of the SLM in pixels.
        slm_height_pixels (int): The height of the SLM in pixels.
        pixel_pitch_um (float): The physical size of each pixel on the SLM in micrometers.
        wavelength_nm (float): The wavelength of the incident light in nanometers.
        bessel_cone_angle_degrees (float): The desired half-cone angle of the Bessel beam
                                            in degrees.

    Returns:
        tuple: A tuple containing (dof_mm, fwhm_um).
               dof_mm (float): Theoretical Depth of Field in millimeters.
               fwhm_um (float): Theoretical FWHM of the central lobe in micrometers.
    """
    # Convert all units to meters for consistent calculations
    pixel_pitch_m = pixel_pitch_um * 1e-6 # um to m
    wavelength_m = wavelength_nm * 1e-9   # nm to m
    
    # Convert cone angle to radians
    alpha_rad = np.deg2rad(bessel_cone_angle_degrees)

    # Calculate the effective diameter of the illuminating beam on the SLM
    # Assuming the beam fills the smaller dimension of the SLM
    # For a circular beam, this would be the diameter of the circle.
    # For a rectangular SLM, it's often limited by the smaller side if the beam is circular.
    # If the beam overfills, it's the smaller SLM dimension.
    # If you have a specific beam diameter D that illuminates the SLM, use that instead.
    # Here, we assume the beam is shaped to fill the SLM's active area,
    # so the effective diameter is the smallest dimension of the SLM.
    D_m = min(slm_width_pixels, slm_height_pixels) * pixel_pitch_m

    # Ensure alpha_rad is not zero to avoid division by zero or tan(0) issues
    if alpha_rad == 0:
        raise ValueError("Bessel cone angle cannot be zero for DOF and FWHM calculation.")
    
    # Theoretical Depth of Field (DOF)
    # DOF = D / (2 * tan(alpha))
    dof_m = D_m / (2 * np.tan(alpha_rad))
    dof_mm = dof_m * 1e3 # Convert to millimeters

    # Theoretical FWHM of the Central Lobe
    # FWHM = 2.405 * lambda / (pi * sin(alpha))
    fwhm_m = (2.405 * wavelength_m) / (math.pi * np.sin(alpha_rad))
    fwhm_um = fwhm_m * 1e6 # Convert to micrometers

    return dof_mm, fwhm_um

# --- Main Script ---
if __name__ == "__main__":
    # --- SLM and Optical System Parameters ---
    slm_target_width = 1920
    slm_target_height = 1152
    slm_pixel_pitch_um = 9.2  # Example: Reflective SLM pixel pitch
    laser_wavelength_nm = 561.0 # Example: Green laser wavelength

    # --- Axicon Parameters ---
    axicon_resolution = (slm_target_height, slm_target_width) 
    desired_bessel_cone_angle_deg = 0.5 # Smaller angle for finer beam/longer DOF

    # --- Blazed Grating Parameters ---
    add_grating_overlay = True
    grating_x_period = 4
    grating_y_period = 0
    grating_angle = 0

    # --- Output File Path ---
    output_dir = "generated_phase_masks"
    os.makedirs(output_dir, exist_ok=True)
    output_axicon_with_grating_bmp = os.path.join(output_dir, "axicon_with_blazed_grating.bmp")
    output_axicon_only_bmp = os.path.join(output_dir, "axicon_only.bmp") # For comparison

    # --- Calculate and Print Theoretical Bessel Beam Properties ---
    print("\n--- Theoretical Bessel Beam Properties ---")
    theoretical_dof_mm, theoretical_fwhm_um = calculate_bessel_beam_properties(
        slm_width_pixels=slm_target_width,
        slm_height_pixels=slm_target_height,
        pixel_pitch_um=slm_pixel_pitch_um,
        wavelength_nm=laser_wavelength_nm,
        bessel_cone_angle_degrees=desired_bessel_cone_angle_deg
    )
    print(f"Theoretical Depth of Field (DOF): {theoretical_dof_mm:.2f} mm")
    print(f"Theoretical FWHM of Central Lobe: {theoretical_fwhm_um:.3f} um")
    print("------------------------------------------")

    print("\n--- Generating Axicon Phase Mask ---")
    axicon_phase_radians = generate_axicon_phase_mask(
        mask_resolution_pixels=axicon_resolution,
        pixel_pitch_um=slm_pixel_pitch_um,
        wavelength_nm=laser_wavelength_nm,
        bessel_cone_angle_degrees=desired_bessel_cone_angle_deg
    )

    print(f"Generated axicon phase mask (raw) shape: {axicon_phase_radians.shape}")
    print(f"Axicon phase values range: [{axicon_phase_radians.min():.4f}, {axicon_phase_radians.max():.4f}] radians")

    # --- Process Axicon ONLY (for comparison) ---
    print("\n--- Processing Axicon Mask WITHOUT Grating Overlay ---")
    process_phase_mask(
        input_phase_array=axicon_phase_radians,
        output_bmp_path=output_axicon_only_bmp,
        slm_width=slm_target_width,
        slm_height=slm_target_height,
        padding_mode=False,
        add_grating=False
    )
    print(f"Axicon only phase mask saved to: {output_axicon_only_bmp}")

    # --- Process Axicon with Blazed Grating Overlay ---
    print("\n--- Processing Axicon Mask WITH Blazed Grating Overlay ---")
    process_phase_mask(
        input_phase_array=axicon_phase_radians,
        output_bmp_path=output_axicon_with_grating_bmp,
        slm_width=slm_target_width,
        slm_height=slm_target_height,
        padding_mode=False,
        add_grating=add_grating_overlay,
        grating_period_x=grating_x_period,
        grating_period_y=grating_y_period,
        grating_angle_deg=grating_angle
    )
    print(f"Axicon with blazed grating phase mask saved to: {output_axicon_with_grating_bmp}")

    print("\n--- Visualization (Optional) ---")
    try:
        plt.figure(figsize=(8, 7))
        plt.imshow(axicon_phase_radians, cmap='hsv', origin='lower')
        plt.colorbar(label='Phase (radians)')
        plt.title('Generated Axicon Phase Mask (Raw)')
        plt.show()

        img_with_grating = Image.open(output_axicon_with_grating_bmp)
        plt.figure(figsize=(8, 7))
        plt.imshow(np.array(img_with_grating), cmap='gray', origin='lower')
        plt.colorbar(label='Grayscale Value (0-255)')
        plt.title('Axicon with Blazed Grating (8-bit BMP)')
        plt.show()

    except Exception as e:
        print(f"Matplotlib visualization failed. Ensure it is installed: {e}")