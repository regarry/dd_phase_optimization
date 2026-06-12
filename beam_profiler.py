import os
import argparse
import numpy as np
import torch
import skimage.io
from skimage.restoration import unwrap_phase
import matplotlib.pyplot as plt
#from datetime import datetime
from data.io import load_config, normalize_to_uint16
#from beam_profile_gen import BeamProfiler
from physics.bessel import generate_axicon_phase_mask
from physics.simulation import OpticsSimulation
from physics.masks import get_initial_phase_mask
plt.rcParams['image.interpolation'] = 'nearest'
def main():
    parser = argparse.ArgumentParser(description="Test light propagation with optional axicon phase mask and save beam profile.")
    parser.add_argument("--config", type=str, required=False, help="Path to config.yaml")
    parser.add_argument("--mask", type=str, default="", help="Optional: path to phase mask tiff (default: zeros)")
    parser.add_argument("--output_dir", type=str, default="beam_profile_test", help="Output directory")
    #parser.add_argument("--fresnel_lens_pattern", action="store_true", help="Use Fresnel lens phase mask")
    #parser.add_argument("--bessel_angle", type=float, default=0.0, help="Bessel cone angle in degrees (default: 0, no axicon)")
    parser.add_argument("--gen_phase_mask", type=str, default="", help="Optional: 'axicon', 'fresnel_lens', or 'empty' to generate a phase mask pattern")
    args = parser.parse_args()

    # Create a timestamped subfolder output dir
    #input_timestamp_epoch = extract_datetime_and_epoch(args.mask)
    #timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_subdir = os.path.join(args.output_dir, "beam_profile")
    os.makedirs(output_subdir, exist_ok=True)
    print(f"Output directory: {output_subdir}")
    if args.mask and not args.config:
        print(f"Using provided config and mask: {args.mask}")
        mask_config_path = os.path.join(os.path.dirname(args.mask), "config.yaml")
        config = load_config(mask_config_path)
    else:
        config = load_config(args.config)
    N = config['phase_mask_pixel_size']
    slm_px = float(config["slm_px"])
    px = float(config['px'])  # pixel size in meters
    config['px'] = px  # Store pixel size in config for later use
    px_mm = px * 1e3 # px in mm
    px_um = px * 1e6 # px in um
    wavelength_nm = config['wavelength'] * 1e9 # wavelength in nm
    beam_fwhm = config['laser_beam_FWHC']
    #bessel_angle = config['bessel_half_cone_angle_degrees']
    #config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phase_mask_upsample_factor = config.get('phase_mask_upsample_factor', 1)
    config["phase_mask_file"] = args.mask
    asm = config.get('angular_spectrum_method', True)
    #initial_phase_mask = config.get('initial_phase_mask', "")

    # Get z_min, z_max, y_min, y_max, and num_z_steps from config
    z_min_mm = config.get('z_min_mm', -10.0) # Default to -10 mm if not in config
    z_max_mm = config.get('z_max_mm', 10.0)  # Default to 10 mm if not in config
    y_min_mm = config.get('y_min_mm', -1.0)  # Default to -1 mm if not in config
    y_max_mm = config.get('y_max_mm', 1.0)   # Default to 1 mm if not in config
    num_z_steps = config.get('num_z_steps', 200) # Default to 200 steps if not in config

    # Convert mm values to pixel values for internal calculations
    z_min_pixels = int(z_min_mm / px_mm)
    z_max_pixels = int(z_max_mm / px_mm)
    y_min_pixels = int(y_min_mm / px_mm)
    y_max_pixels = int(y_max_mm / px_mm)

    # Calculate z_step_pixels
    if num_z_steps > 1:
        z_step_pixels = int((z_max_pixels - z_min_pixels) / (num_z_steps - 1))
    else:
        z_step_pixels = 1 # Handle case of single step to avoid division by zero

    # Ensure z_step_pixels is at least 1
    z_step_pixels = max(1, z_step_pixels)
    #z_step_pixels = 1

    # Generate or load phase mask
    # assert that either bessel_angle > 0 or args.fresnel_lens_pattern is True or args.mask is provided
    #if int(bessel_angle > 0) + int(bool(initial_phase_mask)) + int(bool(args.mask)) > 1:
    #    raise ValueError("Must specify either a Bessel cone angle, a Fresnel lens pattern, or a phase mask file.")
    if args.mask:
        print(f"Loading phase mask from {args.mask}")
        mask_np = skimage.io.imread(args.mask).astype(np.float32)
        
    elif args.gen_phase_mask == "axicon":
        """ print(f"Generating axicon phase mask: {N}x{N}, {slm_px*1e6}um, {wavelength_nm}nm, angle={args.bessel_angle}deg")
        mask_np = generate_axicon_phase_mask(
            mask_resolution_pixels=(N, N),
            pixel_pitch_um=slm_px*1.0e6,
            wavelength_nm=wavelength_nm,
            bessel_half_cone_angle_degrees=args.bessel_angle
        ) """
        #config['bessel_half_cone_angle_degrees'] = args.bessel_angle
        config['initial_phase_mask'] = 'axicon'
        mask_np = get_initial_phase_mask(config)
        
    elif args.gen_phase_mask == "fresnel_lens":
        """ #print(config)
        focal_length = config['lensless_prop_distance'] # in meters
        print(f"Generating a Fresnel lens phase mask: {N}x{N}, {slm_px*1e6}um, {wavelength_nm}nm, focal_length={focal_length}m")
        # Generate Fresnel lens phase mask
        yy, xx = np.meshgrid(np.arange(N) - N // 2, np.arange(N) - N // 2)
        r = np.sqrt(xx**2 + yy**2) * slm_px  # radius in meters
        fresnel_phase = -(np.pi / (wavelength_nm * 1e-9 * focal_length)) * (r ** 2)
        mask_np = np.mod(fresnel_phase, 2 * np.pi).astype(np.float32) """
        config['initial_phase_mask'] = 'lens'
        config['lensless_prop_distance'] = config['fresnel_lens_focal_length']
        mask_np = get_initial_phase_mask(config)
   
    elif args.gen_phase_mask == "empty":
        #print("Using empty phase mask (zeros)")
        #mask_np = np.zeros((N, N), dtype=np.float32)
        config['initial_phase_mask'] = 'empty'
        mask_np = get_initial_phase_mask(config)
    else:
        print("No phase mask specified")
        exit()
        

    # save the mask as png figure for easy viewing
    mask_png_path = os.path.join(output_subdir, "mask.png")
    plt.figure(figsize=(8, 6))
    plt.imshow(mask_np, cmap='hot', aspect='auto')
    plt.colorbar()
    plt.title("Phase Mask")
    plt.tight_layout()
    plt.savefig(mask_png_path)
    print(f"Saved phase mask as PNG to {mask_png_path}")
    # save as tif
    mask_tif_path = os.path.join(output_subdir, "mask.tif")
    skimage.io.imsave(mask_tif_path, mask_np)
    
    unwrapped_phase_mask = unwrap_phase(mask_np - np.pi, wrap_around=False)
    unwrapped_mask_tif_path = os.path.join(output_subdir, "unwrapped_mask.tif")
    skimage.io.imsave(unwrapped_mask_tif_path, unwrapped_phase_mask)

    #mask_tensor = torch.from_numpy(mask_np).type(torch.FloatTensor).to(device)

    config['Nimgs'] = 1
    lens_approach = config['lens_approach']
    # to test 9f try running results with fourier lens
    #lens_approach = 'fourier_lens'  # temporarily force fourier lens for testing
    
    phys_layer = OpticsSimulation(config, device)
    phys_layer.eval()
    if not lens_approach == 'lazy_4f':
        if phase_mask_upsample_factor > 1:
            mask_tensor = torch.from_numpy(mask_np).type(torch.FloatTensor).to(device)
            mask_tensor = OpticsSimulation.expand_matrix_kron_torch(mask_tensor, phase_mask_upsample_factor)
        else:
            mask_tensor = torch.from_numpy(mask_np).type(torch.FloatTensor).to(device)
    else:
        mask_tensor = torch.from_numpy(mask_np).type(torch.FloatTensor).to(device)
        
    #check type of mask_tensor
    print(f"mask_tensor shape: {mask_tensor.shape}, dtype: {mask_tensor.dtype}, device: {mask_tensor.device}")    
    # Convert to radians if max value is 255
    if mask_tensor.max() == 255:
        print("Converting mask from 8-bit to radians")
        mask_tensor = (mask_tensor / 255.0) * 2 * np.pi
    with torch.no_grad():
        if lens_approach == 'against_lens':
            print("are you sure you didnt mean fourier lens?")
            output_layer = phys_layer.against_lens(mask_tensor)
        elif lens_approach == 'fourier_lens' or lens_approach == 'convolution':
            output_layer = phys_layer.fourier_lens(mask_tensor)
        elif lens_approach == 'lensless':
            output_layer = phys_layer.lensless(mask_tensor)
        elif lens_approach == '4f':
            output_layer = phys_layer.fourf(mask_tensor, phys_layer.pad_4f)
        elif lens_approach == '9f':
            output_layer = phys_layer.ninef(mask_tensor)
        elif lens_approach == 'lazy_4f':
            output_layer = phys_layer.lazy_fourf(mask_tensor)
        elif lens_approach == 'sample_4f':
            output_layer = phys_layer.sample_4f(mask_tensor)
        else:
            raise ValueError('lens approach not supported')

        # squeeze the output layer to remove singleton dimensions
        output_layer = output_layer.squeeze()

        print("Generating beam profile...")
        beam_profile, column_sums_per_image, intensities_at_z = phys_layer.generate_beam_cross_section(
            output_layer, output_subdir,
            (z_min_pixels, z_max_pixels, z_step_pixels),
            (y_min_pixels, y_max_pixels), asm = asm
        )

        # Save as TIFF (like mask_inference)
        beam_profile_tiff_path = os.path.join(output_subdir, "beam_profile.tiff")
        if torch.is_tensor(beam_profile):
            # .detach() removes it from the graph
            # .cpu() moves it from GPU to RAM
            # .numpy() converts it to the format skimage expects
            beam_profile_np = beam_profile.detach().cpu().numpy()
        else:
            beam_profile_np = beam_profile

        skimage.io.imsave(beam_profile_tiff_path, beam_profile_np)
        print(f"Saved beam profile as TIFF to {beam_profile_tiff_path}")
        
        column_sums_tiff_path = os.path.join(output_subdir, "column_sums.tiff")
        if torch.is_tensor(column_sums_per_image):
            # .detach() removes it from the graph
            # .cpu() moves it from GPU to RAM
            # .numpy() converts it to the format skimage expects
            column_sums_per_image_np = column_sums_per_image.detach().cpu().numpy()
        else:
            beam_profile_np = beam_profile

        skimage.io.imsave(column_sums_tiff_path, (column_sums_per_image_np))
        print(f"Saved beam profile as TIFF to {column_sums_tiff_path}")

        # Save as PNG for easy viewing
        png_path = os.path.join(output_subdir, "beam_profile.png")
        plt.figure(figsize=(8, 10))
        plt.imshow(normalize_to_uint16(beam_profile), cmap='hot', aspect='auto')
        plt.colorbar()

        # Set axis labels and ticks in mm
        z_range_px = range(z_min_pixels, z_max_pixels, z_step_pixels)
        z_range_mm = np.array(list(z_range_px)) * px_mm  # Convert pixel range to mm
        #z_range_mm = np.linspace(z_min_mm, z_max_mm, num_z_steps)
        #y_range_px = range(y_min_pixels, y_max_pixels, y_step_pixels)
        #y_range_mm = np.array(list(y_range_px)) * px_mm
        y_range_mm = np.linspace(y_min_mm, y_max_mm, y_max_pixels - y_min_pixels + 1) 

        plt.title(
            #f"Timestamp: {timestamp}\n"
            #f"z: {z_min_mm:.2f}mm-{z_max_mm:.2f}mm, steps={num_z_steps},\n"
            #f"axicon angle={bessel_angle}°, \n"
            #f"lens={lens_approach}, \n"
            #f"focal_length_1={1.0e3*config.get('focal_length', 'N/A')}mm, \n"
            #f"focal_length_2={1.0e3*config.get('focal_length_2', 'N/A')}mm \n"
            #f"fresnel_lens_focal_length={1.0e3*config.get('fresnel_lens_focal_length', 'N/A')}mm\n"
            #f"Mask: {initial_phase_mask}\n"
            # display px
            f"Pixel size: {px_um:.2f} um\n"
            #f"{'ASM prop' if asm else 'Fresnel prop'}, "
            f"FWHM: {beam_fwhm*10**6:.2f} um"
        )
        plt.xlabel("y (mm)")
        plt.ylabel("z (mm)")
        num_z_ticks = 20
        plt.yticks(
            ticks=np.linspace(0, len(z_range_mm)-1, num=num_z_ticks),
            labels=[f"{z_range_mm[int(i)]:.2f}" for i in np.linspace(0, len(z_range_mm)-1, num=num_z_ticks)]
        )
        # For y-axis ticks, we need to map pixel indices to mm values
        y_tick_pixels = np.linspace(0, len(y_range_mm)-1, num=5)
        y_tick_mm_labels = [f"{y_range_mm[int(j)]:.2f}" for j in y_tick_pixels]
        plt.xticks(
            ticks=y_tick_pixels,
            labels=y_tick_mm_labels
        )
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()
        print(f"Saved beam profile as PNG to {png_path}")
        
        
        # COLUMN IMAGE VISUALIZATION
        column_visual_path = os.path.join(output_subdir,'column_sums_visualization_in_beam_profiler.png')
        num_slices, num_cols = column_sums_per_image.shape

        # 1. SWAP FIGURE DIMENSIONS FOR RE-ORIENTATION
        # Since we are rotating 90 degrees, width and height logic swaps
        """ fig_height = 12
        fig_width = fig_height * (num_slices * z_step_pixels / num_cols)

        plt.figure(figsize=(fig_width, fig_height))
        plt.rcParams.update({'font.size': 30}) """
        
        # Let Matplotlib handle bounding dimensions. We set a solid base height.
        fig_height = 12
        fig_width = 12
        
        plt.figure(figsize=(fig_width, fig_height))
        plt.rcParams.update({'font.size': 20}) 
        
        # 2. ROTATE THE DATA
        # np.flipud() safely flips the rows vertically, and .T transposes it.
        # Together with origin='lower', this achieves a clean 90-degree clockwise rotation.
        rotated_data = np.flipud(column_sums_per_image).T
        
        adj_aspect = 1 / z_step_pixels # Aspect ratio inverts because axes swapped
        plt.imshow(rotated_data, aspect=adj_aspect, origin='lower', cmap='viridis', interpolation='nearest')
        plt.xlabel('z (mm)')
        plt.ylabel('y (mm)')
        #plt.colorbar(label='')
        
        #save raw column sums visual as a tiff
        column_sums_visual_tiff_path = os.path.join(output_subdir, "column_sums_visual.tiff")
        skimage.io.imsave(column_sums_visual_tiff_path, normalize_to_uint16(column_sums_per_image.T))
        
      # Set X-ticks (formerly Z-ticks)
        num_z_ticks = 11
        # =====================================================================
        # FIXED SYMMETRIC 1-DECIMAL TICK GENERATION
        # =====================================================================
        # 1. Generate clean, mathematically symmetric values for Z (X-axis)
        z_tick_vals = np.linspace(z_min_mm, z_max_mm, num=11)
        
        # Map physical values into coordinate slice-index space: (val - min) / step
        # Since rotated_data is sized by num_slices along the matrix columns
        real_z_min = z_min_pixels * px_mm
        z_tick_indices = (z_tick_vals - real_z_min) / (z_step_pixels * px_mm)
        
        plt.xticks(
            ticks=z_tick_indices,
            labels=[f"{val:.1f}" for val in z_tick_vals]
        )
        
        # 2. Generate clean, mathematically symmetric values for Y (Y-axis)
        y_tick_vals = np.linspace(y_min_mm, y_max_mm, num=5)
        
        # Map physical values into row-index space: (val - min) / step
        # Matrix Y rows are packed 1-to-1 pixel wise with original resolution
        real_y_min = y_min_pixels * px_mm
        y_tick_pixels = (y_tick_vals - real_y_min) / px_mm
        
        plt.yticks(
            ticks=y_tick_pixels,
            labels=[f"{val:.2f}" for val in y_tick_vals]
        )
        # 4. PREVENT CLIPPING
        # bbox_inches='tight' tells matplotlib to dynamically calculate the bounding box 
        # including all large 30pt fonts so nothing gets cut off.
        plt.tight_layout()
        plt.savefig(column_visual_path, bbox_inches='tight', dpi=600) 
        plt.close()

        # save the config used for this test
        config_output_path = os.path.join(output_subdir, "config.yaml")
        with open(config_output_path, "w") as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
        print(f"Saved configuration to {config_output_path}")

if __name__ == "__main__":
    main()