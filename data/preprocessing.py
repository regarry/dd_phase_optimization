import time
from .psf_gen import generate_psf, generate_bead_defocus_stack

def generate_bead_templates(config, results_dir):
    """
    Master function to ensure PSF templates exist.
    """
    # 1. Define paths
    start_time = time.time()
    print("Generating PSF stack...")
    psf_stack = generate_psf(config, results_dir)
    generate_bead_defocus_stack(config, results_dir, psf_stack)
    elapsed = time.time() - start_time
    print(f"Bead template generation completed in {elapsed:.2f} seconds.")