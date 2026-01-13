from .psf_gen_old import generate_psf, generate_bead_defocus_stack

def generate_bead_templates(config, results_dir):
    """
    Master function to ensure PSF templates exist.
    """
    # 1. Define paths
    
    psf_stack = generate_psf(config, results_dir)
    generate_bead_defocus_stack(config, results_dir, psf_stack)
