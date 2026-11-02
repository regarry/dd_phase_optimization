"""Phase arrays are radians; hardware images are integer codes plus JSON metadata."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

TAU = 2 * np.pi

def levels(bits):
    if isinstance(bits, bool) or not isinstance(bits, int) or not 1 <= bits <= 16:
        raise ValueError('slm_bit_depth must be an integer between 1 and 16')
    return (1 << bits) - 1

def quantize_phase(phase, bits=8):
    """Straight-through estimator: quantized forward, identity backward modulo 2pi."""
    import torch
    maximum = levels(bits)
    wrapped = torch.remainder(phase, TAU)
    quantized = torch.round(wrapped / TAU * maximum) / maximum * TAU
    return wrapped + (quantized - wrapped).detach()

def phase_to_codes(phase, bits=8):
    phase = np.asarray(phase)
    if phase.ndim != 2 or not np.isfinite(phase).all():
        raise ValueError('Expected a finite 2D phase array in radians')
    return np.rint(np.remainder(phase, TAU) / TAU * levels(bits)).astype(
        np.uint8 if bits <= 8 else np.uint16)

def load_phase(path, input_units=None, bits=None):
    """Use sidecar metadata, otherwise require units for legacy integer images."""
    path = Path(path)
    array = np.array(Image.open(path))
    sidecar = Path(str(path) + '.json')
    if sidecar.exists():
        meta = json.loads(sidecar.read_text())
        if input_units is not None and input_units != meta['units']:
            raise ValueError('Requested phase units conflict with sidecar metadata')
        input_units = meta['units']
        if bits is not None and bits != meta.get('bit_depth', bits):
            raise ValueError('Requested bit depth conflicts with sidecar metadata')
        bits = meta.get('bit_depth', bits)
    if input_units is None:
        if np.issubdtype(array.dtype, np.floating):
            input_units = 'radians'
        else:
            raise ValueError('Legacy integer mask needs explicit input_units=drive_codes or radians')
    if input_units == 'drive_codes':
        if bits is None:
            raise ValueError('Legacy drive codes need an explicit bit depth')
        maximum = levels(bits)
        if np.any(array < 0) or np.any(array > maximum) or np.any(array != np.rint(array)):
            raise ValueError('Drive codes outside the configured integer range')
        array = array.astype(np.float64) / maximum * TAU
    elif input_units != 'radians':
        raise ValueError('input_units must be radians or drive_codes')
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError('Expected a finite 2D phase mask')
    return array.astype(np.float32)

def load_phase_config(path, config):
    # Metadata is authoritative. YAML options are only for legacy files.
    if Path(str(path) + '.json').exists():
        return load_phase(path)
    return load_phase(path, config.get('phase_mask_input_units'), config.get('slm_bit_depth', 8))

def save_codes(phase, path, bits=8):
    path = Path(path)
    if path.suffix.lower() not in {'.bmp', '.tif', '.tiff', '.png'}:
        raise ValueError('Use BMP, TIFF or PNG for hardware codes')
    if bits > 8 and path.suffix.lower() == '.bmp':
        raise ValueError('More than 8 bits requires TIFF or PNG, not grayscale BMP')
    codes = phase_to_codes(phase, bits)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(codes).save(path)
    Path(str(path) + '.json').write_text(json.dumps({
        'units': 'drive_codes', 'bit_depth': bits, 'phase_period_rad': TAU,
        'mapping': 'linear_0_to_2pi', 'note': 'Apply a measured hardware LUT in the device driver if required.'
    }, indent=2))
    return codes

def resize_phase(phase, size):
    """Interpolate the complex phasor, avoiding artificial pi steps at phase wraps."""
    field = np.exp(1j * np.asarray(phase))
    real = np.asarray(Image.fromarray(field.real.astype(np.float32)).resize(size, Image.Resampling.BILINEAR))
    imag = np.asarray(Image.fromarray(field.imag.astype(np.float32)).resize(size, Image.Resampling.BILINEAR))
    if np.any(np.hypot(real, imag) < 1e-6):
        raise ValueError('Phase resize is ambiguous near cancelling phasors; use padding or a finer grid')
    return np.remainder(np.arctan2(imag, real), TAU)
