"""Array and file entry points sharing one phase export implementation."""
import argparse
import math
import numpy as np
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physics.phase_io import load_phase, resize_phase, save_codes, TAU

def generate_blazed_grating(width, height, period_x=0, period_y=0, angle_rad=0):
    y, x = np.indices((height, width))
    xr = x * np.cos(angle_rad) + y * np.sin(angle_rad)
    yr = -x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return TAU * ((xr / period_x if period_x > 0 else 0) +
                  (yr / period_y if period_y > 0 else 0)) + np.zeros((height, width))

def process_phase_mask(input_phase_array, output_bmp_path, slm_width=None, slm_height=None,
                       padding_mode=False, add_grating=False, grating_period_x=0,
                       grating_period_y=0, grating_angle_deg=0, bit_depth=8):
    phase = np.asarray(input_phase_array, dtype=np.float64)
    if phase.ndim != 2 or not np.isfinite(phase).all():
        raise ValueError('Expected finite 2D radians')
    h, w = phase.shape
    width, height = slm_width or w, slm_height or h
    if (width, height) != (w, h):
        if padding_mode:
            if width < w or height < h:
                raise ValueError('Cannot pad to smaller dimensions')
            padded = np.zeros((height, width))
            y, x = (height-h)//2, (width-w)//2
            padded[y:y+h, x:x+w] = phase
            phase = padded
        else:
            phase = resize_phase(phase, (width, height))
    # Carrier periods refer to final hardware pixels, independent of resampling.
    if add_grating:
        phase = phase + generate_blazed_grating(width, height, grating_period_x,
                                               grating_period_y, math.radians(grating_angle_deg))
    return save_codes(phase, output_bmp_path, bit_depth)

def normalize_phase_image(input_tiff_path, output_bmp_path, slm_width=None, slm_height=None,
                          padding_mode=False, add_grating=False, grating_period_x=0,
                          grating_period_y=0, grating_angle_deg=0, bit_depth=8,
                          input_units=None, input_bit_depth=None):
    phase = load_phase(input_tiff_path, input_units, input_bit_depth)
    return process_phase_mask(phase, output_bmp_path, slm_width, slm_height, padding_mode,
                              add_grating, grating_period_x, grating_period_y,
                              grating_angle_deg, bit_depth)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--bit-depth', type=int, default=8)
    parser.add_argument('--input-units', choices=['radians', 'drive_codes'])
    parser.add_argument('--input-bit-depth', type=int)
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--pad', action='store_true')
    args = parser.parse_args()
    normalize_phase_image(args.input, args.output, args.width, args.height, args.pad,
                          bit_depth=args.bit_depth, input_units=args.input_units,
                          input_bit_depth=args.input_bit_depth)
