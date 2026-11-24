"""Explicit camera exposure policies; fixed scales never depend on a test image."""
import math
import numbers


def exposure_settings(config):
    mode=config.get('camera_exposure_mode','auto_percentile')
    if mode not in ('auto_percentile','fixed'):
        raise ValueError('camera_exposure_mode must be auto_percentile or fixed')
    scale=None
    if mode=='fixed':
        value=config.get('fixed_exposure_scale')
        if isinstance(value,bool) or not isinstance(value,numbers.Real) or not math.isfinite(float(value)) or value<=0:
            raise ValueError('fixed exposure requires a finite positive fixed_exposure_scale')
        scale=float(value)
    return mode,scale
