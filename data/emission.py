"""Optional relative per-emitter brightness, persisted with scene coordinates."""
import numpy as np


def emission_range(config):
    bounds = np.asarray(config.get('emission_strength_range', [1., 1.]), dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] < 0 or bounds[1] < bounds[0] or bounds[1] <= 0:
        raise ValueError('emission_strength_range requires finite nonnegative [min, max], max > 0')
    return bounds


def attach_emission_strength(xyz, config):
    """E3 legacy scenes remain unchanged (including RNG); E4 adds relative photons."""
    lo, hi = emission_range(config)
    if lo == hi == 1:
        return xyz
    strength = np.full((len(xyz), 1), lo) if lo == hi else np.random.uniform(lo, hi, (len(xyz), 1))
    return np.concatenate([xyz, strength], axis=1)
