import numpy as np

from physics.masks import get_initial_phase_mask


def test_gaussian_keeps_incident_amplitude_without_phase_modulation():
    config = dict(initial_phase_mask='gaussian', phase_mask_pixel_size=16,
                  slm_px=1e-6, wavelength=561e-9)
    phase = get_initial_phase_mask(config)
    coords = np.arange(16) - 8
    x, y = np.meshgrid(coords, coords)
    incident = np.exp(-(x*x + y*y) / (2*3.0**2))
    assert phase.shape == (16, 16)
    assert phase.dtype == np.float32
    np.testing.assert_allclose(incident * np.exp(1j * phase), incident, rtol=0, atol=0)
