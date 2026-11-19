import numpy as np
from scripts.airy_orientation_debug import phase_parameters


def test_rotated_cubic_changes_dither_geometry_without_changing_quadratic():
    size=8;pitch=1e-6;k=2*np.pi/561e-9;f=.00375;alpha=1e12
    phase,step=phase_parameters(size,pitch,k,f,alpha,0,pitch)
    y,z=np.meshgrid((np.arange(size)-size//2)*pitch,(np.arange(size)-size//2)*pitch,indexing='ij')
    scale=f*(3*alpha)**(1/3)/k;shift=1.018792971647471*scale-pitch
    expected=-k*(y*y+z*z)/(2*f)-alpha*(y**3+z**3)+k*shift*(y+z)/f
    np.testing.assert_allclose(phase,expected,rtol=2e-6,atol=1e-7)
    rotated,_=phase_parameters(size,pitch,k,f,alpha,45,pitch)
    assert not np.allclose(phase,rotated)


def test_alias_guard_rejects_large_cubic_coefficient_at_existing_grid():
    _,step=phase_parameters(1024,1e-6,2*np.pi/561e-9,.00375,1e14,0,1e-6)
    assert step>np.pi


def test_45_degree_cubic_contains_the_expected_scan_thickness_cross_term():
    size=16;pitch=1e-6;k=2*np.pi/561e-9;f=.00375;alpha=1.1e12
    phase,_=phase_parameters(size,pitch,k,f,alpha,45,pitch)
    y,z=np.meshgrid((np.arange(size)-size//2)*pitch,(np.arange(size)-size//2)*pitch,indexing='ij')
    scale=f*(3*alpha)**(1/3)/k
    dy=-pitch;dz=1.018792971647471*scale*np.sqrt(2)-pitch
    expected=-k*(y*y+z*z)/(2*f)-alpha/np.sqrt(2)*(z**3+3*y*y*z)+k*(dy*y+dz*z)/f
    np.testing.assert_allclose(phase,expected,rtol=2e-6,atol=1e-7)


def test_focused_airy_factory_reproduces_the_tested_phase_recipe():
    from physics.masks import get_initial_phase_mask
    cfg=dict(initial_phase_mask='airy',phase_mask_pixel_size=16,slm_px=1e-6,px=1e-6,
        wavelength=561e-9,refractive_index=1.,lens_approach='sample_4f',
        fresnel_lens_focal_length=.00375,lensless_prop_distance=.00375,
        airy_phase_mode='focused',airy_alpha=1.1e12,airy_rotation_degrees=45)
    phase=get_initial_phase_mask(cfg)
    expected,_=phase_parameters(16,1e-6,2*np.pi/561e-9,.00375,1.1e12,45,1e-6)
    np.testing.assert_array_equal(phase,expected)


def test_legacy_airy_factory_retains_cubic_only_default():
    from physics.masks import get_initial_phase_mask
    cfg=dict(initial_phase_mask='airy',phase_mask_pixel_size=8,slm_px=1e-6,wavelength=561e-9,airy_alpha=1e10)
    coordinate=(np.arange(8)-4)*1e-6;x,y=np.meshgrid(coordinate,coordinate)
    expected=(2*np.pi-np.mod(1e10*(x**3+y**3),2*np.pi)).astype(np.float32)
    np.testing.assert_array_equal(get_initial_phase_mask(cfg),expected)


def test_focused_airy_rejects_an_unmatched_focus_distance():
    import pytest
    from physics.masks import get_initial_phase_mask
    cfg=dict(initial_phase_mask='airy',phase_mask_pixel_size=8,slm_px=1e-6,wavelength=561e-9,
        lens_approach='sample_4f',airy_phase_mode='focused',airy_alpha=1e12,
        fresnel_lens_focal_length=.00375,lensless_prop_distance=.004)
    with pytest.raises(ValueError,match='focal length must match'):get_initial_phase_mask(cfg)
