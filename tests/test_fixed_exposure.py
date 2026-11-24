import numpy as np
import pytest
import torch
from physics.simulation import NoiseLayer,OpticsSimulation
from physics.exposure import exposure_settings


def config(**extra):
    return dict(quantum_efficiency=1.,dark_current_mean=0.,read_noise_std=0.,camera_gain=1.,camera_max_adu=1e9,robust_max_percentile=.9,target_saturation_ratio=.01,epsilon=1e-6,min_clip_value=0.,**extra)


def test_fixed_scale_is_independent_of_scene_brightness_and_batch():
    layer=NoiseLayer(config(camera_exposure_mode='fixed',fixed_exposure_scale=3.))
    image=torch.stack([torch.ones(1,3,4),torch.full((1,3,4),17.)])
    torch.testing.assert_close(layer.scale_for(image),torch.full((2,1,1,1),3.))
    # Expected signal and its gradient remain proportional to dose.
    image.requires_grad_();signal=image*layer.scale_for(image);signal.sum().backward()
    torch.testing.assert_close(image.grad,torch.full_like(image,3.))


def test_default_auto_exposure_retains_formula_and_random_sequence():
    cfg=config();layer=NoiseLayer(cfg);image=torch.arange(1.,25.).reshape(2,1,3,4)
    robust=torch.quantile(image.view(2,-1),cfg['robust_max_percentile'],dim=1)
    scale=(torch.tensor(cfg['camera_max_adu'])/torch.tensor(cfg['camera_gain'])*cfg['target_saturation_ratio']/(robust+cfg['epsilon'])).view(-1,1,1,1)
    torch.testing.assert_close(layer.scale_for(image),scale,rtol=0,atol=0)
    mean=image*scale;torch.manual_seed(4);expected=mean+torch.sqrt(mean+cfg['epsilon'])*torch.randn_like(mean);expected+=cfg['read_noise_std']*torch.randn_like(mean)
    torch.manual_seed(4);torch.testing.assert_close(layer(image),expected,rtol=0,atol=0)


def test_fixed_requires_explicit_positive_scale():
    for value in [None,0,-1,float('inf'),float('nan'),True,'1']:
        with pytest.raises(ValueError):exposure_settings(dict(camera_exposure_mode='fixed',fixed_exposure_scale=value))
    with pytest.raises(ValueError):exposure_settings(dict(camera_exposure_mode='unknown'))


def test_fixed_expected_camera_preserves_absolute_signal_ratio():
    from scripts.optical_report import expected_camera
    cfg=config(camera_exposure_mode='fixed',fixed_exposure_scale=2.)
    a,_,sa=expected_camera(np.ones((4,4)),cfg);b,_,sb=expected_camera(np.ones((4,4))*3,cfg)
    np.testing.assert_array_equal(b,a*3);assert sa==sb==2.


def test_asm_audit_preserves_output_gradient_and_reports_crop_loss():
    sim=OpticsSimulation.__new__(OpticsSimulation);torch.nn.Module.__init__(sim)
    sim.N=8;sim.px=1e-6;sim.wavelength=561e-9;sim.refractive_index=1.;sim.k=2*np.pi/561e-9;sim.device='cpu'
    sim.gamma_padded=sim._generate_gamma((16,16));sim.gamma_unpadded=sim._generate_gamma((8,8));sim.check_sampling_limit=lambda *a,**kw:None
    phase=torch.randn(1,1,8,8,requires_grad=True);field=torch.exp(1j*phase)
    ordinary=sim.angular_spectrum_propagation(field,100.)
    audited,audit=sim.angular_spectrum_propagation(field,100.,return_power_audit=True)
    torch.testing.assert_close(ordinary,audited,rtol=0,atol=0)
    a=torch.autograd.grad(ordinary.abs().square().sum(),phase,retain_graph=True)[0]
    b=torch.autograd.grad(audited.abs().square().sum(),phase)[0]
    torch.testing.assert_close(a,b,rtol=0,atol=0);assert a.abs().max()>0 and torch.isfinite(a).all()
    assert abs(float(audit['full_output_power']/audit['input_power'])-1)<1e-5
    assert audit['cropped_output_power']<audit['full_output_power']
    assert abs(float(audit['spectral_output_power']/audit['full_output_power'])-1)<1e-5


def test_scene_bootstrap_detects_perfect_and_missed_detection():
    from scripts.evaluate_fixed_dose import bootstrap_f1
    assert bootstrap_f1([[2,0,0],[3,0,0]])==[1.,1.]
    assert bootstrap_f1([[0,0,2],[0,0,3]])==[0.,0.]
