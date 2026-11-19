import numpy as np
import pytest
import torch
from data.emission import attach_emission_strength
from physics.simulation import OpticsSimulation, interpolate_defocus_psf
from training.segments import terminal_reason


def test_brightness_default_preserves_coordinates_and_rng():
    xyz=np.arange(12).reshape(4,3)
    np.random.seed(7);expected=np.random.rand(3)
    np.random.seed(7);assert attach_emission_strength(xyz,{}) is xyz
    np.testing.assert_array_equal(np.random.rand(3),expected)


def test_brightness_uniform_is_persistable_and_validated():
    np.random.seed(7)
    out=attach_emission_strength(np.zeros((1200,3)),{'emission_strength_range':[.5,1.5]})
    assert out.shape==(1200,4)
    assert .5<=out[:,3].min()<out[:,3].max()<=1.5
    assert abs(out[:,3].mean()-1)<.03
    for bounds in [[-1,1],[2,1],[0,0],[1,float('nan')],[1]]:
        with pytest.raises(ValueError):attach_emission_strength(np.zeros((1,3)),{'emission_strength_range':bounds})


def test_multiview_shared_illumination_brightness_fractional_psfs_and_gradient():
    # Tiny renderer fixture exercises the actual forward emitter/view loops.
    sim=OpticsSimulation.__new__(OpticsSimulation);torch.nn.Module.__init__(sim)
    sim.lens_approach='sample_4f';sim.phase_mask_upsample_factor=1
    sim.conv3d=False;sim.Nimgs=3;sim.image_volume_size_px=[9,9,8]
    sim.N=8;sim.psf_keep_radius=1;sim.z_depth_list=[-1.5,0,1.5]
    sim.illumination_scaling_factor=torch.tensor(1.);sim.counter=0
    sim.register_buffer('gpu_defocused_beads',torch.stack([torch.full((3,3),float(i+1)) for i in range(6)]))
    calls=[]
    sim.sample_4f=lambda phase: (1+phase).to(torch.complex64)[None,None].expand(1,1,8,8)
    def propagate(field,*args,**kwargs):calls.append(1);return field
    sim.angular_spectrum_propagation=propagate
    phase=torch.nn.Parameter(torch.ones((8,8)))
    xyz=torch.tensor([[[4.,4.,0.,.75]]])
    image=sim(phase,xyz)
    assert len(calls)==1
    torch.testing.assert_close(image[0,0],image[0,2])
    torch.testing.assert_close(image[0,0],image[0,1]*2.5)
    image.sum().backward()
    assert torch.isfinite(phase.grad).all() and phase.grad.norm()>0
    sim.Nimgs=1;sim.z_depth_list=[0]
    legacy=sim(phase,xyz[:,:,:3]);unit=sim(phase,torch.cat([xyz[:,:,:3],torch.ones((1,1,1))],-1))
    torch.testing.assert_close(legacy,unit,rtol=0,atol=0)
    torch.testing.assert_close(sim(phase,xyz),legacy*.75)


def test_psf_interpolation_integer_exact_and_bounds_fail_closed():
    stack=torch.arange(12.).reshape(4,3)
    torch.testing.assert_close(interpolate_defocus_psf(stack,2),stack[2],rtol=0,atol=0)
    torch.testing.assert_close(interpolate_defocus_psf(stack,1.5),(stack[1]+stack[2])/2)
    for d in [-1,3.1,float('nan')]:
        with pytest.raises(ValueError):interpolate_defocus_psf(stack,d)


def test_actual_training_budget_is_terminal_without_early_stopping():
    state=dict(raw_config=dict(max_epochs=100,minimum_training_seconds=10,maximum_training_seconds=10),
               progress=dict(train_seconds=10.2),early_stopping=dict(early_stop=False),epoch=2)
    assert terminal_reason(state)=='training_budget_reached'


def test_float_phase_transfer_roundtrip_preserves_unwrapped_radians(tmp_path):
    from physics.phase_io import save_radians,load_phase
    phase=np.array([[-9,0],[3,12]],dtype=np.float32)
    path=tmp_path/'phase.tif';save_radians(phase,path)
    np.testing.assert_array_equal(load_phase(path),phase)


def test_inconsistent_training_time_bounds_are_rejected():
    state=dict(raw_config=dict(max_epochs=100,minimum_training_seconds=10,maximum_training_seconds=9),
               progress=dict(train_seconds=9.2),early_stopping=dict(early_stop=False),epoch=2)
    with pytest.raises(ValueError,match='below'):terminal_reason(state)
