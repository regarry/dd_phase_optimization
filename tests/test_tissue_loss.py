import copy
import random

import numpy as np
import pytest
import torch

from data.generators import create_random_emitters
from training.tissue_loss import TissueAwareProfileLoss,tissue_profiles,validate_tissue_config


def settings(**changes):
    value=dict(tissue_spatial_loss_weight=1,tissue_profile_x_um=[-2,0,2],
        tissue_profile_z_bounds_um=[-6,6],tissue_z_bounds_um=[-4,4],
        tissue_core_halfwidth_um=1,tissue_core_min_fraction=.8,
        tissue_core_floor_weight=10,tissue_core_uniformity_weight=0,
        spatial_multi_well_loss_weight=0,lens_approach='sample_4f',image_volume=[8,8,12])
    value.update(changes);return value


def pack(intensity,reference=None):
    intensity=torch.as_tensor(intensity,dtype=torch.float32)
    return dict(intensity=intensity,reference_intensity=torch.ones_like(intensity) if reference is None else reference,
                x_um=torch.arange(intensity.shape[0]),z_um=torch.arange(-6,7,dtype=torch.float32))


def test_remote_lobe_does_not_dilute_tissue_penalty():
    loss=TissueAwareProfileLoss(settings())
    p=pack(torch.ones(3,13));baseline=loss(p)
    bright=copy.copy(p);bright['intensity']=p['intensity'].clone()
    bright['intensity'][:,[0,1,11,12]]=1e6
    torch.testing.assert_close(loss(bright),baseline)


def test_loss_uses_thickness_columns_not_propagation_rows():
    loss=TissueAwareProfileLoss(settings())
    core=pack(torch.zeros(3,13));core['intensity'][:,5:8]=1
    off=pack(core['intensity'].clone());off['intensity'][:,3]=2
    assert loss(off)>loss(core)
    # Permuting propagation positions cannot turn thickness into propagation.
    reordered=copy.copy(off);reordered['intensity']=off['intensity'][[2,0,1]]
    torch.testing.assert_close(loss(reordered),loss(off))


def test_turning_beam_off_has_throughput_penalty_and_gradient():
    loss=TissueAwareProfileLoss(settings())
    intensity=torch.zeros(3,13,requires_grad=True);p=pack(intensity)
    value=loss(p);value.backward()
    assert value>0 and torch.isfinite(intensity.grad).all()
    assert torch.all(intensity.grad[:,5:8]<0)  # gradient descent restores core


def test_one_sided_tissue_does_not_penalize_clear_side():
    loss=TissueAwareProfileLoss(settings(tissue_z_bounds_um=[0,4]))
    base=pack(torch.ones(3,13));value=loss(base)
    changed=copy.copy(base);changed['intensity']=base['intensity'].clone()
    changed['intensity'][:,:6]=100
    torch.testing.assert_close(loss(changed),value)


def test_required_tissue_geometry_fails_clearly():
    cfg=settings();del cfg['tissue_z_bounds_um']
    with pytest.raises(ValueError,match='tissue_z_bounds_um'):validate_tissue_config(cfg)
    with pytest.raises(ValueError,match='cover the complete'):validate_tissue_config(settings(tissue_profile_z_bounds_um=[-2,2]))


class TinyPropagation:
    """Differentiable toy propagation, never a physical-model correctness claim."""
    px=1e-6
    N=16

    def sample_4f(self,phase):
        # Position-varying amplitude makes phase-interference gradients nontrivial.
        amplitude=torch.linspace(.4,1,16,device=phase.device)[None,:]
        return (amplitude*torch.exp(1j*phase))[None,None]

    def angular_spectrum_propagation(self,field,x,pad):
        return torch.fft.ifft2(torch.fft.fft2(field)*torch.exp(
            1j*torch.arange(16,device=field.device)[None,:]*(x+.3)/5))


def test_streamed_checkpointed_profile_preserves_raw_phase_gradient():
    torch.manual_seed(19);phase=torch.randn(16,16,requires_grad=True)
    cfg=settings();sim=TinyPropagation();p=tissue_profiles(sim,phase,cfg)
    assert p['intensity'].shape==(3,13)
    value=TissueAwareProfileLoss(cfg)(p);value.backward()
    assert phase.grad is not None and torch.isfinite(phase.grad).all() and phase.grad.abs().max()>0
    # A straightforward dense implementation must agree in output and gradient.
    other=phase.detach().clone().requires_grad_();field=sim.sample_4f(other)
    z=(torch.arange(16)-(16//2-1)).float();selected=(z>=-6)&(z<=6)
    dense=torch.stack([sim.angular_spectrum_propagation(field,x,True).abs().square().sum(-2).squeeze()[selected]
                       for x in [-2,0,2]])
    q=dict(p,intensity=dense);TissueAwareProfileLoss(cfg)(q).backward()
    torch.testing.assert_close(p['intensity'],dense)
    torch.testing.assert_close(phase.grad,other.grad)


def test_one_sided_sample_limits_all_beads_and_pair_connections():
    cfg=dict(num_particles_range=[100,101],particle_spatial_range_x=[0,10],
        particle_spatial_range_y=[0,10],particle_spatial_range_z=[-20,21],
        z_coupled_ratio=1,z_coupled_spacing_range=[1,3],px=1e-6,
        tissue_sample_z_bounds_um=[0,10])
    random.seed(7);np.random.seed(7)
    beads,links=create_random_emitters(cfg)
    assert len(beads)==100 and np.all((beads[:,2]>=0)&(beads[:,2]<=10))
    assert np.all((links[:,2]>=0)&(links[:,2]<=10))
    cfg['tissue_sample_z_bounds_um']=[100,200]
    with pytest.raises(ValueError,match='no allowed'):create_random_emitters(cfg)
