"""Opt-in tissue-aware illumination penalty on raw [propagation, thickness].

Axes follow the renderer: x is illumination propagation, y is scan (summed),
z is detection depth/sheet thickness. All public geometry settings are µm.
No absolute photon or calibrated fixed-dose interpretation is implied.
"""
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def _positive(config,key,allow_zero=False):
    if key not in config:
        raise ValueError('Missing required tissue setting: '+key)
    value=float(config[key])
    if not math.isfinite(value) or (value<0 if allow_zero else value<=0):
        raise ValueError(key+' must be finite and '+('nonnegative' if allow_zero else 'positive'))
    return value


def _bounds(config,key):
    value=config.get(key)
    if not isinstance(value,(list,tuple)) or len(value)!=2:
        raise ValueError(key+' must contain [lower, upper] in µm')
    lo,hi=map(float,value)
    if not math.isfinite(lo+hi) or lo>=hi:
        raise ValueError(key+' must be finite and increasing')
    return lo,hi


def validate_tissue_config(config):
    """Validate only the new explicit objective; legacy configs are untouched."""
    if not config.get('tissue_spatial_loss_weight',0):
        return
    _positive(config,'tissue_spatial_loss_weight')
    xs=config.get('tissue_profile_x_um')
    if not isinstance(xs,(list,tuple)) or not xs or not all(math.isfinite(float(x)) for x in xs):
        raise ValueError('tissue_profile_x_um must explicitly list finite propagation coordinates in µm')
    tlo,thi=_bounds(config,'tissue_z_bounds_um')
    plo,phi=_bounds(config,'tissue_profile_z_bounds_um')
    if plo>tlo or phi<thi:
        raise ValueError('tissue_profile_z_bounds_um must cover the complete fluorescent tissue interval')
    half=_positive(config,'tissue_core_halfwidth_um')
    if max(tlo,-half)>=min(thi,half):
        raise ValueError('The in-focus core must intersect the tissue interval')
    fraction=_positive(config,'tissue_core_min_fraction')
    if fraction>1:
        raise ValueError('tissue_core_min_fraction must be at most 1 relative to the Gaussian reference')
    _positive(config,'tissue_core_floor_weight')
    uniform=float(config.get('tissue_core_uniformity_weight',0))
    if not math.isfinite(uniform) or uniform<0:
        raise ValueError('tissue_core_uniformity_weight must be finite and nonnegative')
    if config.get('spatial_multi_well_loss_weight',0):
        raise ValueError('Do not combine legacy spatial_multi_well_loss with tissue_spatial_loss')


class TissueAwareProfileLoss(nn.Module):
    """Suppress fluorescent off-core excitation without rewarding remote energy.

Off-core tissue power is divided by a *fixed*, zero-phase Gaussian-reference
core power at each x. Thus boosting a distant lobe cannot dilute the penalty.
A separate explicit floor penalizes lost useful core throughput. Outside-tissue
power has zero direct reward/cost. Optional uniformity discourages x roll-off.
The finite floor weight is a soft constraint, not a guarantee of equal dose.
"""
    def __init__(self,config):
        super().__init__()
        validate_tissue_config(config)
        self.tissue=_bounds(config,'tissue_z_bounds_um')
        self.halfwidth=_positive(config,'tissue_core_halfwidth_um')
        self.minimum=_positive(config,'tissue_core_min_fraction')
        self.floor_weight=_positive(config,'tissue_core_floor_weight')
        self.uniformity_weight=float(config.get('tissue_core_uniformity_weight',0))
        self.last_terms=None

    def forward(self,profile):
        if not isinstance(profile,dict):
            raise ValueError('Tissue objective requires raw profile metadata, not a normalized legacy image')
        intensity=profile['intensity'];reference=profile['reference_intensity'].detach()
        z=profile['z_um'].to(intensity.device)
        if intensity.ndim!=2 or reference.shape!=intensity.shape or z.numel()!=intensity.shape[1]:
            raise ValueError('Tissue profiles must be [propagation x, thickness z] with matched reference/axis')
        if not torch.isfinite(intensity).all() or not torch.isfinite(reference).all():
            raise ValueError('Tissue profiles must be finite')
        tissue=(z>=self.tissue[0])&(z<=self.tissue[1])
        core=tissue&(z.abs()<=self.halfwidth)
        off=tissue&~core
        if not core.any() or not off.any():
            raise ValueError('Profile sampling must contain both in-focus core and off-core tissue pixels')
        ref_core=reference[:,core].sum(-1)
        if (ref_core<=0).any():
            raise ValueError('Gaussian reference must have positive core throughput at every sampled x')
        core_power=intensity[:,core].sum(-1)
        outside=intensity[:,off].sum(-1)
        scale=ref_core.clamp_min(torch.finfo(ref_core.dtype).tiny)
        background=(outside/scale).mean()
        relative_core=core_power/scale
        floor=torch.relu(self.minimum-relative_core).square().mean()
        # A global fixed reference scale makes this actual x-uniformity, rather
        # than matching the Gaussian's potentially varying per-x throughput.
        across_x=core_power/ref_core.mean()
        uniformity=(across_x-across_x.mean()).square().mean()
        loss=background+self.floor_weight*floor+self.uniformity_weight*uniformity
        self.last_terms={k:float(v.detach()) for k,v in dict(
            off_core_tissue=background,core_floor=floor,core_uniformity=uniformity,
            mean_relative_core=relative_core.mean(),min_relative_core=relative_core.min()).items()}
        return loss


def tissue_profiles(sim,phase,config,reference_cache=None):
    """Stream/checkpoint FFT profiles instead of retaining a [P,N,N] volume.

Uses sample_4f and pad=True exactly as emitter rendering. The zero-phase
reference is deterministic and gradient-free; caching it cannot detach the
learned path. Per-x checkpointing recomputes propagation during backward.
"""
    validate_tissue_config(config)
    if config['lens_approach']!='sample_4f':
        raise ValueError('The reviewed tissue profile route currently requires sample_4f')
    if config.get('phase_mask_upsample_factor',1)>1:
        phase=sim.expand_matrix_kron_torch(phase,config['phase_mask_upsample_factor'])
    px_um=float(sim.px)*1e6
    requested=torch.as_tensor(config['tissue_profile_x_um'],device=phase.device,dtype=phase.dtype)
    x_pixels=torch.round(requested/px_um).to(torch.int64)
    if torch.unique(x_pixels).numel()!=x_pixels.numel():
        raise ValueError('tissue_profile_x_um positions collapse to duplicate sample pixels')
    width=int(config['image_volume'][1])
    if (x_pixels < -width//2).any() or (x_pixels > width//2-1).any():
        raise ValueError('Tissue profile x positions must stay inside the rendered camera field')
    n=int(sim.N)
    z=(torch.arange(n,device=phase.device,dtype=phase.dtype)-(n//2-1))*px_um
    lo,hi=_bounds(config,'tissue_profile_z_bounds_um')
    if lo<float(z[0]) or hi>float(z[-1]):
        raise ValueError('Tissue profile z bounds exceed the sampled optical field')
    selected=(z>=lo)&(z<=hi)

    def from_field(field,use_checkpoint):
        rows=[]
        for x in x_pixels.tolist():
            def propagate(f,x=x):
                propagated=sim.angular_spectrum_propagation(f,x,pad=True)
                return propagated.abs().square().sum(dim=-2).squeeze()[selected]
            rows.append(checkpoint(propagate,field,use_reentrant=False)
                        if use_checkpoint and torch.is_grad_enabled() and field.requires_grad else propagate(field))
        return torch.stack(rows)

    field=sim.sample_4f(phase)
    intensity=from_field(field,True)
    if reference_cache is None:
        with torch.no_grad():
            reference=from_field(sim.sample_4f(torch.zeros_like(phase)),False).detach()
    else:
        reference=reference_cache
    return dict(intensity=intensity,reference_intensity=reference,
                x_um=x_pixels.to(phase.dtype)*px_um,z_um=z[selected])
