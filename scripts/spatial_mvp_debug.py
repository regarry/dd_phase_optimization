"""One-allocation, one-step joint detector/tissue-loss debugging; never submits."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import yaml

from data.io import expand_config,load_config
from data.preprocessing import generate_bead_templates
from data.datasets import SyntheticMicroscopeData
from models.wrappers import ParallelEndToEndModel
from physics.masks import get_initial_phase_mask
from physics.phase_io import quantize_phase
from training.training import BinaryDiceLoss,MultiClassDiceLoss,compute_total_loss
from training.tissue_loss import TissueAwareProfileLoss


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--particles',type=int,default=32)
    p.add_argument('--initialization',default='axicon');p.add_argument('--side',choices=['one','two'],default='two')
    p.add_argument('--tissue-weight',type=float,default=.01)
    p.add_argument('--templates-from',type=Path,help='Reuse matching trusted PSF/bead templates from another MVP folder')
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count()!=1:
        raise RuntimeError('This debugging helper requires exactly one GPU in a Slurm allocation')
    if not 1<=a.particles<=1200:raise ValueError('particles must be bounded in 1..1200')
    torch.set_num_threads(4);a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    raw=load_config(a.config)
    bounds=[0,100] if a.side=='one' else [-100,100]
    raw.update(num_particles_range=[a.particles,a.particles+1],image_volume=[128,1000,256],
        freeze_unet=False,freeze_slm=False,num_classes=1,initial_phase_mask=a.initialization,
        convolve_psf_ground_truth=False,ce_loss_weight=1,dice_loss_weight=1,
        spatial_multi_well_loss_weight=0,tissue_spatial_loss_weight=a.tissue_weight,
        tissue_profile_x_um=[-400,-200,0,200,400],tissue_profile_z_bounds_um=[-150,150],
        tissue_z_bounds_um=bounds,tissue_sample_z_bounds_um=bounds,tissue_core_halfwidth_um=5,
        tissue_core_min_fraction=.8,tissue_core_floor_weight=10,tissue_core_uniformity_weight=0,
        resume_checkpoint=None)
    for key in ['mae_loss_weight','mse_loss_weight','tv_loss_weight']:raw[key]=0
    (a.output/'input.yaml').write_text(yaml.safe_dump(raw,sort_keys=False))
    cfg=expand_config(copy.deepcopy(raw),str(a.output));(a.output/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    seed=int(cfg['random_seed']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if a.templates_from:
        source=yaml.safe_load((a.templates_from/'config.yaml').read_text())
        keys=['wavelength','numerical_aperture','refractive_index','psf_width_pixels',
              'px','z_depth_list','bead_radius','supersample_factor','image_volume']
        if any(cfg.get(k)!=source.get(k) for k in keys):
            raise ValueError('Reused template scientific geometry differs')
        for name in ['psf_stack.mat',cfg['defocused_beads_filename']]:
            shutil.copyfile(a.templates_from/name,a.output/name)
    else:
        generate_bead_templates(cfg,str(a.output))
    model=ParallelEndToEndModel(cfg).train();phase=torch.nn.Parameter(torch.from_numpy(get_initial_phase_mask(cfg)).float().to('cuda:0'))
    ds=SyntheticMicroscopeData(1,cfg)
    for attempt in range(64):
        xyz,target=ds[0]
        if target.max()>0:break
    else:raise RuntimeError('No in-focus target found in the bounded MVP scene search')
    xyz=xyz[None].to('cuda:0');target=target[None].to('cuda:0')
    funcs={'ce_loss':torch.nn.BCEWithLogitsLoss(),'dice_loss':BinaryDiceLoss(),
           'tissue_spatial_loss':TissueAwareProfileLoss(cfg)}
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    physical=quantize_phase(phase,cfg.get('slm_bit_depth',8));outputs,profile=model(physical,xyz)
    total,weighted=compute_total_loss(outputs,target,physical,cfg,funcs,profile)
    detection=weighted['ce_loss']+weighted['dice_loss']
    dg=torch.autograd.grad(detection,phase,retain_graph=True)[0]
    og=torch.autograd.grad(weighted['tissue_spatial_loss'],phase,retain_graph=True)[0]
    total.backward();torch.cuda.synchronize()
    phase_norm=float(phase.grad.norm());detector_norm=float(torch.sqrt(sum(
        parameter.grad.square().sum() for parameter in model.unet.parameters() if parameter.grad is not None)))
    if not torch.isfinite(total) or not torch.isfinite(phase.grad).all() or phase_norm<=0 or detector_norm<=0:
        raise RuntimeError('Joint MVP has invalid loss, phase gradient or detector gradient')
    before=phase.detach().clone();optimizer=torch.optim.Adam([phase]+list(model.unet.parameters()),lr=cfg['initial_learning_rate'])
    optimizer.step();torch.cuda.synchronize()
    result=dict(success=True,job_id=os.environ['SLURM_JOB_ID'],particles=a.particles,
        target_positive_pixels=int(target.sum()),scene_selection_attempt=attempt+1,
        scene_selection_note='MVP explicitly selects the first nonempty target scene, not representative inference evaluation',
        initialization=a.initialization,side=a.side,weighted_losses={k:float(v.detach()) for k,v in weighted.items()},
        tissue_terms=funcs['tissue_spatial_loss'].last_terms,total_loss=float(total.detach()),
        detection_phase_gradient_norm=float(dg.norm()),weighted_optical_phase_gradient_norm=float(og.norm()),
        combined_phase_gradient_norm=phase_norm,detector_gradient_norm=detector_norm,
        phase_update_norm=float((phase-before).norm()),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        step_with_diagnostic_backwards_seconds=time.monotonic()-started,
        profile_shape=list(profile['intensity'].shape),raw_profile_finite=bool(torch.isfinite(profile['intensity']).all()),
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,python=__import__('sys').version,
        expanded_image_volume=cfg['image_volume'],psf_width_pixels=cfg['psf_width_pixels'],
        safe_y=cfg['particle_spatial_range_y'])
    result['templates_reused_from']=str(a.templates_from) if a.templates_from else None
    (a.output/'debug.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(a.output/'profiles.npz',intensity=profile['intensity'].detach().cpu().numpy(),
        reference=profile['reference_intensity'].cpu().numpy(),x_um=profile['x_um'].cpu().numpy(),z_um=profile['z_um'].cpu().numpy())
    print('SPATIAL_MVP_SUCCESS '+json.dumps(result),flush=True)


if __name__=='__main__':main()
