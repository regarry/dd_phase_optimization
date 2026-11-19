"""Bounded full-density CUDA validation of weighted single- and three-view scenes."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from data.io import expand_config,load_config
from data.datasets import SyntheticMicroscopeData
from data.preprocessing import generate_bead_templates
from models.wrappers import ParallelEndToEndModel
from physics.masks import get_initial_phase_mask
from physics.phase_io import quantize_phase
from training.training import BinaryDiceLoss,MultiClassDiceLoss,compute_total_loss
from training.seeds import seed_all


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Requires a Slurm GPU allocation')
    a.output.mkdir(parents=True,exist_ok=True);results=[]
    for views,classes in [(1,1),(3,3)]:
        folder=a.output/str(views);folder.mkdir(exist_ok=True)
        raw=load_config(a.config);raw.update(emission_strength_range=[.5,1.5],num_classes=classes,
            num_particles_range=[1200,1201],tissue_spatial_loss_weight=0,spatial_multi_well_loss_weight=0,
            tissue_sample_z_bounds_um=[-100,100],initial_phase_mask='random')
        radius_px=raw['bead_radius']/(raw['slm_px']/raw['phase_mask_upsample_factor']*1e6)
        raw['z_depth_list']=[-radius_px,0,radius_px] if views==3 else [0]
        cfg=expand_config(copy.deepcopy(raw),str(folder));generate_bead_templates(cfg,str(folder))
        seed_all(211);model=ParallelEndToEndModel(cfg).train()
        phase=torch.nn.Parameter(torch.from_numpy(get_initial_phase_mask(cfg)).float().cuda())
        xyz,target=SyntheticMicroscopeData(1,cfg)[0]
        assert xyz.shape==(1200,4) and .5<=float(xyz[:,3].min())<=float(xyz[:,3].max())<=1.5
        xyz=xyz[None].cuda();target=target[None].cuda()
        funcs={'ce_loss':torch.nn.CrossEntropyLoss(weight=torch.tensor(cfg['ce_class_weights']).cuda()) if classes==3 else torch.nn.BCEWithLogitsLoss(),
            'dice_loss':MultiClassDiceLoss() if classes==3 else BinaryDiceLoss()}
        for i in range(torch.cuda.device_count()):torch.cuda.reset_peak_memory_stats(i)
        started=time.monotonic();physical=quantize_phase(phase,cfg.get('slm_bit_depth',8))
        outputs=model(physical,xyz);loss,components=compute_total_loss(outputs,target,physical,cfg,funcs,None)
        loss.backward();torch.cuda.synchronize()
        assert torch.isfinite(loss) and torch.isfinite(outputs).all() and torch.isfinite(phase.grad).all() and phase.grad.norm()>0
        unet_gradient=sum(float(x.grad.square().sum()) for x in model.unet.parameters() if x.grad is not None)**.5
        assert unet_gradient>0
        result=dict(views=views,classes=classes,success=True,loss=float(loss),phase_gradient_norm=float(phase.grad.norm()),detector_gradient_norm=unet_gradient,
            seconds=time.monotonic()-started,peak_bytes=[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())],
            gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],camera_channels=views,z_depth_list=cfg['z_depth_list'],
            brightness_range=[float(xyz[...,3].min()),float(xyz[...,3].max())],python=__import__('sys').version,torch=torch.__version__,job_id=os.environ['SLURM_JOB_ID'])
        results.append(result);print('TRANSFER_SMOKE '+json.dumps(result),flush=True)
        (a.output/'results.json').write_text(json.dumps(results,indent=2))
        del outputs,loss,components,model,phase,physical,xyz,target;gc.collect();torch.cuda.empty_cache()
    (a.output/'success.json').write_text(json.dumps(dict(success=True,results=results),indent=2))

if __name__=='__main__':main()
