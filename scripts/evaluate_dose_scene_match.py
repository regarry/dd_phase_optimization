import argparse
import hashlib,json,os,random
from pathlib import Path
import numpy as np
import torch
from skimage.io import imread
from data.datasets import SyntheticMicroscopeData
from physics.phase_io import quantize_phase
from physics.simulation import NoiseLayer
from scripts.evaluate_fixed_dose import load_model,audit_beam,confusion,class_scores

parser = argparse.ArgumentParser(description='Compare the historical Gaussian snapshot and the fixed-dose checkpoint on paired scenes.')
parser.add_argument('--fixed-run', required=True, type=Path)
parser.add_argument('--auto-run', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
fixed, auto, out = args.fixed_run.resolve(), args.auto_run.resolve(), args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
state,cfg,model,fixed_phase=load_model(dict(checkpoint=str(fixed/'checkpoints/latest.pt'),artifacts=str(fixed)))
assert state['epoch']==265 and state['progress']['train_scenes']==34048
del state
sim=model.physics.replicas[0]
snapshot=auto/'models/net_34.pt'
phase_path=auto/'learned_phase_masks/radians/mask_phase_epoch_34.tif'
auto_phase=quantize_phase(torch.from_numpy(imread(phase_path)).float().cuda(),cfg.get('slm_bit_depth',8))
assert auto_phase.shape==fixed_phase.shape
seed=2026112519
np.random.seed(seed);random.seed(seed);torch.manual_seed(seed)
dataset=SyntheticMicroscopeData(128,cfg)
scenes=[dataset[i] for i in range(128)]
profiles={};results={}
xs=np.unique(np.r_[np.rint(np.linspace(-500,499,65)),0].astype(int))
zs=(np.arange(sim.N)-(sim.N//2-1))*cfg['px']*1e6
with torch.no_grad():
    for name,phase in [('fixed',fixed_phase),('auto',auto_phase)]:
        if name=='auto':
            weights=torch.load(snapshot,map_location='cpu',weights_only=True)
            model.unet.load_state_dict({k[5:]:v for k,v in weights.items() if k.startswith('unet.')},strict=True)
            del weights
        field=sim.sample_4f(phase)
        profiles[name]=np.stack([sim.angular_spectrum_propagation(field,int(x),pad=True).abs().square().sum(-2).squeeze().cpu().numpy() for x in xs])
        np.save(out/(name+'_phase_rad.npy'),phase.cpu().numpy())
        cameras={mode:NoiseLayer(dict(cfg,camera_exposure_mode=mode)) for mode in ('fixed','auto_percentile')}
        counts={mode:[] for mode in cameras}
        for i,(xyz,target) in enumerate(scenes):
            expected=sim(phase,xyz[None].cuda())
            truth=(np.asarray(target).squeeze()==1).astype(np.int64)
            for mode,noise in cameras.items():
                torch.manual_seed(seed+1000+i)
                camera=noise(expected)/cfg['camera_max_adu']
                logits=model.unet(camera)
                assert torch.isfinite(camera).all() and torch.isfinite(logits).all()
                prob=logits.sigmoid().squeeze().cpu().numpy()
                pred=(prob>=.5).astype(np.int64)
                counts[mode].append(confusion(truth,pred,2))
                if i==0:
                    np.savez_compressed(out/f'{name}_{mode}_example.npz',camera=camera.squeeze().cpu().numpy(),truth=truth,probability=prob,prediction=pred)
        results[name]=dict(power=audit_beam(sim,phase),exposure={})
        for mode,items in counts.items():
            a=np.stack(items)
            pooled=a.sum(0)
            results[name]['exposure'][mode]=dict(confusion=pooled.tolist(),classes=class_scores(pooled),per_scene=a.tolist())
        print(name,{k:v['classes'][1] for k,v in results[name]['exposure'].items()},flush=True)
np.savez_compressed(out/'beams.npz',x_um=xs*cfg['px']*1e6,z_um=zs,**profiles)
(out/'results.json').write_text(json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'],scene_seed=seed,scenes=128,training_scenes=dict(fixed=34048,auto=33600),fixed_scale=cfg['fixed_exposure_scale'],protocol='Shared fixed-dose study geometry, illumination optics, noise and bead scenes; detector weights and phase from respective snapshots.',hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (snapshot,phase_path,fixed/'checkpoints/latest.pt')},results=results),indent=2))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
peak=max(p.max() for p in profiles.values())
for col,(name,profile) in enumerate(profiles.items()):
    for row,maximum in enumerate((profile.max(),peak)):
        ax=axes[row,col]
        image=ax.pcolormesh(xs*cfg['px']*1e6,zs,profile.T,shading='auto',cmap='magma',vmin=0,vmax=maximum)
        ax.set(ylim=(-150,150),xlabel='Propagation x (µm)',ylabel='Sheet-normal z (µm)',title=f'{name} trained · '+('own peak' if row==0 else 'shared scale'))
        fig.colorbar(image,ax=ax,label='Scan-integrated intensity (a.u.)')
fig.savefig(out/'beam_profiles.png',dpi=150)
lines=['# Exposure comparison at similar training scene counts','','128 paired fresh scenes on the fixed-dose study geometry. The historical run trained on different geometry, bead brightness, and validation cadence.','', '| Trained with | Evaluated with | Bead F1 | Precision | Recall |','|---|---|---:|---:|---:|']
for name,result in results.items():
    for mode,value in result['exposure'].items():
        c=value['classes'][1]
        values=' | '.join('—' if c[k] is None else f'{c[k]:.4f}' for k in ('f1','precision','recall'))
        lines.append(f'| {name} | {mode} | {values} |')
lines.extend(['','![Beam profiles](beam_profiles.png)',''])
(out/'REPORT.md').write_text('\n'.join(lines))
