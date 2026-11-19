"""Bounded interactive-GPU Airy estimates through the renderer's scanned path."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from physics.simulation import OpticsSimulation
from physics.masks import get_initial_phase_mask
from physics.phase_io import quantize_phase
from utils.beam_metrics import beam_metrics


def main(config,output):
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count()!=1:raise RuntimeError('One allocated GPU required')
    cfg=json.loads(Path(config).read_text())['configuration'];out=Path(output);out.mkdir(exist_ok=True,parents=True)
    torch.cuda.init();sim=OpticsSimulation(cfg,'cuda:0');f=cfg['lensless_prop_distance'];k=sim.k
    theory=json.loads((out/'airy_theory.json').read_text());a0=theory['estimated_alpha_rad_per_m3']
    xs=np.unique(np.r_[np.linspace(-500,500,81).round(),0]).astype(int);z=(np.arange(sim.N)-(sim.N//2-1))*sim.px*1e6
    u=(torch.arange(sim.N,device='cuda:0')-sim.N//2)*cfg['slm_px'];v,w=torch.meshgrid(u,u,indexing='ij')
    lens=torch.from_numpy(get_initial_phase_mask(dict(cfg,initial_phase_mask='gaussian',gaussian_phase_mode='focused'))).to('cuda:0')
    candidates=[('gaussian',lens,None),('axicon',torch.from_numpy(get_initial_phase_mask(dict(cfg,initial_phase_mask='axicon'))).float().to('cuda:0'),None)]
    candidates.append(('legacy_cubic',-(1e10)*(v**3+w**3),1e10))
    for factor in (.25,.5,1,2,4,8,1e12/a0,2e12/a0):
        alpha=a0*factor;x0=f*(3*alpha)**(1/3)/k
        # Predicted main-lobe offset plus renderer's +1 pixel z origin.
        shift=1.018792971647471*x0-sim.px
        mask=lens-alpha*(v**3+w**3)+(k*shift/f)*w
        candidates.append(('focused_airy_%g'%factor,mask,alpha))
    profiles={};records=[];started=time.monotonic()
    with torch.no_grad():
        for name,mask,alpha in candidates:
            phase=quantize_phase(mask,cfg.get('slm_bit_depth',8));field=sim.sample_4f(phase);rows=[]
            for x in xs:
                propagated=sim.angular_spectrum_propagation(field,int(x),pad=True)
                rows.append(propagated.abs().square().sum(-2).squeeze().cpu().numpy())
            rows=np.asarray(rows);profiles[name]=rows
            m=beam_metrics(xs*sim.px*1e6,z,rows);m.update(name=name,alpha_rad_per_m3=alpha)
            records.append(m);np.save(out/(name+'_phase_radians.npy'),phase.cpu().numpy())
            print('AIRY_SCAN',name,alpha,m['main_lobe_1e_um'],m['on_axis_lobe_1e_um'],flush=True)
    target=records[1]['on_axis_lobe_1e_um']
    choices=[r for r in records if r['name'].startswith('focused_') and r['on_axis_lobe_1e_um']]
    best=min(choices,key=lambda r:abs(np.log(r['on_axis_lobe_1e_um']/target)))
    np.savez_compressed(out/'airy_scan.npz',x_um=xs*sim.px*1e6,z_um=z,**profiles)
    (out/'airy_scan.json').write_text(json.dumps(dict(records=records,selected=best,target_axicon_width_um=target,
        selection='closest scanned on-axis 1/e thickness; OS/range reported independently, not jointly optimized',
        configuration=cfg,job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(0),elapsed_seconds=time.monotonic()-started),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.config,a.output)
