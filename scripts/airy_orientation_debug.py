"""Bounded same-pupil Airy phase-rotation diagnostic, not training."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from physics.simulation import OpticsSimulation
from physics.masks import get_initial_phase_mask
from physics.phase_io import quantize_phase
from utils.beam_metrics import beam_metrics,lobe_width,power_interval


def phase_parameters(size,pitch,k,f,alpha,angle,renderer_pixel):
    from physics.masks import focused_airy_phase_parameters
    return focused_airy_phase_parameters(size,pitch,k,f,alpha,angle,renderer_pixel)


def scan(config,output,alphas=None,angles=None):
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count()!=1:raise RuntimeError('One allocated GPU required')
    cfg=json.loads(Path(config).read_text())['configuration'];out=Path(output);out.mkdir(exist_ok=True,parents=True)
    torch.cuda.init();sim=OpticsSimulation(cfg,'cuda:0');z=(np.arange(sim.N)-(sim.N//2-1))*sim.px*1e6
    started=time.monotonic();records=[];masks={};focus_profiles={}
    def profile(mask,x=0):
        phase=quantize_phase(torch.from_numpy(mask).to('cuda:0'),cfg.get('slm_bit_depth',8));field=sim.sample_4f(phase)
        propagated=sim.angular_spectrum_propagation(field,int(x),pad=True)
        return propagated.abs().square().sum(-2).squeeze().cpu().numpy()
    with torch.no_grad():
        axicon=get_initial_phase_mask(dict(cfg,initial_phase_mask='axicon'));reference=profile(axicon)
        target_width=lobe_width(z,reference);target_os=power_interval(z,reference)['width_um']
        for alpha in (alphas or (1e11,3e11,6e11,1e12,1.5e12,2e12)):
            for angle in (angles or (-45,-30,-15,0,15,30,45)):
                name='alpha_%g_angle_%g'%(alpha,angle);mask,step=phase_parameters(sim.N,cfg['slm_px'],sim.k,cfg['lensless_prop_distance'],alpha,angle,sim.px)
                if step>=np.pi:
                    records.append(dict(name=name,alpha=alpha,angle_degrees=angle,max_phase_step_rad=step,rejected='phase gradient exceeds Nyquist pi per pixel'));continue
                row=profile(mask);width=lobe_width(z,row);oswidth=power_interval(z,row)['width_um'];peak=float(z[row.argmax()])
                score=abs(np.log(width/target_width))+abs(np.log(oswidth/target_os)) if width else None
                records.append(dict(name=name,alpha=alpha,angle_degrees=angle,max_phase_step_rad=step,width_1e_um=width,os63_um=oswidth,peak_z_um=peak,score=score));masks[name]=mask;focus_profiles[name]=row
                print('ORIENTATION_SCREEN',name,width,oswidth,peak,flush=True)
        valid=[r for r in records if r.get('score') is not None and abs(r['peak_z_um'])<=20]
        ranked=sorted(valid,key=lambda r:r['score'])[:4]
        xs=np.linspace(-500,500,81).round().astype(int);profiles={};details=[]
        for name,mask in [('axicon',axicon)]+[(r['name'],masks[r['name']]) for r in ranked]:
            phase=quantize_phase(torch.from_numpy(mask).to('cuda:0'),cfg.get('slm_bit_depth',8));field=sim.sample_4f(phase);rows=[]
            for x in xs:
                propagated=sim.angular_spectrum_propagation(field,int(x),pad=True)
                rows.append(propagated.abs().square().sum(-2).squeeze().cpu().numpy())
            rows=np.asarray(rows);profiles[name]=rows;m=beam_metrics(xs*sim.px*1e6,z,rows);m['name']=name;details.append(m)
            np.save(out/(name+'_phase_radians.npy'),phase.cpu().numpy())
    np.savez_compressed(out/'orientation_profiles.npz',x_um=xs*sim.px*1e6,z_um=z,**profiles)
    np.savez_compressed(out/'focus_screens.npz',z_um=z,axicon=reference,**focus_profiles)
    result=dict(job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),configuration=cfg,screen=records,selected_focus_matches=ranked,propagation_metrics=details,target_width_1e_um=target_width,target_os63_um=target_os,elapsed_seconds=time.monotonic()-started,selection='Sum of absolute log errors in 1/e width and OS63 versus same-pupil scanned axicon; |peak z|<=20um; reject analytic phase steps >=pi. Not a global optimum or equality-of-propagation-length proof.')
    (out/'orientation_results.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--alphas');p.add_argument('--angles');a=p.parse_args();scan(a.config,a.output,[float(v) for v in a.alphas.split(',')] if a.alphas else None,[float(v) for v in a.angles.split(',')] if a.angles else None)
