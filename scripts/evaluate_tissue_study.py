"""Bounded checkpoint-snapshot evaluation of binary and three-class tissue studies."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np

# Support a direct repository script invocation while respecting a supplied
# frozen backend on PYTHONPATH. Artifact copies rely on that explicit backend.
import importlib.util
import sys
if __package__ in (None,'') and importlib.util.find_spec('scripts') is None:
    source_root=Path(__file__).resolve().parents[1]
    if (source_root/'physics').is_dir():sys.path.insert(0,str(source_root))


def confusion(target,prediction,classes):
    target=np.asarray(target,dtype=np.int64);prediction=np.asarray(prediction,dtype=np.int64)
    if target.shape!=prediction.shape or np.any(target<0) or np.any(target>=classes) or np.any(prediction<0) or np.any(prediction>=classes):
        raise ValueError('Invalid class labels/shapes')
    return np.bincount((target*classes+prediction).ravel(),minlength=classes**2).reshape(classes,classes)


def class_scores(matrix):
    result=[]
    for c in range(len(matrix)):
        tp=int(matrix[c,c]);fp=int(matrix[:,c].sum()-tp);fn=int(matrix[c].sum()-tp)
        result.append(dict(class_id=c,tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,
                           recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None))
    return result


def spatial_regions(z,cfg):
    tissue=(z>=cfg['tissue_z_bounds_um'][0])&(z<=cfg['tissue_z_bounds_um'][1])
    core=tissue&(np.abs(z)<=cfg['tissue_core_halfwidth_um'])
    return tissue,core,tissue&~core


def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def inference_state_dict(saved,target,gpu_count):
    """Drop only verified duplicate optical buffers when reducing GPU replicas.

    This is inference only. Optimizer and CUDA RNG resume state are untouched.
    Every required target key is loaded strictly, with no learned-state fallback.
    """
    import torch
    prefix='physics.replicas.'
    result={}
    for key,value in saved.items():
        if key.startswith(prefix):
            parts=key.split('.',3);replica=int(parts[2]);suffix=parts[3]
            reference=prefix+'0.'+suffix
            if reference not in saved or not torch.equal(value,saved[reference]):
                raise ValueError('Optical replica state differs: '+key)
            if replica>=gpu_count:
                continue
        result[key]=value
    if set(result)!=set(target):
        raise ValueError('Inference state keys differ from model; no partial loading')
    return result


def initialize_cuda_metrics(cuda,gpu_count):
    # device_count may use NVML without initializing the caching allocator.
    # reset_peak_memory_stats itself does not call Torch's lazy initializer.
    cuda.init()
    for i in range(gpu_count):cuda.reset_peak_memory_stats(i)


def capture_one(entry,count):
    import torch
    from data.datasets import SyntheticMicroscopeData
    from models.wrappers import ParallelEndToEndModel
    from physics.phase_io import quantize_phase
    from physics.masks import get_initial_phase_mask
    from training.checkpoint import read_checkpoint
    from training.tissue_loss import TissueAwareProfileLoss,tissue_profiles
    gpu_count=torch.cuda.device_count()
    if not os.environ.get('SLURM_JOB_ID') or gpu_count not in (1,2):
        raise RuntimeError('Inference requires one or two Slurm-allocated GPUs')
    initialize_cuda_metrics(torch.cuda,gpu_count)
    started=time.monotonic();checkpoint=Path(entry['checkpoint']);output=Path(entry['output'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=entry['checkpoint_sha256']:raise RuntimeError('Snapshot changed')
    state=read_checkpoint(checkpoint);original=dict(state['config']);cfg=dict(original)
    # Suppress auxiliary training-loss output only. Image formation is unchanged.
    cfg['SpatialMulitWellLoss']=False
    torch.set_num_threads(4);torch.manual_seed(20261112);np.random.seed(20261112);random.seed(20261112)
    model=ParallelEndToEndModel(cfg).eval()
    model.load_state_dict(inference_state_dict(state['model'],model.state_dict(),gpu_count),strict=True)
    raw=state['raw_phase'];phase=quantize_phase(raw.to('cuda:0'),cfg.get('slm_bit_depth',8))
    random.seed(20261112);np.random.seed(20261112)
    fresh=SyntheticMicroscopeData(count,cfg)
    cases=[('fresh',i,*fresh[i]) for i in range(count)]
    cases += [('saved_validation',i,*v) for i,v in enumerate(state['val_samples'][:count])]
    metadata=[];classes=2 if cfg['num_classes']==1 else cfg['num_classes']
    ideal_capture={}
    def save_ideal(module,args):ideal_capture['image']=args[0].detach().cpu().numpy()
    hook=model.physics.replicas[0].noise.register_forward_pre_hook(save_ideal)
    with torch.no_grad():
        for group,i,xyz,target in cases:
            seed=20261212+i+(1000 if group=='saved_validation' else 0)
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            xyz=torch.as_tensor(xyz).float()
            if xyz.ndim != 2 or xyz.shape[1] not in (3,4):raise ValueError("Invalid emitter scene shape")
            target=np.asarray(target).squeeze().astype(np.int64)
            camera=model.physics(phase,xyz[None].to('cuda:0'));logits=model.unet(camera)
            if not torch.isfinite(camera).all() or not torch.isfinite(logits).all():raise RuntimeError('Nonfinite inference')
            if classes==2:
                bead=logits.sigmoid().squeeze().cpu().numpy();probability=np.stack([1-bead,bead]);pred=(bead>=.5).astype(np.int64)
            else:
                probability=logits.softmax(1).squeeze(0).cpu().numpy();pred=probability.argmax(0)
            matrix=confusion(target,pred,classes)
            file='%s_%02d.npz'%(group,i)
            np.savez_compressed(output/file,xyz=xyz.numpy(),target=target,prediction=pred,probability=probability,
                                camera_adu=camera.squeeze().cpu().numpy()*cfg['camera_max_adu'],ideal_photons=ideal_capture['image'].squeeze())
            metadata.append(dict(file=file,dataset=group,index=i,noise_seed=seed,confusion=matrix.tolist()))
        sim=model.physics.replicas[0]
        xs=np.unique(np.r_[np.rint(np.linspace(-500,499,61)),[-400,-200,0,200,400]].astype(int))
        profiles={}
        masks={'learned':phase}
        for name in ('gaussian','axicon'):
            reference_cfg=dict(cfg,initial_phase_mask=name,gaussian_phase_mode='focused')
            masks[name]=quantize_phase(torch.from_numpy(get_initial_phase_mask(reference_cfg)).to('cuda:0'),cfg.get('slm_bit_depth',8))
        for name,mask in masks.items():
            field=sim.sample_4f(mask);rows=[]
            for x in xs:
                propagated=sim.angular_spectrum_propagation(field,int(x),pad=True)
                rows.append(propagated.abs().square().sum(-2).squeeze().cpu().numpy())
            profiles[name]=np.asarray(rows)
        z=(np.arange(sim.N)-(sim.N//2-1))*sim.px*1e6
        np.savez_compressed(output/'beam_profiles.npz',x_um=xs*sim.px*1e6,z_um=z,**profiles)
        optical={}
        if original.get('tissue_spatial_loss_weight',0):
            criterion=TissueAwareProfileLoss(original);value=criterion(tissue_profiles(sim,phase,original))
            optical=dict(unweighted=float(value),weight=original['tissue_spatial_loss_weight'],terms=criterion.last_terms)
    hook.remove()
    meta=dict(run=entry['name'],completed_epochs=state['epoch']+1,progress=state.get('progress',{}),
              checkpoint_sha256=entry['checkpoint_sha256'],configuration=original,cases=metadata,classes=classes,
              optical_loss=optical,scene_seed=20261112,elapsed_seconds=time.monotonic()-started,
              job_id=os.environ['SLURM_JOB_ID'],gpu_names=[torch.cuda.get_device_name(i) for i in range(gpu_count)],
              inference_gpu_count=gpu_count,training_gpu_count=len(state['cuda_rng']),
              peak_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(gpu_count)],
              replica_note='Verified identical optical replica buffers; emitter summation grouping can change floating-point rounding',
              torch=torch.__version__,cuda=torch.version.cuda,auxiliary_training_profile_disabled=True)
    write_json(output/'capture.json',meta)
    del model,state,phase,masks;gc.collect()
    for i in range(gpu_count):
        with torch.cuda.device(i):torch.cuda.empty_cache()
    print('CAPTURE_SUCCESS',entry['name'],meta['completed_epochs'],meta['elapsed_seconds'],flush=True)


def capture_airy_references(plan):
    """Add only missing analytic Airy scans; preserve original captured arrays."""
    import torch
    from physics.simulation import OpticsSimulation
    from physics.masks import get_initial_phase_mask
    from physics.phase_io import quantize_phase
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count()!=1:
        raise RuntimeError('Airy reference capture requires one Slurm GPU')
    initialize_cuda_metrics(torch.cuda,1)
    for entry in plan['runs']:
        if not entry['name'].startswith('one_sided_'):continue
        out=Path(entry['output']);meta=json.loads((out/'capture.json').read_text(encoding='utf-8'));cfg=dict(meta['configuration'])
        with np.load(out/'beam_profiles.npz') as saved:
            x=saved['x_um'].copy();z=saved['z_um'].copy()
        with torch.no_grad():
            sim=OpticsSimulation(cfg,'cuda:0')
            alpha=plan.get('airy_reference_alpha',cfg['airy_alpha'])
            angle=float(plan.get('airy_reference_angle_degrees',0))
            if angle and not plan.get('airy_reference_focus',False):
                raise ValueError('Rotated Airy reference currently requires explicit focusing')
            mask=get_initial_phase_mask(dict(cfg,initial_phase_mask='airy',airy_alpha=alpha,
                airy_phase_mode='focused' if plan.get('airy_reference_focus',False) else 'cubic_only',
                airy_rotation_degrees=angle))
            phase=quantize_phase(torch.from_numpy(mask).to('cuda:0'),cfg.get('slm_bit_depth',8))
            field=sim.sample_4f(phase);rows=[]
            for position in np.rint(x/(sim.px*1e6)).astype(int):
                propagated=sim.angular_spectrum_propagation(field,int(position),pad=True)
                rows.append(propagated.abs().square().sum(-2).squeeze().cpu().numpy())
            intensity=np.asarray(rows)
            if not np.isfinite(intensity).all() or intensity.max()<=0:raise RuntimeError('Invalid Airy profile')
        np.savez_compressed(out/'airy_reference.npz',x_um=x,z_um=z,airy=intensity)
        np.save(out/'airy_reference_phase_radians.npy',phase.cpu().numpy())
        write_json(out/'airy_reference.json',dict(airy_alpha=alpha,angle_degrees=angle,focused=plan.get('airy_reference_focus',False),alpha_units='rad/m^3',
                   job_id=os.environ['SLURM_JOB_ID'],gpu_name=torch.cuda.get_device_name(0),
                   note='Analytic cubic phase; same input amplitude, sample_4f path and padded propagation'))
        del sim,phase,field;gc.collect();torch.cuda.empty_cache()
        print('AIRY_REFERENCE_SUCCESS',entry['name'],flush=True)


def comparison_names(name):
    return ('learned','gaussian','axicon','airy') if name.startswith('one_sided_') else ('learned','gaussian','axicon')


def render_beams(entry):
    import matplotlib.pyplot as plt
    out=Path(entry['output']);meta=json.loads((out/'capture.json').read_text(encoding='utf-8'));cfg=meta['configuration']
    with np.load(out/'beam_profiles.npz') as saved:b={key:saved[key].copy() for key in saved.files}
    names=comparison_names(entry['name'])
    if 'airy' in names:
        with np.load(out/'airy_reference.npz') as airy:
            np.testing.assert_allclose(airy['x_um'],b['x_um']);np.testing.assert_allclose(airy['z_um'],b['z_um'])
            b['airy']=airy['airy'].copy()
    x=b['x_um'];z=b['z_um'];fig,axes=plt.subplots(2,len(names),figsize=(5*len(names),8))
    shared_peak=None # Each beam is normalized independently, as requested.
    for i,name in enumerate(names):
        intensity=b[name];shared_peak=float(intensity.max());im=axes[0,i].imshow((intensity/shared_peak).T,origin='lower',extent=[x[0],x[-1],z[0],z[-1]],aspect='auto',cmap='inferno',vmin=0,vmax=1)
        airy_meta=json.loads((out/'airy_reference.json').read_text(encoding='utf-8')) if name=='airy' else {}
        title=name+(' | alpha=%.2e rad/m³, angle=%.0f°'%(airy_meta['airy_alpha'],airy_meta.get('angle_degrees',0)) if name=='airy' else '')
        axes[0,i].set(ylim=(-150,150),xlabel='Propagation x (µm)',ylabel='Thickness z (µm)',title=title)
        if cfg.get('tissue_z_bounds_um'):
            for bound in cfg['tissue_z_bounds_um']:axes[0,i].axhline(bound,color='cyan',ls='--',lw=1)
        fig.colorbar(im,ax=axes[0,i],label='Intensity / own beam peak (linear)')
        for loc in (-400,0,400):
            j=np.argmin(abs(x-loc));axes[1,i].plot(z,intensity[j]/shared_peak,label='x=%.0f µm'%x[j])
        axes[1,i].set(xlim=(-150,150),ylim=(0,1),xlabel='Thickness z (µm)',ylabel='Intensity / own beam peak (linear)');axes[1,i].legend()
    fig.suptitle(entry['name']+' | independent linear normalization; scan-y integrated; no fixed-dose claim');fig.tight_layout();fig.savefig(out/'beam_profiles.png',dpi=150);plt.close(fig)


def render_phase(entry):
    import torch
    import matplotlib.pyplot as plt
    state=torch.load(entry['checkpoint'],map_location='cpu',weights_only=False)
    phase=state['raw_phase'].numpy();output=Path(entry['output']);cfg=state['config']
    np.save(output/'raw_phase_radians.npy',phase)
    from physics.phase_io import quantize_phase
    physical=quantize_phase(torch.from_numpy(phase),cfg.get('slm_bit_depth',8)).numpy()
    np.save(output/'quantized_phase_radians.npy',physical)
    fig,ax=plt.subplots(figsize=(6,5));extent=np.array([-phase.shape[1]/2,phase.shape[1]/2,-phase.shape[0]/2,phase.shape[0]/2])*cfg['slm_px']*1e6
    im=ax.imshow(np.mod(physical,2*np.pi),origin='lower',extent=extent,cmap='twilight',vmin=0,vmax=2*np.pi)
    fig.colorbar(im,ax=ax,label='Quantized wrapped phase (rad)');ax.set(xlabel='Mask x (µm; configured pitch)',ylabel='Mask y (µm)',title='%s | epoch %d'%(entry['name'],state['epoch']+1))
    fig.tight_layout();fig.savefig(output/'phase_mask.png',dpi=150);plt.close(fig)
    return state


def render_spatial(plan):
    import matplotlib.pyplot as plt
    import yaml
    fig,axes=plt.subplots(2,2,figsize=(13,8));z=np.arange(-150,151)
    for i,name in enumerate(('two_sided','one_sided')):
        entry=next(e for e in plan['runs'] if e['name']==name+'_gaussian')
        cfg=yaml.safe_load(Path(entry['input_yaml']).read_text(encoding='utf-8'));tissue,core,off=spatial_regions(z,cfg)
        a=axes[i,0];a.fill_between(z,0,off.astype(float),step='mid',color='#e67e22',alpha=.5,label='Penalized fluorescent off-core region')
        a.fill_between(z,0,core.astype(float),step='mid',color='#2ca02c',alpha=.7,label='Useful core; throughput floor')
        a.plot(z,off.astype(float),color='#bf5310');a.set(xlabel='Thickness z (µm)',ylabel='Off-core spatial weight',title=name+' | applied at x = −400, −200, 0, 200, 400 µm');a.legend(fontsize=8)
        a.axvspan(*cfg['tissue_sample_z_bounds_um'],color='gray',alpha=.12,label='Bead-centre support')
        a=axes[i,1];q=np.linspace(0,1.5,300);floor=cfg['tissue_core_floor_weight']*np.maximum(cfg['tissue_core_min_fraction']-q,0)**2
        a.plot(q,floor);a.axvline(.8,color='gray',ls='--');a.set(xlabel='Core power / fixed reference core power',ylabel='Unweighted throughput penalty',title='10 × max(0, 0.8 − relative core power)²')
    fig.suptitle('Spatial objective: off-core tissue power / fixed reference core power + throughput floor\nMean across propagation positions; multiply by 0.01; outside-tissue energy has no direct reward.');fig.tight_layout();fig.savefig(Path(plan['output'])/'spatial_cost_functions.png',dpi=160);plt.close(fig)


def render_run(entry):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    out=Path(entry['output']);meta=json.loads((out/'capture.json').read_text(encoding='utf-8'));cfg=meta['configuration']
    render_beams(entry)
    pooled={};class_names=['background','bead'] if meta['classes']==2 else ['background','bead','connection']
    for group in ('fresh','saved_validation'):
        cases=[c for c in meta['cases'] if c['dataset']==group];matrix=np.sum([np.array(c['confusion']) for c in cases],axis=0)
        pooled[group]=dict(scenes=len(cases),confusion=matrix.tolist(),classes=class_scores(matrix))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,(group,m) in zip(axes,pooled.items()):
        matrix=np.array(m['confusion']);ax.imshow(matrix+1,cmap='Blues',norm=LogNorm(vmin=1,vmax=max(2,matrix.max()+1)))
        for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,str(v),ha='center',va='center',color='crimson')
        ax.set(xticks=range(meta['classes']),yticks=range(meta['classes']),xticklabels=class_names,yticklabels=class_names,xlabel='Predicted',ylabel='True',title=group)
    fig.tight_layout();fig.savefig(out/'confusion.png',dpi=150);plt.close(fig)
    for group in pooled:
        case=next(c for c in meta['cases'] if c['dataset']==group and c['index']==0)
        with np.load(out/case['file']) as d:
            fig,axes=plt.subplots(2,2,figsize=(15,7));camera=d['camera_adu'];lo,hi=np.percentile(camera,[1,99.8])
            axes[0,0].imshow(camera,cmap='gray',aspect='equal',vmin=lo,vmax=hi);axes[0,0].set_title('Exact detector camera input | ADU; display stretch; auto-exposure')
            axes[0,1].imshow(d['target'],aspect='equal',cmap='viridis',vmin=0,vmax=meta['classes']-1);axes[0,1].set_title('Target classes: '+', '.join(class_names))
            axes[1,0].imshow(d['probability'][1],aspect='equal',cmap='magma',vmin=0,vmax=1);axes[1,0].set_title('Bead probability')
            axes[1,1].imshow(camera,aspect='equal',cmap='gray',vmin=lo,vmax=hi);truth=d['target']==1;pred=d['prediction']==1
            for mask,color,label,marker in ((truth&pred,'lime','TP','o'),(truth&~pred,'red','FN','x'),(~truth&pred,'cyan','FP','+')):
                yy,xx=np.where(mask);axes[1,1].scatter(xx,yy,c=color,label=label,marker=marker,s=20)
            axes[1,1].legend();axes[1,1].set_title('Bead-class pixel detection (markers enlarged)')
            for ax in axes.ravel():ax.set(xlabel='Camera x / propagation pixel',ylabel='Camera y / scan pixel')
            fig.suptitle('%s | epoch %d | %s sample 0'%(entry['name'],meta['completed_epochs'],group));fig.tight_layout();fig.savefig(out/(group+'_sample.png'),dpi=150);plt.close(fig)
    from scripts.optical_report import augment_run
    extra=augment_run(entry,meta,pooled)
    meta['optical_metrics']=extra
    write_json(out/'metrics.json',pooled)
    write_json(out/'SUCCESS.json',dict(success=True,checkpoint_sha256=meta['checkpoint_sha256'],completed_epochs=meta['completed_epochs']))
    return meta,pooled


def render_overview(plan):
    import matplotlib.pyplot as plt
    runs=plan['runs'];columns=4;rows=(len(runs)+columns-1)//columns
    profiles=[]
    for entry in runs:
        with np.load(Path(entry['output'])/'beam_profiles.npz') as data:
            profiles.append({k:data[k].copy() for k in ('x_um','z_um','learned')})
    peak=None # Independent per-beam normalization for visualization only.
    fig,axes=plt.subplots(rows,columns,figsize=(17,3.3*rows),squeeze=False)
    for entry,b,ax in zip(runs,profiles,axes.ravel()):
        x=b['x_um'];z=b['z_um'];peak=float(b['learned'].max())
        im=ax.imshow((b['learned']/peak).T,cmap='inferno',origin='lower',extent=[x[0],x[-1],z[0],z[-1]],aspect='auto',vmin=0,vmax=1)
        ax.set(title=entry['name'],xlabel='Propagation x (µm)',ylabel='Thickness z (µm)',ylim=(-150,150))
        meta=json.loads((Path(entry['output'])/'capture.json').read_text(encoding='utf-8'))
        for bound in meta['configuration'].get('tissue_z_bounds_um',[]):ax.axhline(bound,color='cyan',ls='--',lw=.8)
    for ax in axes.ravel()[len(runs):]:ax.set_visible(False)
    fig.suptitle('Learned beam profiles: study plus Dice + BCE baselines\nIndependent linear normalization per beam; cyan = optical exclusion bounds; baseline geometry/budget differ')
    fig.tight_layout(rect=[0,0,.94,.94]);cax=fig.add_axes([.95,.16,.012,.66]);fig.colorbar(im,cax=cax,label='Intensity / own beam peak (linear)');fig.savefig(Path(plan['output'])/'beam_profiles_all_runs.png',dpi=160);plt.close(fig)


def gallery(plan):
    out=Path(plan['output']);rows=[];html=['<!doctype html><meta charset="utf-8"><title>Deep Design beam and inference comparisons</title><style>body{font:16px system-ui;margin:24px}img{max-width:100%}section{margin-bottom:60px}a{margin-right:15px}</style><h1>Deep Design checkpoint comparisons</h1><p>Two fresh and two reused validation scenes per run. Not final/converged performance. Camera auto-exposure enabled. Profiles use independent per-beam linear normalization; no fixed-dose claim.</p><img src="spatial_cost_functions.png">']
    html+=['<p><a href="beam_profiles_all_runs.png">All learned beam profiles (linear)</a> | <a href="phase_masks_all_12.png">12 study phase masks</a></p><p>Dice + BCE baselines are shown separately: their specimen depth and training budget differ from the tissue study. The full random baseline has no checkpoint and is unavailable.</p>']
    for e in plan['runs']:
        out_run=Path(e['output']);html+=['<section><h2>'+e['name']+'</h2><a href="'+e['name']+'/phase_mask.png">Phase mask</a>']
        if (out_run/'capture.json').exists():
            meta,metrics=render_run(e);rows.append(dict(run=e['name'],epochs=meta['completed_epochs'],metrics=metrics,optical=meta['optical_loss']))
            hours=meta['progress'].get('train_seconds')
            timing='%.2f actual train hours'%(hours/3600) if hours is not None else 'training-time counter unavailable in historical checkpoint'
            html+=['<p>%d completed epochs; %s</p>'%(meta['completed_epochs'],timing)]
            for image in ('beam_profiles','phase_mask','fresh_sample','saved_validation_sample','confusion'):
                html+=['<img loading="lazy" src="%s/%s.png">'%(e['name'],image)]
                if image=='confusion':
                    from scripts.optical_report import metrics_html
                    html+=[metrics_html(e,meta,metrics)]
            html+=['<img loading="lazy" src="%s/camera_diagnostics.png">'%e['name']]
            if e['name'].startswith(('one_sided_','two_sided_')):html+=['<img loading="lazy" src="%s/bead_xz_overlay.png">'%e['name']]
        else:html+=['<p>GPU inference pending.</p><img loading="lazy" src="'+e['name']+'/phase_mask.png">']
        html+=['</section>']
    if len(rows)==len(plan['runs']):render_overview(plan)
    (out/'index.html').write_text('\n'.join(html),encoding='utf-8');write_json(out/'evaluation_summary.json',rows)
    if len(rows)==len(plan['runs']):write_json(out/'SUCCESS.json',dict(success=True,evaluated_runs=len(rows)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('mode',choices=['phases','capture','references','render','all']);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--count',type=int,default=2);a=parser.parse_args()
    if not 1<=a.count<=4:parser.error('Bounded 1..4 test scenes per group')
    plan=json.loads(a.plan.read_text(encoding='utf-8'))
    if a.mode in ('phases','render','all'):
        import matplotlib
        matplotlib.use('Agg')
        if a.mode=='phases':
            for entry in plan['runs']:render_phase(entry)
        render_spatial(plan)
    if a.mode=='references':capture_airy_references(plan)
    if a.mode in ('capture','all'):
        failures=[]
        for entry in plan['runs']:
            try:capture_one(entry,a.count)
            except Exception as exc:
                import traceback
                details=traceback.format_exc();print(details,flush=True)
                write_json(Path(entry['output'])/'ERROR.json',dict(error=repr(exc),traceback=details));failures.append(entry['name'])
        if failures:raise RuntimeError('Inference failures: '+', '.join(failures))
    if a.mode in ('phases','render','all'):gallery(plan)
