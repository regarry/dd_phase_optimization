"""Bounded, Slurm-only trained inference capture and CPU artifact rendering."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np


def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def pixel_counts(probability,target,threshold=.5):
    pred=probability>=threshold;truth=target.astype(bool)
    return np.array([[np.sum(~truth&~pred),np.sum(~truth&pred)],
                     [np.sum(truth&~pred),np.sum(truth&pred)]],dtype=np.int64)


def scores(matrix):
    tn,fp,fn,tp=map(int,np.asarray(matrix).ravel())
    return dict(tn=tn,fp=fp,fn=fn,tp=tp,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def detection_counts(probability,target,threshold=.5,radius=2):
    """8-connected predicted components, peak pixels, one-to-one 2D matching."""
    from scipy.ndimage import label,maximum_position
    from scipy.optimize import linear_sum_assignment
    truth=np.argwhere(target.astype(bool))
    labels,n=label(probability>=threshold,structure=np.ones((3,3)))
    peaks=np.asarray(maximum_position(probability,labels,index=list(range(1,n+1)))).reshape(-1,2)
    matches=0
    if len(truth) and len(peaks):
        distance=np.linalg.norm(truth[:,None]-peaks[None],axis=-1)
        cost=np.where(distance<=radius,distance/(radius+1),1e6)
        i,j=linear_sum_assignment(cost);matches=int(np.sum(distance[i,j]<=radius))
    return dict(tp=matches,fp=len(peaks)-matches,fn=len(truth)-matches,
                target_points=len(truth),predicted_components=n)


def capture(checkpoint,output,count,seed=1729):
    import torch
    from data.datasets import SyntheticMicroscopeData
    from models.wrappers import ParallelEndToEndModel
    from physics.phase_io import quantize_phase
    from training.checkpoint import read_checkpoint
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count()!=2:
        raise RuntimeError('Capture requires a Slurm allocation with exactly two visible GPUs')
    if not 1<=count<=32:raise ValueError('Bounded evaluation allows 1..32 scenes per dataset')
    output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    state=read_checkpoint(checkpoint);cfg=state['config'].copy()
    if cfg['num_classes']!=1 or cfg.get('SpatialMulitWellLoss',False):
        raise ValueError('This evaluation supports the binary detection path only')
    if cfg['lens_approach']!='sample_4f' or cfg.get('phase_mask_upsample_factor',1)!=1:
        raise ValueError('Profile capture requires the reviewed sample_4f path with phase upsample 1')
    if count>len(state['val_samples']):raise ValueError('Not enough cached validation scenes')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(4)
    model=ParallelEndToEndModel(cfg).eval();model.load_state_dict(state['model'],strict=True)
    phase=quantize_phase(state['raw_phase'].to('cuda:0'),cfg.get('slm_bit_depth',8))
    # Reset scene RNG after construction, keeping scientific generator/settings.
    random.seed(seed);np.random.seed(seed)
    dataset=SyntheticMicroscopeData(count,cfg)
    fresh=[dataset[i] for i in range(count)]
    cases=[('fresh',i,*v) for i,v in enumerate(fresh)]
    cases += [('saved_validation',i,*v) for i,v in enumerate(state['val_samples'][:count])]
    case_meta=[]
    with torch.no_grad():
        for group,i,xyz,target in cases:
            xyz=torch.as_tensor(xyz).float().reshape(-1,3)
            target=np.asarray(target).squeeze()
            if not np.all(np.isin(target,[0,1])):raise ValueError('Expected binary point targets')
            noise_seed=seed+100000+i+(1000 if group=='saved_validation' else 0)
            torch.manual_seed(noise_seed);torch.cuda.manual_seed_all(noise_seed)
            camera=model.physics(phase,xyz[None].to('cuda:0'))
            logits=model.unet(camera)
            if not torch.isfinite(camera).all() or not torch.isfinite(logits).all():
                raise RuntimeError('Nonfinite camera or inference tensor')
            probability=logits.sigmoid().squeeze().cpu().numpy()
            if probability.shape!=target.shape:raise ValueError('Prediction/target shape mismatch')
            name='%s_%03d.npz'%(group,i)
            np.savez_compressed(output/name,xyz=xyz.numpy(),target=target,
                probability=probability,camera_adu=camera.squeeze().cpu().numpy()*cfg['camera_max_adu'])
            case_meta.append(dict(file=name,dataset=group,index=i,noise_seed=noise_seed))
            print('TRAINED_CAPTURE',group,i,flush=True)
        # Same sample-plane field and padded angular-spectrum path as the renderer.
        sim=model.physics.replicas[0]
        xs=np.unique(np.rint(np.linspace(-cfg['image_volume'][1]//2,
                       cfg['image_volume'][1]//2-1,101)).astype(int))
        profiles={}
        for name,mask in [('learned',phase),('flat_reference',torch.zeros_like(phase))]:
            field=sim.sample_4f(mask)
            rows=[]
            for x in xs:
                propagated=sim.angular_spectrum_propagation(field,int(x),pad=True)
                rows.append(propagated.abs().square().sum(dim=-2).squeeze().cpu().numpy())
            profiles[name]=np.asarray(rows)
        # Forward samples thickness at N//2-1+z, rather than N//2+z.
        zs=(np.arange(sim.N)-(sim.N//2-1))*sim.px*1e6
        np.savez_compressed(output/'beam_profiles.npz',propagation_x_um=xs*sim.px*1e6,
                            thickness_z_um=zs,**profiles)
        np.save(output/'raw_phase_radians.npy',state['raw_phase'].numpy())
    metadata=dict(trained=True,checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint_epoch_index=state['epoch'],completed_epochs=state['epoch']+1,
        configuration=cfg,raw_config=state['raw_config'],cases=case_meta,
        scene_seed=seed,threshold=.5,object_tolerance_pixels=2,
        python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        gpu_names=[torch.cuda.get_device_name(i) for i in range(2)],
        visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),job_id=os.environ['SLURM_JOB_ID'],
        elapsed_seconds=time.monotonic()-started,
        axes='x: illumination propagation/camera columns; y: scan/camera rows; z: sheet thickness/detection depth',
        profile_note='Raw scan-summed intensity, identical sample_4f input and pad=True propagation; flat reference has no phase modulation; camera auto-exposure remains enabled',
        train_losses=state['train_losses'],validation_losses=state['val_losses'])
    write_json(output/'capture_metadata.json',metadata)


def render(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    meta=json.loads((output/'capture_metadata.json').read_text());summary={};rows=[]
    thresholds=np.unique(np.r_[np.linspace(.01,.99,50),.5])
    fig,axs=plt.subplots(1,2,figsize=(10,4.5))
    prfig,prax=plt.subplots(figsize=(6,5))
    for gindex,group in enumerate(['fresh','saved_validation']):
        cases=[c for c in meta['cases'] if c['dataset']==group]
        pooled=np.zeros((2,2),dtype=np.int64);curve=np.zeros((len(thresholds),2,2),dtype=np.int64)
        objects=dict(tp=0,fp=0,fn=0);sample=None
        for case in cases:
            with np.load(output/case['file']) as data:
                p=data['probability'];target=data['target'];m=pixel_counts(p,target);pooled+=m
                for ti,threshold in enumerate(thresholds):curve[ti]+=pixel_counts(p,target,threshold)
                d=detection_counts(p,target)
                for k in objects:objects[k]+=d[k]
                rows.append(dict(dataset=group,index=case['index'],**scores(m),
                                 object_tp=d['tp'],object_fp=d['fp'],object_fn=d['fn']))
                if case['index']==0:sample={k:data[k].copy() for k in data.files}
        om=scores([[0,objects['fp']],[objects['fn'],objects['tp']]]);om.pop('tn')
        summary[group]=dict(scenes=len(cases),pixel=scores(pooled),object_2d=om,
                            confusion_matrix=pooled.tolist(),threshold=.5,
                            object_definition='8-connected components, peak-probability pixel, one-to-one match within 2 sample pixels; ground truth is unique in-focus projected points')
        ax=axs[gindex];ax.imshow(pooled+1,cmap='Blues',norm=LogNorm(vmin=1,vmax=max(2,pooled.max()+1)))
        for i in range(2):
            for j in range(2):ax.text(j,i,str(pooled[i,j]),ha='center',va='center',color='#b11d24')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['background','bead'],yticklabels=['background','bead'],
               xlabel='Predicted',ylabel='True',title=group+' | %d scenes'%len(cases))
        sm=summary[group]['pixel'];figtitle='Pixel confusion matrices | threshold 0.5 | counts, logarithmic color'
        curve_scores=[scores(m) for m in curve]
        prax.plot([m['recall'] if m['recall'] is not None else np.nan for m in curve_scores],
                  [m['precision'] if m['precision'] is not None else np.nan for m in curve_scores],label=group)
        if sm['recall'] is not None and sm['precision'] is not None:prax.scatter(sm['recall'],sm['precision'],marker='x',s=80)
        write_json(output/(group+'_threshold_curve.json'),[dict(threshold=float(t),**s) for t,s in zip(thresholds,curve_scores)])
        if sample is not None:
            camera=sample['camera_adu'];p=sample['probability'];target=sample['target'].astype(bool);pred=p>=.5
            panel,axes=plt.subplots(2,2,figsize=(15,7))
            vmin,vmax=np.percentile(camera,[1,99.8]);axes[0,0].imshow(camera,cmap='gray',aspect='auto',vmin=vmin,vmax=vmax)
            axes[0,0].set_title('Exact synthetic camera tensor supplied to detector\nADU; 1–99.8% display stretch; auto-exposure')
            axes[0,1].imshow(target,cmap='gray',aspect='auto');axes[0,1].set_title('In-focus bead target; one-pixel projected centers')
            im=axes[1,0].imshow(p,cmap='magma',vmin=0,vmax=1,aspect='auto');panel.colorbar(im,ax=axes[1,0]);axes[1,0].set_title('Trained bead probability')
            axes[1,1].imshow(camera,cmap='gray',aspect='auto',vmin=vmin,vmax=vmax)
            for mask,color,label,marker in [(target&pred,'#00dc65','TP','o'),(target&~pred,'#ff2727','FN','x'),(~target&pred,'#26a2ff','FP','+')]:
                yy,xx=np.where(mask);axes[1,1].scatter(xx,yy,c=color,label=label,marker=marker,s=22,linewidths=1)
            axes[1,1].legend();axes[1,1].set_title('Pixel detection overlay (markers enlarged for visibility)')
            for ax in axes.ravel():ax.set(xlabel='Camera x pixel / propagation',ylabel='Camera y pixel / scan')
            panel.suptitle('Trained Gaussian-initialized model, epoch %d | %s case 0'%(meta['completed_epochs'],group));panel.tight_layout()
            panel.savefig(output/(group+'_case0.png'),dpi=150,bbox_inches='tight');plt.close(panel)
    fig.suptitle(figtitle);fig.tight_layout();fig.savefig(output/'confusion_matrices.png',dpi=150);plt.close(fig)
    prax.set(xlabel='Pixel recall',ylabel='Pixel precision',xlim=(0,1.02),ylim=(0,1.02),title='Fixed threshold sweep; × marks threshold 0.5');prax.legend();prfig.tight_layout();prfig.savefig(output/'precision_recall.png',dpi=150);plt.close(prfig)
    with (output/'per_scene_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with np.load(output/'beam_profiles.npz') as b:
        x=b['propagation_x_um'];z=b['thickness_z_um'];beamfig,baxs=plt.subplots(2,2,figsize=(13,8))
        for i,name in enumerate(['learned','flat_reference']):
            intensity=b[name];peak=intensity.max()
            if not np.isfinite(intensity).all() or peak<=0:raise ValueError('Invalid raw beam intensity')
            im=baxs[0,i].imshow((intensity/peak).T,origin='lower',extent=[x[0],x[-1],z[0],z[-1]],aspect='auto',cmap='inferno',vmin=0,vmax=1)
            baxs[0,i].set(xlabel='Propagation x (µm)',ylabel='Sheet thickness z (µm)',ylim=(-100,100),title=name+' | normalized to its own peak');beamfig.colorbar(im,ax=baxs[0,i])
            for loc in [-400,0,400]:
                j=np.argmin(abs(x-loc));baxs[1,i].plot(z,intensity[j]/peak,label='x = %.0f µm'%x[j])
            baxs[1,i].set(xlabel='Sheet thickness z (µm)',ylabel='Intensity / global profile peak',xlim=(-100,100));baxs[1,i].legend()
        beamfig.suptitle('Same simulator input and pupil; scan y integrated\nUncalibrated simulation; camera auto-exposure prevents fixed-dose performance claims');beamfig.tight_layout();beamfig.savefig(output/'beam_profiles.png',dpi=150);plt.close(beamfig)
    lf,la=plt.subplots(figsize=(7,4));la.plot(np.arange(1,len(meta['train_losses'])+1),meta['train_losses'],label='train');la.plot(np.arange(1,len(meta['validation_losses'])+1),meta['validation_losses'],label='validation');la.set(xlabel='Completed epoch',ylabel='CE + Dice loss');la.legend();lf.tight_layout();lf.savefig(output/'loss_history.png',dpi=150);plt.close(lf)
    write_json(output/'metrics.json',summary)
    lines=['# Trained Gaussian detection evaluation','',
           'Final checkpoint: **%d completed epochs**, not the best-validation epoch checkpoint.'%meta['completed_epochs'],'',
           '| Dataset | Scenes | Pixel precision | Pixel recall | Pixel F1 | 2D point F1 (2 px tolerance) |',
           '|---|---:|---:|---:|---:|---:|']
    def fmt(v):return 'undefined' if v is None else '%.4f'%v
    for name,s in summary.items():lines.append('| %s | %d | %s | %s | %s | %s |'%(name,s['scenes'],fmt(s['pixel']['precision']),fmt(s['pixel']['recall']),fmt(s['pixel']['f1']),fmt(s['object_2d']['f1'])))
    lines+=['','Fresh scenes use the recorded scene seed and the unchanged scientific generator. Saved-validation scenes are reused training-time validation examples. Fresh-scene performance is synthetic same-distribution testing, not real-cell or instrument validation. The noise seed is recorded per case. Threshold 0.5 is fixed in advance; the threshold sweep is exploratory and does not select a replacement operating point. Point matching uses unique projected in-focus centers, not independently depth-resolved beads.','',
       'Camera images are the exact tensors passed to the detector, converted back to ADU. Display contrast is explicitly stretched; raw arrays are in the case NPZ files. Camera auto-exposure remains enabled (target saturation ratio 0.01, robust percentile 0.999). The model is a bead detector; no cell annotations are available. The PSF-safe y range is only six central rows in these settings.','',
       'Beam profiles follow the trained sample_4f route and pad=True propagation. Raw scan-integrated intensities and physical axes are in beam_profiles.npz. The z=0 index follows the renderer at N//2−1. The flat reference uses the same optical model/input amplitude with zero phase, without running an untrained reference detector. Independently peak-normalized display does not compare absolute incident power or establish fixed-dose benefits.','']
    for name in ['fresh_case0','saved_validation_case0','confusion_matrices','precision_recall','beam_profiles','loss_history']:lines+=['![%s](%s.png)'%(name,name),'']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    html='<!doctype html><meta charset="utf-8"><title>Trained Gaussian evaluation</title><style>body{font:16px system-ui;margin:30px;max-width:1500px}img{max-width:100%%}pre{white-space:pre-wrap}</style><h1>Trained Gaussian bead detection</h1><p>Final epoch %d; fixed threshold 0.5. Fresh and reused validation scenes reported separately. Camera auto-exposure enabled; no fixed-dose claim.</p><pre>%s</pre>'%(meta['completed_epochs'],json.dumps(summary,indent=2))
    html+=''.join('<img loading="lazy" src="%s.png">'%n for n in ['fresh_case0','saved_validation_case0','confusion_matrices','precision_recall','beam_profiles','loss_history'])
    (output/'index.html').write_text(html)
    write_json(output/'SUCCESS.json',dict(success=True,checkpoint_sha256=meta['checkpoint_sha256'],datasets=summary))
    print('EVALUATION_RENDERED',json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=['capture','render','all']);p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=16);p.add_argument('--seed',type=int,default=1729);a=p.parse_args()
    if a.mode in ['capture','all']:
        if a.checkpoint is None:p.error('--checkpoint required for capture')
        capture(a.checkpoint,a.output,a.count,a.seed)
    if a.mode in ['render','all']:render(a.output)
