"""Render archive-review figures from saved TIFFs only; no model execution."""
import csv
import hashlib
import html
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy.signal import find_peaks

from review_training_archive import config, images, counts_metrics

OUT=None
ROWS=[]
RUNS={}
SELECTED=[]
plt.rcParams.update({'font.size':9,'figure.dpi':120,'axes.spines.top':False,
                     'axes.spines.right':False,'savefig.bbox':'tight'})


def save(fig,name):
    fig.savefig(OUT/(name+'.png'));plt.close(fig)


def relative(path):
    return os.path.relpath(path,OUT)


def evaluation(run):
    return next((e for e in run['evaluations'] if e['path']==run['best_saved_inference']),None)


def profile(path):
    path=Path(path);a=tifffile.imread(path).astype(float).squeeze()
    cfg=config(path.parent/'config.yaml');px=float(cfg.get('px',1e-6));pxmm=px*1000
    start=int(float(cfg.get('z_min_mm',-1))/pxmm)
    stop=int(float(cfg.get('z_max_mm',1))/pxmm)
    step=max(1,int((stop-start)/(int(cfg.get('num_z_steps',200))-1)))
    propagation=np.array(list(range(start,stop,step)))*px*1e6
    lo=int(float(cfg.get('y_min_mm',-.3))/pxmm)
    hi=int(float(cfg.get('y_max_mm',.3))/pxmm)
    thickness=np.arange(lo,hi)*px*1e6
    verified=(a.ndim==2 and a.shape==(len(propagation),len(thickness)))
    if not verified:
        propagation=np.arange(a.shape[0]);thickness=np.arange(a.shape[1])
    center=np.argmin(abs(propagation));line=a[center];peak=int(np.argmax(line))
    left=right=peak
    while left>0 and line[left-1]>=line[peak]/2:left-=1
    while right<len(line)-1 and line[right+1]>=line[peak]/2:right+=1
    width=(right-left+1)*(px*1e6 if verified else 1)
    central=abs(thickness)<=5 if verified else np.zeros(len(line),bool)
    peaks,_=find_peaks(line,height=line.max()*.1,prominence=line.max()*.05)
    return a,propagation,thickness,dict(path=str(path),axis_shape_verified=verified,
        central_peak_fwhm_sampled_um=width if verified else None,
        focus_normalized_profile_fraction_within_5um=float(line[central].sum()/line.sum()) if verified and line.sum() else None,
        focus_peak_count_10percent=int(len(peaks)),normalization='archived profiler intensity normalization; not incident-power calibrated')


def main_profiles(run,e):
    # Select the same timestamp evaluation as inference, including its controls.
    root=Path(e['path']).parent if e else None
    if root and root.name=='inference':root=root.parent
    primary=[Path(p) for p in run['profile_files'] if Path(p).parent.name=='beam_profile'
             and 'axicon_beam_profile' not in p and 'fresnel_lens_beam_profile' not in p]
    chosen=next((p for p in primary if root and root in p.parents),primary[-1] if primary else None)
    if chosen is None:return []
    parent=chosen.parent.parent
    return [(label,p) for label,p in [('learned',chosen),
        ('axicon control',parent/'axicon_beam_profile/beam_profile/column_sums.tiff'),
        ('Fresnel control',parent/'fresnel_lens_beam_profile/beam_profile/column_sums.tiff')] if p.exists()]


def diagnostics(run,e):
    fig,axs=plt.subplots(2,3,figsize=(16,8));fig.suptitle(run['id']+' | '+run['initialization']+' | saved inference (selected evaluation)')
    h=run['history']
    for key,color in [('train','#2065b0'),('validation','#df7624')]:
        if h[key]:axs[0,0].plot(np.arange(1,len(h[key])+1),h[key],label=key,color=color)
    axs[0,0].set(title='Recorded loss (within this run)',xlabel='Completed epoch',ylabel='Objective value')
    axs[0,0].set_yscale('symlog',linthresh=1e-4);axs[0,0].legend()
    if e and e['cases']:
        # Fixed case 0, never cherry-pick a pretty frame.
        c=sorted(e['cases'],key=lambda c:c['index'])[0]
        dec=images(tifffile.imread(c['prediction']),tifffile.imread(c['target']))
        cp=Path(c['prediction']).parent/('camera_image_'+str(c['index'])+'.tif')
        if cp.exists():
            camera=tifffile.imread(cp).squeeze();v=np.percentile(camera,[1,99.8])
            axs[0,1].imshow(camera,cmap='gray',vmin=v[0],vmax=v[1],aspect='auto')
            axs[0,1].set_title('Saved camera, case 0\n1–99.8% display stretch; auto-exposure')
        else:axs[0,1].text(.1,.5,'Camera file absent',transform=axs[0,1].transAxes)
        axs[0,2].imshow(dec['truth'],cmap='gray',aspect='auto');axs[0,2].set_title('Bead target, case 0 (not all emitters)')
        axs[1,0].imshow(dec['probability'],cmap='magma',vmin=0,vmax=1,aspect='auto');axs[1,0].set_title('Bead probability, case 0')
        overlay=np.zeros(dec['truth'].shape+(3,));overlay[dec['truth']&dec['pred']]=[0,.8,.3]
        overlay[dec['truth']&~dec['pred']]=[1,.2,.2];overlay[~dec['truth']&dec['pred']]=[.2,.6,1]
        axs[1,1].imshow(overlay,aspect='auto');axs[1,1].set_title('Case 0: green TP / red FN / blue FP')
        m=e['recomputed'];matrix=np.array([[m['tn'],m['fp']],[m['fn'],m['tp']]])
        axs[1,2].imshow(np.log10(matrix+1),cmap='Blues')
        for i in range(2):
            for j in range(2):axs[1,2].text(j,i,str(matrix[i,j]),ha='center',va='center',color='red')
        axs[1,2].set(xticks=[0,1],yticks=[0,1],xticklabels=['background','bead'],yticklabels=['background','bead'],
            xlabel='Predicted',ylabel='True',title='Pooled pixel confusion, %d scenes\nP %.3f | R %.3f | F1 %.3f'%(e['paired_cases'],m['precision'] or 0,m['recall'] or 0,m['f1'] or 0))
    else:
        for ax in axs.ravel()[1:]:ax.text(.1,.5,'No paired saved inference',transform=ax.transAxes)
    for ax in [axs[0,1],axs[0,2],axs[1,0],axs[1,1]]:ax.set(xlabel='Camera x pixel',ylabel='Camera y pixel')
    fig.tight_layout(rect=[0,0,1,.96]);save(fig,run['id']+'_diagnostics')


def beams(run,e):
    paths=main_profiles(run,e)
    if not paths:return []
    fig,axs=plt.subplots(2,len(paths),figsize=(6*len(paths),7),squeeze=False)
    results=[]
    for j,(label,path) in enumerate(paths):
        a,p,t,m=profile(path);results.append(dict(run=run['id'],label=label,**m))
        normalized=a/a.max() if a.max() else a
        im=axs[0,j].imshow(normalized.T,origin='lower',aspect='auto',extent=[p[0],p[-1],t[0],t[-1]],vmin=0,vmax=1,cmap='inferno')
        axs[0,j].set(title=label+' | independently normalized',xlabel='Propagation distance (µm)' if m['axis_shape_verified'] else 'Propagation index',ylabel='Sheet thickness coordinate (µm)' if m['axis_shape_verified'] else 'Thickness index')
        if m['axis_shape_verified']:axs[0,j].set_ylim(-100,100)
        fig.colorbar(im,ax=axs[0,j],label='Relative intensity')
        for loc in [-400,0,400]:
            i=np.argmin(abs(p-loc));line=a[i];axs[1,j].plot(t,line/(line.max() or 1),label='prop %.0f'%p[i])
        axs[1,j].set(xlabel='Sheet thickness (µm)' if m['axis_shape_verified'] else 'Thickness index',ylabel='Row-normalized intensity',title='Transverse profiles; no absolute-power comparison')
        if m['axis_shape_verified']:axs[1,j].set_xlim(-80,80)
        axs[1,j].legend()
    fig.suptitle(run['id']+' | archived beam profiles | scan coordinate integrated out');fig.tight_layout(rect=[0,0,1,.96]);save(fig,run['id']+'_beams')
    return results


def cohort(ids,title,name):
    fig,axs=plt.subplots(1,3,figsize=(16,4.5))
    for id in ids:
        run=RUNS[id];e=evaluation(run);name_label=run['initialization']+' '+id[:8]
        if e:
            m=e['recomputed'];axs[0].scatter(m['recall'],m['precision'],s=80,label=name_label)
            axs[0].annotate(run['initialization'],(m['recall'],m['precision']),xytext=(4,4),textcoords='offset points')
        vals=run['history']['validation'];axs[1].plot(np.arange(1,len(vals)+1),vals,label=name_label)
        pp=main_profiles(run,e)
        if pp:
            a,p,t,m=profile(pp[0][1]);line=a[np.argmin(abs(p))];axs[2].plot(t,line/(line.max() or 1),label=name_label)
    axs[0].set(xlabel='Pooled pixel recall',ylabel='Pooled pixel precision',xlim=(0,1.02),ylim=(0,1.02))
    axs[1].set(xlabel='Completed epoch',ylabel='Validation loss');axs[1].set_yscale('symlog',linthresh=1e-4)
    axs[2].set(xlabel='Sheet thickness coordinate (µm)',ylabel='Normalized focus profile',xlim=(-80,80));axs[2].legend(fontsize=7)
    fig.suptitle(title);fig.tight_layout(rect=[0,0,1,.93]);save(fig,name)


def main(review,selected,cohorts):
    global OUT,ROWS,RUNS,SELECTED
    OUT=review.resolve()
    ROWS=json.loads((OUT/'inventory.json').read_text())
    RUNS={r['id']:r for r in ROWS}
    SELECTED=selected
    if not all(id in RUNS for id in selected):raise ValueError('Unknown run ID')
    months=sorted(set(r['id'][:6] for r in ROWS));fig,axs=plt.subplots(1,2,figsize=(13,4.5))
    axs[0].bar(months,[sum(r['id'][:6]==m for r in ROWS) for m in months],label='Run records')
    axs[0].bar(months,[sum(r['id'][:6]==m and r['epochs']>0 for r in ROWS) for m in months],label='Recorded completed epochs')
    axs[0].tick_params(axis='x',rotation=45);axs[0].legend();axs[0].set_ylabel('Runs')
    for task,color in [('binary_bead_centers','#176b45'),('soft_target_overlap','#cc842c'),('three_class_bead_centers','#a84674')]:
        rr=[r for r in ROWS if task in r['best_saved_task'] and r['best_saved_f1'] is not None]
        axs[1].scatter([r['best_saved_recall'] or 0 for r in rr],[r['best_saved_precision'] or 0 for r in rr],label=task,s=20,color=color,alpha=.65)
    axs[1].set(xlabel='Saved-evaluation recall',ylabel='Saved-evaluation precision',title='Selected best saved evaluation per run\nDifferent tasks/settings: not a leaderboard');axs[1].legend(fontsize=7)
    fig.tight_layout();save(fig,'archive_overview')
    metrics=[];summary=[]
    for id in SELECTED:
        run=RUNS[id];e=evaluation(run);diagnostics(run,e);metrics+=beams(run,e)
        cfg=run['config'];entry=dict(id=id,initialization=run['initialization'],epochs=run['epochs'],objective=run['objective'],config=cfg,
            evaluation=e['path'] if e else None,evaluation_epoch=e['epoch'] if e else None,
            metrics=e['recomputed'] if e else None,cases=e['paired_cases'] if e else 0,
            source_config=run['config_path'],target_hashes=[hashlib.sha256(Path(c['target']).read_bytes()).hexdigest() for c in e['cases']] if e else [])
        summary.append(entry);print('RENDERED',id,flush=True)
    (OUT/'selected_runs.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'beam_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    for i,ids in enumerate(cohorts):cohort(ids,'Selected cohort; matching must be checked in configuration and scene metadata','cohort_%d'%i)
    chunks=['<!doctype html><html><meta charset="utf-8"><title>Deep Design archive review</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:20px;background:#f6f7fa;color:#18202b}img{max-width:100%;background:white}article{margin:40px 0;border-top:2px solid #abc}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #bcc}pre{white-space:pre-wrap}a{color:#1657a0}select{font:inherit}</style><h1>Deep Design: Saved training archive review</h1><p>Read <a href="REPORT.md">REPORT.md</a> for conclusions, provenance and limitations. Saved inference only; no new training or GPU inference.</p><p>Scores are pixel bead-class metrics on reused validation scenes, not object localization or independent test performance. Historical camera auto-exposure and normalized profiles do not establish fixed-dose benefit.</p><img src="archive_overview.png"><p>Show run: <select id="filter"><option value="all">All selected runs</option>']
    chunks += ['<option>'+id+'</option>' for id in SELECTED]
    chunks += ['</select></p>']
    chunks += ['<img src="cohort_%d.png">'%i for i in range(len(cohorts))]
    for x in summary:
        id=x['id'];cfg=x['config'];chunks+=['<article data-run="'+id+'"><h2>'+id+' — '+html.escape(x['initialization'])+'</h2><p><a href="'+html.escape(relative(x['source_config']))+'">Original YAML</a> | <a href="'+html.escape(relative(RUNS[id]['path']))+'">Original run</a></p>']
        keys=['num_particles_range','bead_radius','image_volume','px','random_seed','ntrain','nvalid','batch_size','accumulation_steps','initial_learning_rate','freeze_unet','freeze_slm','num_classes','convolve_psf_ground_truth','quantum_efficiency','dark_current_mean','read_noise_std','target_saturation_ratio','robust_max_percentile','lens_approach','numerical_aperture','wavelength']
        chunks+=['<details><summary>Training parameters, evaluation and confusion counts</summary><pre>'+html.escape(json.dumps(dict(parameters={k:cfg.get(k,'unrecorded') for k in keys},objective=x['objective'],completed_epochs=x['epochs'],evaluation_epoch=x['evaluation_epoch'],evaluation=x['evaluation'],pooled_pixel_metrics=x['metrics']),indent=2))+'</pre></details>']
        chunks+=['<img loading="lazy" src="'+id+'_diagnostics.png">']
        if (OUT/(id+'_beams.png')).exists():chunks+=['<img loading="lazy" src="'+id+'_beams.png">']
        if (OUT/(id+'_sample3d.png')).exists():chunks+=['<img loading="lazy" src="'+id+'_sample3d.png">']
        chunks+=['</article>']
    chunks+=['<script>document.querySelector("#filter").onchange=function(){document.querySelectorAll("article").forEach(x=>x.hidden=this.value!=="all"&&x.dataset.run!==this.value)}</script></html>']
    (OUT/'index.html').write_text('\n'.join(chunks))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(__doc__);p.add_argument('--review',type=Path,required=True);p.add_argument('--runs',nargs='+',required=True);p.add_argument('--cohort',nargs='+',action='append',default=[]);a=p.parse_args();main(a.review,a.runs,a.cohort)
