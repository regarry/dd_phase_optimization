"""Seeded diagnostics for initialized, untrained reference networks.

capture: Slurm GPU allocation only; reconstruct initialized network and phase.
render: CPU plots from captured arrays. Does not train or resume any model.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys

import numpy as np


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def confusion(probability, target, threshold=.5):
    pred = probability >= threshold
    truth = target.astype(bool)
    return np.array([[np.sum(~truth & ~pred), np.sum(~truth & pred)],
                     [np.sum(truth & ~pred), np.sum(truth & pred)]], dtype=np.int64)


def metrics(matrix):
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, precision=tp/max(1,tp+fp),
                recall=tp/max(1,tp+fn), dice=2*tp/max(1,2*tp+fp+fn),
                specificity=tn/max(1,tn+fp), accuracy=(tp+tn)/max(1,tn+fp+fn+tp),
                prevalence=(tp+fn)/max(1,tn+fp+fn+tp))


def bead_categories(xyz, probability, cfg):
    x = xyz[:,0].astype(int); y = xyz[:,1].astype(int)
    focus = (xyz[:,2] >= cfg['z_range_cost_function'][0]) & (xyz[:,2] <= cfg['z_range_cost_function'][1])
    positive = probability[y,x] >= .5
    # 0=correctly detected in-focus; 1=missed in-focus; 2=out-of-focus, negative
    # projection; 3=out-of-focus, positive projection (ambiguous, not 3D attribution).
    return np.where(focus, np.where(positive,0,1), np.where(positive,3,2))


def capture(sweep, output, count):
    import torch
    import yaml
    from data.datasets import SyntheticMicroscopeData
    from models.wrappers import ParallelEndToEndModel
    from physics.masks import get_initial_phase_mask
    from physics.phase_io import quantize_phase

    if not 1 <= torch.cuda.device_count() <= 2:
        raise RuntimeError('Diagnostic capture requires one or two allocated GPUs')
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'trained': False, 'purpose':'Untrained initialization reference; not a trained-model result',
                'common_scene_seed':1729, 'scenes_per_model':count,
                'camera_noise_seed_base':101729, 'threshold':.5,
                'python':sys.version, 'torch':torch.__version__, 'cuda':torch.version.cuda,
                'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                'note':'Inference only; emitter sums may have small device-count rounding differences', 'models':{}}
    reference = None
    for name in ('random','gaussian','axicon'):
        run = next((sweep/name/'outputs').glob('*/config.yaml')).parent
        cfg = yaml.safe_load((run/'config.yaml').read_text())
        with (run/'labels.pickle').open('rb') as f:
            saved = pickle.load(f)
        # Identical fresh scenes for a meaningful initialization comparison.
        if reference is None:
            random.seed(1729); np.random.seed(1729)
            ds = SyntheticMicroscopeData(count, cfg)
            reference = [(xyz.numpy(),target.numpy()) for xyz,target in (ds[i] for i in range(count))]
            np.savez_compressed(output/'common_scenes.npz',
                xyz=np.stack([v[0] for v in reference]), target=np.stack([v[1] for v in reference]))
        seed = cfg['random_seed']
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = ParallelEndToEndModel(cfg).eval()
        phase = torch.from_numpy(get_initial_phase_mask(cfg)).to('cuda:0')
        phase = quantize_phase(phase, cfg['slm_bit_depth'])
        dest = output/name; dest.mkdir(exist_ok=True)
        np.save(dest/'initialized_phase_radians.npy',phase.cpu().numpy())
        cases = [('common',i,xyz,target) for i,(xyz,target) in enumerate(reference)]
        cases += [('saved_validation',i,np.asarray(saved[i]['xyz'],np.float32),
                   np.asarray(saved[i]['target'],np.float32)) for i in sorted(saved)[:4]]
        for dataset,i,xyz,target in cases:
            with torch.no_grad():
                torch.manual_seed(101729+i+(1000 if dataset=='saved_validation' else 0))
                # Capture exactly the camera tensor supplied to the detector.
                camera = model.physics(phase, torch.from_numpy(xyz[None]).float().to('cuda:0'))
                logits = model.unet(camera)
                if not torch.isfinite(logits).all():
                    raise RuntimeError('Nonfinite diagnostic logits')
                probability = logits.sigmoid().squeeze().cpu().numpy()
                camera_adu = camera.squeeze().cpu().numpy()*cfg['camera_max_adu']
            np.savez_compressed(dest/('%s_%03d.npz'%(dataset,i)), xyz=xyz,
                    target=target.squeeze(), probability=probability, camera_adu=camera_adu)
            print('UNTRAINED_CAPTURE',name,dataset,i,flush=True)
        metadata['models'][name] = dict(original_run=str(run), saved_validation_scenes=len(saved),
                 configuration=cfg, checkpoint_available=False,
                 detector_initialization_sha256=hashlib.sha256(b''.join(
                     t.detach().cpu().numpy().tobytes() for t in model.unet.state_dict().values())).hexdigest())
        del model,phase
        torch.cuda.empty_cache()
    dump(output/'capture_metadata.json',metadata)


def interactive_cloud(path, xyz, categories, px):
    """Self-contained rotatable 3D canvas; no network/CDN dependency."""
    payload = json.dumps({'xyz':(xyz*px*1e6).tolist(),'category':categories.tolist()})
    html = '''<!doctype html><meta charset="utf-8"><title>Untrained diagnostic bead sample</title>
<style>body{font:16px system-ui;background:#111827;color:#f3f4f6;margin:24px}canvas{width:100%;height:70vh;background:#0b1020;border:1px solid #475569}label{margin-right:18px}h1{font-size:22px}.note{max-width:1000px}</style>
<h1>UNTRAINED INITIALIZATION REFERENCE</h1>
<p class="note">Drag to rotate; wheel to zoom; hover a bead for its coordinates. Axis ranges are independently scaled for inspection; coordinates are micrometers. The network predicts a 2D in-focus center map, not bead depth.</p>
<p><label><input id="off" type="checkbox" checked>Show out-of-focus beads</label><label><input id="scale" type="checkbox" checked>Equalize axis scales</label><button id="reset">Reset view</button></p>
<p><span style="color:#22c55e">● in-focus detected</span> &nbsp;<span style="color:#ef4444">● in-focus missed</span> &nbsp;<span style="color:#94a3b8">● out-of-focus / negative projection</span> &nbsp;<span style="color:#f59e0b">● out-of-focus / positive projection (ambiguous)</span></p>
<canvas id="c"></canvas><p id="tip">Hover a bead</p>
<p>Orange is a projection association, not proof that this bead caused a false positive. False-positive background pixels have no known 3D bead location. All probabilities come from the seeded, untrained detector.</p>
<script>const d=PAYLOAD;const canvas=document.querySelector('#c'),ctx=canvas.getContext('2d');let yaw=.7,pitch=.4,zoom=1,drag=false,last=[],projected=[];
const colors=['#22c55e','#ef4444','#94a3b8','#f59e0b'];const labels=['in-focus detected','in-focus missed','out-of-focus / negative projection','out-of-focus / positive projection (ambiguous)'];
const lo=[0,1,2].map(k=>Math.min(...d.xyz.map(p=>p[k]))),hi=[0,1,2].map(k=>Math.max(...d.xyz.map(p=>p[k]))),span=hi.map((v,k)=>Math.max(v-lo[k],1));
function project(p){let a=p.map((v,k)=>(v-(lo[k]+hi[k])/2)/(document.querySelector('#scale').checked?span[k]:Math.max(...span)));let x=a[0]*Math.cos(yaw)-a[2]*Math.sin(yaw),z=a[0]*Math.sin(yaw)+a[2]*Math.cos(yaw),y=a[1]*Math.cos(pitch)-z*Math.sin(pitch);return[canvas.width/2+x*canvas.height*.8*zoom,canvas.height/2-y*canvas.height*.8*zoom,z]}
function draw(){canvas.width=canvas.clientWidth*devicePixelRatio;canvas.height=canvas.clientHeight*devicePixelRatio;ctx.clearRect(0,0,canvas.width,canvas.height);projected=[];const points=d.xyz.map((p,i)=>({p:project(p),i})).sort((a,b)=>a.p[2]-b.p[2]);for(const {p,i} of points){let k=d.category[i];if(k>1&&!document.querySelector('#off').checked)continue;ctx.globalAlpha=k===2?.25:.95;ctx.fillStyle=colors[k];ctx.beginPath();ctx.arc(p[0],p[1],(k<2?4.5:2.4)*devicePixelRatio,0,7);ctx.fill();projected.push([p[0],p[1],i]);}ctx.globalAlpha=1;ctx.fillStyle='#e5e7eb';ctx.font=(14*devicePixelRatio)+'px sans-serif';ctx.fillText('x: propagation / camera columns; y: scan / camera rows; z: detection depth',20*devicePixelRatio,28*devicePixelRatio);ctx.fillText('Range (µm): x '+lo[0].toFixed(1)+'…'+hi[0].toFixed(1)+'; y '+lo[1].toFixed(1)+'…'+hi[1].toFixed(1)+'; z '+lo[2].toFixed(1)+'…'+hi[2].toFixed(1),20*devicePixelRatio,50*devicePixelRatio)}
canvas.onpointerdown=e=>{drag=true;last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointerup=()=>drag=false;canvas.onpointermove=e=>{if(drag){yaw+=(e.clientX-last[0])*.008;pitch+=(e.clientY-last[1])*.008;last=[e.clientX,e.clientY];draw();return;}let r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)*devicePixelRatio,y=(e.clientY-r.top)*devicePixelRatio,best=null,dist=100*devicePixelRatio**2;for(const p of projected){let s=(p[0]-x)**2+(p[1]-y)**2;if(s<dist){dist=s;best=p[2]}}if(best!==null)document.querySelector('#tip').textContent='Bead '+best+': ('+d.xyz[best].map(v=>v.toFixed(2)).join(', ')+') µm — '+labels[d.category[best]]};canvas.onwheel=e=>{e.preventDefault();zoom*=Math.exp(-e.deltaY*.001);draw()};document.querySelector('#off').onchange=draw;document.querySelector('#scale').onchange=draw;document.querySelector('#reset').onclick=()=>{yaw=.7;pitch=.4;zoom=1;draw()};window.onresize=draw;draw();</script>'''
    path.write_text(html.replace('PAYLOAD',payload))


def render(sweep, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LogNorm
    from matplotlib.patches import Patch
    meta = json.loads((output/'capture_metadata.json').read_text())
    summary = {}
    colors = ['#22c55e','#ef4444','#94a3b8','#f59e0b']
    names = ('random','gaussian','axicon')
    matrix_fig, matrix_axes = plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    pr_fig, pr_ax = plt.subplots(figsize=(7,5),constrained_layout=True)
    hist_fig, hist_axes = plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for column,name in enumerate(names):
        dest = output/name; cfg = meta['models'][name]['configuration']
        counts=np.zeros((2,2),dtype=np.int64); hist=np.zeros((2,512),dtype=np.int64)
        for file in sorted(dest.glob('common_*.npz')):
            sample=np.load(file); p=sample['probability']; t=sample['target'].astype(bool)
            counts+=confusion(p,t)
            hist[0]+=np.histogram(p[~t],bins=512,range=(0,1))[0]
            hist[1]+=np.histogram(p[t],bins=512,range=(0,1))[0]
        result=metrics(counts); summary[name]=result
        dump(dest/'untrained_metrics.json',dict(**result,scenes=meta['scenes_per_model'],trained=False,
             dataset='common fresh scenes; untrained diagnostic only',threshold=.5))
        normalized=counts/np.maximum(counts.sum(axis=1,keepdims=True),1)
        for row,values in enumerate((counts,normalized)):
            ax=matrix_axes[row,column]
            im=ax.imshow(values,cmap='Blues',norm=LogNorm(vmin=1,vmax=max(2,int(counts.max())))) if row==0 else ax.imshow(values,vmin=0,vmax=1,cmap='Blues')
            for y in range(2):
                for x in range(2): ax.text(x,y,format(values[y,x],',d') if row==0 else '%.2f%%'%(values[y,x]*100),ha='center',va='center',color='white' if values[y,x]>(counts.max()*.15 if row==0 else .5) else 'black')
            ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Background','In-focus center'],yticklabels=['Background','In-focus center'],xlabel='Predicted',ylabel='True',title=name+(' counts (log colors)' if row==0 else ' row-normalized'))
        tp=np.cumsum(hist[1,::-1]);fp=np.cumsum(hist[0,::-1]);recall=tp/max(1,hist[1].sum());precision=tp/np.maximum(1,tp+fp)
        pr_ax.plot(recall,precision,label=name)
        bins=(np.arange(512)+.5)/512
        for k,label in enumerate(('Background','In-focus center')):
            hist_axes[column].plot(bins,hist[k]/max(1,hist[k].sum()),label=label)
        hist_axes[column].set(title=name,xlabel='Untrained predicted probability',ylabel='Fraction within class',yscale='log');hist_axes[column].legend()
        sample=np.load(dest/'common_000.npz')
        p,t,camera,xyz=sample['probability'],sample['target'],sample['camera_adu'],sample['xyz']
        pred=p>=.5;truth=t.astype(bool)
        error=np.zeros(t.shape,dtype=np.uint8);error[truth&pred]=1;error[truth&~pred]=2;error[~truth&pred]=3
        fig,axes=plt.subplots(5,1,figsize=(15,10),constrained_layout=True)
        for ax,img,title,cmap in zip(axes,[camera,t,p,pred,error],['Actual detector input: simulated camera (ADU)','In-focus bead-center truth','UNTRAINED detector probability','Thresholded untrained prediction (0.5)','Error map: green TP, red FN, magenta FP'],['gray','gray','viridis','gray',ListedColormap(['#111827','#22c55e','#ef4444','#d946ef'])]):
            im=ax.imshow(img,aspect='auto',origin='lower',cmap=cmap,vmin=0,vmax=3 if ax is axes[-1] else (1 if ax is not axes[0] else max(1,camera.max())))
            ax.set(title=title,xlabel='Camera column x (pixels)',ylabel='Row y');fig.colorbar(im,ax=ax,shrink=.7)
        fig.suptitle(name+' — UNTRAINED initialization reference',fontsize=14)
        fig.savefig(dest/'camera_truth_inference.png',dpi=170);plt.close(fig)
        # Full field has a very large aspect ratio; include a readable crop with
        # the most in-focus centers, selected using truth only, not predictions.
        start=max(0,min(t.shape[1]-128,int(np.argmax(np.convolve(t.sum(axis=0),np.ones(128),'valid')))))
        fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        for ax,img,title in zip(axes,[camera,t,p],['Camera ADU','Target centers','Untrained probability']):
            im=ax.imshow(img[:,start:start+128],origin='lower',aspect='equal',
                         cmap='gray' if title!='Untrained probability' else 'viridis',
                         vmin=0,vmax=max(1,float(camera.max())) if title=='Camera ADU' else 1)
            ax.set(title=title,xlabel='Crop x (pixels)',ylabel='y (pixels)')
            fig.colorbar(im,ax=ax,shrink=.7)
        fig.suptitle(name+' truth-selected crop, columns %d–%d; UNTRAINED'%(start,start+127));fig.savefig(dest/'camera_crop.png',dpi=180);plt.close(fig)
        category=bead_categories(xyz,p,cfg)
        positions=xyz*cfg['px']*1e6
        fig=plt.figure(figsize=(12,7));ax=fig.add_subplot(111,projection='3d')
        labels=['In-focus detected','In-focus missed','Out-of-focus / negative projection','Out-of-focus / positive projection (ambiguous)']
        for k,label in enumerate(labels):
            select=category==k
            ax.scatter(*positions[select].T,s=20 if k<2 else 5,c=colors[k],alpha=.95 if k<2 else .25,label=label+' (%d)'%select.sum())
        ax.set(xlabel='x: propagation / columns (µm)',ylabel='y: scan / rows (µm)',zlabel='z: detection depth (µm)',title=name+' — 3D ground-truth sample, UNTRAINED 2D prediction association')
        ax.set_box_aspect((2,1,1));ax.legend(loc='upper left',fontsize=8)
        fig.text(.05,.02,'Axis scales differ for readability. Colors use predictions at projected x,y; this is not inferred 3D localization.',fontsize=10)
        fig.savefig(dest/'beads_3d.png',dpi=180,bbox_inches='tight');plt.close(fig)
        interactive_cloud(dest/'beads_3d.html',xyz,category,cfg['px'])
        np.savetxt(dest/'bead_performance.csv',np.column_stack([positions,category,p[xyz[:,1].astype(int),xyz[:,0].astype(int)]]),delimiter=',',header='x_um,y_um,z_um,category,projected_probability',comments='')
    matrix_fig.suptitle('UNTRAINED initialization-reference confusion matrices',fontsize=14)
    matrix_fig.savefig(output/'confusion_matrices.png',dpi=180);plt.close(matrix_fig)
    pr_ax.set(xlabel='Recall',ylabel='Precision',title='UNTRAINED precision–recall curves (512 probability bins)',xlim=(0,1),ylim=(0,1));pr_ax.legend();pr_ax.grid(alpha=.2)
    pr_fig.savefig(output/'precision_recall.png',dpi=180);plt.close(pr_fig)
    hist_fig.suptitle('UNTRAINED probability distributions');hist_fig.savefig(output/'probability_distributions.png',dpi=180);plt.close(hist_fig)
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    ax.text(.5,.55,'Initialization reference only\nTraining loss is not measured by this capture',ha='center',va='center',fontsize=18,transform=ax.transAxes)
    ax.text(.5,.23,'All three runs stalled before the first optimizer step: AF_UNIX path too long',ha='center',va='center',transform=ax.transAxes)
    ax.set(xlabel='Completed epoch',ylabel='Loss (unavailable)',title='Initialization reference — training history not evaluated');ax.set_xticks([]);ax.set_yticks([])
    fig.savefig(output/'loss_unavailable.png',dpi=180);plt.close(fig)
    dump(output/'untrained_summary.json',summary)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('action',choices=['capture','render'])
    p.add_argument('--sweep',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=16)
    args=p.parse_args()
    if args.action=='capture':capture(args.sweep.resolve(),args.output.resolve(),args.count)
    else:render(args.sweep.resolve(),args.output.resolve())


if __name__=='__main__':main()
