"""Cross-evaluate saved detector/phase pairs on identical confined bead volumes.

This matches inference geometry, not training history; no model is retrained.
"""
import argparse,json,os,random
from pathlib import Path
import numpy as np
import torch
from data.datasets import SyntheticMicroscopeData
from models.wrappers import ParallelEndToEndModel
from physics.phase_io import quantize_phase
from training.checkpoint import read_checkpoint
from scripts.evaluate_tissue_study import confusion,class_scores


def capture(plan_path):
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count()!=1:raise RuntimeError('One allocated GPU required')
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8'));out=Path(plan['output'])/'matched_volume';out.mkdir(exist_ok=True)
    entries=[e for e in plan['runs'] if e['name'].startswith(('two_sided_','dice_bce_'))]
    reference=next(e for e in entries if e['name']=='two_sided_gaussian')
    cfg=dict(read_checkpoint(reference['checkpoint'])['config']);cfg['SpatialMulitWellLoss']=False
    random.seed(773211);np.random.seed(773211);torch.manual_seed(773211)
    dataset=SyntheticMicroscopeData(4,cfg);cases=[dataset[i] for i in range(4)];results=[]
    for entry in entries:
        state=read_checkpoint(entry['checkpoint']);model=ParallelEndToEndModel(cfg).eval()
        weights={k[len('unet.'):]:v for k,v in state['model'].items() if k.startswith('unet.')}
        model.unet.load_state_dict(weights,strict=True);phase=quantize_phase(state['raw_phase'].to('cuda:0'),cfg.get('slm_bit_depth',8));matrix=np.zeros((2,2),dtype=np.int64)
        with torch.no_grad():
            for i,(xyz,target) in enumerate(cases):
                torch.manual_seed(775500+i);torch.cuda.manual_seed_all(775500+i)
                camera=model.physics(phase,xyz[None].to('cuda:0'));prob=model.unet(camera).sigmoid().squeeze().cpu().numpy();truth=np.asarray(target).squeeze().astype(np.int64)
                matrix+=confusion(truth,(prob>=.5).astype(np.int64),2)
                np.savez_compressed(out/('%s_%02d.npz'%(entry['name'],i)),xyz=xyz.numpy(),target=truth,probability=prob,camera_adu=camera.squeeze().cpu().numpy()*cfg['camera_max_adu'])
        results.append(dict(run=entry['name'],completed_epochs=state['epoch']+1,classes=class_scores(matrix),confusion=matrix.tolist(),scenes=4))
        del model,state,phase;torch.cuda.empty_cache();print('MATCHED_EVALUATION',entry['name'],flush=True)
    (out/'metrics.json').write_text(json.dumps(dict(results=results,geometry=cfg,seed=773211,noise_seeds=[775500+i for i in range(4)],note='Identical 1200-emitter volumes with z centres [-50,50]um; common PSF/camera/generator and paired noise seeds. Original detector weights and phase; historical baselines had different training distributions/budgets. Auto-exposure on, not fixed dose or causal objective ablation.'),indent=2)+'\n')


def render(plan_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8'));root=Path(plan['output']);out=root/'matched_volume';meta=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    names=['two_sided_gaussian','dice_bce_gaussian','two_sided_axicon','dice_bce_axicon'];fig,axes=plt.subplots(2,2,figsize=(14,8))
    for name,ax in zip(names,axes.ravel()):
        entry=next(e for e in plan['runs'] if e['name']==name)
        with np.load(Path(entry['output'])/'beam_profiles.npz') as b:
            p=b['learned'];x=b['x_um'];z=b['z_um'];ax.imshow((p/p.max()).T,origin='lower',extent=[x[0],x[-1],z[0],z[-1]],cmap='inferno',vmin=0,vmax=1,aspect='auto')
        ax.axhspan(-50,50,fill=False,edgecolor='cyan',hatch='//',alpha=.3)
        ax.set(title=name,xlabel='Propagation x (µm)',ylabel='Thickness z (µm)',ylim=(-150,150))
    fig.suptitle('Two-sided + detection objective versus detection-only baselines\nSame inference bead volume within ±50 µm; each learned beam normalized independently; training budgets/distributions differ');fig.tight_layout();fig.savefig(out/'beam_comparison.png',dpi=150);plt.close(fig)
    html='<h2>Matched confined-volume inference</h2><p>'+meta['note']+'</p><img src="matched_volume/beam_comparison.png"><table border=1 cellpadding=5><tr><th>Model</th><th>Epochs</th><th>Pixel Dice</th><th>Precision</th><th>Recall</th><th>Global OS63 (µm)</th><th>Sample-restricted OS63 (µm)</th></tr>'
    for r in meta['results']:
        s=r['classes'][1]
        from utils.beam_metrics import power_interval
        entry=next(e for e in plan['runs'] if e['name']==r['run'])
        optical=json.loads((Path(entry['output'])/'optical_metrics.json').read_text(encoding='utf-8'))['learned']
        with np.load(Path(entry['output'])/'beam_profiles.npz') as b:
            z=b['z_um'];j=int(np.argmin(abs(b['x_um'])));profile=b['learned'][j];support=(z>=-50)&(z<=50)
            restricted=power_interval(z,profile*support)['width_um']
        html+='<tr><td>%s</td><td>%d</td><td>%.4f</td><td>%.4f</td><td>%.4f</td><td>%.2f</td><td>%.2f</td></tr>'%(r['run'],r['completed_epochs'],s['f1'] or 0,s['precision'] or 0,s['recall'] or 0,optical['optical_sectioning_63_um'],restricted)
    html+='</table>';(out/'report_fragment.html').write_text(html,encoding='utf-8')
    index=root/'index.html';text=index.read_text(encoding='utf-8');text=text.replace('<h1>Deep Design checkpoint comparisons</h1>','<h1>Deep Design checkpoint comparisons</h1>'+html,1);index.write_text(text,encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['capture','render','all']);p.add_argument('--plan',required=True);a=p.parse_args()
    if a.mode in ('capture','all'):capture(a.plan)
    if a.mode in ('render','all'):render(a.plan)
