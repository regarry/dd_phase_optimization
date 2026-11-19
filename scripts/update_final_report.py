"""Refresh saved report presentation and FOV metrics without optical inference."""
import argparse,json,re,shutil
from pathlib import Path
import numpy as np
from utils.beam_metrics import lobe_width,power_interval


def update(root):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import torch
 root=Path(root).resolve();plan=json.loads((root/'plan.json').read_text());index=root/'index.html'
 original=index.read_text(encoding='utf-8')
 if not (root/'index_before_widths.html').exists():shutil.copyfile(index,root/'index_before_widths.html')
 ref=next(e for e in plan['runs'] if e['name']=='one_sided_airy');refpath=Path(ref['output']);refcfg=json.loads((refpath/'capture.json').read_text())['configuration']
 # The analytic Airy field is independent of targets/detection geometry. Reuse
 # only after verifying the entire illumination configuration and axis grid.
 keys=['N','px','slm_px','phase_mask_pixel_size','phase_mask_upsample_factor','wavelength','refractive_index','lens_approach','angular_spectrum_method','lensless_prop_distance','fresnel_lens_focal_length','laser_beam_FWHC','aperature','aberration','c5','c13','lowpass_pinhole_radius_mm','focal_length_2','focal_length_3','focal_length_4','slm_bit_depth']
 allmetrics=[];sources=[]
 for entry in plan['runs']:
  out=Path(entry['output']);meta=json.loads((out/'capture.json').read_text());cfg=meta['configuration']
  with np.load(out/'beam_profiles.npz') as b:x=b['x_um'].copy();z=b['z_um'].copy();profiles={n:b[n].copy() for n in ['learned','gaussian','axicon']}
  ap=out/'airy_reference.npz'
  if not ap.exists():
   assert all(cfg.get(k)==refcfg.get(k) for k in keys),'Illumination differs; cannot reuse Airy: '+entry['name']
   shutil.copyfile(refpath/'airy_reference.npz',ap);shutil.copyfile(refpath/'airy_reference.json',out/'airy_reference.json');sources.append(dict(run=entry['name'],reference_source=ref['name'],verified_keys=keys))
  with np.load(ap) as b:
   np.testing.assert_array_equal(x,b['x_um']);np.testing.assert_array_equal(z,b['z_um']);profiles['airy']=b['airy'].copy()
  rows=[]
  for beam,p in profiles.items():
   for location,j in [('left edge',0),('center',int(np.argmin(abs(x)))),('right edge',len(x)-1)]:
    row=p[j];peak=int(row.argmax());interval=power_interval(z,row)
    rows.append(dict(beam=beam,location=location,x_um=float(x[j]),main_lobe_1e_um=lobe_width(z,row,peak),peak_z_um=float(z[peak]),optical_sectioning_63_um=interval['width_um']))
  (out/'fov_widths.json').write_text(json.dumps(rows,indent=2)+'\n');allmetrics.append(dict(run=entry['name'],measurements=rows))
  table='<h3>Main-lobe thickness and optical sectioning across the FOV</h3><p>Scan-y integrated intensity; first1/e crossings around the strongest lobe and shortest interval carrying1−1/e power. Edges are the actual sampled FOV boundaries, not the OS-doubling positions used to define propagation length in Remacha et al. Widths are illumination-only, finite-grid measurements; peak z identifies displaced lobes.</p><table class="fov-widths"><thead><tr><th>Beam</th><th>Location</th><th>x (µm)</th><th>Main lobe1/e (µm)</th><th>OS63% (µm)</th><th>Peak z (µm)</th></tr></thead><tbody>'
  f=lambda v:'unresolved' if v is None else '%.2f'%v
  for r in rows:table+='<tr>'+''.join('<td>'+str(v)+'</td>' for v in [r['beam'],r['location'],f(r['x_um']),f(r['main_lobe_1e_um']),f(r['optical_sectioning_63_um']),f(r['peak_z_um'])])+'</tr>'
  table+='</tbody></table>'
  # Preserve all other original sections; replace only this run's content block.
  pattern=r'(<section><h2>'+re.escape(entry['name'])+r'</h2>)(.*?)(</section>)';match=re.search(pattern,original,re.S)
  assert match,entry['name'];body=match.group(2)
  body=re.sub(r'<h3>Main-lobe thickness and optical sectioning across the FOV</h3>.*?</tbody></table>','',body,flags=re.S)
  body=re.sub(r'<img[^>]*src="'+re.escape(entry['name'])+r'/camera_diagnostics.png"[^>]*>','',body)
  # New camera panels contain only measured synthetic detector input.
  case=next(c for c in meta['cases'] if c['dataset']=='fresh' and c['index']==0)
  with np.load(out/case['file']) as d:camera=d['camera_adu'];truth=d['target']
  points=np.argwhere(truth==1);cy,cx=points[0] if len(points) else np.array(camera.shape)//2
  y0=max(0,int(cy)-24);x0=max(0,int(cx)-24);crop=camera[y0:min(y0+49,camera.shape[0]),x0:min(x0+49,camera.shape[1])]
  fig,axes=plt.subplots(2,1,figsize=(13,6));hi=max(1,float(np.quantile(camera,.998)))
  for ax,image,title in [(axes[0],camera,'Synthetic camera input | simulated ADU; display0–99.8th percentile'),(axes[1],crop,'Local camera crop | equal pixel aspect')]:
   im=ax.imshow(image,cmap='gray',vmin=0,vmax=hi,aspect='equal');ax.set_title(title);ax.set(xlabel='Camera x pixel',ylabel='Camera y pixel');fig.colorbar(im,ax=ax,label='ADU')
  fig.suptitle(entry['name']);fig.tight_layout();fig.savefig(out/'camera_diagnostics.png',dpi=150);plt.close(fig)
  state=torch.load(entry['checkpoint'],map_location='cpu',weights_only=False);train=np.asarray(state['train_losses']);valid=np.asarray(state['val_losses']);assert len(train)==len(valid)==state['epoch']+1
  fig,ax=plt.subplots(figsize=(9,4));epochs=np.arange(1,len(train)+1);ax.plot(epochs,train,label='Training');ax.plot(epochs,valid,label='Validation');ax.set(xlabel='Completed epoch',ylabel='Total configured loss',title=entry['name']+' | %d epochs'%len(train));ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'loss_history.png',dpi=150);plt.close(fig)
  (out/'loss_history.json').write_text(json.dumps(dict(epochs=epochs.tolist(),train=train.tolist(),validation=valid.tolist()))+'\n')
  body=re.sub(r'<h3>Training and validation loss</h3><img[^>]*>','',body)
  body+='<h3>Training and validation loss</h3><img src="%s/loss_history.png"><p>Total losses have different definitions across objectives and should not be ranked directly. Training data are generated on demand.</p>%s<img src="%s/camera_diagnostics.png">'%(entry['name'],table,entry['name'])
  original=original[:match.start(2)]+body+original[match.end(2):]
 (root/'fov_widths_all_runs.json').write_text(json.dumps(allmetrics,indent=2)+'\n');(root/'airy_reference_reuse.json').write_text(json.dumps(sources,indent=2)+'\n')
 original=original.replace('loading="lazy"','loading="eager"')
 css='''<style id="report-print">body{max-width:1500px;margin:24px auto;font-family:Arial,sans-serif}img{display:block;max-width:100%;height:auto}table{border-collapse:collapse;max-width:100%;margin-bottom:18px}th,td{border:1px solid #999;padding:5px;text-align:left}thead{display:table-header-group}h2,h3{break-after:avoid}img,tr{break-inside:avoid}@page{size:A3 landscape;margin:12mm}@media print{body{max-width:none;margin:0}img{max-width:100%;max-height:245mm;object-fit:contain}a{color:inherit}section{margin-bottom:20mm}h2{break-before:page}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}</style>'''
 original=re.sub(r'<style id="report-print">.*?</style>','',original,flags=re.S);original=original.replace('</style>','</style>'+css,1)
 original=original.replace('<h1>Final-epoch Deep Design evaluation — November18</h1>','<h1>Final-epoch Deep Design evaluation — November18</h1><p>Updated: expected-signal panels removed; all14 loss histories and center/edge widths added. <a href="report.pdf">PDF report</a></p>',1)
 index.write_text(original,encoding='utf-8');print('Updated14 loss histories,168 beam-position measurements, and camera-only panels. No GPU inference required.')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--report-dir',required=True);a=p.parse_args();update(a.report_dir)
