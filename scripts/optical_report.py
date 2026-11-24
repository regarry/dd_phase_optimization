"""CPU analysis of captured dithered beam profiles and exact camera arrays."""
import json
from pathlib import Path
import numpy as np
from utils.beam_metrics import beam_metrics


def number(value):return 'unavailable' if value is None else '%.3g'%value


def expected_camera(ideal,cfg):
    from physics.exposure import exposure_settings
    mode,scale=exposure_settings(cfg)
    if mode=='auto_percentile':
        robust=float(np.quantile(ideal,cfg['robust_max_percentile']))
        scale=(cfg['camera_max_adu']/cfg['camera_gain'])*cfg['target_saturation_ratio']/(robust+cfg['epsilon'])
    electrons=np.maximum(ideal*scale*cfg['quantum_efficiency']+cfg['dark_current_mean'],cfg['min_clip_value'])
    return electrons*cfg['camera_gain'],np.sqrt(electrons+cfg['read_noise_std']**2)*cfg['camera_gain'],scale


def measure_profiles(cfg,x,z,profiles):
    optical={name:beam_metrics(x,z,p) for name,p in profiles.items()}
    if cfg.get('tissue_sample_z_bounds_um'):
        from utils.beam_metrics import power_interval
        lower,upper=cfg['tissue_sample_z_bounds_um'];support=(z>=lower)&(z<=upper)
        focus=int(np.argmin(abs(x)))
        for name,p in profiles.items():
            row=p[focus];restricted=power_interval(z,row*support)
            optical[name]['sample_restricted_os63_um']=restricted['width_um'] if restricted else None
            optical[name]['sample_support_um']=[lower,upper]
            optical[name]['fraction_profile_power_in_sample']=float(row[support].sum()/row.sum())
    return optical


def augment_run(entry,meta,pooled):
    import matplotlib.pyplot as plt
    out=Path(entry['output']);cfg=meta['configuration']
    with np.load(out/'beam_profiles.npz') as b:
        x=b['x_um'].copy();z=b['z_um'].copy();profiles={k:b[k].copy() for k in ('learned','gaussian','axicon')}
    if (out/'airy_reference.npz').exists():
        with np.load(out/'airy_reference.npz') as b:profiles['airy']=b['airy'].copy()
    optical=measure_profiles(cfg,x,z,profiles)
    (out/'optical_metrics.json').write_text(json.dumps(optical,indent=2)+'\n')
    case=next(c for c in meta['cases'] if c['dataset']=='fresh' and c['index']==0)
    with np.load(out/case['file']) as saved:d={k:saved[k].copy() for k in saved.files}
    camera=d['camera_adu'];ideal=d.get('ideal_photons');stats=dict(camera_min_adu=float(camera.min()),camera_max_adu=float(camera.max()),median_adu=float(np.median(camera)),percentile_99_9_adu=float(np.quantile(camera,.999)),saturated_fraction=float((camera>=cfg['camera_max_adu']).mean()),zero_clipped_fraction=float((camera<=0).mean()),expected_dark_mean_adu=cfg['dark_current_mean']*cfg['camera_gain'],expected_dark_noise_std_adu=np.sqrt(cfg['dark_current_mean']+cfg['read_noise_std']**2)*cfg['camera_gain'])
    if ideal is not None:
        expected,std,scale=expected_camera(ideal,cfg);stats['exposure_scale']=scale
    else:expected=camera
    (out/'camera_stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    # Equal x/y sampling; display the complete strip plus a true-aspect local crop.
    xy=d['xyz'];target=np.asarray(d['target']);positions=np.argwhere(target==1)
    cy,cx=positions[0] if len(positions) else np.array(camera.shape)//2
    y0=max(0,int(cy)-24);y1=min(camera.shape[0],y0+49);x0=max(0,int(cx)-24);x1=min(camera.shape[1],x0+49)
    fig,axes=plt.subplots(2,2,figsize=(13,7));high=max(1,float(np.quantile(camera,.998)))
    im=axes[0,0].imshow(camera,cmap='gray',vmin=0,vmax=high,aspect='equal');fig.colorbar(im,ax=axes[0,0],label='ADU');axes[0,0].set_title('Exact noisy detector input; 0–99.8% display stretch')
    im=axes[0,1].imshow(expected,cmap='gray',vmin=0,vmax=high,aspect='equal');fig.colorbar(im,ax=axes[0,1],label='ADU');axes[0,1].set_title('Expected camera signal incl. dark offset; same display scale')
    for ax,image,title in zip(axes[1],(camera,expected),('Noisy crop, equal pixel aspect','Expected-signal crop, equal pixel aspect')):
        im=ax.imshow(image[y0:y1,x0:x1],cmap='gray',vmin=0,vmax=high,aspect='equal',extent=[x0,x1,y1,y0]);ax.set_title(title);fig.colorbar(im,ax=ax,label='ADU')
    for ax in axes.ravel():ax.set(xlabel='Camera x pixel (1 µm configured)',ylabel='Camera y pixel (1 µm configured)')
    fig.suptitle(entry['name']+' | physical pixel aspect; camera values are simulated ADU, not normalized beam intensity');fig.tight_layout();fig.savefig(out/'camera_diagnostics.png',dpi=150);plt.close(fig)
    if entry['name'].startswith(('one_sided_','two_sided_')):
        fig,axes=plt.subplots(1,2,figsize=(14,5));p=profiles['learned'];extent=[x[0],x[-1],z[0],z[-1]]
        for ax in axes:
            ax.imshow((p/p.max()).T,origin='lower',extent=extent,aspect='auto',cmap='inferno',vmin=0,vmax=1)
            ax.set(xlabel='Propagation x (µm)',ylabel='Thickness z (µm)',ylim=(-150,150))
            for bound in cfg['tissue_z_bounds_um']:ax.axhline(bound,color='cyan',ls='--')
        axes[0].set_title('Dithered learned intensity / own peak')
        physical_x=(xy[:,0]-cfg['image_volume'][1]//2)*cfg['px']*1e6;physical_z=xy[:,2]*cfg['px']*1e6
        focus=(xy[:,2]>=cfg['z_range_cost_function'][0])&(xy[:,2]<=cfg['z_range_cost_function'][1])
        for mask,color,label in ((~focus,'white','Out-of-focus bead centres'),(focus,'lime','In-focus bead centres')):
            axes[1].scatter(physical_x[mask],physical_z[mask],s=6,c=color,alpha=.35,label=label)
        axes[1].set_title('Same fresh test sample projected onto x–z');axes[1].legend(fontsize=8)
        fig.suptitle(entry['name']+' | points show known synthetic 3D coordinates projected along scan y');fig.tight_layout();fig.savefig(out/'bead_xz_overlay.png',dpi=150);plt.close(fig)
    return optical


def metrics_html(entry,meta,metrics):
    out=Path(entry['output']);optical=json.loads((out/'optical_metrics.json').read_text(encoding='utf-8'));camera=json.loads((out/'camera_stats.json').read_text(encoding='utf-8'))
    html='<h3>Pixel detection metrics (hard Dice = F1)</h3><table border=1 cellpadding=5><tr><th>Scenes</th><th>Class</th><th>Dice</th><th>Precision</th><th>Recall</th></tr>'
    names=['background','bead','connection']
    for group,values in metrics.items():
        for c in values['classes']:
            html+='<tr><td>%s (%d)</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>'%(group,values['scenes'],names[c['class_id']],number(c['f1']),number(c['precision']),number(c['recall']))
    html+='</table><h3>Scanned illumination metrics at x=0</h3><table border=1 cellpadding=5><tr><th>Beam</th><th>Main lobe 1/e (µm)</th><th>Peak z (µm)</th><th>On-axis lobe (µm)</th><th>Optical sectioning 63% (µm)</th><th>Generalized Rayleigh range: OS waist x; doubling left / right (µm)</th></tr>'
    for name,m in optical.items():
        ranges=['%s%s'%('≥' if m[k]['censored'] else '',number(m[k]['distance_um'])) for k in ('generalized_range_left','generalized_range_right')]
        if m.get('waist_at_field_edge'):ranges=['unresolved: minimum at field edge','unresolved']
        html+='<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s / %s</td></tr>'%(name,number(m['main_lobe_1e_um']),number(m['main_lobe_peak_z_um']),number(m['on_axis_lobe_1e_um']),number(m['optical_sectioning_63_um']),*['x=%s; %s'%(number(m['os_waist_x_um']),ranges[0]),ranges[1]])
    html+='</table><p>Generalized ranges start at the minimum observed OS waist and follow OS doubling, not Gaussian √2 Rayleigh range. ≥ means crossing outside sampled propagation window. Total energy is limited to the simulated z window; illumination widths are not a measured full-microscope axial PSF. Definition source: <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC6968747/">Remacha et al. (2020)</a>.</p>'
    html+='<p>Camera: median %s ADU, 99.9th percentile %s ADU, max %s ADU, saturated fraction %s. Dark mean %s ADU; dark+read σ %s ADU. Auto-exposure stays enabled.</p>'%tuple(number(camera[k]) for k in ('median_adu','percentile_99_9_adu','camera_max_adu','saturated_fraction','expected_dark_mean_adu','expected_dark_noise_std_adu'))
    return html
