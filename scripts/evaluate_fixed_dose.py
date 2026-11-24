"""Matched relative fixed-incident-dose evaluation of trusted checkpoint snapshots."""
import argparse,gc,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from data.datasets import SyntheticMicroscopeData
from models.wrappers import ParallelEndToEndModel
from physics.masks import get_initial_phase_mask
from physics.phase_io import quantize_phase
from physics.simulation import NoiseLayer
from scripts.evaluate_tissue_study import confusion,class_scores,inference_state_dict
from training.checkpoint import read_checkpoint


def bootstrap_f1(scene_counts,seed=23):
    a=np.asarray(scene_counts,dtype=float)
    rng=np.random.RandomState(seed);samples=a[rng.randint(0,len(a),(2000,len(a)))].sum(1)
    denominator=2*samples[:,0]+samples[:,1]+samples[:,2]
    score=np.divide(2*samples[:,0],denominator,out=np.zeros(len(samples)),where=denominator>0)
    return np.percentile(score,[2.5,97.5]).tolist()


def audit_beam(sim,phase):
    original=sim.angular_spectrum_propagation;stages=[]
    def traced(*args,**kw):
        field,data=original(*args,return_power_audit=True,**kw)
        stages.append({key:float(value) for key,value in data.items()});return field
    sim.angular_spectrum_propagation=traced
    try:field=sim.sample_4f(phase)
    finally:sim.angular_spectrum_propagation=original
    assert len(stages)==1 and float(sim.gamma_padded.min())>0,'Audit restricted to the tested scalar propagating-frequency grid'
    launch=stages[0];xs=[]
    for x in [-500,-250,0,250,499]:
        _,a=original(field,x,pad=True,return_power_audit=True);xs.append({k:float(v) for k,v in a.items()})
    for a in [launch]+xs:
        assert abs(a['spectral_output_power']/a['full_output_power']-1)<2e-5
        assert a['full_output_power']<=a['input_power']*(1+2e-5)
        assert a['cropped_output_power']<=a['full_output_power']*(1+2e-5)
        a['full_transmission']=a['full_output_power']/a['input_power']
        a['retained_after_crop']=a['cropped_output_power']/a['input_power']
        a['crop_loss_fraction_of_full']=1-a['cropped_output_power']/a['full_output_power']
    return dict(launch=launch,propagation=xs,input_gaussian_power=float(sim.incident_gaussian.abs().square().sum(dtype=torch.float64)*sim.px**2),power_units='sampled scalar |U|^2 integral in arbitrary amplitude^2 m^2; not watts',interpretation='Configured spectral filtering and numerical crop losses kept separate; no per-mask power renormalization')


def load_model(entry):
    state=read_checkpoint(entry['checkpoint']);cfg=dict(state['config']);cfg.update(training_results_dir=entry['artifacts'],defocused_beads_filename='defocused_beads.mat',SpatialMulitWellLoss=False)
    model=ParallelEndToEndModel(cfg).eval();model.load_state_dict(inference_state_dict(state['model'],model.state_dict(),1),strict=True)
    phase=quantize_phase(state['raw_phase'].cuda(),cfg.get('slm_bit_depth',8))
    return state,cfg,model,phase


def capture(plan):
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.device_count()==1
    root=Path(plan['output']);torch.set_num_threads(4);started=time.monotonic();entries=plan['runs'];reference=entries[0]
    state,cfg,model,phase=load_model(reference);sim=model.physics.replicas[0]
    canonical=dict(cfg,num_classes=3,tissue_sample_z_bounds_um=[-100,100])
    def scenes(seed,n):
        np.random.seed(seed);random.seed(seed);torch.manual_seed(seed)
        ds=SyntheticMicroscopeData(n,canonical);return [ds[i] for i in range(n)]
    calibration=scenes(2026112301,4);test=scenes(2026112302,16)
    gaussian_cfg=dict(cfg,initial_phase_mask='gaussian',gaussian_phase_mode='focused')
    gaussian=quantize_phase(torch.from_numpy(get_initial_phase_mask(gaussian_cfg)).float().cuda(),cfg.get('slm_bit_depth',8))
    with torch.no_grad():calib=torch.cat([sim(gaussian,xyz[None].cuda()) for xyz,_ in calibration],dim=0)
    robust=float(torch.quantile(calib.flatten(),cfg['robust_max_percentile']));assert robust>0
    scale=(cfg['camera_max_adu']/cfg['camera_gain'])*cfg['target_saturation_ratio']/(robust+cfg['epsilon'])
    ref_input=sim.incident_gaussian.detach().cpu().clone();ref_psf=sim.gpu_defocused_beads[:103].detach().cpu().clone()
    settings=['N','px','slm_px','wavelength','refractive_index','lens_approach','angular_spectrum_method','phase_mask_pixel_size','phase_mask_upsample_factor','laser_beam_FWHC','lowpass_pinhole_radius_mm','lensless_prop_distance','fresnel_lens_focal_length','focal_length_2','focal_length_3','focal_length_4','aperature','aberration','slm_bit_depth','illumination_scaling_factor','bead_radius','quantum_efficiency','camera_gain','dark_current_mean','read_noise_std','camera_max_adu','epsilon','min_clip_value','robust_max_percentile','target_saturation_ratio','image_volume','bead_intensity','z_range_cost_function','c5','c13']
    refcfg={k:cfg.get(k) for k in settings}
    protocol=dict(fixed_reference_signal_scale=scale,calibration_gaussian_ideal_percentile=robust,reference_percentile=cfg['robust_max_percentile'],reference_target_saturation_ratio=cfg['target_saturation_ratio'],calibration_seed=2026112301,test_seed=2026112302,calibration_scenes=4,test_scenes=16,dose_multipliers=plan['doses'],multi_view_policy='Equal total incident dose: signal scale divided by number of sequential focal views; same per-frame integration/dark/read noise; incident intensity adjusted, not imaging time',dose_definition='Relative incident illumination energy before phase/pupil/filter. Same incident amplitude, sampled pupil and uniform discrete scan dwell. No scene/mask throughput renormalization. Reference 1x scale set once using disjoint Gaussian calibration scenes.',limitations='Arbitrary signal units, not J/cm^2. Linear fluorescence, no bleaching/saturation of fluorophores. Detector trained with auto-exposure; posthoc fixed-dose domain shift. Numerical crop and filtering loss reported, not repaired.',verified_common_settings=refcfg)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    np.savez_compressed(root/'calibration_scenes.npz',xyz=np.stack([x.numpy() for x,_ in calibration]))
    np.savez_compressed(root/'test_scenes.npz',xyz=np.stack([x.numpy() for x,_ in test]),target=np.stack([t.numpy() for _,t in test]))
    # Bounded CUDA check of the real fixed-scale image-to-raw-phase path.
    trial=state['raw_phase'].cuda().clone().requires_grad_(True)
    signal=sim(quantize_phase(trial,cfg.get('slm_bit_depth',8)),test[0][0][None,:32].cuda())
    fixed_noise=NoiseLayer(dict(cfg,camera_exposure_mode='fixed',fixed_exposure_scale=scale))
    gradient=torch.autograd.grad(fixed_noise(signal).mean(),trial)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().max()>0
    (root/'fixed_gradient_check.json').write_text(json.dumps(dict(success=True,emitters=32,raw_phase_gradient_norm=float(gradient.norm()),raw_phase_gradient_max=float(gradient.abs().max())),indent=2))
    del trial,signal,gradient,fixed_noise
    del state,model,phase,sim,gaussian,calib;gc.collect();torch.cuda.empty_cache()
    cache={};audits={};results=[]
    with torch.no_grad():
        for entry in entries:
            state,cfg,model,phase=load_model(entry);sim=model.physics.replicas[0]
            assert all(cfg.get(k)==v for k,v in refcfg.items()),'Different common optical/camera setting: '+entry['name']
            torch.testing.assert_close(sim.incident_gaussian.cpu(),ref_input,rtol=0,atol=0);torch.testing.assert_close(sim.gpu_defocused_beads[:103].cpu(),ref_psf,rtol=0,atol=0)
            views=len(cfg['z_depth_list']);key=hashlib.sha256(state['raw_phase'].numpy().tobytes()).hexdigest()+str(cfg['z_depth_list'])
            if key not in cache:
                audits[key]=audit_beam(sim,phase)
                cache[key]=[sim(phase,xyz[None].cuda()).cpu() for xyz,_ in test]
                np.savez_compressed(root/('ideal_'+hashlib.sha256(key.encode()).hexdigest()[:12]+'.npz'),ideal=torch.cat(cache[key]).numpy())
            audit=audits[key];assert abs(audit['launch']['input_power']/audit['input_gaussian_power']-1)<2e-5
            assert abs(audit['input_gaussian_power']/float(ref_input.square().sum(dtype=torch.float64)*cfg['px']**2)-1)<2e-5
            levels=[('auto',None)]+[('dose_%g'%d,d) for d in plan['doses']];records=[]
            folder=root/entry['name'];folder.mkdir(exist_ok=True)
            for label,dose in levels:
                noise=NoiseLayer(dict(cfg,camera_exposure_mode='auto_percentile') if dose is None else dict(cfg,camera_exposure_mode='fixed',fixed_exposure_scale=scale*dose/views))
                classes=2 if cfg['num_classes']==1 else 3;matrix=np.zeros((classes,classes),dtype=np.int64);counts=[];effective=[];saturated=[];zero=[];mean_signal=[]
                for i,((xyz,target3),cached) in enumerate(zip(test,cache[key])):
                    ideal=cached.cuda();torch.manual_seed(2026112350+i);torch.cuda.manual_seed_all(2026112350+i)
                    camera=noise(ideal)/cfg['camera_max_adu'];logits=model.unet(camera)
                    assert torch.isfinite(camera).all() and torch.isfinite(logits).all()
                    truth=np.asarray(target3).squeeze().astype(np.int64)
                    if classes==2:truth=(truth==1).astype(np.int64);pred=(logits.sigmoid().squeeze()>=.5).cpu().numpy().astype(np.int64)
                    else:pred=logits.argmax(1).squeeze().cpu().numpy().astype(np.int64)
                    cm=confusion(truth,pred,classes);matrix+=cm;c=class_scores(cm)[1];counts.append([c['tp'],c['fp'],c['fn']])
                    coefficient=float(noise.scale_for(ideal).flatten()[0]);effective.append(coefficient*views/scale)
                    saturated.append(float((camera>=1).float().mean()));zero.append(float((camera<=0).float().mean()));mean_signal.append(float(ideal.mean())*coefficient*cfg['quantum_efficiency']*cfg['camera_gain'])
                    if i==0 and (dose in [1.,4.] or dose is None):
                        np.savez_compressed(folder/(label+'_sample.npz'),xyz=xyz.numpy(),target=truth,prediction=pred,probability=(logits.sigmoid() if classes==2 else logits.softmax(1)).squeeze(0).cpu().numpy(),camera_adu=camera.squeeze(0).cpu().numpy()*cfg['camera_max_adu'])
                c=class_scores(matrix)[1];records.append(dict(level=label,dose=dose,classes=class_scores(matrix),confusion=matrix.tolist(),bead_f1_ci95=bootstrap_f1(counts),scene_counts=counts,relative_incident_dose=dict(min=min(effective),median=float(np.median(effective)),max=max(effective)),saturated_pixel_fraction=float(np.mean(saturated)),zero_clipped_fraction=float(np.mean(zero)),mean_expected_signal_adu=float(np.mean(mean_signal))))
            result=dict(run=entry['name'],cohort=entry['cohort'],epochs=state['epoch']+1,train_hours=state.get('progress',{}).get('train_seconds',0)/3600,checkpoint_sha256=entry['checkpoint_sha256'],views=views,power_audit=audit,levels=records)
            results.append(result);(folder/'metrics.json').write_text(json.dumps(result,indent=2));(root/'results.json').write_text(json.dumps(results,indent=2));print('FIXED_DOSE_CAPTURE_SUCCESS',entry['name'],flush=True)
            del model,state,phase,sim,ideal,camera,logits;gc.collect();torch.cuda.empty_cache()
    (root/'SUCCESS.json').write_text(json.dumps(dict(success=True,models=len(results),test_scenes=16,scenarios=6,job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),torch=torch.__version__,elapsed_seconds=time.monotonic()-started),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();capture(json.loads(a.plan.read_text()))
