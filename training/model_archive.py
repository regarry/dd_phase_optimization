"""Self-contained inference bundle: frozen source + state + data + environment.

A Python module pickle alone is intentionally not used as a portability promise.
"""
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import warnings

import torch

from training.checkpoint import read_checkpoint
from training.runtime import atomic_json


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def archive_model(run_dir,source_snapshot,reason):
    run=Path(run_dir).resolve();source=Path(source_snapshot).resolve()
    if not source.is_dir():raise RuntimeError('Frozen source snapshot is missing')
    checkpoint=run/'checkpoints/latest.pt';state=read_checkpoint(checkpoint)
    if not torch.isfinite(state['raw_phase']).all() or not all(
            torch.isfinite(value).all() for value in state['model'].values() if value.is_floating_point()):
        raise RuntimeError('Refusing to archive nonfinite model/phase tensors')
    destination=run/'model_bundle'
    if destination.exists():
        old=json.loads((destination/'manifest.json').read_text())
        if old['checkpoint_sha256']==sha256(checkpoint):return old
        raise RuntimeError('Existing deployment bundle belongs to a different checkpoint')
    temporary=run/('.model_bundle.tmp.'+str(os.getpid()));temporary.mkdir()
    (temporary/'code').mkdir();(temporary/'artifacts').mkdir()
    for name in ['data','physics','models','training','utils']:
        folder=source/name
        if not folder.is_dir():raise RuntimeError('Snapshot lacks module '+name)
        shutil.copytree(folder,temporary/'code'/name,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['pyproject.toml','requirements.txt']:
        if (source/name).exists():shutil.copyfile(source/name,temporary/'code'/name)
    shutil.copyfile(checkpoint,temporary/'full_state.pt')
    config=state['config']
    filename=Path(config['defocused_beads_filename'])
    bead_file=filename if filename.is_absolute() else run/filename
    if not bead_file.is_file():raise RuntimeError('Inference bead-template file missing')
    shutil.copyfile(bead_file,temporary/'artifacts'/filename.name)
    if (run/'psf_stack.mat').exists():shutil.copyfile(run/'psf_stack.mat',temporary/'artifacts/psf_stack.mat')
    environment=dict(python=sys.version,torch=torch.__version__,cuda_build=torch.version.cuda,
                     platform=platform.platform(),packages={})
    for package in ['numpy','scipy','torchvision','scikit-image','tifffile','matplotlib','PyYAML']:
        try:environment['packages'][package]=metadata.version(package)
        except metadata.PackageNotFoundError:environment['packages'][package]='unavailable'
    atomic_json(temporary/'environment.json',environment)
    atomic_json(temporary/'config.json',config)
    atomic_json(temporary/'raw_config.json',state['raw_config'])
    detector=dict(available=False)
    try:
        from models.unet import OpticsDesignUnet
        model=OpticsDesignUnet(config).cpu().eval()
        weights={key[len('unet.'):]:value for key,value in state['model'].items() if key.startswith('unet.')}
        model.load_state_dict(weights,strict=True)
        shape=(1,int(config.get('Nimgs',len(config.get('z_depth_list',[0])))),
               int(config['image_volume'][0]),int(config['image_volume'][1]))
        if config.get('conv3d',False):raise RuntimeError('Standalone detector trace supports reviewed 2D path only')
        example=torch.zeros(shape)
        with warnings.catch_warnings(record=True) as caught,torch.no_grad():
            traced=torch.jit.trace(model,example,check_trace=True)
            torch.testing.assert_close(traced(example),model(example),rtol=1e-4,atol=1e-5)
            traced.save(str(temporary/'detector.ts'))
        detector=dict(available=True,input_shape=shape,
                      note='Detector only; optical forward model uses frozen source and full state',
                      warnings=[str(item.message) for item in caught])
    except Exception as exc:
        # The full eager optical model remains reproducible from the bundle.
        detector['error']=str(exc)
    loader='''"""Load only trusted bundles, with the original logical GPU count/order."""
import json
from pathlib import Path
import sys


def load_full_model(bundle):
    root=Path(bundle).resolve()
    for name in ['models','physics','training','data','utils']:
        loaded=sys.modules.get(name)
        origin=getattr(loaded,'__file__',None)
        if origin and root/'code' not in Path(origin).resolve().parents:
            raise RuntimeError('Load the frozen bundle in a fresh Python process')
    sys.path.insert(0,str(root/'code'))
    import torch
    from models.wrappers import ParallelEndToEndModel
    from physics.phase_io import quantize_phase
    state=torch.load(root/'full_state.pt',map_location='cpu',weights_only=False)
    if torch.cuda.device_count()!=len(state['cuda_rng']):
        raise RuntimeError('Expose the original logical GPU count/order')
    cfg=dict(state['config'])
    cfg['training_results_dir']=str(root/'artifacts')
    cfg['defocused_beads_filename']=Path(cfg['defocused_beads_filename']).name
    model=ParallelEndToEndModel(cfg).eval()
    model.load_state_dict(state['model'],strict=True)
    phase=quantize_phase(state['raw_phase'].to('cuda:0'),cfg.get('slm_bit_depth',8))
    return model,phase,cfg
'''
    (temporary/'load_bundle.py').write_text(loader)
    (temporary/'README.md').write_text('''# Frozen full-model inference bundle

This bundle includes the complete trained state, raw phase, architecture source,
scientific configuration, inference bead/PSF templates and environment versions.
Use `load_bundle.load_full_model(BUNDLE)` with the recorded logical GPU count and
compatible dependencies. It loads the bundled implementation rather than future
edits to the original checkout. Load trusted bundles only: the checkpoint contains
Python RNG/NumPy objects and is not a safe interchange format for untrusted data.

If manifest.json reports detector.available=true, detector.ts is a standalone
TorchScript detector that can run without importing the Python architecture.
It accepts the recorded camera tensor shape; it does not include the optical
simulation, camera model or phase. The complete eager optical pipeline still
requires a compatible Torch/CUDA environment and the frozen source. Future
framework/driver compatibility is not guaranteed by saving a Python module pickle.
Scientific settings must not be silently changed when relocating this bundle.
''')
    manifest=dict(format_version=1,reason=reason,completed_epochs=state['epoch']+1,
                  progress=state.get('progress',{}),checkpoint_sha256=sha256(checkpoint),
                  detector=detector,files={})
    for path in temporary.rglob('*'):
        if path.is_file():manifest['files'][str(path.relative_to(temporary))]=sha256(path)
    atomic_json(temporary/'manifest.json',manifest)
    temporary.rename(destination)
    return manifest
