"""Paired inference at the common zero-based epoch 310."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from data.datasets import SyntheticMicroscopeData
from models.wrappers import ParallelEndToEndModel
from physics.phase_io import quantize_phase
from scripts.evaluate_tissue_study import class_scores, confusion, inference_state_dict
from training.checkpoint import read_checkpoint

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--study', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
study, out = args.study.resolve(), args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
manifest = json.loads((study / 'manifest.json').read_text())
runs = [(name, entry) for name, entry in manifest['runs'].items()
        if entry['class_weights'] is not None]
assert len(runs) == 9 and torch.cuda.device_count() == 1
assert os.environ.get('SLURM_JOB_ID')
scene_seed, noise_base, count = 2026112401, 2026112500, 128
results, matrices, cameras, scene_hashes = [], [], [], []
for name, entry in runs:
    run = Path(entry['run_dir'])
    snapshot = run / 'models/net_310.pt'
    state = read_checkpoint(run / 'checkpoints/latest.pt')
    cfg = dict(state['config'])
    assert state['raw_config']['freeze_slm'] and cfg['num_classes'] == 3
    assert cfg.get('camera_exposure_mode', 'auto_percentile') == 'auto_percentile'
    cfg['SpatialMulitWellLoss'] = False
    phase = quantize_phase(state['raw_phase'].cuda(), cfg.get('slm_bit_depth', 8))
    del state
    model = ParallelEndToEndModel(cfg).eval()
    saved = torch.load(snapshot, map_location='cpu', weights_only=True)
    model.load_state_dict(inference_state_dict(saved, model.state_dict(), 1), strict=True)
    del saved
    np.random.seed(scene_seed)
    random.seed(scene_seed)
    torch.manual_seed(scene_seed)
    scenes = SyntheticMicroscopeData(count, cfg)
    per_scene = []
    with torch.no_grad():
        for i in range(count):
            xyz, target = scenes[i]
            xyz = torch.as_tensor(xyz).float()
            truth = np.asarray(target).squeeze().astype(np.int64)
            identity = hashlib.sha256(xyz.numpy().tobytes() + truth.tobytes()).hexdigest()
            torch.manual_seed(noise_base + i)
            torch.cuda.manual_seed_all(noise_base + i)
            camera = model.physics(phase, xyz[None].cuda())
            assert torch.isfinite(camera).all()
            if not results:
                scene_hashes.append(identity)
                cameras.append(camera.cpu())
            else:
                assert identity == scene_hashes[i], (name, i)
                assert torch.equal(camera.cpu(), cameras[i]), (name, i, 'camera mismatch')
            logits = model.unet(camera)
            assert torch.isfinite(logits).all()
            prediction = logits.argmax(1).squeeze().cpu().numpy()
            per_scene.append(confusion(truth, prediction, 3))
    counts = np.stack(per_scene)
    matrices.append(counts)
    np.save(out / (name + '_confusion.npy'), counts)
    results.append(dict(run=name, weights=entry['class_weights'],
                        snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                        classes=class_scores(counts.sum(axis=0))))
    print(name, results[-1]['classes'][1], flush=True)
    del model, phase, camera, logits
    torch.cuda.empty_cache()

def bead_f1(counts):
    tp = counts[..., 1, 1]
    denominator = counts[..., 1, :].sum(axis=-1) + counts[..., :, 1].sum(axis=-1)
    return np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)

counts = np.stack(matrices)
rng = np.random.default_rng(2026112402)
draws = rng.multinomial(count, np.full(count, 1/count), size=10000)
scores = bead_f1(np.einsum('bi,rijk->brjk', draws, counts))
order = np.argsort(-bead_f1(counts.sum(axis=1)))
leader = order[0]
for idx, row in enumerate(results):
    row['bead_f1_ci95_exploratory'] = np.quantile(scores[:, idx], [.025, .975]).tolist()
    row['leader_minus_run_ci95_exploratory'] = np.quantile(
        scores[:, leader] - scores[:, idx], [.025, .975]).tolist()
report = dict(success=True, job_id=os.environ['SLURM_JOB_ID'], gpu=torch.cuda.get_device_name(),
              epoch_index=310, completed_epochs=311, training_scenes=39808,
              fresh_scenes=count, scene_seed=scene_seed, noise_seed_base=noise_base,
              camera_exposure_mode='auto_percentile', identical_camera_inputs_verified=True,
              scene_hashes=scene_hashes, ranking=[results[i] for i in order])
(out / 'results.json').write_text(json.dumps(report, indent=2))
