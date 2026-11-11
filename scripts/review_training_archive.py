"""Read-only archive inventory and saved-inference audit; no GPU/training jobs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import tifffile
import yaml

PRUNE={'models','learned_phase_masks','psf_imgs','bead_defocus_pngs','checkpoints',
       'code','tmp','mpl_cache','cpu_tests','validation'}


def config(path):
    if not path.exists():return {}
    text=path.read_text(errors='replace')
    try:return yaml.safe_load(text) or {}
    except yaml.YAMLError:
        try:return yaml.load(text,Loader=yaml.BaseLoader) or {}
        except yaml.YAMLError:return {}


def numeric(value,default=None):
    try:return float(value)
    except (TypeError,ValueError):return default


def boolean(value):
    return str(value).lower()=='true'


def objective(cfg):
    keys=['ce_loss_weight','bce_loss_weight','dice_loss_weight','mae_loss_weight',
          'mse_loss_weight','spatial_multi_well_loss_weight','tv_loss_weight']
    active={k.replace('_loss_weight',''):numeric(cfg.get(k),0) for k in keys if numeric(cfg.get(k),0)>0}
    return active or {'legacy_unrecorded':None}


def history(path):
    if not path.exists():return []
    try:return np.atleast_1d(np.loadtxt(path)).astype(float).tolist()
    except (ValueError,OSError):return []


def counts_metrics(counts):
    tn,fp,fn,tp=map(int,np.asarray(counts).ravel())
    return dict(tn=tn,fp=fp,fn=fn,tp=tp,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                prevalence=(tp+fn)/(tn+fp+fn+tp) if tn+fp+fn+tp else None)


def images(prediction,target):
    p=np.asarray(prediction).squeeze();t=np.asarray(target).squeeze()
    if t.ndim==3 and t.shape[0]==3 and t.shape[-1]!=3:t=np.moveaxis(t,0,-1)
    if p.ndim==3 and p.shape[0]==3 and p.shape[-1]!=3:p=np.moveaxis(p,0,-1)
    if t.ndim==3 and t.shape[-1]==3:
        labels=t.argmax(axis=-1);truth=labels==1;task='three_class_bead_centers'
    elif t.ndim==2:
        if np.nanmax(t)>1 and np.all(np.isin(t,[0,1,2])):
            labels=t.astype(int);truth=labels==1;task='three_class_bead_centers'
        else:
            if np.nanmax(t)>1 or np.nanmin(t)<0:raise ValueError('Unknown target intensity encoding')
            labels=(t>=.5).astype(int);truth=labels==1
            task='soft_target_overlap' if np.any((t>0)&(t<1)) else 'binary_bead_centers'
    else:raise ValueError('Unsupported target shape '+str(t.shape))
    if p.ndim==3 and p.shape[-1]==3:
        predicted_labels=p.argmax(axis=-1);pred=predicted_labels==1
        probability=p[...,1].astype(float)
        if np.issubdtype(p.dtype,np.integer):probability/=np.iinfo(p.dtype).max
        rule='argmax class=1'
    elif p.ndim==2:
        probability=p.astype(float)
        if np.issubdtype(p.dtype,np.integer) and probability.max()>1:
            probability/=np.iinfo(p.dtype).max
        if not np.isfinite(probability).all() or probability.min()<0 or probability.max()>1.00001:
            raise ValueError('Predictions are not finite probabilities')
        pred=probability>=.5;predicted_labels=pred.astype(int);rule='probability >= 0.5'
    else:raise ValueError('Unsupported prediction shape '+str(p.shape))
    if pred.shape!=truth.shape:raise ValueError('Prediction/target shape mismatch')
    matrix=np.array([[np.sum(~truth&~pred),np.sum(~truth&pred)],
                     [np.sum(truth&~pred),np.sum(truth&pred)]],dtype=np.int64)
    return dict(probability=probability,truth=truth,pred=pred,labels=labels,
                predicted_labels=predicted_labels,counts=matrix,task=task,rule=rule)


def audit_inference(folder):
    cfg=config(folder/'config.yaml');cases=[];errors=[];pooled=np.zeros((2,2),dtype=np.int64)
    paths=sorted(folder.glob('inference_*.tif'))+sorted(folder.glob('inference_*.tiff'))
    for p in paths:
        match=re.fullmatch(r'inference_(\d+)\.tiff?',p.name)
        if not match:continue
        i=match.group(1)
        target=next(iter(list(folder.glob('ground_truth_'+i+'.tif'))+list(folder.glob('ground_truth_'+i+'.tiff'))),None)
        if not target:continue
        try:
            decoded=images(tifffile.imread(p),tifffile.imread(target))
            row=dict(index=int(i),prediction=str(p),target=str(target),task=decoded['task'],rule=decoded['rule'],
                     **counts_metrics(decoded['counts']))
            pooled+=decoded['counts'];cases.append(row)
        except Exception as exc:errors.append(dict(file=str(p),error=str(exc)))
    text=(folder/'metrics.txt').read_text(errors='replace') if (folder/'metrics.txt').exists() else ''
    # Older logs repeat thirds; only parse their explicitly labeled full image.
    legacy=[tuple(map(float,m)) for m in re.findall(r'Full image:\s*Precision:\s*([\deE.+-]+),\s*Recall:\s*([\deE.+-]+),\s*F1:\s*([\deE.+-]+)',text)]
    if not legacy and 'precision' in text.lower():
        try:
            legacy=[(float(row['precision']),float(row['recall']),float(row.get('f1',row.get('f1_score'))))
                    for row in csv.DictReader(text.splitlines())]
        except (KeyError,ValueError,TypeError):pass
    return dict(path=str(folder),saved_prediction_files=len(paths),paired_cases=len(cases),cases=cases,
                recomputed=counts_metrics(pooled) if cases else None,errors=errors,
                tasks=sorted(set(c['task'] for c in cases)),
                reported_mean=dict(precision=float(np.mean([v[0] for v in legacy])),
                    recall=float(np.mean([v[1] for v in legacy])),f1=float(np.mean([v[2] for v in legacy]))) if legacy else None,
                reported_cases=len(legacy),config=cfg,
                epoch=cfg.get('inference_epoch',cfg.get('epoch')),model_path=cfg.get('model_path'))


def scan(root,output,since=None,until=None):
    output.mkdir(parents=True,exist_ok=True)
    runs=sorted(p for p in root.iterdir() if p.is_dir() and p.name[:8].isdigit()
                and (since is None or since<=p.name[:8]) and (until is None or p.name[:8]<=until))
    for parent in sorted(root.glob('phase_initialization_*')):
        runs.extend(sorted(p for p in parent.glob('*/outputs/20*')
                           if (since is None or since<=p.name[:8]) and (until is None or p.name[:8]<=until)))
    records=[]
    for index,run in enumerate(runs):
        cfg=config(run/'config.yaml') or config(run/'input.yaml')
        train=history(run/'train_losses.txt');valid=history(run/'val_losses.txt')
        finite=np.asarray(valid)[np.isfinite(valid)] if valid else np.array([])
        infer_folders=set();profiles=[];metric_files=[]
        for current,dirs,files in os.walk(run):
            dirs[:]=[d for d in dirs if d not in PRUNE and not d.startswith('.')]
            for name in files:
                path=Path(current)/name
                if re.fullmatch(r'inference_\d+\.tiff?',name):infer_folders.add(Path(current))
                if name=='metrics.txt':metric_files.append(str(path));infer_folders.add(Path(current))
                if name=='column_sums.tiff':profiles.append(str(path))
        evaluations=[audit_inference(p) for p in sorted(infer_folders)]
        best=max((e for e in evaluations if e['recomputed'] and e['recomputed']['f1'] is not None),
                 key=lambda e:e['recomputed']['f1'],default=None)
        models=run/'models'
        saved_epochs=[int(m.group(1)) for p in models.iterdir() if (m:=re.fullmatch(r'net_(\d+)\.pt',p.name))] if models.exists() else []
        config_path=run/'config.yaml' if (run/'config.yaml').exists() else run/'input.yaml'
        row=dict(id=run.name,path=str(run),config=cfg,config_path=str(config_path),
             config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.exists() else None,
             objective=objective(cfg),initialization=cfg.get('initial_phase_mask','unknown'),
             epochs=len(train),validation_epochs=len(valid),finite_validation_epochs=len(finite),
             best_val=float(finite.min()) if len(finite) else None,
             best_val_epoch=int(np.nanargmin(np.where(np.isfinite(valid),valid,np.nan)))+1 if len(finite) else None,
             first_val=float(valid[0]) if valid and np.isfinite(valid[0]) else None,
             last_val=float(valid[-1]) if valid and np.isfinite(valid[-1]) else None,
             model_files=len(saved_epochs),last_model_epoch=max(saved_epochs) if saved_epochs else None,
             profile_files=profiles,metric_files=metric_files,evaluations=evaluations,
             best_saved_inference=best['path'] if best else None,
             best_saved_f1=best['recomputed']['f1'] if best else None,
             best_saved_precision=best['recomputed']['precision'] if best else None,
             best_saved_recall=best['recomputed']['recall'] if best else None,
             best_saved_cases=best['paired_cases'] if best else 0,
             best_saved_task=best['tasks'] if best else [],
             history={'train':train,'validation':valid})
        records.append(row)
        if index%20==0:print('SCANNED',index+1,'/',len(runs),run.name,flush=True)
    # JSON disallows NaN; preserve absent/nonfinite history values as null.
    def clean(value):
        if isinstance(value,float) and not np.isfinite(value):return None
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
        if isinstance(value,list):return [clean(v) for v in value]
        return value
    (output/'inventory.json').write_text(json.dumps(clean(records),indent=2,default=str)+'\n')
    fields=['id','path','initialization','objective','epochs','best_val','best_val_epoch','best_saved_f1',
            'best_saved_precision','best_saved_recall','best_saved_cases','best_saved_task','best_saved_inference']
    with (output/'inventory.csv').open('w') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader()
        for row in records:writer.writerow({k:row[k] for k in fields})
    print('COMPLETE',len(records),'runs;',sum(len(r['evaluations']) for r in records),'inference evaluations;',
          sum(e['paired_cases'] for r in records for e in r['evaluations']),'paired saved images',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,default=Path('training_results'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--since',help='Inclusive YYYYMMDD');p.add_argument('--until',help='Inclusive YYYYMMDD');args=p.parse_args()
    for bound in [args.since,args.until]:
        if bound is not None and not re.fullmatch(r'[0-9]{8}',bound):p.error('Date bounds require YYYYMMDD')
    scan(args.root.resolve(),args.output.resolve(),args.since,args.until)
