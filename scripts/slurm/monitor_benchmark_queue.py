"""Bounded login-node monitor; retarget one pending job, never submit/cancel.

Optional retargeting requires an explicitly configured capacity helper; select between validated
48-GB L40/L40S and 80-GB H100 pairs. Scheduling remains subject to Slurm/QOS limits.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


def command(*args):
    return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=15)


def free_nodes(text):
    result={'l40s':[], 'h100':[], 'l40':[]}
    for line in text.splitlines():
        fields=dict(re.findall(r'(\w+)=(\S+)',line))
        state=fields.get('State','')
        if os.environ.get('DD_SLURM_PARTITION','gpu') not in fields.get('Partitions','') or any(v in state for v in ('DRAIN','DOWN','NOT_RESPONDING')):
            continue
        cfg=dict(v.split('=',1) for v in fields.get('CfgTRES','').split(',') if '=' in v)
        used=dict(v.split('=',1) for v in fields.get('AllocTRES','').split(',') if '=' in v)
        cpu=int(cfg.get('cpu',0))-int(used.get('cpu',0))
        memory=int(fields.get('RealMemory',0))-int(fields.get('AllocMem',0))
        for model in result:
            key='gres/gpu:'+model
            if int(cfg.get(key,0))-int(used.get(key,0))>=2 and cpu>=4 and memory>=32768:
                result[model].append(fields['NodeName'])
    return result


def qos_available(text):
    result={'l40s':0,'h100':0,'l40':0}
    for line in text.splitlines():
        parts=line.split()
        if len(parts)<6 or parts[0]!=os.environ.get('DD_SLURM_PARTITION','gpu'):continue
        model={'L40S':'l40s','H100':'h100','L40':'l40'}.get(parts[1])
        if model:
            try:result[model]=int(parts[4])
            except ValueError:pass
    return result


def monitor(job,state_path,duration,interval):
    start=time.monotonic();history=[];last_change=-300
    manifest_path=state_path.parent/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    benchmark=manifest.get('runs',{}).get('benchmark',{})
    if benchmark.get('job_id')!=job:
        raise ValueError('Monitor may retarget only the benchmark recorded in this manifest')
    progress_path=Path(benchmark['run_dir'])/'live_progress.json' if benchmark.get('job_id')==job else None
    base=dict(job_id=job,eligible_gpu_types={'l40s':'48 GB','h100':'80 GB','l40':'48 GB if deployed'},
              excluded={'h200':'not configured for this monitor',
                        'rtx_6000_pro':'Blackwell not validated with frozen Torch 2.4.1/CUDA 12.1 environment',
                        'a100/a30/a10':'not configured for this monitor'},
              benchmark_wall_time='00:20:00',training_wall_time='1-00:00:00',
              monitor_duration_seconds=duration)
    while time.monotonic()-start<duration:
        data={**base,'checked_utc':datetime.now(timezone.utc).isoformat(),'history':history,'monitor_active':True}
        try:
            info=command('scontrol','show','job',str(job),'-o')
            fields=dict(re.findall(r'(\w+)=(\S+)',info))
            data.update(state=fields.get('JobState'),reason=fields.get('Reason'),node=fields.get('NodeList'))
            match=re.search(r'TresPerNode=gres/gpu:([^:,\s]+):2',info)
            current=match.group(1) if match else None
            data['requested_gpu_type']=current
            if data['state'] not in ('PENDING','RUNNING','CONFIGURING'):
                data['monitor_active']=False
            elif data['state']=='PENDING' and current in ('l40s','h100','l40') and os.environ.get('DD_GPU_CAPACITY_HELPER'):
                nodes=free_nodes(command('scontrol','show','node','-o'))
                capacity=qos_available(command(*shlex.split(os.environ['DD_GPU_CAPACITY_HELPER'])))
                data.update(nodes_with_free_pairs=nodes,qos_report_available=capacity,
                            capacity_note='Helper capacity is cached; Slurm makes the final scheduling decision')
                choices=[kind for kind in ('l40s','h100','l40') if capacity[kind]>=2 and nodes[kind]]
                if choices and current not in choices and time.monotonic()-start-last_change>=120:
                    chosen=choices[0]
                    response=command('scontrol','update','JobId='+str(job),'QOS='+os.environ.get('DD_SLURM_SMOKE_QOS','short_gpu'),'Gres=gpu:'+chosen+':2')
                    history.append(dict(time_utc=data['checked_utc'],previous=current,selected=chosen,response=response.strip()))
                    data['requested_gpu_type']=chosen;last_change=time.monotonic()-start
            if progress_path and progress_path.exists():
                data['training_progress']=json.loads(progress_path.read_text())
        except Exception as exc:
            data['error']=str(exc)
        temporary=state_path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(state_path)
        print(json.dumps({k:v for k,v in data.items() if k not in ('history','excluded','nodes_with_free_pairs')}),flush=True)
        if not data['monitor_active']:return
        time.sleep(min(interval,max(0,duration-(time.monotonic()-start))))
    data['monitor_active']=False;data['monitor_stop_reason']='bounded monitoring interval ended'
    state_path.write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--job',type=int,required=True)
    p.add_argument('--state-file',type=Path,required=True);p.add_argument('--duration',type=int,default=1200)
    p.add_argument('--interval',type=int,default=30);args=p.parse_args()
    if not 0<args.duration<=3600 or args.interval<15:
        p.error('Use a positive duration <=3600 seconds and an interval >=15 seconds')
    monitor(args.job,args.state_file.resolve(),args.duration,args.interval)
