"""Read-only audit of bounded study progress and scheduler state."""
import argparse,csv,json,subprocess
from pathlib import Path


def audit(study,recovery=None):
    study=Path(study);manifest=json.loads((study/'study_manifest.json').read_text());extra=json.loads(Path(recovery).read_text()) if recovery else []
    jobs=[str(j) for r in manifest['runs'].values() for j in r['job_ids']]+[str(j) for r in extra for j in r['job_ids']]
    data=subprocess.check_output(['sacct','-j',','.join(jobs),'-X','-n','-P','--format=JobID,State,ElapsedRaw,NodeList'],text=True)
    states={row[0]:dict(state=row[1],elapsed_seconds=int(row[2]),node=row[3]) for row in csv.reader(data.splitlines(),delimiter='|') if row}
    results=[]
    for name,r in manifest['runs'].items():
        run=Path(r['run_dir']);progress=json.loads((run/'progress.json').read_text());terminal=json.loads((run/'terminal.json').read_text()) if (run/'terminal.json').exists() else None
        ids=r['job_ids']+[j for e in extra if e['run']==name for j in e['job_ids']];own={str(j):states.get(str(j),dict(state='unobserved')) for j in ids}
        results.append(dict(run=name,progress=progress,terminal=terminal,jobs=own,
                       executed_allocations=sum(v['state'] not in ['PENDING','CANCELLED','unobserved'] for v in own.values()),
                       active_or_pending=any(v['state'] in ['RUNNING','PENDING','COMPLETING'] for v in own.values())))
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--recovery');p.add_argument('--output',required=True);a=p.parse_args()
    result=audit(a.study,a.recovery);Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    for r in result:print(r['run'],r['progress']['completed_epochs'],'epochs','%.2fh'%(r['progress']['train_seconds']/3600),r['terminal']['reason'] if r['terminal'] else 'active' if r['active_or_pending'] else 'STOPPED; investigate',r['executed_allocations'],'executed allocations')
