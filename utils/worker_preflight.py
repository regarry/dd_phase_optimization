"""Bounded CPU test of real Torch worker tensor transfer before a GPU run."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--worker-timeout', type=float, default=60,
                        help='Tensor-transfer timeout, matching the training loader (seconds)')
    parser.add_argument('--report', type=Path,
                        help='Persistent JSON result written by the bounded parent')
    args = parser.parse_args()
    if args.worker_timeout <= 0 or not args.worker_timeout < float('inf'):
        parser.error('--worker-timeout must be finite and positive')
    started=time.monotonic()
    def stage(name):
        print('WORKER_PREFLIGHT '+json.dumps(dict(stage=name,
              elapsed_seconds=time.monotonic()-started,host=socket.gethostname(),
              pid=os.getpid())),flush=True)
    if not args.child:
        # Allow interpreter/Torch import and worker teardown outside the loader's
        # own bound. Never retry or fall back to zero workers on failure.
        bound=args.worker_timeout+120
        result=dict(success=False,host=socket.gethostname(),
                    worker_timeout_seconds=args.worker_timeout,
                    total_timeout_seconds=bound,tmpdir=os.environ.get('TMPDIR'))
        stage('parent_started')
        try:
            subprocess.run([sys.executable, '-m', 'utils.worker_preflight', '--child',
                            '--worker-timeout',str(args.worker_timeout)],
                           timeout=bound,check=True)
            result['success']=True
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:
            result['error']=str(exc)
            raise
        finally:
            result['elapsed_seconds']=time.monotonic()-started
            if args.report:
                args.report.parent.mkdir(parents=True,exist_ok=True)
                temporary=args.report.with_suffix(args.report.suffix+'.tmp')
                temporary.write_text(json.dumps(result,indent=2)+'\n')
                temporary.replace(args.report)
            stage('parent_finished' if result['success'] else 'parent_failed')
        return
    import tempfile
    stage('torch_import_started')
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    stage('torch_import_complete')
    # Fail explicitly before a worker can hang on a too-long AF_UNIX socket.
    if len(tempfile.gettempdir().encode()) + 48 >= 108:
        raise RuntimeError('TMPDIR is too long for multiprocessing AF_UNIX sockets')
    loader = DataLoader(TensorDataset(torch.arange(8)), batch_size=2,
                        num_workers=4, timeout=args.worker_timeout)
    stage('worker_transfer_started')
    values = torch.cat([batch[0] for batch in loader])
    if not torch.equal(values, torch.arange(8)):
        raise RuntimeError('Worker tensor-transfer preflight failed')
    stage('worker_transfer_complete')
    print('Four-worker tensor-transfer preflight passed; TMPDIR=' + tempfile.gettempdir(),flush=True)


if __name__ == '__main__':
    main()
