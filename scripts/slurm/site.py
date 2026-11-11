"""Portable Slurm settings supplied locally, never copied from an account."""
import os
import shlex
import sys


def settings():
    return dict(python=os.environ.get('DD_PYTHON',sys.executable),
                account=os.environ.get('DD_SLURM_ACCOUNT',''),
                partition=os.environ.get('DD_SLURM_PARTITION',''),
                qos=os.environ.get('DD_SLURM_QOS',''),
                smoke_qos=os.environ.get('DD_SLURM_SMOKE_QOS',''),
                gres=os.environ.get('DD_SLURM_GRES','gpu:2'),
                modules=os.environ.get('DD_MODULES',''))


def directives(qos=None):
    site=settings();lines=[]
    for key,value in [('account',site['account']),('partition',site['partition']),
                      ('qos',site['qos'] if qos is None else qos)]:
        if value:
            if not all(c.isalnum() or c in '_.:-' for c in value):
                raise ValueError('Invalid Slurm '+key)
            lines.append('#SBATCH --'+key+'='+value)
    if not all(c.isalnum() or c in '_.:-,' for c in site['gres']):
        raise ValueError('Invalid Slurm GRES')
    lines.append('#SBATCH --gres='+site['gres'])
    return '\n'.join(lines)


def module_setup():
    names=shlex.split(settings()['modules'])
    return '\n'.join('module load '+shlex.quote(name) for name in names)
