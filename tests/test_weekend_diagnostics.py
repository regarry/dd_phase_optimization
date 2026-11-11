import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.weekend_diagnostics import bead_categories, confusion, metrics


def test_exact_confusion_and_zero_denominators():
    matrix = confusion(np.array([[.1,.9],[.1,.9]]), np.array([[0,0],[1,1]]))
    np.testing.assert_array_equal(matrix, [[1,1],[1,1]])
    result=metrics(matrix)
    assert result['precision']==result['recall']==result['dice']==.5
    assert metrics(np.zeros((2,2),dtype=int))['precision']==0


def test_3d_colors_preserve_projection_ambiguity():
    xyz=np.array([[0,0,0],[1,0,0],[0,0,1],[1,0,1]])
    categories=bead_categories(xyz,np.array([[.9,.1]]),{'z_range_cost_function':[0,0]})
    np.testing.assert_array_equal(categories,[0,1,3,2])


def test_preflight_rejects_long_path_before_workers(tmp_path):
    directory=tmp_path/('long_'+'x'*160)
    directory.mkdir()
    result=subprocess.run([sys.executable,'-m','utils.worker_preflight','--child'],
                          env={**os.environ,'TMPDIR':str(directory)},capture_output=True,
                          text=True,timeout=35)
    assert result.returncode!=0
    assert 'TMPDIR is too long' in result.stderr
