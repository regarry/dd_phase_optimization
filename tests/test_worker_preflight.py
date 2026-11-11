"""The startup gate must remain bounded and fail closed with persistent evidence."""
import json
import subprocess
from unittest.mock import patch

import pytest

from utils.worker_preflight import main


@pytest.mark.parametrize('failure', [subprocess.TimeoutExpired('worker',180),
                                    subprocess.CalledProcessError(1,'worker')])
def test_failure_is_reported_and_propagated(tmp_path,failure):
    report=tmp_path/'result.json'
    with patch('sys.argv',['preflight','--report',str(report)]), \
         patch('utils.worker_preflight.subprocess.run',side_effect=failure) as run:
        with pytest.raises(type(failure)):
            main()
    result=json.loads(report.read_text())
    assert not result['success'] and result['error']
    assert result['total_timeout_seconds']>result['worker_timeout_seconds']
    assert run.call_count==1  # No silent retry or zero-worker fallback.


@pytest.mark.parametrize('value',['0','-1','nan','inf'])
def test_invalid_timeout_fails_before_starting_workers(value):
    with patch('sys.argv',['preflight','--worker-timeout',value]), \
         patch('utils.worker_preflight.subprocess.run') as run:
        with pytest.raises(SystemExit):
            main()
    run.assert_not_called()
