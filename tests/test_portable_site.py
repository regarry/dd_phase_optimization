from scripts.slurm.site import directives,settings
import pytest


def test_site_defaults_do_not_embed_an_account(monkeypatch):
    for key in ['DD_SLURM_ACCOUNT','DD_SLURM_PARTITION','DD_SLURM_QOS','DD_SLURM_GRES']:
        monkeypatch.delenv(key,raising=False)
    assert settings()['account']==''
    text=directives()
    assert '--account=' not in text and '--partition=' not in text
    assert '--gres=gpu:2' in text


def test_account_and_gpu_request_are_explicit_and_validated(monkeypatch):
    monkeypatch.setenv('DD_SLURM_ACCOUNT','example_account')
    monkeypatch.setenv('DD_SLURM_GRES','gpu:example_type:2')
    text=directives('example_qos')
    assert '--account=example_account' in text and '--gres=gpu:example_type:2' in text
    monkeypatch.setenv('DD_SLURM_ACCOUNT','unsafe\n#SBATCH --extra')
    with pytest.raises(ValueError):directives()
