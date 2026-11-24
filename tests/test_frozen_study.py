import pytest
from scripts.slurm.prepare_frozen_gaussian_study import configurations


def test_frozen_study_covers_relative_grid_and_preserves_controls():
    configs=configurations({'initial_learning_rate':.1,'bead_radius':1.5,'z_coupled_ratio':.5},8)
    assert len(configs)==12
    grid=[c for c in configs.values() if c['num_classes']==3]
    assert {tuple(c['ce_class_weights']) for c in grid}=={(a,1.,b) for a in [.05,.1,.5] for b in [.5,2.,4.]}
    for c in configs.values():
        assert c['freeze_slm'] and not c['freeze_unet']
        assert c['initial_phase_mask']=='gaussian' and c['gaussian_phase_mode']=='focused'
        assert c['z_depth_list']==[0] and c['emission_strength_range']==[.5,1.5]
        assert c['initial_learning_rate']==.1 and c['bead_radius']==1.5
        assert c['minimum_training_seconds']==c['maximum_training_seconds']==28800
    assert configs['one_sided']['tissue_sample_z_bounds_um']==[0,100]
    assert configs['two_sided']['tissue_sample_z_bounds_um']==[-100,100]
    assert configs['binary_bce_dice']['tissue_spatial_loss_weight']==0


def test_budget_must_be_explicit_and_bounded():
    for n in [-1,0,25]:
        with pytest.raises(ValueError):configurations({},n)
