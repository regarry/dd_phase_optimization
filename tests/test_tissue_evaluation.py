import numpy as np
import pytest
from scripts.evaluate_tissue_study import confusion,class_scores,spatial_regions


def test_three_class_bead_metrics_do_not_credit_connection_predictions():
    matrix=confusion(np.array([[0,1,1,2]]),np.array([[0,1,2,1]]),3)
    np.testing.assert_array_equal(matrix,[[1,0,0],[0,1,1],[0,1,0]])
    score=class_scores(matrix)[1]
    assert score['tp']==score['fp']==score['fn']==1
    assert score['f1']==.5


def test_invalid_prediction_does_not_silently_change_confusion_size():
    with pytest.raises(ValueError):confusion(np.array([0]),np.array([3]),3)


def test_spatial_regions_match_tissue_loss_including_boundary():
    z=np.array([-100,-5,0,5,6,100,101])
    tissue,core,off=spatial_regions(z,dict(tissue_z_bounds_um=[0,100],tissue_core_halfwidth_um=5))
    np.testing.assert_array_equal(core,[False,False,True,True,False,False,False])
    np.testing.assert_array_equal(off,[False,False,False,False,True,True,False])


def test_single_gpu_inference_removes_only_identical_optical_replicas():
    import torch
    from scripts.evaluate_tissue_study import inference_state_dict
    saved={'physics.replicas.0.buffer':torch.tensor([1.]),
           'physics.replicas.1.buffer':torch.tensor([1.]),'unet.weight':torch.tensor([7.])}
    target={key:value for key,value in saved.items() if '.replicas.1.' not in key}
    actual=inference_state_dict(saved,target,1)
    assert set(actual)==set(target)
    assert actual['unet.weight'] is saved['unet.weight']
    saved['physics.replicas.1.buffer']=torch.tensor([2.])
    with pytest.raises(ValueError,match='replica state differs'):inference_state_dict(saved,target,1)


def test_inference_adapter_refuses_missing_learned_state():
    import torch
    from scripts.evaluate_tissue_study import inference_state_dict
    with pytest.raises(ValueError,match='no partial loading'):
        inference_state_dict({'unet.weight':torch.tensor([1.])},{'unet.bias':torch.tensor([1.])},1)


def test_cuda_allocator_initializes_before_resetting_peak_metrics():
    from scripts.evaluate_tissue_study import initialize_cuda_metrics
    calls=[]
    class Cuda:
        def init(self):calls.append('initialized')
        def reset_peak_memory_stats(self,device):
            assert calls and calls[0]=='initialized'
            calls.append(device)
    initialize_cuda_metrics(Cuda(),2)
    assert calls==['initialized',0,1]


def test_one_sided_comparison_includes_airy_reference():
    from scripts.evaluate_tissue_study import comparison_names
    assert comparison_names('one_sided_gaussian')==('learned','gaussian','axicon','airy')
    assert comparison_names('dice_bce_axicon')==('learned','gaussian','axicon')


def test_expected_camera_dark_offset_noise_and_percentile_target():
    from scripts.optical_report import expected_camera
    cfg=dict(robust_max_percentile=.999,camera_max_adu=65535.,camera_gain=.46,target_saturation_ratio=.01,epsilon=1e-6,quantum_efficiency=.6,dark_current_mean=120.,read_noise_std=60.,min_clip_value=0.)
    expected,std,scale=expected_camera(np.zeros((8,8)),cfg)
    np.testing.assert_allclose(expected,55.2)
    np.testing.assert_allclose(std,np.sqrt(3720)*.46)
    expected,std,scale=expected_camera(np.ones((8,8)),cfg)
    np.testing.assert_allclose(expected,(65535*.01*.6+55.2),rtol=2e-6)
