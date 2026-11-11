import numpy as np

from scripts.evaluate_checkpoint import detection_counts,pixel_counts,scores


def test_pixel_confusion_counts_and_undefined_empty_scores():
    np.testing.assert_array_equal(pixel_counts(np.array([[.8,.8],[.2,.2]]),
                                   np.array([[1,0],[1,0]])),[[1,1],[1,1]])
    s=scores(np.zeros((2,2),dtype=int))
    assert s['precision'] is None and s['recall'] is None and s['f1'] is None


def test_component_matching_is_one_to_one_and_respects_radius():
    truth=np.zeros((10,10));truth[3,3]=truth[3,5]=truth[9,9]=1
    probability=np.zeros_like(truth);probability[3,4]=.9
    # One component near two targets can match only one; the distant bead is missed.
    assert detection_counts(probability,truth)==dict(tp=1,fp=0,fn=2,target_points=3,predicted_components=1)
    assert detection_counts(probability,truth,radius=0)['tp']==0


def test_empty_target_retains_false_positive_components():
    p=np.zeros((5,5));p[0,0]=p[4,4]=.9
    d=detection_counts(p,np.zeros_like(p))
    assert d['tp']==0 and d['fn']==0 and d['fp']==2
