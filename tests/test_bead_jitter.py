import random

import numpy as np
import pytest

import data.generators as generators


def config(**changes):
    return dict(num_particles_range=[400,401], particle_spatial_range_x=[-50,51],
                particle_spatial_range_y=[-50,51], particle_spatial_range_z=[-20,21],
                z_coupled_ratio=1.0, z_coupled_spacing_range=[2,3], **changes)


@pytest.mark.parametrize('explicit_zero',[False,True])
def test_zero_jitter_preserves_prechange_seeded_scene(explicit_zero):
    cfg={'num_particles_range':[8,9],'particle_spatial_range_x':[0,10],
         'particle_spatial_range_y':[0,10],'particle_spatial_range_z':[-6,7],
         'z_coupled_ratio':1.0,'z_coupled_spacing_range':[2,3]}
    if explicit_zero:cfg['z_coupled_lateral_jitter_px']=0
    random.seed(37);np.random.seed(23)
    beads,connections=generators.create_random_emitters(cfg)
    # Captured from the original generator before adding jitter support.
    np.testing.assert_array_equal(beads,[[7,9,-2],[6,6,1],[6,6,3],[5,7,2],
                                       [7,9,6],[9,1,3],[5,7,4],[9,1,5]])
    np.testing.assert_array_equal(connections,[[9,1,4],[5,7,3],[6,6,2]])


def test_offsets_are_inclusive_per_axis_and_not_twice_radius(monkeypatch):
    monkeypatch.setattr(random,'shuffle',lambda beads:None)
    random.seed(91);np.random.seed(91)
    cfg=config(z_coupled_lateral_jitter_px=3)
    beads,_=generators.create_random_emitters(cfg)
    pairs=beads.reshape(-1,2,3)
    delta=pairs[:,1,:2]-pairs[:,0,:2]
    assert (np.abs(delta)<=3).all()
    for axis in range(2):
        assert {-3,0,3} <= set(delta[:,axis])
    assert len(beads)==400 and np.issubdtype(beads.dtype,np.integer)


@pytest.mark.parametrize('radius',[1,3,1000])
def test_narrow_edge_ranges_stay_psf_safe(monkeypatch,radius):
    monkeypatch.setattr(random,'shuffle',lambda beads:None)
    random.seed(2);np.random.seed(2)
    cfg=config(z_coupled_lateral_jitter_px=radius)
    cfg.update(particle_spatial_range_x=[0,2],particle_spatial_range_y=[-2,0])
    beads,_=generators.create_random_emitters(cfg)
    assert set(beads[:,0]) <= {0,1}
    assert set(beads[:,1]) <= {-2,-1}
    assert (np.abs(beads[1::2,:2]-beads[::2,:2])<=radius).all()


def test_sparse_allowed_positions_are_respected(monkeypatch):
    monkeypatch.setattr(random,'shuffle',lambda beads:None)
    random.seed(5);np.random.seed(5)
    cfg=config(z_coupled_lateral_jitter_px=4)
    cfg.update(particle_spatial_range_x=range(0,31,3),particle_spatial_range_y=[-5,-1,3,8])
    beads,_=generators.create_random_emitters(cfg)
    assert set(beads[:,0]) <= set(cfg['particle_spatial_range_x'])
    assert set(beads[:,1]) <= set(cfg['particle_spatial_range_y'])
    assert (np.abs(beads[1::2,:2]-beads[::2,:2])<=4).all()


def test_connection_labels_follow_jittered_endpoints(monkeypatch):
    monkeypatch.setattr(random,'shuffle',lambda beads:None)
    monkeypatch.setattr(random,'choice',lambda values:values[0])
    monkeypatch.setattr(generators,'_jitter_coordinate',lambda anchor,allowed,radius:anchor+radius)
    cfg=config(z_coupled_lateral_jitter_px=3)
    cfg.update(num_particles_range=[2,3],particle_spatial_range_x=[0,10],
               particle_spatial_range_y=[0,10],particle_spatial_range_z=[-2,3],
               z_coupled_spacing_range=[3,3])
    beads,connections=generators.create_random_emitters(cfg)
    np.testing.assert_array_equal(beads,[[0,0,-2],[3,3,1]])
    np.testing.assert_array_equal(connections,[[1,1,-1],[2,2,0]])


def test_uncoupled_beads_and_rng_are_unchanged():
    cfg=config();cfg['z_coupled_ratio']=0
    def sample(radius):
        random.seed(12);np.random.seed(12)
        result=generators.create_random_emitters({**cfg,'z_coupled_lateral_jitter_px':radius})
        return result,random.getstate(),np.random.get_state()
    a,python_a,numpy_a=sample(0);b,python_b,numpy_b=sample(3)
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    assert python_a==python_b
    np.testing.assert_array_equal(numpy_a[1],numpy_b[1])
    assert numpy_a[2:]==numpy_b[2:]


@pytest.mark.parametrize('radius',[-1,1.5,3.0,True,False,'3',None])
def test_invalid_jitter_fails_clearly(radius):
    with pytest.raises(ValueError,match='nonnegative integer'):
        generators.create_random_emitters(config(z_coupled_lateral_jitter_px=radius))
