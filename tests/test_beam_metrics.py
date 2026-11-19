import numpy as np
from scipy.special import erfinv
from utils.beam_metrics import beam_metrics,lobe_width,power_interval


def test_gaussian_width_and_energy_interval():
    z=np.linspace(-20,20,4001);sigma=2.;p=np.exp(-z*z/(2*sigma*sigma))
    assert abs(lobe_width(z,p)-2*np.sqrt(2)*sigma)<.001
    expected=2*np.sqrt(2)*sigma*erfinv(1-1/np.e)
    assert abs(power_interval(z,p)['width_um']-expected)<.002


def test_distant_lobes_expand_sectioning_without_expanding_local_lobe():
    z=np.linspace(-50,50,2001);p=np.exp(-z*z/2)+.8*np.exp(-(z-20)**2/2)+.8*np.exp(-(z+20)**2/2)
    assert lobe_width(z,p)<3
    assert power_interval(z,p)['width_um']>20


def test_unobserved_os_doubling_is_censored_and_asymmetry_preserved():
    x=np.arange(-5,6);z=np.linspace(-20,20,401);profiles=np.array([np.exp(-z*z/(2*(1+max(v,0))**2)) for v in x])
    metrics=beam_metrics(x,z,profiles)
    assert metrics['generalized_range_left']['censored']
    assert not metrics['generalized_range_right']['censored']


def test_gaussian_os_doubling_range_is_sqrt_three_times_classic_rayleigh():
    x=np.linspace(-15,15,61);z=np.linspace(-80,80,1001);zr=5.
    profile=np.array([np.exp(-z*z/(2*(1+(v/zr)**2))) for v in x])
    metrics=beam_metrics(x,z,profile)
    for name in ('generalized_range_left','generalized_range_right'):
        assert not metrics[name]['censored']
        assert abs(metrics[name]['distance_um']-np.sqrt(3)*zr)<.06


def test_zero_padding_does_not_smear_power_across_empty_sample_space():
    z=np.arange(5.);p=np.array([0.,1.,1.,1.,0.])
    base=power_interval(z,p,fraction=1.)
    wide_z=np.arange(-400.,405.);wide=np.zeros_like(wide_z);wide[400:405]=p
    extended=power_interval(wide_z,wide,fraction=1.)
    assert base==extended
    assert base['width_um']==4.


def test_shortest_power_window_can_start_after_a_dark_gap():
    z=np.arange(101.);p=np.zeros_like(z);p[1:4]=1.;p[80:84]=3.
    interval=power_interval(z,p,.6)
    assert interval['lower_um']>75
    assert interval['width_um']<5
