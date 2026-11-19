"""Scanned-sheet intensity metrics, with explicit finite-window censoring.

Remacha et al. BOE 11:8–26 (2020), doi:10.1364/BOE.11.000008:
main-lobe width at 1/e intensity, optical sectioning at 1-1/e power,
and propagation length to doubling of optical-sectioning width. Our x denotes
their propagation y. All supplied coordinates are micrometres.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar


def lobe_width(z,intensity,peak=None,fraction=1/np.e):
    z=np.asarray(z,float);p=np.asarray(intensity,float)
    j=int(np.argmax(p) if peak is None else peak);level=p[j]*fraction
    if p[j]<=0:return None
    left=j;right=j
    while left>0 and p[left-1]>level:left-=1
    while right<len(p)-1 and p[right+1]>level:right+=1
    if left==0 or right==len(p)-1:return None
    lo=float(np.interp(level,[p[left-1],p[left]],[z[left-1],z[left]]))
    hi=float(np.interp(level,[p[right+1],p[right]],[z[right+1],z[right]]))
    return hi-lo


def power_interval(z,intensity,fraction=1-1/np.e):
    z=np.asarray(z,float);p=np.maximum(np.asarray(intensity,float),0)
    if z.ndim!=1 or p.shape!=z.shape or len(z)<2 or not np.all(np.diff(z)>0):
        raise ValueError('Power integration needs matching increasing coordinates')
    if not 0<fraction<=1:raise ValueError('Power fraction must be in (0,1]')
    c=np.r_[0.,cumulative_trapezoid(p,z)];total=c[-1]
    if not total>0:return None
    c=c/total
    def inverse(q,side):
        # A zero-intensity gap creates a flat CDF. At exactly that probability,
        # starts use its right edge and ends its left edge. Collapsing duplicate
        # CDF values before interpolation would smear power across dark regions.
        q=np.asarray(q);right=np.clip(np.searchsorted(c,q,side=side),1,len(c)-1)
        left=right-1;span=c[right]-c[left]
        return z[left]+np.divide(q-c[left],span,out=np.zeros_like(q,dtype=float),where=span>0)*(z[right]-z[left])
    def width(q):return float(inverse(q+fraction,'left')-inverse(q,'right'))
    starts=c[(c>=0)&(c<=1-fraction)];ends=c-fraction;ends=ends[(ends>=0)&(ends<=1-fraction)]
    grid=np.unique(np.r_[np.linspace(0,1-fraction,1025),starts,ends])
    values=inverse(grid+fraction,'left')-inverse(grid,'right');i=values.argmin()
    lo=grid[max(0,i-1)];hi=grid[min(len(grid)-1,i+1)]
    fit=minimize_scalar(width,bounds=(lo,hi),method='bounded',options={'xatol':1e-10}) if hi>lo else None
    q=float(fit.x) if fit is not None and fit.fun<values[i] else float(grid[i])
    a=float(inverse(q,'right'));b=float(inverse(q+fraction,'left'))
    return dict(width_um=b-a,lower_um=a,upper_um=b,power_fraction=fraction)


def doubling_distance(x,widths,focus_index,direction):
    target=2*widths[focus_index];j=focus_index
    while 0<=j+direction<len(x):
        k=j+direction
        if widths[k]>=target:
            coordinate=float(np.interp(target,[widths[j],widths[k]],[x[j],x[k]]))
            return dict(distance_um=abs(coordinate-x[focus_index]),censored=False)
        j=k
    return dict(distance_um=abs(float(x[j]-x[focus_index])),censored=True)


def beam_metrics(x,z,profiles):
    x=np.asarray(x,float);z=np.asarray(z,float);p=np.asarray(profiles,float)
    if p.shape!=(len(x),len(z)) or not np.isfinite(p).all() or np.min(p)<0:raise ValueError('Invalid profile grid')
    focus=int(np.argmin(abs(x)));row=p[focus];peak=int(np.argmax(row))
    intervals=[power_interval(z,r) for r in p]
    if any(i is None for i in intervals):raise ValueError('Zero-power profile')
    widths=np.array([i['width_um'] for i in intervals]);near=np.flatnonzero(abs(z)<=20)
    central=int(near[np.argmax(row[near])])
    candidates=np.flatnonzero(np.isclose(widths,widths.min(),rtol=1e-5,atol=1e-8))
    waist=int(candidates[np.argmin(abs(x[candidates]))])
    left=doubling_distance(x,widths,waist,-1);right=doubling_distance(x,widths,waist,1)
    return dict(focus_x_um=float(x[focus]),main_lobe_1e_um=lobe_width(z,row,peak),
        main_lobe_peak_z_um=float(z[peak]),on_axis_lobe_1e_um=lobe_width(z,row,central),
        on_axis_peak_z_um=float(z[central]),fwhm_um=lobe_width(z,row,peak,.5),
        optical_sectioning_63_um=widths[focus],power_interval_at_focus=intervals[focus],
        os_waist_x_um=float(x[waist]),os_waist_width_um=float(widths[waist]),waist_at_field_edge=bool(waist in (0,len(x)-1)),
        os_widths_um=widths.tolist(),propagation_x_um=x.tolist(),
        generalized_range_left=left,generalized_range_right=right,
        generalized_length_lower_bound_um=left['distance_um']+right['distance_um'],
        range_note='OS doubling distances from minimum observed OS waist; not a Gaussian sqrt(2) Rayleigh fit. Censored values are lower bounds.',
        finite_z_window_um=[float(z[0]),float(z[-1])],
        edge_power_fraction=float((row[:8].sum()+row[-8:].sum())/row.sum()),
        width_note='Illumination only after scan integration, not combined detection PSF resolution; 1um grid limits thin-lobe accuracy.')
