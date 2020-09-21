import numpy as np
import pyprop8 as pp
from utils import stf_trapezoidal

def test_derivatives(model,source,stations,nt=257,dt=0.5,alpha = 0.023,pad_frac = 1,derivatives=None,delta = 1e-3,source_time_function=None):
    '''Compute finite-difference derivatives and compare with those output by code'''
    if source_time_function is None: source_time_function = lambda w:stf_trapezoidal(w,3,6)
    if derivatives is None:
        derivatives = pp.DerivativeSwitches(r=True,phi=True)
    nDimSta = stations.nDim
    if derivatives.nderivs==0:
        raise ValueError("All derivatives are 'switched off'!")
    elif derivatives.nderivs>1:
        nDimDerivs = 1
    else:
        nDimDerivs = 0
    nDimChannels = 1
    if nt>1:
        nDimTime = 1
    else:
        nDimTime = 0

    deriv_comp = lambda sl: tuple(nDimSta*[slice(None)]+nDimDerivs*[sl]+nDimChannels*[slice(None)]+nDimTime*[slice(None)])


    tt,seis0,drv = pp.compute_seismograms(model,source,stations,nt,dt,alpha,pad_frac = pad_frac,derivatives=derivatives,source_time_function=source_time_function)
    assert len(seis0.shape)==nDimSta+nDimChannels+nDimTime
    assert len(drv.shape) == nDimSta+nDimChannels+nDimTime+nDimDerivs
    if derivatives.r:
        sta_pert = stations.copy()
        sta_pert.rr+=delta
        tt,seis = pp.compute_seismograms(model,source,sta_pert,nt,dt,alpha,pad_frac = pad_frac,derivatives=None,source_time_function=source_time_function)
        fd = (seis - seis0)/delta
        err = drv[deriv_comp(derivatives.i_r)]-fd




    return err
