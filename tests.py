import numpy as np
import pyprop8 as pp

def test_derivatives(model,source,stations,nt=257,dt=0.5,alpha = 0.023,pad_frac = 1,derivatives=None):
    '''Compute finite-difference derivatives and compare with those output by code'''

    if derivatives is None:
        derivatives = pp.DerivativeSwitches(r=True,phi=True)
    tt,seis0,drv = pp.compute_seismograms(model,source,stations,nt,dt,alpha,pad_frac = pad_frac)
