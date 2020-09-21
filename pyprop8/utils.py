import numpy as np

def stf_trapezoidal(omega,trise,trupt):
    uu = np.ones(omega.shape,dtype='complex128')
    uxx = np.ones_like(uu)
    uex=np.ones_like(uu)
    wp = omega!=0
    uu[wp] = omega[wp]*trise*1j
    uu[wp] = (1-np.exp(-uu[wp]))/uu[wp]
    uxx[wp] = 1j*omega[wp]*trupt/2
    uex[wp] = np.exp(uxx[wp])
    uxx[wp] = (uex[wp]-1/uex[wp])/(2*uxx[wp])
    return uu*uxx
