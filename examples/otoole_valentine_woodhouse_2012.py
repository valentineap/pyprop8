import pyprop8 as pp
from pyprop8.utils import rtf2xyz,make_moment_tensor,stf_trapezoidal,stf_cosine,latlon2xy,clp_filter
import numpy as np
import matplotlib.pyplot as plt

'''This example aims to reproduce the first couple of figures presented in
O'Toole, Valentine & Woodhouse (2012, doi: 10.1111/j.1365-246X.2012.05608.x).'''

# Table 1:
model = pp.LayeredStructureModel(np.array([[ 0.10, 3.20, 2.00, 2.10],
                                           [ 1.90, 5.15, 2.85, 2.50],
                                           [ 3.00, 5.50, 3.20, 2.60],
                                           [13.00, 6.00, 3.46, 2.70],
                                           [14.00, 6.70, 3.87, 2.80],
                                           [np.inf,7.70, 4.30, 3.30]]))
print(model)

stations = pp.RegularlyDistributedReceivers(39.6,39.6,1,90-118.2,90-118.2,1,degrees=True).asListOfReceivers()

# Table 2, 'Iteration 0' column
event = pp.PointSource(0,0,35,rtf2xyz(np.array([[ 0.3406, 0.0005, 0.1610],
                                                [ 0.0005, 0.7798, 0.1430],
                                                [ 0.1610, 0.1430, 0.6349]])),np.array([[0.],[0.],[0.]]),0)
drv = pp.DerivativeSwitches(moment_tensor=True,z=True,x=True,y=True,time=True)

stf = lambda w: stf_trapezoidal(w,3,6)*clp_filter(w,0.05*2*np.pi,0.2*2*np.pi)
tt,seis,deriv = pp.compute_seismograms(model,event,stations,81,0.5,source_time_function = stf,derivatives=drv,pad_frac=0.5)

nez = [1,0,2] #Reorder seismogram components to match O'Toole's figure
# Native ordering of moment tensor components is as follows:
#    Mxx, Myy, Mzz, Mxy, Mxz, Myz
# Geometrical conversions:
#    dx -> dphi
#    dy -> -dtheta
#    dz -> dr
# Hence native ordering is equivalent to:
#    Mpp, Mtt, Mrr, -Mtp, Mrp -Mrt
dcomp = [2,1,0,5,4,3]
dcompsign = [1,1,1,-1,1,-1]
complabel=[r"$M_{rr}$ or $M_{zz}$",
           r"$M_{\theta\theta}$ or $M_{yy}$",
           r"$M_{\phi\phi}$ or $M_{xx}$",
           r"$M_{r\theta}$ or $-M_{yz}$",
           r"$M_{r\phi}$ or $M_{xz}$",
           r"$M_{\theta\phi}$ or $-M_{xy}$"]

amax = 1.025*abs(deriv[drv.i_mt:drv.i_mt+6,:,:]).max()
fig = plt.figure()
for idrv in range(6):
    for icomp in range(3):
        ax = fig.add_subplot(6,3,(3*idrv)+icomp+1)
        if idrv==0:
            if icomp == 0:
                ax.set_title("North")
            elif icomp == 1:
                ax.set_title("East")
            elif icomp == 2:
                ax.set_title("Vertical")
        ax.plot(tt,np.zeros_like(tt),'k:')
        ax.plot(tt,dcompsign[idrv]*deriv[drv.i_mt+dcomp[idrv],nez[icomp],:])
        ax.set_ylim(-amax,amax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        if icomp==0: ax.text(0,.7,complabel[idrv],transform=ax.transAxes)
plt.tight_layout()
plt.show()

fig = plt.figure()
deriv[drv.i_z,:,:]*=10
deriv[drv.i_x,:,:]*=10
deriv[drv.i_y,:,:]*=10
deriv[drv.i_time,:,:]*=2
amax = abs(deriv[drv.i_mt+6:,:,:]).max()
amax = 1.025*max([amax,abs(seis).max()])
dcomp = [drv.i_z,drv.i_y,drv.i_x,drv.i_time]
complabel = ["Depth","Latitude","Longitude","Time"]
for idrv in range(4):
    for icomp in range(3):
        ax = fig.add_subplot(5,3,(3*idrv)+icomp+1)
        if idrv==0:
            if icomp == 0:
                ax.set_title("North")
            elif icomp == 1:
                ax.set_title("East")
            elif icomp == 2:
                ax.set_title("Vertical")
        ax.plot(tt,np.zeros_like(tt),'k:')
        ax.plot(tt,deriv[dcomp[idrv],nez[icomp],:])
        ax.set_ylim(-amax,amax)
        ax.axis('off')
        if icomp==0: ax.text(0,.7,complabel[idrv],transform=ax.transAxes)
for icomp in range(3):
    ax = fig.add_subplot(5,3,13+icomp)
    ax.plot(tt,np.zeros_like(tt),'k:')
    ax.plot(tt,seis[nez[icomp],:])
    ax.set_ylim(-amax,amax)
    ax.axis('off')
    if icomp==0: ax.text(0,.7,"HRGPS synthetic",transform=ax.transAxes)
plt.tight_layout()
plt.show()
