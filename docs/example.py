import numpy as np
import matplotlib.pyplot as plt

import pyprop8 as pp
from pyprop8.utils import stf_trapezoidal, make_moment_tensor, rtf2xyz


model = pp.LayeredStructureModel([[ 3.00, 1.80, 0.00, 1.02],
                                  [ 2.00, 4.50, 2.40, 2.57],
                                  [ 5.00, 5.80, 3.30, 2.63],
                                  [20.00, 6.50, 3.65, 2.85],
                                  [np.inf,8.00, 4.56, 3.34]])



strike = 340
dip = 90
rake = 0
scalar_moment = 2.4E8
Mxyz = rtf2xyz(make_moment_tensor(strike, dip, rake, scalar_moment,0,0))
F =np.zeros([3,1])

event_x = 0
event_y = 0
event_depth = 34
event_time = 0
source =  pp.PointSource(event_x, event_y, event_depth, Mxyz, F, event_time)

stations = pp.ListOfReceivers(xx = np.zeros(18),yy=np.linspace(30,200,18),depth=3)

nt = 181
dt = 0.5
tt,seis = pp.compute_seismograms(model, source, stations, nt, dt,
                                 xyz=False,source_time_function=lambda w:stf_trapezoidal(w,3,6))


fig = plt.figure(figsize=(5,8))
ax = fig.subplots(3,1)
ax[0].set_title("Radial")
ax[1].set_title("Transverse")
ax[2].set_title("Vertical")
for i in range(18):
    ax[0].plot(tt,seis[i,0,:]-50*i,'k')
    ax[1].plot(tt,seis[i,1,:]-50*i,'k')
    ax[2].plot(tt,seis[i,2,:]-50*i,'k')
for i in range(3):
    ax[i].set_xlim(0,90)
    ax[i].set_ylim(-1000,400)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
ax[2].set_xticks([0,30,60,90])
plt.tight_layout()
plt.savefig('fig1.png',dpi=300)
plt.show()



model_halfspace = pp.LayeredStructureModel([[np.inf,6.00,3.46,2.70]])
stations = pp.RegularlyDistributedReceivers(1,151,300,0,360,72)
static = pp.compute_static(model_halfspace,source,stations)
amax = abs(static).max()

fig = plt.figure(figsize=(8,5))
ax = fig.subplots(2,3)
ax[0,0].contourf(*stations.as_xy(),static[:,:,1],levels=np.linspace(-1.05*amax,1.05*amax,101),cmap=plt.cm.RdBu)
ax[0,1].contourf(*stations.as_xy(),static[:,:,0],levels=np.linspace(-1.05*amax,1.05*amax,101),cmap=plt.cm.RdBu)
sc = ax[0,2].contourf(*stations.as_xy(),static[:,:,2],levels=np.linspace(-1.05*amax,1.05*amax,101),cmap=plt.cm.RdBu)
for i in range(3):
    ax[0,i].set_xlim(-100,100)
    ax[0,i].set_ylim(-100,100)
    ax[0,i].set_aspect(1)
    ax[0,i].set_xticks([])
    ax[0,i].set_yticks([])
ax[0,0].set_title('North')
ax[0,1].set_title('East')
ax[0,2].set_title('Vertical')

ax[1,1].set_aspect(.1)
c = plt.colorbar(sc,cax=ax[1,1],orientation='horizontal',label='Displacement (mm)')
c.set_ticks([-amax,0,amax])
ax[1,0].set_visible(False)
ax[1,2].set_visible(False)
plt.tight_layout()
plt.savefig('fig2.png',dpi=300)
plt.show()
