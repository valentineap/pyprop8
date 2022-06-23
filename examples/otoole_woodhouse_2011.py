import pyprop8 as pp
from pyprop8.utils import rtf2xyz,make_moment_tensor,stf_trapezoidal,stf_cosine,latlon2xy
import numpy as np
import matplotlib.pyplot as plt

'''This example aims to reproduce the various figures presented in
O'Toole & Woodhouse (2011, doi: 10.1111/j.1365-246X.2011.05210.x).

As various details are not fully and unambiguously defined in the paper, there
may be some minor differences between the figures there and those output by this
code.
'''


model_table_1 = pp.LayeredStructureModel([[ 3.00, 1.80, 0.00, 1.02],
                                          [ 2.00, 4.50, 2.40, 2.57],
                                          [ 5.00, 5.80, 3.30, 2.63],
                                          [20.00, 6.50, 3.65, 2.85],
                                          [np.inf,8.00, 4.56, 3.34]])

model_halfspace = pp.LayeredStructureModel([[np.inf,6.00,3.46,2.70]])

#Table 2 does not mention the properties of the underlying infinite halfspace.
#The following assumes this shares properties with the model of Table 1.
model_table_2 = pp.LayeredStructureModel([[ 1.50, 2.20, 1.00, 2.20],
                                          [ 6.50, 4.30, 2.30, 2.60],
                                          [ 7.00, 6.00, 3.40, 2.70],
                                          [ 6.00, 6.60, 3.70, 2.90],
                                          [ 6.00, 7.20, 4.00, 3.05],
                                          [np.inf,8.00, 4.56, 3.34]])

### Figure 1 ###
# The following two specifications of station location ought to be equivalent.
#stations = pp.RegularlyDistributedReceivers(30,200,18,90,90,1,depth=3)
stations = pp.ListOfReceivers(xx = np.zeros(18),yy=np.linspace(30,200,18),depth=3)
source =  pp.PointSource(0,0,34,rtf2xyz(make_moment_tensor(340,90,0,2.4E8,0,0)),np.zeros([3,1]), 0.)
tt,seis = pp.compute_seismograms(model_table_1, source, stations, 181,.5,xyz=False,source_time_function=lambda w:stf_trapezoidal(w,3,6))

fig = plt.figure(figsize=(6,10))
ax = fig.add_subplot(311)
ax.set_title("Radial")
for i in range(18):
    ax.plot(tt,seis[i,0,:]-50*i,'k')

ax = fig.add_subplot(312)
ax.set_title("Transverse")
for i in range(18):
    ax.plot(tt,seis[i,1,:]-50*i,'k')

ax = fig.add_subplot(313)
ax.set_title("Vertical")
for i in range(18):
    ax.plot(tt,seis[i,2,:]-50*i,'k')
plt.show()

### Figure 2 ###

# Compute on a regular grid of points and then interpolate
stations = pp.RegularlyDistributedReceivers(1,151,300,0,360,72)
static = pp.compute_static(model_halfspace,source,stations)
amax = abs(static).max()

fig = plt.figure(figsize=(3,8))
ax = fig.add_subplot(411)
ax.contourf(*stations.as_xy(),static[:,:,1],levels=np.linspace(-1.05*amax,1.05*amax,101),cmap=plt.cm.RdBu)
ax.set_xlim(-100,100)
ax.set_ylim(-100,100)
ax.set_aspect(1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('North')

ax = fig.add_subplot(412)
ax.contourf(*stations.as_xy(),static[:,:,0],levels=np.linspace(-1.05*amax,1.05*amax,101),cmap=plt.cm.RdBu)
ax.set_xlim(-100,100)
ax.set_ylim(-100,100)
ax.set_aspect(1)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('East')

ax = fig.add_subplot(413)
sc = ax.contourf(*stations.as_xy(),static[:,:,2],levels=np.linspace(-1.05*amax,1.05*amax,101),cmap=plt.cm.RdBu)
ax.set_xlim(-100,100)
ax.set_ylim(-100,100)
ax.set_aspect(1)
ax.set_title('Vertical')
ax = fig.add_subplot(414)
ax.set_aspect(.1)
c = plt.colorbar(sc,cax=ax,orientation='horizontal',label='Displacement (mm)')
c.set_ticks([-amax,0,amax])
plt.tight_layout()
plt.show()

### Figure 3
stations = pp.RegularlyDistributedReceivers(100,110,2,80,80,1)
source = pp.PointSource(0,0,10,rtf2xyz(make_moment_tensor(0,90,180,1.1E7,0,0)),np.zeros([3,1]), 0.)
tt,seis = pp.compute_seismograms(model_halfspace,source,stations,240,0.5,xyz=True,source_time_function = lambda w:stf_trapezoidal(w,6,3))

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(2,3,1)
ax.set_title("North")
ax.plot(tt,seis[0,1,:])
ax.set_xlim(0,120)
ax.set_xticks([0,60,120])
ax.set_xticklabels([])
ax = fig.add_subplot(2,3,2)
ax.set_title("East")
ax.plot(tt,seis[0,0,:])
ax.set_xlim(0,120)
ax.set_xticks([0,60,120])
ax.set_xticklabels([])
ax = fig.add_subplot(2,3,3)
ax.text(1.1,0.5,"r = 100km",transform=ax.transAxes)
ax.set_title("Vertical")
ax.plot(tt,seis[0,2,:])
ax.set_xlim(0,120)
ax.set_xticks([0,60,120])
ax.set_xticklabels([])
ax = fig.add_subplot(2,3,4)
ax.set_ylabel("Displacement (mm)")
ax.plot(tt,seis[1,1,:])
ax.set_xlim(0,120)
ax.set_xticks([0,60,120])
ax.set_xticklabels([])
ax = fig.add_subplot(2,3,5)
ax.plot(tt,seis[1,0,:])
ax.set_xlim(0,120)
ax.set_xticks([0,60,120])
ax.set_xticklabels([])
ax = fig.add_subplot(2,3,6)
ax.text(1.1,0.5,"r = 110km",transform=ax.transAxes)
ax.plot(tt,seis[1,2,:])
ax.set_xlim(0,120)
ax.set_xticks([0,60,120])
ax.set_xlabel("Time (s)")
plt.tight_layout()
plt.show()

### Figure 4 ###
# Only the model is different...
tt,seis = pp.compute_seismograms(model_table_2,source,stations,240,0.5,xyz=True,source_time_function = lambda w:stf_trapezoidal(w,6,3))

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(2,3,1)
ax.set_title("North")
ax.plot(tt,seis[0,1,:])
ax.set_xlim(0,120)
ax.set_xticks([0,60,120])
ax.set_xticklabels([])
ax = fig.add_subplot(2,3,2)
ax.set_title("East")
ax.plot(tt,seis[0,0,:])
ax.set_xlim(0,120)
ax.set_xticks([0,60,120])
ax.set_xticklabels([])
ax = fig.add_subplot(2,3,3)
ax.text(1.1,0.5,"r = 100km",transform=ax.transAxes)
ax.set_title("Vertical")
ax.plot(tt,seis[0,2,:])
ax.set_xlim(0,120)
ax.set_xticks([0,60,120])
ax.set_xticklabels([])
ax = fig.add_subplot(2,3,4)
ax.set_ylabel("Displacement (mm)")
ax.plot(tt,seis[1,1,:])
ax.set_xlim(0,120)
ax.set_xticks([0,60,120])
ax.set_xticklabels([])
ax = fig.add_subplot(2,3,5)
ax.plot(tt,seis[1,0,:])
ax.set_xlim(0,120)
ax.set_xticks([0,60,120])
ax.set_xticklabels([])
ax = fig.add_subplot(2,3,6)
ax.text(1.1,0.5,"r = 110km",transform=ax.transAxes)
ax.plot(tt,seis[1,2,:])
ax.set_xlim(0,120)
ax.set_xticks([0,60,120])
ax.set_xlabel("Time (s)")
plt.tight_layout()
plt.show()

### Figure 5 ###
# The paper does not unambiguously define the precise setup and processing used
# in this experiment, and the code I have from O'Toole is not sufficient to
# enable this to be reconstructed. The following gives results that are clearly
# close to the original, but not identical.

stations = pp.ListOfReceivers(np.array([22.383514667]),np.array([36.493270697]),depth=3,geometry='spherical')
source = pp.PointSource(21.79,36.24,30,rtf2xyz(make_moment_tensor(332,6,120,2.4E7,0,0)),np.zeros([3,1]), 0.)
tt,seis = pp.compute_seismograms(model_table_1,source,stations,240,0.5,xyz=True,source_time_function=lambda w:stf_cosine(w,4.5)*(1-np.exp(-1j*w*5))/(5*1j*w))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(tt,seis[0,:])
ax.set_title('East component at Kerya')
ax.set_xlim(0,120)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Displacement (mm)")
plt.tight_layout()
plt.show()
