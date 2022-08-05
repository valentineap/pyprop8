import numpy as np 
import matplotlib.pyplot as plt 
import pyprop8 as pp
from pyprop8.utils import rtf2xyz, make_moment_tensor,stf_trapezoidal
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.tri import Triangulation, CubicTriInterpolator
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d import art3d as art3d
import tqdm 

# This script generates an animated gif showing ground motion and
# line-of-sight static displacement (~ InSAR) for an event

lo_res = False # Set to True to generate a quick-and-dirty plot for checking layout etc
if lo_res:
    nt=60
    delta_t = 1
    ngrid=40
    ncontour=28
    dpi=100
    framestep=5
    nr = 10
    nphi=36
else:
    nt = 240      # Number of time-steps in seismogram calculation
    delta_t = 0.25 # Time interval in seismogram calculation
    ngrid=200     # Number of grid points in Cartesian grid for interpolation
    dpi=300       # resolution of output
    ncontour=280  # Number of contours in static displacement plot
    framestep=1   # Stride for animation
    nr=50         # Number of radii for seismogram calculation
    nphi=72       # Number of azimuths for seismogram calculation
box_size = 50

# Earth model to use

# These values are Based on Crust1.0 for Idaho
model = pp.LayeredStructureModel([( 16.5, 6.1, 3.6 , 2.7),
                                 ( 14.6, 6.3, 3.7, 2.8),
                                 (  6.4, 7.0, 4.0, 3.0),
                                 (np.inf,7.9, 4.4, 3.3)])

# Earthquake parameters
# strike = 180
# dip = 0
# rake = 0
# M0=1E7
# GCMT catalog values
depth = 13.8
moment_tensor_rtf = 1e6*np.array([[-2.320, 1.120, 1.150],
                                  [ 1.120, 1.780,-5.970],
                                  [ 1.150,-5.970, 0.535]])

moment_tensor = rtf2xyz(moment_tensor_rtf)  #make_moment_tensor(strike, dip, rake, M0, 0, 0))
source = pp.PointSource(0,0,depth,moment_tensor,np.zeros([3,1]),0)
stf = lambda w: stf_trapezoidal(w, 4.2, 8.4)

receivers = pp.RegularlyDistributedReceivers(1,50,nr,0,360,nphi,depth=0)
xx,yy = receivers.as_xy()
tri = Triangulation(xx.flatten(),yy.flatten())

# To ensure a high-quality resolution of the surface, increase the
# spatial (k-space) sampling
stencil_args = {'kmin':0,'kmax':5,'nk':12000}

time, seismograms = pp.compute_seismograms(model,source,receivers,nt,delta_t,source_time_function=stf,stencil_kwargs=stencil_args)

fig = plt.figure(figsize=(8,8))
ax1 = fig.add_axes((0.0,0.0,1,1),projection='3d')
ax2 = fig.add_axes((0.85,0.05,0.1,0.1),projection='3d')
ax3 = fig.add_axes((0.05,0.05,0.1,0.1))


grid_x,grid_y=np.meshgrid(np.linspace(-box_size,box_size,ngrid),np.linspace(-box_size,box_size,ngrid))


max_norm = np.linalg.norm(seismograms,2,axis=2).max()

scale_horiz = box_size/max_norm
scale_vert = scale_horiz

#cmax=abs(seismograms).max()*1.05
cmax = max_norm*1.05
#scale_horiz=0.01
#scale_vert = 0.1
view_azim_init = -45
view_elev = 30
nstatic = nt

def animate(i):
    # Plot the ground surface (with displacement)
    azim = view_azim_init
    if i>nstatic:
        azim = view_azim_init+i-nstatic
    ax1.azim = azim
    ax1.elev = view_elev
    ls = LightSource(90-ax1.azim+view_azim_init, 30)
    if i<nt:
        fx = CubicTriInterpolator(tri,seismograms[:,:,0,i].flatten())
        fy = CubicTriInterpolator(tri,seismograms[:,:,1,i].flatten())
        fz = CubicTriInterpolator(tri,seismograms[:,:,2,i].flatten())
    else:
        fx = CubicTriInterpolator(tri,seismograms[:,:,0,-1].flatten())
        fy = CubicTriInterpolator(tri,seismograms[:,:,1,-1].flatten())
        fz = CubicTriInterpolator(tri,seismograms[:,:,2,-1].flatten())
    ax1.clear()

    zz = fz(grid_x,grid_y)
    

    rgb = ls.shade(zz, cmap=plt.cm.coolwarm_r,vmin=-cmax,vmax=cmax,blend_mode='overlay')
    surf = ax1.plot_surface(grid_x+scale_horiz*fx(grid_x,grid_y),
                           grid_y+scale_horiz*fy(grid_x,grid_y),
                           scale_vert*zz,
                           facecolors=rgb,antialiased=True,
                           rcount=ngrid,ccount=ngrid)
    ax1.set_xlim(-50,50)
    ax1.set_ylim(-50,50)
    ax1.set_zlim(-2*scale_vert*cmax,scale_vert*cmax)
    ax1.set_axis_off()
    # Plot the compass rose
    ax2.clear()
    ax2.set_xlim(-2,2)
    ax2.set_ylim(-2,2)
    ths = np.linspace(0,np.pi*2,1000)
    view, = ax2.plot(2*np.sin(ths),2*np.cos(ths),zs=0,color='k')
    view, = ax2.plot([-0.2,0,0.2,-0.2],[1.75,2.5,1.75,1.75],zs=0,color='k')
    view, = ax2.plot([0,0],[-2,-2.25],zs=0,color='k')
    view, = ax2.plot([2,2.25],[0,0],zs=0,color='k')
    view, = ax2.plot([-2,-2.25],[0,0],zs=0,color='k')
    ax2.azim = azim
    ax2.elev = view_elev
    ax2.set_axis_off()
    # Plot the clock
    ax3.clear()
    ax3.set_xlim(-2,2)
    ax3.set_ylim(-2,2)
    ax3.set_aspect(1.0)
    ax3.add_patch(plt.Circle((0,0),1,color='lightgrey'))
    ax3.set_axis_off()
    if i<nt:
        clock = ax3.arrow(0,0,np.sin((i*delta_t)*np.pi/30),np.cos((i*delta_t)*np.pi/30))
        clock2 = ax3.arrow(0,0,0.6*np.sin((i*delta_t)*np.pi/1800),0.6*np.cos((i*delta_t)*np.pi/1800))
        ax3.text(0,-1.2,"%.1f s"%(i*delta_t),ha='center',va='top')
    else:  
        clock = ax3.arrow(0,0,np.sin((nt*delta_t)*np.pi/30),np.cos((nt*delta_t)*np.pi/30))
        clock2 = ax3.arrow(0,0,0.6*np.sin((nt*delta_t)*np.pi/1800),0.6*np.cos((i*delta_t)*np.pi/1800))
        ax3.text(0,-1.2,"%.1f s"%(nt*delta_t),ha='center',va='top')
    # Plot the InSAR
    if i>=nt:
        los_vector = np.array([np.cos(np.deg2rad(azim))*np.sin(np.deg2rad(90-view_elev)),
                       np.sin(np.deg2rad(azim))*np.sin(np.deg2rad(90-view_elev)),
                       np.cos(np.deg2rad(90-view_elev))])
        insar = seismograms[:,:,:,-1].dot(los_vector)
        insar = fx(grid_x,grid_y)*los_vector[0]+fy(grid_x,grid_y)*los_vector[1]+zz*los_vector[2]
        # interferogram = ax1.contourf(xx,yy,insar%28,ncontour,offset=-2*scale_vert*cmax,cmap=plt.cm.jet)
        interferogram = ax1.contourf(grid_x,grid_y,insar%28,ncontour,offset=-2*scale_vert*cmax,cmap=plt.cm.jet)
    return surf, view, clock
    #return  view, clock
anim = FuncAnimation(fig,animate,interval=10,blit=True,frames=tqdm.tqdm(np.arange(0,nt+360,framestep)))
anim.save("test.gif",dpi=dpi,writer=PillowWriter(fps=25))