import numpy as np 
import matplotlib.pyplot as plt 
import pyprop8 as pp
from pyprop8.utils import rtf2xyz, make_moment_tensor,stf_trapezoidal
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.tri import Triangulation, CubicTriInterpolator
from matplotlib.colors import LightSource
import tqdm 
model = pp.LayeredStructureModel([[ 3.00, 1.80, 0.00, 1.02],
                                  [ 2.00, 4.50, 2.40, 2.57],
                                  [ 5.00, 5.80, 3.30, 2.63],
                                  [20.00, 6.50, 3.65, 2.85],
                                  [np.inf,8.00, 4.56, 3.34]])

strike = 180
dip = 0
rake = 0
M0=1E7
depth = 15
moment_tensor = rtf2xyz(make_moment_tensor(strike, dip, rake, M0, 0, 0))
source = pp.PointSource(0,0,depth,moment_tensor,np.zeros([3,1]),0)
stf = lambda w: stf_trapezoidal(w, 3, 6)

receivers = pp.RegularlyDistributedReceivers(1,50,50,0,360,72,depth=3)
xx,yy = receivers.as_xy()
tri = Triangulation(xx.flatten(),yy.flatten())

nt = 120
delta_t = 0.5
time, seismograms = pp.compute_seismograms(model,source,receivers,nt,delta_t,source_time_function=stf)

fig = plt.figure()
ax1 = fig.add_axes((0.05,0.05,0.9,0.9),projection='3d')
ax2 = fig.add_axes((0.05,0.05,0.1,0.1))
#ax3 = fig.add_axes((0.7,0.05,0.25,0.25))

ngrid=200
grid_x,grid_y=np.meshgrid(np.linspace(-50,50,ngrid),np.linspace(-50,50,ngrid))
cmax=abs(seismograms).max()*1.05
scale_horiz=0.1
scale_vert = 0.1
view_azim_init = -45
view_elev = 30
nstatic = 120

def animate(i):
    azim = view_azim_init
    if i>nstatic:
        azim = view_azim_init+i-nstatic
    ax1.azim = azim
    ax1.elev = view_elev
    #view_azim=10
    #view_elev=30
    ls = LightSource(90, 30)
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
    

    rgb = ls.shade(zz, cmap=plt.cm.coolwarm_r,vmin=-cmax,vmax=cmax)
    surf = ax1.plot_surface(grid_x+scale_horiz*fx(grid_x,grid_y),
                           grid_y+scale_horiz*fy(grid_x,grid_y),
                           scale_vert*zz,
                           facecolors=rgb,antialiased=False,
                           rcount=ngrid,ccount=ngrid)
    ax1.set_xlim(-50,50)
    ax1.set_ylim(-50,50)
    ax1.set_xlabel("West-East")
    ax1.set_ylabel("South-North")
    ax1.set_zlim(-2*scale_vert*cmax,scale_vert*cmax)
    ax1.set_axis_off()
    ax2.clear()
    ax2.set_xlim(-2,2)
    ax2.set_ylim(-2,2)
    ax2.set_aspect(1.0)
    ax2.add_patch(plt.Circle((0,0),1,color='lightgrey'))
    ax2.text(0,1.2,"N",ha='center',va='bottom')
    ax2.text(0,-1.2,"S",ha='center',va='top')
    ax2.text(-1.2,0,"W",ha='right',va='center')
    ax2.text(1.2,0,"E",ha='left',va='center')
    ax2.set_axis_off()
    view = ax2.arrow(0,0,-np.cos(np.deg2rad(azim)),-np.sin(np.deg2rad(azim)))
    # ax3.clear()
    # ax3.set_xlim(-50,50)
    # ax3.set_ylim(-50,50)
    # ax3.set_aspect(1.0)
    if i>=nt:
        los_vector = np.array([np.cos(np.deg2rad(azim))*np.sin(np.deg2rad(90-view_elev)),
                       np.sin(np.deg2rad(azim))*np.sin(np.deg2rad(90-view_elev)),
                       np.cos(np.deg2rad(90-view_elev))])
        insar = seismograms[:,:,:,-1].dot(los_vector)
        interferogram = ax1.contourf(xx,yy,insar%28,280,offset=-2*scale_vert*cmax,cmap=plt.cm.jet)
    return surf, view, 
anim = FuncAnimation(fig,animate,interval=10,blit=True,frames=tqdm.tqdm(np.arange(0,nt+360)))
anim.save("test.gif",dpi=300,writer=PillowWriter(fps=25))