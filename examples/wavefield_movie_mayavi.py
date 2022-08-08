import numpy as np 
import pyprop8 as pp
from pyprop8.utils import rtf2xyz, make_moment_tensor,stf_trapezoidal
import tqdm 
from mayavi import mlab
from scipy.interpolate import griddata

print(pp.__file__)
# This script generates an animated gif showing ground motion and
# line-of-sight static displacement (~ InSAR) for an event
lo_res = False # Set to True to generate a quick-and-dirty plot for checking layout etc
if lo_res:
    nt=180
    delta_t = 0.25
    ngrid=50
    ncontour=28
    dpi=100
    framestep=5
    nr = 20
    nphi=36
    stencil_args = {'kmin':0,'kmax':2,'nk':1200}
else:
    nt = 900      # Number of time-steps in seismogram calculation
    delta_t = 0.05 # Time interval in seismogram calculation
    ngrid=250     # Number of grid points in Cartesian grid for interpolation
    dpi=300       # resolution of output
    ncontour=140  # Number of contours in static displacement plot
    framestep=1   # Stride for animation
    nr=100         # Number of radii for seismogram calculation
    nphi=360       # Number of azimuths for seismogram calculation
    # To ensure a high-quality resolution of the surface, increase the
    # spatial (k-space) sampling
    stencil_args = {'kmin':0,'kmax':5,'nk':12000}
box_size = 100
# Earth model to use

# These values are Based on Crust1.0 for Idaho
model = pp.LayeredStructureModel([( 16.5, 6.1, 3.6 , 2.7),
                                 ( 14.6, 6.3, 3.7, 2.8),
                                 (  6.4, 7.0, 4.0, 3.0),
                                 (np.inf,7.9, 4.4, 3.3)])

# Earthquake parameters
# GCMT catalog values
depth = 13.8
moment_tensor_rtf = 1e6*np.array([[-2.320, 1.120, 1.150],
                                  [ 1.120, 1.780,-5.970],
                                  [ 1.150,-5.970, 0.535]])

moment_tensor = rtf2xyz(moment_tensor_rtf)  #make_moment_tensor(strike, dip, rake, M0, 0, 0))
source = pp.PointSource(0,0,depth,moment_tensor,np.zeros([3,1]),0)
stf = lambda w: stf_trapezoidal(w, 4.2, 8.4)

receivers = pp.RegularlyDistributedReceivers(1,100,nr,0,360,nphi,depth=0)
xx,yy = receivers.as_xy()

if False:
    time, seismograms = pp.compute_seismograms(model,source,receivers,nt,delta_t,source_time_function=stf,stencil_kwargs=stencil_args)
    with open('data.bin','wb') as fp:
        np.save(fp, time)
        np.save(fp, seismograms)
else:
    with open('data.bin','rb') as fp:
        time = np.load(fp)
        seismograms = np.load(fp)


xgrid, ygrid = np.meshgrid(np.linspace(-box_size,box_size,ngrid),np.linspace(-box_size,box_size,ngrid))
insar = np.zeros([ngrid,ngrid,3])
insar[:,:,0] = griddata((xx.flatten(),yy.flatten()),seismograms[:,:,0,-1].flatten(),(xgrid.flatten(),ygrid.flatten())).reshape((ngrid,ngrid))
insar[:,:,1] = griddata((xx.flatten(),yy.flatten()),seismograms[:,:,1,-1].flatten(),(xgrid.flatten(),ygrid.flatten())).reshape((ngrid,ngrid))
insar[:,:,2] = griddata((xx.flatten(),yy.flatten()),seismograms[:,:,2,-1].flatten(),(xgrid.flatten(),ygrid.flatten())).reshape((ngrid,ngrid))
mask = (xgrid**2+ygrid**2)**0.5 > 100
insar[mask,:] = np.nan


max_norm = np.linalg.norm(seismograms,2,axis=2).max()
max_vert = abs(seismograms[:,:,2,:]).max()

print([abs(seismograms[:,:,i,:]).max() for i in range(3)] )
scale_horiz = 0. #box_size/max_norm
scale_vert = 0.5 #scale_horiz
cmax = 1.05*scale_vert*max_vert
j=0
print("cmax=%f"%cmax,scale_vert,scale_horiz)
mlab.options.offscreen=True
fig = mlab.figure(size=(1024,1024))
#surf = mlab.mesh(xx+scale_horiz*seismograms[:,:,0,j],yy+scale_horiz*seismograms[:,:,1,j],scale_vert*seismograms[:,:,2,j])
# for i in tqdm.tqdm(range(nt)):
#     mlab.clf()
    
#     surf = mlab.mesh(xx+scale_horiz*seismograms[:,:,0,i],
#                      yy+scale_horiz*seismograms[:,:,1,i],
#                      scale_vert*seismograms[:,:,2,i],
#                      vmin=-cmax,vmax=cmax,#extent=[-100,100,-100,100,-cmax,cmax],
#                      colormap='RdYlBu',reset_zoom=False)
#     #mlab.outline()
#     mlab.axes(ranges=(-100,100,-100,100,-cmax,cmax),extent=(-100,100,-100,100,-cmax,cmax),x_axis_visibility=True,y_axis_visibility=True,z_axis_visibility=True)
#     mlab.view(azimuth=-45,elevation=60,focalpoint=(0,0,0),distance=400)


#     mlab.savefig("animation/test%04i.png"%i)
for i in tqdm.tqdm(range(nt,nt+360)):
    azim = -45+(i+1-nt)
    view_elev = 60
    mlab.clf()
    surf = mlab.mesh(xx+scale_horiz*seismograms[:,:,0,-1],
                     yy+scale_horiz*seismograms[:,:,1,-1],
                     scale_vert*seismograms[:,:,2,-1],
                     vmin=-cmax,vmax=cmax,#extent=[-100,100,-100,100,-cmax,cmax],
                     colormap='RdYlBu')
    los_vector = np.array([np.cos(np.deg2rad(azim))*np.sin(np.deg2rad(view_elev)),
                           np.sin(np.deg2rad(azim))*np.sin(np.deg2rad(view_elev)),
                           np.cos(np.deg2rad(view_elev))])
    
    img = mlab.imshow(insar.dot(los_vector)%28,colormap='jet',extent=[-100,100,-100,100,-2*cmax,-2*cmax])
    img.module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0
    img.update_pipeline()
    mlab.axes(ranges=(-100,100,-100,100,-2*cmax,cmax),extent=(-100,100,-100,100,-2*cmax,cmax),x_axis_visibility=True,y_axis_visibility=True,z_axis_visibility=True)
    mlab.view(azimuth=azim,elevation=view_elev,focalpoint=(0,0,-cmax),distance=400)

    

    mlab.savefig("animation/test%04i.png"%i)
mlab.show()

#ffmpeg -f image2 -i animation/test%04d.png test2.gif