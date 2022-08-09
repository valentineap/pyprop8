import numpy as np 
import pyprop8 as pp
from pyprop8.utils import rtf2xyz, make_moment_tensor,stf_trapezoidal
import tqdm 
from mayavi import mlab
from scipy.interpolate import griddata

print(pp.__file__)
# This script generates an animated gif showing ground motion and
# line-of-sight static displacement (~ InSAR) for an event

nt = 401      # Number of time-steps in seismogram calculation
delta_t = 0.1 # Time interval in seismogram calculation
ngrid=560     # Number of grid points in Cartesian grid for interpolation
nr=200         # Number of radii for seismogram calculation
nphi=360       # Number of azimuths for seismogram calculation

frame_file_fmt = 'animation/frame_%04i.png'

# To ensure a high-quality resolution of the surface, increase the
# spatial (k-space) sampling. Probably doesn't need to be this high!
stencil_args = {'kmin':0,'kmax':7,'nk':20000}
box_size = 65


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


seismogram_file = 'seismograms.bin'
compute_seismograms=False # Set to True to compute seismograms rather than load
if compute_seismograms:
    time, seismograms = pp.compute_seismograms(model,source,receivers,nt,delta_t,
                                            source_time_function=stf,stencil_kwargs=stencil_args)
    if seismogram_file is not None:
        with open(seismogram_file,'wb') as fp:
            np.save(fp, time)
            np.save(fp, seismograms)
else:
    print("Attempting to load seismograms from file:")
    try:
        with open(seismogram_file,'rb') as fp:
            time = np.load(fp)
            seismograms = np.load(fp)
    except:
        print("Failed. Consider re-running with `compute_seismograms=True`")
        raise


# Construct regular grid for interpolation of functions
xgrid, ygrid = np.meshgrid(np.linspace(-box_size,box_size,ngrid),
                                np.linspace(-box_size,box_size,ngrid))

# Differentiate seismograms to get accelerograms and calculate overall amplitude -
# used to plot wavefronts
acc = np.linalg.norm(np.diff(seismograms,n=2),axis=2)
maxacc = acc.max()

# Assume that `nt` has been chosen so that seismograms[:,:,:,-1] represent static offset.
# Could use `pyprop8.compute_static` here but there's not really any need.
# Interpolate onto regular Cartesian grid (so that we can use `imshow` for plotting)
insar = np.zeros([ngrid,ngrid,3])
insar[:,:,0] = griddata((xx.flatten(),yy.flatten()),seismograms[:,:,0,-1].flatten(),(xgrid.flatten(),ygrid.flatten())).reshape((ngrid,ngrid))
insar[:,:,1] = griddata((xx.flatten(),yy.flatten()),seismograms[:,:,1,-1].flatten(),(xgrid.flatten(),ygrid.flatten())).reshape((ngrid,ngrid))
insar[:,:,2] = griddata((xx.flatten(),yy.flatten()),seismograms[:,:,2,-1].flatten(),(xgrid.flatten(),ygrid.flatten())).reshape((ngrid,ngrid))
# mask = (xgrid**2+ygrid**2)**0.5 > 100
# insar[mask,:] = np.nan


# Maximum vertical displacement - defines z-axis scaling in plot
max_vert = abs(seismograms[:,:,2,:]).max()

# Scale factors to convert seismic displacements (in mm) to graph coordinates (km)
scale_horiz = 0.5 #km/mm i.e. amplification of 500,000 x
scale_vert = 0.5 #km/mm

#Maximum for colour/z axis is 'a little more' than maximum displacement
cmax = 1.05*scale_vert*max_vert

# Get started with mayavi. Turn off interactive mode and create a figure.
mlab.options.offscreen=True
fig = mlab.figure(size=(720,720))

# Set various parameters controlling camera and display
ax_visible = False
ax_range = (-100,100,-100,100,-2*cmax,cmax) #Defines volume of view
focalpoint=(0,0,-cmax) # Where we are looking
focaldist=400 # How far away we are
view_elev = 60
hover = 0.02*cmax # Annotations etc hover above the z=0 plane so as not to be disrupted by waves
boxcolor=(0,0,0) # Color for box corner annotation
ticksize=5 # Size of box corners

# mayavi is supposed to do animations but I can't make it work properly. #ust generate each frame 
# separately and then stitch them together at the end.
#
# The plotting code could doubtless be made cleaner and nicer...
#
# First a loop for plotting wavefield evolution over time
do_wavefield=True # Useful to be able to turn this off and skip straight to second loop
if do_wavefield:
    for i in tqdm.tqdm(range(nt)):
        # Clear the image and start again
        mlab.clf()

        # Compute data for the image of wavefronts -> magnitude of acceleration at this instant
        accstep = np.zeros([ngrid,ngrid])
        if i==0 or i==nt-1:
            pass # We have lost two time steps due to the numerical differentiation
        else:
            # Interpolate the acceleration data.
            # Note that if timestep * wavespeed < distance between sampling points in xx/yy, aliasing will happen
            accstep = griddata((xx.flatten(),yy.flatten()),acc[:,:,i-1].flatten(),(xgrid.flatten(),ygrid.flatten())).reshape((ngrid,ngrid))
        
        # Plot the main surface-displacement image
        surf = mlab.mesh(xx+scale_horiz*seismograms[:,:,0,i],
                        yy+scale_horiz*seismograms[:,:,1,i],
                        scale_vert*seismograms[:,:,2,i],
                        vmin=-cmax,vmax=cmax,
                        colormap='seismic',reset_zoom=False)

        # Plot corners of square region used for lower plot
        mlab.plot3d([box_size-ticksize,box_size,box_size],[-box_size,-box_size,-box_size+ticksize],[hover,hover,hover],tube_radius=0.2,color=boxcolor)
        mlab.plot3d([-box_size+ticksize,-box_size,-box_size],[-box_size,-box_size,-box_size+ticksize],[hover,hover,hover],tube_radius=0.2,color=boxcolor)
        mlab.plot3d([box_size-ticksize,box_size,box_size],[box_size,box_size,box_size-ticksize],[hover,hover,hover],tube_radius=0.2,color=boxcolor)
        mlab.plot3d([-box_size+ticksize,-box_size,-box_size],[box_size,box_size,box_size-ticksize],[hover,hover,hover],tube_radius=0.2,color=boxcolor)

        # Plot the lower image showing wavefronts. Remember imshow needs array transposed!
        mlab.imshow(accstep.T,colormap='bone',extent=[-box_size,box_size,-box_size,box_size,-2*cmax,-2*cmax],vmin=0,vmax=maxacc)

        # Label the lower image
        mlab.text3d(-box_size-ticksize,-4*ticksize,-2*cmax,"%i x %ikm"%(2*box_size,2*box_size),scale=5,orient_to_camera=False,orientation=(0,0,90))
        mlab.text3d(-box_size+5*ticksize,box_size+ticksize,-2*cmax,"Ground acceleration",scale=5,orient_to_camera=False,orientation=(0,0,0))

        # Set the visible volume
        mlab.axes(ranges=ax_range,extent=ax_range,
                    x_axis_visibility=ax_visible,y_axis_visibility=ax_visible,z_axis_visibility=ax_visible)

        # Add some annotations
        mlab.text(0.05,0.025,"t=%05.02fs"%time[i],width=0.2)
        mlab.text(0.05,0.93,"Mw6.5, Stanley, Idaho",width=0.45)

        # Compass arrow
        mlab.quiver3d(np.array([0]),np.array([80]),np.array([hover]),np.array([0]),np.array([20]),np.array([0]),line_width=3,scale_factor=1,color=boxcolor)
        mlab.text(0,105,"N",z=0,width=0.01,color=boxcolor)
        mlab.text(0.45,0.075,"Made with pyprop8",width=0.45)
        mlab.text(0.4,0.025,"github.com/valentineap/pyprop8",width=0.55)

        # Set the camera position
        mlab.view(azimuth=-45,elevation=view_elev,focalpoint=focalpoint,distance=focaldist)

        # And save!
        mlab.savefig(frame_file_fmt%i)

# Second loop for rotating camera + InSAR
for i in tqdm.tqdm(range(nt,nt+360)):
    # View azimuth will increment frame-by-frame
    azim = -45+(i+1-nt)

    # Clear the figure
    mlab.clf()

    # Plot the main surface-displacement image
    surf = mlab.mesh(xx+scale_horiz*seismograms[:,:,0,-1],
                     yy+scale_horiz*seismograms[:,:,1,-1],
                     scale_vert*seismograms[:,:,2,-1],
                     vmin=-cmax,vmax=cmax,
                     colormap='seismic',
                    )
                
    # Plot 'corners' of square region used for lower plot
    mlab.plot3d([box_size-ticksize,box_size,box_size],[-box_size,-box_size,-box_size+ticksize],[hover,hover,hover],tube_radius=0.2,color=boxcolor)
    mlab.plot3d([-box_size+ticksize,-box_size,-box_size],[-box_size,-box_size,-box_size+ticksize],[hover,hover,hover],tube_radius=0.2,color=boxcolor)
    mlab.plot3d([box_size-ticksize,box_size,box_size],[box_size,box_size,box_size-ticksize],[hover,hover,hover],tube_radius=0.2,color=boxcolor)
    mlab.plot3d([-box_size+ticksize,-box_size,-box_size],[box_size,box_size,box_size-ticksize],[hover,hover,hover],tube_radius=0.2,color=boxcolor)

    # Compute line-of-sight vector corresponding to current look direction
    # Camera is pointed at a point *below* the z=0 plane so the elevation angle for 
    # our view of the z=0 surface is not the same as the elevation angle for the
    # camera. A bit of trigonometry gives
    #     tan phi = b sin th / (b cos th + a)
    # where a is the z-coordinate of the focal point, b is the camera-focal point 
    # distance, th is the camera elevation and phi is the effective viewing angle.
    los_elev = np.rad2deg(np.arctan2(focaldist*np.sin(view_elev),(focaldist*np.cos(view_elev) + focalpoint[2])))
    # Want vector *from* surface *to* satellite
    # This means that a +ve value in our image corresponds to motion *towards* the satellite
    los_vector = -np.array([np.cos(np.deg2rad(azim))*np.sin(np.deg2rad(los_elev)),
                            np.sin(np.deg2rad(azim))*np.sin(np.deg2rad(los_elev)),
                            np.cos(np.deg2rad(los_elev))])
    print(los_vector)

    # Make InSAR image. Again remember the transpose!
    img = mlab.imshow((insar.dot(los_vector).T+14)%28-14,colormap='jet',
                    extent=[-box_size,box_size,-box_size,box_size,-2*cmax,-2*cmax],
                    vmin=-14,vmax=14)
    img.module_manager.scalar_lut_manager.reverse_lut = True # Reverse the colormap

    # Add the colorbar and resize it
    colorbar = mlab.colorbar(img,orientation='vertical',nb_labels=3,label_fmt="%.0f mm")
    colorbar.scalar_bar_representation.position = [0.025, 0.15]
    colorbar.scalar_bar_representation.position2 = [0.1, 0.3]

    # Label the lower image
    mlab.text3d(-box_size-ticksize,-4*ticksize,-2*cmax,"%i x %ikm"%(2*box_size,2*box_size),scale=5,orient_to_camera=False,orientation=(0,0,90))
    mlab.text3d(-box_size-0.5*ticksize,box_size+ticksize,-2*cmax,"Wrapped line-of-sight displacement",scale=5,orient_to_camera=False,orientation=(0,0,0))

    # Set the visible volume
    mlab.axes(ranges=ax_range,extent=ax_range,
                x_axis_visibility=ax_visible,
                y_axis_visibility=ax_visible,
                z_axis_visibility=ax_visible)

    # Add some annotations
    mlab.text(0.05,0.025,"t=%05.02fs"%time[-1],width=0.2)
    mlab.text(0.05,0.93,"Mw6.5, Stanley, Idaho",width=0.45)
    # Compass arrow
    mlab.quiver3d(np.array([0]),np.array([80]),np.array([hover]),
                    np.array([0]),np.array([20]),np.array([0]),
                    line_width=3,scale_factor=1,color=boxcolor)
    mlab.text(0,105,"N",z=0,width=0.01,color=boxcolor)
    mlab.text(0.45,0.075,"Made with pyprop8",width=0.45)
    mlab.text(0.4,0.025,"github.com/valentineap/pyprop8",width=0.55)

    # Set the camera position
    mlab.view(azimuth=azim,elevation=view_elev,focalpoint=focalpoint,distance=focaldist)

    # And save!
    mlab.savefig(frame_file_fmt%i)
mlab.show()

# Now build a movie using the following command:
# ffmpeg  -stream_loop 3 -i animation/frame_%04d.png -vcodec libx264 idaho.mp4