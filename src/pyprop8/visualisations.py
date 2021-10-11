import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.colors import LightSource,Normalize
from mpl_toolkits.mplot3d import Axes3D
import tqdm
from pyprop8 import *

def generate_surface_movie(structure,source,stf = None):
    depth = 0
    if structure.mu[0] == 0:
        print("Model appears to have ocean layer; movie will show seafloor.")
        depth = structure.dz[0] # Skip ocean layer

    print ("Generating seismograms...")
    stations = RegularlyDistributedReceivers(1,100,100,0,360,360,depth = depth, degrees=True)
    # stations = ListOfReceivers()
    # ngrid = 25
    # xx,yy = np.meshgrid(np.linspace(-100,100,ngrid),np.linspace(-100,100,ngrid))
    # stations.from_xy(xx.flatten(),yy.flatten(),depth=depth)
    nt = 500
    dt = 0.1
    alpha=0.023
    tt,seis = compute_seismograms(structure,source,stations,nt,dt,alpha,source_time_function = stf,pad_frac=2)
    amax = abs(seis[:,:,2,:]).max()
    norm = Normalize(-amax,amax)
    xx,yy = stations.as_xy()
    ls = LightSource()
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.set_xticks([])
    ax.set_yticks([])
    ngrid = 200
    xxg,yyg = np.meshgrid(np.linspace(-50,50,ngrid),np.linspace(-50,50,ngrid))
    zero_centre = lambda m:np.where(np.sqrt(xxg**2+yyg**2)<5,0,m)
    c = ax.imshow(ls.shade(zero_centre(griddata(np.array([xx.flatten(),yy.flatten()]).T,seis[:,:,2,0].flatten(),\
                    np.array([xxg.flatten(),yyg.flatten()]).T,method='linear').reshape(ngrid,ngrid)),cmap=plt.cm.RdBu,norm=norm))
    #c.set_clim(-amax,amax)
    t = ax.text(0.8,0.05,'t=%.2f'%(0),transform=ax.transAxes)
    elements = [c,t]
    #prog = tqdm.tqdm(total=nt-1)
    def make_frame(n):
        # for x in elements[0].collections:
        #     x.remove()
        elements[0].set_data(ls.shade(zero_centre(griddata(np.array([xx.flatten(),yy.flatten()]).T,seis[:,:,2,n].flatten(),\
                        np.array([xxg.flatten(),yyg.flatten()]).T,method='linear').reshape(ngrid,ngrid)),cmap=plt.cm.RdBu,norm=norm))
        #elements[0].set_clim(-amax,amax)
        elements[1].set_text('t=%.2f'%tt[n])

        return elements
    a = anim.FuncAnimation(fig,make_frame,tqdm.tqdm(np.arange(1,nt)),blit=True,interval=40)
    #rprog.close()
    a.save('test.mp4')

    plt.show()


def generate_surface_movie2(structure,source,stf = None):
    depth = 0
    if structure.mu[0] == 0:
        print("Model appears to have ocean layer; movie will show seafloor.")
        depth = structure.dz[0] # Skip ocean layer

    print ("Generating seismograms...")
    stations = RegularlyDistributedReceivers(1,100,101,0,360,360,depth = depth, degrees=True)
    # stations = ListOfReceivers()
    # ngrid = 25
    # xx,yy = np.meshgrid(np.linspace(-100,100,ngrid),np.linspace(-100,100,ngrid))
    # stations.from_xy(xx.flatten(),yy.flatten(),depth=depth)
    nt = 200
    dt = .125
    alpha=0.023
    tt,seis = compute_seismograms(structure,source,stations,nt,dt,alpha,source_time_function = stf,pad_frac=2)
    amax = abs(seis[:,:,2,:]).max()
    norm = Normalize(-amax,amax)
    xx,yy = stations.as_xy()
    ls = LightSource(altdeg=45,azdeg=120)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111,projection='3d')
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zlim(-1.05*amax,amax*1.05)
    ax.axes.set_box_aspect((2,2,1))
    ngrid = 40
    xxg,yyg = np.meshgrid(np.linspace(-50,50,ngrid),np.linspace(-50,50,ngrid))
    zero_centre = lambda m:np.where(np.sqrt(xxg**2+yyg**2)<5,0,m)
    # print(ls.shade(zero_centre(griddata(np.array([xx.flatten(),yy.flatten()]).T,seis[:,:,2,0].flatten(),\
    #                 np.array([xxg.flatten(),yyg.flatten()]).T,method='linear').reshape(ngrid,ngrid)),cmap=plt.cm.RdBu,norm=norm).shape)
    zzg = zero_centre(griddata(np.array([xx.flatten(),yy.flatten()]).T,seis[:,:,2,0].flatten(),\
                    np.array([xxg.flatten(),yyg.flatten()]).T,method='linear').reshape(ngrid,ngrid))
    rgb = ls.shade(zzg,cmap=plt.cm.RdBu,norm=norm,blend_mode='soft')
    c = ax.plot_surface(xxg,yyg,zzg,facecolors=rgb,linewidth=0,rstride=1,cstride=1,antialiased=True,shade=False)
    c._facecolors2d=c._facecolors3d
    c._edgecolors2d=c._edgecolors3d
    #c.set_clim(-amax,amax)
    #t = ax.text(0.8,0.05,'t=%.2f'%(0),transform=ax.transAxes)
    elements = [c]
    #prog = tqdm.tqdm(total=nt-1)
    def make_frame(n):
        # for x in elements[0].collections:
        #     x.remove()
        elements[0].remove()
        zzg = zero_centre(griddata(np.array([xx.flatten(),yy.flatten()]).T,seis[:,:,2,n].flatten(),\
                        np.array([xxg.flatten(),yyg.flatten()]).T,method='linear').reshape(ngrid,ngrid))
        rgb = ls.shade(zzg,cmap=plt.cm.RdBu,norm=norm,blend_mode='soft')
        elements[0] = ax.plot_surface(xxg,yyg,zzg,facecolors=rgb,linewidth=0,rstride=1,cstride=1,antialiased=True,shade=False)
                    # np.array([xxg.flatten(),yyg.flatten()]).T,method='linear').reshape(ngrid,ngrid)),cmap=plt.cm.RdBu,norm=norm,lightsource=ls,linewidth=0,rstride=1,cstride=1)
        elements[0]._facecolors2d=elements[0]._facecolors3d
        elements[0]._edgecolors2d=elements[0]._edgecolors3d
        #elements[0].set_clim(-amax,amax)
        #elements[0].set_clim(-amax,amax)
        #elements[1].set_text('t=%.2f'%tt[n])

        return elements
    a = anim.FuncAnimation(fig,make_frame,tqdm.tqdm(np.arange(1,nt)),blit=True,interval=50)
    #rprog.close()
    a.save('test.mp4')
    #plt.close(fig)
    plt.show()
