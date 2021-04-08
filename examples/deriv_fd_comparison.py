import pyprop8 as pp
from pyprop8.utils import rtf2xyz,make_moment_tensor,stf_trapezoidal,stf_cosine,latlon2xy
import numpy as np
import matplotlib.pyplot as plt


# The main point of this example is to compare the analytic derivatives computed by
# pyprop to finite-difference approximations. We compute spectra and seismograms, and
# check the radial, azimuthal, x and y derivatives.
#
# For each case, a figure is produced. This shows:
# - A histogram of the element-by-element fractional error, i.e. the difference betwen
#   FD and analytical derivatives as a proportion of the analytical derivative for every
#   point in space and time or frequency
# - A histogram of the trace-by-trace fractional error, defined as max(abs(fd-drv))/max(abs(drv))
#   for each trace (seismogram or spectrum)
# - Plots of FD and analytical derivatives for the trace with the largest trace-by-trace fractional
#   error
# - A plot of the difference between the two traces in the previous plot.
#
# In general a good agreement should be seen. The exception is where stations lie on
# (or close to) the nodal planes in the wavefield. 



def compare(a,b,title='',aname='a',bname='b',eps=1e-8,thresh=1e-6,fig=True):
    '''Compare two arrays and print various statistics quantifying their similarity. Optionally generate figure.'''
    assert a.shape==b.shape,'Cannot compare arrays of different shape'
    print('')
    print('%s   [eps = %e thresh = %e]'%(title,eps,thresh))
    print('='*len(title))


    lname = len(aname)+2*len(bname)+12

    header = ' '*(lname+1)+'max(|X|)'.rjust(13)+'mean(|X|)'.rjust(13)+'median(|X|)'.rjust(12)
    print(header)
    print(' '+'-'*(len(header)-1))
    for x,xn in zip((a,b,a-b,(a-b)/(abs(b)+eps)),(aname,bname,' %s-%s'%(aname,bname),'(%s-%s)/(|%s|+eps)'%(aname,bname,bname))):
        print('%s %12.5e %12.5e %12.5e'%(xn.rjust(lname+1),abs(x).max(),abs(x).mean(),np.median(abs(x))))
    print('')
    worst = np.unravel_index(np.argmax(abs(a-b).max(-1)/abs(b).max(-1)),a.shape[:-1])+(slice(None),)
    worst_score = abs(a-b)[worst].max()/abs(b[worst]).max()
    print(" Worst trace-by-trace fractional error: %.5e"%worst_score)
    if worst_score<thresh:
        print(" *** PASS ***")
        print(" %s and %s agree"%(aname,bname))
        rval = True
    else:
        print(" *** FAIL ***")
        print(" %s and %s do not agree"%(aname,bname))
        rval = False
    print('')
    print('')

    if fig:
        f = plt.figure()
        ax = f.add_subplot(311)
        hbins = np.linspace(-8,-2,100)
        ax.hist(np.log10(eps+abs((a-b)/(abs(b)+eps)).flatten()),hbins,density=True,label='Element-by-element')
        ax.hist(np.log10((abs(a-b).max(-1)/abs(b).max(-1)).flatten()),hbins,density=True,alpha = 0.5,label='Trace-by-trace')
        ax.set_title("Fractional error")
        ax.set_xlabel("log10(error)")
        ax.set_yticks([])
        ax.legend()




        ax = f.add_subplot(312)
        if np.any(np.iscomplex(a)):
            ax.plot(np.real(a[worst]),'k-',label='real(%s)'%aname)
            ax.plot(np.imag(a[worst]),'k--',label='imag(%s)'%aname)
        else:
            ax.plot(np.real(a[worst]),'k-',label=aname)

        if np.any(np.iscomplex(b)):
            ax.plot(np.real(b[worst]),'r-',label='real(%s)'%bname)
            ax.plot(np.imag(b[worst]),'r--',label='imag(%s)'%bname)
        else:
            ax.plot(b[worst],'r-',label=bname)
        ax.legend()
        ax.set_xticks([])
        ax = f.add_subplot(313)
        if np.any(np.iscomplex(a-b)):
            ax.plot(np.real((a-b)[worst]),'k-',label='real(%s-%s)'%(aname,bname))
            ax.plot(np.imag((a-b)[worst]),'k--',label='imag(%s-%s)'%(aname,bname))
        else:
            ax.plot((a-b)[worst],'k-',label='%s-%s'%(aname,bname))
        ax.legend()
        ax.set_xticks([])
        f.suptitle(title)
        plt.tight_layout()

        plt.show()
    return rval

# Model from O'Toole & Woodhouse (2009)
model = pp.LayeredStructureModel([[ 3.00, 1.80, 0.00, 1.02],
                                          [ 2.00, 4.50, 2.40, 2.57],
                                          [ 5.00, 5.80, 3.30, 2.63],
                                          [20.00, 6.50, 3.65, 2.85],
                                          [np.inf,8.00, 4.56, 3.34]])

# We want to check both spectra and derivatives, and use common settings across both.
nt = 121
dt = 0.5
pad_frac = 0.5
npad = int(pad_frac*nt)
tt = np.arange(nt+npad)*dt
alpha = None
stf = lambda w:stf_trapezoidal(w,3,6)

if alpha is None:
    # Use 'rule of thumb' given in O'Toole & Woodhouse (2011)
    alpha = np.log(10)/tt[-1]
ww = 2*np.pi*np.fft.rfftfreq(nt+npad,dt)
delta_omega = ww[1]
ww=ww-alpha*1j

delta = 1e-6
stations = pp.RegularlyDistributedReceivers(30,180,100,0,360,10,depth=3)
source =  pp.PointSource(0,0,34,rtf2xyz(make_moment_tensor(339,90,0,2.4E8,0,0)),np.zeros([3,1]), 0.)
derivatives = pp.DerivativeSwitches(r=True,phi=True,x=True,y=True)

# Radial perturbation
stations_r_pert = stations.copy()
stations_r_pert.rmin += delta
stations_r_pert.rmax += delta

spec, dspec = pp.compute_spectra(model,source,stations,ww,derivatives)
spec_r_pert = pp.compute_spectra(model,source,stations_r_pert,ww)
fd = (spec_r_pert - spec)/delta
compare(fd,dspec[:,:,derivatives.i_r,:,:],title='Radial derivative: spectrum',aname='fd',bname='drv')

tt,seis,dseis = pp.compute_seismograms(model,source,stations,nt,dt,derivatives=derivatives,xyz=False,source_time_function=stf)
tt,seis_r_pert = pp.compute_seismograms(model,source,stations_r_pert,nt,dt,xyz=False,source_time_function=stf)
fd = (seis_r_pert - seis)/delta
compare(fd,dseis[:,:,derivatives.i_r,:,:],title='Radial derivative: time series',aname='fd',bname='drv')

# Azimuthal perturbation
stations_phi_pert = stations.copy()
delta_phi = 1e-6
stations_phi_pert.phimin += delta_phi
stations_phi_pert.phimax += delta_phi

spec_phi_pert = pp.compute_spectra(model,source,stations_phi_pert,ww)
fd = (spec_phi_pert - spec)/(2*np.pi*delta_phi/360)
compare(fd,dspec[:,:,derivatives.i_phi,:,:],title='Azimuthal derivative: spectrum',aname='fd',bname='drv')

tt,seis_phi_pert = pp.compute_seismograms(model,source,stations_phi_pert,nt,dt,xyz=False,source_time_function=stf)
fd = (seis_phi_pert - seis)/(2*np.pi*delta_phi/360)
compare(fd,dseis[:,:,derivatives.i_phi,:,:],title='Azimuthal derivative: time series',aname='fd',bname='drv')

# Convert to ListOfReceivers to allow x/y finite difference calculations
# without implicitly moving stations
xx,yy = stations.as_xy()
stations_xy = pp.ListOfReceivers(xx.flatten(),yy.flatten(),depth=3)
spec_xy,dspec_xy = pp.compute_spectra(model,source,stations_xy,ww,derivatives)

# x perturbation
source_x_pert = source.copy()
source_x_pert.x+=delta
spec_x_pert = pp.compute_spectra(model,source_x_pert,stations_xy,ww)
fd = (spec_x_pert-spec_xy)/delta
compare(fd,dspec_xy[:,derivatives.i_x,:,:],title='x derivative: spectrum',aname='fd',bname='drv')

tt,seis_xy,dseis_xy = pp.compute_seismograms(model,source,stations_xy,nt,dt,xyz=False,derivatives=derivatives,source_time_function=stf)
tt,seis_x_pert = pp.compute_seismograms(model,source_x_pert,stations_xy,nt,dt,xyz=False,source_time_function=stf)
fd = (seis_x_pert-seis_xy)/delta
compare(fd,dseis_xy[:,derivatives.i_x,:,:],title='x derivative: time series',aname='fd',bname='drv')

# y perturbation
source_y_pert = source.copy()
source_y_pert.y+=delta

spec_y_pert = pp.compute_spectra(model,source_y_pert,stations_xy,ww)
fd = (spec_y_pert-spec_xy)/delta
compare(fd,dspec_xy[:,derivatives.i_y,:,:],title='y derivative: spectrum',aname='fd',bname='drv')

tt,seis_y_pert = pp.compute_seismograms(model,source_y_pert,stations_xy,nt,dt,xyz=False,source_time_function=stf)
fd = (seis_y_pert-seis_xy)/delta
compare(fd,dseis_xy[:,derivatives.i_y,:,:],title='y derivative: time series',aname='fd',bname='drv')
