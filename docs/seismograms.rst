
.. _walkthrough:

===============================
Calculating seismic observables
===============================
``pyprop8`` can calculate three kinds of seismic observables: seismic spectra, displacement seismograms, and static offsets (e.g. for simulating GPS or InSAR data). This document describes how to set up and perform such simulations. For additional details, see the :ref:`reference`.

-------------
Problem setup
-------------
In order to calculate seismograms or seismic spectra, we need to specify three things:

1. The properties of the earth model;
2. The seismic source; and
3. The receivers at which the wavefield is to be recorded.

Earth model
===========

In ``pyprop8``, the earth model is assumed to be an infinite, layered halfspace. Each layer is characterised by four parameters: a layer thickness ('thickness', km), a P-wave velocity ('vp', km/s), an S-wave velocity ('vs', km/s), and a density ('rho', g/cm^3).

To represent a model, we need to create an instance of :py:class:`~pyprop8.LayeredStructureModel`. The easiest way to achieve this is to initialise the class, passing a list (or list-like object) containing the properties of each layer. Two formats are supported.

1. Each entry in the list is a tuple ``(thickness, vp, vs, rho)``. The first entry in the list is assumed to correspond to the uppermost (surface) layer, with subsequent entries representing successively deeper layers. The final entry in the list specifies the properties of the underlying half-space, and its thickness must be specified as ``np.inf``. This is the default format for :py:class:`~pyprop8.LayeredStructureModel`.

2. Alternatively, each entry in the list may be a tuple ``(ztop, vp, vs, rho)``, where ``ztop`` represents the depth (in km) to the interface defining the top of the layer. In this case, list entries need *not* be ordered. One entry must have `ztop=0`, defining the properties of the surface layer; the ordering and thickness of all other layers will be automatically inferred. If there is duplication of interface depths within the list, only the first will be retained. The properties associated with the lowermost interface specified are deemed to represent the underlying halfspace. To use this format, pass the optional argument ``interface_depth_form=True`` when initialising :py:class:`~pyprop8.LayeredStructureModel`.

The choice of format is simply a matter of user preference and convenience; it has no impact on the manner in which the simulations proceed. Thus::

  layers = [(3.0,    1.8,   0, 1.02),
            (5.0,    4.5, 2.4, 2.5),
            (np.inf, 8.0, 4.5, 3.0)]
  model = pp.LayeredStructureModel(layers)

will behave identically to::

  layers = [(0.0, 1.8,   0, 1.02),
            (3.0, 4.5, 2.4, 2.5),
            (8.0,   8, 4.5, 3.0)]
  model = pp.LayeredStructureModel(layers, interface_depth_form=True)

It will be noted that the surface layer in this case has a shear-wave velocity of 0 km/s. This signifies that it is a fluid -- this particular model might correspond to a location within an ocean. Only the uppermost layer is allowed to be a fluid.

Once you have created your model, you can call ``print(model)`` to obtain a graphical representation:

  >>> print(model)
  ------------------------------------------------------- z = 0.00 km
    vp = 1.80 km/s       FLUID        rho = 1.02 g/cm^3
  ------------------------------------------------------- z = 3.00 km
    vp = 4.50 km/s   vs = 2.40 km/s   rho = 2.50 g/cm^3
  ------------------------------------------------------- z = 8.00 km
    vp = 8.00 km/s   vs = 4.50 km/s   rho = 3.00 g/cm^3



Receivers
=========

Next we need to specify receiver locations. ``pyprop8`` supports two formats, which result in different computation paths internally and hence different costs.

The first is designed to be used when the wavefield is sought at isolated, pre-determined locations - perhaps locations chosen to correspond to a deployment of instruments. In this case, we need to create an instance of :py:class:`~pyprop8.ListOfReceivers`. This requires the x- and y-coordinates of each station, and the depth at which instruments are buried (which should be given as '0' if receivers lie on the surface)::

   stations = np.array([[ 5.3,  6.2],
                        [-2.1,  0.3],
                        [ 1.5, -3.1]]) # 3 pairs of (x,y) coordinates
   depth = 4 # Seafloor instruments
   receivers = pp.ListOfReceivers(stations[:,0],stations[:,1],depth)

The computation algorithm currently requires all receivers to lie at a common depth. If multiple depths are required, it will be necessary to divide receivers between multiple :py:class:`~pyprop8.ListOfReceivers` objects and perform simulations for each.

:py:class:`~pyprop8.ListOfReceivers` also accepts an optional argument, ``geometry``. By default, this has the value ``geometry='cartesian'``, implying that x- and y-coordinates are expressed relative to a Cartesian kilometre grid (i.e. a location (1,2) lies 1km east and 2km north of some arbitrarily-chosen origin point). The source location (see below) is assumed to be specified within the same coordinate system, and this is used to determine the source-receiver distances and azimuths used internally within the code. Alternatively, ``geometry='spherical'`` may be specified. In this case, x- and y-coordinates are treated as degrees longitude and degrees latitude. This is convenient for approximating real-world scenarios, and source-receiver distances and azimuths are calculated assuming great-circle propagation on the surface of a sphere of radius 6371km. However, ``pyprop8`` remains based on a flat-Earth approximation, and its validity degrades as the source-receiver distance increases.

Alternatively, we may wish to achieve a general characterisation of the wavefield throughout a region. For this case, ``pyprop8`` provides :py:class:`~pyprop8.RegularlyDistributedReceivers`, which assumes that receivers lie on a regular polar grid, centred upon the event location. This creates geometric simplifications that can be exploited by the computation algorithm, substantially accelerating calculations. For this case, we must specify ranges for radius and azimuth, and the number of grid points for each::

   # 5 equally spaced radii between 10km and 50km (i.e. [10 20 30 40 50])
   rmin, rmax = (10, 50)
   nr = 5
   # 8 equally spaced azimuths between 0 and 315 degrees (i.e. every 45 degrees)
   phimin, phimax = (0, 360)
   nphi = 8
   # Receivers may still be buried
   depth = 0
   receivers = pp.RegularlyDistributedReceivers(rmin,rmax,nr,phimin,phimax,nphi,depth)

This will result in a regular polar grid of 40 stations. By default, it is assumed that the minimum and maximum azimuths are measured in degrees (counter-clockwise from the x/East axis, when viewed from above); pass an optional argument ``degrees=False`` to use radians.

As already mentioned, the use of :py:class:`~pyprop8.RegularlyDistributedReceivers` requires the event to lie at the centre of the polar grid. The location of this central point can be specified with respect to a global Cartesian coordinate system by passing optional arguments ``x0=`` and ``y0=``. If these are omitted, the grid is assumed to be centred on the origin of the Cartesian system, and the source location should also be specified (see below) as ``(x,y) = (0,0)``.

Seismic source
==============

``pyprop8`` assumes that the seismic source acts at a single point in space (and time; but see the discussion of source time-functions, below). To represent a seismic source, we need to create an instance of :py:class:`~pyprop8.PointSource`. This requires us to specify the source location (an x-coordinate, a y-coordinate, and a depth); the source mechanism (expressed as a co-located moment tensor and force vector); and an event time::

   event_x, event_y, event_dep = ( 5, -3, 10) # Spatial location
   M = np.array([[ 1, 0, 0],
                 [ 0,-1, 0],
                 [ 0, 0, 0]]) # Moment tensor, expressed in Cartesian system
   F = np.array([[ 0],
                 [ 0],
                 [ 0]]) # Force vector, expressed in Cartesian system
   event_time = datetime.datetime.fromisoformat("2021-08-17T03:45:37") # Date/time
   source = pp.PointSource(event_x, event_y, event_dep, M, F, event_time)

The lateral (x/y) source coordinates should be specified in the same coordinate system as is used for the receivers (see above). This may be either a Cartesian kilometre grid, with arbitrary origin; or longitude and latitude coordinates specified in degrees.

Moment tensors and force vectors are expressed in a Cartesian, z-up system. The function :py:func:`utils.rtf2xyz` is available to convert moment tensors from the spherical-polar definition that is ubiquitous in global seismology (e.g. if catalogue source mechanisms are to be used). Additionally, the function :py:func:`utils.make_moment_tensor` is available to construct a moment tensor (in a spherical-polar system) given strike, dip and rake angles and magnitude information. Note that both moment tensor and force vector must be supplied, although all entries may be zero.

The moment tensor is represented by a 3x3 array, and the force vector by a 1x3 array. It is also possible to specify multiple moment tensor/force vector combinations within a single instance of :py:class:`~pyprop8.PointSource`::

   M = np.array([[[ 1, 0, 0],
                  [ 0,-1, 0],
                  [ 0, 0, 0]],
                 [[ 0, 0, 0],
                  [ 0, 0, 0],
                  [ 0, 0, 0]]]) # Two 3x3 moment tensors
   F = np.array([[[ 0],
                  [ 0],
                  [ 0]],
                 [[ 1],
                  [ 0],
                  [ 0]]]) # Two 3x1 force vectors
   source = pp.PointSource(event_x, event_y, event_dep, M, F, event_time)

In this case, the shape of the moment tensor array becomes Nx3x3, and that of the force vector Nx1x3. As described below, ``pyprop8`` will perform separate simulations for each of the N moment tensor/force vector pairs, returning N sets of simulated data. This capability is exploited internally to assist in calculation of derivatives, but may have limited value for general use: any computational efficiencies are modest, and potentially offset by the growth in memory demands.


-------------------------
Obtaining seismic spectra
-------------------------

Once the model, receivers and source have been created, obtaining seismic spectra (i.e., the spectrum of the wavefield at each receiver) can be as simple as::

    om_min, om_max = (0,5*2*np.pi)
    nfreq = 1000
    omegas = np.linspace(om_min,om_max,nfreq) # Frequencies at which spectrum is sought
    spectra = compute_spectra(model, source, receivers, omegas, squeeze_outputs=False) # Do the calculation

This creates the (complex) array ``spectra``, containing a spectrum of the wavefield at each receiver. The shape of the array depends on the choice made when specifying the receivers.

1. If ``receivers`` is an instance of :py:class:`~pyprop8.ListOfReceivers`, ``spectra`` will have shape ``(source.nsources, receivers.nstations, 3, nfreq)``, where ``source.nsources`` is the number of moment tensor/source vector pairs specified within the ``source`` object, ``receivers.nstations`` is the total number of receivers, and ``nfreq`` is the number of frequency points at which evaluation was requested. The third dimension indexes the three components of motion: radial, transverse, and vertical.

2. If ``receivers`` is an instance of :py:class:`~pyprop8.RegularlyDistributedReceivers`, ``spectra`` will have shape ``(source.nsources, receivers.nr, receivers.nphi, 3, nfreq)``, where ``receivers.nr`` and ``receivers.nphi`` are the number of grid points in the radial and azimuthal directions, respectively.

However, if ``squeeze_outputs=True`` (which is the default if this argument is omitted), :py:func:`numpy.squeeze` will be applied to the output of ``compute_spectra``. This discards any dimensions that have size `1`.


Derivatives
===========

To obtain derivatives of spectra with respect to source parameters, we need to first create an instance of :py:class:`~pyprop8.DerivativeSwitches`. This is used to specify the derivatives that are sought, for example::

   derivs = DerivativeSwitches(moment_tensor=True, force=False, x=True, y=True, z=True)

We can then call ``compute_spectra``, passing this object as an optional ``derivatives`` argument::

   spectra, derivatives = compute_spectra(model, source, receivers, omegas, derivatives=derivs, squeeze_outputs=False)

Notice that when ``derivatives`` is set (or more precisely, when it receives any value other than ``None``), ``compute_spectra`` now returns *two* arrays. The ``spectra`` array is organised precisely as described above; the ``derivatives`` array has an additional dimension. Again, there are two possibilities:

1. If ``receivers`` is an instance of :py:class:`~pyprop8.ListOfReceivers`, ``derivatives`` will have shape ``(source.nsources, receivers.nstations, derivs.nderivs, 3, nfreq)``, where ``source.nderivs`` is the total number of derivatives requested.

2. If ``receivers`` is an instance of :py:class:`~pyprop8.RegularlyDistributedReceivers`, ``derivatives`` will have shape ``(source.nsources, receivers.nr, receivers.nphi, derivs.nderivs, 3, nfreq)``.

If ``squeeze_outputs=True``, :py:func:`numpy.squeeze` will also be applied to this array, discarding any dimensions of size '1'.

The index of specific derivatives (e.g. the 'depth' derivative) within the array will vary depending on the components that are requested. Therefore, :py:class:`~pyprop8.DerivativeSwitches` provides a mechanism for automatically determining the appropriate index. For example::

   # Derivatives wrt the six independent moment tensor components
   dmt = derivatives[:,:,derivs.i_mt:derivs.i_mt+6,:,:]
   # Derivatives wrt the event depth
   ddepth = derivatives[:,:,derivs.i_dep,:,:]

---------------------
Obtaining seismograms
---------------------

``pyprop8`` provides a separate routine to compute seismograms (i.e., time series). Fundamentally, this is simply a matter of obtaining spectra and then taking the Fourier transform; however, :py:func:`~pyprop8.compute_seismograms` handles various additional book-keeping and processing tasks. At the simplest::

   nt = 200 # Number of time-series points
   dt = 0.5   # Sampling interval (s)
   tt, seismograms = compute_seismograms(model, source, receivers, nt, dt, squeeze_outputs=False):

This computes an ``nt``-point displacement time series for each receiver, sampled every ``dt`` seconds. Two arrays are returned. The first (``tt``) has dimension ``(nt, )`` and contains the time-points at which the seismogram is obtained (i.e. the sequence ``[0, dt, 2*dt, ..., (nt-1)*dt]``). The second is again dependent on the manner in which receivers are specified:

1. If ``receivers`` is an instance of :py:class:`~pyprop8.ListOfReceivers`, ``seismograms`` will have shape ``(source.nsources, receivers.nstations, 3, nt)``, where ``source.nsources`` is the number of moment tensor/source vector pairs specified within the ``source`` object, ``receivers.nstations`` is the total number of receivers, and ``nfreq`` is the number of frequency points at which evaluation was requested. The third dimension indexes the three components of motion: x, y, and z.

2. If ``receivers`` is an instance of :py:class:`~pyprop8.RegularlyDistributedReceivers`, ``seismograms`` will have shape ``(source.nsources, receivers.nr, receivers.nphi, 3, nt)``, where ``receivers.nr`` and ``receivers.nphi`` are the number of grid points in the radial and azimuthal directions, respectively.

Again, using the default option of ``squeeze_outputs=True`` will apply :py:func:`numpy.squeeze` to the the output array.

By default, seismograms represent ground displacment relative to a Cartesian coordinate system. By passing the optional argument ``xyz=False`` it is possible to instead obtain seismograms relative to a polar basis, i.e. radial/transverse/vertical motion.

Source time-functions
=====================

By default, ``pyprop8`` is based on the assumption that the source behaves like a step-function in time: all energy is released instantaneously. This is common in seismological simulations, but is not a realistic representation of most seismic sources. It is therefore important to convolve the seismograms with a source time-function that provides a more reasonable representation of energy release. It is efficient to implement this convolution as a multiplication in the frequency domain, and :py:func:`~pyprop8.compute_seismograms` has the facility to apply such a transformation to the seismic spectra prior to taking the Fourier transform.

To use this, it is necessary to determine the frequency spectrum of the desired source time-function. This should be implemented as a Python function (or other callable) that takes an angular frequency as its sole argument, and returns the (complex) value of the source spectrum at that point, e.g.::

   def stf(om):
       if om==0:
           f = 1
       else:
           f = np.sin(om)/om
      return f

This callable should then be passed via the ``source_time_function`` keyword argument, e.g. ``source_time_function=stf``. For convenience and illustration, some standard functions are provided in the :py:mod:`~pyprop8.utils` module.

Derivatives
===========

Computing derivatives of seismograms with respect to moment tensor components follows the pattern already described in the context of spectra. As before, it is necessary to create an instance of :py:class:`~pyprop8.DerivativeSwitches`, and pass it to :py:func:`~pyprop8.compute_seismograms` via the keyword argument ``derivatives``::

   derivs = DerivativeSwitches(moment_tensor=True, force=False, x=True, y=True, depth=True)
   tt, seismograms, derivatives = compute_seismograms(model, source, receivers, nt, dt, derivatives=derivs, squeeze_outputs=False)

Again, this will cause :py:func:`~pyprop8.compute_seismograms` to return a third array, ``derivatives``.

1. If ``receivers`` is an instance of :py:class:`~pyprop8.ListOfReceivers`, ``derivatives`` will have shape ``(source.nsources, receivers.nstations, derivs.nderivs, 3, nt)``, where ``source.nderivs`` is the total number of derivatives requested.

2. If ``receivers`` is an instance of :py:class:`~pyprop8.RegularlyDistributedReceivers`, ``derivatives`` will have shape ``(source.nsources, receivers.nr, receivers.nphi, derivs.nderivs, 3, nt)``.

If ``squeeze_outputs=True``, :py:func:`numpy.squeeze` will be applied to this array. Again, it is recommended to index the array using the functionality provided in :py:class:`~pyprop8.DerivativeSwitches`.

------------------------------------
Obtaining static offset measurements
------------------------------------
Finally, it is also possible to simulate only the static offset (i.e. the part of seismic displacement that remains once transient motion has ceased). This follows the same pattern as the functions already described; usage can be as simple as::

   static = compute_static(model, source, receivers)

This will return an an array of shape ``(source.nsources, receivers.nstations, 3)`` or ``(source.nsources, receivers.nr, receivers.nphi, 3)``, depending on the manner in which ``receivers`` is specified. This contains displacements expressed relative to a Cartesian x/y/z basis, indexed by the final dimension. Alternatively, it is possible to specify one or more 'line of sight' vectors, which determine the direction(s) in which displacement is to be measured. Thus, for example::

   los = np.array([[1,0],
                   [1,0],
                   [0,1]])
   static = compute_static(model, source, receivers,los_vector=los)

will return an array of shape ``(source.nsources, receivers.nstations, 2)`` or or ``(source.nsources, receivers.nr, receivers.nphi, 2)``, containing displacements relative to the two specified directions. Again, passing ``squeeze_outputs=True`` will eliminate unnecessary dimensions from the array.

As before, derivatives may be obtained by passing an additional ``derivatives=derivs`` argument, where ``derivs`` is an instance of :py:class:`~pyprop8.DerivativeSwitches`. Thus::

   static, derivatives = compute_static(model, source, receivers, los_vector=los, derivatives=derivs)

will result in ``derivatives`` having shape ``(source.nsources, receivers.nstations, derivs.nderivs, nlos)`` or ``(source.nsources, receivers.nr, receivers.nphi, derivs.nderivs, nlos)``, where ``nlos`` is the number of line-of-sight vectors provided (or '3' in the default case).
