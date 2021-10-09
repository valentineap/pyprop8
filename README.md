# pyprop8

This package provides a pure-Python implementation of the seismogram calculation algorithm set out in [O'Toole & Woodhouse (2011)](https://doi.org/10.1111/j.1365-246X.2011.05210.x), together with the source derivatives set out in [O'Toole, Valentine & Woodhouse (2012)](https://doi.org/10.1111/j.1365-246X.2012.05608.x). It is intended to provide a lightweight, easy-to-install seismological forward model suitable for use in teaching and research (in particular, to provide a computationally-cheap yet physically-realistic forward problem for use in the development and testing of inversion algorithms).

Specifically, this algorithm enables:

- Computation of the seismic response of a plane-layered elastic half-space
- Due to a buried point source (moment-tensor or force representation)
- Stable at zero-frequency (allowing straightforward calculation of the static  deformation attributable to the event)

It is based on a Thompson-Haskell propagator matrix method, using a minor vector formalism to ensure numerical stability for the P-SV system.

## Contents
- [Introduction](#introduction)
- [Getting started](#getting-started)
  - [Specifying earth models](#specifying-earth-models)
  - [Specifying sources](#specifying-sources)
  - [Specifying receivers](#specifying-receivers)
  - [Computing seismic observables](#computing-seismic-observables)
    - [Spectra](#spectra)
    - [Time series](#time-series)
    - [Static displacement](#static-displacement)
  - [A minimal example](#a-minimal-example)
- [Citing this package](#citing-this-package)
- [Acknowledgements](#acknowledgements)

## Introduction
The current package represents a complete re-implementation of the original Fortran code developed by Woodhouse and O'Toole. It is designed to use array operations to compute seismograms for many receivers simultaneously. This enables rapid calculations, at the expense of memory consumption. If memory is limited, it is straightforward to divide receivers into subsets, and perform calculations for each subset separately.

Since the algorithm is formulated in a Cartesian (i.e. non-spherical) geometry, it is only intended for simulation of ground motion in the near field. As a rule of thumb, the flat-Earth approximation is applicable for distances up to around 200 km from the seismic source. It will be noted that the theory underpinning this code presently omits many physical phenomena relevant to the real Earth, including:

- Fault finiteness;
- Lateral heterogeneity;
- Topography (of the free surface and of layer interfaces);
- Anelasticity and anisotropy.

Nevertheless, it allows reasonably realistic simulation of the seismic wavefield, and can be employed for first-order analysis of observed waveforms. It also serves as a useful exemplar for teaching, research and development purposes across the broad areas of seismology, geophysical inverse theory and geophysical machine learning, exhibiting phenomena characteristic of a range of real-world problems with modest computational cost.

The present implementation is written solely in Python, and uses only standard libraries (primarily NumPy). As such it is straightforward to deploy and use across a range of platforms, and can easily be incorporated into other Python-based applications.

## Getting started
To access core functionality, import `pyprop8` within your Python code. This
provides access to the following:

- A class for specifying the earth model, `LayeredStructureModel`;
- A class for specifying the source, `PointSource`;
- Two classes for specifying receiver locations, `RegularlyDistributedReceivers` and `ListOfReceivers`;
- A class for specifying which derivatives are sought, `DerivativeSwitches`;
- Two routines, `compute_spectra()` and `compute_seismograms()`, which perform calculations and return seismic observables in the spectral and time domains, respectively.

In order to generate seismograms, you need to:

1. Specify an earth model;
2. Specify the source representation;
3. Specify the receivers at which seismograms are to be 'recorded';

and then call one of the computational routines.

### Specifying earth models
The algorithm underpinning `pyprop8` assumes that an earth model consists of horizontal flat layers, with each layer being homogenous and isotropic. The lowermost layer is assumed to extend to infinite depth. A Cartesian (rather than spherical) geometry is assumed.

From a computational perspective, `pyprop8` represents an earth model using the class `LayeredStructureModel`. Our first task is therefore to create an instance of this class. To do so, we must pass the initialiser a list (or other list-like type, e.g. a NumPy array) specifying the layers. This can be done in two ways:

1. Each entry in the list corresponds to a *layer*, and is a tuple of `(thickness,vp,vs,rho)` where `thickness` is the layer thickness, `vp` and `vs` are P- and S-wave velocities (in units of km/s), and `rho` is a density (in g/cc). The first entry in the list describes the uppermost (i.e., surface) layer; subsequent entries correspond to successively deeper layers. The final entry *must* have thickness `np.inf`, and represents the properties of the underlying halfspace. If using this approach, pass `interface_depth_form=False` to the `LayeredStructureModel` initialiser. (This is also the default if `interface_depth_form` is not explicitly specified.)

2. Each entry in the list corresponds to an *interface*. In this case, the entry is a tuple `(interface_depth,vp,vs,rho)`, with `vp`,`vs` and `rho` describing the seismic properties *below* the interface. The interface depths should all be positive, and specified in units of km. The list does *not* need to be ordered, but *must* contain one entry with an interface depth of '0' (corresponding to the surface). If there is duplication of interface depths within the list, only the first will be retained. The properties associated with the lowermost interface specified are deemed to represent the underlying halfspace. If using this form, pass `interface_depth_form=True` to the `LayeredStructureModel` initialiser.

Thus, in the following, the models `m1`,`m2` and `m3` are all identical:
```python
import pyprop8 as pp
import numpy as np

m1 = pp.LayeredStructureModel([(3,1.8,0,1.02),
                               (5,4.5,2.4,2.5),
                               (np.inf,8,4.5,3.0)])
m2 = pp.LayeredStructureModel([(0,1.8,0,1.02),
                               (3,4.5,2.4,2.5),
                               (8,8,4.5,3.0)],interface_depth_form=True)
model_array = np.array([[3,1.8,0,1.02],[5,4.5,2.4,2.5],[np.inf,8,4.5,3.0]])
m3 = pp.LayeredStructureModel(model_array,interface_depth_form = False)
```
A visual representation of the layers and their properties can be obtained by calling `print()` on any instance of `LayeredStructureModel`, and may be useful for verifying that everything is as intended.

### Specifying sources
A seismic source is represented as an instance of `PointSource`, and both moment tensor and force vector representations are supported. The moment tensor is expressed relative to a Cartesian coordinate system (x,y,z). If using catalogue source parameters, these are likely to be expressed in a spherical coordinate system (r,theta,phi); a conversion function is provided as `pyprop8.utils.rtf2xyz()`. In addition, a routine to generate (r, theta,phi) moment tensors from (strike,dip,rake,moment) is available, `pyprop8.utils.make_moment_tensor()`.

Once the moment tensor and force vector have been obtained, creating the `PointSource` object is straightforward:
```python
Mxyz = np.array([3,3])
Mxyz[:,:] = ...
F = np.array([3,1])
F[:,0] = ...
source = pp.PointSource(event_x, event_y, event_depth,
                        Mxyz, F, event_time)
```
Note that the moment tensor and force vector here are deemed to act *simultaneously*. In most circumstances one or other of these should be set to `np.zeros(...)`.

If one wishes to evaluate seismograms for multiple distinct moment tensors/force vectors *at a single location*, the arrays `Mxyz` and `F` can be specified with shapes `(nsources,3,3)` and `(nsources,3,1)` respectively. In this case, `Mxyz[i,:,:]` acts simultaneously with `F[i,:,:]`. If multiple sources at distinct locations are required, it will be necessary to create separate `PointSource` objects for each.

### Specifying receivers
Two different paradigms can be used for specifying receiver locations. If one wishes to obtain a comprehensive sampling of the seismic wavefield (e.g. for visualisation purposes), `RegularlyDistributedReceivers` provides a set of receivers distributed on a regular grid in polar coordinates (i.e. where, given a list of radii and a list of azimuths, a receiver exists for every possible (radius, azimuth) pair). `pyprop8` is then able to exploit this regularity to accelerate computations. Alternatively, if one wishes to sample the wavefield at specific arbitrary locations (e.g. those corresponding to real observations), `ListOfReceivers` should be used.

Initialisation is as follows:
- `stations = RegularlyDistributedReceivers(rmin,rmax,nr,phimin,phimax,nphi,depth=depth,x0=x0,y0,degrees=phi_is_degrees)`, where
  - `rmin,rmax`: minimum and maximum radii at which to place receivers;
  - `nr`: number of equally-spaced radii to generate in specified range;
  - `phimin,phimax`: minimum and maximum azimuths, measured anticlockwise from the x (East) axis;
  - `nphi`: number of equally-spaced azimuths to generate in specified range;
  - `depth`: depth at which receivers lie within model (km). Default: 0.
  - `x0, y0`: Location of polar origin in global Cartesian system. Must coincide with event location.
  - `phi_is_degrees`: boolean, True if `phimin` and `phimax` are specified in degrees, False if in radians. Default: degrees.


  For convenience, the resulting object provdes a function `stations.as_xy()` which returns the Cartesian coordinates of each station.
- `stations = ListOfReceivers(xlocations,ylocations,depth=depth,geometry=geometry)`, where

  - `xlocations`: list or 1D array containing x-coordinates of each receiver;
  - `ylocations`: list or 1D array containing y-coordinates of each receiver (in the same order, so that the Nth element of each list refers to the same receiver);
  - `depth`: depth at which receivers lie within model (km). Default: 0.
  - `geometry`: Whether locations should be interpreted as x/y coordinates in a Cartesian geometry (`geometry='cartesian'`), or as lon/lat (*note order!*) coordinates in a spherical geometry (`geometry='spherical'`). This choice also governs the interpretation of the event location coordinates.

It will be noted that in either case, receivers must all lie at a single depth. In the event that multiple receiver depths must be handled, it will be necessary to create multiple receiver objects and call the computational routines on each individually.

### Computing seismic observables
Once the model, source and receivers are all set up, it is straightforward to compute seismic observables. Three routines are provided. One, `compute_spectra()`, computes and returns velocity spectra for each receiver. The second, `compute_seismograms()`, acts as a wrapper around `compute_spectra()` to post-process and Fourier-transform the spectra yielding time series at each station. The third, `compute_static()`, is another wrapper that determines the static-offset component of the displacement field at each location, and (optionally) computes the line-of-sight displacement to simulat (e.g.) an InSAR image. All routines can optionally also return derivatives of spectra/seismograms with respect to the source parameters.

#### Spectra

To obtain velocity spectra, call
```
spectra [, dspectra] = compute_spectra(structure, source, stations, omegas,
                                       derivatives = None, show_progress = True,
                                       stencil = kIntegrationStencil, stencil_kwargs = {'kmin':0,'kmax':2.04,'nk':1200},
                                       squeeze_outputs=True )
```
where

- `structure` is an instance of `LayeredStructureModel` ([see above](#specifying-earth-models));
- `source` is an instance of `PointSource` ([see above](#specifying-sources));
- `stations` is an instance of either `RegularlyDistributedReceivers` or `ListOfReceivers` ([see above](#specifying-receivers));
- `omegas` is a (possibly complex) array of shape `(nomegas,)` specifying the frequencies for which spectra should be performed (in units of rad/s);
- `derivatives` is an instance of `DerivativeSwitches` specifying any derivatives that are sought;
- `show_progress` is a boolean value; set to `True` to display a progress indicator using `tqdm`;
- `stencil` is a callable that determines the quadrature scheme used for integration over spatial wavenumber, k. It should have signature `kvals,wts = stencil(**stencil_kwargs)`, where `kvals` is a 1D array containing the k values to be evaluated, and `wts` is a 1D array containing the quadrature weight associated with each evaluation point. By default a trapezium-rule integrator is used;
- `stencil_kwargs` is a dictionary containing any arguments required by the stencil function;
- `squeeze_outputs` is a boolean value; set to `True` to call `np.squeeze()` on the spectra before returning, to discard any unnecessary dimensions in the array (i.e, dimensions of length 1).

The function `compute_spectra` returns either one or two arrays, depending on whether derivatives are requested. The first (or only) array contains all requested spectra. Its shape will depend on how receivers were specified. If an object of class `RegularlyDistributedReceivers` was used, then it has shape `(nsources, nr, nphi, 3, nomegas)`, where:

- `nsources` represents the number of moment tensors/force vectors specified in the `source` object;
- `nr, nphi` represent the number of radii and azimuths specified in the `stations` object;
- `3` represents the three spatial dimensions (ordered as 'radial', 'transverse', 'vertical'); and
- `nomegas` represents the number of frequencies requested.

If, instead, `stations` is of class `ListOfReceivers`, the result will have shape `(nsources, nstations, 3, nomegas)`. Here `nstations` is the number of distinct receivers in the `stations` object, and other dimensions are as described above. Note that if `squeeze_outputs=True`, any 'unnecessary' dimensions will be removed. This is convenient when (say) only one source is employed.

If derivatives are requested, these are contained in the second array returned by `compute_spectra`. This has shape `(nsources, nr, nphi, nderivs, 3, nomegas)` or `(nsources, nstations, nderivs, 3, nomegas)`, where `nderivs` is the number of derivative components requested. The ordering of the derivatives within this array is governed by the `DerivativeSwitches` object. Again, `squeeze_outputs=True` will affect the shape of the array. If a `DerivativeSwitches` object is provided, but no derivative components are activated, `compute_spectra` will return `None` in place of `d_spectra`.

#### Time series
To compute seismograms, call
```
tt,seismograms [, dseismograms] = compute_seismograms(structure, source, stations
                     nt, dt, alpha = None,
                     source_time_function=None,pad_frac=0.5,kind='displacement',
                     xyz = True, derivatives = None,
                     show_progress = True, **kwargs)
```
This is a wrapper around `compute_spectra`, and the arguments `structure`, `source` and `stations` are as defined above. Additional arguments are:
- `nt` is the number of time samples in the output time series.
- `dt` is the time-step for the output time series (i.e., total length will be `dt*(nt-1)`.)
- `alpha` is a parameter describing how far off the real axis computations should be undertaken. In effect, we calculate `exp(-alpha t).s(t)`, and then 'undo' the exponential scaling before returning `s(t)`. The purpose of this is to avoid singularities on the real axis that are associated with surface wave propagation. If `alpha=None` (default), we use rule of thumb proposed in O'Toole & Woodhouse (2011): `alpha = log(10)/tmax`.
- `source_time_function` is a callable, `stf(omega)`, taking a single complex-valued argument (a frequency, in units of rad/s), and returning a scalar value (possibly complex). If provided, the seismic spectra are multiplied by this function prior to the inverse Fourier transform being taken. This allows a source time-function (or other filtering operations) to be applied. Some appropriate functions are available in  `pyprop8.utils`.
- `pad_frac` is a float representing any additional time series length to be computed and then discarded (to circumvent ringing and other 'edge' effects). This is expressed as a fraction of `nt`, i.e. a time series of length `(1 + pad_frac)*nt` samples is initially computed, but only the first `nt` points are returned.
- `kind` is a string: permissible values are `displacement`, `velocity` or `acceleration`. This determines the class of observable that is simulated.
- `xyz` is a boolean value; set to True (default) to obtain seismograms relative to a Cartesian [x/y/z or east/north/vertical] basis; otherwise they will be expressed in a cylindrical coordinate system [radial/transverse/vertical].
- `derivatives` should be passed a `DerivativeSwitches` object if any derivatives are sought.
- `show_progress` is a boolean value; set to True for a progress bar to be displayed.
- `squeeze_outputs` is a boolean value; set to `True` to call `np.squeeze()` on the spectra before returning, to discard any unnecessary dimensions in the array (i.e, dimensions of length 1).
Any additional keyword arguments are passed to `compute_spectra`.

`compute_seismograms` returns two or three arrays, depending on whether derivatives are requested. The first, `tt`, is a one-dimensional array of length `nt`, containing the time-steps at which the seismogram has been computed. The second, `seismograms`, has a shape that depends on the class of `stations`. If this is `RegularlyDistributedReceivers`, then the shape is `(nsources, nr, nphi, 3, nt)`; if `ListOfReceivers`, it will be `(nsources, nstations, 3, nt)`. If derivatives are requested, the third array will have shape `(nsources, nr, nphi, nderivs, 3, nt)` or `(nsources, nstations, nderivs, 3, nt)`. Again, if `squeeze_outputs=True`, any dimensions containing only one entry will be discarded.

#### Static displacement
A second wrapper around `compute_spectra` handles the zero-frequency case, outputting static displacement data. This has signature

```
static [, d_static] = compute_static(structure, source, stations,
                                     los_vector = None, derivatives = None,
                                     squeeze_outputs=True,**kwargs)
```
Again, `structure`, `source` and `stations` are as defined above. Additional arguments are:
- `los_vector` describes the 'line of sight' along which the displacement field is to be viewed. If a 1-D array is provided, the static displacement field is projected onto this direction (expressed relative to a Cartesian basis). If a 2-D array is provided, each column of the array is treated as an independent line-of-sight, and the static field corresponding to each is returned. Thus, the default value of `np.eye(3)` returns the static displacement relative to each coordinate axis in the Cartesian basis. Finally, setting `los_vector=None` will cause the displacement field to be returned in the cylindrical [radial/transverse/vertical] basis that is employed for computation.
- `derivatives` is a `DerivativesSwitches` object describing any derivatives that should be computed.
- `squeeze_outputs` is a boolean value; set to `True` to call `np.squeeze()` on the spectra before returning, to discard any unnecessary dimensions in the array (i.e, dimensions of length 1).
All other keyword arguments are passed to `compute_spectra`.

One or two arrays are returned, depending on the value of `derivatives`. The first, containing the static displacements, has shape `(nsources, nr, nphi, nlos)` or `(nsources, nstations, nlos)` depending on the type of the `stations` object used (as discussed above); here, `nlos` is the number of columns in the `los_vector` array, or 3 if `los_vector=None`. The second array, if present, contains the derivatives. This  has shape `(nsources, nr, nphi, nderivs, nlos)` or `(nsources, nstations, nderivs, nlos)`. Again, if a `DerivativeSwitches` object is passed to `compute_static` but all derivative components are switched off, `None` will be returned for `d_static`, and the value of the `squeeze_outputs` argument may modify the array shapes.
### A minimal example

Summarising all the above, a minimal working example might be:

```python
import pyprop8 as pp
from pyprop8.utils import rtf2xyz,make_moment_tensor,stf_trapezoidal

model = pp.LayeredStructureModel([(3,1.8,0,1.02),(5,4.5,2.4,2.5),(np.inf,8,4.5,3)]) # Ocean layer on surface
stations = pp.RegularlyDistributedReceivers(10,100,9,0,360,36,depth=3) # Receivers on sea floor
source = pp.PointSource(0,0,12,rtf2xyz(make_moment_tensor(135,30,20,1E8,0,0)),np.zeros((3,1)),None)
nt = 257
timestep = 0.5
tt, seis = pp.compute_seismograms(model,source,stations,
                              nt,timestep,0.023,
                              source_time_function=lambda om:stf_trapezoidal(om,3,6))
xlocs,ylocs = stations.as_xy()    
```

This will result in `tt` of shape (257,) containing timesteps, and `seis` of shape `(9,36,3,257)` containing displacement seismograms. Specifically, `seis[i,j,:,:]` will represent one 3-component seismogram sampled at 2Hz, for the station at `(xlocs[i,j],ylocs[i,j])`.

## Citing this package
If you make use of this code, please acknowledge the work that went into developing it! In particular, if you are preparing a publication, we would appreciate it if you cite both the paper describing the general method, and this specific implementation:

- [O'Toole, T.B. & J.H. Woodhouse (2011)](https://doi.org/10.1111/j.1365-246X.2011.05210.x), "Numerically stable computation of complete synthetic seismograms including the static displacement in plane layered media", Geophysical Journal International, 187, pp.1516-1536, doi:10.1111/j.1365-246X.2011.05210.x
- (tbc)

An appropriate form of words might be, "We calculate seismograms using the software package `pyprop8` (REF tbc), which is based on the approach of O'Toole & Woodhouse (2011)."

If your work relies on being able to calculate the source parameter derivatives, you should also cite the paper describing how these are obtained:

- [O'Toole, Valentine & Woodhouse (2012)](https://doi.org/10.1111/j.1365-246X.2012.05608.x), "Centroid–moment tensor inversions using high-rate GPS waveforms", Geophysical Journal International, 191, pp.257-270, doi:10.1111/j.1365-246X.2012.05608.x

## Acknowledgements

This package was developed at the Australian National University by Andrew Valentine. It builds on earlier work by Tom O'Toole and John Woodhouse, and has benefited from sight of code written by those authors. In particular, the exponential rescaling of the propagator matrices is inspired by their implementation. However, the present code has been developed 'from scratch' based on the published algorithms.

This work has received support from the Australian Research Council under grants DE180100040 and DP200100053.
