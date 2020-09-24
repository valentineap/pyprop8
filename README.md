# pyprop8

This package provides a pure-Python implementation of the seismogram calculation algorithm set out in [O'Toole & Woodhouse (2011)](https://doi.org/10.1111/j.1365-246X.2011.05210.x), together with the source derivatives set out in [O'Toole, Valentine & Woodhouse (2012)](https://doi.org/10.1111/j.1365-246X.2012.05608.x).

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
A seismic source is represented as an instance of `PointSource`, and both moment tensor and force vector representations are supported. The moment tensor is expressed relative to a Cartesian coordinate system (x,y,z). If using catalogue source parameters, these are likely to be expressed in a spherical coordinate system (r,theta,phi); a conversion function is provided as `pyprop8.utils.rtf2xyz()`. In addition, a routine to generate moment tensors from (strike,dip,rake,moment) is available, `pyprop8.utils.make_moment_tensor()`.

Once the moment tensor and force vector have been obtained, creating the `PointSource` object is straightforward:
```python
Mxyz = np.array([3,3])
Mxyz[:,:] = ...
F = np.array([3,1])
F[:,0] = ...
source = pp.PointSource(event_latitude, event_longitude, event_depth,
                        Mxyz, F, event_time)
```
Note that the moment tensor and force vector here are deemed to act *simultaneously*. In most circumstances one or other of these should be set to `np.zeros(...)`.

If one wishes to evaluate seismograms for multiple distinct moment tensors/force vectors *at a single location*, the arrays `Mxyz` and `F` can be specified with shapes `(nsources,3,3)` and `(nsources,3,1)` respectively. In this case, `Mxyz[i,:,:]` acts simultaneously with `F[i,:,:]`. If multiple sources at distinct locations are required, it will be necessary to create separate `PointSource` objects for each.

### Specifying receivers
Two different paradigms can be used for specifying receiver locations. If one wishes to obtain a comprehensive sampling of the seismic wavefield (e.g. for visualisation purposes), `RegularlyDistributedReceivers` provides a set of receivers distributed on a regular grid in polar coordinates (i.e. where, given a list of radii and a list of azimuths, a receiver exists for every possible (radius, azimuth) pair). `pyprop8` is then able to exploit this regularity to accelerate computations. Alternatively, if one wishes to sample the wavefield at specific arbitrary locations (e.g. those corresponding to real observations), `ListOfReceivers` should be used.

For advanced use, both `RegularlyDistributedReceivers` and `ListOfReceivers` can be initialised without arguments to yield an 'empty' object, which can then be populated as required. In general, however, it is anticipated that they will be constructed as follows:
- `stations = RegularlyDistributedReceivers(rmin,rmax,nr,phimin,phimax,nphi,depth=depth,degrees=phi_is_degrees)`, where
  - `rmin,rmax`: minimum and maximum radii at which to place receivers;
  - `nr`: number of equally-spaced radii to generate in specified range;
  - `phimin,phimax`: minimum and maximum azimuths;
  - `nphi`: number of equally-spaced azimuths to generate in specified range;
  - `depth`: depth at which receivers lie within model (km). Default: 0.
  - `phi_is_degrees`: boolean, True if `phimin` and `phimax` are specified in degrees, False if in radians. Default: degrees.

  For convenience, the resulting object provdes a function `stations.as_xy()` which returns the Cartesian coordinates of each station.
- `stations = ListOfReceivers(xlocations,ylocations,x0=x0,y0=y0,depth=depth)`, where
  - `xlocations`: list or 1D array containing x-coordinates of each receiver;
  - `ylocations`: list or 1D array containing y-coordinates of each receiver (in the same order, so that the Nth element of each list refers to the same receiver);
  - `x0,y0`: coordinates of the axis of (assumed) cylindrical symmetry (which must correspond to the source coordinates). Default: (0,0);
  - `depth`: depth at which receivers lie within model (km). Default: 0.

It will be noted that in either case, receivers must all lie at a single depth. In the event that multiple receiver depths must be handled, it will be necessary to create multiple receiver objects and call the computational routines on each individually.

### Computing seismic observables
Once the model, source and receivers are all set up, it is straightforward to compute seismic observables. Two routines are provided. One, `compute_spectra()`, computes and returns velocity spectra for each receiver. The other, `compute_seismograms()`, acts as a wrapper around `compute_spectra()` to post-process and Fourier-transform the spectra yielding time series at each station. Both routines can optionally also return derivatives of spectra/seismograms with respect to the source parameters.

#### Spectra

To obtain spectra,

#### Time series

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
