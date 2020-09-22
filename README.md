# pyprop8

This package implements the seismogram calculation algorithm set out in [O'Toole & Woodhouse (2011)](https://doi.org/10.1111/j.1365-246X.2011.05210.x), together with the source derivatives set out in [O'Toole, Valentine & Woodhouse (2012)](https://doi.org/10.1111/j.1365-246X.2012.05608.x).

## Contents
- [Introduction](#introduction)
- [Getting started](#getting-started)
  - [Specifying earth models](#specifying-earth-models)
  - [Specifying sources](#specifying-sources)
  - [Specifying receivers](#specifying-receivers)

## Introduction
Specifically, the algorithm enables:

- Computation of the seismic response of a plane-layered elastic half-space
- Due to a buried point source (moment-tensor or force representation)
- Stable at zero-frequency (allowing straightforward calculation of the static  deformation attributable to the event)

It is based on a Thompson-Haskell propagator matrix method, using a minor vector formalism to ensure numerical stability for the P-SV system.

The current package represents a complete re-implementation of the original Fortran code developed by Woodhouse and O'Toole, which is not readily available. It is designed to use array operations to compute seismograms for many receivers simultaneously. This enables rapid calculations, at the expense of memory consumption. If memory is limited, it is straightforward to divide receivers into subsets, and perform calculations for each subset separately.

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
Note that the moment tensor and force vector here are deemed to act *simultaneously*. In most circumstances you probably want one of these to be set to `np.zeros(...)`.

If one wishes to evaluate seismograms for multiple distinct moment tensors/force vectors *at a single location*, the arrays `Mxyz` and `F` can be specified with shapes `(nsources,3,3)` and `(nsources,3,1)` respectively. In this case, `Mxyz[i,:,:]` acts simultaneously with `F[i,:,:]`. If multiple sources at distinct locations are required, it will be necessary to create separate `PointSource` objects for each.

### Specifying receivers

Two different paradigms can be used for specifying receiver locations. If one wishes to obtain a comprehensive sampling of the seismic wavefield (e.g. for visualisation purposes), `RegularlyDistributedReceivers` provides a set of receivers distributed on a regular grid in polar coordinates (i.e. at equally spaced radii and azimuths). `pyprop8` is then able to exploit this regularity to accelerate computations. Alternatively, if one wishes to sample the wavefield at specific arbitrary locations (e.g. those corresponding to available observations), `ListOfReceivers` should be used.
