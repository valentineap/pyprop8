---
title: 'pyprop8: A lightweight code to simulate seismic observables in a layered half-space'
tags:
  - Python
  - seismology
  - earthquakes
authors:
  - name: Andrew P. Valentine
    orcid: 0000-0001-6134-9351
    affiliation: 1 # (Multiple affiliations must be quoted)
    email: andrew.valentine@durham.ac.uk
  - name: Malcolm Sambridge
    orcid: 0000-0003-2858-3175
    affiliation: 2
affiliations:
 - name: Department of Earth Sciences, Durham University.
   index: 1
 - name: Research School of Earth Sciences, The Australian National University.
   index: 2
date: 9 October 2021
bibliography: paper.bib
---
# Summary
The package `pyprop8` enables calculation of the response of a 1-D layered halfspace to a seismic source, and also derivatives ('sensitivity kernels') of the wavefield with respect to source parameters. Seismograms, seismic spectra, and measures of static displacement (e.g. GPS, InSAR and field observations) may all be simulated. The method is based on a Thompson-Haskell propagator matrix algorithm, described in @OToole2011 and @OToole2012. The package is entirely written in Python, dependent only on the mainstream libraries `numpy` and `scipy`. As such it is lightweight and easy to deploy across a variety of platforms, making it particularly suited to use for teaching and outreach purposes.

# Statement of need
Many different tools and packages exist which may be used to simulate seismic wave propagation. Generally, these are designed to support the needs of the seismological research community. As such, they tend to be optimised for computational efficiency, and are partly or entirely dependent on code written in compiled languages. In many cases they interface with a variety of external libraries---for example, those designed to implement particular data formats. As such, installation, deployment and use can be cumbersome, especially for non-expert users and on non-Unix-like operating systems. This can create substantial barriers in contexts such as teaching and outreach, where it may be necessary to accommodate and support users across a wide variety of platforms.

To address this, `pyprop8` is written entirely in Python, and the core seismogram-calculation routines do not have any dependencies except the mainstream libraries `numpy` and `scipy`. We envisage that this package will serve a diverse range of users. For those studying or teaching seismology, `pyprop8` provides a simulation tool with a reasonable level of physical realism and flexibility, which can underpin a wide range of demonstrations and practical exercises. Similarly, for those studying or teaching inverse theory, `pyprop8` may be used as an exemplar forward problem with realistic features and scalable complexity. In this context, the ability to compute partial derivatives with respect to source parameters is valuable. The package may also prove beneficial to the research community, particularly as a source of test problems for use in the development of inversion algorithms.

# Implementation details
The theoretical basis for `pyprop8` is described in detail in @OToole2011, with the calculation of derivatives set out in an appendix to @OToole2012; `pyprop8` provides a full implementation of this published theory. The algorithm is based on a Thompson-Haskell propagator matrix method, using a minor-vector formalism to ensure numerical stability. A novel feature of the approach is that it is designed to remain stable at zero frequency, allowing the static offset (i.e., permanent seismic deformation) to be captured within simulations. As a result, a wide range of seismic observables can be simulated within a common framework, including conventional seismograms, GPS records, InSAR images and direct field observations of slip.

## Limitations
A number of assumptions or approximations are inherent to the formulation of `pyprop8`, and we summarise the main ones here:

- The algorithm is framed in a Cartesian geometry, i.e. it assumes a flat Earth. This is a reasonable approximation close to the seismic source, but degrades beyond modest distances (~100km).
- The earth structure is assumed to be a stack of homogeneous, isotropic layers. Real-world features such as lateral heterogeneity, anisotropy and topography cannot be accounted for.
- The seismic source is assumed to act at a single point in space: any finite spatial extent of real-world sources is neglected. This approximation degrades close to the source location and as the seismic magnitude increases.

These factors should be borne in mind in any case where outputs from `pyprop8` are to be compared to observational data.

# Applications
The computational approach described by @OToole2011 (but not the `pyprop8` implementation) has underpinned a number of studies---particularly those focussed on earthquake early warning using continuous GPS data [@OToole2012; @OToole2013; @Kaufl2014]. It has also been exploited for monitoring of microseismicity, e.g. during hydraulic fracturing [@OToole2013a]. We anticipate that the release of `pyprop8` will enable further applications of this kind.

Much of our work focusses on the development of new strategies for solving inverse problems. In this context, `pyprop8` provides a valuable resource for implementing test problems. In particular, it offers scalable complexity across both model and data spaces. It can support examples that are linear, weakly non-linear, or highly non-linear, and datasets may range from isolated slip-vectors through to dense arrays of time-series. This allows a structured approach to algorithm development, progressing from simpler to more complex test problems [@Kaufl2014; @Kaufl2015; @Kaufl2016a].

# Acknowledgements
The theory described in @OToole2011 was originally implemented in Fortran 77 by John Woodhouse and Tom O'Toole. We have benefited from access to a copy of this code (`prop8`), which was distributed informally but never publicly released. This has informed certain aspects of our own implementation, and enabled us to validate our version against theirs. Our work has benefited from financial support from the Australian Research Council under grants DE180100040 and DP200100053, and from the CSIRO Future Science Platform in Deep Earth Imaging.

# References
