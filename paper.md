---
title: 'pyprop8: A lightweight code to simulate seismic observables in a layered half-space'
tags:
  - Python
  - seismology
  - earthquakes
authors:
  - name: Andrew P Valentine # note this makes a footnote saying 'co-first author'
    orcid: 0000-0001-6134-9351
    affiliation: 1 # (Multiple affiliations must be quoted)
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
The package `pyprop8` enables calculation of the response of a 1-D layered halfspace to a seismic source, and also derivatives ('sensitivity kernels') of the wavefield with respect to source parameters. It is based on a Thompson-Haskell propagator matrix algorithm, described in [@OToole2011] and [@OToole2012]. The package is entirely written in Python, dependent only on the mainstream libraries `numpy` and `scipy` (plus `matplotlib` for visualisation of the included examples). As such it is lightweight and easy to deploy across a variety of platforms, making it particularly suited to use for teaching and outreach purposes.

# Statement of need
Many different tools and packages exist which may be used to simulate seismic wave propagation. Generally, these are designed to support the needs of the seismological research community. As such, they tend to be partly or entirely dependent on code written in compiled languages, and may interface with a variety of external libraries---for example, those designed to implement standard data formats. As such, installation, deployment and use can be cumbersome, especially for non-expert users and on non-Unix-like operating systems. This can create substantial barriers in contexts such as teaching and outreach, where it may be necessary to accommodate and support users across a wide variety of platforms.

To address this, `pyprop8` is written entirely in Python, and the core seismogram-calculation routines do not have any dependencies except the mainstream libraries `numpy` and `scipy`. We envisage that this package will serve a diverse range of users. For those studying or teaching seismology, it offers a simulation tool that is computationally-lightweight but with a reasonable level of physical realism, and may be used to enable a wide range of demonstrations or practical exercises. For those studying or teaching inverse theory, it can serve as a exemplar forward model with realistic features and scalable complexity. Since it provides the ability to compute derivatives of the wavefield with respect to source parameters, it can be used in conjunction with both gradient- and sampling-based inversion algorithms. Similarly, we believe that it will prove useful to the research community---particularly those engaged in the development of inversion algorithms, where straightforward yet realistic test problems are often required.

# Implementation details
The theoretical basis for `pyprop8` is set out in [@OToole2011], with the calculation of derivatives described in [@OToole2012]. The algorithm is based on a Thompson-Haskell propagator matrix method, using a minor-vector formalism to ensure numerical stability. A novel feature of this approach is that it is designed to remain stable at zero frequency, allowing the static offset (i.e., permanent seismic deformation) to be captured within simulations. As a result, a wide range of seismic observables can be simulated within a common framework, including conventional seismograms, GPS records, InSAR images and direct field observations of slip. The algorithm is framed in a Cartesian geometry, i.e. it assumes a flat Earth. As a result, its ability to reproduce real-world seismograms degrades beyond modest distances (~100km).

# Acknowledgements
This work has benefited from financial support from the Australian Research Council under grants DE180100040 and DP200100053, and from the CSIRO Future Science Platform in Deep Earth Imaging.