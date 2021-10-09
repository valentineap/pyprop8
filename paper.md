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
 - name: Department of Earth Sciences, Durham University, South Road, Durham, DH1 3LE, UK.
   index: 1
 - name: Research School of Earth Sciences, The Australian National University, 142 Mills Road, Acton ACT 2601, Australia.
   index: 2
date: 9 October 2021
bibliography: paper.bib
---
# Introduction
Many different tools and packages exist which may be used to simulate seismic wave propagation. Generally, these are designed to support the needs of the seismological research community. As such, they tend to be partly or entirely dependent on code written in compiled languages, and may interface with a variety of external libraries---for example, those designed to implement standard data formats. As such, installation, deployment and use can be cumbersome, especially for non-expert users and on non-Unix-like operating systems. This can create substantial barriers in contexts such as teaching and outreach, where it may be necessary to accommodate users across a wide variety of platforms.

To address this, we present `pyprop8`. This is an implementation of the sythetic seismogram calculation algorithms presented in [@OToole2011] and [@OToole2012]. It is written entirely in Python, depending only on the mainstream packages `numpy` and `scipy` (plus `matplotlib` for visualisation). It enables calculation of the seismic wavefield, including any static offset, within a 1-D layered half-space. The source may be specified as either a moment tensor or a point force, and it is also possible to calculate derivatives of the wavefield with respect to the source parameters. In addition to computing time series, the package also provides efficient routines for obtaining only the static offset (i.e. the permanent displacement), enabling simulation of seismic observations made using InSAR, GPS or field measurements of slip.

We envisage that this package may serve a diverse range of users. For those studying or teaching seismology it provides a simulation tool that is computationally-lightweight but provides a reasonable level of physical realism, which may be used to enable a wide range of demonstrations or practical exercises. For those studying or teaching inverse theory, it can serve as a exemplar forward model that has realistic features and scalable complexity, and which is compatible with both gradient- and sampling-based inversion algorithms. 
