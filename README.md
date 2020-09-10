# prop8py

This package implements the seismogram calculation algorithm set out in O'Toole & Woodhouse (2011), together with the source derivatives set out in O'Toole, Valentine & Woodhouse (2012).

Specifically, the algorithm enables:
- Computation of the seismic response of a plane-layered elastic half-space
- Due to a buried point source (moment-tensor or force representation)
- Stable at zero-frequency (allowing straightforward calculation of the static  deformation attributable to the event)
It is based on a Thompson-Haskell propagator matrix method, using a minor vector formalism to ensure numerial stability for the P-SV system.

This package represents a complete re-implementation of the original Fortran code developed by Woodhouse and O'Toole, which is not readily available. The present version relies on array operations to compute seismograms for many receivers simultaneously. This enables rapid calculations, at the expense of memory consumption. If memory is limited, it is straightforward to divide receivers into subsets, and perform calculations for each subset separately.

Since the algorithm is formulated in a Cartesian (i.e. non-spherical) geometry, it is only intended for simulation of ground motion in the near field. As a rule of thumb, the flat-Earth approximation is applicable for distances up to around 200 km from the seismic source. It will be noted that the theory underpinning this code presently omits many physical phenomena relevant to the real Earth, including:

- Fault finiteness;
- Lateral heterogeneity;
- Topography (of the free surface and of layer interfaces);
- Anelasticity and anisotropy.

Nevertheless, it allows reasonably realistic simulation of the seismic wavefield, and can be employed for first-order analysis of observed waveforms. It also serves as a useful exemplar for teaching, research and development purposes across the broad areas of seismology, geophysical inverse theory and geophysical machine learning, exhibiting phenomena characteristic of a range of real-world problems with modest computational cost.

The present implementation is written solely in Python, and uses only standard libraries (primarily NumPy). As such it is straightforward to deploy and use across a range of platforms, and can easily be incorporated into other Python-based applications.
