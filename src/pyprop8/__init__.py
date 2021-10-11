__doc__='''
pyprop8
-------
This package enables computation of the seismic response of a layered elastic
half-space due to a buried point source. It uses an approach described in:

O'Toole, T.B. & J.H. Woodhouse (2011), "Numerically stable computation of
      complete synthetic seismograms including the static displacement in plane
      layered media", Geophysical Journal International, 187, pp.1516--1536,
      doi:10.1111/j.1365-246X.2011.05210.x

In addition, the package has the facility to compute partial derivatives of the
seismic wavefield with respect to seismic source parameters. This relies on
results described in:

O'Toole, T.B., A.P. Valentine & J.H. Woodhouse (2012), "Centroid–moment tensor
      inversions using high-rate GPS waveforms", Geophysical Journal
      International, 191, pp.257--270, doi:10.1111/j.1365-246X.2012.05608.x

For the avoidance of doubt: the examples presented in the above papers were
generated using a Fortran implementation of the algorithm (`prop8`), and not
with the current package.
'''


from ._core import LayeredStructureModel, \
                    PointSource, \
                    RegularlyDistributedReceivers,ListOfReceivers,\
                    DerivativeSwitches, \
                    compute_seismograms,compute_spectra,compute_static, \
                    kIntegrationStencil
