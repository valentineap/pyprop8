.. _reference:

*************
API reference
*************
.. py:module:: pyprop8

Classes
=======

Earth model
-----------

.. autoclass:: LayeredStructureModel
   :members: vp, vs, copy

Receivers
---------
.. autoclass:: ListOfReceivers
   :members: copy, nstations
.. autoclass:: RegularlyDistributedReceivers
   :members: copy, nstations, asListOfReceivers


Seismic source
--------------
.. autoclass:: PointSource
   :members: copy

Derivatives
-----------
.. autoclass:: DerivativeSwitches

Main computational routines
===========================
The following routines perform simulations of seismic response.

.. autofunction:: compute_spectra

.. autofunction:: compute_seismograms

.. autofunction:: compute_static
