===============
Getting started
===============

------------
Installation
------------

The easiest way to obtain ``pyprop8`` is using ``pip``, e.g.::

  pip install pyprop8

Alternatively you can clone the `github repository <https://github.com/valentineap/pyprop8>`_, and then make sure that the directory ``src/pyprop8`` is included in your Python search path.

``pyprop8`` is designed to have minimal dependencies, so that it is easy to deploy. The main non-standard requirements are ``numpy`` and ``scipy``.

-----
Usage
-----
``pyprop8`` is packaged in a number of submodules. Core computational routines are available by importing ``pyprop8``. In the examples given here, this will always be done as follows::

  import pyprop8 as pp

Some additional convenience functions are found in the module ``pyprop8.utils``. In this document we will always import these by name, e.g.::

  from pyprop8.utils import rtf2xyz

The package includes further modules that are used internally, but which are unlikely to be useful to end-users.

For more details, read :ref:`walkthrough` and take a look at the :ref:`example`.

-------------------------
Testing your installation
-------------------------
To verify that your installation is working correctly, you may wish to run the tests that are distributed as part of the package. To access these, enter the following in a Python interpreter::

  from pyprop8 import tests
  tests.tests()

This will run and report a number of calculations.