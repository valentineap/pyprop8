# Examples

This directory contains various notebooks and scripts that demonstrate some of the things that `pyprop8` can do.

## Jupyter Notebooks
- [demo.ipynb](examples/demo.ipynb) - A simple demonstration of `pyprop8`, showing computation of seismograms and static displacement fields. Intended to provide a 'first look' at `pyprop8`, which can be run [using Binder](https://mybinder.org/v2/gh/valentineap/pyprop8/HEAD?labpath=examples%2Fdemo.ipynb).

## Python Scripts
- [otoole_woodhouse_2011.py](examples/otoole_woodhouse_2011.py) - Code that aims to reproduce the figures presented in [O'Toole & Woodhouse (2011)](https://doi.org/10.1111/j.1365-246X.2011.05210.x). Replication is not perfect: various details and choices are not fully- and unambiguously documented in the paper. In particular, no formulae are given for source time functions, and this is (probably) the source of the very minor differences that can be seen between the waveforms plotted here and those of the paper.

- [otoole_valentine_woodhouse_2012.py](examples/otoole_valentine_woodhouse_2012.py) - Code that aims to reproduce the figures presented in [O'Toole et al. (2012)](https://doi.org/10.1111/j.1365-246X.2012.05608.x). Again, some minor differences are likely due to imperfect replication of how the source time function is implemented.
