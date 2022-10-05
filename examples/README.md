# Examples

This directory contains various notebooks and scripts that demonstrate some of the things that `pyprop8` can do.

## Jupyter Notebooks
- [demo.ipynb](demo.ipynb) - A simple demonstration of `pyprop8`, showing computation of seismograms and static displacement fields. Intended to provide a 'first look' at `pyprop8`, which can be run [using Binder](https://mybinder.org/v2/gh/valentineap/pyprop8/HEAD?labpath=examples%2Fdemo.ipynb). The `requirements.txt` file for this example can be found at `/binder/requirements.txt` (which is where Binder expects it!).

## Python Scripts
- [otoole_papers/otoole_woodhouse_2011.py](otoole_papers/otoole_woodhouse_2011.py) - Code that aims to reproduce the figures presented in [O'Toole & Woodhouse (2011)](https://doi.org/10.1111/j.1365-246X.2011.05210.x). Replication is not perfect: various details and choices are not fully- and unambiguously documented in the paper. In particular, no formulae are given for source time functions, and this is (probably) the source of the very minor differences that can be seen between the waveforms plotted here and those of the paper.

- [otoole_papers/otoole_valentine_woodhouse_2012.py](otoole_papers/otoole_valentine_woodhouse_2012.py) - Code that aims to reproduce the figures presented in [O'Toole et al. (2012)](https://doi.org/10.1111/j.1365-246X.2012.05608.x). Again, some minor differences are likely due to imperfect replication of how the source time function is implemented.

- [idaho_wavefield_movie/idaho_wavefield_movie.py](idaho_wavefield_movie/idaho_wavefield_movie.py) - Generate an animation visualizing the wavefield evolution and line-of-sight static displacement for a Mw6.4 earthquake in Idaho on 31 March 2020 (the [Stanley earthquake](https://www.idahogeology.org/geologic-hazards/earthquake-hazards/stanley-earthquake)).

- [petrinja_wavefield_movie/petrinja_wavefield_movie.py](petrinja_wavefield_movie/petrinja_wavefield_movie.py) - A similar animation, but for the Mw6.4 earthquake that occurred in Petrinja, Croatia on 29th December 2020.

## See Also

- The [`tests`](/src/pyprop8/tests.py) module.

- The [list of projects](/USERS.md) that have made use of `pyprop8`.
