# Petrinja earthquake animation

This script computes seimic observables for the Mw6.4 earthquake that occurred in Petrinja, Croatia on 29th December 2020. It then generates a movie visualising:
- The surface displacement, including both static and dynamic parts;
- The amplitude of ground acceleration (highlighting the dynamic component of the response);
- The line-of-sight static displacement, as might be measured using InSAR (and demonstrating how this changes when the surface is viewed from different directions).
Source parameters are drawn from the [Global CMT catalogue](https://www.globalcmt.org). Earth model is based on [Crust1.0](https://igppweb.ucsd.edu/~gabi/crust1.html).

The visualisation is done using [`mayavi`](https://pypi.org/project/mayavi/). In theory this should be pip-installable, but this does not seem as robust as it could be. I found `macports` to be the most straightforward route, by installing packages `vtk +python39` and `py39-mayavi`. Obviously this is only a viable route on a Mac!

The script generates each frame of the movie as a separate `.png` file within the `animation/` subdirectory. These can then be assembled into a movie using an appropriate tool, e.g. [`ffmpeg`](https://ffmpeg.org/). For example, the following command will make a `.mp4` file that loops through all frames 3 times:

> ffmpeg  -stream_loop 3 -i animation/frame_%04d.png -vcodec libx264 petrinja.mp4

The `ffmpeg` program can generate movie files in a wide range of formats.

