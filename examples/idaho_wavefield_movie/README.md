# Idaho earthquake animation

This script computes seimic observables for the Mw6.5 earthquake that occurred in Stanley, Idaho on 31st March 2020. It then generates a movie visualising:
- The surface displacement, including both static and dynamic parts;
- The amplitude of ground acceleration (highlighting the dynamic component of the response);
- The line-of-sight static displacement, as might be measured using InSAR (and demonstrating how this changes when the surface is viewed from different directions).

The script generates each frame of the movie as a separate `.png` file. These can then be assembled into a movie using an appropriate tool, e.g. ffmpeg.

Source parameters are drawn from the [Global CMT catalogue](https://www.globalcmt.org). Earth model is based on [Crust1.0](https://igppweb.ucsd.edu/~gabi/crust1.html).

The visualisation is done using [`mayavi`](https://pypi.org/project/mayavi/). In theory this should be pip-installable, but this does not seem as robust as it could be. I found `macports` to be the most straightforward route, by installing packages `vtk +python39` and `py39-mayavi`. Obviously this is only a viable route on a Mac!
