# Idaho earthquake animation

This script generates each frame of the movie as a separate `.png` file. These can then be assembled into a movie using an appropriate tool, e.g. ffmpeg.

Source parameters are drawn from the [Global CMT catalogue](https://www.globalcmt.org). Earth model is based on [Crust1.0](https://igppweb.ucsd.edu/~gabi/crust1.html).

The visualisation is done using [`mayavi`](https://pypi.org/project/mayavi/). In theory this should be pip-installable, but this does not seem as robust as it could be. I found `macports` to be the most straightforward route, by installing packages `vtk +python39` and `py39-mayavi`. Obviously this is only a viable route on a Mac!

![Example video](./idaho.mp4 video/mp4)

<video width="320" height="240" controls>
  <source src="./idaho.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>
