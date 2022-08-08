# Contributing to `pyprop8`

Contributions to the `pyprop8` project are welcome! 

## Using and citing `pyprop8`
If you find `pyprop8` useful in your work, please tell us! A list of projects and publications that rely on `pyprop8` can be found [here](USERS.md), and we would love to add more. The best way to do this is to fork the repository, edit your copy, and submit a pull request. If you're not confident doing this, you can use the [issue tracker](https://github.com/valentineap/pyprop8/issues) to send us the relevant information, and we will add it for you.

If you are preparing a paper, we would appreciate it if you cite both the paper describing the general method, and this specific implementation:

O'Toole, T.B. & J.H. Woodhouse (2011), "Numerically stable computation of complete synthetic seismograms including the static displacement in plane layered media", Geophysical Journal International, 187, pp.1516-1536, doi:10.1111/j.1365-246X.2011.05210.x

(JOSS, tbc)

If your work relies on being able to calculate the source parameter derivatives, you should also cite the paper describing how these are obtained:

O'Toole, Valentine & Woodhouse (2012), "Centroid–moment tensor inversions using high-rate GPS waveforms", Geophysical Journal International, 191, pp.257-270, doi:10.1111/j.1365-246X.2012.05608.x



## Reporting bugs and suggesting improvements
Bug reports should be submitted via the [issue tracker](https://github.com/valentineap/pyprop8/issues). This can also be used to propose ideas for future development. 

## Fixing bugs and making improvements
You are welcome to implement bug fixes and other enhancements. The best way to do this is to:

1. Create a fork of [the main repository](https://github.com/valentineap/pyprop8) in your own Github account.
2. Make and test your changes.
3. Once you are satisfied, create a pull request to the main repository. Please make sure your pull request explains what your edits are designed to fix/add!
4. Respond to any questions/comments raised during code review.

If your pull request is accepted, your contribution will appear in the main code base. 

Anyone considering making substantial additions or changes to the code is encouraged to [contact Andrew Valentine](mailto:andrew.valentine@durham.ac.uk) to discuss your plans and ensure that development efforts are coordinated. 

## Using a local copy of `pyprop8`

The easiest way for most users to install `pyprop8` will be via `pip`. This installs the latest stable release somewhere on your system, so that it can be found when you do `import pyprop8`.

During development, you may want `import pyprop8` to load your working copy, and not the stable release. Since `pyprop8` is a pure-Python module, this is reasonably straightforward: you just need to ensure that `<repository>/src` (where `<repository>` is your working copy of the repository) appears in your module search path before the system directories. One easy way forward is to use a '[virtual environment](https://docs.python.org/3/library/venv.html)':

```bash
% python -m venv my_env # Creates `my_env` directory
% source my_env/bin/activate 
(my_env) % pip install -r requirements.txt # Installs numpy, scipy etc
(my_env) % export PYTHON_PATH=<repository>/src
```
This should ensure that Python uses the `<repository>` version of `pyprop8`. You can verify this by checking which file is loaded when you import the module:

```python
> import pyprop8 as pp
> print(pp.__file__) # Should be <repository>/src/pyprop8/__init__.py
``` 




