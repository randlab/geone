# GEONE

[![Documentation Status](https://readthedocs.org/projects/geone/badge/?version=latest)](https://geone.readthedocs.io/en/latest/?badge=latest)

**Current version : 1.3.0** <!-- Update manually here! see src/geone/_version.py -->

GEONE is a Python3 package providing a set of tools for geostatistical modeling, including:

- multiple-point statistics (MPS) simulation as "DEESSE wrapper"
- geostatistical tools based on two-point statistics, including  "GEOSCLASSIC wrapper"
- miscellaneous algorithms based on random processes

## Documentation, examples and references

The documentation of GEONE is on https://geone.readthedocs.io.

<!-- The notebooks (examples) from the documentation are available in [docs/source/notebooks](./docs/source/notebooks). -->
The notebooks (examples) from the documentation are available in [docs/source/notebooks](https://github.com/randlab/geone/tree/master/docs/source/notebooks).

## Installation

GEONE is available on:

- [PyPI](https://pypi.org/project/geone) (The Python Package Index)
- [Github repository](https://github.com/randlab/geone)

GEONE relies on pre-compiled C libraries (DEESSE and GEOSCLASSIC core), available for:
- linux (x86_64 with GLIBC 2.35 or GLIBC 2.27) and python 3.9 to 3.13
- mac (x86_64 or arm64) and python 3.9 to 3.13
- windows and python 3.9 to 3.13

### Installation from [PyPI](https://pypi.org/project/geone)

In a terminal type
```
pip install geone
```
Or, equivalently: `python -m pip install geone`.

### Installation from the [Github repository](https://github.com/randlab/geone)

In a terminal, change directory where to download GEONE, and type
```
git clone https://github.com/randlab/geone.git
cd geone
pip install .
```

*Note:* use `pip install . --verbose` or `pip install . -v` for printing (more) messages during the installation.

Alternatively:

- Instead of `git clone ...`, you can download GEONE from the [Github repository](https://github.com/randlab/geone): click on the green button "code" and choose "Download ZIP".
- Then, unzip the archive on your computer
- Finally, in a terminal, go into the unzipped directory, and type `pip install .`

**Warning - Using GEONE**

If the installation has been done from github, do not launch python from the directory containing the downloaded sources and where the installation has been done (with `pip`), otherwise `import geone` will fail.

### Requirements

The following python packages are used by GEONE:

- `matplotlib`
- `multiprocessing` (for parallel processes)
- `numpy`
- `pandas`
- `pyvista`
- `scipy`

**Note: `numpy` version >= 2 is used.**

**Remark: the package `ipykernel` is required to run the notebooks.**

GEONE has been tested with the following settings:
```
platform_system : Linux
python version  : sys.version_info(major=3, minor=13, micro=7, releaselevel='final', serial=0)
glibc_str       : glibc 2.35
glibc_version   : (2, 35)
machine         : x86_64
matplotlib.__version__ = 3.10.7
numpy.__version__      = 2.3.3
pandas.__version__     = 2.3.3
pyvista.__version__    = 0.46.3
scipy.__version__      = 1.16.2
```

### Removing GEONE
In a terminal type

`pip uninstall -y geone`

*Note: First remove the directory 'geone.egg-info' from the current directory (if present).*

<!--
## References

### Some references about DEESSE
- J. Straubhaar, P. Renard (2021) Conditioning Multiple-Point Statistics Simulation to Inequality Data. Earth and Space Science, [doi:10.1029/2020EA001515](https://dx.doi.org/10.1029/2020EA001515)
- J. Straubhaar, P. Renard, T. Chugunova (2020) Multiple-point statistics using multi-resolution images. Stochastic Environmental Research and Risk Assessment 20, 251-273, [doi:10.1007/s00477-020-01770-8](https://dx.doi.org/10.1007/s00477-020-01770-8)
- J. Straubhaar, P. Renard, G. Mariethoz (2016) Conditioning multiple-point statistics simulations to block data. Spatial Statistics 16, 53-71, [doi:10.1016/j.spasta.2016.02.005](https://dx.doi.org/10.1016/j.spasta.2016.02.005)
- G. Mariethoz, J. Straubhaar, P. Renard, T. Chugunova, P. Biver (2015) Constraining distance-based multipoint simulations to proportions and trends. Environmental Modelling & Software 72, 184-197, [doi:10.1016/j.envsoft.2015.07.007](https://dx.doi.org/10.1016/j.envsoft.2015.07.007)
- G. Mariethoz, P. Renard, J. Straubhaar (2010) The Direct Sampling method to perform multiple-point geostatistical simulation. Water Resources Research 46, W11536, [doi:10.1029/2008WR007621](https://dx.doi.org/10.1029/2008WR007621)

### Reference about DEESSEX
- A. Comunian, P. Renard, J. Straubhaar (2012) 3D multiple-point statistics simulation using 2D training images. Computers & Geosciences 40, 49-65, [doi:10.1016/j.cageo.2011.07.009](https://dx.doi.org/10.1016/j.cageo.2011.07.009)

### Some references about GRF
- J. W. Cooley and J. W. Tukey (1965) An algorithm for machine calculation of complex fourier series. Mathematics of Computation 19(90):297-301, [doi:10.2307/2003354](https://dx.doi.org/10.2307/2003354)
- C. R. Dietrich and G. N. Newsam (1993) A fast and exact method for multidimensional gaussian stochastic simulations. Water Resources Research 29(8):2861-2869, [doi:10.1029/93WR01070](https://dx.doi.org/10.1029/93WR01070)
- A. T. A. Wood and G. Chan (1994) Simulation of stationary gaussian processes in [0,1]^d. Journal of Computational and Graphical Statistics 3(4):409-432, [doi:10.2307/1390903](https://dx.doi.org/10.2307/1390903)

### Other references
- C. Lantuéjoul (2002) Geostatistical Simulation, Models and Algorithms. Springer Verlag, Berlin, 256 p.
- P. Renard, D. Allard (2013), Connectivity metrics for subsurface flow and transport. Advances in Water Resources 51:168-196, `doi:10.1016/j.advwatres.2011.12.001 <https://doi.org/10.1016/j.advwatres.2011.12.001>`_
- J. Straubhaar, P. Renard (2024), Exploring substitution random functions composed of stationary multi-Gaussian processes. Stochastic Environmental Research and Risk Assessment, `doi:10.1007/s00477-024-02662-x <https://doi.org/10.1007/s00477-024-02662-x>`_
 -->

## License

<!-- See [LICENSE](LICENSE) file. -->
<!-- See [LICENSE](https://geone.readthedocs.io/en/latest/LICENSE.html) file. -->
See [LICENSE](https://github.com/randlab/geone/blob/master/LICENSE) file.

## Authors
GEONE is developed by [Julien Straubhaar](https://www.unine.ch/philippe.renard/home/the-team/julien-straubhaar.html) and [Philippe Renard](https://www.unine.ch/philippe.renard/home/the-team/philippe-renard.html).
