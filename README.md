# GEONE

[![Documentation Status](https://readthedocs.org/projects/geone/badge/?version=latest)](https://geone.readthedocs.io/en/latest/?badge=latest)

GEONE is a Python3 package providing a set of tools for geostatistical modeling, comprising:
- a DEESSE wrapper for multiple-point statistics (MPS) simulation
- geostatistical tools based on two-point statistics
- miscellaneous algorithms based on random processes

See brief description and examples below.

## Important notes

DEESSE and some other geostatistical tools (GEOSCLASSIC) provided by GEONE are compiled in C for windows, linux and mac, and for python 3.7 to 3.12 (from python 3.8 for mac). Hence, DEESSE and GEOSCLASSIC wrappers use pre-compiled C libraries. The installation process detects the operating system and the python version, ensuring the installation of the right libraries. Note that for linux, libraries depending on the library GLIBC 2.35 or GLIBC 2.27 are provided, hence the library GLIBC of your OS has to be compatible with one of those versions to ensure proper operation of GEONE.

## Installation

### Installation from [PyPI](https://pypi.org/) (The Python Package Index)

In a terminal type 
```
pip install geone
```
(or `python -m pip install geone`).

### Installation from github
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

**Using GEONE - Important note**

If the installation has been done from github, do not launch python from the directory containing the downloaded sources and where the installation has been done (with `pip`), otherwise `import geone` will fail.

### Requirements
The following python packages are used by GEONE (tested on python 3.11.5):
   - matplotlib (3.8.1)
   - multiprocessing (for parallel processes)
   - numpy (tested with version 1.26.0)
   - pandas (tested with version 2.1.2)
   - pyvista (tested with version 0.42.3)
   - scipy (tested with version 1.11.3)

### Removing GEONE
In a terminal type 

`pip uninstall -y geone`

*Note: first remove the directory 'geone.egg-info' from the current directory (if present).*

## Brief GEONE's description

### Multiple-point statistics (MPS)

#### DEESSE
DEESSE is a parallel software for multiple point statistics (MPS) simulations. MPS allows to generate random fields reproducing the spatial statistics - within and between variable(s) - of a training data set (TDS), also often called training image (TI). DEESSE follows an approach based on the *direct sampling* of the TDS. The simulation grid is sequentially populated by values borrowed from the TDS, selected after a random search for similar patterns. Many features are handled by DEESSE and illustrated in the proposed examples below.

#### DEESSEX (crossing DEESSE simulation)
DEESSE can also be used for 3D simulation based on 2D TDS which give the spatial statistics in sections (slices) of the 3D simulation domain. (More generally, 1D or 2D TDS can be used for higher dimensional simulation.) The principle consists in filling the simulation domain by successively simulating sections (according the the given orientation) conditionally to the sections previously simulated. A wrapper for this approach (named DEESSEX referring to crossing-simulation / X-simulation with DEESSE) is proposed in the package GEONE.

### Two-point statistics tools
- Gaussian random fields (GRF)
    - simulation and estimation (kriging) based on Fast Fourier Transform (FFT) (via a (block) circulant embedding of the covariance matrix)
    - simulation (sequential Gaussian simulation, SGS) and estimation (kriging) based on search neighborhood
- Pluri-Gaussian simulation (PGS)
- Other classical geostatistical tools
    - two-point statistics analysis: covariance, variogram, connectivity, etc.

### Miscellaneous algorithms
- Other algorithms based on random processes
    - accept-reject sampler in one or multi-dimension
    - Poisson point process in one or multi-dimension
    - Chentsov simulation in 1D, 2D, 3D

## Examples - Tutorials

*Note: some modules in the package GEONE can be run as a script ('\_\_main\_\_' scope) and provide examples by this way.*

Various examples are provided in notebooks (in [examples](examples) directory) to get started with GEONE, as described below.

### Images and point sets
The following notebooks show how to deal with the classes `geone.img.Img` for "images" and `geone.img.PointSet` for "point sets".

- `ex_a_01_image_and_pointset.ipynb`: classes for images and point sets, reading from files, writing to files, plotting
- `ex_a_02_image2d_rgb.ipynb`: reading / writing RGB 2D images, files in png format
- `ex_a_03_image_interpolation.ipynb`: some tools to interpolate images (e.g. make them finer or coarser)

### Multiple-point statistics - DEESSE
The following notebooks show how to run DEESSE with its functionalities (options).

- `ex_deesse_01_basics.ipynb`: basic DEESSE (categorical) simulations
- `ex_deesse_02_additional_outputs_and_simulation_paths.ipynb`: retrieving additional output maps and setting the simulation path
- `ex_deesse_03_search_neigbhorhood.ipynb`: advanced setting for the search neighborhood ellipsoid
- `ex_deesse_04_continuous_sim.ipynb`: continous simulations
- `ex_deesse_05_geom_transformation.ipynb`: simulations with geometrical transformations (rotation / scaling)
- `ex_deesse_06_proba_constraint.ipynb`: simulations with probability (proportion) constraints
- `ex_deesse_07_connectivity_data.ipynb`: simulations with connectivity data
- `ex_deesse_08_multivariate_sim.ipynb`: bivariate simulations - stationary case
- `ex_deesse_09_multivariate_sim2.ipynb`: bivariate simulations - setting an auxiliary variable to deal with non-stationarity
- `ex_deesse_10_incomplete_image.ipynb`: reconstruction of an image using a training data set
- `ex_deesse_11_using_mask.ipynb`: simulation using a mask
- `ex_deesse_12_multiple_TIs.ipynb`: simulation using multiple training images
- `ex_deesse_13_inequality_data.ipynb`: simulations with inequality data
- `ex_deesse_14_rotation3D.ipynb`: simulations with rotation in 3D
- `ex_deesse_15_block_data.ipynb`: simulation with block data, *i.e* target mean values over block of cells
- `ex_deesse_16_advanced_use_of_pyramids.ipynb`: simulation using pyramids (retrieving pyramids, conditioning within pyramids)

### Multiple-point statistics - DEESSEX ("X-simulation")
The following notebooks show how some examples of crossing-simulation (X-simulation) with DEESSEX.

- `ex_deesseX_01_getting_started.ipynb`: getting starting with deesseX, simulation based on XZ and YZ sections
- `ex_deesseX_02.ipynb`: simulation based on XY, XZ and YZ sections
- `ex_deesseX_03.ipynb`: simulation based on XY, XZ and YZ sections and simulation based on XY 2D-section and Z 1D-section
- `ex_deesseX_04.ipynb`: simulation based on XZ and YZ sections, and accounting for non-stationarity (vertical trend)

### MultiGaussian estimation and simulation (general function)
The following notebook shows elementary covariance models and the use of a general function (wrapper) allowing to launch the other functions of GEONE for multiGaussian estimation and simulation (based on FFT / search neighborhood (GEOSCLASSIC), see below).

- `ex_general_multiGaussian.ipynb`: functions for multiGaussian estimation and simulation in a grid, and elementary covariance/variogram models (in 1D)

### GRF based on FFT
Gaussian random fields (GRF) - simulation and estimation (kriging) in a grid - based on Fast Fourier Transform (FFT).

- `ex_grf_1d.ipynb`: example for the generation of 1D fields
- `ex_grf_2d.ipynb`: example for the generation of 2D fields
- `ex_grf_3d.ipynb`: example for the generation of 3D fields

### SGS / SIS and kriging based on search neighborhood
Sequential Gaussian Simulation (SGS), Sequential Indicator Simulation (SIS) and estimation (kriging) in a grid - based on (limited) search neigborhood; tools for image analysis : covariance variogram, connectivity of images (GEOSCLASSIC wrapper).

- `ex_geosclassic_1d_1.ipynb`:example in 1D for two-point statistics simulation and estimation
- `ex_geosclassic_1d_2_non_stat_cov.ipynb`:example in 1D with non-stationary covariance model
- `ex_geosclassic_2d_1.ipynb`:example in 2D for two-point statistics simulation and estimation
- `ex_geosclassic_2d_2_non_stat_cov.ipynb`:example in 2D with non-stationary covariance model
- `ex_geosclassic_3d_1.ipynb`:example in 3D for two-point statistics simulation and estimation
- `ex_geosclassic_3d_2_non_stat_cov.ipynb`:example in 3D with non-stationary covariance model
- `ex_geosclassic_indicator_1d.ipynb`:example in 1D for two-point statistics simulation and estimation of indicator variables
- `ex_geosclassic_indicator_2d.ipynb`:example in 2D for two-point statistics simulation and estimation of indicator variables
- `ex_geosclassic_indicator_3d.ipynb`:example in 3D for two-point statistics simulation and estimation of indicator variables
- `ex_geosclassic_image_analysis.ipynb`:example for two-point statistics analysis (covariance, variogram, connectivity, ...) of images (maps)

### Variogram analysis tools
Tools for variogram analysis - variogram fitting - illustrated in various cases.

- `ex_vario_analysis_data1D_1.ipynb`: example for variogram analysis and ordinary kriging for data in 1D
- `ex_vario_analysis_data1D_2_non_stationary.ipynb`: example how dealing with non stationary data set in 1D
- `ex_vario_analysis_data2D_1_omnidirectional.ipynb`: example for variogram analysis and ordinary kriging for data in 2D (omni-directional)
- `ex_vario_analysis_data2D_2_general.ipynb`: example for variogram analysis and ordinary kriging for data in 2D (general)
- `ex_vario_analysis_data2D_3_non_stationary.ipynb`: example how dealing with non stationary data set in 2D
- `ex_vario_analysis_data3D_1_omnidirectional.ipynb`: example for variogram analysis and ordinary kriging for data in 3D (omni-directional)
- `ex_vario_analysis_data3D_2_general.ipynb`: example for variogram analysis and ordinary kriging for data in 3D (general)
- `ex_vario_analysis_data3D_3_non_stationary.ipynb`: example how dealing with non stationary data set in 3D

### Pluri-Gaussian simulation (PGS)
- `ex_pgs.ipynb`: example of pluri-Gaussian simulations in 1D, 2D and 3D (categorical, conditional or not), based on two latent Gaussian fields

### Miscellaneous algorithms based on random processes
Accept-reject sampler and other algorithms such as homogeneous and non-homogeneous Poisson point process, Chentsov simulations.

- `ex_acceptRejectSampler.ipynb`: example of accept-reject sampler for generating samples according to given density function (uni- or multi-variate)
- `ex_randProcess.ipynb`: example of Poisson point process, and Chentsov simulation in 1D, 2D and 3D

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

## License

See [LICENSE](LICENSE) file.

## Authors
GEONE is developed by [Julien Straubhaar](https://www.unine.ch/philippe.renard/home/the-team/julien-straubhaar.html) and [Philippe Renard](https://www.unine.ch/philippe.renard/home/the-team/philippe-renard.html).
