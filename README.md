# Package GEONE
GEONE is a python package providing a set of tools for geostatistical and multiple-point statistics modeling, comprising:
   - multiple-point statistics (MPS) - DEESSE wrapper
   - Gaussian random fields (GRF) - following a method based on (block) circulant embedding of the covariance matrix and Fast Fourier Transform (FFT)
   - other classical geostatistical tools (two-point statistics analysis (covariance, variogram, connectivity) / simulation (SGS, SIS) / estimation (kriging))
   - pluri-Gaussian simulation (PGS)
   - other algorithms based on random processes (Poisson point process, Chentsov simulation)


## Installation
```
git clone https://github.com/randlab/geone.git
cd geone
pip install .
```

 **Remove geone**

`pip uninstall -y geone`

*Note: first remove the directory 'geone.egg-info' from the current directory (if present).*

## Using GEONE

Do not launch python from the directory where the installation has been done (with `pip`), otherwise `import geone` will fail.

## Requirements
The following python packages are used by GEONE (tested on python 3.10.9):
   - matplotlib (3.6.2)
   - multiprocessing (for parallel processes)
   - numpy (tested with version 1.23.5)
     pandas (tested with version 1.5.2)
   - pyvista (tested with version 0.38.5)
   - scipy (tested with version 1.10.0)

## Important notes
- GEONE includes a *DEESSE wrapper* to directly launch DEESSE within python. The DEESSE version provided with GEONE is a test version with restricted capabilities. **To unlock the full capabilities of DEESSE, the user must obtain a commercial or academic license from the University of Neuchâtel. See LICENSE file for details.**
- DEESSE and some other geostatistical tools provided by GEONE are compiled in C for windows and linux, and for python3.6 to python3.10. **Note that for linux, the provided libraries depend on the library GLIBC 2.35, hence the library GLIBC of your OS has to be compatible with that version to ensure proper operation of GEONE.**

## DEESSE - Introduction
DEESSE is a parallel software for multiple point statistics (MPS) simulations. MPS allows to generate random fields reproducing the spatial statistics -- within and between variable(s) -- of a training data set (TDS). DEESSE follows an approach based on the *direct sampling* of the TDS. The simulation grid is sequentially populated by values borrowed from the TDS, selected after a random search for similar patterns. Many features are handled by DEESSE and illustrated in the proposed examples below.

DEESSE can also be used for 3D simulation based on 2D TDS which give the spatial statistics in sections (slices) of the 3D simulation domain. (More generally, 1D or 2D TDS can be used for higher dimensional simulation.) The principle consists in filling the simulation domain by successively simulating sections (according the the given orientation) conditionally to the sections previously simulated. A wrapper for this approach (named DEESSEX referring to crossing-simulation / X-simulation with DEESSE) is proposed in the package GEONE.

## Examples
Some modules in the package GEONE can be run as a script ('\_\_main\_\_' scope) and provide examples by this way.

Various examples are provided (notebooks in 'examples' directory) to get started with GEONE, as described below.

#### Images and point sets in geone
- `ex_a_01_image_and_pointset.ipynb`: classes for images and point sets, reading from files, writing to files, plotting
- `ex_a_02_image2d_rgb.ipynb`: reading / writing RGB 2D images, files in png format

#### DEESSE Examples
Multiple-point statistics - simulation using the DEESSE wrapper:
- `ex_deesse_01_basics.ipynb`: basic DEESSE (categorical) simulations
- `ex_deesse_02_additional_outputs_and_simulation_paths.ipynb`: retrieving additional output maps and setting the simulation path
- `ex_deesse_03_search_neigbhorhood.ipynb`: advanced setting for the search neighborhood ellipsoid
- `ex_deesse_04_continous_sim.ipynb`: continous simulations
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

#### DEESSEX Examples
Multiple-point statistics - X-simulation using the DEESSEX wrapper:
- `ex_deesseX_01_getting_started.ipynb`: getting starting with deesseX, simulation based on XZ and YZ sections
- `ex_deesseX_02.ipynb`: simulation based on XY, XZ and YZ sections
- `ex_deesseX_03.ipynb`: simulation based on XY, XZ and YZ sections and simulation based on XY 2D-section and Z 1D-section
- `ex_deesseX_04.ipynb`: simulation based on XZ and YZ sections, and accounting for non-stationarity (vertical trend)

#### General use for multiGaussian estimation and simulation in a grid
- `ex_general_multiGaussian.ipynb`: functions for multiGaussian estimation and simulation in a grid, and elementary covariance/variogram models (in 1D)

#### Examples - Gaussian random fields in a grid (GRF)
- `ex_grf_1d.ipynb`: example for the generation of 1D fields
- `ex_grf_2d.ipynb`: example for the generation of 2D fields
- `ex_grf_3d.ipynb`: example for the generation of 3D fields

#### Examples - Simulation and estimation with kriging in a grid (GeosClassic wrapper)
- `ex_geosclassic_1d.ipynb`:example in 1D for two-point statistics simulation and estimation
- `ex_geosclassic_1d_non_stat_cov.ipynb`:example in 1D with non-stationary covariance model
- `ex_geosclassic_2d.ipynb`:example in 2D for two-point statistics simulation and estimation
- `ex_geosclassic_2d_non_stat_cov.ipynb`:example in 2D with non-stationary covariance model
- `ex_geosclassic_3d.ipynb`:example in 3D for two-point statistics simulation and estimation
- `ex_geosclassic_3d_non_stat_cov.ipynb`:example in 3D with non-stationary covariance model
- `ex_geosclassic_indicator_1d.ipynb`:example in 1D for two-point statistics simulation and estimation of indicator variables
- `ex_geosclassic_indicator_2d.ipynb`:example in 2D for two-point statistics simulation and estimation of indicator variables
- `ex_geosclassic_indicator_3d.ipynb`:example in 3D for two-point statistics simulation and estimation of indicator variables
- `ex_geosclassic_image_analysis.ipynb`:example for two-point statistics analysis (covariance, variogram, connectivity, ...) of images (maps)

#### Examples - Variogram analysis tools and kriging:
- `ex_vario_analysis_data1D.ipynb`: example for variogram analysis and ordinary kriging for data in 1D
- `ex_vario_analysis_data2D_1_omnidirectional.ipynb`: example for variogram analysis and ordinary kriging for data in 2D (omni-directional)
- `ex_vario_analysis_data2D_2_general.ipynb`: example for variogram analysis and ordinary kriging for data in 2D (general)
- `ex_vario_analysis_data3D_1_omnidirectional.ipynb`: example for variogram analysis and ordinary kriging for data in 3D (omni-directional)
- `ex_vario_analysis_data3D_2_general.ipynb`: example for variogram analysis and ordinary kriging for data in 3D (general)

#### Examples - Pluri-Gaussian simulation
- `ex_pgs.ipynb`: example of pluri-Gaussian simulations in 1D, 2D and 3D (categorical, conditional or not), based on two latent Gaussian fields

#### Examples - Ohter algorithms based on random processes
- `ex_randProcess.ipynb`: example of Poisson point process, and Chentsov simulation in 1D, 2D and 3D

## Some references about DEESSE
- J. Straubhaar, P. Renard (2021) Conditioning Multiple-Point Statistics Simulation to Inequality Data. Earth and Space Science, [doi:10.1029/2020EA001515](https://dx.doi.org/10.1029/2020EA001515)
- J. Straubhaar, P. Renard, T. Chugunova (2020) Multiple-point statistics using multi-resolution images. Stochastic Environmental Research and Risk Assessment 20, 251-273, [doi:10.1007/s00477-020-01770-8](https://dx.doi.org/10.1007/s00477-020-01770-8)
- J. Straubhaar, P. Renard, G. Mariethoz (2016) Conditioning multiple-point statistics simulations to block data. Spatial Statistics 16, 53-71, [doi:10.1016/j.spasta.2016.02.005](https://dx.doi.org/10.1016/j.spasta.2016.02.005)
- G. Mariethoz, J. Straubhaar, P. Renard, T. Chugunova, P. Biver (2015) Constraining distance-based multipoint simulations to proportions and trends. Environmental Modelling & Software 72, 184-197, DOI:
   10.1016/j.envsoft.2015.07.007
- G. Mariethoz, P. Renard, J. Straubhaar (2010) The Direct Sampling method to perform multiple-point geostatistical simulation. Water Resources Research 46, W11536, [doi:10.1029/2008WR007621](https://dx.doi.org/10.1029/2008WR007621)

## Reference about DEESSEX
- A. Comunian, P. Renard, J. Straubhaar (2012) 3D multiple-point statistics simulation using 2D training images. Computers & Geosciences 40, 49-65, [doi:10.1016/j.cageo.2011.07.009](https://dx.doi.org/10.1016/j.cageo.2011.07.009)

## Some references about GRF
- C. R. Dietrich and G. N. Newsam. A fast and exact method for multidimensional gaussian stochastic simulations. Water Resour. Res., 29(8):2861-2869, 1993, [doi:10.1029/93WR01070](https://dx.doi.org/10.1029/93WR01070)
- A. T. A. Wood and G. Chan. Simulation of stationary gaussian processes in [0,1]^d. J. Comput. Graph. Stat., 3(4):409-432, 1994, url: http://www.jstor.org/stable/1390903.
- J. W. Cooley and J. W. Tukey. An algorithm for machine calculation of complex fourier series. Math. Comput., 19(90):297-301, 1965, [doi:10.2307/2003354](https://dx.doi.org/10.2307/2003354)
