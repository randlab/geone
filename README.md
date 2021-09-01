# Package GEONE
GEONE is a python package providing a set of tools for geostatistical and multiple-point statistics modeling, comprising:
   - multiple-point statistics (MPS) - DEESSE wrapper
   - gaussian random fields (GRF)
   - other classical geostatistical tools (two-point statistics analysis (covariance, variogram, connectivity) / simulation (SGS, SIS) / estimation (kriging))

## Installation
To install the package: `python3 -m pip install .`

To uninstall the package: `python3 -m pip uninstall -y geone`

(To install the package in development mode (enable editing): `python3 -m pip install -e .`)

## Requirements
The following python packages are used by 'geone':
   - numpy
   - scipy
   - matplotlib
   - mpl_toolkits
   - pyvista

**Important note**:  
GEONE includes a *DEESSE wrapper* to directly launch DEESSE within python. The DEESSE version provided with GEONE is a test version with restricted capabilities. To unlock the full capabilities of DEESSE, the user must obtain a commercial or academic license from the University of Neuchâtel. See LICENSE file for details.

Note also that the DEESSE wrapper is built for python3.6 / python3.7 / python3.8 / python3.9.

## Examples
Some modules in the package GEONE can be run as a script ('\_\_main\_\_' scope) and provide examples by this way.

Various examples are provided (notebooks in 'examples' directory) to get started with GEONE, as described below.
- Multiple-point statistics - simulation using the DEESSE wrapper:
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
- Gaussian random fields (GRF):
   - `ex_grf_1d.ipynb`: example for the generation of 1D fields
   - `ex_grf_2d.ipynb`: example for the generation of 2D fields
   - `ex_grf_3d.ipynb`: example for the generation of 3D fields
- Variogram analysis tools and kriging:
   - `ex_vario_analysis_data1D.ipynb`: example for variogram analysis and ordinary kriging for data in 1D
   - `ex_vario_analysis_data2D_1_omnidirectional.ipynb`: example for variogram analysis and ordinary kriging for data in 2D (omni-directional)
   - `ex_vario_analysis_data2D_2_general.ipynb`: example for variogram analysis and ordinary kriging for data in 2D (general)
   - `ex_vario_analysis_data3D_1_omnidirectional.ipynb`: example for variogram analysis and ordinary kriging for data in 3D (omni-directional)
   - `ex_vario_analysis_data3D_2_general.ipynb`: example for variogram analysis and ordinary kriging for data in 3D (general)
- Simulation and estimation with kriging (GeosClassic wrapper):
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
