# Package GEONE
GEONE is a python package providing a set of tools for geostatistical and multiple-point statistics modeling, comprising:
   - gaussian random fields (GRF)
   - multiple-point statistics (MPS) - DEESSE wrapper

## Installation
To install the package: `python3 -m pip install install .`

To uninstall the package: `python3 -m pip uninstall -y geone`

To install the package in development mode (enable editing): `python3 -m pip install -e .`

## Requirements
The following python packages are used by 'geone':
   - numpy
   - matplotlib
   - mpl_toolkits
   - pyvista

**Important note**:  
GEONE includes a *DEESSE wrapper* to directly launch DEESSE within python. The DEESSE version provided with GEONE is a test version with restricted capabilities. To unlock the full capabilities of DEESSE, the user must obtain a commercial or academic license from the University of Neuchâtel. See LICENSE file for details.

Note also that the DEESSE wrapper is built for python3.6 / python3.7 / python3.8.

## Examples
Most of the modules in the package GEONE can be run as a script ('\_\_main\_\_' scope) and provide examples by this way.

Various examples are provided in notebooks, as described below.
- Gaussian random fields (GRF):
   - `ex_grf_1d.ipynb`: example for the generation of 1D fields
   - `ex_grf_2d.ipynb`: example for the generation of 2D fields
   - `ex_grf_3d.ipynb`: example for the generation of 3D fields
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
