# Package 'geone'
This python3 project provides tools for goestatistics simulations:
   - gaussian random fields (GRF)
   - multiple-point statistics (MPS) - deesse wrapper

Note: most of the modules in this package can be run as a script ('\_\_main\_\_' scope) and provide an example by this way.

## Installation
To install the package (in your home directory): `pip3 install .`  
To uninstall the package: `pip3 uninstall -y geone`

To install the package in development mode (enable editing): `pip3 install -e .`

## Requirements
The following python packages are used by 'geone':
   - numpy
   - matplotlib
   - mpl_toolkits
   - pyvista

**Important note**:  
The package 'geone' includes a *deesse wrapper*, i.e. an *interface* to directly launch deesse with python. Whereas this wrapper is provided as an open source software (under MIT license), the deesse software itself is patent-protected by the University of Neuchâtel. Free licenses for deesse are provided on request, for non-commercial academic research and education only.

Note also that the deesse wrapper is built for python3.6.

## Examples
Various examples are provided in notebooks.
- Gaussian random fields (GRF):
   - `ex_grf_1d.ipynb`: example for the generation of 1D fields
   - `ex_grf_2d.ipynb`: example for the generation of 2D fields
   - `ex_grf_3d.ipynb`: example for the generation of 3D fields
- Multiple-point statistics - simulation using the deesse wrapper:
   - `ex_deesse_01_basics.ipynb`: basic deesse (categorical) simulations
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
