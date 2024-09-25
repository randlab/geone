.. _examples_gallery_file:

Examples - Notebooks Gallery
****************************

Switch to :ref:`Notebooks List <examples_list_file>`

Various examples are provided in notebooks below to get started with GEONE. 

The notebooks (with related data files) from this doc are available on Github at `<https://github.com/randlab/geone/docs/source/notebooks>`_

.. note::
    In the gallery below, abbreviations are used for the beginning of the original notebook file names:

    - `ex_gc` stands for `ex_geosclassic`
    - `ex_VA` stands for `ex_vario_analysis`


Images and point sets
---------------------
The following notebooks show how to deal with the classes `geone.img.Img` for "images" and `geone.img.PointSet` for "point sets".

    .. nbgallery::
        :maxdepth: 1

        ex_a_01 - Images and point sets         <notebooks/ex_a_01_image_and_pointset>
        ex_a_02 - Dealing with RGB images (png) <notebooks/ex_a_02_image2d_rgb>
        ex_a_03 - Interpolating images          <notebooks/ex_a_03_image_interpolation>
    

Multiple-point statistics - DEESSE
----------------------------------
The following notebooks show how to run DEESSE with its functionalities (options).

    .. nbgallery::
        :maxdepth: 1

        ex_deesse_01 - Getting started                         <notebooks/ex_deesse_01_basics>
        ex_deesse_02 - Simulation path and additional outputs  <notebooks/ex_deesse_02_additional_outputs_and_simulation_paths>
        ex_deesse_03 - Search neighborhood                     <notebooks/ex_deesse_03_search_neigbhorhood>
        ex_deesse_04 - Continuous simulations                  <notebooks/ex_deesse_04_continuous_sim>
        ex_deesse_05 - Geometrical transformation              <notebooks/ex_deesse_05_geom_transformation>
        ex_deesse_06 - Proportion constraints                  <notebooks/ex_deesse_06_proba_constraint>
        ex_deesse_07 - Connectivity data                       <notebooks/ex_deesse_07_connectivity_data>
        ex_deesse_08 - Multivariate simulations (I)            <notebooks/ex_deesse_08_multivariate_sim>
        ex_deesse_09 - Multivariate simulations (II)           <notebooks/ex_deesse_09_multivariate_sim2>
        ex_deesse_10 - Incomplete images                       <notebooks/ex_deesse_10_incomplete_image>
        ex_deesse_11 - Using a mask                            <notebooks/ex_deesse_11_using_mask>
        ex_deesse_12 - Multiple training data sets             <notebooks/ex_deesse_12_multiple_TIs>
        ex_deesse_13 - Inequality data                         <notebooks/ex_deesse_13_inequality_data>
        ex_deesse_14 - Rotation in 3D                          <notebooks/ex_deesse_14_rotation3D>
        ex_deesse_15 - Block data                              <notebooks/ex_deesse_15_block_data>
        ex_deesse_16 - Advanced use of pyramids                <notebooks/ex_deesse_16_advanced_use_of_pyramids>

Multiple-point statistics - DEESSEX ("X-simulation")
----------------------------------------------------
The following notebooks show how some examples of crossing-simulation (X-simulation) with DEESSEX.

    .. nbgallery::
        :maxdepth: 1

        ex_deesseX_01 - Getting started                        <notebooks/ex_deesseX_01_getting_started>
        ex_deesseX_02 - Example I                              <notebooks/ex_deesseX_02>
        ex_deesseX_03 - Example II                             <notebooks/ex_deesseX_03>
        ex_deesseX_04 - Example III (with non-stationarity)    <notebooks/ex_deesseX_04>

MultiGaussian estimation and simulation (general function)
----------------------------------------------------------
The following notebook shows elementary covariance models and the use of a general function (wrapper) allowing to launch the other functions of GEONE for multiGaussian estimation and simulation (based on FFT / search neighborhood (GEOSCLASSIC), see below).

    .. nbgallery::
        :maxdepth: 1

        ex_general_multiGaussian <notebooks/ex_general_multiGaussian>

GRF based on FFT
----------------
Gaussian random fields (GRF) - simulation and estimation (kriging) in a grid - based on Fast Fourier Transform (FFT).

    .. nbgallery::
        :maxdepth: 1

        ex_grf_1d - 1D <notebooks/ex_grf_1d>
        ex_grf_2d - 2D <notebooks/ex_grf_2d>
        ex_grf_3d - 3D <notebooks/ex_grf_3d>

SGS / SIS and kriging based on search neighborhood
--------------------------------------------------
Sequential Gaussian Simulation (SGS), Sequential Indicator Simulation (SIS) and estimation (kriging) in a grid - based on (limited) search neigborhood; tools for image analysis : covariance variogram, connectivity of images (GEOSCLASSIC wrapper).

    .. nbgallery::
        :maxdepth: 1

        ex_gc_1d_1 - 1D                                 <notebooks/ex_geosclassic_1d_1>
        ex_gc_1d_2 - 1D with non stationary covariance  <notebooks/ex_geosclassic_1d_2_non_stat_cov>
        ex_gc_2d_1 - 2D                                 <notebooks/ex_geosclassic_2d_1>
        ex_gc_2d_2 - 2D with non stationary covariance  <notebooks/ex_geosclassic_2d_2_non_stat_cov>
        ex_gc_3d_1 - 3D                                 <notebooks/ex_geosclassic_3d_1>
        ex_gc_3d_2 - 3D with non stationary covariance  <notebooks/ex_geosclassic_3d_2_non_stat_cov>
        ex_gc_indicator_1d - indicator variable in 1D   <notebooks/ex_geosclassic_indicator_1d>
        ex_gc_indicator_2d - indicator variable in 2D   <notebooks/ex_geosclassic_indicator_2d>
        ex_gc_indicator_3d - indicator variable in 3D   <notebooks/ex_geosclassic_indicator_3d>
        ex_gc_image_analysis - tools for image analysis <notebooks/ex_geosclassic_image_analysis>

Variogram analysis tools
------------------------
Tools for variogram analysis - variogram fitting - illustrated in various cases.

    .. nbgallery::
        :maxdepth: 1

        ex_VA_data1D_1 - 1D                       <notebooks/ex_vario_analysis_data1D_1>
        ex_VA_data1D_2 - 1D with non-stationarity <notebooks/ex_vario_analysis_data1D_2_non_stationary>
        ex_VA_data2D_1 - 2D omni-directional      <notebooks/ex_vario_analysis_data2D_1_omnidirectional>
        ex_VA_data2D_2 - 2D with anisotropy       <notebooks/ex_vario_analysis_data2D_2_general>
        ex_VA_data2D_3 - 2D with non-stationarity <notebooks/ex_vario_analysis_data2D_3_non_stationary>
        ex_VA_data3D_1 - 3D omni-directional      <notebooks/ex_vario_analysis_data3D_1_omnidirectional>
        ex_VA_data3D_2 - 3D with anisotropy       <notebooks/ex_vario_analysis_data3D_2_general>
        ex_VA_data3D_3 - 3D with non-stationarity <notebooks/ex_vario_analysis_data3D_3_non_stationary>

Pluri-Gaussian simulation (PGS)
-------------------------------
    .. nbgallery::
        :maxdepth: 1
        
        ex_pgs - PGS in 1D, 2D, 3D <notebooks/ex_pgs>

Miscellaneous algorithms based on random processes
--------------------------------------------------
Accept-reject sampler and other algorithms such as homogeneous and non-homogeneous Poisson point process, Chentsov simulations.

    .. nbgallery::
        :maxdepth: 1
        
        ex_acceptRejectSampler            <notebooks/ex_acceptRejectSampler>
        ex_randProcess - various examples <notebooks/ex_randProcess>
