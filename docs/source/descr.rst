Brief GEONE's description
*************************

Multiple-point statistics (MPS)
===============================

DEESSE
------
DEESSE is a parallel software for multiple point statistics (MPS) simulations. MPS allows to generate random fields reproducing the spatial statistics - within and between variable(s) - of a training data set (TDS), also often called training image (TI). DEESSE follows an approach based on the *direct sampling* of the TDS. The simulation grid is sequentially populated by values borrowed from the TDS, selected after a random search for similar patterns. Many features are handled by DEESSE and illustrated in the proposed examples below.

DEESSEX (crossing DEESSE simulation)
------------------------------------
DEESSE can also be used for 3D simulation based on 2D TDS which give the spatial statistics in sections (slices) of the 3D simulation domain. (More generally, 1D or 2D TDS can be used for higher dimensional simulation.) The principle consists in filling the simulation domain by successively simulating sections (according the the given orientation) conditionally to the sections previously simulated. A wrapper for this approach (named DEESSEX referring to crossing-simulation / X-simulation with DEESSE) is proposed in the package GEONE.

Two-point statistics tools
==========================
- Gaussian random fields (GRF)
    - simulation and estimation (kriging) based on Fast Fourier Transform (FFT) (via a (block) circulant embedding of the covariance matrix)
    - simulation (sequential Gaussian simulation, SGS) and estimation (kriging) based on search neighborhood
- Pluri-Gaussian simulation (PGS)
- Other classical geostatistical tools
    - two-point statistics analysis: covariance, variogram, connectivity, etc.

Miscellaneous algorithms
========================
- Other algorithms based on random processes
    - accept-reject sampler in one or multi-dimension
    - Poisson point process in one or multi-dimension
    - Chentsov simulation in 1D, 2D, 3D
