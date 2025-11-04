#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# Python module:  'srf.py'
# author:         Julien Straubhaar
# date:           sep-2024
# -------------------------------------------------------------------------

"""
Module for the generation of random fields based on substitution random function (SRF).
Random fields in 1D, 2D, 3D.

References
----------
- J. Straubhaar, P. Renard (2024), \
Exploring substitution random functions composed of stationary multi-Gaussian processes. \
Stochastic Environmental Research and Risk Assessment, \
`doi:10.1007/s00477-024-02662-x <https://doi.org/10.1007/s00477-024-02662-x>`_
- C. Lantuéjoul (2002) Geostatistical Simulation, Models and Algorithms. \
Springer Verlag, Berlin, 256 p.
"""

import numpy as np
import scipy.stats as stats
from scipy.interpolate import interp1d
from geone import covModel as gcm
from geone import markovChain as mc
from geone import multiGaussian

# ============================================================================
class SrfError(Exception):
    """
    Custom exception related to `srf` module.
    """
    pass
# ============================================================================

# ============================================================================
# Tools for simulating categorical SRF with
#     - multi-Gaussian simulation as directing function (latent field)
#     - Markov chain as coding process
# ============================================================================

# ----------------------------------------------------------------------------
def srf_mg_mc(
        cov_model_T, kernel_Y,
        dimension, spacing=None, origin=None,
        spacing_Y=0.001,
        categVal=None,
        x=None, v=None,
        t=None, yt=None,
        algo_T='fft', params_T=None,
        mh_iter=100, ntry_max=1,
        nreal=1,
        full_output=True,
        verbose=1,
        logger=None):
    """
    Substitution Random Function (SRF) - multi-Gaussian + Markov chain (on finite set).

    This function allows to generate categorical random fields in 1D, 2D, 3D, based on
    a SRF Z defined as

    - Z(x) = Y(T(x))

    where

    - T is the directing function, a multi-Gaussian random field (latent field)
    - Y is the coding process, a Markov chain on finite sets (of categories) (1D)

    Z and T are fields in 1D, 2D or 3D.

    Notes
    -----
    The module :mod:`multiGaussian` is used for the multi-Gaussian field T, and the
    module :mod:`markovChain` is used for the markov chain Y.

    Parameters
    ----------
    cov_model_T : :class:`geone.covModel.CovModel<d>D`
        covariance model for T, in 1D or 2D or 3D

    kernel_Y : 2d-array of shape (n, n)
        transition kernel for Y of a Markov chain on a set of states
        :math:`S=\\{0, \\ldots, n-1\\}`, where `n` is the number of categories
        (states); the element at row `i` and column `j` is the probability to have
        the state of index `j` at the next step given the state `i` at the current
        step, i.e.

        - :math:`kernel[i][j] = P(Y_{k+1}=j\\ \\vert\\ Y_{k}=i)`

        where the sequence of random variables :math:`(Y_k)` is a Markov chain
        on `S` defined by the kernel `kernel`.

        In particular, every element of `kernel` is positive or zero, and its
        rows sum to one.

    dimension : [sequence of] int(s)
        number of cells along each axis, for simulation in:

        - 1D: `dimension=nx`
        - 2D: `dimension=(nx, ny)`
        - 3D: `dimension=(nx, ny, nz)`

    spacing : [sequence of] float(s), optional
        cell size along each axis, for simulation in:

        - 1D: `spacing=sx`
        - 2D: `spacing=(sx, sy)`
        - 3D: `spacing=(sx, sy, sz)`

        by default (`None`): 1.0 along each axis

    origin : [sequence of] float(s), optional
        origin of the grid ("corner of the first cell"), for simulation in:

        - 1D: `origin=ox`
        - 2D: `origin=(ox, oy)`
        - 3D: `origin=(ox, oy, oz)`

        by default (`None`): 0.0 along each axis

    spacing_Y : float, default: 0.001
        positive value, resolution of the Y process, spacing along abscissa
        between two steps in the Markov chain Y (btw. two adjacent cell in
        1D-grid for Y)

    categVal : 1d-array of shape (n,), optional
        values of categories (one value for each state `0, ..., n-1`);
        by default (`None`) : `categVal` is set to `[0, ..., n-1]`

    x : array-like of floats, optional
        data points locations (float coordinates), for simulation in:

        - 1D: 1D array-like of floats
        - 2D: 2D array-like of floats of shape (m, 2)
        - 3D: 2D array-like of floats of shape (m, 3)

        note: if one point (m=1), a float in 1D, a 1D array of shape (2,) in 2D,
        a 1D array of shape (3,) in 3D, is accepted

    v : 1d-array-like of floats, optional
        data values at `x` (`v[i]` is the data value at `x[i]`)

    t : 1d-array-like of floats, or float, optional
        values of T considered as conditioning point for Y(T) (additional constraint)

    yt : 1d-array-like of floats, or float, optional
        value of Y at the conditioning point `t` (same length as `t`)

    algo_T : str
        defines the algorithm used for generating multi-Gaussian field T:

        - 'fft' or 'FFT' (default): based on circulant embedding and FFT, \
        function called for <d>D (d = 1, 2, or 3): `geone.grf.grf<d>D`
        - 'classic' or 'CLASSIC': classic algorithm, based on the resolution \
        of kriging system considered points in a search ellipsoid, function \
        called for <d>D (d = 1, 2, or 3): `geone.geoscalassicinterface.simulate<d>D`

    params_T : dict, optional
        keyword arguments (additional parameters) to be passed to the function
        corresponding to what is specified by the argument `algo_T` (see the
        corresponding function for its keyword arguments), in particular the key
        'mean' can be specified (set to value 0 if not specified)

    mh_iter : int, default: 100
        number of iteration for Metropolis-Hasting algorithm, for conditional
        simulation only; note: used only if `x` or `t` is not `None`

    ntry_max : int, default: 1
        number of tries per realization before giving up if something goes wrong

    nreal : int, default: 1
        number of realization(s)

    full_output : bool, default: True
        - if `True`: simulation(s) of Z, T, and Y are retrieved in output
        - if `False`: simulation(s) of Z only is retrieved in output

    verbose : int, default: 1
        verbose mode, integer >=0, higher implies more display

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    Z : nd-array
        all realizations, `Z[k]` is the `k`-th realization:

        - for 1D: `Z` of shape (nreal, nx), where nx = dimension
        - for 2D: `Z` of shape (nreal, ny, nx), where nx, ny = dimension
        - for 3D: `Z` of shape (nreal, nz, ny, nx), where nx, ny, nz = dimension

    T : nd-array
        latent fields of all realizations, `T[k]` for the `k`-th realization:

        - for 1D: `T` of shape (nreal, nx), where nx = dimension
        - for 2D: `T` of shape (nreal, ny, nx), where nx, ny = dimension
        - for 3D: `T` of shape (nreal, nz, ny, nx), where nx, ny, nz = dimension

        returned if `full_output=True`

    Y : list of length nreal
        markov chains of all realizations, `Y[k]` is a list of length 4 for
        the `k`-th realization:

        - Y[k][0]: int, Y_nt (number of cell along t-axis)
        - Y[k][1]: float, Y_st (cell size along t-axis)
        - Y[k][2]: float, Y_ot (origin)
        - Y[k][3]: 1d-array of shape (Y_nt,), values of Y[k]

        returned if `full_output=True`
    """
    fname = 'srf_mg_mc'

    if algo_T not in ('fft', 'FFT', 'classic', 'CLASSIC'):
        err_msg = f"{fname}: `algo_T` invalid, should be 'fft' (default) or 'classic'"
        if logger: logger.error(err_msg)
        raise SrfError(err_msg)

    # Set space dimension (of grid) according to covariance model for T
    if isinstance(cov_model_T, gcm.CovModel1D):
        d = 1
    elif isinstance(cov_model_T, gcm.CovModel2D):
        d = 2
    elif isinstance(cov_model_T, gcm.CovModel3D):
        d = 3
    else:
        err_msg = f'{fname}: `cov_model_T` invalid, should be a class `geone.covModel.CovModel1D`, `geone.covModel.CovModel2D` or `geone.covModel.CovModel3D`'
        if logger: logger.error(err_msg)
        raise SrfError(err_msg)

    # Check argument 'dimension'
    if hasattr(dimension, '__len__') and len(dimension) != d:
        err_msg = f'{fname}: `dimension` of incompatible length'
        if logger: logger.error(err_msg)
        raise SrfError(err_msg)

    if d == 1:
        grid_size = dimension
    else:
        grid_size = np.prod(dimension)

    # Check (or set) argument 'spacing'
    if spacing is None:
        if d == 1:
            spacing = 1.0
        else:
            spacing = tuple(np.ones(d))
    else:
        if hasattr(spacing, '__len__') and len(spacing) != d:
            err_msg = f'{fname}: `spacing` of incompatible length'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

    # Check (or set) argument 'origin'
    if origin is None:
        if d == 1:
            origin = 0.0
        else:
            origin = tuple(np.zeros(d))
    else:
        if hasattr(origin, '__len__') and len(origin) != d:
            err_msg = f'{fname}: `origin` of incompatible length'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

    # if not cov_model_T.is_stationary(): # prevent calculation if covariance model is not stationary
    #     if verbose > 0:
    #         print(f'ERROR ({fname}): `cov_model_T` is not stationary')

    # Check kernel for Y
    if not isinstance(kernel_Y, np.ndarray) or kernel_Y.ndim != 2 or kernel_Y.shape[0] != kernel_Y.shape[1]:
        err_msg = f'{fname}: `kernel_Y` is not a square matrix (2d array)'
        if logger: logger.error(err_msg)
        raise SrfError(err_msg)

    if np.any(kernel_Y < 0) or not np.all(np.isclose(kernel_Y.sum(axis=1), 1.0)):
        err_msg = f'{fname}: `kernel_Y` is not a transition probability matrix'
        if logger: logger.error(err_msg)
        raise SrfError(err_msg)

    # Number of categories (order of the kernel)
    n = kernel_Y.shape[0]

    # Check category values
    if categVal is None:
        categVal = np.arange(n)
    else:
        categVal = np.asarray(categVal)
        if categVal.ndim != 1 or categVal.shape[0] != n:
            err_msg = f'{fname}: `categVal` invalid'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

        if len(np.unique(categVal)) != len(categVal):
            err_msg = f'{fname}: `categVal` contains duplicated values'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

    # Check additional constraint t (conditioning point for T), yt (corresponding value for Y)
    if t is None:
        if yt is not None:
            err_msg = f'{fname}: `t` is not given (`None`) but `yt` is given (not `None`)'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

    else:
        if yt is None:
            err_msg = f'{fname}: `t` is given (not `None`) but `yt` is not given (`None`)'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

        t = np.asarray(t, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        yt = np.asarray(yt, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(yt) != len(t):
            err_msg = f'{fname}: length of `yt` is not valid'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

        # Check values
        if not np.all([yv in categVal for yv in yt]):
            err_msg = f'{fname}: `yt` contains an invalid value'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

    # Initialize dictionary params_T
    if params_T is None:
        params_T = {}

    # Compute meshgrid over simulation domain if needed (see below)
    if ('mean' in params_T.keys() and callable(params_T['mean'])) or ('var' in params_T.keys() and callable(params_T['var'])):
        if d == 1:
            xi = origin + spacing*(0.5+np.arange(dimension)) # x-coordinate of cell center
        elif d == 2:
            xi = origin[0] + spacing[0]*(0.5+np.arange(dimension[0])) # x-coordinate of cell center
            yi = origin[1] + spacing[1]*(0.5+np.arange(dimension[1])) # y-coordinate of cell center
            yyi, xxi = np.meshgrid(yi, xi, indexing='ij')
        elif d == 3:
            xi = origin[0] + spacing[0]*(0.5+np.arange(dimension[0])) # x-coordinate of cell center
            yi = origin[1] + spacing[1]*(0.5+np.arange(dimension[1])) # y-coordinate of cell center
            zi = origin[2] + spacing[2]*(0.5+np.arange(dimension[2])) # z-coordinate of cell center
            zzi, yyi, xxi = np.meshgrid(zi, yi, xi, indexing='ij')

    # Set mean_T (as array) from params_T
    if 'mean' not in params_T.keys():
        mean_T = np.array([0.0])
    else:
        mean_T = params_T['mean']
        if mean_T is None:
            mean_T = np.array([0.0])
        elif callable(mean_T):
            if d == 1:
                mean_T = mean_T(xi).reshape(-1) # replace function 'mean_T' by its evaluation on the grid
            elif d == 2:
                mean_T = mean_T(xxi, yyi).reshape(-1) # replace function 'mean_T' by its evaluation on the grid
            elif d == 3:
                mean_T = mean_T(xxi, yyi, zzi).reshape(-1) # replace function 'mean_T' by its evaluation on the grid
        else:
            mean_T = np.asarray(mean_T).reshape(-1)
            if mean_T.size not in (1, grid_size):
                err_msg = f"{fname}: 'mean' parameter for T (in `params_T`) has incompatible size"
                if logger: logger.error(err_msg)
                raise SrfError(err_msg)

    # Set var_T (as array) from params_T, if given
    var_T = None
    if 'var' in params_T.keys():
        var_T = params_T['var']
        if var_T is not None:
            if callable(var_T):
                if d == 1:
                    var_T = var_T(xi).reshape(-1) # replace function 'var_T' by its evaluation on the grid
                elif d == 2:
                    var_T = var_T(xxi, yyi).reshape(-1) # replace function 'var_T' by its evaluation on the grid
                elif d == 3:
                    var_T = var_T(xxi, yyi, zzi).reshape(-1) # replace function 'var_T' by its evaluation on the grid
            else:
                var_T = np.asarray(var_T).reshape(-1)
                if var_T.size not in (1, grid_size):
                    err_msg = f"{fname}: 'var' parameter for T (in `params_T`) has incompatible size"
                    if logger: logger.error(err_msg)
                    raise SrfError(err_msg)

    # Number of realization(s)
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if full_output:
            if verbose > 0:
                if logger:
                    logger.warning(f'{fname}: `nreal` <= 0: `None`, `None`, `None` is returned')
                else:
                    print(f'{fname}: WARNING: `nreal` <= 0: `None`, `None`, `None` is returned')
            return None, None, None
        else:
            if verbose > 0:
                if logger:
                    logger.warning(f'{fname}: `nreal` <= 0: `None` is returned')
                else:
                    print(f'{fname}: WARNING: `nreal` <= 0: `None` is returned')
            return None

    # Note: format of data (x, v) not checked !

    if x is None:
        # Preparation for unconditional case
        if v is not None:
            err_msg = f'{fname}: `x` is not given (`None`) but `v` is given (not `None`)'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

    else:
        # Preparation for conditional case
        if v is None:
            err_msg = f'{fname}: `x` is given (not `None`) but `v` is not given (`None`)'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

        x = np.asarray(x, dtype='float').reshape(-1, d) # cast in d-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

        # Check values
        if not np.all([yv in categVal for yv in v]):
            err_msg = f'{fname}: `v` contains an invalid value'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

        # Number of conditioning points
        npt = x.shape[0]

        # Get index in mean_T for each conditioning points
        x_mean_T_grid_ind = None
        if mean_T.size == 1:
            x_mean_T_grid_ind = np.zeros(npt, dtype='int')
        else:
            indc_f = (x-origin)/spacing
            indc = indc_f.astype(int)
            indc = indc - 1 * np.all((indc == indc_f, indc > 0), axis=0)
            if d == 1:
                x_mean_T_grid_ind = 1 * indc[:, 0] # multiply by 1.0 makes a copy of the array !
            elif d == 2:
                x_mean_T_grid_ind = indc[:, 0] + dimension[0] * indc[:, 1]
            elif d == 3:
                x_mean_T_grid_ind = indc[:, 0] + dimension[0] * (indc[:, 1] + dimension[1] * indc[:, 2])
        #
        # Get index in var_T (if not None) for each conditioning points
        if var_T is not None:
            if var_T.size == 1:
                x_var_T_grid_ind = np.zeros(npt, dtype='int')
            else:
                if x_mean_T_grid_ind is not None:
                    x_var_T_grid_ind = x_mean_T_grid_ind
                else:
                    indc_f = (x-origin)/spacing
                    indc = indc_f.astype(int)
                    indc = indc - 1 * np.all((indc == indc_f, indc > 0), axis=0)
                    if d == 1:
                        x_var_T_grid_ind = 1 * indc[:, 0] # multiply by 1.0 makes a copy of the array !
                    elif d == 2:
                        x_var_T_grid_ind = indc[:, 0] + dimension[0] * indc[:, 1]
                    elif d == 3:
                        x_var_T_grid_ind = indc[:, 0] + dimension[0] * (indc[:, 1] + dimension[1] * indc[:, 2])

        # Get covariance function for T
        cov_func_T = cov_model_T.func() # covariance function

        # Get evaluation of covariance function for T at 0
        cov0_T = cov_func_T(np.zeros(d))

        # Set kriging matrix for T (mat_T) of order npt, "over every conditioining point"
        mat_T = np.ones((npt, npt))
        for i in range(npt-1):
            # lag between x[i] and x[j], j=i+1, ..., npt-1
            h = x[(i+1):] - x[i]
            cov_h_T = cov_func_T(h)
            mat_T[i, (i+1):npt] = cov_h_T
            mat_T[(i+1):npt, i] = cov_h_T
            mat_T[i, i] = cov0_T

        mat_T[-1,-1] = cov0_T

        if var_T is not None:
            varUpdate = np.sqrt(var_T[x_var_T_grid_ind]/cov0_T)
            mat_T = varUpdate*(mat_T.T*varUpdate).T

        # Initialize
        #   - npt_ext: number of total conditioning point for Y, "point T(x) + additional constraint t"
        #   - v_T: values of T(x) (that are defined later) followed by values yt at additional constraint t"
        #   - v_ext: values for Y at "point T(x) + additional constraint (t)"
        if t is None:
            npt_ext = npt
            v_T = np.zeros(npt)
            v_ext = v
        else:
            npt_ext = npt + len(t)
            v_T = np.hstack((np.zeros(npt), t))
            v_ext = np.hstack((v, yt))

        # Set index in categVal of values v_ext
        v_ext_cat = np.array([np.where(categVal==yv)[0][0] for yv in v_ext], dtype='int')

        if npt_ext <= 1:
            mh_iter = 0 # unnecessary to apply Metropolis update !

    # Preparation of
    #     - pinv : invariant distribution
    #     - kernel_Y_rev (reverse transition kernel)
    #     - kernel_Y_pow (kernel raised to power 0, 1, 2, ...)
    #     - kernel_Y_rev_pow (reverse kernel raised to power 0, 1, 2, ...)
    try:
        pinv_Y = mc.compute_mc_pinv(kernel_Y, logger=logger)
    except Exception as exc:
        err_msg = f'{fname}: computing invariant distribution for Y failed'
        if logger: logger.error(err_msg)
        raise SrfError(err_msg) from exc

    try:
        kernel_Y_rev = mc.compute_mc_kernel_rev(kernel_Y, pinv=pinv_Y, logger=logger)
    except Exception as exc:
        err_msg = f'{fname}: kernel for Y not reversible'
        if logger: logger.error(err_msg)
        raise SrfError(err_msg) from exc

    m_pow = 1
    kernel_Y_pow = np.zeros((m_pow, n, n))
    kernel_Y_pow[0] = np.eye(n)

    m_rev_pow = 1
    kernel_Y_rev_pow = np.zeros((m_rev_pow, n, n))
    kernel_Y_rev_pow[0] = np.eye(n)

    if t is not None and len(t) > 1:
        # Check validity of additional constraint (t, yt):
        #    check the compatibility with kernel_Y, i.e. that the probabilities:
        #       Prob(Y[t[k1]]=yt[k1], Y[t[k2]]=yt[k2]) > 0, for all pairs t[k1] < t[k2]
        #
        # Compute
        #    yind: node index of conditioning node (nearest node),
        #          rounded to lower index if between two grid node and index is positive
        yind_f = (t-t.min())/spacing_Y
        yind = yind_f.astype(int)
        yind = yind - 1 * np.all((yind == yind_f, yind > 0), axis=0)
        #
        # Set index in categVal of values yt
        yval_cat = np.array([np.where(categVal==yv)[0][0] for yv in yt], dtype='int')
        #
        inds = np.argsort(yind)
        i0 = max(np.diff([yind[j] for j in inds]))
        if i0 >= m_pow:
            kernel_Y_pow = np.concatenate((kernel_Y_pow, np.zeros((i0-m_pow+1, n, n))), axis=0)
            for i in range(m_pow, i0+1):
                kernel_Y_pow[i] = kernel_Y_pow[i-1].dot(kernel_Y)
            m_pow = i0+1
        # check if Prob(Y[t[inds[i+1]]]=yt[inds[i+1]], Y[t[inds[i]]]=yt[inds[i]]) = kernel^(inds[i+1]-inds[i])[yval_cat[inds[i]], yval_cat[inds[i+1]]] > 0, for all i
        if np.any(np.isclose([kernel_Y_pow[yind[inds[i+1]]-yind[inds[i]], int(yval_cat[inds[i]]), int(yval_cat[inds[i+1]])] for i in range(len(t)-1)], 0)):
        # if np.any([kernel_Y_pow[yind[inds[i+1]]-yind[inds[i]], int(yval_cat[inds[i]]), int(yval_cat[inds[i+1]])] < 1.e-20 for i in range(len(t)-1)]):
            err_msg = f'{fname}: invalid additional constraint on Markov chain Y wrt. kernel'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

    # Set (again if given) default parameter 'mean' and 'var' for T
    params_T['mean'] = mean_T
    params_T['var'] = var_T

    # Set default parameter 'verbose' for params_T
    if 'verbose' not in params_T.keys():
        params_T['verbose'] = 0
        # params_T['verbose'] = verbose

    # Initialization for output
    Z = []
    if full_output:
        T = []
        Y = []

    for ireal in range(nreal):
        # Generate ireal-th realization
        if verbose > 1:
            if logger:
                logger.info(f'{fname}: simulation {ireal+1} of {nreal}...')
            else:
                print(f'{fname}: simulation {ireal+1} of {nreal}...')
        for ntry in range(ntry_max):
            sim_ok = True
            if verbose > 2 and ntry > 0:
                if logger:
                    logger.info(f'{fname}:   ... new trial ({ntry+1} of {ntry_max}) for simulation {ireal+1} of {nreal}...')
                else:
                    print(f'{fname}:   ... new trial ({ntry+1} of {ntry_max}) for simulation {ireal+1} of {nreal}...')
            if x is None:
                # Unconditional case
                # ------------------
                # Generate T (one real)
                try:
                    sim_T = multiGaussian.multiGaussianRun(
                            cov_model_T, dimension, spacing, origin,
                            mode='simulation', algo=algo_T, output_mode='array',
                            **params_T, nreal=1, logger=logger)
                except:
                    sim_ok = False
                    if verbose > 2:
                        if logger:
                            logger.info(f'{fname}:   ... simulation of T failed')
                        else:
                            print(f'{fname}:   ... simulation of T failed')
                    continue
                # except Exception as exc:
                #     err_msg = f'{fname}: simulation of T failed'
                #     if logger: logger.error(err_msg)
                #     raise SrfError(err_msg) from exc

                # -> nd-array of shape
                #      (1, dimension) (for T in 1D)
                #      (1, dimension[1], dimension[0]) (for T in 2D)
                #      (1, dimension[2], dimension[1], dimension[0]) (for T in 3D)

                # Set origin and dimension for Y
                min_T = np.min(sim_T)
                max_T = np.max(sim_T)
                if t is not None:
                    min_T = min(t.min(), min_T)
                    max_T = max(t.max(), max_T)
                min_T = min_T - 0.5 * spacing_Y
                max_T = max_T + 0.5 * spacing_Y
                dimension_Y = int(np.ceil((max_T - min_T)/spacing_Y))
                origin_Y = min_T - 0.5*(dimension_Y*spacing_Y - (max_T - min_T))

                if t is not None:
                    # Compute
                    #    yind: node index of conditioning node (nearest node),
                    #          rounded to lower index if between two grid node and index is positive
                    yind_f = (t-origin_Y)/spacing_Y
                    yind = yind_f.astype(int)
                    yind = yind - 1 * np.all((yind == yind_f, yind > 0), axis=0)
                    #
                    yval = yt
                else:
                    yind, yval = None, None

                # Generate Y conditional to possible additional constraint (t, yt) (one real)
                try:
                    mc_Y = mc.simulate_mc(
                            kernel_Y, dimension_Y,
                            categVal=categVal, data_ind=yind, data_val=yval,
                            pinv=pinv_Y, kernel_rev=kernel_Y_rev, kernel_pow=kernel_Y_pow,
                            nreal=1,
                            logger=logger)
                except:
                    sim_ok = False
                    if verbose > 2:
                        if logger:
                            logger.info(f'{fname}:   ...  simulation of Markov chain Y failed')
                        else:
                            print(f'{fname}:   ...  simulation of Markov chain Y failed')
                    continue
                # except Exception as exc:
                #     err_msg = f'{fname}: simulation of Markov chain Y failed'
                #     if logger: logger.error(err_msg)
                #     raise SrfError(err_msg) from exc

                # -> 2d-array of shape (1, dimension_Y)

            else:
                # Conditional case
                # ----------------
                # Initialize: unconditional simulation of T at x (values in v_T)
                ind = np.random.permutation(npt)
                for j, k in enumerate(ind):
                    # Simulate value at x[k] (= x[ind[j]]), conditionally to the previous ones
                    # Solve the kriging system (for T)
                    try:
                        w = np.linalg.solve(
                                mat_T[ind[:j], :][:, ind[:j]], # kriging matrix
                                mat_T[ind[:j], ind[j]], # second member
                            )
                    except:
                        sim_ok = False
                        break

                    # Mean (kriged) value at x[k]
                    mu_T_k = mean_T[x_mean_T_grid_ind[k]] + (v_T[ind[:j]] - mean_T[x_mean_T_grid_ind[ind[:j]]]).dot(w)
                    # Standard deviation (of kriging) at x[k]
                    std_T_k = np.sqrt(np.maximum(0, cov0_T - np.dot(w, mat_T[ind[:j], ind[j]])))
                    # Draw value in N(mu_T_k, std_T_k^2)
                    v_T[k] = np.random.normal(loc=mu_T_k, scale=std_T_k)

                if not sim_ok:
                    sim_ok = False
                    if verbose > 2:
                        if logger:
                            logger.info(f'{fname}:    ... cannot solve kriging system (for T, initialization)')
                        else:
                            print(f'{fname}:    ... cannot solve kriging system (for T, initialization)')
                    continue

                # Update simulated values v_T at x using Metropolis-Hasting (MH) algorithm
                for nit in range(mh_iter):
                    if verbose > 3:
                        if logger:
                            logger.info(f'{fname}:   ... sim {ireal+1} of {nreal}: MH iter {nit+1} of {mh_iter}...')
                        else:
                            print(f'{fname}:   ... sim {ireal+1} of {nreal}: MH iter {nit+1} of {mh_iter}...')
                    ind = np.random.permutation(npt)
                    for k in ind:
                        # Sequence of indexes without k
                        indmat = np.hstack((np.arange(k), np.arange(k+1, npt)))
                        # Simulate possible new value v_T_new at x[k], conditionally to all the ohter ones
                        #
                        # Solve the kriging system for T
                        try:
                            w = np.linalg.solve(
                                    mat_T[indmat, :][:, indmat], # kriging matrix
                                    mat_T[indmat, k], # second member
                                )
                        except:
                            sim_ok = False
                            if verbose > 2:
                                if logger:
                                    logger.info(f'{fname}:   ... cannot solve kriging system (for T)')
                                else:
                                    print(f'{fname}:   ... cannot solve kriging system (for T)')
                            break
                        #
                        # Mean (kriged) value at x[k]
                        mu_T_k = mean_T[x_mean_T_grid_ind[k]] + (v_T[indmat] - mean_T[x_mean_T_grid_ind[indmat]]).dot(w)
                        # Standard deviation (of kriging) at x[k]
                        std_T_k = np.sqrt(np.maximum(0, cov0_T - np.dot(w, mat_T[indmat, k])))
                        # Draw value in N(mu, std^2)
                        v_T_k_new = np.random.normal(loc=mu_T_k, scale=std_T_k)
                        #
                        # Compute MH quotient defined as
                        #    p_new / p
                        # where:
                        #    p_new = prob(Y[v_T_k_new] = v[k] | Y[indmat] = v[indmat], Y[t] = yt)
                        #    p = prob(Y[v_T[k]] = v[k] | Y[indmat] = v[indmat], Y[t] = yt)
                        inds = np.argsort(v_T)
                        # --- Compute p ---
                        v_T_k_i = np.where(inds==k)[0][0]
                        # v_T[k] = v_T[inds[v_T_k_i]]
                        if v_T_k_i == 0:
                            # v_T[k] is the smallest value in v_T
                            # we have
                            #     v_T[k]  <= v_T[inds[v_T_k_i+1]]
                            #     p = prob(Y[v_T[k]] = v_ext[k] | Y[v_T[inds[v_T_k_i+1]]] = v_ext[inds[v_T_k_i+1]])
                            i1 = int((v_T[inds[v_T_k_i+1]] - v_T[k]) / spacing_Y)
                            #     p = kernel_Y_rev^i1[v_ext_cat[inds[v_T_k_i+1]], v_ext_cat[k]]
                            if i1 >= m_rev_pow:
                                kernel_Y_rev_pow = np.concatenate((kernel_Y_rev_pow, np.zeros((i1-m_rev_pow+1, n, n))), axis=0)
                                for i in range(m_rev_pow, i1+1):
                                    kernel_Y_rev_pow[i] = kernel_Y_rev_pow[i-1].dot(kernel_Y_rev)
                                m_rev_pow = i1+1
                            p = kernel_Y_rev_pow[i1, v_ext_cat[inds[v_T_k_i+1]], v_ext_cat[k]]
                        elif v_T_k_i == npt_ext-1:
                            # v_T[k] is the largest value in v_T
                            # we have
                            #     v_T[inds[v_T_k_i-1]] <= v_T[k]
                            #     p = prob(Y[v_T[k]] = v_ext[k] | Y[v_T[inds[v_T_k_i-1]]] = v_ext[inds[v_T_k_i-1]])
                            i0 = int((v_T[k] - v_T[inds[v_T_k_i-1]]) / spacing_Y)
                            #     p = kernel_Y^i0[v_ext_cat[inds[v_T_k_i-1]], v_ext_cat[k]]
                            if i0 >= m_pow:
                                kernel_Y_pow = np.concatenate((kernel_Y_pow, np.zeros((i0-m_pow+1, n, n))), axis=0)
                                for i in range(m_pow, i0+1):
                                    kernel_Y_pow[i] = kernel_Y_pow[i-1].dot(kernel_Y)
                                m_pow = i0+1
                            p = kernel_Y_pow[i0, v_ext_cat[inds[v_T_k_i-1]], v_ext_cat[k]]
                        else:
                            # v_T[k] is neither the smallest nor the largest value in v_T
                            # we have
                            #     v_T[inds[v_T_k_i-1]] <= v_T[k] <= v_T[inds[v_T_k_i+1]]
                            #     p = prob(Y[v_T[k]] = v_ext[k] | Y[v_T[inds[v_T_k_i-1]]] = v_ext[inds[v_T_k_i-1]], Y[v_T[inds[v_T_k_i+1]]] = v_ext[inds[v_T_k_i+1]])
                            i0 = int((v_T[k] - v_T[inds[v_T_k_i-1]]) / spacing_Y)
                            i1 = int((v_T[inds[v_T_k_i+1]] - v_T[k]) / spacing_Y)
                            #     p = kernel_Y^i0[v_ext_cat[inds[v_T_k_i-1]], v_ext_cat[k]] * kernel_Y^i1[v_ext_cat[k], v_ext_cat[inds[v_T_k_i+1]]] / kernel_Y^(i0+i1)[v_ext_cat[inds[v_T_k_i-1]], v_ext_cat[inds[v_T_k_i+1]]]
                            ii = i0+i1
                            if ii >= m_pow:
                                kernel_Y_pow = np.concatenate((kernel_Y_pow, np.zeros((ii-m_pow+1, n, n))), axis=0)
                                for i in range(m_pow, ii+1):
                                    kernel_Y_pow[i] = kernel_Y_pow[i-1].dot(kernel_Y)
                                m_pow = ii+1
                            denom = kernel_Y_pow[ii, v_ext_cat[inds[v_T_k_i-1]], v_ext_cat[inds[v_T_k_i+1]]]
                            if np.isclose(denom, 0):
                                # p = 0.
                                # Accept new value v_T_new at x[k]
                                v_T[k] = v_T_k_new
                                continue
                            else:
                                p = kernel_Y_pow[i0, v_ext_cat[inds[v_T_k_i-1]], v_ext_cat[k]] * kernel_Y_pow[i1, v_ext_cat[k], v_ext_cat[inds[v_T_k_i+1]]] / denom
                        # --- Compute p_new ---
                        v_T_k_new_i = npt_ext
                        for i in range(npt_ext):
                            if v_T[inds[i]]>=v_T_k_new:
                                v_T_k_new_i = i
                                break
                        if v_T_k_new_i == 0:
                            # v_T_k_new <= v_T[i] for all i
                            # we have
                            #     v_T_k_new <= v_T[inds[0]]
                            #     p_new = prob(Y[v_T_k_new] = v_ext[k] | Y[v_T[inds[0]]] = v_ext[inds[0]])
                            i1 = int((v_T[inds[0]] - v_T_k_new) / spacing_Y)
                            #     p_new = kernel_Y_rev^i1[v_ext_cat[inds[0]], v_ext_cat[k]]
                            if i1 >= m_rev_pow:
                                kernel_Y_rev_pow = np.concatenate((kernel_Y_rev_pow, np.zeros((i1-m_rev_pow+1, n, n))), axis=0)
                                for i in range(m_rev_pow, i1+1):
                                    kernel_Y_rev_pow[i] = kernel_Y_rev_pow[i-1].dot(kernel_Y_rev)
                                m_rev_pow = i1+1
                            p_new = kernel_Y_rev_pow[i1, v_ext_cat[inds[0]], v_ext_cat[k]]
                        elif v_T_k_new_i == npt_ext:
                            # v_T_k_new > v_T[i] for all i
                            # we have
                            #     v_T[inds[npt_ext-1] <= v_T_k_new
                            #     p_new = prob(Y[v_T_k_new] = v_ext[k] | Y[v_T[inds[npt_ext-1]]] = v_ext[inds[npt_ext-1]])
                            i0 = int((v_T_k_new - v_T[inds[npt_ext-1]]) / spacing_Y)
                            #     p_new = kernel_Y^i0[v_ext_cat[inds[npt_ext-1]], v_ext_cat[k]]
                            if i0 >= m_pow:
                                kernel_Y_pow = np.concatenate((kernel_Y_pow, np.zeros((i0-m_pow+1, n, n))), axis=0)
                                for i in range(m_pow, i0+1):
                                    kernel_Y_pow[i] = kernel_Y_pow[i-1].dot(kernel_Y)
                                m_pow = i0+1
                            p_new = kernel_Y_pow[i0, v_ext_cat[npt_ext-1], v_ext_cat[k]]
                        else:
                            # we have
                            #     v_T[inds[v_T_k_new_i-1]] < v_T_k_new <= v_T[inds[v_T_k_new_i]]
                            #     p_new = prob(Y[v_T_k_new] = v_ext[k] | Y[v_T[inds[v_T_k_new_i-1]]] = v_ext[inds[v_T_k_new_i-1]], Y[v_T[inds[v_T_k_new_i]]] = v_ext[inds[v_T_k_new_i-1]])
                            i0 = int((v_T_k_new - v_T[inds[v_T_k_new_i-1]]) / spacing_Y)
                            i1 = int((v_T[inds[v_T_k_new_i]] - v_T_k_new) / spacing_Y)
                            #     p = kernel_Y^i0[v_ext_cat[inds[v_T_k_new_i-1]], v_ext_cat[k]] * kernel_Y^i1[v_ext_cat[k], v_ext_cat[inds[v_T_k_new_i]]] / kernel_Y^(i0+i1)[v_ext_cat[inds[v_T_k_new_i-1]], v_ext_cat[inds[v_T_k_new_i]]]
                            ii = i0+i1
                            if ii >= m_pow:
                                kernel_Y_pow = np.concatenate((kernel_Y_pow, np.zeros((ii-m_pow+1, n, n))), axis=0)
                                for i in range(m_pow, ii+1):
                                    kernel_Y_pow[i] = kernel_Y_pow[i-1].dot(kernel_Y)
                                m_pow = ii+1
                            denom = kernel_Y_pow[ii, v_ext_cat[inds[v_T_k_new_i-1]], v_ext_cat[inds[v_T_k_new_i]]]
                            if np.isclose(denom, 0):
                                p_new = 0.
                            else:
                                p_new = kernel_Y_pow[i0, v_ext_cat[inds[v_T_k_new_i-1]], v_ext_cat[k]] * kernel_Y_pow[i1, v_ext_cat[k], v_ext_cat[inds[v_T_k_new_i]]] / denom
                        #
                        mh_quotient = p_new/p
                        if mh_quotient >= 1.0 or np.random.random() < mh_quotient:
                            # Accept new value v_T_new at x[k]
                            v_T[k] = v_T_k_new
                    #
                    if not sim_ok:
                        break
                #
                if not sim_ok:
                    continue

                # Generate T conditional to (x, v_T[0:npt]) (one real)
                try:
                    sim_T = multiGaussian.multiGaussianRun(
                            cov_model_T, dimension, spacing, origin, x=x, v=v_T[:npt],
                            mode='simulation', algo=algo_T, output_mode='array',
                            **params_T, nreal=1, logger=logger)
                except:
                    sim_ok = False
                    if verbose > 2:
                        if logger:
                            logger.info(f'{fname}:   ... conditional simulation of T failed')
                        else:
                            print(f'{fname}:   ... conditional simulation of T failed')
                    continue
                # except Exception as exc:
                #     err_msg = f'{fname}: conditional simulation of T failed'
                #     if logger: logger.error(err_msg)
                #     raise SrfError(err_msg) from exc

                # -> nd-array of shape
                #      (1, dimension) (for T in 1D)
                #      (1, dimension[1], dimension[0]) (for T in 2D)
                #      (1, dimension[2], dimension[1], dimension[0]) (for T in 3D)

                # Set origin and dimension for Y
                min_T = np.min(sim_T)
                max_T = np.max(sim_T)
                if t is not None:
                    min_T = min(t.min(), min_T)
                    max_T = max(t.max(), max_T)
                min_T = min_T - 0.5 * spacing_Y
                max_T = max_T + 0.5 * spacing_Y
                dimension_Y = int(np.ceil((max_T - min_T)/spacing_Y))
                origin_Y = min_T - 0.5*(dimension_Y*spacing_Y - (max_T - min_T))

                # Compute
                #    yind: node index (nearest node),
                #          rounded to lower index if between two grid nodes and index is positive
                yind_f = (v_T-origin_Y)/spacing_Y
                yind = yind_f.astype(int)
                yind = yind - 1 * np.all((yind == yind_f, yind > 0), axis=0)

                # Generate Y conditional to (v_T, v_ext) (one real)
                try:
                    mc_Y = mc.simulate_mc(
                            kernel_Y, dimension_Y,
                            categVal=categVal, data_ind=yind, data_val=v_ext,
                            pinv=pinv_Y, kernel_rev=kernel_Y_rev, kernel_pow=kernel_Y_pow,
                            nreal=1,
                            logger=logger)
                except:
                    sim_ok = False
                    if verbose > 2:
                        if logger:
                            logger.info(f'{fname}:   ... conditional simulation of Markov chain Y failed')
                        else:
                            print(f'{fname}:   ... conditional simulation of Markov chain Y failed')
                    continue
                # except Exception as exc:
                #     err_msg = f'{fname}: conditional simulation of Markov chain Y failed'
                #     if logger: logger.error(err_msg)
                #     raise SrfError(err_msg) from exc

                # -> 2d-array of shape (1, dimension_Y)

            # Generate Z (one real)
            # Compute
            #    ind: node index (nearest node),
            #         rounded to lower index if between two grid nodes and index is positive
            ind_f = (sim_T.reshape(-1) - origin_Y)/spacing_Y
            ind = ind_f.astype(int)
            ind = ind - 1 * np.all((ind == ind_f, ind > 0), axis=0)
            Z_real = mc_Y[0][ind]
            # Z_real = mc_Y[0][np.floor((sim_T.reshape(-1) - origin_Y)/spacing_Y).astype(int)]
            if sim_ok:
                Z.append(Z_real)
                if full_output:
                    T.append(sim_T[0])
                    Y.append([dimension_Y, spacing_Y, origin_Y, mc_Y.reshape(dimension_Y)])
                break

    # Get Z
    if verbose > 0 and len(Z) < nreal:
        if logger:
            logger.warning(f'{fname}: some realization failed (missing)')
        else:
            print(f'{fname}: WARNING: some realization failed (missing)')

    Z = np.asarray(Z).reshape(len(Z), *np.atleast_1d(dimension)[::-1])

    if full_output:
        T = np.asarray(T).reshape(len(T), *np.atleast_1d(dimension)[::-1])
        return Z, T, Y
    else:
        return Z
# ----------------------------------------------------------------------------

# ============================================================================
# Tools for simulating continuous SRF with
#     - multi-Gaussian simulation as directing function (latent field)
#     - multi-Gaussian simulation as coding process
# ============================================================================

# ----------------------------------------------------------------------------
def srf_mg_mg(
        cov_model_T, cov_model_Y,
        dimension, spacing=None, origin=None,
        spacing_Y=0.001,
        x=None, v=None,
        t=None, yt=None,
        vmin=None, vmax=None,
        algo_T='fft', params_T=None,
        algo_Y='fft', params_Y=None,
        target_distrib=None,
        initial_distrib=None,
        mh_iter=100,
        ntry_max=1,
        nreal=1,
        full_output=True,
        verbose=1,
        logger=None):
    """
    Substitution Random Function (SRF) - multi-Gaussian + multi-Gaussian.

    This function allows to generate continuous random fields in 1D, 2D, 3D, based on
    a SRF Z defined as

    - Z(x) = Y(T(x))

    where

    - T is the directing function, a multi-Gaussian random field (latent field)
    - Y is the coding process, a multi-Gaussian random process (1D)

    Z and T are fields in 1D, 2D or 3D.

    Notes
    -----
    The module :mod:`multiGaussian` is used for the multi-Gaussian fields T and Y.

    Parameters
    ----------
    cov_model_T : :class:`geone.covModel.CovModel<d>D`
        covariance model for T, in 1D or 2D or 3D

    cov_model_Y : :class:`geone.covModel.CovModel1D`
        covariance model for Y, in 1D

    dimension : [sequence of] int(s)
        number of cells along each axis, for simulation in:

        - 1D: `dimension=nx`
        - 2D: `dimension=(nx, ny)`
        - 3D: `dimension=(nx, ny, nz)`

    spacing : [sequence of] float(s), optional
        cell size along each axis, for simulation in:

        - 1D: `spacing=sx`
        - 2D: `spacing=(sx, sy)`
        - 3D: `spacing=(sx, sy, sz)`

        by default (`None`): 1.0 along each axis

    origin : [sequence of] float(s), optional
        origin of the grid ("corner of the first cell"), for simulation in:

        - 1D: `origin=ox`
        - 2D: `origin=(ox, oy)`
        - 3D: `origin=(ox, oy, oz)`

        by default (`None`): 0.0 along each axis

    spacing_Y : float, default: 0.001
        positive value, resolution of the Y process, spacing along abscissa
        between two cells in the field Y (btw. two adjacent cell in 1D-grid
        for Y)

    x : array-like of floats, optional
        data points locations (float coordinates), for simulation in:

        - 1D: 1D array-like of floats
        - 2D: 2D array-like of floats of shape (m, 2)
        - 3D: 2D array-like of floats of shape (m, 3)

        note: if one point (m=1), a float in 1D, a 1D array of shape (2,) in 2D,
        a 1D array of shape (3,) in 3D, is accepted

    v : 1d-array-like of floats, optional
        data values at `x` (`v[i]` is the data value at `x[i]`)

    t : 1d-array-like of floats, or float, optional
        values of T considered as conditioning point for Y(T) (additional constraint)

    yt : 1d-array-like of floats, or float, optional
        value of Y at the conditioning point `t` (same length as `t`)

    vmin : float, optional
        minimal value for Z (or Y); simulation are rejected if not honoured

    vmax : float, optional
        maximal value for Z (or Y); simulation are rejected if not honoured

    algo_T : str
        defines the algorithm used for generating multi-Gaussian field T:

        - 'fft' or 'FFT' (default): based on circulant embedding and FFT, \
        function called for <d>D (d = 1, 2, or 3): `geone.grf.grf<d>D`
        - 'classic' or 'CLASSIC': classic algorithm, based on the resolution \
        of kriging system considered points in a search ellipsoid, function \
        called for <d>D (d = 1, 2, or 3): `geone.geoscalassicinterface.simulate<d>D`

    params_T : dict, optional
        keyword arguments (additional parameters) to be passed to the function
        corresponding to what is specified by the argument `algo_T` (see the
        corresponding function for its keyword arguments), in particular the key
        'mean' can be specified (set to value 0 if not specified)

    algo_Y : str
        defines the algorithm used for generating 1D multi-Gaussian field Y:

        - 'fft' or 'FFT' (default): based on circulant embedding and FFT, \
        function called: :func:`geone.grf.grf1D`
        - 'classic' or 'CLASSIC': classic algorithm, based on the resolution \
        of kriging system considered points in a search ellipsoid, function \
        called: :func:`geone.geoscalassicinterface.simulate`

    params_Y : dict, optional
        keyword arguments (additional parameters) to be passed to the function
        corresponding to what is specified by the argument `algo_Y` (see the
        corresponding function for its keyword arguments), in particular the key
        'mean' can be specified (if not specified, set to the mean value of `v`
        if `v` is not `None`, set to 0 otherwise)

    target_distrib : class
        target distribution for the value of a single realization of Z, with
        attributes:

        - target_distrib.cdf : (`func`) cdf
        - target_distrib.ppf : (`func`) inverse cdf

        See `initial_distrib` below.

    initial_distrib : class
        initial distribution for the value of a single realization of Z, with
        attributes:

        - initial_distrib.cdf : (`func`) cdf
        - initial_distrib.ppf : (`func`) inverse cdf

        The procedure is the following:

        1. conditioning data value `v` (if present) are transormed:
            * `v_tilde = initial_distrib.ppf(target_distrib.cdf(v))`
        2. SRF realization of `z_tilde` (conditionally to `v_tilde` if present) \
        is generated
        3. back-transform is applied to obtain the final realization:
            * `z = target_distrib.ppf(initial_distrib.cdf(z_tilde))`

        By default:

        - `target_distrib = None`
        - `initial_distrib = None`

        * For unconditional case:
            - if `target_distrib` is `None`: no transformation is applied
            - otherwise (not `None`): transformation is applied (step 3. above)
        * For conditional case:
            - if `target_distrib` is `None`: no transformation is applied
            - otherwise (not `None`): transformation is applied (steps 1 and 3. \
            above); this requires that `initial_distrib` is specified (not `None`), \
            or that `t` and `yt` are specified with the value "mean_T" given in `t`

        The distribution `initial_distrib` is used when needed:

        * as specified (if not `None`, be sure of what is given)
        * computed automatically otherwise (if `None`): \
        the distribution returned by the function \
        :func:`compute_distrib_Z_given_Y_of_mean_T`, with the keyword arguments \
        (only for unconditional case)
            - mean_Y  : set to `mean_Y` (see above)
            - cov_T_0 : set to the covariance of T evaluated at 0 (`cov_model_T.func()(0)[0]`)
            - y_mean_T: set to `yt[i0]`, where `t[i0]=mean_T` (if exists, see `t`, `yt` above) \
            or set to Y(mean(T)) computed after step 2 above (otherwise)

    mh_iter : int, default: 100
        number of iteration for Metropolis-Hasting algorithm, for conditional
        simulation only; note: used only if `x` or `t` is not `None`

    ntry_max : int, default: 1
        number of tries per realization before giving up if something goes wrong

    nreal : int, default: 1
        number of realization(s)

    full_output : bool, default: True
        - if `True`: simulation(s) of Z, T, and Y are retrieved in output
        - if `False`: simulation(s) of Z only is retrieved in output

    verbose : int, default: 1
        verbose mode, integer >=0, higher implies more display

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    Z : nd-array
        all realizations, `Z[k]` is the `k`-th realization:

        - for 1D: `Z` of shape (nreal, nx), where nx = dimension
        - for 2D: `Z` of shape (nreal, ny, nx), where nx, ny = dimension
        - for 3D: `Z` of shape (nreal, nz, ny, nx), where nx, ny, nz = dimension

    T : nd-array
        latent fields of all realizations, `T[k]` for the `k`-th realization:

        - for 1D: `T` of shape (nreal, nx), where nx = dimension
        - for 2D: `T` of shape (nreal, ny, nx), where nx, ny = dimension
        - for 3D: `T` of shape (nreal, nz, ny, nx), where nx, ny, nz = dimension

        returned if `full_output=True`

    Y : list of length nreal
        1D random fields of all realizations, `Y[k]` is a list of length 4 for
        the `k`-th realization:

        - Y[k][0]: int, Y_nt (number of cell along t-axis)
        - Y[k][1]: float, Y_st (cell size along t-axis)
        - Y[k][2]: float, Y_ot (origin)
        - Y[k][3]: 1d-array of shape (Y_nt,), values of Y[k]

        returned if `full_output=True`
    """
    fname = 'srf_mg_mg'

    if algo_T not in ('fft', 'FFT', 'classic', 'CLASSIC'):
        err_msg = f"{fname}: `algo_T` invalid, should be 'fft' (default) or 'classic'"
        if logger: logger.error(err_msg)
        raise SrfError(err_msg)

    if algo_Y not in ('fft', 'FFT', 'classic', 'CLASSIC'):
        err_msg = f"{fname}: `algo_Y` invalid, should be 'fft' (default) or 'classic'"
        if logger: logger.error(err_msg)
        raise SrfError(err_msg)

    # Set space dimension (of grid) according to covariance model for T
    if isinstance(cov_model_T, gcm.CovModel1D):
        d = 1
    elif isinstance(cov_model_T, gcm.CovModel2D):
        d = 2
    elif isinstance(cov_model_T, gcm.CovModel3D):
        d = 3
    else:
        err_msg = f'{fname}: `cov_model_T` invalid, should be a class `geone.covModel.CovModel1D`, `geone.covModel.CovModel2D` or `geone.covModel.CovModel3D`'
        if logger: logger.error(err_msg)
        raise SrfError(err_msg)

    # Check argument 'dimension'
    if hasattr(dimension, '__len__') and len(dimension) != d:
        err_msg = f'{fname}: `dimension` of incompatible length'
        if logger: logger.error(err_msg)
        raise SrfError(err_msg)

    if d == 1:
        grid_size = dimension
    else:
        grid_size = np.prod(dimension)

    # Check (or set) argument 'spacing'
    if spacing is None:
        if d == 1:
            spacing = 1.0
        else:
            spacing = tuple(np.ones(d))
    else:
        if hasattr(spacing, '__len__') and len(spacing) != d:
            err_msg = f'{fname}: `spacing` of incompatible length'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

    # Check (or set) argument 'origin'
    if origin is None:
        if d == 1:
            origin = 0.0
        else:
            origin = tuple(np.zeros(d))
    else:
        if hasattr(origin, '__len__') and len(origin) != d:
            err_msg = f'{fname}: `origin` of incompatible length'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

    # if not cov_model_T.is_stationary(): # prevent calculation if covariance model is not stationary
    #     if verbose > 0:
    #         print(f'ERROR ({fname}): `cov_model_T` is not stationary')

    # Check covariance model for Y
    if not isinstance(cov_model_Y, gcm.CovModel1D):
        err_msg = f'{fname}: `cov_model_Y` invalid'
        if logger: logger.error(err_msg)
        raise SrfError(err_msg)

    # elif not cov_model_Y.is_stationary(): # prevent calculation if covariance model is not stationary
    #     err_msg = f'{fname}: `cov_model_Y` is not stationary'
    #     if logger: logger.error(err_msg)
    #     raise SrfError(err_msg)

    # Check additional constraint t (conditioning point for T), yt (corresponding value for Y)
    if t is None:
        if yt is not None:
            err_msg = f'{fname}: `t` is not given (`None`) but `yt` is given (not `None`)'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

    else:
        if yt is None:
            err_msg = f'{fname}: `t` is given (not `None`) but `yt` is not given (`None`)'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

        t = np.asarray(t, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        yt = np.asarray(yt, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(yt) != len(t):
            err_msg = f'{fname}: length of `yt` is not valid'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

    # Initialize dictionary params_T
    if params_T is None:
        params_T = {}

    # Compute meshgrid over simulation domain if needed (see below)
    if ('mean' in params_T.keys() and callable(params_T['mean'])) or ('var' in params_T.keys() and callable(params_T['var'])):
        if d == 1:
            xi = origin + spacing*(0.5+np.arange(dimension)) # x-coordinate of cell center
        elif d == 2:
            xi = origin[0] + spacing[0]*(0.5+np.arange(dimension[0])) # x-coordinate of cell center
            yi = origin[1] + spacing[1]*(0.5+np.arange(dimension[1])) # y-coordinate of cell center
            yyi, xxi = np.meshgrid(yi, xi, indexing='ij')
        elif d == 3:
            xi = origin[0] + spacing[0]*(0.5+np.arange(dimension[0])) # x-coordinate of cell center
            yi = origin[1] + spacing[1]*(0.5+np.arange(dimension[1])) # y-coordinate of cell center
            zi = origin[2] + spacing[2]*(0.5+np.arange(dimension[2])) # z-coordinate of cell center
            zzi, yyi, xxi = np.meshgrid(zi, yi, xi, indexing='ij')

    # Set mean_T (as array) from params_T
    if 'mean' not in params_T.keys():
        mean_T = np.array([0.0])
    else:
        mean_T = params_T['mean']
        if mean_T is None:
            mean_T = np.array([0.0])
        elif callable(mean_T):
            if d == 1:
                mean_T = mean_T(xi).reshape(-1) # replace function 'mean_T' by its evaluation on the grid
            elif d == 2:
                mean_T = mean_T(xxi, yyi).reshape(-1) # replace function 'mean_T' by its evaluation on the grid
            elif d == 3:
                mean_T = mean_T(xxi, yyi, zzi).reshape(-1) # replace function 'mean_T' by its evaluation on the grid
        else:
            mean_T = np.asarray(mean_T).reshape(-1)
            if mean_T.size not in (1, grid_size):
                err_msg = f"{fname}: 'mean' parameter for T (in `params_T`) has incompatible size"
                if logger: logger.error(err_msg)
                raise SrfError(err_msg)

    # Set var_T (as array) from params_T, if given
    var_T = None
    if 'var' in params_T.keys():
        var_T = params_T['var']
        if var_T is not None:
            if callable(var_T):
                if d == 1:
                    var_T = var_T(xi).reshape(-1) # replace function 'var_T' by its evaluation on the grid
                elif d == 2:
                    var_T = var_T(xxi, yyi).reshape(-1) # replace function 'var_T' by its evaluation on the grid
                elif d == 3:
                    var_T = var_T(xxi, yyi, zzi).reshape(-1) # replace function 'var_T' by its evaluation on the grid
            else:
                var_T = np.asarray(var_T).reshape(-1)
                if var_T.size not in (1, grid_size):
                    err_msg = f"{fname}: 'var' parameter for T (in `params_T`) has incompatible size"
                    if logger: logger.error(err_msg)
                    raise SrfError(err_msg)

    # Initialize dictionary params_Y
    if params_Y is None:
        params_Y = {}

    # Set mean_Y from params_Y (if given, and check if it is a unique value)
    mean_Y = None
    if 'mean' in params_Y.keys():
        mean_Y = params_Y['mean']
        if callable(mean_Y):
            err_msg = f"{fname}: 'mean' parameter for Y (in `params_Y`) must be a unique value (float) if given"
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

        else:
            mean_Y = np.asarray(mean_Y, dtype='float').reshape(-1)
            if mean_Y.size != 1:
                err_msg = f"{fname}: 'mean' parameter for Y (in `params_Y`) must be a unique value (float) if given"
                if logger: logger.error(err_msg)
                raise SrfError(err_msg)

            mean_Y = mean_Y[0]

    # Check var_Y from params_Y
    if 'var' in params_Y.keys() and params_Y['var'] is not None:
        err_msg = f"{fname}: 'var' parameter for Y (in `params_Y`) must be `None`"
        if logger: logger.error(err_msg)
        raise SrfError(err_msg)

    # Check input for distribution transform
    if target_distrib is None:
        if initial_distrib is not None and verbose > 0:
            if logger:
                logger.warning(f'{fname}: target distribution not handled (`initial_distrib` ignored) because `target_distrib` is not given (`None`)')
            else:
                print(f'{fname}: WARNING: target distribution not handled (`initial_distrib` ignored) because `target_distrib` is not given (`None`)')
    else:
        if mean_T.size != 1:
            err_msg = f'{fname}: target distribution cannot be handled with non-stationary mean for T (in `params_T`)'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

        if x is not None:
            if initial_distrib is None:
                if 'mean' not in params_Y.keys():
                    err_msg = f"{fname}: target distribution cannot be handled (cannot set `initial_distrib`: 'mean' for Y must be specified (in `params_Y`)"
                    if logger: logger.error(err_msg)
                    raise SrfError(err_msg)

                else:
                    if t is not None:
                        ind = np.where(t==mean_T[0])[0]
                    else:
                        ind = []
                    if len(ind) == 0:
                        err_msg = f'{fname}: target distribution cannot be handled (cannot set `initial_distrib`: value of mean(T) should be specified in `t`)'
                        if logger: logger.error(err_msg)
                        raise SrfError(err_msg)

    # Number of realization(s)
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if full_output:
            if verbose > 0:
                if logger:
                    logger.warning(f'{fname}: `nreal` <= 0: `None`, `None`, `None` is returned')
                else:
                    print(f'{fname}: WARNING: `nreal` <= 0: `None`, `None`, `None` is returned')
            return None, None, None
        else:
            if verbose > 0:
                if logger:
                    logger.warning(f'{fname}: `nreal` <= 0: `None` is returned')
                else:
                    print(f'{fname}: WARNING: `nreal` <= 0: `None` is returned')
            return None

    # Note: format of data (x, v) not checked !

    if x is None:
        if v is not None:
            err_msg = f'{fname}: `x` is not given (`None`) but `v` is given (not `None`)'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

        # Preparation for unconditional case
        # Set mean_Y
        if mean_Y is None:
            mean_Y = 0.0
        #
        # Preparation for distribution transform
        if target_distrib is None:
            # no distribution transform
            distrib_transf = 0
        else:
            distrib_transf = 1
            if initial_distrib is None:
                if t is not None:
                    ind = np.where(t==mean_T[0])[0]
                else:
                    ind = []
                if len(ind):
                    y_mean_T = yt[ind[0]]
                    cov_T_0 = cov_model_T.func()(np.zeros(d))[0]
                    cov_Y_0 = cov_model_Y.func()(0.)[0]
                    std_Y_0 = np.sqrt(cov_Y_0)
                    initial_distrib = compute_distrib_Z_given_Y_of_mean_T(
                        np.linspace(min(y_mean_T, mean_Y)-5.*std_Y_0, max(y_mean_T, mean_Y)+5.*std_Y_0, 501),
                        cov_model_Y, mean_Y=mean_Y, y_mean_T=y_mean_T, cov_T_0=cov_T_0,
                        fstd=4.5, nint=2001, assume_sorted=True
                        )
                    compute_initial_distrib = False
                else:
                    # initial_distrib will be computed for each realization
                    cov_T_0 = cov_model_T.func()(np.zeros(d))[0]
                    cov_Y_0 = cov_model_Y.func()(0.)[0]
                    std_Y_0 = np.sqrt(cov_Y_0)
                    compute_initial_distrib = True
            else:
                compute_initial_distrib = False
    #
    else:
        # Preparation for conditional case
        if v is None:
            err_msg = f'{fname}: `x` is given (not `None`) but `v` is not given (`None`)'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

        x = np.asarray(x, dtype='float').reshape(-1, d) # cast in d-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

        # Number of conditioning points
        npt = x.shape[0]

        # Get index in mean_T for each conditioning points
        x_mean_T_grid_ind = None
        if mean_T.size == 1:
            x_mean_T_grid_ind = np.zeros(npt, dtype='int')
        else:
            indc_f = (x-origin)/spacing
            indc = indc_f.astype(int)
            indc = indc - 1 * np.all((indc == indc_f, indc > 0), axis=0)
            if d == 1:
                x_mean_T_grid_ind = 1 * indc[:, 0] # multiply by 1.0 makes a copy of the array !
            elif d == 2:
                x_mean_T_grid_ind = indc[:, 0] + dimension[0] * indc[:, 1]
            elif d == 3:
                x_mean_T_grid_ind = indc[:, 0] + dimension[0] * (indc[:, 1] + dimension[1] * indc[:, 2])

        # Get index in var_T (if not None) for each conditioning points
        if var_T is not None:
            if var_T.size == 1:
                x_var_T_grid_ind = np.zeros(npt, dtype='int')
            else:
                if x_mean_T_grid_ind is not None:
                    x_var_T_grid_ind = x_mean_T_grid_ind
                else:
                    indc_f = (x-origin)/spacing
                    indc = indc_f.astype(int)
                    indc = indc - 1 * np.all((indc == indc_f, indc > 0), axis=0)
                    if d == 1:
                        x_var_T_grid_ind = 1 * indc[:, 0] # multiply by 1.0 makes a copy of the array !
                    elif d == 2:
                        x_var_T_grid_ind = indc[:, 0] + dimension[0] * indc[:, 1]
                    elif d == 3:
                        x_var_T_grid_ind = indc[:, 0] + dimension[0] * (indc[:, 1] + dimension[1] * indc[:, 2])

        # Get covariance function for T and Y
        cov_func_T = cov_model_T.func() # covariance function
        cov_func_Y = cov_model_Y.func() # covariance function

        # Get evaluation of covariance function for T and Y at 0
        cov0_T = cov_func_T(np.zeros(d))
        cov0_Y = cov_func_Y(np.zeros(1))

        # Set mean_Y
        if mean_Y is None:
            mean_Y = np.mean(v)

        # Preparation for distribution transform
        if target_distrib is None:
            # no distribution transform
            distrib_transf = 0
        else:
            distrib_transf = 1
            if initial_distrib is None:
                if t is not None:
                    ind = np.where(t==mean_T[0])[0]
                else:
                    ind = []
                if len(ind):
                    y_mean_T = yt[ind[0]]
                    cov_T_0 = cov0_T[0]
                    cov_Y_0 = cov0_Y[0]
                    std_Y_0 = np.sqrt(cov_Y_0)
                    initial_distrib = compute_distrib_Z_given_Y_of_mean_T(
                        np.linspace(min(y_mean_T, mean_Y)-5.*std_Y_0, max(y_mean_T, mean_Y)+5.*std_Y_0, 501),
                        cov_model_Y, mean_Y=mean_Y, y_mean_T=y_mean_T, cov_T_0=cov_T_0,
                        fstd=4.5, nint=2001, assume_sorted=True
                        )
                else:
                    distrib_transf = 0

        if distrib_transf:
            # Transform the conditioning data value
            v = initial_distrib.ppf(target_distrib.cdf(v))

        # Set kriging matrix for T (mat_T) of order npt, "over every conditioining point"
        mat_T = np.ones((npt, npt))
        for i in range(npt-1):
            # lag between x[i] and x[j], j=i+1, ..., npt-1
            h = x[(i+1):] - x[i]
            cov_h_T = cov_func_T(h)
            mat_T[i, (i+1):npt] = cov_h_T
            mat_T[(i+1):npt, i] = cov_h_T
            mat_T[i, i] = cov0_T

        mat_T[-1,-1] = cov0_T

        if var_T is not None:
            varUpdate = np.sqrt(var_T[x_var_T_grid_ind]/cov0_T)
            mat_T = varUpdate*(mat_T.T*varUpdate).T

        # Initialize
        #   - npt_ext: number of total conditioning point for Y, "point T(x) + additional constraint t"
        #   - v_T: values of T(x) (that are defined later) followed by values yt at additional constraint t"
        #   - v_ext: values for Y at "point T(x) + additional constraint (t)"
        #   - mat_Y: kriging matrix for Y of order npt_ext, over "point T(x) + additional constraint t"
        if t is None:
            npt_ext = npt
            v_T = np.zeros(npt)
            v_ext = v
            mat_Y = np.ones((npt_ext, npt_ext))
        else:
            npt_ext = npt + len(t)
            v_T = np.hstack((np.zeros(npt), t))
            v_ext = np.hstack((v, yt))
            mat_Y = np.ones((npt_ext, npt_ext))
            for i in range(len(t)-1):
                # lag between t[i] and t[j], j=i+1, ..., len(t)-1
                h = t[(i+1):] - t[i]
                cov_h_Y = cov_func_Y(h)
                k = i + npt
                mat_Y[k, (k+1):] = cov_h_Y
                mat_Y[(k+1):, k] = cov_h_Y
                #mat_Y[k, k] = cov0_Y

            #mat_Y[-1,-1] = cov0_Y
        for i in range(npt_ext):
            mat_Y[i, i] = cov0_Y

        if npt_ext <= 1:
            mh_iter = 0 # unnecessary to apply Metropolis update !

    # Set (again if given) default parameter 'mean' and 'var' for T, and 'mean' for Y
    params_T['mean'] = mean_T
    params_T['var'] = var_T
    params_Y['mean'] = mean_Y

    # Set default parameter 'verbose' for params_T and params_Y
    if 'verbose' not in params_T.keys():
        params_T['verbose'] = 0
        # params_T['verbose'] = verbose
    if 'verbose' not in params_Y.keys():
        params_Y['verbose'] = 0
        # params_Y['verbose'] = verbose

    # Initialization for output
    Z = []
    if full_output:
        T = []
        Y = []

    for ireal in range(nreal):
        # Generate ireal-th realization
        if verbose > 1:
            if logger:
                logger.info(f'{fname}: simulation {ireal+1} of {nreal}...')
            else:
                print(f'{fname}: simulation {ireal+1} of {nreal}...')
        for ntry in range(ntry_max):
            sim_ok = True
            Y_cond_aggregation = False
            if verbose > 2 and ntry > 0:
                if logger:
                    logger.info(f'{fname}:   ... new trial ({ntry+1} of {ntry_max}) for simulation {ireal+1} of {nreal}...')
                else:
                    print(f'{fname}:   ... new trial ({ntry+1} of {ntry_max}) for simulation {ireal+1} of {nreal}...')
            if x is None:
                # Unconditional case
                # ------------------
                # Generate T (one real)
                try:
                    sim_T = multiGaussian.multiGaussianRun(
                            cov_model_T, dimension, spacing, origin,
                            mode='simulation', algo=algo_T, output_mode='array',
                            **params_T, nreal=1, logger=logger)
                except:
                    sim_ok = False
                    if verbose > 2:
                        if logger:
                            logger.info(f'{fname}:   ... simulation of T failed')
                        else:
                            print(f'{fname}:   ... simulation of T failed')
                    continue
                # except Exception as exc:
                #     err_msg = f'{fname}: simulation of T failed'
                #     if logger: logger.error(err_msg)
                #     raise SrfError(err_msg) from exc

                # -> nd-array of shape
                #      (1, dimension) (for T in 1D)
                #      (1, dimension[1], dimension[0]) (for T in 2D)
                #      (1, dimension[2], dimension[1], dimension[0]) (for T in 3D)

                # Set origin and dimension for Y
                min_T = np.min(sim_T)
                max_T = np.max(sim_T)
                if t is not None:
                    min_T = min(t.min(), min_T)
                    max_T = max(t.max(), max_T)
                min_T = min_T - 0.5 * spacing_Y
                max_T = max_T + 0.5 * spacing_Y
                dimension_Y = int(np.ceil((max_T - min_T)/spacing_Y))
                origin_Y = min_T - 0.5*(dimension_Y*spacing_Y - (max_T - min_T))

                # Generate Y conditional to possible additional constraint (t, yt) (one real)
                try:
                    sim_Y = multiGaussian.multiGaussianRun(
                            cov_model_Y, dimension_Y, spacing_Y, origin_Y, x=t, v=yt,
                            mode='simulation', algo=algo_Y, output_mode='array',
                            **params_Y, nreal=1, logger=logger)
                except:
                    sim_ok = False
                    if verbose > 2:
                        if logger:
                            logger.info(f'{fname}:   ... simulation of Y failed')
                        else:
                            print(f'{fname}:   ... simulation of Y failed')
                    continue
                # except Exception as exc:
                #     err_msg = f'{fname}: simulation of Y failed'
                #     if logger: logger.error(err_msg)
                #     raise SrfError(err_msg) from exc

                # -> 2d-array of shape (1, dimension_Y)

                if distrib_transf:
                    if compute_initial_distrib:
                        # Compute initial_distrib
                        # (approximately based on mean(T) and y_mean_T)
                        # print('... computing initial_distrib ...')
                        sim_T_mean = sim_T.reshape(-1).mean()
                        # Compute
                        #    ind: node index (nearest node),
                        #         rounded to lower index if between two grid nodes and index is positive
                        ind_f = (sim_T_mean - origin_Y)/spacing_Y
                        ind = ind_f.astype(int)
                        ind = ind - 1 * np.all((ind == ind_f, ind > 0), axis=0)
                        y_mean_T = sim_Y[0][ind]
                        #y_mean_T = sim_Y[0][np.floor((sim_T_mean - origin_Y)/spacing_Y).astype(int)]
                        initial_distrib = compute_distrib_Z_given_Y_of_mean_T(
                            np.linspace(min(y_mean_T, mean_Y)-5.*std_Y_0, max(y_mean_T, mean_Y)+5.*std_Y_0, 501),
                            cov_model_Y, mean_Y=mean_Y, y_mean_T=y_mean_T, cov_T_0=cov_T_0,
                            fstd=4.5, nint=2001, assume_sorted=True
                            )
                    #
                    # (Back-)transform sim_Y value
                    sim_Y = target_distrib.ppf(initial_distrib.cdf(sim_Y))
            #
            else:
                # Conditional case
                # ----------------
                # Initialize: unconditional simulation of T at x (values in v_T)
                ind = np.random.permutation(npt)
                for j, k in enumerate(ind):
                    # Simulate value at x[k] (= x[ind[j]]), conditionally to the previous ones
                    # Solve the kriging system (for T)
                    try:
                        w = np.linalg.solve(
                                mat_T[ind[:j], :][:, ind[:j]], # kriging matrix
                                mat_T[ind[:j], ind[j]], # second member
                            )
                    except:
                        sim_ok = False
                        break

                    # Mean (kriged) value at x[k]
                    mu_T_k = mean_T[x_mean_T_grid_ind[k]] + (v_T[ind[:j]] - mean_T[x_mean_T_grid_ind[ind[:j]]]).dot(w)
                    # Standard deviation (of kriging) at x[k]
                    std_T_k = np.sqrt(np.maximum(0, cov0_T - np.dot(w, mat_T[ind[:j], ind[j]])))
                    # Draw value in N(mu_T_k, std_T_k^2)
                    v_T[k] = np.random.normal(loc=mu_T_k, scale=std_T_k)

                if not sim_ok:
                    sim_ok = False
                    if verbose > 2:
                        if logger:
                            logger.info(f'{fname}:    ... cannot solve kriging system (for T, initialization)')
                        else:
                            print(f'{fname}:    ... cannot solve kriging system (for T, initialization)')
                    continue

                # Updated kriging matrix for Y (mat_Y) according to value in v_T[0:npt]
                for i in range(npt-1):
                    # lag between v_T[i] and v_T[j], j=i+1, ..., npt-1
                    h = v_T[(i+1):npt] - v_T[i]
                    cov_h_Y = cov_func_Y(h)
                    mat_Y[i, (i+1):npt] = cov_h_Y
                    mat_Y[(i+1):npt, i] = cov_h_Y
                    # mat_Y[i, i] = cov0_Y

                for i, k in enumerate(range(npt, npt_ext)):
                    # lag between t[i] and v_T[j], j=0, ..., npt-1
                    h = v_T[0:npt] - t[i]
                    cov_h_Y = cov_func_Y(h)
                    mat_Y[k, 0:npt] = cov_h_Y
                    mat_Y[0:npt, k] = cov_h_Y
                    # mat_Y[i, i] = cov0_Y

                # mat_Y[-1,-1] = cov0_Y

                # Update simulated values v_T at x using Metropolis-Hasting (MH) algorithm
                for nit in range(mh_iter):
                    if verbose > 3:
                        if logger:
                            logger.info(f'{fname}:   ... sim {ireal+1} of {nreal}: MH iter {nit+1} of {mh_iter}...')
                        else:
                            print(f'{fname}:   ... sim {ireal+1} of {nreal}: MH iter {nit+1} of {mh_iter}...')
                    ind = np.random.permutation(npt)
                    for k in ind:
                        # Sequence of indexes without k
                        indmat = np.hstack((np.arange(k), np.arange(k+1, npt)))
                        # Simulate possible new value v_T_new at x[k], conditionally to all the ohter ones
                        #
                        # Solve the kriging system for T
                        try:
                            w = np.linalg.solve(
                                    mat_T[indmat, :][:, indmat], # kriging matrix
                                    mat_T[indmat, k], # second member
                                )
                        except:
                            sim_ok = False
                            if verbose > 2:
                                if logger:
                                    logger.info(f'{fname}:   ... cannot solve kriging system (for T)')
                                else:
                                    print(f'{fname}:   ... cannot solve kriging system (for T)')
                            break
                        #
                        # Mean (kriged) value at x[k]
                        mu_T_k = mean_T[x_mean_T_grid_ind[k]] + (v_T[indmat] - mean_T[x_mean_T_grid_ind[indmat]]).dot(w)
                        # Standard deviation (of kriging) at x[k]
                        std_T_k = np.sqrt(np.maximum(0, cov0_T - np.dot(w, mat_T[indmat, k])))
                        # Draw value in N(mu, std^2)
                        v_T_k_new = np.random.normal(loc=mu_T_k, scale=std_T_k)
                        #
                        # Compute MH quotient defined as
                        #    prob(Y[v_T_k_new] = v[k] | Y[indmat] = v[indmat], Y[t] = yt) / prob(Y[v_T[k]] = v[k] | Y[indmat] = v[indmat], Y[t] = yt)
                        # (where Y[t]=yt are the possible additional constraint)
                        #
                        # New lag from v_T_k_new and corresponding covariance for Y
                        h_k_new = v_T_k_new - np.hstack((v_T[:k], v_T_k_new, v_T[k+1:]))
                        cov_h_Y_k_new = cov_func_Y(h_k_new)
                        # Solve the kriging system for Y for simulation at v_T[k] and at v_T_k_new
                        indmat_ext = np.hstack((indmat, np.arange(npt, npt_ext)))
                        try:
                            w = np.linalg.solve(
                                    mat_Y[indmat_ext, :][:, indmat_ext], # kriging matrix
                                    np.vstack((mat_Y[indmat_ext, k], cov_h_Y_k_new[indmat_ext])).T # both second members
                                )
                        except:
                            sim_ok = False
                            if verbose > 2:
                                if logger:
                                    logger.info(f'{fname}:   ... cannot solve kriging system (for Y)')
                                else:
                                    print(f'{fname}:   ... cannot solve kriging system (for Y)')
                            break
                        # Mean (kriged) values at v_T[k] and v_T_k_new
                        mu_Y_k = mean_Y + (v_ext[indmat_ext] - mean_Y).dot(w) # mu_k of shape(2, )
                        # Variance (of kriging) at v_T[k] and v_T_k_new
                        var_Y_k = np.maximum(1.e-20, cov0_Y - np.array([np.dot(w[:,0], mat_Y[indmat_ext, k]), np.dot(w[:,1], cov_h_Y_k_new[indmat_ext])]))
                        # Set minimal variance to 1.e-20 to avoid division by zero
                        #
                        # MH quotient is
                        #    phi_{mean=mu_Y_k[1], var=var_Y_k[1]}(v[k]) / phi_{mean=mu_Y_k[0], var=var_Y_k[0]}(v[k])
                        # where phi_{mean, var} is the pdf of the normal law of given mean and var
                        # To avoid overflow in exp, compute log of mh quotient...
                        log_mh_quotient = 0.5 * (np.log(var_Y_k[0]) + (v[k]-mu_Y_k[0])**2/var_Y_k[0] - np.log(var_Y_k[1]) - (v[k]-mu_Y_k[1])**2/var_Y_k[1])
                        if log_mh_quotient >= 0.0 or np.random.random() < np.exp(log_mh_quotient):
                            # Accept new value v_T_new at x[k]
                            v_T[k] = v_T_k_new
                            # Update kriging matrix for Y
                            mat_Y[k,:] = cov_h_Y_k_new
                            mat_Y[:,k] = cov_h_Y_k_new
                    if not sim_ok:
                        break

                if not sim_ok:
                    continue

                # Generate T conditional to (x, v_T[0:npt]) (one real)
                try:
                    sim_T = multiGaussian.multiGaussianRun(
                            cov_model_T, dimension, spacing, origin, x=x, v=v_T[:npt],
                            mode='simulation', algo=algo_T, output_mode='array',
                            **params_T, nreal=1, logger=logger)
                except:
                    sim_ok = False
                    if verbose > 2:
                        if logger:
                            logger.info(f'{fname}:   ... conditional simulation of T failed')
                        else:
                            print(f'{fname}:   ... conditional simulation of T failed')
                    continue
                # except Exception as exc:
                #     err_msg = f'{fname}: conditional simulation of T failed'
                #     if logger: logger.error(err_msg)
                #     raise SrfError(err_msg) from exc

                # -> nd-array of shape
                #      (1, dimension) (for T in 1D)
                #      (1, dimension[1], dimension[0]) (for T in 2D)
                #      (1, dimension[2], dimension[1], dimension[0]) (for T in 3D)

                # Set origin and dimension for Y
                min_T = np.min(sim_T)
                max_T = np.max(sim_T)
                if t is not None:
                    min_T = min(t.min(), min_T)
                    max_T = max(t.max(), max_T)
                min_T = min_T - 0.5 * spacing_Y
                max_T = max_T + 0.5 * spacing_Y
                dimension_Y = int(np.ceil((max_T - min_T)/spacing_Y))
                origin_Y = min_T - 0.5*(dimension_Y*spacing_Y - (max_T - min_T))

                # Compute
                #    indc: node index of conditioning node (nearest node),
                #          rounded to lower index if between two grid node and index is positive
                indc_f = (v_T-origin_Y)/spacing_Y
                indc = indc_f.astype(int)
                indc = indc - 1 * np.all((indc == indc_f, indc > 0), axis=0)
                indc_unique, indc_inv = np.unique(indc, return_inverse=True)
                if len(indc_unique) == len(indc):
                    v_T_unique = v_T
                    v_ext_unique = v_ext
                else:
                    Y_cond_aggregation = True
                    v_T_unique = np.array([v_T[indc_inv==j].mean() for j in range(len(indc_unique))])
                    v_ext_unique = np.array([v_ext[indc_inv==j].mean() for j in range(len(indc_unique))])

                # Generate Y conditional to (v_T, v_ext) (one real)
                try:
                    sim_Y = multiGaussian.multiGaussianRun(
                            cov_model_Y, dimension_Y, spacing_Y, origin_Y, x=v_T_unique, v=v_ext_unique,
                            mode='simulation', algo=algo_Y, output_mode='array',
                            **params_Y, nreal=1, logger=logger)
                except:
                    sim_ok = False
                    if verbose > 2:
                        if logger:
                            logger.info(f'{fname}:   ... conditional simulation of Y failed')
                        else:
                            print(f'{fname}:   ... conditional simulation of Y failed')
                    continue
                # except Exception as exc:
                #     err_msg = f'{fname}: conditional simulation of Y failed'
                #     if logger: logger.error(err_msg)
                #     raise SrfError(err_msg) from exc

                # -> 2d-array of shape (1, dimension_Y)

                if distrib_transf:
                    # Back-transform sim_Y value
                    sim_Y = target_distrib.ppf(initial_distrib.cdf(sim_Y))

            # Generate Z (one real)
            # Compute
            #    ind: node index (nearest node),
            #         rounded to lower index if between two grid nodes and index is positive
            ind_f = (sim_T.reshape(-1) - origin_Y)/spacing_Y
            ind = ind_f.astype(int)
            ind = ind - 1 * np.all((ind == ind_f, ind > 0), axis=0)
            Z_real = sim_Y[0][ind]
            #Z_real = sim_Y[0][np.floor((sim_T.reshape(-1) - origin_Y)/spacing_Y).astype(int)]
            if vmin is not None and Z_real.min() < vmin:
                sim_ok = False
                if verbose > 2:
                    if logger:
                        logger.info(f'{fname}:   ... specified minimal value not honoured')
                    else:
                        print(f'{fname}:   ... specified minimal value not honoured')
                continue
            if vmax is not None and Z_real.max() > vmax:
                sim_ok = False
                if verbose > 2:
                    if logger:
                        logger.info(f'{fname}:   ... specified maximal value not honoured')
                    else:
                        print(f'{fname}:   ... specified maximal value not honoured')
                continue

            if sim_ok:
                if Y_cond_aggregation and verbose > 0:
                    if logger:
                        logger.warning(f'{fname}: conditioning points for Y falling in a same grid cell have been aggregated (mean) (real index {ireal})')
                    else:
                        print(f'{fname}: WARNING: conditioning points for Y falling in a same grid cell have been aggregated (mean) (real index {ireal})')
                Z.append(Z_real)
                if full_output:
                    T.append(sim_T[0])
                    Y.append([dimension_Y, spacing_Y, origin_Y, sim_Y.reshape(dimension_Y)])
                break

    # Get Z
    if verbose > 0 and len(Z) < nreal:
        if logger:
            logger.warning(f'{fname}: some realization failed (missing)')
        else:
            print(f'{fname}: WARNING: some realization failed (missing)')

    Z = np.asarray(Z).reshape(len(Z), *np.atleast_1d(dimension)[::-1])

    if full_output:
        T = np.asarray(T).reshape(len(T), *np.atleast_1d(dimension)[::-1])
        return Z, T, Y
    else:
        return Z
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
class Distrib (object):
    """
    Class defining a distribution by a pdf, cdf, and ppf.
    """
    def __init__(self, pdf=None, cdf=None, ppf=None):
        self.pdf = pdf
        self.cdf = cdf
        self.ppf = ppf
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def compute_distrib_Z_given_Y_of_mean_T(
        z, cov_model_Y,
        mean_Y=0.,
        y_mean_T=0.,
        cov_T_0=1.0,
        fstd=4.5,
        nint=2001,
        assume_sorted=False):
    """
    Computes the distribution of Z given Y(mean(T)), for a SRF Z = Y(T).

    With a SRF Z = Y(T), compute the pdf, cdf and ppf (inverse cdf) of
    Z given Y(mean(T))=y_mean_T (applicable for a (large) ensemble of realizations).

    The cdf is given by the equation (26) in the reference below. This equation
    requires expectations wrt. :math:`\\mathcal{N}(0, c\\_T\\_0)`, which are approximated
    using `nint` values in the interval :math:`\\pm fstd \\cdot \\sqrt{c\\_T\\_0}`.

    Parameters
    ----------
    z : 1d-array of floats
        values at which the conditional cdf and pdf are evaluated before interpolation, e.g
        `numpy.linspace(z_min, z_max, n)` with given `z_min`, `z_max`, and `n`

    cov_model_Y : :class:`geone.covModel.CovModel1D`
        covariance model for Y (coding process), in 1D

    mean_Y : float, default: 0.0
        mean of Y

    y_mean_T : float, default: 0.0
        imposed value for Y(mean(T))

    cov_T_0 : float, default: 1.0
        covariance model of T (latent field, directing function) evaluated at 0

    fstd : float, default: 4.5
        positive value used for computing approximation (see above)

    nint : int, defualt: 2001
        positive integer used for computing approximation (see above)

    assume_sorted : bool, default: False
        if `True`: `z` has to be an array of monotonically increasing values

    Returns
    -------
    distrib : :class:`Distrib`
        distribution, where each attribute is a function (obtained by
        interpolation of its approximated evaluation at `z`):

        - distrib.pdf: (func) pdf f_{Z|Y(mean(T))=y_mean_T}
        - distrib.cdf: (func) cdf F_{Z|Y(mean(T))=y_mean_T}
        - distrib.ppf: (func) inverse cdf

    References
    ----------
    - J. Straubhaar, P. Renard (2024), \
    Exploring substitution random functions composed of stationary multi-Gaussian processes. \
    Stochastic Environmental Research and Risk Assessment, \
    `doi:10.1007/s00477-024-02662-x <https://doi.org/10.1007/s00477-024-02662-x>`_
    """
    # The cdf is given by (eq. 26 of the ref):
    #     F_{Z|Y(mean(T))=y_mean_T}(z)
    #         = P(Z < z | Y(mean(T)) = y_mean_T)
    #         = E_{h~N(0, c_T(0))} [F(z)], F cdf of N(mean_Y + C_Y(h)/C_Y(0)*(y_mean_T-mean_Y), C_Y(0) - C_Y(h)**2/C_Y(0))
    # the pdf is given by:
    #     f_{Z|Y(mean(T))=y_mean_T}(z)
    #         = d/dz[P(Z < z | Y(mean(T)) = y_mean_T)]
    #         = E_{h~N(0, c_T(0))} [f(z)], f cdf of N(mean_Y + C_Y(h)/C_Y(0)*(y_mean_T-mean_Y), C_Y(0) - C_Y(h)**2/C_Y(0))
    # where
    #     C_Y: covariance function of the coding process Y
    #     C_T: covariance function of the directing function T
    # The function F_{Z|Y(mean(T))=y_mean_T} and f_{Z|Y(mean(T))=y_mean_T} are set by interpolation of the evaluation at z
    # (can be np.linspace(z_min, z_max, n), i.e. at z_min + i * (z_max-z_min)/(n-1), i=0, ... n-1)

    # Approximation is computed, using 'nint' values of h in the interval
    # +/-'fstd'*np.sqrt(2.0*c_T(0)) for the mean wrt. N(0, c_T(0))

    # fname = 'compute_distrib_Z_given_Y_of_mean_T'

    std_T_0 = np.sqrt(cov_T_0)
    a = fstd*std_T_0
    h = np.linspace(-a, a, nint)
    h_weight = np.exp(-0.5*h**2/cov_T_0)
    h_weight = h_weight / h_weight.sum()

    z = np.asarray(z, dtype='float').reshape(-1)

    cov_Y_h = cov_model_Y.func()(h)
    cov_Y_0 = cov_model_Y.func()(0.)
    pdf_value = np.array([np.sum(h_weight*stats.norm.pdf(zi, loc=mean_Y + cov_Y_h/cov_Y_0 * (y_mean_T - mean_Y), scale=np.maximum(np.sqrt(cov_Y_0-cov_Y_h**2/cov_Y_0), 1.e-20))) for zi in z])
    cdf_value = np.array([np.sum(h_weight*stats.norm.cdf(zi, loc=mean_Y + cov_Y_h/cov_Y_0 * (y_mean_T - mean_Y), scale=np.maximum(np.sqrt(cov_Y_0-cov_Y_h**2/cov_Y_0), 1.e-20))) for zi in z])

    pdf = interp1d(z, pdf_value, assume_sorted=assume_sorted, bounds_error=False, fill_value=(0., 0.))
    cdf = interp1d(z, cdf_value, assume_sorted=assume_sorted, bounds_error=False, fill_value=(0., 1.))
    ppf = interp1d(cdf_value, z, assume_sorted=assume_sorted, bounds_error=False, fill_value=(z.min(), z.max()))

    distrib = Distrib(pdf, cdf, ppf)
    return distrib
# ----------------------------------------------------------------------------

# ============================================================================
# Tools for simulating continuous SRF with
#     - two multi-Gaussian simulation as directing function
#     - 2D multi-Gaussian simulation as coding process
# ============================================================================

# ----------------------------------------------------------------------------
def srf_bimg_mg(
        cov_model_T1, cov_model_T2, cov_model_Y,
        dimension, spacing=None, origin=None,
        spacing_Y=(0.001, 0.001),
        x=None, v=None,
        t=None, yt=None,
        vmin=None, vmax=None,
        algo_T1='fft', params_T1=None,
        algo_T2='fft', params_T2=None,
        algo_Y='fft', params_Y=None,
        mh_iter=100,
        ntry_max=1,
        nreal=1,
        full_output=True,
        verbose=1,
        logger=None):
    """
    Substitution Random Function (SRF) - multi-Gaussian + multi-Gaussian.

    This function allows to generate continuous random fields in 1D, 2D, 3D, based on
    a SRF Z defined as

    - Z(x) = Y(T1(x), T2(x))

    where

    - T1, T1 are the directing functions (independent), two multi-Gaussian random fields \
    (latent fields)
    - Y is the coding process, a 2D multi-Gaussian random field

    Z and T1, T2 are fields in 1D, 2D or 3D.

    Notes
    -----
    The module :mod:`multiGaussian` is used for the multi-Gaussian fields T1, T2 and Y.

    Parameters
    ----------
    cov_model_T1 : :class:`geone.covModel.CovModel<d>D`
        covariance model for T1, in 1D or 2D or 3D;
        note: can be set to `None`; in this case, `algo_T1='deterministic'`
        is requiered and `params_T1['mean']` defines the field T1

    cov_model_T2 : :class:`geone.covModel.CovModel<d>D`
        covariance model for T2, in 1D or 2D or 3D
        note: can be set to `None`; in this case, `algo_T2='deterministic'`
        is requiered and `params_T2['mean']` defines the field T2

    cov_model_Y : :class:`geone.covModel.CovModel2D`
        covariance model for Y, in 2D

    dimension : [sequence of] int(s)
        number of cells along each axis, for simulation in:

        - 1D: `dimension=nx`
        - 2D: `dimension=(nx, ny)`
        - 3D: `dimension=(nx, ny, nz)`

    spacing : [sequence of] float(s), optional
        cell size along each axis, for simulation in:

        - 1D: `spacing=sx`
        - 2D: `spacing=(sx, sy)`
        - 3D: `spacing=(sx, sy, sz)`

        by default (`None`): 1.0 along each axis

    origin : [sequence of] float(s), optional
        origin of the grid ("corner of the first cell"), for simulation in:

        - 1D: `origin=ox`
        - 2D: `origin=(ox, oy)`
        - 3D: `origin=(ox, oy, oz)`

        by default (`None`): 0.0 along each axis

    spacing_Y : sequence of 2 floats, default: (0.001, 0.001)
        two positive values, resolution of the 2D Y field, along the two
        dimensions (corresponding to T1 and T2), spacing between two adjacent
        cells in the two directions

    x : array-like of floats, optional
        data points locations (float coordinates), for simulation in:

        - 1D: 1D array-like of floats
        - 2D: 2D array-like of floats of shape (m, 2)
        - 3D: 2D array-like of floats of shape (m, 3)

        note: if one point (m=1), a float in 1D, a 1D array of shape (2,) in 2D,
        a 1D array of shape (3,) in 3D, is accepted

    v : 1d-array-like of floats, optional
        data values at `x` (`v[i]` is the data value at `x[i]`)

    t : 2d-array of floats or sequence of 2 floats, optional
        values of (T1, T2) considered as conditioning point for Y(T) (additional constraint)m
        each row corresponding to one point;
        note: if only one point, a sequence of 2 floats is accepted

    yt : 1d-array-like of floats, or float, optional
        value of Y at the conditioning point `t`

    vmin : float, optional
        minimal value for Z (or Y); simulation are rejected if not honoured

    vmax : float, optional
        maximal value for Z (or Y); simulation are rejected if not honoured

    algo_T1 : str
        defines the algorithm used for generating multi-Gaussian field T1:

        - 'fft' or 'FFT' (default): based on circulant embedding and FFT, \
        function called for <d>D (d = 1, 2, or 3): `geone.grf.grf<d>D`
        - 'classic' or 'CLASSIC': classic algorithm, based on the resolution \
        of kriging system considered points in a search ellipsoid, function \
        called for <d>D (d = 1, 2, or 3): `geone.geoscalassicinterface.simulate<d>D`
        - 'deterministic' or 'DETERMINISTIC': use a deterministic field defined \
        by `params_T1['mean']`

    params_T1 : dict, optional
        keyword arguments (additional parameters) to be passed to the function
        corresponding to what is specified by the argument `algo_T1` (see the
        corresponding function for its keyword arguments), in particular the key
        'mean' can be specified (set to value 0 if not specified)

    algo_T2 : str
        defines the algorithm used for generating multi-Gaussian field T2:

        - 'fft' or 'FFT' (default): based on circulant embedding and FFT, \
        function called for <d>D (d = 1, 2, or 3): `geone.grf.grf<d>D`
        - 'classic' or 'CLASSIC': classic algorithm, based on the resolution \
        of kriging system considered points in a search ellipsoid, function \
        called for <d>D (d = 1, 2, or 3): `geone.geoscalassicinterface.simulate<d>D`
        - 'deterministic' or 'DETERMINISTIC': use a deterministic field defined \
        by `params_T2['mean']`

    params_T2 : dict, optional
        keyword arguments (additional parameters) to be passed to the function
        corresponding to what is specified by the argument `algo_T2` (see the
        corresponding function for its keyword arguments), in particular the key
        'mean' can be specified (set to value 0 if not specified)

    algo_Y : str
        defines the algorithm used for generating 2D multi-Gaussian field Y:

        - 'fft' or 'FFT' (default): based on circulant embedding and FFT, \
        function called: :func:`geone.grf.grf2D`
        - 'classic' or 'CLASSIC': classic algorithm, based on the resolution \
        of kriging system considered points in a search ellipsoid, function \
        called: :func:`geone.geoscalassicinterface.simulate`

    params_Y : dict, optional
        keyword arguments (additional parameters) to be passed to the function
        corresponding to what is specified by the argument `algo_Y` (see the
        corresponding function for its keyword arguments), in particular the key
        'mean' can be specified (if not specified, set to the mean value of `v`
        if `v` is not `None`, set to 0 otherwise)

    mh_iter : int, default: 100
        number of iteration for Metropolis-Hasting algorithm, for conditional
        simulation only; note: used only if `x` or `t` is not `None`

    ntry_max : int, default: 1
        number of tries per realization before giving up if something goes wrong

    nreal : int, default: 1
        number of realization(s)

    full_output : bool, default: True
        - if `True`: simulation(s) of Z, T1, T2, and Y are retrieved in output
        - if `False`: simulation(s) of Z only is retrieved in output

    verbose : int, default: 1
        verbose mode, integer >=0, higher implies more display

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    Z : nd-array
        all realizations, `Z[k]` is the `k`-th realization:

        - for 1D: `Z` of shape (nreal, nx), where nx = dimension
        - for 2D: `Z` of shape (nreal, ny, nx), where nx, ny = dimension
        - for 3D: `Z` of shape (nreal, nz, ny, nx), where nx, ny, nz = dimension

    T1 : nd-array
        latent fields of all realizations, `T1[k]` for the `k`-th realization:

        - for 1D: `T1` of shape (nreal, nx), where nx = dimension
        - for 2D: `T1` of shape (nreal, ny, nx), where nx, ny = dimension
        - for 3D: `T1` of shape (nreal, nz, ny, nx), where nx, ny, nz = dimension

        returned if `full_output=True`

    T2 : nd-array
        latent fields of all realizations, `T2[k]` for the `k`-th realization:

        - for 1D: `T2` of shape (nreal, nx), where nx = dimension
        - for 2D: `T2` of shape (nreal, ny, nx), where nx, ny = dimension
        - for 3D: `T2` of shape (nreal, nz, ny, nx), where nx, ny, nz = dimension

        returned if `full_output=True`

    Y : list of length nreal
        2D random fields of all realizations, `Y[k]` is a list of length 4 for
        the `k`-th realization:

        - Y[k][0]: 1d-array of shape (2,): (Y_nx, Y_ny)
        - Y[k][1]: 1d-array of shape (2,): (Y_sx, Y_sy)
        - Y[k][2]: 1d-array of shape (2,): (Y_ox, Y_oy)
        - Y[k][3]: 2d-array of shape (Y_ny, Y_nx): values of Y[k]

        returned if `full_output=True`
    """
    fname = 'srf_bimg_mg'

    if algo_T1 not in ('fft', 'FFT', 'classic', 'CLASSIC', 'deterministic', 'DETERMINISTIC'):
        err_msg = f"{fname}: `algo_T1` invalid, should be 'fft' (default) or 'classic'"
        if logger: logger.error(err_msg)
        raise SrfError(err_msg)

    if algo_T2 not in ('fft', 'FFT', 'classic', 'CLASSIC', 'deterministic', 'DETERMINISTIC'):
        err_msg = f"{fname}: `algo_T2` invalid, should be 'fft' (default) or 'classic'"
        if logger: logger.error(err_msg)
        raise SrfError(err_msg)

    if algo_Y not in ('fft', 'FFT', 'classic', 'CLASSIC'):
        err_msg = f"{fname}: `algo_Y` invalid, should be 'fft' (default) or 'classic'"
        if logger: logger.error(err_msg)
        raise SrfError(err_msg)

    # Ignore covariance model if 'algo' is deterministic for T1, T2
    if algo_T1 in ('deterministic', 'DETERMINISTIC'):
        cov_model_T1 = None

    if algo_T2 in ('deterministic', 'DETERMINISTIC'):
        cov_model_T2 = None

    # Set space dimension (of grid) according to covariance model for T1
    d = 0
    if cov_model_T1 is None:
        if algo_T1 not in ('deterministic', 'DETERMINISTIC'):
            err_msg = f"{fname}: `cov_model_T1` is `None`, then `algo_T1` must be 'deterministic'"
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

    elif isinstance(cov_model_T1, gcm.CovModel1D):
        d = 1
    elif isinstance(cov_model_T1, gcm.CovModel2D):
        d = 2
    elif isinstance(cov_model_T1, gcm.CovModel3D):
        d = 3
    else:
        err_msg = f'{fname}: `cov_model_T1` invalid, should be a class `geone.covModel.CovModel1D`, `geone.covModel.CovModel2D` or `geone.covModel.CovModel3D`'
        if logger: logger.error(err_msg)
        raise SrfError(err_msg)

    if cov_model_T2 is None:
        if algo_T2 not in ('deterministic', 'DETERMINISTIC'):
            err_msg = f"{fname}: `cov_model_T2` is `None`, then `algo_T2` must be 'deterministic'"
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

        # if d == 0:
        #     err_msg = f'{fname}: `cov_model_T1` and `cov_model_T2` are `None`, at least one covariance model is required'
        #     if logger: logger.error(err_msg)
        #     raise SrfError(err_msg)

    elif (d == 1 and not isinstance(cov_model_T2, gcm.CovModel1D)) or (d == 2 and not isinstance(cov_model_T2, gcm.CovModel2D)) or (d == 3 and not isinstance(cov_model_T2, gcm.CovModel3D)):
        err_msg = f'{fname}: `cov_model_T1` and `cov_model_T2` not compatible (dimensions differ)'
        if logger: logger.error(err_msg)
        raise SrfError(err_msg)

    if d == 0:
        # Set space dimension (of grid) according to 'dimension'
        if hasattr(dimension, '__len__'):
            d = len(dimension)
        else:
            d = 1

    # Check argument 'dimension'
    if hasattr(dimension, '__len__') and len(dimension) != d:
        err_msg = f'{fname}: `dimension` of incompatible length'
        if logger: logger.error(err_msg)
        raise SrfError(err_msg)

    if d == 1:
        grid_size = dimension
    else:
        grid_size = np.prod(dimension)

    # Check (or set) argument 'spacing'
    if spacing is None:
        if d == 1:
            spacing = 1.0
        else:
            spacing = tuple(np.ones(d))
    else:
        if hasattr(spacing, '__len__') and len(spacing) != d:
            err_msg = f'{fname}: `spacing` of incompatible length'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

    # Check (or set) argument 'origin'
    if origin is None:
        if d == 1:
            origin = 0.0
        else:
            origin = tuple(np.zeros(d))
    else:
        if hasattr(origin, '__len__') and len(origin) != d:
            err_msg = f'{fname}: `origin` of incompatible length'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

    # if not cov_model_T1.is_stationary(): # prevent calculation if covariance model is not stationary
    #     if verbose > 0:
    #         print(f'ERROR ({fname}): `cov_model_T1` is not stationary')

    # if not cov_model_T2.is_stationary(): # prevent calculation if covariance model is not stationary
    #     if verbose > 0:
    #         print(f'ERROR ({fname}): `cov_model_T2` is not stationary')

    # Check covariance model for Y
    if not isinstance(cov_model_Y, gcm.CovModel2D):
        err_msg = f'{fname}: `cov_model_Y` invalid'
        if logger: logger.error(err_msg)
        raise SrfError(err_msg)

    # elif not cov_model_Y.is_stationary(): # prevent calculation if covariance model is not stationary
    #     err_msg = f'{fname}: `cov_model_Y` is not stationary'
    #     if logger: logger.error(err_msg)
    #     raise SrfError(err_msg)

    # Check additional constraint t (conditioning point for (T1, T2)), yt (corresponding value for Y)
    if t is None:
        if yt is not None:
            err_msg = f'{fname}: `t` is not given (`None`) but `yt` is given (not `None`)'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

    else:
        if yt is None:
            err_msg = f'{fname}: `t` is given (not `None`) but `yt` is not given (`None`)'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

        t = np.asarray(t, dtype='float').reshape(-1, 2) # cast in 2-dimensional array if needed
        yt = np.asarray(yt, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(yt) != len(t):
            err_msg = f'{fname}: length of `yt` is not valid'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

    # Initialize dictionaries params_T1, params_T2
    if params_T1 is None:
        params_T1 = {}
    if params_T2 is None:
        params_T2 = {}

    # Compute meshgrid over simulation domain if needed (see below)
    if ('mean' in params_T1.keys() and callable(params_T1['mean'])) or ('var' in params_T1.keys() and callable(params_T1['var'])) \
    or ('mean' in params_T2.keys() and callable(params_T2['mean'])) or ('var' in params_T2.keys() and callable(params_T2['var'])):
        if d == 1:
            xi = origin + spacing*(0.5+np.arange(dimension)) # x-coordinate of cell center
        elif d == 2:
            xi = origin[0] + spacing[0]*(0.5+np.arange(dimension[0])) # x-coordinate of cell center
            yi = origin[1] + spacing[1]*(0.5+np.arange(dimension[1])) # y-coordinate of cell center
            yyi, xxi = np.meshgrid(yi, xi, indexing='ij')
        elif d == 3:
            xi = origin[0] + spacing[0]*(0.5+np.arange(dimension[0])) # x-coordinate of cell center
            yi = origin[1] + spacing[1]*(0.5+np.arange(dimension[1])) # y-coordinate of cell center
            zi = origin[2] + spacing[2]*(0.5+np.arange(dimension[2])) # z-coordinate of cell center
            zzi, yyi, xxi = np.meshgrid(zi, yi, xi, indexing='ij')

    # Set mean_T1 (as array) from params_T1
    if 'mean' not in params_T1.keys():
        mean_T1 = np.array([0.0])
    else:
        mean_T1 = params_T1['mean']
        if mean_T1 is None:
            mean_T1 = np.array([0.0])
        elif callable(mean_T1):
            if d == 1:
                mean_T1 = mean_T1(xi).reshape(-1) # replace function 'mean_T1' by its evaluation on the grid
            elif d == 2:
                mean_T1 = mean_T1(xxi, yyi).reshape(-1) # replace function 'mean_T1' by its evaluation on the grid
            elif d == 3:
                mean_T1 = mean_T1(xxi, yyi, zzi).reshape(-1) # replace function 'mean_T1' by its evaluation on the grid
        else:
            mean_T1 = np.asarray(mean_T1).reshape(-1)
            if mean_T1.size not in (1, grid_size):
                err_msg = f"{fname}: 'mean' parameter for T1 (in `params_T1`) has incompatible size"
                if logger: logger.error(err_msg)
                raise SrfError(err_msg)

    # Set var_T1 (as array) from params_T1, if given
    var_T1 = None
    if 'var' in params_T1.keys():
        var_T1 = params_T1['var']
        if var_T1 is not None:
            if callable(var_T1):
                if d == 1:
                    var_T1 = var_T1(xi).reshape(-1) # replace function 'var_T1' by its evaluation on the grid
                elif d == 2:
                    var_T1 = var_T1(xxi, yyi).reshape(-1) # replace function 'var_T1' by its evaluation on the grid
                elif d == 3:
                    var_T1 = var_T1(xxi, yyi, zzi).reshape(-1) # replace function 'var_T1' by its evaluation on the grid
            else:
                var_T1 = np.asarray(var_T1).reshape(-1)
                if var_T1.size not in (1, grid_size):
                    err_msg = f"{fname}: 'var' parameter for T1 (in `params_T1`) has incompatible size"
                    if logger: logger.error(err_msg)
                    raise SrfError(err_msg)

    # Set mean_T2 (as array) from params_T2
    if 'mean' not in params_T2.keys():
        mean_T2 = np.array([0.0])
    else:
        mean_T2 = params_T2['mean']
        if mean_T2 is None:
            mean_T2 = np.array([0.0])
        elif callable(mean_T2):
            if d == 1:
                mean_T2 = mean_T2(xi).reshape(-1) # replace function 'mean_T2' by its evaluation on the grid
            elif d == 2:
                mean_T2 = mean_T2(xxi, yyi).reshape(-1) # replace function 'mean_T2' by its evaluation on the grid
            elif d == 3:
                mean_T2 = mean_T2(xxi, yyi, zzi).reshape(-1) # replace function 'mean_T2' by its evaluation on the grid
        else:
            mean_T2 = np.asarray(mean_T2).reshape(-1)
            if mean_T2.size not in (1, grid_size):
                err_msg = f"{fname}: 'mean' parameter for T2 (in `params_T2`) has incompatible size"
                if logger: logger.error(err_msg)
                raise SrfError(err_msg)

    # Set var_T2 (as array) from params_T2, if given
    var_T2 = None
    if 'var' in params_T2.keys():
        var_T2 = params_T2['var']
        if var_T2 is not None:
            if callable(var_T2):
                if d == 1:
                    var_T2 = var_T2(xi).reshape(-1) # replace function 'var_T2' by its evaluation on the grid
                elif d == 2:
                    var_T2 = var_T2(xxi, yyi).reshape(-1) # replace function 'var_T2' by its evaluation on the grid
                elif d == 3:
                    var_T2 = var_T2(xxi, yyi, zzi).reshape(-1) # replace function 'var_T2' by its evaluation on the grid
            else:
                var_T2 = np.asarray(var_T2).reshape(-1)
                if var_T2.size not in (1, grid_size):
                    err_msg = f"{fname}: 'var' parameter for T2 (in `params_T2`) has incompatible size"
                    if logger: logger.error(err_msg)
                    raise SrfError(err_msg)

    # Initialize dictionary params_Y
    if params_Y is None:
        params_Y = {}

    # Set mean_Y from params_Y (if given, and check if it is a unique value)
    mean_Y = None
    if 'mean' in params_Y.keys():
        mean_Y = params_Y['mean']
        if callable(mean_Y):
            err_msg = f"{fname}: 'mean' parameter for Y (in `params_Y`) must be a unique value (float) if given"
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

        else:
            mean_Y = np.asarray(mean_Y, dtype='float').reshape(-1)
            if mean_Y.size != 1:
                err_msg = f"{fname}: 'mean' parameter for Y (in `params_Y`) must be a unique value (float) if given"
                if logger: logger.error(err_msg)
                raise SrfError(err_msg)

            mean_Y = mean_Y[0]

    # Check var_Y from params_Y
    if 'var' in params_Y.keys() and params_Y['var'] is not None:
        err_msg = f"{fname}: 'var' parameter for Y (in `params_Y`) must be `None`"
        if logger: logger.error(err_msg)
        raise SrfError(err_msg)

    # Number of realization(s)
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if full_output:
            if verbose > 0:
                if logger:
                    logger.warning(f'{fname}: `nreal` <= 0: `None`, `None`, `None` is returned')
                else:
                    print(f'{fname}: WARNING: `nreal` <= 0: `None`, `None`, `None` is returned')
            return None, None, None
        else:
            if verbose > 0:
                if logger:
                    logger.warning(f'{fname}: `nreal` <= 0: `None` is returned')
                else:
                    print(f'{fname}: WARNING: `nreal` <= 0: `None` is returned')
            return None

    # Note: format of data (x, v) not checked !

    if x is None:
        if v is not None:
            err_msg = f'{fname}: `x` is not given (`None`) but `v` is given (not `None`)'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

        # Preparation for unconditional case
        # Set mean_Y
        if mean_Y is None:
            mean_Y = 0.0
    #
    else:
        # Preparation for conditional case
        if v is None:
            err_msg = f'{fname}: `x` is given (not `None`) but `v` is not given (`None`)'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

        x = np.asarray(x, dtype='float').reshape(-1, d) # cast in d-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise SrfError(err_msg)

        # Number of conditioning points
        npt = x.shape[0]

        # Get index in mean_T1 for each conditioning points
        x_mean_T1_grid_ind = None
        if mean_T1.size == 1:
            x_mean_T1_grid_ind = np.zeros(npt, dtype='int')
        else:
            indc_f = (x-origin)/spacing
            indc = indc_f.astype(int)
            indc = indc - 1 * np.all((indc == indc_f, indc > 0), axis=0)
            if d == 1:
                x_mean_T1_grid_ind = 1 * indc[:, 0] # multiply by 1.0 makes a copy of the array !
            elif d == 2:
                x_mean_T1_grid_ind = indc[:, 0] + dimension[0] * indc[:, 1]
            elif d == 3:
                x_mean_T1_grid_ind = indc[:, 0] + dimension[0] * (indc[:, 1] + dimension[1] * indc[:, 2])

        # Get index in var_T1 (if not None) for each conditioning points
        if var_T1 is not None:
            if var_T1.size == 1:
                x_var_T1_grid_ind = np.zeros(npt, dtype='int')
            else:
                if x_mean_T1_grid_ind is not None:
                    x_var_T1_grid_ind = x_mean_T1_grid_ind
                else:
                    indc_f = (x-origin)/spacing
                    indc = indc_f.astype(int)
                    indc = indc - 1 * np.all((indc == indc_f, indc > 0), axis=0)
                    if d == 1:
                        x_var_T1_grid_ind = 1 * indc[:, 0] # multiply by 1.0 makes a copy of the array !
                    elif d == 2:
                        x_var_T1_grid_ind = indc[:, 0] + dimension[0] * indc[:, 1]
                    elif d == 3:
                        x_var_T1_grid_ind = indc[:, 0] + dimension[0] * (indc[:, 1] + dimension[1] * indc[:, 2])

        # Get index in mean_T2 for each conditioning points
        x_mean_T2_grid_ind = None
        if mean_T2.size == 1:
            x_mean_T2_grid_ind = np.zeros(npt, dtype='int')
        else:
            indc_f = (x-origin)/spacing
            indc = indc_f.astype(int)
            indc = indc - 1 * np.all((indc == indc_f, indc > 0), axis=0)
            if d == 1:
                x_mean_T2_grid_ind = 1 * indc[:, 0] # multiply by 1.0 makes a copy of the array !
            elif d == 2:
                x_mean_T2_grid_ind = indc[:, 0] + dimension[0] * indc[:, 1]
            elif d == 3:
                x_mean_T2_grid_ind = indc[:, 0] + dimension[0] * (indc[:, 1] + dimension[1] * indc[:, 2])

        # Get index in var_T2 (if not None) for each conditioning points
        if var_T2 is not None:
            if var_T2.size == 1:
                x_var_T2_grid_ind = np.zeros(npt, dtype='int')
            else:
                if x_mean_T2_grid_ind is not None:
                    x_var_T2_grid_ind = x_mean_T2_grid_ind
                else:
                    indc_f = (x-origin)/spacing
                    indc = indc_f.astype(int)
                    indc = indc - 1 * np.all((indc == indc_f, indc > 0), axis=0)
                    if d == 1:
                        x_var_T2_grid_ind = 1 * indc[:, 0] # multiply by 1.0 makes a copy of the array !
                    elif d == 2:
                        x_var_T2_grid_ind = indc[:, 0] + dimension[0] * indc[:, 1]
                    elif d == 3:
                        x_var_T2_grid_ind = indc[:, 0] + dimension[0] * (indc[:, 1] + dimension[1] * indc[:, 2])

        # Get covariance function for T1, T2 and Y, and their evaluation at 0
        if cov_model_T1 is not None:
            cov_func_T1 = cov_model_T1.func() # covariance function
            cov0_T1 = cov_func_T1(np.zeros(d))
        if cov_model_T2 is not None:
            cov_func_T2 = cov_model_T2.func() # covariance function
            cov0_T2 = cov_func_T2(np.zeros(d))
        cov_func_Y = cov_model_Y.func() # covariance function
        cov0_Y = cov_func_Y(np.zeros(2))

        # Set mean_Y
        if mean_Y is None:
            mean_Y = np.mean(v)

        if cov_model_T1 is not None:
            # Set kriging matrix for T1 (mat_T1) of order npt, "over every conditioining point"
            mat_T1 = np.ones((npt, npt))
            for i in range(npt-1):
                # lag between x[i] and x[j], j=i+1, ..., npt-1
                h = x[(i+1):] - x[i]
                cov_h_T1 = cov_func_T1(h)
                mat_T1[i, (i+1):npt] = cov_h_T1
                mat_T1[(i+1):npt, i] = cov_h_T1
                mat_T1[i, i] = cov0_T1

            mat_T1[-1,-1] = cov0_T1

            if var_T1 is not None:
                varUpdate = np.sqrt(var_T1[x_var_T1_grid_ind]/cov0_T1)
                mat_T1 = varUpdate*(mat_T1.T*varUpdate).T

        if cov_model_T2 is not None:
            # Set kriging matrix for T2 (mat_T2) of order npt, "over every conditioining point"
            mat_T2 = np.ones((npt, npt))
            for i in range(npt-1):
                # lag between x[i] and x[j], j=i+1, ..., npt-1
                h = x[(i+1):] - x[i]
                cov_h_T2 = cov_func_T2(h)
                mat_T2[i, (i+1):npt] = cov_h_T2
                mat_T2[(i+1):npt, i] = cov_h_T2
                mat_T2[i, i] = cov0_T2

            mat_T2[-1,-1] = cov0_T2

            if var_T2 is not None:
                varUpdate = np.sqrt(var_T2[x_var_T2_grid_ind]/cov0_T2)
                mat_T2 = varUpdate*(mat_T2.T*varUpdate).T

        # Initialize
        #   - npt_ext: number of total conditioning point for Y, "point (T1(x), T2(x)) + additional constraint t"
        #   - v_T: values of (T1(x), T2(x)) (that are defined later) followed by values yt at additional constraint t"
        #   - v_ext: values for Y at "point (T1(x), T2(x)) + additional constraint (t)"
        #   - mat_Y: kriging matrix for Y of order npt_ext, over "point (T1(x), T2(x)) + additional constraint t"
        if t is None:
            npt_ext = npt
            v_T = np.zeros((npt, 2))
            v_ext = v
            mat_Y = np.ones((npt_ext, npt_ext))
        else:
            npt_ext = npt + len(t)
            v_T = np.vstack((np.zeros((npt, 2)), t))
            v_ext = np.hstack((v, yt))
            mat_Y = np.ones((npt_ext, npt_ext))
            for i in range(len(t)-1):
                # lag between t[i] and t[j], j=i+1, ..., len(t)-1
                h = t[(i+1):] - t[i]
                cov_h_Y = cov_func_Y(h)
                k = i + npt
                mat_Y[k, (k+1):] = cov_h_Y
                mat_Y[(k+1):, k] = cov_h_Y
                #mat_Y[k, k] = cov0_Y

            #mat_Y[-1,-1] = cov0_Y
        for i in range(npt_ext):
            mat_Y[i, i] = cov0_Y
        #
        if npt_ext <= 1:
            mh_iter = 0 # unnecessary to apply Metropolis update !

    # Set (again if given) default parameter 'mean' and 'var' for T1, T2, and 'mean' for Y
    if cov_model_T1 is not None:
        params_T1['mean'] = mean_T1
        params_T1['var'] = var_T1
    else:
        if mean_T1.size == grid_size:
            params_T1['mean'] = mean_T1.reshape(*dimension[::-1])
        else:
            params_T1['mean'] = mean_T1 * np.ones(dimension[::-1])
    if cov_model_T2 is not None:
        params_T2['mean'] = mean_T2
        params_T2['var'] = var_T2
    else:
        if mean_T2.size == grid_size:
            params_T2['mean'] = mean_T2.reshape(*dimension[::-1])
        else:
            params_T2['mean'] = mean_T2 * np.ones(dimension[::-1])
    params_Y['mean'] = mean_Y

    # Set default parameter 'verbose' for params_T1, params_T2 and params_Y
    if 'verbose' not in params_T1.keys():
        params_T1['verbose'] = 0
        # params_T1['verbose'] = verbose
    if 'verbose' not in params_T2.keys():
        params_T2['verbose'] = 0
        # params_T2['verbose'] = verbose
    if 'verbose' not in params_Y.keys():
        params_Y['verbose'] = 0
        # params_Y['verbose'] = verbose

    # Initialization for output
    Z = []
    if full_output:
        T1 = []
        T2 = []
        Y = []

    for ireal in range(nreal):
        # Generate ireal-th realization
        if verbose > 1:
            if logger:
                logger.info(f'{fname}: simulation {ireal+1} of {nreal}...')
            else:
                print(f'{fname}: simulation {ireal+1} of {nreal}...')
        for ntry in range(ntry_max):
            sim_ok = True
            Y_cond_aggregation = False
            if verbose > 2 and ntry > 0:
                if logger:
                    logger.info(f'{fname}:   ... new trial ({ntry+1} of {ntry_max}) for simulation {ireal+1} of {nreal}...')
                else:
                    print(f'{fname}:   ... new trial ({ntry+1} of {ntry_max}) for simulation {ireal+1} of {nreal}...')
            if x is None:
                # Unconditional case
                # ------------------
                # Generate T1 (one real)
                if cov_model_T1 is not None:
                    try:
                        sim_T1 = multiGaussian.multiGaussianRun(
                                cov_model_T1, dimension, spacing, origin,
                                mode='simulation', algo=algo_T1, output_mode='array',
                                **params_T1, nreal=1, logger=logger)
                    except:
                        sim_ok = False
                        if verbose > 2:
                            if logger:
                                logger.info(f'{fname}:   ... simulation of T1 failed')
                            else:
                                print(f'{fname}:   ... simulation of T1 failed')
                        continue
                    # except Exception as exc:
                    #     err_msg = f'{fname}: simulation of T1 failed'
                    #     if logger: logger.error(err_msg)
                    #     raise SrfError(err_msg) from exc

                else:
                    sim_T1 = params_T1['mean'].reshape(1,*dimension[::-1])
                # -> sim_T1: nd-array of shape
                #      (1, dimension) (for T1 in 1D)
                #      (1, dimension[1], dimension[0]) (for T1 in 2D)
                #      (1, dimension[2], dimension[1], dimension[0]) (for T1 in 3D)

                # Generate T2 (one real)
                if cov_model_T2 is not None:
                    try:
                        sim_T2 = multiGaussian.multiGaussianRun(
                                cov_model_T2, dimension, spacing, origin,
                                mode='simulation', algo=algo_T2, output_mode='array',
                                **params_T2, nreal=1, logger=logger)
                    except:
                        sim_ok = False
                        if verbose > 2:
                            if logger:
                                logger.info(f'{fname}:   ... simulation of T2 failed')
                            else:
                                print(f'{fname}:   ... simulation of T2 failed')
                        continue
                    # except Exception as exc:
                    #     err_msg = f'{fname}: simulation of T2 failed'
                    #     if logger: logger.error(err_msg)
                    #     raise SrfError(err_msg) from exc
                else:
                    sim_T2 = params_T2['mean'].reshape(1,*dimension[::-1])
                # -> sim_T2: nd-array of shape
                #      (1, dimension) (for T2 in 1D)
                #      (1, dimension[1], dimension[0]) (for T2 in 2D)
                #      (1, dimension[2], dimension[1], dimension[0]) (for T2 in 3D)

                # Set origin and dimension for Y
                origin_Y = [0.0, 0.0]
                dimension_Y = [0, 0]

                min_T1 = np.min(sim_T1)
                max_T1 = np.max(sim_T1)
                if t is not None:
                    min_T1 = min(t[:, 0].min(), min_T1)
                    max_T1 = max(t[:, 0].max(), max_T1)
                min_T1 = min_T1 - 0.5 * spacing_Y[0]
                max_T1 = max_T1 + 0.5 * spacing_Y[0]
                dimension_Y[0] = int(np.ceil((max_T1 - min_T1)/spacing_Y[0]))
                origin_Y[0] = min_T1 - 0.5*(dimension_Y[0]*spacing_Y[0] - (max_T1 - min_T1))

                min_T2 = np.min(sim_T2)
                max_T2 = np.max(sim_T2)
                if t is not None:
                    min_T2 = min(t[:, 1].min(), min_T2)
                    max_T2 = max(t[:, 1].max(), max_T2)
                min_T2 = min_T2 - 0.5 * spacing_Y[1]
                max_T2 = max_T2 + 0.5 * spacing_Y[1]
                dimension_Y[1] = int(np.ceil((max_T2 - min_T2)/spacing_Y[1]))
                origin_Y[1] = min_T2 - 0.5*(dimension_Y[1]*spacing_Y[1] - (max_T2 - min_T2))

                # Generate Y conditional to possible additional constraint (t, yt) (one real)
                try:
                    sim_Y = multiGaussian.multiGaussianRun(
                            cov_model_Y, dimension_Y, spacing_Y, origin_Y, x=t, v=yt,
                            mode='simulation', algo=algo_Y, output_mode='array',
                            **params_Y, nreal=1, logger=logger)
                except:
                    sim_ok = False
                    if verbose > 2:
                        if logger:
                            logger.info(f'{fname}:   ... simulation of Y failed')
                        else:
                            print(f'{fname}:   ... simulation of Y failed')
                    continue
                # except Exception as exc:
                #     err_msg = f'{fname}: simulation of Y failed'
                #     if logger: logger.error(err_msg)
                #     raise SrfError(err_msg) from exc

                # -> 3d-array of shape (1, dimension_Y[1], dimension_Y[0])

            else:
                # Conditional case
                # ----------------
                # Initialize: unconditional simulation of T1 at x (values in v_T[:,0])
                ind = np.random.permutation(npt)
                for j, k in enumerate(ind):
                    if cov_model_T1 is not None:
                        # Simulate value at x[k] (= x[ind[j]]), conditionally to the previous ones
                        # Solve the kriging system (for T1)
                        try:
                            w = np.linalg.solve(
                                    mat_T1[ind[:j], :][:, ind[:j]], # kriging matrix
                                    mat_T1[ind[:j], ind[j]], # second member
                                )
                        except:
                            sim_ok = False
                            break

                        # Mean (kriged) value at x[k]
                        mu_T1_k = mean_T1[x_mean_T1_grid_ind[k]] + (v_T[ind[:j], 0] - mean_T1[x_mean_T1_grid_ind[ind[:j]]]).dot(w)
                        # Standard deviation (of kriging) at x[k]
                        std_T1_k = np.sqrt(np.maximum(0, cov0_T1 - np.dot(w, mat_T1[ind[:j], ind[j]])))
                        # Draw value in N(mu_T1_k, std_T1_k^2)
                        v_T[k, 0] = np.random.normal(loc=mu_T1_k, scale=std_T1_k)
                    else:
                        v_T[k, 0] = mean_T1[x_mean_T1_grid_ind[k]]

                if not sim_ok:
                    sim_ok = False
                    if verbose > 2:
                        if logger:
                            logger.info(f'{fname}:    ... cannot solve kriging system (for T1, initialization)')
                        else:
                            print(f'{fname}:    ... cannot solve kriging system (for T1, initialization)')
                    continue

                # Initialize: unconditional simulation of T2 at x (values in v_T[:,1])
                ind = np.random.permutation(npt)
                for j, k in enumerate(ind):
                    if cov_model_T2 is not None:
                        # Simulate value at x[k] (= x[ind[j]]), conditionally to the previous ones
                        # Solve the kriging system (for T2)
                        try:
                            w = np.linalg.solve(
                                    mat_T2[ind[:j], :][:, ind[:j]], # kriging matrix
                                    mat_T2[ind[:j], ind[j]], # second member
                                )
                        except:
                            sim_ok = False
                            break

                        # Mean (kriged) value at x[k]
                        mu_T2_k = mean_T2[x_mean_T2_grid_ind[k]] + (v_T[ind[:j], 1] - mean_T2[x_mean_T2_grid_ind[ind[:j]]]).dot(w)
                        # Standard deviation (of kriging) at x[k]
                        std_T2_k = np.sqrt(np.maximum(0, cov0_T2 - np.dot(w, mat_T2[ind[:j], ind[j]])))
                        # Draw value in N(mu_T2_k, std_T2_k^2)
                        v_T[k, 1] = np.random.normal(loc=mu_T2_k, scale=std_T2_k)
                    else:
                        v_T[k, 1] = mean_T2[x_mean_T2_grid_ind[k]]

                if not sim_ok:
                    sim_ok = False
                    if verbose > 2:
                        if logger:
                            logger.info(f'{fname}:    ... cannot solve kriging system (for T2, initialization)')
                        else:
                            print(f'{fname}:    ... cannot solve kriging system (for T2, initialization)')
                    continue

                # Updated kriging matrix for Y (mat_Y) according to value in v_T[0:npt]
                for i in range(npt-1):
                    # lag between v_T[i] and v_T[j], j=i+1, ..., npt-1
                    h = v_T[(i+1):npt] - v_T[i]
                    cov_h_Y = cov_func_Y(h)
                    mat_Y[i, (i+1):npt] = cov_h_Y
                    mat_Y[(i+1):npt, i] = cov_h_Y
                    # mat_Y[i, i] = cov0_Y

                for i, k in enumerate(range(npt, npt_ext)):
                    # lag between t[i] and v_T[j], j=0, ..., npt-1
                    h = v_T[0:npt] - t[i]
                    cov_h_Y = cov_func_Y(h)
                    mat_Y[k, 0:npt] = cov_h_Y
                    mat_Y[0:npt, k] = cov_h_Y
                    # mat_Y[i, i] = cov0_Y

                # mat_Y[-1,-1] = cov0_Y

                # Update simulated values v_T at x using Metropolis-Hasting (MH) algorithm
                v_T_k_new = np.zeros(2)
                for nit in range(mh_iter):
                    if verbose > 3:
                        if logger:
                            logger.info(f'{fname}:   ... sim {ireal+1} of {nreal}: MH iter {nit+1} of {mh_iter}...')
                        else:
                            print(f'{fname}:   ... sim {ireal+1} of {nreal}: MH iter {nit+1} of {mh_iter}...')
                    ind = np.random.permutation(npt)
                    for k in ind:
                        # Sequence of indexes without k
                        indmat = np.hstack((np.arange(k), np.arange(k+1, npt)))
                        # Simulate possible new value v_T_new at x[k], conditionally to all the ohter ones
                        #
                        if cov_model_T1 is not None:
                            # Solve the kriging system for T1
                            try:
                                w = np.linalg.solve(
                                        mat_T1[indmat, :][:, indmat], # kriging matrix
                                        mat_T1[indmat, k], # second member
                                    )
                            except:
                                sim_ok = False
                                if verbose > 2:
                                    if logger:
                                        logger.info(f'{fname}:   ... cannot solve kriging system (for T1)')
                                    else:
                                        print(f'{fname}:   ... cannot solve kriging system (for T1)')
                                break
                            #
                            # Mean (kriged) value at x[k]
                            mu_T1_k = mean_T1[x_mean_T1_grid_ind[k]] + (v_T[indmat, 0] - mean_T1[x_mean_T1_grid_ind[indmat]]).dot(w)
                            # Standard deviation (of kriging) at x[k]
                            std_T1_k = np.sqrt(np.maximum(0, cov0_T1 - np.dot(w, mat_T1[indmat, k])))
                            # Draw value in N(mu, std^2)
                            v_T_k_new[0] = np.random.normal(loc=mu_T1_k, scale=std_T1_k)
                        else:
                            v_T_k_new[0] = mean_T1[x_mean_T1_grid_ind[k]]
                        #
                        # Solve the kriging system for T2
                        if cov_model_T2 is not None:
                            try:
                                w = np.linalg.solve(
                                        mat_T2[indmat, :][:, indmat], # kriging matrix
                                        mat_T2[indmat, k], # second member
                                    )
                            except:
                                sim_ok = False
                                if verbose > 2:
                                    if logger:
                                        logger.info(f'{fname}:   ... cannot solve kriging system (for T2)')
                                    else:
                                        print(f'{fname}:   ... cannot solve kriging system (for T2)')
                                break
                            #
                            # Mean (kriged) value at x[k]
                            mu_T2_k = mean_T2[x_mean_T2_grid_ind[k]] + (v_T[indmat, 1] - mean_T2[x_mean_T2_grid_ind[indmat]]).dot(w)
                            # Standard deviation (of kriging) at x[k]
                            std_T2_k = np.sqrt(np.maximum(0, cov0_T2 - np.dot(w, mat_T2[indmat, k])))
                            # Draw value in N(mu, std^2)
                            v_T_k_new[1] = np.random.normal(loc=mu_T2_k, scale=std_T2_k)
                        else:
                            v_T_k_new[1] = mean_T2[x_mean_T2_grid_ind[k]]
                        #
                        # Compute MH quotient defined as
                        #    prob(Y[v_T_k_new] = v[k] | Y[indmat] = v[indmat], Y[t] = yt) / prob(Y[v_T[k]] = v[k] | Y[indmat] = v[indmat], Y[t] = yt)
                        # (where Y[t]=yt are the possible additional constraint)
                        #
                        # New lag from v_T_k_new and corresponding covariance for Y #################
                        h_k_new = v_T_k_new - np.vstack((v_T[:k], v_T_k_new, v_T[k+1:]))
                        cov_h_Y_k_new = cov_func_Y(h_k_new)
                        # Solve the kriging system for Y for simulation at v_T[k] and at v_T_k_new
                        indmat_ext = np.hstack((indmat, np.arange(npt, npt_ext)))
                        try:
                            w = np.linalg.solve(
                                    mat_Y[indmat_ext, :][:, indmat_ext], # kriging matrix
                                    np.vstack((mat_Y[indmat_ext, k], cov_h_Y_k_new[indmat_ext])).T # both second members
                                )
                        except:
                            sim_ok = False
                            if verbose > 2:
                                if logger:
                                    logger.info(f'{fname}:   ... cannot solve kriging system (for Y)')
                                else:
                                    print(f'{fname}:   ... cannot solve kriging system (for Y)')
                            break
                        # Mean (kriged) values at v_T[k] and v_T_k_new
                        mu_Y_k = mean_Y + (v_ext[indmat_ext] - mean_Y).dot(w) # mu_k of shape(2, )
                        # Variance (of kriging) at v_T[k] and v_T_k_new
                        var_Y_k = np.maximum(1.e-20, cov0_Y - np.array([np.dot(w[:,0], mat_Y[indmat_ext, k]), np.dot(w[:,1], cov_h_Y_k_new[indmat_ext])]))
                        # Set minimal variance to 1.e-20 to avoid division by zero
                        #
                        # MH quotient is
                        #    phi_{mean=mu_Y_k[1], var=var_Y_k[1]}(v[k]) / phi_{mean=mu_Y_k[0], var=var_Y_k[0]}(v[k])
                        # where phi_{mean, var} is the pdf of the normal law of given mean and var
                        # To avoid overflow in exp, compute log of mh quotient...
                        log_mh_quotient = 0.5 * (np.log(var_Y_k[0]) + (v[k]-mu_Y_k[0])**2/var_Y_k[0] - np.log(var_Y_k[1]) - (v[k]-mu_Y_k[1])**2/var_Y_k[1])
                        if log_mh_quotient >= 0.0 or np.random.random() < np.exp(log_mh_quotient):
                            # Accept new value v_T_new at x[k]
                            v_T[k] = v_T_k_new
                            # Update kriging matrix for Y
                            mat_Y[k,:] = cov_h_Y_k_new
                            mat_Y[:,k] = cov_h_Y_k_new
                    if not sim_ok:
                        break

                if not sim_ok:
                    continue

                # Generate T1 conditional to (x, v_T[0:npt, 0]) (one real)
                if cov_model_T1 is not None:
                    try:
                        sim_T1 = multiGaussian.multiGaussianRun(
                                cov_model_T1, dimension, spacing, origin, x=x, v=v_T[:npt, 0],
                                mode='simulation', algo=algo_T1, output_mode='array',
                                **params_T1, nreal=1, logger=logger)
                    except:
                        sim_ok = False
                        if verbose > 2:
                            if logger:
                                logger.info(f'{fname}:   ... conditional simulation of T1 failed')
                            else:
                                print(f'{fname}:   ... conditional simulation of T1 failed')
                        continue
                    # except Exception as exc:
                    #     err_msg = f'{fname}: conditional simulation of T1 failed'
                    #     if logger: logger.error(err_msg)
                    #     raise SrfError(err_msg) from exc
                else:
                    sim_T1 = params_T1['mean'].reshape(1,*dimension[::-1])
                # -> sim_T1: nd-array of shape
                #      (1, dimension) (for T1 in 1D)
                #      (1, dimension[1], dimension[0]) (for T1 in 2D)
                #      (1, dimension[2], dimension[1], dimension[0]) (for T1 in 3D)

                # Generate T2 conditional to (x, v_T[0:npt, 1]) (one real)
                if cov_model_T2 is not None:
                    try:
                        sim_T2 = multiGaussian.multiGaussianRun(
                                cov_model_T2, dimension, spacing, origin, x=x, v=v_T[:npt, 1],
                                mode='simulation', algo=algo_T2, output_mode='array',
                                **params_T2, nreal=1, logger=logger)
                    except:
                        sim_ok = False
                        if verbose > 2:
                            if logger:
                                logger.info(f'{fname}:   ... conditional simulation of T2 failed')
                            else:
                                print(f'{fname}:   ... conditional simulation of T2 failed')
                        continue
                    # except Exception as exc:
                    #     err_msg = f'{fname}: conditional simulation of T2 failed'
                    #     if logger: logger.error(err_msg)
                    #     raise SrfError(err_msg) from exc
                else:
                    sim_T2 = params_T2['mean'].reshape(1,*dimension[::-1])
                # -> sim_T2: nd-array of shape
                #      (1, dimension) (for T2 in 1D)
                #      (1, dimension[1], dimension[0]) (for T2 in 2D)
                #      (1, dimension[2], dimension[1], dimension[0]) (for T2 in 3D)

                # Set origin and dimension for Y
                origin_Y = [0.0, 0.0]
                dimension_Y = [0, 0]

                min_T1 = np.min(sim_T1)
                max_T1 = np.max(sim_T1)
                if t is not None:
                    min_T1 = min(t[:, 0].min(), min_T1)
                    max_T1 = max(t[:, 0].max(), max_T1)
                min_T1 = min_T1 - 0.5 * spacing_Y[0]
                max_T1 = max_T1 + 0.5 * spacing_Y[0]
                dimension_Y[0] = int(np.ceil((max_T1 - min_T1)/spacing_Y[0]))
                origin_Y[0] = min_T1 - 0.5*(dimension_Y[0]*spacing_Y[0] - (max_T1 - min_T1))

                min_T2 = np.min(sim_T2)
                max_T2 = np.max(sim_T2)
                if t is not None:
                    min_T2 = min(t[:, 1].min(), min_T2)
                    max_T2 = max(t[:, 1].max(), max_T2)
                min_T2 = min_T2 - 0.5 * spacing_Y[1]
                max_T2 = max_T2 + 0.5 * spacing_Y[1]
                dimension_Y[1] = int(np.ceil((max_T2 - min_T2)/spacing_Y[1]))
                origin_Y[1] = min_T2 - 0.5*(dimension_Y[1]*spacing_Y[1] - (max_T2 - min_T2))

                # Compute
                #    indc: node index of conditioning node (nearest node),
                #          rounded to lower index if between two grid node and index is positive
                indc_f = (v_T-origin_Y)/spacing_Y
                indc = indc_f.astype(int)
                indc = indc - 1 * np.all((indc == indc_f, indc > 0), axis=0)
                indc = indc[0] + indc[1] * dimension_Y[0] # single-indices

                indc_unique, indc_inv = np.unique(indc, return_inverse=True)
                if len(indc_unique) == len(indc):
                    v_T_unique = v_T
                    v_ext_unique = v_ext
                else:
                    Y_cond_aggregation = True
                    v_T_unique = np.array([v_T[indc_inv==j].mean() for j in range(len(indc_unique))])
                    v_ext_unique = np.array([v_ext[indc_inv==j].mean() for j in range(len(indc_unique))])

                # Generate Y conditional to (v_T, v_ext) (one real)
                try:
                    sim_Y = multiGaussian.multiGaussianRun(
                            cov_model_Y, dimension_Y, spacing_Y, origin_Y, x=v_T_unique, v=v_ext_unique,
                            mode='simulation', algo=algo_Y, output_mode='array',
                            **params_Y, nreal=1, logger=logger)
                except:
                    sim_ok = False
                    if verbose > 2:
                        if logger:
                            logger.info(f'{fname}:   ... conditional simulation of Y failed')
                        else:
                            print(f'{fname}:   ... conditional simulation of Y failed')
                    continue
                # except Exception as exc:
                #     err_msg = f'{fname}: conditional simulation of Y failed'
                #     if logger: logger.error(err_msg)
                #     raise SrfError(err_msg) from exc

                # -> 3d-array of shape (1, dimension_Y[1], dimension_Y[0])

            # Generate Z (one real)
            # Compute
            #    ind1, ind2: node index (nearest node),
            #                rounded to lower index if between two grid nodes and index is positive
            ind_f = (sim_T1.reshape(-1) - origin_Y[0])/spacing_Y[0]
            ind1 = ind_f.astype(int)
            ind1 = ind1 - 1 * np.all((ind1 == ind_f, ind1 > 0), axis=0)
            ind_f = (sim_T2.reshape(-1) - origin_Y[1])/spacing_Y[1]
            ind2 = ind_f.astype(int)
            ind2 = ind2 - 1 * np.all((ind2 == ind_f, ind2 > 0), axis=0)
            Z_real = np.array([sim_Y[0, jj, ii] for ii, jj in zip(ind1, ind2)])
            #Z_real = np.array([sim_Y[0, j, i] for i, j in zip(np.floor((sim_T1.reshape(-1) - origin_Y[0])/spacing_Y[0]).astype(int), np.floor((sim_T2.reshape(-1) - origin_Y[1])/spacing_Y[1]).astype(int))])
            if vmin is not None and Z_real.min() < vmin:
                sim_ok = False
                if verbose > 2:
                    if logger:
                        logger.info(f'{fname}:   ... specified minimal value not honoured')
                    else:
                        print(f'{fname}:   ... specified minimal value not honoured')
                continue
            if vmax is not None and Z_real.max() > vmax:
                sim_ok = False
                if verbose > 2:
                    if logger:
                        logger.info(f'{fname}:   ... specified maximal value not honoured')
                    else:
                        print(f'{fname}:   ... specified maximal value not honoured')
                continue

            if sim_ok:
                if Y_cond_aggregation and verbose > 0:
                    if logger:
                        logger.warning(f'{fname}: conditioning points for Y falling in a same grid cell have been aggregated (mean) (real index {ireal})')
                    else:
                        print(f'{fname}: WARNING: conditioning points for Y falling in a same grid cell have been aggregated (mean) (real index {ireal})')
                Z.append(Z_real)
                if full_output:
                    T1.append(sim_T1[0])
                    T2.append(sim_T2[0])
                    Y.append([dimension_Y, spacing_Y, origin_Y, sim_Y.reshape(dimension_Y[::-1])])
                break

    # Get Z
    if verbose > 0 and len(Z) < nreal:
        if logger:
            logger.warning(f'{fname}: some realization failed (missing)')
        else:
            print(f'{fname}: WARNING: some realization failed (missing)')

    Z = np.asarray(Z).reshape(len(Z), *np.atleast_1d(dimension)[::-1])

    if full_output:
        T1 = np.asarray(T1).reshape(len(T1), *np.atleast_1d(dimension)[::-1])
        T2 = np.asarray(T2).reshape(len(T2), *np.atleast_1d(dimension)[::-1])
        return Z, T1, T2, Y
    else:
        return Z
# ----------------------------------------------------------------------------

# # =============================================================================
# # Function to plot details of a SRF
# # =============================================================================

# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# from matplotlib.markers import MarkerStyle

# from geone import imgplot as imgplt

# # ----------------------------------------------------------------------------
# def plot_srf1D_details(im_Z, im_T, Y=None,
#                        x=None, v=None, t=None, yt=None,
#                        im_Z_display=None, im_T_display=None,
#                        plot_dens_Z=True, plot_dens_T=True,
#                        quant_Z=None, quant_T=None,
#                        col_stat_Z='green', col_stat_T='orange',
#                        col_x_in_im_Z='red', marker_x_in_im_Z='x', markersize_x_in_im_Z=75,
#                        col_x_in_im_T='red', marker_x_in_im_T='x', markersize_x_in_im_T=75,
#                        col_x_in_Y='red', marker_x_in_Y='x', markersize_x_in_Y=75,
#                        col_t_in_Y='purple', marker_t_in_Y='.', markersize_t_in_Y=100,
#                        ireal=0):
#     """
#     Displays (in the current figure) the details of one realization of a 1D SRF.

#     Three following plots are displayed:

#     - result for Z (resulting SRF)
#     - result for T (latent field, directing function)
#     - result for Y (coding process), and some statistics

#     Note: if `Y` is not given (`None`), then only the two first plots are displayed.

#     Z and T fields are displayed as 2D maps, with one cell along y-axis, and the
#     function :func:`plot_srf2D_details` is used with the same parameters except:

#     **Parameters (differing)**
#     --------------------------
#     x : 1d-array of floats, or float, optional
#         data points locations (float coordinates)
#     """
#     if x is not None:
#         x = np.vstack((x, (im_Z.oy + 0.5*im_Z.sy) * np.ones_like(x))).T

#     plot_srf2D_details(im_Z, im_T, Y=Y,
#                        x=x, v=v, t=t, yt=yt,
#                        im_Z_display=im_Z_display, im_T_display=im_T_display,
#                        plot_dens_Z=plot_dens_Z, plot_dens_T=plot_dens_T,
#                        quant_Z=quant_Z, quant_T=quant_T,
#                        col_stat_Z=col_stat_Z, col_stat_T=col_stat_T,
#                        col_x_in_im_Z=col_x_in_im_Z, marker_x_in_im_Z=marker_x_in_im_Z, markersize_x_in_im_Z=markersize_x_in_im_Z,
#                        col_x_in_im_T=col_x_in_im_T, marker_x_in_im_T=marker_x_in_im_T, markersize_x_in_im_T=markersize_x_in_im_T,
#                        col_x_in_Y=col_x_in_Y, marker_x_in_Y=marker_x_in_Y, markersize_x_in_Y=markersize_x_in_Y,
#                        col_t_in_Y=col_t_in_Y, marker_t_in_Y=marker_t_in_Y, markersize_t_in_Y=markersize_t_in_Y,
#                        ireal=ireal)
# # ----------------------------------------------------------------------------

# # ----------------------------------------------------------------------------
# def plot_srf2D_details(im_Z, im_T, Y=None,
#                        x=None, v=None,
#                        t=None, yt=None,
#                        im_Z_display=None, im_T_display=None,
#                        plot_dens_Z=True, plot_dens_T=True,
#                        quant_Z=None, quant_T=None,
#                        col_stat_Z='green', col_stat_T='orange',
#                        col_x_in_im_Z='red', marker_x_in_im_Z='x', markersize_x_in_im_Z=75,
#                        col_x_in_im_T='red', marker_x_in_im_T='x', markersize_x_in_im_T=75,
#                        col_x_in_Y='red', marker_x_in_Y='x', markersize_x_in_Y=75,
#                        col_t_in_Y='purple', marker_t_in_Y='.', markersize_t_in_Y=100,
#                        ireal=0):
#     """
#     Displays (in the current figure) the details of one realization of a 1D SRF.

#     The following plots are displayed:

#     - result for Z (resulting SRF)
#     - result for T (latent field, directing function)
#     - result for Y (coding process), and some statistics

#     Note: if `Y` is not given (`None`), then only the two first plots are displayed.

#     Parameters
#     ----------
#     im_Z : :class:`geone.img.Img`
#         image containing the realizations of Z (resulting SRF),
#         each variable is one realization

#     im_T : :class:`geone.img.Img`
#         image containing the realizations of T (latent field, directing function),
#         each variable is one realization

#     Y : list, optional
#         list containing the realizations of Y (coding process), `Y[k]` is a list of
#         length of length 4 for the k-th realization of `Y`, with:

#         - Y[k][0]: int, Y_nt (number of cell along t-axis)
#         - Y[k][1]: float, Y_st (cell size along t-axis)
#         - Y[k][2]: float, Y_ot (origin)
#         - Y[k][3]: 1d-array of shape (Y_nt,), values of Y[k]

#     x : array of floats, optional
#         data points locations (float coordinates);
#         2d-array of floats of two columns, each row being the location
#         of one conditioning point;
#         note: if only one point, a 1d-array of 2 floats is accepted

#     v : 1d-array-like of floats, optional
#         data values at `x` (`v[i]` is the data value at `x[i]`)

#     t : 1d-array-like of floats, or float, optional
#         values of T considered as conditioning point for Y(T) (additional constraint)

#     yt : 1d-array-like of floats, or float, optional
#         value of Y at the conditioning point `t` (same length as `t`)

#     im_Z_display : dict, optional
#         additional parameters for displaying im_Z (on 1st plot),
#         passed to the function :func:`geone.imgplot.drawImage2D`

#     im_T_display : dict, optional
#         additional parameters for displaying im_T (on 2nd plot)
#         passed to the function :func:`geone.imgplot.drawImage2D`

#     plot_dens_Z : bool, default: True
#         indicates if density of Z is displayed (on 3rd plot)

#     plot_dens_T : bool, default: True
#         indicates if density of T is displayed (on 3rd plot)

#     quant_Z: 1d-array of floats or float, optional
#         probability values in [0, 1] for quantiles of T to be displayed
#         (on 3rd plot), e.g.
#         `numpy.array([0., 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.])`

#     quant_T: 1d-array of floats or float, optional
#         probability values in [0, 1] for quantiles of T to be displayed
#         (on 3rd plot), e.g.
#         `numpy.array([0., 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.])`

#     col_stat_Z : color, default: 'green'
#         color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
#         displaying statistics about Z (density, quantiles) (on 3rd plot)

#     col_stat_T : color, default: 'orange'
#         color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
#         displaying statistics about T (density, quantiles) (on 3rd plot)

#     col_x_in_im_Z : color, default: 'red'
#         color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
#         plotting x locations in map of T (on 1st plot)

#     marker_x_in_im_Z : marker, default: 'x'
#         marker used for plotting x location in map of T (on 1st plot)

#     markersize_x_in_im_Z : int, default: 75
#         marker size used for plotting x location in map of T (on 1st plot)

#     col_x_in_im_T : color, default: 'red'
#         color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
#         plotting x locations in map of T (on 2nd plot)

#     marker_x_in_im_T : marker, default: 'x'
#         marker used for plotting x location in map of T (on 2nd plot)

#     markersize_x_in_im_T : int, default: 75
#         marker size used for plotting x location in map of T (on 2nd plot)

#     col_x_in_Y : color, default: 'purple'
#         color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
#         plotting t locations in map of Y (on 3rd plot)

#     marker_x_in_im_Y : marker, default: 'x'
#         marker used for t locations in map of Y (on 3rd plot)

#     markersize_x_in_im_Y : int, default: 75
#         marker size used for t locations in map of Y (on 3rd plot)

#     col_x_in_Y : color, default: 'red'
#         color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
#         plotting T(x) locations in map of Y (on 3rd plot)

#     marker_x_in_im_Y : marker, default: 'x'
#         marker used for T(x) locations in map of Y (on 3rd plot)

#     markersize_x_in_im_Y : int, default: 75
#         marker size used for T(x) locations in map of Y (on 3rd plot)

#     col_t_in_Y : color, default: 'purple'
#         color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
#         plotting t locations in map of Y (on 3rd plot)

#     marker_t_in_im_Y : marker, default: '.'
#         marker used for t locations in map of Y (on 3rd plot)

#     markersize_t_in_im_Y : int, default: 100
#         marker size used for t locations in map of Y (on 3rd plot)

#     ireal : int, default: 0
#         index of the realization to be displayed
#     """
#     # Initialize dictionary im_Z_display
#     if im_Z_display is None:
#         im_Z_display = {}

#     # Initialize dictionary im_T_display
#     if im_T_display is None:
#         im_T_display = {}

#     # Prepare figure layout
#     fig = plt.gcf()
#     fig.set_constrained_layout(True)
#     #fig = plt.figure(figsize=figsize, constrained_layout=True)

#     nr = 2 + int(Y is not None)
#     gs = GridSpec(nr, 4, figure=fig)
#     ax1 = fig.add_subplot(gs[0:2, 0:2]) # ax for T
#     ax2 = fig.add_subplot(gs[0:2, 2:4]) # ax for Z

#     if x is not None:
#         x = np.asarray(x, dtype='float').reshape(-1, 2) # cast in 2-dimensional array if needed
#         v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed

#     plt.sca(ax1)
#     imgplt.drawImage2D(im_T, iv=ireal, **im_T_display)
#     if 'title' not in im_T_display.keys():
#         plt.title('Realization of T (#{})'.format(ireal))
#     if x is not None:
#         #plt.plot(x[:,0], x[:,1], ls='', color=col_x_in_im_T, marker=marker_x_in_im_T, markersize=markersize_x_in_im_T)
#         if not isinstance(col_x_in_im_T, list):
#             col_x_in_im_T = [col_x_in_im_T]
#         if not isinstance(marker_x_in_im_T, list):
#             marker_x_in_im_T = [marker_x_in_im_T]
#         if not isinstance(markersize_x_in_im_T, list):
#             markersize_x_in_im_T = [markersize_x_in_im_T]
#         for k in range(x.shape[0]):
#             marker = marker_x_in_im_T[k%len(marker_x_in_im_T)]
#             col = col_x_in_im_T[k%len(col_x_in_im_T)]
#             markersize = markersize_x_in_im_T[k%len(markersize_x_in_im_T)]
#             if MarkerStyle(marker).is_filled():
#                 color={'c':'none', 'edgecolor':col}
#             else:
#                 color={'c':col}
#             plt.scatter(x[k,0], x[k,1], marker=marker, s=markersize, **color)

#     plt.sca(ax2)
#     imgplt.drawImage2D(im_Z, iv=ireal, **im_Z_display)#, yticklabels=[])#yaxis=False)
#     if 'title' not in im_Z_display.keys():
#         plt.title('Realization of Z (#{})'.format(ireal))

#     if x is not None:
#         #plt.plot(x[:,0], x[:,1], ls='', color=col_x_in_im_Z, marker=marker_x_in_im_Z, markersize=markersize_x_in_im_Z)
#         if not isinstance(col_x_in_im_Z, list):
#             col_x_in_im_Z = [col_x_in_im_Z]
#         if not isinstance(marker_x_in_im_Z, list):
#             marker_x_in_im_Z = [marker_x_in_im_Z]
#         if not isinstance(markersize_x_in_im_Z, list):
#             markersize_x_in_im_Z = [markersize_x_in_im_Z]
#         for k in range(x.shape[0]):
#             marker = marker_x_in_im_Z[k%len(marker_x_in_im_Z)]
#             col = col_x_in_im_Z[k%len(col_x_in_im_Z)]
#             markersize = markersize_x_in_im_Z[k%len(markersize_x_in_im_Z)]
#             if MarkerStyle(marker).is_filled():
#                 color={'c':'none', 'edgecolor':col}
#             else:
#                 color={'c':col}
#             plt.scatter(x[k,0], x[k,1], marker=marker, s=markersize, **color)

#     if Y is not None:
#         ax3 = fig.add_subplot(gs[2, :])
#         plt.sca(ax3)

#         Y_nx = Y[ireal][0]
#         Y_sx = Y[ireal][1]
#         Y_ox = Y[ireal][2]
#         Y_val = Y[ireal][3]
#         y_abscissa = Y_ox + (np.arange(Y_nx)+0.5)*Y_sx
#         plt.plot(y_abscissa, Y_val)

#         if x is not None:
#             jx = (x[:,0]-im_T.ox)/im_T.sx
#             jy = (x[:,1]-im_T.oy)/im_T.sy
#             jz = np.zeros(x.shape[0]) # (x[:,2]-im_T.oz)/im_T.sz

#             ix = [int(a) for a in jx]
#             iy = [int(a) for a in jy]
#             iz = [int(a) for a in jz]

#             # round to lower index if between two grid node
#             ix = [a-1 if a == b and a > 0 else a for a, b in zip(ix, jx)]
#             iy = [a-1 if a == b and a > 0 else a for a, b in zip(iy, jy)]
#             iz = [a-1 if a == b and a > 0 else a for a, b in zip(iz, jz)]
#             # plt.plot([im_T.val[ireal, izz, iyy, ixx] for ixx, iyy, izz in zip(ix, iy, iz)],
#             #         [im_Z.val[ireal, izz, iyy, ixx] for ixx, iyy, izz in zip(ix, iy, iz)],
#             #         ls='', color=col_x_in_Y, marker=marker_x_in_Y, markersize=markersize_x_in_Y)
#             if not isinstance(col_x_in_Y, list):
#                 col_x_in_Y = [col_x_in_Y]
#             if not isinstance(marker_x_in_Y, list):
#                 marker_x_in_Y = [marker_x_in_Y]
#             if not isinstance(markersize_x_in_Y, list):
#                 markersize_x_in_Y = [markersize_x_in_Y]
#             for k, (ixx, iyy, izz) in enumerate(zip(ix, iy, iz)):
#                 marker = marker_x_in_Y[k%len(marker_x_in_Y)]
#                 col = col_x_in_Y[k%len(col_x_in_Y)]
#                 markersize = markersize_x_in_Y[k%len(markersize_x_in_Y)]
#                 if MarkerStyle(marker).is_filled():
#                     color={'c':'none', 'edgecolor':col}
#                 else:
#                     color={'c':col}
#                 # plt.scatter(im_T.val[ireal, izz, iyy, ixx], im_Z.val[ireal, izz, iyy, ixx], marker=marker, s=markersize, **color)
#                 plt.scatter(im_T.val[ireal, izz, iyy, ixx], v[k], marker=marker, s=markersize, **color)

#         if t is not None:
#             #plt.plot(t, interp1d(y_abscissa, Y[ireal][2])(t), ls='', color=col_t_in_Y, marker=marker_t_in_Y, markersize=markersize_t_in_Y)
#             #plt.scatter(t, interp1d(y_abscissa, Y[ireal][2])(t), marker=marker_t_in_Y, s=markersize_t_in_Y, c=col_t_in_Y)
#             plt.scatter(t, yt, marker=marker_t_in_Y, s=markersize_t_in_Y, c=col_t_in_Y)

#         plt.title('Y(t)')
#         plt.grid()

#         if quant_T is not None:
#             quant_T = np.atleast_1d(quant_T)
#             tq = np.quantile(im_T.val[ireal], quant_T)
#             ypos = plt.ylim()[0] + 0.05*np.diff(plt.ylim())
#             for xx, p in zip(tq, quant_T):
#                 plt.axvline(xx, c=col_stat_T, ls='dashed', alpha=0.5)
#                 plt.text(xx, ypos, 'p={}'.format(p), ha='center', va='bottom',
#                          bbox={'facecolor':col_stat_T, 'alpha':0.2})

#         if quant_Z is not None:
#             quant_Z = np.atleast_1d(quant_Z)
#             zq = np.quantile(im_Z.val[ireal], quant_Z)
#             xpos = plt.xlim()[0] + 0.05*np.diff(plt.xlim())
#             for yy, p in zip(zq, quant_Z):
#                 plt.axhline(yy, c=col_stat_Z, ls='dashed', alpha=0.5)
#                 plt.text(xpos, yy, 'p={}'.format(p), ha='center', va='bottom',
#                          bbox={'facecolor':col_stat_Z, 'alpha':0.2})

#         if plot_dens_T:
#             ax3b = ax3.twinx()  # instantiate a second axes that shares the same x-axis
#             ax3b.set_ylabel('T density', color=col_stat_T)  # we already handled the x-label with ax3
#             ax3b.tick_params(axis='y', labelcolor=col_stat_T)
#             plt.sca(ax3b)
#             plt.hist(im_T.val[ireal].reshape(-1), density=True, bins=40, color=col_stat_T, alpha=0.2)
#             tt = np.linspace(y_abscissa.min(), y_abscissa.max(), 100)
#             plt.plot(tt, stats.gaussian_kde(im_T.val[ireal].reshape(-1))(tt), color=col_stat_T, ls='dashed', alpha=0.5)

#         if plot_dens_Z:
#             ax3c = ax3.twiny()  # instantiate a second axes that shares the same y-axis
#             ax3c.set_xlabel('Z density', color=col_stat_Z)  # we already handled the y-label with ax3
#             ax3c.tick_params(axis='x', labelcolor=col_stat_Z)
#             plt.sca(ax3c)
#             plt.hist(im_Z.val[ireal].reshape(-1), density=True, bins=40, color=col_stat_Z, alpha=0.2, orientation='horizontal')
#             z = np.linspace(im_Z.val[ireal].min(), im_Z.val[ireal].max(), 100)
#             plt.plot(stats.gaussian_kde(im_Z.val[ireal].reshape(-1))(z), z, color=col_stat_Z, ls='dashed', alpha=0.5)

#         plt.sca(ax3)
#     #plt.show()
# # ----------------------------------------------------------------------------
