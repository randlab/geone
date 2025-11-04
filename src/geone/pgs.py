#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# Python module:  'pgs.py'
# author:         Julien Straubhaar
# date:           may-2022
# -------------------------------------------------------------------------

"""
Module for plurig-Gaussian simulations in 1D, 2D and 3D.
"""

import numpy as np
from geone import covModel as gcm
from geone import multiGaussian

# ============================================================================
class PgsError(Exception):
    """
    Custom exception related to `pgs` module.
    """
    pass
# ============================================================================

# ----------------------------------------------------------------------------
def pluriGaussianSim_unconditional(
        cov_model_T1, cov_model_T2, flag_value,
        dimension, spacing=None, origin=None,
        algo_T1='fft', params_T1={},
        algo_T2='fft', params_T2={},
        nreal=1,
        full_output=True,
        verbose=1,
        logger=None):
    """
    Generates unconditional pluri-Gaussian simulations.

    The simulated variable Z at a point x is defined as

    * Z(x) = flag_value(T1(x), T2(x))

    where

    * T1, T2 are two multi-Gaussian random fields (latent fields)
    * `flag_value` is a function of two variables defining the final value \
    (given as a "flag")

    Z and T1, T2 are fields in 1D, 2D or 3D.

    Parameters
    ----------
    cov_model_T1 : :class:`geone.covModel.CovModel<d>D`
        covariance model for T1, in 1D or 2D or 3D (same space dimension for T1 and T2);
        note: if `algo_T1='deterministic'`, `cov_model_T1` can be `None` (unused)

    cov_model_T2 : :class:`geone.covModel.CovModel<d>D`
        covariance model for T2, in 1D or 2D or 3D (same space dimension for T1 and T2);
        note: if `algo_T2='deterministic'`, `cov_model_T2` can be `None` (unused)

    flag_value : function (`callable`)
        function of tow arguments (xi, yi) that returns the "flag_value" at
        location (xi, yi)

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

    algo_T1 : str {'fft', 'classic', 'deterministic'}, default: 'fft'
        defines the algorithm used for T1:

        - 'fft': algorithm based on circulant embedding and FFT, function \
        called for <d>D (d = 1, 2, or 3): 'geone.grf.grf<d>D'
        - 'classic': "classic" algorithm, based on the resolution of \
        kriging system considered points in a search ellipsoid, function called \
        for <d>D (d = 1, 2, or 3): \
        'geone.geoscalassicinterface.simulate<d>D'
        - 'deterministic': use a deterministic field, given by `param_T1['mean']`

    algo_T2 : str {'fft', 'classic', 'deterministic'}, default: 'fft'
        defines the algorithm used for T2 (see `algo_T1` for detail)

    params_T1 : dict
        keyword arguments (additional parameters) to be passed to the function
        that is called (according to `algo_T1` and space dimension) for simulation
        of T1

    params_T2 : dict
        keyword arguments (additional parameters) to be passed to the function
        that is called (according to `algo_T2` and space dimension) for simulation
        of T2

    nreal : int, default: 1
        number of realization(s)

    full_output : bool, default: True
        - if `True`: simulation(s) of Z, T1, and T2 are retrieved in output
        - if `False`: simulation(s) of Z only is retrieved in output

    verbose : int, default: 1
        verbose mode, higher implies more printing (info)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    Z : ndarray
        array of shape

        - for 1D: (nreal, nx)
        - for 2D: (nreal, ny, nx)
        - for 3D: (nreal, nz, ny, nx)

        Z[k] is the k-th realization of Z

    T1 : ndarray, optional
        array of shape

        - for 1D: (nreal, nx)
        - for 2D: (nreal, ny, nx)
        - for 3D: (nreal, nz, ny, nx)

        T1[k] is the k-th realization of T1;
        returned if `full_output=True`

    T2 : ndarray, optional
        array of shape

        - for 1D: (nreal, nx)
        - for 2D: (nreal, ny, nx)
        - for 3D: (nreal, nz, ny, nx)

        T2[k] is the k-th realization of T2;
        returned if `full_output=True`
    """
    fname = 'pluriGaussianSim_unconditional'

    if not callable(flag_value):
        err_msg = f'{fname}: `flag_value` invalid, should be a function (callable) of two arguments'
        if logger: logger.error(err_msg)
        raise PgsError(err_msg)

    if algo_T1 not in ('fft', 'FFT', 'classic', 'CLASSIC', 'deterministic', 'DETERMINISTIC'):
        err_msg = f"{fname}: `algo_T1` invalid, should be 'fft' (default) or 'classic' or 'deterministic'"
        if logger: logger.error(err_msg)
        raise PgsError(err_msg)

    if algo_T2 not in ('fft', 'FFT', 'classic', 'CLASSIC', 'deterministic', 'DETERMINISTIC'):
        err_msg = f"{fname}: `algo_T2` invalid, should be 'fft' (default) or 'classic' or 'deterministic'"
        if logger: logger.error(err_msg)
        raise PgsError(err_msg)

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
            raise PgsError(err_msg)

    elif isinstance(cov_model_T1, gcm.CovModel1D):
        d = 1
    elif isinstance(cov_model_T1, gcm.CovModel2D):
        d = 2
    elif isinstance(cov_model_T1, gcm.CovModel3D):
        d = 3
    else:
        err_msg = f'{fname}: `cov_model_T1` invalid, should be a class `geone.covModel.CovModel1D`, `geone.covModel.CovModel2D` or `geone.covModel.CovModel3D`'
        if logger: logger.error(err_msg)
        raise PgsError(err_msg)

    if cov_model_T2 is None:
        if algo_T2 not in ('deterministic', 'DETERMINISTIC'):
            err_msg = f"{fname}: `cov_model_T2` is `None`, then `algo_T2` must be 'deterministic'"
            if logger: logger.error(err_msg)
            raise PgsError(err_msg)

        # if d == 0:
        #     err_msg = f'{fname}: `cov_model_T1` and `cov_model_T2` are `None`, at least one covariance model is required'
        #     if logger: logger.error(err_msg)
        #     raise PgsError(err_msg)

    elif (d == 1 and not isinstance(cov_model_T2, gcm.CovModel1D)) or (d == 2 and not isinstance(cov_model_T2, gcm.CovModel2D)) or (d == 3 and not isinstance(cov_model_T2, gcm.CovModel3D)):
        err_msg = f'{fname}: `cov_model_T1` and `cov_model_T2` not compatible (dimensions differ)'
        if logger: logger.error(err_msg)
        raise PgsError(err_msg)

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
        raise PgsError(err_msg)

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
            raise PgsError(err_msg)

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
            raise PgsError(err_msg)

#    if not cov_model_T1.is_stationary(): # prevent calculation if covariance model is not stationary
#         if verbose > 0:
#             print(f"ERROR ({fname}): `cov_model_T1` is not stationary")

#    if not cov_model_T2.is_stationary(): # prevent calculation if covariance model is not stationary
#         if verbose > 0:
#             print(f"ERROR ({fname}): `cov_model_T2` is not stationary")

    # Set default parameter 'verbose' for params_T1, params_T2
    if 'verbose' not in params_T1.keys():
        params_T1['verbose'] = 0
        # params_T1['verbose'] = verbose
    if 'verbose' not in params_T2.keys():
        params_T2['verbose'] = 0
        # params_T2['verbose'] = verbose

    # Generate T1
    if cov_model_T1 is not None:
        try:
            sim_T1 = multiGaussian.multiGaussianRun(
                    cov_model_T1, dimension, spacing, origin,
                    mode='simulation', algo=algo_T1, output_mode='array',
                    **params_T1, nreal=nreal)
        except Exception as exc:
            err_msg = f'{fname}: simulation of T1 failed'
            if logger: logger.error(err_msg)
            raise PgsError(err_msg) from exc

    else:
        sim_T1 = np.array([params_T1['mean'].reshape(1,*dimension[::-1]) for _ in range(nreal)])
    # -> sim_T1: nd-array of shape
    #      (nreal_T, dimension) (for T1 in 1D)
    #      (nreal_T, dimension[1], dimension[0]) (for T1 in 2D)
    #      (nreal_T, dimension[2], dimension[1], dimension[0]) (for T1 in 3D)
    #
    # Generate T2
    if cov_model_T2 is not None:
        try:
            sim_T2 = multiGaussian.multiGaussianRun(
                    cov_model_T2, dimension, spacing, origin,
                    mode='simulation', algo=algo_T2, output_mode='array',
                    **params_T2, nreal=nreal)
        except Exception as exc:
            err_msg = f'{fname}: simulation of T2 failed'
            if logger: logger.error(err_msg)
            raise PgsError(err_msg) from exc
    else:
        sim_T2 = np.array([params_T2['mean'].reshape(1,*dimension[::-1]) for _ in range(nreal)])
    # -> sim_T2: nd-array of shape
    #      (nreal_T, dimension) (for T2 in 1D)
    #      (nreal_T, dimension[1], dimension[0]) (for T2 in 2D)
    #      (nreal_T, dimension[2], dimension[1], dimension[0]) (for T2 in 3D)

    # Generate Z
    if verbose > 1:
        if logger:
            logger.info(f'{fname}: retrieving Z...')
        else:
            print(f'{fname}: retrieving Z...')
    Z = flag_value(sim_T1, sim_T2)
    # Z = np.asarray(Z).reshape(len(Z), *np.atleast_1d(dimension)[::-1])

    if full_output:
        return Z, sim_T1, sim_T2
    else:
        return Z
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def pluriGaussianSim(
        cov_model_T1, cov_model_T2, flag_value,
        dimension, spacing=None, origin=None,
        x=None, v=None,
        algo_T1='fft', params_T1={},
        algo_T2='fft', params_T2={},
        accept_init=0.25, accept_pow=2.0,
        mh_iter_min=100, mh_iter_max=200,
        ntry_max=1,
        retrieve_real_anyway=False,
        nreal=1,
        full_output=True,
        verbose=1,
        logger=None):
    """
    Generates (conditional) pluri-Gaussian simulations.

    The simulated variable Z at a point x is defined as

    * Z(x) = flag_value(T1(x), T2(x))

    where

    * T1, T2 are two multi-Gaussian random fields (latent fields)
    * `flag_value` is a function of two variables defining the final value \
    (given as a "flag")

    Z and T1, T2 are fields in 1D, 2D or 3D.

    Parameters
    ----------
    cov_model_T1 : :class:`geone.covModel.CovModel<d>D`
        covariance model for T1, in 1D or 2D or 3D (same space dimension for T1 and T2);
        note: if `algo_T1='deterministic'`, `cov_model_T1` can be `None` (unused)

    cov_model_T2 : :class:`geone.covModel.CovModel<d>D`
        covariance model for T2, in 1D or 2D or 3D (same space dimension for T1 and T2);
        note: if `algo_T2='deterministic'`, `cov_model_T2` can be `None` (unused)

    flag_value : function (`callable`)
        function of tow arguments (xi, yi) that returns the "flag_value" at
        location (xi, yi)

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

    x : array-like of floats, optional
        data points locations (float coordinates), for simulation in:

        - 1D: 1D array-like of floats
        - 2D: 2D array-like of floats of shape (n, 2)
        - 3D: 2D array-like of floats of shape (n, 3)

        note: if one point (n=1), a float in 1D, a 1D array of shape (2,) in 2D,
        a 1D array of shape (3,) in 3D, is accepted

    v : 1D array-like of floats, optional
        data values at `x` (`v[i]` is the data value at `x[i]`)

    algo_T1 : str {'fft', 'classic', 'deterministic'}, default: 'fft'
        defines the algorithm used for T1:

        - 'fft': algorithm based on circulant embedding and FFT, function \
        called for <d>D (d = 1, 2, or 3): 'geone.grf.grf<d>D'
        - 'classic': "classic" algorithm, based on the resolution of \
        kriging system considered points in a search ellipsoid, function called \
        for <d>D (d = 1, 2, or 3): \
        'geone.geoscalassicinterface.simulate<d>D'
        - 'deterministic': use a deterministic field, given by `param_T1['mean']`

    algo_T2 : str {'fft', 'classic', 'deterministic'}, default: 'fft'
        defines the algorithm used for T2 (see `algo_T1` for detail)

    params_T1 : dict
        keyword arguments (additional parameters) to be passed to the function
        that is called (according to `algo_T1` and space dimension) for simulation
        of T1

    params_T2 : dict
        keyword arguments (additional parameters) to be passed to the function
        that is called (according to `algo_T2` and space dimension) for simulation
        of T2

    accept_init : float, default: 0.25
        initial acceptation probability
        (see parameters `mh_iter_min`, `mh_iter_max`)

    accept_pow : float, default: 2.0
        power for computing acceptation probability
        (see parameters `mh_iter_min`, `mh_iter_max`)

    mh_iter_min : int, default: 100
        see parameter `mh_iter_max`

    mh_iter_max : int, default: 200
        `mh_iter_min` and `mh_iter_max` are the number of iterations
        (min and max) for Metropolis-Hasting algorithm
        (for conditional simulation) when updating T1 and T2 at conditioning
        locations at iteration `nit` (in 0, ..., `mh_iter_max-1`):

        * if `nit < mh_iter_min`: for any k:
            - simulate new candidate at `x[k]`: `(T1(x[k]), T2(x[k]))`
            - if `flag_value(T1(x[k]), T2(x[k])=v[k]` (conditioning ok): \
            accept the new candidate
            - else (conditioning not ok): \
            accept the new candidate with probability
                * p = `accept_init * (1 - 1/mh_iter_min)**accept_pow`

        * if nit >= mh_iter_min:
            - if conditioning ok at every `x[k]`: stop and exit the loop,
            - else: for any k:
                - if conditioning ok at `x[k]`: skip
                - else:
                    * simulate new candidate at `x[k]`: `(T1(x[k]), T2(x[k]))`
                    * if `flag_value(T1(x[k]), T2(x[k])=v[k]` (conditioning ok): \
                    accept the new candidate
                    * else (conditioning not ok): \
                    reject the new candidate

    ntry_max : int, default: 1
        number of trial(s) per realization before giving up if something goes
        wrong

    retrieve_real_anyway : bool, default: False
        if after `ntry_max` trial(s) a conditioning data is not honoured, then
        the realization is:

        - retrieved, if `retrieve_real_anyway=True`
        - not retrieved (missing realization), if `retrieve_real_anyway=False`

    nreal : int, default: 1
        number of realization(s)

    full_output : bool, default: True
        - if `True`: simulation(s) of Z, T1, T2, and `n_cond_ok` are \
        retrieved in output
        - if `False`: simulation(s) of Z only is retrieved in output

    verbose : int, default: 1
        verbose mode, higher implies more printing (info)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    Z : ndarray
        array of shape

        - for 1D: (nreal, nx)
        - for 2D: (nreal, ny, nx)
        - for 3D: (nreal, nz, ny, nx)

        Z[k] is the k-th realization of Z

    T1 : ndarray, optional
        array of shape

        - for 1D: (nreal, nx)
        - for 2D: (nreal, ny, nx)
        - for 3D: (nreal, nz, ny, nx)

        T1[k] is the k-th realization of T1;
        returned if `full_output=True`

    T2 : ndarray, optional
        array of shape

        - for 1D: (nreal, nx)
        - for 2D: (nreal, ny, nx)
        - for 3D: (nreal, nz, ny, nx)

        T2[k] is the k-th realization of T2;
        returned if `full_output=True`

    n_cond_ok : list of 1D array
        list of length `nreal`

        - n_cond_ok[k]: 1D array of ints
            number of conditioning locations honoured at each iteration of the
            Metropolis-Hasting algorithm for the k-th realization, in particular
            `len(n_cond_ok[k])` is the number of iteration done,
            `n_cond_ok[k][-1]` is the number of conditioning locations honoured
            at the end;

        returned if `full_output=True`
    """
    fname = 'pluriGaussianSim'

    if not callable(flag_value):
        err_msg = f'{fname}: `flag_value` invalid, should be a function (callable) of two arguments'
        if logger: logger.error(err_msg)
        raise PgsError(err_msg)

    if algo_T1 not in ('fft', 'FFT', 'classic', 'CLASSIC', 'deterministic', 'DETERMINISTIC'):
        err_msg = f"{fname}: `algo_T1` invalid, should be 'fft' (default) or 'classic' or 'deterministic'"
        if logger: logger.error(err_msg)
        raise PgsError(err_msg)

    if algo_T2 not in ('fft', 'FFT', 'classic', 'CLASSIC', 'deterministic', 'DETERMINISTIC'):
        err_msg = f"{fname}: `algo_T2` invalid, should be 'fft' (default) or 'classic' or 'deterministic'"
        if logger: logger.error(err_msg)
        raise PgsError(err_msg)

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
            raise PgsError(err_msg)

    elif isinstance(cov_model_T1, gcm.CovModel1D):
        d = 1
    elif isinstance(cov_model_T1, gcm.CovModel2D):
        d = 2
    elif isinstance(cov_model_T1, gcm.CovModel3D):
        d = 3
    else:
        err_msg = f'{fname}: `cov_model_T1` invalid, should be a class `geone.covModel.CovModel1D`, `geone.covModel.CovModel2D` or `geone.covModel.CovModel3D`'
        if logger: logger.error(err_msg)
        raise PgsError(err_msg)

    if cov_model_T2 is None:
        if algo_T2 not in ('deterministic', 'DETERMINISTIC'):
            err_msg = f"{fname}: `cov_model_T2` is `None`, then `algo_T2` must be 'deterministic'"
            if logger: logger.error(err_msg)
            raise PgsError(err_msg)

        # if d == 0:
        #     err_msg = f'{fname}: `cov_model_T1` and `cov_model_T2` are `None`, at least one covariance model is required'
        #     if logger: logger.error(err_msg)
        #     raise PgsError(err_msg)

    elif (d == 1 and not isinstance(cov_model_T2, gcm.CovModel1D)) or (d == 2 and not isinstance(cov_model_T2, gcm.CovModel2D)) or (d == 3 and not isinstance(cov_model_T2, gcm.CovModel3D)):
        err_msg = f'{fname}: `cov_model_T1` and `cov_model_T2` not compatible (dimensions differ)'
        if logger: logger.error(err_msg)
        raise PgsError(err_msg)

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
        raise PgsError(err_msg)

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
            raise PgsError(err_msg)

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
            raise PgsError(err_msg)

#    if not cov_model_T1.is_stationary(): # prevent calculation if covariance model is not stationary
#         if verbose > 0:
#             print(f"ERROR ({fname}): `cov_model_T1` is not stationary")

#    if not cov_model_T2.is_stationary(): # prevent calculation if covariance model is not stationary
#         if verbose > 0:
#             print(f"ERROR ({fname}): `cov_model_T2` is not stationary")

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
                raise PgsError(err_msg)

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
                    raise PgsError(err_msg)

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
                raise PgsError(err_msg)

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
                    raise PgsError(err_msg)

    # Note: format of data (x, v) not checked !

    if x is None:
        if v is not None:
            err_msg = f'{fname}: `x` is not given (`None`) but `v` is given (not `None`)'
            if logger: logger.error(err_msg)
            raise PgsError(err_msg)

    else:
        # Preparation for conditional case
        if v is None:
            err_msg = f'{fname}: `x` is given (not `None`) but `v` is not given (`None`)'
            if logger: logger.error(err_msg)
            raise PgsError(err_msg)

        x = np.asarray(x, dtype='float').reshape(-1, d) # cast in d-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise PgsError(err_msg)

        # Compute
        #    indc: node index of conditioning node (nearest node),
        #          rounded to lower index if between two grid node and index is positive
        indc_f = (x-origin)/spacing
        indc = indc_f.astype(int)
        indc = indc - 1 * np.all((indc == indc_f, indc > 0), axis=0)
        if d == 1:
            indc = 1 * indc[:, 0] # multiply by 1.0 makes a copy of the array !
        elif d == 2:
            indc = indc[:, 0] + dimension[0] * indc[:, 1]
        elif d == 3:
            indc = indc[:, 0] + dimension[0] * (indc[:, 1] + dimension[1] * indc[:, 2])
        indc_unique, indc_inv = np.unique(indc, return_inverse=True)
        if len(indc_unique) != len(x):
            if np.any([len(np.unique(v[indc_inv==j])) > 1 for j in range(len(indc_unique))]):
                err_msg = f'{fname}: more than one conditioning point fall in a same grid cell and have different conditioning values'
                if logger: logger.error(err_msg)
                raise PgsError(err_msg)

            else:
                if verbose > 0:
                    if logger:
                        logger.warning(f'{fname}: more than one conditioning point fall in a same grid cell with same conditioning value (consistent)')
                    else:
                        print(f'{fname}: WARNING: more than one conditioning point fall in a same grid cell with same conditioning value (consistent)')
                x = np.array([x[indc_inv==j][0] for j in range(len(indc_unique))])
                v = np.array([v[indc_inv==j][0] for j in range(len(indc_unique))])

        # Number of conditioning points
        npt = x.shape[0]
        #
        # Get index in mean_T1 for each conditioning point
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

        # Get index in var_T1 (if not None) for each conditioning point
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

        # Get index in mean_T2 for each conditioning point
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

        # Get index in var_T2 (if not None) for each conditioning point
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

    # Set (again if given) default parameter 'mean' and 'var' for T1, T2
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

    # Set default parameter 'verbose' for params_T1, params_T2
    if 'verbose' not in params_T1.keys():
        params_T1['verbose'] = 0
        # params_T1['verbose'] = verbose
    if 'verbose' not in params_T2.keys():
        params_T2['verbose'] = 0
        # params_T2['verbose'] = verbose

    # Initialization for output
    Z = []
    if full_output:
        T1 = []
        T2 = []
        n_cond_ok = []

    for ireal in range(nreal):
        # Generate ireal-th realization
        if verbose > 1:
            if logger:
                logger.info(f'{fname}: simulation {ireal+1} of {nreal}...')
            else:
                print(f'{fname}: simulation {ireal+1} of {nreal}...')
        for ntry in range(ntry_max):
            sim_ok = True
            nhd_ok = [] # to be appended for full output...
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
                                **params_T1, nreal=1)
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
                    #     raise PgsError(err_msg) from exc

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
                                **params_T2, nreal=1)
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
                    #     raise PgsError(err_msg) from exc

                else:
                    sim_T2 = params_T2['mean'].reshape(1,*dimension[::-1])
                # -> sim_T2: nd-array of shape
                #      (1, dimension) (for T2 in 1D)
                #      (1, dimension[1], dimension[0]) (for T2 in 2D)
                #      (1, dimension[2], dimension[1], dimension[0]) (for T2 in 3D)

            else:
                # Conditional case
                # ----------------
                v_T = np.zeros((npt, 2))
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
                    if verbose > 2:
                        if logger:
                            logger.info(f'{fname}:    ... cannot solve kriging system (for T2, initialization)')
                        else:
                            print(f'{fname}:    ... cannot solve kriging system (for T2, initialization)')
                    continue

                # Update simulated values v_T at x using Metropolis-Hasting (MH) algorithm
                v_T_k_new = np.zeros(2)
                stop_mh = False
                for nit in range(mh_iter_max):
                    #hd_ok = np.array([flag_value(v_T[k, 0], v_T[k, 1]) == v[k] for k in range(npt)])
                    hd_ok = flag_value(v_T[:, 0], v_T[:, 1]) == v
                    nhd_ok.append(np.sum(hd_ok))
                    if nit >= mh_iter_min:
                        if nhd_ok[-1] == npt:
                            stop_mh = True
                            break
                    else:
                        # Set acceptation probability for bad case
                        p_accept = accept_init * np.power(1.0 - nit/mh_iter_min, accept_pow)
                    if verbose > 3:
                        if logger:
                            logger.info(f'{fname}:   ... sim {ireal+1} of {nreal}: MH iter {nit+1} of {mh_iter_min},  {mh_iter_max}...')
                        else:
                            print(f'{fname}:   ... sim {ireal+1} of {nreal}: MH iter {nit+1} of {mh_iter_min},  {mh_iter_max}...')
                    ind = np.random.permutation(npt)
                    for k in ind:
                        if nit >= mh_iter_min and hd_ok[k]:
                           #print('skip')
                           continue
                        #
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
                        # Accept or not the new candidate
                        if flag_value(v_T_k_new[0], v_T_k_new[1]) == v[k]:
                            # Accept the new candidate
                            v_T[k] = v_T_k_new
                        elif nit < mh_iter_min and np.random.random() < p_accept:
                            # Accept the new candidate
                            v_T[k] = v_T_k_new

                    if not sim_ok:
                        break

                if not sim_ok:
                    continue

                if not stop_mh:
                    hd_ok = flag_value(v_T[:, 0], v_T[:, 1]) == v
                    nhd_ok.append(np.sum(hd_ok))
                if nhd_ok[-1] != npt:
                    # sim_ok kept to True
                    if verbose > 2:
                        if logger:
                            logger.info(f'{fname}:   ... conditioning failed')
                        else:
                            print(f'{fname}:   ... conditioning failed')

                    if ntry < ntry_max - 1 or not retrieve_real_anyway:
                        continue

                # Generate T1 conditional to (x, v_T[:, 0]) (one real)
                if cov_model_T1 is not None:
                    try:
                        sim_T1 = multiGaussian.multiGaussianRun(
                                cov_model_T1, dimension, spacing, origin, x=x, v=v_T[:, 0],
                                mode='simulation', algo=algo_T1, output_mode='array',
                                **params_T1, nreal=1)
                    except:
                        sim_ok = False
                        if verbose > 2:
                            if logger:
                                logger.info(f'{fname}:   ... conditional simulation of T1 failed')
                            else:
                                print(f'{fname}:   ... conditional simulation of T1 failed')
                        continue

                else:
                    sim_T1 = params_T1['mean'].reshape(1,*dimension[::-1])
                # -> sim_T1: nd-array of shape
                #      (1, dimension) (for T1 in 1D)
                #      (1, dimension[1], dimension[0]) (for T1 in 2D)
                #      (1, dimension[2], dimension[1], dimension[0]) (for T1 in 3D)

                # Generate T2 conditional to (x, v_T[:, 1]) (one real)
                if cov_model_T2 is not None:
                    try:
                        sim_T2 = multiGaussian.multiGaussianRun(
                                cov_model_T2, dimension, spacing, origin, x=x, v=v_T[:, 1],
                                mode='simulation', algo=algo_T2, output_mode='array',
                                **params_T2, nreal=1)
                    except:
                        sim_ok = False
                        if verbose > 2:
                            if logger:
                                logger.info(f'{fname}:   ... conditional simulation of T2 failed')
                            else:
                                print(f'{fname}:   ... conditional simulation of T2 failed')
                        continue
                else:
                    sim_T2 = params_T2['mean'].reshape(1,*dimension[::-1])
                # -> sim_T2: nd-array of shape
                #      (1, dimension) (for T2 in 1D)
                #      (1, dimension[1], dimension[0]) (for T2 in 2D)
                #      (1, dimension[2], dimension[1], dimension[0]) (for T2 in 3D)

            # Generate Z (one real)
            if sim_ok:
                if x is not None:
                    if nhd_ok[-1] != npt:
                        if not retrieve_real_anyway:
                            break
                        else:
                            if verbose > 0:
                                if logger:
                                    logger.warning(f'{fname}: realization does not honoured all data, but retrieved anyway')
                                else:
                                    print(f'{fname}: WARNING: realization does not honoured all data, but retrieved anyway')
                Z_real = flag_value(sim_T1[0], sim_T2[0])
                Z.append(Z_real)
                if full_output:
                    T1.append(sim_T1[0])
                    T2.append(sim_T2[0])
                    n_cond_ok.append(np.asarray(nhd_ok))
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
        return Z, T1, T2, n_cond_ok
    else:
        return Z
# ----------------------------------------------------------------------------
