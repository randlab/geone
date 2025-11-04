#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# Python module:  'multiGaussian.py'
# author:         Julien Straubhaar
# date:           may-2022
# -------------------------------------------------------------------------

"""
Module for multi-Gaussian simulation and estimation in 1D, 2D and 3D,
based on functions in other geone modules (wrapper).
"""

import numpy as np
import inspect
from geone import covModel as gcm
from geone import img
from geone import geosclassicinterface as gci
from geone import grf

# ============================================================================
class MultiGaussianError(Exception):
    """
    Custom exception related to `multiGaussian` module.
    """
    pass
# ============================================================================

# ----------------------------------------------------------------------------
def multiGaussianRun(
        cov_model,
        dimension, spacing=None, origin=None,
        mode='simulation',
        algo='fft',
        output_mode='img',
        retrieve_warnings=False,
        verbose=2,
        use_multiprocessing=False,
        logger=None,
        **kwargs):
    """
    Runs multi-Gaussian simulation or estimation.

    Wrapper of other functions, the space dimension (1, 2, or 3) is detected
    (from the parameter `dimension`), and the method is selected according to 
    the parameters `mode` and `algo`. Moreover, the type of output can be 
    specified (parameter `output_mode`).

    Parameters
    ----------
    cov_model : :class:`geone.covModel.CovModel<d>D`
        covariance model in 1D or 2D or 3D

    dimension : [sequence of] int(s)
        number of cells along each axis, for simulation in:

        - 1D: `dimension=nx`
        - 2D: `dimension=(nx, ny)`
        - 3D: `dimension=(nx, ny, nz)`

        note : this parameter determines the space dimension (1, 2, or 3)

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

    mode : str {'simulation', 'estimation'}, default: 'simulation'
        mode of computation:

        - `mode='simulation'`: generates multi-Gaussian simulations
        - `mode='estimation'`: computes multi-Gaussian estimation (and st. dev.)

    algo : str {'fft', 'classic', 'classic_old'}, default: 'fft'
        defines the algorithm used:

        - `algo='fft'`: algorithm based on circulant embedding and FFT, function \
        called for <d>D (d = 1, 2, or 3):
            - 'geone.grf.grf<d>D',   `if mode='simulation'`
            - 'geone.grf.krige<d>D', `if mode='estimation'`
        - `algo='classic'`: "classic" algorithm, based on the resolution of \
        kriging system considering points in a search ellipsoid, function called:
            - 'geone.geoscalassicinterface.simulate', `if mode='simulation'`
            - 'geone.geoscalassicinterface.estimate', `if mode='estimation'`
        - `algo='classic_old'`: "classic" algorithm (old version), \
        based on the resolution of kriging system considering points in a search \
        ellipsoid, function called for <d>D (d = 1, 2, or 3):
            - 'geone.geoscalassicinterface.simulate<d>D', `if mode='simulation'`
            - 'geone.geoscalassicinterface.estimate<d>D', `if mode='estimation'`

    output_mode : str {'array', 'img'}, default: 'img'
        defines the type of output returned (see below)

    retrieve_warnings : bool, default: False
        indicates if the warnings encountered during the run are retrieved in
        output (`True`) or not (`False`) (see below)

    verbose : int, default: 2
        verbose mode, higher implies more printing (info)

    use_multiprocessing : bool, default: False
        indicates if multiprocessing is used in the case: 
        `algo='classic_old'`, `mode='simulation'`); 
        if `use_multiprocessing=True`, the function
        `geone.geoscalassicinterface.simulate<d>D_mp` is used instead of
        `geone.geoscalassicinterface.simulate<d>D`; 
        in other cases `use_multiprocessing` is ignored

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs : dict
        keyword arguments (additional parameters) to be passed to the function
        that is called (according to `algo` and space dimension);
        note: argument `mask` can also be used with `algo='fft'`; in this case
        the mask is applied afterward

    Returns
    -------
    output : ndarray or :class:`geone.img.Img`
        - if `output_mode='array'`: output is a ndarray of shape
            * for 1D:
                * (1, nx) if `mode='estimation'` with kriging estimate only
                * (2, nx) if `mode='estimation'` with kriging estimate and st. dev.
                * (nreal, nx) if `mode='simulation'`
            * for 2D:
                * (1, ny, nx) if `mode='estimation'` with kriging estimate only
                * (2, ny, nx) if `mode='estimation'` with kriging estimate and st. dev.
                * (nreal, ny, nx) if `mode='simulation'`
            * for 3D:
                * (1, nz, ny, nx) if `mode='estimation'` with krig. est. only
                * (2, nz, ny, nx) if `mode='estimation'` with krig. est. and st. dev.
                * (nreal, nz, ny, nx) if `mode='simulation'`
        - if `output_mode='img'`: output is an instance of :class:`geone.img.Img` \
        an image with `output.nv` variables:
            - `output.nv=1` if `mode='estimation'` with kriging estimate only
            - `output.nv=2` if `mode='estimation'` with krig. est. and st. dev.
            - `output.nv=nreal` if `mode='simulation'`

    warnings : list of strs, optional
        list of distinct warnings encountered (can be empty) during the run
        (no warning (empty list) if `algo='fft'`);
        returned if `retrieve_warnings=True`
    """
    fname = 'multiGaussianRun'

    if mode.lower() not in ('simulation', 'estimation'):
        err_msg = f"{fname}: `mode` invalid, should be 'simulation' or 'estimation' (default)"
        if logger: logger.error(err_msg)
        raise MultiGaussianError(err_msg)

    if algo.lower() not in ('fft', 'classic', 'classic_old'):
        err_msg = f"{fname}: `algo` invalid, should be 'fft' (default) or 'classic' or 'classic_old'"
        if logger: logger.error(err_msg)
        raise MultiGaussianError(err_msg)

    if output_mode.lower() not in ('array', 'img'):
        err_msg = f"{fname}: `output_mode` invalid, should be 'array' or 'img' (default)"
        if logger: logger.error(err_msg)
        raise MultiGaussianError(err_msg)

    # Convert strings in lowercase
    mode = mode.lower()
    algo = algo.lower()
    output_mode = output_mode.lower()

    # Set space dimension: d
    if hasattr(dimension, '__len__'):
        d = len(dimension)
    else:
        # assume dimension is an int, nx
        d = 1

    if d not in (1, 2, 3):
        err_msg = f'{fname}: space dimension not valid'
        if logger: logger.error(err_msg)
        raise MultiGaussianError(err_msg)

    # Check (or set) argument 'spacing'
    if spacing is None:
        spacing = tuple(np.ones(d))
    else:
        if hasattr(spacing, '__len__') and len(spacing) != d:
            err_msg = f'{fname}: `spacing` of incompatible length'
            if logger: logger.error(err_msg)
            raise MultiGaussianError(err_msg)

    # Check (or set) argument 'origin'
    if origin is None:
        origin = tuple(np.zeros(d))
    else:
        if hasattr(origin, '__len__') and len(origin) != d:
            err_msg = f'{fname}: `origin` of incompatible length'
            if logger: logger.error(err_msg)
            raise MultiGaussianError(err_msg)

    # Check dimension of cov_model wrt. space dimension
    if algo == 'classic_indicator' or algo == 'classic_indicator_ind':
        cov_model_list = list(np.asarray(cov_model).reshape(-1))
        for cm in cov_model_list:
            if not isinstance(cm, gcm.CovModel1D):
                if cm.__class__.__name__ != f'CovModel{d}D':
                    err_msg = f'{fname}: `cov_model` dimension is incompatible with `dimension`'
                    if logger: logger.error(err_msg)
                    raise MultiGaussianError(err_msg)
    else:
        if not isinstance(cov_model, gcm.CovModel1D):
            if cov_model.__class__.__name__ != f'CovModel{d}D':
                err_msg = f'{fname}: `cov_model` dimension is incompatible with `dimension`'
                if logger: logger.error(err_msg)
                raise MultiGaussianError(err_msg)

    if algo == 'fft':
        if mode == 'estimation':
            if d == 1:
                run_f = grf.krige1D
            elif d == 2:
                run_f = grf.krige2D
            elif d == 3:
                run_f = grf.krige3D
        elif mode == 'simulation':
            if d == 1:
                run_f = grf.grf1D
            elif d == 2:
                run_f = grf.grf2D
            elif d == 3:
                run_f = grf.grf3D

        # Filter unused keyword arguments
        run_f_set_of_all_args = set([val.name for val in inspect.signature(run_f).parameters.values()])
        kwargs_common_keys = run_f_set_of_all_args.intersection(kwargs.keys())
        kwargs_new = {key: kwargs[key] for key in kwargs_common_keys}
        kwargs_unexpected_keys = set(kwargs.keys()).difference(run_f_set_of_all_args)
        apply_mask = False # default
        if kwargs_unexpected_keys:
            if 'mask' in kwargs_unexpected_keys:
                kwargs_unexpected_keys.remove('mask')
                if kwargs['mask'] is not None:
                    try:
                        mask = np.asarray(kwargs['mask']).reshape(np.atleast_1d(dimension)[::-1]).astype('bool')
                    except:
                        err_msg = f'{fname}: `mask` invalid'
                        if logger: logger.error(err_msg)
                        raise MultiGaussianError(err_msg)
                    
                    apply_mask = True # to apply mask afterward

            if kwargs_unexpected_keys:
                # set kwargs_unexpected_keys is not empty
                if verbose > 0:
                    s = "`, `".join(kwargs_unexpected_keys)
                    if logger:
                        logger.warning(f"{fname}: unexpected keyword arguments (`{s}`) passed to function `{run_f.__module__}.{run_f.__name__}` were ignored")
                    else:
                        print(f"{fname}: WARNING: unexpected keyword arguments (`{s}`) passed to function `{run_f.__module__}.{run_f.__name__}` were ignored")

        try:
            output = run_f(cov_model, dimension, spacing=spacing, origin=origin, verbose=verbose, logger=logger, **kwargs_new)
        except Exception as exc:
            err_msg = f'{fname}: computation failed'
            if logger: logger.error(err_msg)
            raise MultiGaussianError(err_msg) from exc

        # -> if mode = 'simulation':
        #    output is an array with (d+1) dimension (axis 0 corresponds to realizations)
        # -> if mode = 'estimation':
        #    output is an array (kriging estimate only) or a 2-tuple of array (kriging estimate and standard deviation);
        #    each array with d dimension
        if mode == 'estimation':
            if isinstance(output, tuple):
                output = np.asarray(output)
            else:
                output = output[np.newaxis, :]
        if apply_mask:
            output[:, ~mask] = np.nan
        if output_mode == 'img':
            output = img.Img(
                *np.hstack((np.atleast_1d(dimension), np.ones(3-d, dtype='int'))),
                *np.hstack((np.atleast_1d(spacing), np.ones(3-d))),
                *np.hstack((np.atleast_1d(origin), np.zeros(3-d))),
                nv=output.shape[0], val=output,
                logger=logger)
        warnings = [] # no warning available if algo = 'fft'

    elif algo in ('classic'):
        if mode == 'estimation':
            run_f = gci.estimate
        elif mode == 'simulation':
            run_f = gci.simulate

        # Filter unused keyword arguments
        run_f_set_of_all_args = set([val.name for val in inspect.signature(run_f).parameters.values()])
        kwargs_common_keys = run_f_set_of_all_args.intersection(kwargs.keys())
        kwargs_new = {key: kwargs[key] for key in kwargs_common_keys}
        kwargs_unexpected_keys = set(kwargs.keys()).difference(run_f_set_of_all_args)
        if kwargs_unexpected_keys:
            # set kwargs_unexpected_keys is not empty
            if verbose > 0:
                s = "`, `".join(kwargs_unexpected_keys)
                if logger:
                    logger.warning(f"{fname}: unexpected keyword arguments (`{s}`) passed to function `{run_f.__module__}.{run_f.__name__}` were ignored")
                else:
                    print(f"{fname}: WARNING: unexpected keyword arguments (`{s}`) passed to function `{run_f.__module__}.{run_f.__name__}` were ignored")

        try:
            output = run_f(cov_model, dimension, spacing=spacing, origin=origin, verbose=verbose, logger=logger, **kwargs_new)
        except Exception as exc:
            err_msg = f'{fname}: computation failed'
            if logger: logger.error(err_msg)
            raise MultiGaussianError(err_msg) from exc

        warnings = output['warnings']
        output = output['image']
        if output_mode == 'array':
            # get the array of value and remove extra dimension for 1D and 2D
            output = output.val.reshape(-1, *np.atleast_1d(dimension)[::-1])

    elif algo in ('classic_old'):
        if mode == 'estimation':
            if d == 1:
                run_f = gci.estimate1D
            elif d == 2:
                run_f = gci.estimate2D
            elif d == 3:
                run_f = gci.estimate3D
        elif mode == 'simulation':
            if use_multiprocessing:
                if d == 1:
                    run_f = gci.simulate1D_mp
                elif d == 2:
                    run_f = gci.simulate2D_mp
                elif d == 3:
                    run_f = gci.simulate3D_mp
            else:
                if d == 1:
                    run_f = gci.simulate1D
                elif d == 2:
                    run_f = gci.simulate2D
                elif d == 3:
                    run_f = gci.simulate3D

        # Filter unused keyword arguments
        run_f_set_of_all_args = set([val.name for val in inspect.signature(run_f).parameters.values()])
        kwargs_common_keys = run_f_set_of_all_args.intersection(kwargs.keys())
        kwargs_new = {key: kwargs[key] for key in kwargs_common_keys}
        kwargs_unexpected_keys = set(kwargs.keys()).difference(run_f_set_of_all_args)
        if kwargs_unexpected_keys:
            # set kwargs_unexpected_keys is not empty
            if verbose > 0:
                s = "', '".join(kwargs_unexpected_keys)
                if logger:
                    logger.warning(f"{fname}: unexpected keyword arguments (`{s}`) passed to function '{run_f.__module__}.{run_f.__name__}' were ignored")
                else:
                    print(f"{fname}: WARNING: unexpected keyword arguments (`{s}`) passed to function '{run_f.__module__}.{run_f.__name__}' were ignored")

        try:
            output = run_f(cov_model, dimension, spacing=spacing, origin=origin, verbose=verbose, logger=logger, **kwargs_new)
        except Exception as exc:
            err_msg = f'{fname}: computation failed'
            if logger: logger.error(err_msg)
            raise MultiGaussianError(err_msg) from exc

        warnings = output['warnings']
        output = output['image']
        if output_mode == 'array':
            # get the array of value and remove extra dimension for 1D and 2D
            output = output.val.reshape(-1, *np.atleast_1d(dimension)[::-1])

    if retrieve_warnings:
        return output, warnings
    else:
        return output
# ----------------------------------------------------------------------------
