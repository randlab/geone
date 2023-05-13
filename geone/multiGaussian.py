#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python module:  'multiGaussian.py'
author:         Julien Straubhaar
date:           may-2022

Module for multi-Gaussian simulation and estimation in 1D, 2D and 3D,
based on functions in other geone modules.
"""

import numpy as np
import inspect
from geone import covModel as gcm
from geone import img
from geone import geosclassicinterface as gci
from geone import grf

# ----------------------------------------------------------------------------
def multiGaussianRun(
    cov_model,
    dimension, spacing=None, origin=None,
    x=None, v=None,
    mode='simulation',
    algo='fft',
    output_mode='img',
    retrieve_warnings=False,
    verbose=1,
    use_multiprocessing=False,
    **kwargs):
    """
    Runs multi-Gaussian simulation or estimation (according to the argument
    'mode', see below) in 1D, 2D or 3D, using other function (according to
    the argument 'algo', see below).

    :param cov_model:   (CovModel1D or CovModel2D or CovModel3D class)
                            covariance model in 1D or 2D or 3D, see definition of
                            the class in module geone.covModel
    :param dimension:   number of cells along each axis,
                        for simulation in
                            - 1D: (int, or sequence of 1 int): nx
                            - 2D: (sequence of 2 ints): (nx, ny)
                            - 3D: (sequence of 3 ints): (nx, ny, nz)
    :param spacing:     spacing between two adjacent cells along each axis,
                        for simulation in
                            - 1D: (float, or sequence of 1 float): sx
                            - 2D: (sequence of 2 floats): (sx, sy)
                            - 3D: (sequence of 3 floats): (sx, sy, sz)
                            (if None, set to 1.0 along each axis)
    :param origin:      origin of the simulation grid (corner of first grid
                        cell), for simulation in
                            - 1D: (float, or sequence of 1 float): ox
                            - 2D: (sequence of 2 floats): (ox, oy)
                            - 3D: (sequence of 3 floats): (ox, oy, oz)
                            (if None, set to 0.0 along each axis)
    :param x:           coordinate of data points,
                        for simulation in
                            - 1D: (1-dimensional array or float)
                            - 2D: (2-dimensional array of dim n x 2, or
                                   1-dimensional array of dim 2)
                            - 3D: (2-dimensional array of dim n x 3, or
                                   1-dimensional array of dim 3)
                            (None if no data)
    :param v:           value at data points,
                        for simulation in
                            - 1D: (1-dimensional array or float)
                            - 2D: (1-dimensional array of length n)
                            - 3D: (1-dimensional array of length n)
                            (None if no data)
    :param mode:        (str) mode of computation, can be 'estimation' or
                            'simulation' (default):
                            - 'simulation': generates multi-Gaussian simulations
                            - 'estimation': computes multi-Gaussian estimation
    :param algo:        (str) defines the algorithm used:
                            - 'fft' or 'FFT' (default): based on circulant
                                embedding and FFT, function called for <d>D
                                (d = 1, 2, or 3):
                                - 'geone.grf.grf<d>D' if 'mode' = 'simulation'
                                - 'geone.grf.krige<d>D' if 'mode' = 'estimation'
                            - 'classic' or 'CLASSIC': classic algorithm, based on
                                the resolution of kriging system considered points
                                in a search ellipsoid, function called for <d>D
                                (d = 1, 2, or 3):
                                - 'geone.geoscalassicinterface.simulate<d>D'
                                    if 'mode' = 'simulation'
                                - 'geone.geoscalassicinterface.estimate<d>D'
                                    if 'mode' = 'estimation'
    :param output_mode: (str) indicates the output mode, can be 'array' or 'img'
                            (default), see 'return' below
    :param retrieve_warnings:
                        (bool) indicates if the possible warnings are retrieved
                            in output see 'return' below

    :param verbose:     (int) verbose mode, integer >=0, higher implies more
                            display

    :param use_multiprocessing:
                        (bool) indicates if multiprocessing is used:
                            - multiprocessing can be used only with
                                 mode = 'simulation' and algo = 'classic'
                              in any other case use_multiprocessing is ignored
                            - with mode = 'simulation' and algo = 'classic', if
                              use_multiprocessing is True, then simulations are
                              generated through multiple processes, i.e. function
                                 geone.geoscalassicinterface.simulate<d>D_mp
                              is used instead of
                                 geone.geoscalassicinterface.simulate<d>D

    :param kwargs:      (dict) keyword arguments (additional parameters) to be
                            passed to the function corresponding to what is
                            specified by the argument 'algo' (see the
                            corresponding function for its keyword arguments)

    :return:    depends on 'output_mode' and 'retrieve_warnings',
                    - if retrieve_warnings is False: return output
                    - if retrieve_warnings is True: return (output, warnings)
                where:
                output:
                - if output_mode = 'array':
                    (nd-array) array of shape:
                    - for 1D:
                    (1, nx) for mode = 'estimation' with krig. estimate only
                    (2, nx) for mode = 'estimation' with krig. estimate and std
                    (nreal, nx) for mode = 'simulation' (nreal realization(s))
                    - for 2D:
                    (1, ny, nx) for mode = 'estimation' with krig. est. only
                    (2, ny, nx) for mode = 'estimation' with krig. est. and std
                    (nreal, ny, nx) for mode = 'simulation' (nreal real.)
                    - for 3D:
                    (1, nz, ny, nx) for mode = 'estimation' with krig. est. only
                    (2, nz, ny, nx) for mode = 'estimation' with krig. est.
                                                                        and std
                    (nreal, nz, ny, nx) for mode = 'simulation' (nreal real.)
                - if output_mode = 'img':
                    (Img (class)) image, with output.nv variables:
                    - output.nv = 1, for mode = 'estimation' with krig. est. only
                    - output.nv = 2, for mode = 'estimation' with krig. est.
                                                                        and std
                    - output.nv = nreal, for mode = 'simulation' (nreal real.))
                warnings:
                    (list of strings) list of distinct warnings encountered
                        (can be empty) (get no warnings if 'algo' = 'fft' or
                        'FFT')
    """
    if retrieve_warnings:
        out = None, None
    else:
        out = None

    if mode not in ('simulation', 'estimation'):
        print("ERROR (MULTIGAUSSIANRUN): 'mode' invalid, should be 'simulation' or 'estimation' (default)")
        return out

    if algo not in ('fft', 'FFT', 'classic', 'CLASSIC'):
        print("ERROR (MULTIGAUSSIANRUN): 'algo' invalid, should be 'fft' (default) or 'classic'")
        return out

    if output_mode not in ('array', 'img'):
        print("ERROR (MULTIGAUSSIANRUN): 'output_mode' invalid, should be 'array' or 'img' (default)")
        return out

    # Set space dimension: d
    if hasattr(dimension, '__len__'):
        d = len(dimension)
    else:
        # assume dimension is an int, nx
        d = 1

    # Check space dimension and covariance model
    if d == 1:
        if not isinstance(cov_model, gcm.CovModel1D):
            print("ERROR (MULTIGAUSSIANRUN): 'cov_model' invalid for 1D grid, should be a class: <geone.covModel.CovModel1D> ")
            return out
    elif d == 2:
        if not isinstance(cov_model, gcm.CovModel2D) and not isinstance(cov_model, gcm.CovModel1D):
            print("ERROR (MULTIGAUSSIANRUN): 'cov_model' invalid for 2D grid, should be a class: <geone.covModel.CovModel2D> or <geone.covModel.CovModel1D>")
            return out
    elif d == 3:
        if not isinstance(cov_model, gcm.CovModel3D) and not isinstance(cov_model, gcm.CovModel1D):
            print("ERROR (MULTIGAUSSIANRUN): 'cov_model' invalid for 3D grid, should be a class: <geone.covModel.CovModel3D> or <geone.covModel.CovModel1D>")
            return out
    else:
        print("ERROR (MULTIGAUSSIANRUN): unknown space dimension (check 2nd argurment 'dimension')")
        return out

    # Check (or set) argument 'spacing'
    if spacing is None:
        spacing = tuple(np.ones(d))
    else:
        if hasattr(spacing, '__len__') and len(spacing) != d:
            print("ERROR (MULTIGAUSSIANRUN): 'spacing' of incompatible length")
            return out

    # Check (or set) argument 'origin'
    if origin is None:
        origin = tuple(np.zeros(d))
    else:
        if hasattr(origin, '__len__') and len(origin) != d:
            print("ERROR (MULTIGAUSSIANRUN): 'origin' of incompatible length")
            return out

    # Note: data (x, v) not checked here, directly passed to further function

    if algo in ('fft', 'FFT'):
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
        if kwargs_unexpected_keys:
            # set kwargs_unexpected_keys is not empty
            s = "', '".join(kwargs_unexpected_keys)
            print(f"WARNING (MULTIGAUSSIANRUN): unexpected keyword arguments ('{s}') passed to function '{run_f.__module__}.{run_f.__name__}' were ignored")

        output = run_f(cov_model, dimension, spacing=spacing, origin=origin, x=x, v=v, verbose=verbose, **kwargs_new)
        # -> if mode = 'simulation':
        #    output is an array with (d+1) dimension (axis 0 corresponds to realizations)
        # -> if mode = 'estimation':
        #    output is an array (kriging estimate only) or a 2-tuple of array (kriging estimate and standard deviation);
        #    each array with d dimension
        if (isinstance(output, tuple) and np.any([a is None for a in output])) or output is None:
            print("ERROR (MULTIGAUSSIANRUN): an error occurred...")
            return out
        if mode == 'estimation':
            if isinstance(output, tuple):
                output = np.asarray(output)
            else:
                output = output[np.newaxis, :]
        if output_mode == 'img':
            output = img.Img(
                *np.hstack((np.atleast_1d(dimension), np.ones(3-d, dtype='int'))),
                *np.hstack((np.atleast_1d(spacing), np.ones(3-d))),
                *np.hstack((np.atleast_1d(origin), np.zeros(3-d))),
                nv=output.shape[0], val=output)
        warnings = [] # no warning available if algo = 'fft'

    elif algo in ('classic', 'CLASSIC'):
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
            s = "', '".join(kwargs_unexpected_keys)
            print(f"WARNING (MULTIGAUSSIANRUN): unexpected keyword arguments ('{s}') passed to function '{run_f.__module__}.{run_f.__name__}' were ignored")

        output = run_f(cov_model, dimension, spacing=spacing, origin=origin, x=x, v=v, verbose=verbose, **kwargs)
        if output is None:
            print("ERROR (MULTIGAUSSIANRUN): an error occurred...")
            return out

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
