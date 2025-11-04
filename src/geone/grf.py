#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# Python module:  'grf.py'
# author:         Julien Straubhaar
# date:           jan-2018
# -------------------------------------------------------------------------


"""
Module for gaussian random fields (GRF) simulations in 1D, 2D and 3D,
based on (block) circulant embedding of the covariance matrix and
Fast Fourier Transform (FFT).

References
----------
- J. W. Cooley, J. W. Tukey (1965) \
An algorithm for machine calculation of complex Fourier series. \
Mathematics of Computation 19(90):297-301, \
`doi:10.2307/2003354 <https://dx.doi.org/10.2307/2003354>`_
- C. R. Dietrich, G. N. Newsam (1993) \
A fast and exact method for multidimensional gaussian stochastic simulations. \
Water Resources Research 29(8):2861-2869 \
`doi:10.1029/93WR01070 <https://dx.doi.org/10.1029/93WR01070>`_
- A. T. A. Wood, G. Chan (1994) \
Simulation of Stationary Gaussian Processes in :math:`[0, 1]^d`. \
Journal of Computational and Graphical Statistics 3(4):409-432, \
`doi:10.2307/1390903 <https://dx.doi.org/10.2307/1390903>`_
"""

import numpy as np
from geone import covModel as gcm
from geone import img

# ============================================================================
class GrfError(Exception):
    """
    Custom exception related to `grf` module.
    """
    pass
# ============================================================================

# ----------------------------------------------------------------------------
def extension_min(r, n, s=1.0):
    """
    Computes minimal extension along one direction, for GRF simulations.

    The extension is computed such that a GRF reproduces the covariance model
    appropriately.

    Parameters
    ----------
    r : float
        range (max) along the considered direction
    n : int
        number of cells (dimension) in the considered direction
    s : float, default: 1.0
        cell size in the considered direction

    Returns
    -------
    ext_min : int
        minimal extension in number of cells along the considered direction
        for appropriate GRF simulations
    """
    # fname = 'extension_min'

    k = int(np.ceil(r/s))
    return max(k-n, 0) + k - 1
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def grf1D(
        cov_model,
        dimension, spacing=1.0, origin=0.0,
        x=None, v=None,
        aggregate_data_op=None,
        aggregate_data_op_kwargs=None,
        mean=None, var=None,
        nreal=1,
        extensionMin=None, rangeFactorForExtensionMin=1.0,
        crop=True,
        method=3, conditioningMethod=2,
        measureErrVar=0.0, tolInvKappa=1.e-10,
        verbose=1,
        printInfo=None,
        logger=None):
    """
    Generates Gaussian Random Fields (GRF) in 1D via Fast Fourier Transform (FFT).

    In brief, the GRFs

    - are generated using the given covariance model (`cov_model`),
    - may have a specified mean (`mean`) and variance (`var`), which can be non stationary,
    - may be conditioned to location(s) `x` with value(s) `v`.

    Parameters
    ----------
    cov_model : :class:`geone.covModel.CovModel1D`, or function (`callable`)
        covariance model in 1D or directly a function of covariance

    dimension : int
        `dimension=nx`, number of cells in the 1D simulation grid

    spacing : float, default: 1.0
        `spacing=sx`, cell size

    origin : float, default: 0.0
        `origin=ox`, origin of the 1D simulation grid (left border)

    x : 1D array-like of floats, optional
        data points locations (float coordinates); note: if one point, a float
        is accepted

    v : 1D array-like of floats, optional
        data values at `x` (`v[i]` is the data value at `x[i]`), array of same
        length as `x` (or float if one point)

    aggregate_data_op : str {'sgs', 'krige', 'min', 'max', 'mean', 'quantile', 'most_freq', 'random'}, optional
        operation used to aggregate data points falling in the same grid cells

        - if `aggregate_data_op='sgs'`: function :func:`geone.covModel.sgs` is used \
        with the covariance model `cov_model` given in arguments
        - if `aggregate_data_op='krige'`: function :func:`geone.covModel.krige` is used \
        with the covariance model `cov_model` given in arguments
        - if `aggregate_data_op='most_freq'`: most frequent value is selected \
        (smallest one if more than one value with the maximal frequence)
        - if `aggregate_data_op='random'`: value from a random point is selected
        - otherwise: the function `numpy.<aggregate_data_op>` is used with the \
        additional parameters given by `aggregate_data_op_kwargs`, note that, e.g. \
        `aggregate_data_op='quantile'` requires the additional parameter \
        `q=<quantile_to_compute>`

        Note: if `aggregate_data_op='sgs'` or `aggregate_data_op='random'`, the
        aggregation is done for each realization (simulation), i.e. each simulation
        on the grid starts with a new set of values in conditioning grid cells;
        if `aggregate_data_op='sgs'` or `aggregate_data_op='krige'`, then
        `cov_model` must be a covariance model and not directly the covariance
        function

        By default (`None`): `aggregate_data_op='sgs'` is used

    aggregate_data_op_kwargs : dict, optional
        keyword arguments to be passed to `geone.covModel.sgs`,
        `geone.covModel.krige`, or `numpy.<aggregate_data_op>`, according to
        the parameter `aggregate_data_op`

    mean : function (`callable`), or array-like of floats, or float, optional
        kriging mean value:

        - if a function: function of one argument (xi) that returns the mean at \
        location xi
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), mean values at grid cells (for \
        non-stationary mean)
        - if a float: same mean value at every grid cell

        By default (`None`): the mean of data value (`v`) (0.0 if no data) is
        considered at every grid cell

    var : function (`callable`), or array-like of floats, or float, optional
        kriging variance value:

        - if a function: function of one argument (xi) that returns the variance \
        at location xi
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), variance values at grid cells (for \
        non-stationary variance)
        - if a float: same variance value at every grid cell

        By default (`None`): not used (use of covariance model only)

    nreal : int, default: 1
        number of realization(s)

    extensionMin : int, optional
        minimal extension in cells (see note 1 below)

        By default (`None`): minimal extension is automatically computed:

        - based on the range of the covariance model, if `cov_model` is given as \
        an instance of :class:`geone.covModel.CovModel1D`)
        - set to `nx-1`, if `cov_model` is given as a function (`callable`)

    rangeFactorForExtensionMin : float, default: 1.0
        factor by which the range of the covariance model is multiplied before
        computing the default minimal extension, if `cov_model` is given as
        an instance of :class:`geone.covModel.CovModel1D`) and if
        `extensionMin=None` (not used otherwise)

    crop : bool, default: True
        indicates if the extended generated field (simulation) will be cropped to
        original dimension; note that `crop=False` is not valid with conditioning
        or non-stationary mean or non-stationary variance

    method : int, default: 3
        indicates which method is used to generate unconditional simulations;
        for each method the Discrete Fourier Transform (DFT) "lam" of the
        circulant embedding of the covariance matrix is used, and periodic and
        stationary GRFs are generated

        - `method=1` (method A): generate one GRF Z as follows:
            - generate one real gaussian white noise W
            - apply fft (or fft inverse) on W to get X
            - multiply X by "lam" (term by term)
            - apply fft inverse (or fft) to get Z
        - `method=2` (method B): generate one GRF Z as follows:
           - generate directly X (from method A)
           - multiply X by lam (term by term)
           - apply fft inverse (or fft) to get Z
        - `method=3` (method C, default): generate two independent GRFs Z1, Z2 as follows:
           - generate two independant real gaussian white noises W1, W2 and set \
           W = W1 + i * W2
           - apply fft (or fft inverse) on W to get X
           - multiply X by "lam" (term by term)
           - apply fft inverse (or fft) to get Z, and set Z1 = Re(Z), Z2 = Im(Z); \
           note: if `nreal` is odd, the last field is generated using method A

    conditioningMethod : int, default: 2
        indicates which method is used to update the simulations to account for
        conditioning data; let

        * A: index of conditioning cells
        * B: index of non-conditioning cells
        * Zobs: vector of values of the unconditional simulation Z at conditioning cells
        * :math:`r = \\left(\\begin{array}{cc} r_{AA} & r_{AB}\\\\r_{BA} & r_{BB}\\end{array}\\right)` \
        the covariance matrix, where index A (resp. B) refers to conditioning \
        (resp. non-conditioning) index in the grid;

        an unconditional simulation Z is updated into a conditional simulation ZCond as
        follows; let

        * ZCond[A] = Zobs
        * ZCond[B] = Z[B] + rBA * rAA^(-1) * (Zobs - Z[A])

        i.e. the update consists in adding the kriging estimates of the residues
        to an unconditional simulation

        * `conditioningMethod=1` (method CondtioningA): the matrix M = rBA * rAA^(-1) \
        is explicitly computed (warning: could require large amount of memory), \
        then all the simulations are updated by a sum and a multiplication by the \
        matrix M
        * `conditioningMethod=2` (method CondtioningB, default): for each simulation, \
        the linear system rAA * x = Zobs - Z[A] is solved and then, the multiplication \
        by rBA is done via fft

        Note: parameter `conditioningMethod` is used only for conditional simulation

    measureErrVar : float, default: 0.0
        measurement error variance; the error on conditioning data is assumed to
        follow the distrubution N(0, `measureErrVar` * I); i.e.
        rAA + `measureErrVar` * I is considered instead of rAA for stabilizing the
        linear system for this matrix;
        note: parameter `measureErrVar` is used only for conditional simulation

    tolInvKappa : float, default: 1.e-10
        the simulation is stopped if the inverse of the condition number of rAA
        is above `tolInvKappa`;
        note: parameter `tolInvKappa` is used only for conditional simulation

    verbose : int, default: 1
        verbose mode, higher implies more printing (info):

        - 0: no display
        - 1: warnings
        - 2: warnings + basic info
        - 3 (or >2): all information

        note that if an error occurred, it is raised

    printInfo : bool, optional
        deprecated, use `verbose` instead;

        - if `printInfo=False`, `verbose` is set to 1 (overwritten)
        - if `printInfo=True`, `verbose` is set to 3 (overwritten)
        - if `printInfo=None` (default): not used

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    grf : 2D array of shape (`nreal`, n1)
        GRF realizations, with n1 = nx (= dimension) if `crop=True`, but
        n1 >= nx if `crop=False`;
        `grf[i, j]`: value of the i-th realisation at grid cell of index j

    Notes
    -----
    1. For reproducing covariance model, the dimension of GRF should be large
    enough; let K an integer such that K*`spacing` is greater or equal to the
    correlation range, then:

    - correlation accross opposite border should be removed by extending \
    the domain sufficiently, i.e.

        `extensionMin` >= K - 1

    - two cells could not be correlated simultaneously regarding both \
    distances between them (with respect to the periodic grid), i.e. one \
    should have

        `dimension+extensionMin` >= 2*K - 1.

    To sum up, `extensionMin` should be chosen such that

        `dimension+extensionMin` >= max(`dimension`, K) + K - 1

    i.e.

        `extensionMin` >= max(K-1, 2*K-`dimension`-1)

    2. For large data set:

    - `conditioningMethod` should be set to 2 for using FFT

    - `measureErrVar` can be set to a small positive value to stabilize the \
    covariance matrix for conditioning locations (solving linear system).

    3. Some mathematical details:

    Discrete Fourier Transform (DFT) of a vector x of length N is given by

        c = DFT(x) = F * x

    where F is the N x N matrix with coefficients

        F(j,k) = [exp(-i*2*pi*j*k/N)], 0 <= j,k <= N-1

    We have

        F^(-1) = 1/N * F^(*)

    where ^(*) denotes the conjugate transpose.

    Let

        Q = 1/N^(1/2) * F

    Then Q is unitary, i.e. Q^(-1) = Q^(*)

    Then, we have

        DFT = F = N^(1/2) * Q,

        DFT^(-1) = 1/N * F^(*) = 1/N^(1/2) * Q^(*)

    Using `numpy` package:

        `numpy.fft.fft()` = DFT()

        `numpy.fft.ifft()` = DFT^(-1)()
    """
    fname = 'grf1D'

    # Set verbose mode according to printInfo (if given)
    if printInfo is not None:
        if printInfo:
            verbose = 3
        else:
            verbose = 1

    # Check first argument and get covariance function
    if cov_model.__class__.__name__ == 'function':
        # covariance function is given
        cov_func = cov_model
        cov_range = None # unknown range
    elif isinstance(cov_model, gcm.CovModel1D):
        # Prevent calculation if covariance model is not stationary
        if not cov_model.is_stationary():
            err_msg = f'{fname}: `cov_model` is not stationary: {fname} cannot be applied (use `geone.geosclassicinterface` package)'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        cov_func = cov_model.func() # covariance function
        cov_range = cov_model.r()
    else:
        err_msg = f'{fname}: `cov_model` invalid'
        if logger: logger.error(err_msg)
        raise GrfError(err_msg)

    # aggregate_data_op (default)
    if aggregate_data_op is None:
        aggregate_data_op = 'sgs'

    # Number of realization(s)
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose > 0:
            if logger:
                logger.warning(f'{fname}: `nreal` <= 0: `None` is returned')
            else:
                print(f'{fname}: WARNING: `nreal` <= 0: `None` is returned')
        return None

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: Preliminary computation...')
        else:
            print(f'{fname}: Preliminary computation...')

    #### Preliminary computation ####
    nx = dimension
    sx = spacing
    ox = origin

    if method not in (1, 2, 3):
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise GrfError(err_msg)

    if x is None and v is not None:
        err_msg = f'{fname}: `x` is not given (`None`) but `v` is given (not `None`)'
        if logger: logger.error(err_msg)
        raise GrfError(err_msg)

    if x is not None:
        if conditioningMethod not in (1, 2):
            err_msg = f'{fname}: `conditioningMethod` invalid'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        if v is None:
            err_msg = f'{fname}: `x` is given (not `None`) but `v` is not given (`None`)'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        x = np.asarray(x, dtype='float').reshape(-1, 1) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

    mean_x = mean
    if mean is not None:
        if callable(mean):
            if x is not None:
                mean_x = mean(x[:, 0])
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            mean = mean(xi) # replace function 'mean' by its evaluation on the grid
        else:
            mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if mean.size == 1:
                if x is not None:
                    mean_x = mean
            elif mean.size == nx:
                # mean = mean.reshape(nx)
                if x is not None:
                    mean_x = img.Img_interp_func(img.Img(nx, 1, 1, sx, 1., 1., ox, 0., 0., nv=1, val=mean, logger=logger), iy=0, iz=0, logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `mean` is not valid'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

    var_x = var
    if var is not None:
        if callable(var):
            if x is not None:
                var_x = var(x[:, 0])
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            var = var(xi) # replace function 'var' by its evaluation on the grid
        else:
            var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if var.size == 1:
                if x is not None:
                    var_x = var
            elif var.size == nx:
                # var = var.reshape(nx)
                if x is not None:
                    var_x = img.Img_interp_func(img.Img(nx, 1, 1, sx, 1., 1., ox, 0., 0., nv=1, val=var, logger=logger), iy=0, iz=0, logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `var` is not valid'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

    # data point set from x, v
    if x is not None:
        if aggregate_data_op_kwargs is None:
            aggregate_data_op_kwargs = {}
        if aggregate_data_op == 'krige' or aggregate_data_op == 'sgs':
            if cov_range is None:
                # cov_model is directly the covariance function
                err_msg = f"{fname}: `cov_model` must be a model (not directly a function) when `aggregate_data_op='{aggregate_data_op}'` is used"
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

            # Get grid cell with at least one data point:
            # x_agg: 2D array, each row contains the coordinates of the center of such cell
            try:
                im_tmp = img.imageFromPoints(
                        x, values=None, varname=None,
                        nx=nx, sx=sx, ox=ox,
                        indicator_var=True, 
                        count_var=False,
                        logger=logger)
            except Exception as exc:
                err_msg = f'{fname}: cannot set image from points'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg) from exc

            ind_agg = np.where(im_tmp.val[0])
            if len(ind_agg[0]) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

            x_agg = im_tmp.xx()[ind_agg].reshape(-1, 1)
            # x_agg = im_tmp.xx()[*ind_agg].reshape(-1, 1) # ok from python 3.11 only ?
            ind_agg = ind_agg[2:] # remove index along z and y axes
            del(im_tmp)
            # Compute
            # - kriging estimate (v_agg) and kriging std (v_agg_std) at x_agg,
            # - or nreal simulation(s) (v_agg) at x_agg
            if mean is not None and mean.size > 1:
                mean_x_agg = mean[ind_agg]
                # mean_x_agg = mean[*ind_agg]
            else:
                mean_x_agg = mean
            if var is not None and var.size > 1:
                var_x_agg = var[ind_agg]
                # var_x_agg = var[*ind_agg]
            else:
                var_x_agg = var
            if aggregate_data_op == 'krige':
                try:
                    v_agg, v_agg_std = gcm.krige(
                            x, v, x_agg, cov_model, method='simple_kriging',
                            mean_x=mean_x, mean_xu=mean_x_agg,
                            var_x=var_x, var_xu=var_x_agg,
                            verbose=0, logger=logger,
                            **aggregate_data_op_kwargs)
                except Exception as exc:
                    err_msg = f"{fname}: aggratating data points in grid failed (`aggregate_data_op='{aggregate_data_op}'`)"
                    if logger: logger.error(err_msg)
                    raise GrfError(err_msg) from exc

                # all real (same values)
                v_agg = np.tile(v_agg, nreal).reshape(nreal, -1)
            else:
                try:
                    v_agg = gcm.sgs(
                            x, v, x_agg, cov_model, method='simple_kriging',
                            mean_x=mean_x, mean_xu=mean_x_agg,
                            var_x=var_x, var_xu=var_x_agg,
                            nreal=nreal, seed=None,
                            verbose=0, logger=logger,
                            **aggregate_data_op_kwargs)
                except Exception as exc:
                    err_msg = f"{fname}: aggratating data points in grid failed (`aggregate_data_op='{aggregate_data_op}'`)"
                    if logger: logger.error(err_msg)
                    raise GrfError(err_msg) from exc

            xx_agg = x_agg[:, 0]
            # yy_agg = 0.5*np.ones_like(xx_agg)
            # zz_agg = 0.5*np.ones_like(xx_agg)
        elif aggregate_data_op == 'random':
            # Aggregate data on grid cell by taking random point
            xx = x[:, 0]
            yy = 0.5*np.ones_like(xx)
            zz = 0.5*np.ones_like(xx)
            # first realization of v_agg
            try:
                xx_agg, yy_agg, zz_agg, v_agg, i_inv = img.aggregateDataPointsWrtGrid(
                        xx, yy, zz, v,
                        nx, 1, 1, sx, 1.0, 1.0, ox, 0.0, 0.0,
                        op=aggregate_data_op, return_inverse=True,
                        logger=logger,
                        **aggregate_data_op_kwargs)
            except Exception as exc:
                err_msg = f"{fname}: aggratating data points in grid failed (`aggregate_data_op='{aggregate_data_op}'`)"
                if logger: logger.error(err_msg)
                raise GrfError(err_msg) from exc

            if len(xx_agg) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

            # next realizations of v_agg
            v_agg = np.vstack((v_agg, np.zeros((nreal-1, v_agg.size))))
            for i in range(1, nreal):
                v_agg[i] = [v[np.random.choice(np.where(i_inv==j)[0])] for j in range(len(xx_agg))]
        else:
            # Aggregate data on grid cell by using the given operation
            xx = x[:, 0]
            yy = 0.5*np.ones_like(xx)
            zz = 0.5*np.ones_like(xx)
            try:
                xx_agg, yy_agg, zz_agg, v_agg = img.aggregateDataPointsWrtGrid(
                        xx, yy, zz, v,
                        nx, 1, 1, sx, 1.0, 1.0, ox, 0.0, 0.0,
                        op=aggregate_data_op, 
                        logger=logger,
                        **aggregate_data_op_kwargs)
            except Exception as exc:
                err_msg = f"{fname}: aggratating data points in grid failed (`aggregate_data_op='{aggregate_data_op}'`)"
                if logger: logger.error(err_msg)
                raise GrfError(err_msg) from exc

            if len(xx_agg) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

            # all real (same values)
            v_agg = np.tile(v_agg, nreal).reshape(nreal, -1)

    if not crop:
        if x is not None: # conditional simulation
            err_msg = f'{fname}: `crop=False` cannot be used with conditional simulation'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        if mean is not None and mean.size > 1:
            err_msg = f'{fname}: `crop=False` cannot be used with non-stationary mean'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        if var is not None and var.size > 1:
            err_msg = f'{fname}: `crop=False` cannot be used with non-stationary variance'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

    if extensionMin is None:
        # default extensionMin
        if cov_range is not None: # known range
            # ... based on range of covariance model
            extensionMin = extension_min(rangeFactorForExtensionMin*cov_range, nx, s=sx)
        else:
            # ... based on dimension
            extensionMin = dimension - 1

    Nmin = nx + extensionMin

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: Computing circulant embedding...')
        else:
            print(f'{fname}: Computing circulant embedding...')

    # Circulant embedding of the covariance matrix
    # --------------------------------------------
    # The embedding matrix is a circulant matrix of size N x N, computed from
    # the covariance function.
    # To take a maximal benefit of Fast Fourier Transform (FFT) for computing DFT,
    # we choose
    #    N = 2^g (a power of 2), with N >= Nmin, N >= 2
    g = int(max(np.ceil(np.log2(Nmin)), 1.0))
    N = int(2**g)

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: embedding dimension: {N}')
        else:
            print(f'{fname}: embedding dimension: {N}')

    # ccirc: coefficient of the embedding matrix (first line), vector of size N
    L = int (N/2)
    h = np.arange(-L, L, dtype=float) * sx # [-L ... 0 ... L-1] * sx
    ccirc = cov_func(h)

    del(h)

    # ...shift first L index to the end of the axis, i.e.:
    #    [-L ... 0 ... L-1] -> [0 ... L-1 -L ... -1]
    ind = np.arange(L)
    ccirc = ccirc[np.hstack((ind+L, ind))]

    del(ind)

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: Computing FFT of circulant matrix...')
        else:
            print(f'{fname}: Computing FFT of circulant matrix...')

    # Compute the Discrete Fourier Transform (DFT) of ccric, via FFT
    # --------------------------------------------------------------
    # The DFT coefficients
    #   lam = DFT(ccirc) = (lam(0),lam(1),...,lam(N-1))
    # are the eigen values of the embedding matrix.
    # We have:
    #   a) lam are real coefficients, because the embedding matrix is symmetric
    #   b) lam(k) = lam(N-k), k=1,...,N-1, because the coefficients ccirc are real
    lam = np.real(np.fft.fft(ccirc))
    # ...note that the imaginary parts are equal to 0

    # Eventual use of approximate embedding
    # -------------------------------------
    # If some DFT coefficients are negative, then set them to zero
    # and update them to fit the marginals distribution (approximate embedding)
    if np.min(lam) < 0:
        lam = np.sum(lam)/np.sum(np.maximum(lam, 0.)) * np.maximum(lam, 0.)

    # Take the square root of the (updated) DFT coefficients
    # ------------------------------------------------------
    lamSqrt = np.sqrt(lam)

    if x is None or conditioningMethod == 1:
        del(lam)

    # For specified variance
    # ----------------------
    # Compute updating factor
    if var is not None:
        varUpdate = np.sqrt(var/cov_func(0.))

    # Dealing with conditioning
    # -------------------------
    if x is not None:
        if verbose > 1:
            if logger:
                logger.info(f'{fname}: Treatment of conditioning data...')
            else:
                print(f'{fname}: Treatment of conditioning data...')
        # Compute the part rAA of the covariance matrix
        #        +         +
        #        | rAA rAB |
        #    r = |         |
        #        | rBA rBB |
        #        +         +
        # where index A (resp. B) refers to
        # conditioning (resp. non-conditioning) index in the grid.

        if verbose > 1:
            if logger:
                logger.info(f'{fname}: Computing covariance matrix (rAA) for conditioning locations...')
            else:
                print(f'{fname}: Computing covariance matrix (rAA) for conditioning locations...')

        # Compute
        #    indc: node index of conditioning node,
        #          rounded to lower index if between two grid node and index is positive
        indc_f = (xx_agg-origin)/spacing
        indc = indc_f.astype(int)
        indc = indc - 1 * np.all((indc == indc_f, indc > 0), axis=0)

        nc = len(xx_agg)

        # rAA
        rAA = np.zeros((nc, nc))

        diagEntry = ccirc[0] + measureErrVar
        for i in range(nc):
            rAA[i,i] = diagEntry
            for j in range(i+1, nc):
                rAA[i,j] = ccirc[np.mod(indc[j]-indc[i], N)]
                rAA[j,i] = rAA[i,j]

        # Test if rAA is almost singular...
        if 1./np.linalg.cond(rAA) < tolInvKappa:
            err_msg = f'{fname}: conditioning issue: condition number of matrix rAA is too big'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        # Compute:
        #    indnc: node index of non-conditioning node (nearest node)
        indnc = np.asarray(np.setdiff1d(np.arange(nx), indc), dtype=int)
        nnc = len(indnc)

        if conditioningMethod == 1:
            # Method ConditioningA
            # --------------------
            if verbose > 1:
                if logger:
                    logger.info(f'{fname}: Computing covariance matrix (rBA) for non-conditioning / conditioning locations...')
                else:
                    print(f'{fname}: Computing covariance matrix (rBA) for non-conditioning / conditioning locations...')

            # Compute the parts rBA of the covariance matrix (see above)
            # rBA
            rBA = np.zeros((nnc, nc))
            for j in range(nc):
                k = np.mod(indc[j] - indnc, N)
                rBA[:,j] = ccirc[k]

            if verbose > 1:
                if logger:
                    logger.info(f'{fname}: Computing rBA * rAA^(-1)...')
                else:
                    print(f'{fname}: Computing rBA * rAA^(-1)...')

            # compute rBA * rAA^(-1)
            rBArAAinv = np.dot(rBA, np.linalg.inv(rAA))

            del(rAA, rBA)

            # If a variance var is specified, then the matrix r should be updated
            # by the following operation:
            #    diag((var/cov_func(0))^1/2) * r * diag((var/cov_func(0))^1/2)
            # Hence, if a non-stationary variance is specified,
            # the matrix rBA * rAA^(-1) should be consequently updated
            # by multiplying its columns by 1/varUpdate[indc] and its rows by varUpdate[indnc]
            if var is not None and var.size > 1:
                rBArAAinv = np.transpose(varUpdate[indnc] * np.transpose(1./varUpdate[indc] * rBArAAinv))

        elif conditioningMethod == 2:
            # Method ConditioningB
            # --------------------
            if verbose > 1:
                if logger:
                    logger.info(f'{fname}: Computing index in the embedding grid for non-conditioning / conditioning locations...')
                else:
                    print(f'{fname}: Computing index in the embedding grid for non-conditioning / conditioning locations...')

            # Compute index in the embedding grid for indc and indnc
            # (to allow use of fft)
            indcEmb = indc
            indncEmb = indnc

        if mean is None:
            # Set mean for grf
            mean = np.array([np.mean(v)])

    else: # x is None (unconditional)
        if mean is None:
            # Set mean for grf
            mean = np.array([0.0])

    del(ccirc)
    #### End of preliminary computation ####

    # Unconditional simulation
    # ========================
    # Method A: Generating one real GRF Z
    # --------
    # 1. Generate a real gaussian white noise W ~ N(0,1) on G (1D grid)
    # 2. Compute Z = Q^(*) D Q * W
    #    [OR: Z = Q D Q^(*) * W], where
    #       Q is normalized DFT matrix
    #       D = diag(lamSqrt)
    #    i.e:
    #       Z = DFT^(-1)(D * DFT(W))
    #       [OR: Z = DFT(D * DFT^(-1)(W))]
    #
    # Method B: Generating one real GRF Z
    # --------
    # 1. Assuming N=2L even, generate
    #       V1 = (V1(1),...,V1(L-1)) ~ 1/sqrt(2) N(0, 1)
    #       V2 = (V2(1),...,V2(L-1)) ~ 1/sqrt(2) N(0, 1)
    #    and set
    #       X = (X(0),...,X(N-1)) on G
    #    with
    #       X(0) ~ N(0,1)
    #       X(L) ~ N(0,1)
    #    and
    #       X(k) = V1(k) + i V2(k)
    #       X(N-k) = V1(k) - i V2(k)
    #    for k = 1,...,L-1
    # 2. Compute Z = Q^(*) D * X
    #    [OR: Z = Q D * X], where
    #       Q is normalized DFT matrix
    #       D = diag(lamSqrt)
    #    i.e:
    #       Z = N^(1/2) * DFT^(-1)(D * X)
    #       [OR: Z = 1/N^(1/2) * DFT(D * X]
    #
    # Method C: Generating two independent real GRFs Z1, Z2
    # --------
    # (If nreal is odd, the last realization is generated using method A.)
    # 1. Generate two independent real gaussian white noises W1,W2 ~ N(0,1) on G (1D grid)
    #    and let W = W1 + i * W2 (complex value)
    # 2. Compute Z = Q^(*) D * W
    #    [OR: Z = Q D * W], where
    #       Q is normalized DFT matrix
    #       D = diag(lamSqrt)
    #    i.e:
    #       Z = N^(1/2) * DFT^(-1)(D * W)
    #       [OR: Z = 1/N^(1/2) * DFT(D * W)]
    #    Then the real and imaginary parts of Z are two independent GRFs
    if crop:
        grfNx = nx
    else:
        grfNx = N

    grf = np.zeros((nreal, grfNx))

    if method == 1:
        # Method A
        # --------
        for i in range(nreal):
            if verbose > 2:
                if logger:
                    logger.info(f'{fname}: unconditional simulation {i+1:4d} of {nreal:4d}...')
                else:
                    print(f'{fname}: unconditional simulation {i+1:4d} of {nreal:4d}...')

            W = np.random.normal(size=N)

            Z = np.fft.ifft(lamSqrt * np.fft.fft(W))
            # ...note that Im(Z) = 0
            grf[i] = np.real(Z[0:grfNx])

    elif method == 2:
        # Method B
        # --------
        for i in range(nreal):
            if verbose > 2:
                if logger:
                    logger.info(f'{fname}: unconditional simulation {i+1:4d} of {nreal:4d}...')
                else:
                    print(f'{fname}: unconditional simulation {i+1:4d} of {nreal:4d}...')

            X1 = np.zeros(N)
            X2 = np.zeros(N)

            X1[[0,L]] = np.random.normal(size=2)
            X1[range(1,L)] = 1./np.sqrt(2) * np.random.normal(size=L-1)
            X1[list(reversed(range(L+1,N)))] = X1[range(1,L)]

            X2[range(1,L)] = 1./np.sqrt(2) * np.random.normal(size=L-1)
            X2[list(reversed(range(L+1,N)))] = - X2[range(1,L)]

            X = np.array(X1, dtype=complex)
            X.imag = X2

            Z = np.sqrt(N) * np.fft.ifft(lamSqrt * X)

            grf[i] = np.real(Z[0:grfNx])

    elif method == 3:
        # Method C
        # --------
        for i in np.arange(0, nreal-1, 2):
            if verbose > 2:
                if logger:
                    logger.info(f'{fname}: unconditional simulation {i+1:4d}-{i+2:4d} of {nreal:4d}...')
                else:
                    print(f'{fname}: unconditional simulation {i+1:4d}-{i+2:4d} of {nreal:4d}...')

            W = np.array(np.random.normal(size=N), dtype=complex)
            W.imag = np.random.normal(size=N)
            Z = np.sqrt(N) * np.fft.ifft(lamSqrt * W)
            #  Z = 1/sqrt(N) * np.fft.fft(lamSqrt * W)] # see above: [OR:...]

            grf[i] = np.real(Z[0:grfNx])
            grf[i+1] = np.imag(Z[0:grfNx])

        if np.mod(nreal, 2) == 1:
            if verbose > 2:
                if logger:
                    logger.info(f'{fname}: unconditional simulation {nreal:4d} of {nreal:4d}...')
                else:
                    print(f'{fname}: unconditional simulation {nreal:4d} of {nreal:4d}...')

            W = np.random.normal(size=N)
            Z = np.fft.ifft(lamSqrt * np.fft.fft(W))

            grf[nreal-1] = np.real(Z[0:grfNx])

    if var is not None:
        grf = varUpdate * grf

    grf = mean + grf

    # Conditional simulation
    # ----------------------
    # Let
    #    A: index of conditioning nodes
    #    B: index of non-conditioning nodes
    #    Zobs: vector of values at conditioning nodes
    # and
    #        +         +
    #        | rAA rAB |
    #    r = |         |
    #        | rBA rBB |
    #        +         +
    # the covariance matrix, where index A (resp. B) refers to
    # conditioning (resp. non-conditioning) index in the grid.
    #
    # Then, from an unconditional simulation Z, we retrieve a conditional
    # simulation ZCond as follows.
    # Let
    #    ZCond[A] = Zobs
    #    ZCond[B] = Z[B] + rBA * rAA^(-1) * (Zobs - Z[A])
    if x is not None:
        if conditioningMethod == 1:
            # Method ConditioningA
            # --------------------
            if verbose > 1:
                if logger:
                    logger.info(f'{fname}: updating conditional simulations...')
                else:
                    print(f'{fname}: updating conditional simulations...')

            # Update all simulations at a time,
            # use the matrix rBA * rAA^(-1) already computed
            grf[:,indnc] = grf[:,indnc] + np.transpose(np.dot(rBArAAinv, np.transpose(v_agg - grf[:,indc])))
            grf[:,indc] = v_agg

        elif conditioningMethod == 2:
            # Method ConditioningB
            # --------------------
            # Update each simulation successively as follows:
            #    - solve rAA * x = Zobs - z[A]
            #    - do the multiplication rBA * x via the circulant embedding of the
            #      covariance matrix (using fft)
            rAAinvResiduEmb = np.zeros(N)

            for i in range(nreal):
                if verbose > 2:
                    if logger:
                        logger.info(f'{fname}: updating conditional simulation {i+1:4d} of {nreal:4d}...')
                    else:
                        print(f'{fname}: updating conditional simulation {i+1:4d} of {nreal:4d}...')

                # Compute residue
                residu = v_agg[i] - grf[i,indc]
                # ... update if non-stationary variance is specified
                if var is not None and var.size > 1:
                    residu = 1./varUpdate[indc] * residu

                # Compute
                #    x = rAA^(-1) * residu, and then
                #    Z = rBA * x via the circulant embedding of the covariance matrix
                rAAinvResiduEmb[indcEmb] = np.linalg.solve(rAA, residu)
                Z = np.fft.ifft(lam * np.fft.fft(rAAinvResiduEmb))
                # ...note that Im(Z) = 0
                Z = np.real(Z[indncEmb])

                # ... update if non-stationary covariance is specified
                if var is not None and var.size > 1:
                    Z = varUpdate[indnc] * Z

                grf[i, indnc] = grf[i, indnc] + Z
                grf[i, indc] = v_agg[i]

    return grf
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def krige1D(
        cov_model,
        dimension, spacing=1.0, origin=0.0,
        x=None, v=None,
        aggregate_data_op=None,
        aggregate_data_op_kwargs=None,
        mean=None, var=None,
        extensionMin=None, rangeFactorForExtensionMin=1.0,
        conditioningMethod=1, # note: set conditioningMethod=2 if unable to allocate memory
        measureErrVar=0.0, tolInvKappa=1.e-10,
        computeKrigSD=True,
        verbose=1,
        printInfo=None,
        logger=None):
    """
    Computes kriging estimates and standard deviations in 1D via FFT.

    It is a simple kriging

    - of value(s) `v` at location(s) `x`,
    - based on the given covariance model (`cov_model`),
    - it may account for a specified mean (`mean`) and variance (`var`), which can be non stationary.

    Parameters
    ----------
    cov_model : :class:`geone.covModel.CovModel1D`, or function (`callable`)
        covariance model in 1D or directly a function of covariance

    dimension : int
        `dimension=nx`, number of cells in the 1D simulation grid

    spacing : float, default: 1.0
        `spacing=sx`, cell size

    origin : float, default: 0.0
        `origin=ox`, origin of the 1D simulation grid (left border)

    x : 1D array-like of floats, optional
        data points locations (float coordinates); note: if one point, a float
        is accepted

    v : 1D array-like of floats, optional
        data values at `x` (`v[i]` is the data value at `x[i]`), array of same
        length as `x` (or float if one point)

    aggregate_data_op : str {'krige', 'min', 'max', 'mean', 'quantile', 'most_freq'}, optional
        operation used to aggregate data points falling in the same grid cells

        - if `aggregate_data_op='krige'`: function :func:`geone.covModel.krige` is used \
        with the covariance model `cov_model` given in arguments
        - if `aggregate_data_op='most_freq'`: most frequent value is selected \
        (smallest one if more than one value with the maximal frequence)
        - otherwise: the function `numpy.<aggregate_data_op>` is used with the \
        additional parameters given by `aggregate_data_op_kwargs`, note that, e.g. \
        `aggregate_data_op='quantile'` requires the additional parameter \
        `q=<quantile_to_compute>`

        Note: if `aggregate_data_op='krige'`, then `cov_model` must be a
        covariance model and not directly the covariance function

        By default (`None`): `aggregate_data_op='krige'` is used

    aggregate_data_op_kwargs : dict, optional
        keyword arguments to be passed to `geone.covModel.krige`,
        `geone.covModel.krige`, or `numpy.<aggregate_data_op>`, according to
        the parameter `aggregate_data_op`

    mean : function (`callable`), or array-like of floats, or float, optional
        kriging mean value:

        - if a function: function of one argument (xi) that returns the mean at \
        location xi
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), mean values at grid cells (for \
        non-stationary mean)
        - if a float: same mean value at every grid cell

        By default (`None`): the mean of data value (`v`) (0.0 if no data) is
        considered at every grid cell

    var : function (`callable`), or array-like of floats, or float, optional
        kriging variance value:

        - if a function: function of one argument (xi) that returns the variance \
        at location xi
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), variance values at grid cells (for \
        non-stationary variance)
        - if a float: same variance value at every grid cell

        By default (`None`): not used (use of covariance model only)

    extensionMin : int, optional
        minimal extension in cells (see note 1 below)

        By default (`None`): minimal extension is automatically computed:

        - based on the range of the covariance model, if `cov_model` is given as \
        an instance of :class:`geone.covModel.CovModel1D`)
        - set to `nx-1`, if `cov_model` is given as a function (`callable`)

    rangeFactorForExtensionMin : float, default: 1.0
        factor by which the range of the covariance model is multiplied before
        computing the default minimal extension, if `cov_model` is given as
        an instance of :class:`geone.covModel.CovModel1D`) and if
        `extensionMin=None` (not used otherwise)

    conditioningMethod : int, default: 1
        indicates which method is used to update the simulations to account for
        conditioning data; let

        * A: index of conditioning cells
        * B: index of non-conditioning cells
        * :math:`r = \\left(\\begin{array}{cc} r_{AA} & r_{AB}\\\\r_{BA} & r_{BB}\\end{array}\\right)` \
        the covariance matrix, where index A (resp. B) refers to conditioning \
        (resp. non-conditioning) index in the grid;

        then, thre kriging estimates and kriging variances are

        * krig[B]    = mean + rBA * rAA^(-1) * (v - mean)
        * krigVar[B] = diag(rBB - rBA * rAA^(-1) * rAB)

        and the computation is done according to `conditioningMethod`:

        * `conditioningMethod=1` (method CondtioningA, default): the matrices \
        rBA, RAA^(-1) are explicitly computed (warning: could require large \
        amount of memory)
        * `conditioningMethod=2` (method CondtioningB): for kriging estimates, \
        the linear system rAA * y = (v - mean) is solved, and then mean + rBA*y is \
        computed; for kriging variances, for each column u[j] of rAB, the linear \
        system rAA * y = u[j] is solved, and then rBB[j,j] - y^t*y is computed

        Note: set `conditioningMethod=2` if unable to allocate memory

    measureErrVar : float, default: 0.0
        measurement error variance; the error on conditioning data is assumed to
        follow the distrubution N(0, `measureErrVar` * I); i.e.
        rAA + `measureErrVar` * I is considered instead of rAA for stabilizing the
        linear system for this matrix

    tolInvKappa : float, default: 1.e-10
        the computation is stopped if the inverse of the condition number of rAA
        is above `tolInvKappa`

    computeKrigSD : bool, default: True
        indicates if the kriging standard deviations are computed

    verbose : int, default: 2
        verbose mode, higher implies more printing (info):

        - 0: no display
        - 1: warnings
        - 2: warnings + basic info
        - 3 (or >2): all information

        note that if an error occurred, it is raised

    printInfo : bool, optional
        deprecated, use `verbose` instead;

        - if `printInfo=False`, `verbose` is set to 1 (overwritten)
        - if `printInfo=True`, `verbose` is set to 3 (overwritten)
        - if `printInfo=None` (default): not used

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    krig : 1D array of shape (nx,)
        kriging estimates, with nx (= dimension);
        `krig[j]`: value at grid cell of index j

    krigSD : 1D array of shape (nx,), optional
        kriging standard deviations, with nx (= dimension);
        `krigSD[j]`: value at grid cell of index j;
        returned if `computeKrigSD=True`

    Notes
    -----
    1. For reproducing covariance model, the dimension of GRF should be large
    enough; let K an integer such that K*`spacing` is greater or equal to the
    correlation range, then:

    - correlation accross opposite border should be removed by extending \
    the domain sufficiently, i.e.

        `extensionMin` >= K - 1

    - two cells could not be correlated simultaneously regarding both \
    distances between them (with respect to the periodic grid), i.e. one \
    should have

        `dimension+extensionMin` >= 2*K - 1.

    To sum up, `extensionMin` should be chosen such that

        `dimension+extensionMin` >= max(`dimension`, K) + K - 1

    i.e.

        `extensionMin` >= max(K-1, 2*K-`dimension`-1)

    2. For large data set:

    - `conditioningMethod` should be set to 2 for using FFT

    - `measureErrVar` can be set to a small positive value to stabilize the \
    covariance matrix for conditioning locations (solving linear system).

    3. Some mathematical details:

    Discrete Fourier Transform (DFT) of a vector x of length N is given by

        c = DFT(x) = F * x

    where F is the N x N matrix with coefficients

        F(j,k) = [exp(-i*2*pi*j*k/N)], 0 <= j,k <= N-1

    We have

        F^(-1) = 1/N * F^(*)

    where ^(*) denotes the conjugate transpose.

    Let

        Q = 1/N^(1/2) * F

    Then Q is unitary, i.e. Q^(-1) = Q^(*)

    Then, we have

        DFT = F = N^(1/2) * Q,

        DFT^(-1) = 1/N * F^(*) = 1/N^(1/2) * Q^(*)

    Using `numpy` package:

        `numpy.fft.fft()` = DFT()

        `numpy.fft.ifft()` = DFT^(-1)()
    """
    fname = 'krige1D'

    # Set verbose mode according to printInfo (if given)
    if printInfo is not None:
        if printInfo:
            verbose = 3
        else:
            verbose = 1

    # Check third argument and get covariance function
    if cov_model.__class__.__name__ == 'function':
        # covariance function is given
        cov_func = cov_model
        cov_range = None # unknown range
    elif isinstance(cov_model, gcm.CovModel1D):
        # Prevent calculation if covariance model is not stationary
        if not cov_model.is_stationary():
            err_msg = f'{fname}: `cov_model` is not stationary: {fname} cannot be applied (use `geone.geosclassicinterface` package)'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        cov_func = cov_model.func() # covariance function
        cov_range = cov_model.r()
    else:
        err_msg = f'{fname}: `cov_model` invalid'
        if logger: logger.error(err_msg)
        raise GrfError(err_msg)

    # aggregate_data_op (default)
    if aggregate_data_op is None:
        aggregate_data_op = 'krige'

    nx = dimension
    sx = spacing
    ox = origin

    if x is None and v is not None:
        err_msg = f'{fname}: `x` is not given (`None`) but `v` is given (not `None`)'
        if logger: logger.error(err_msg)
        raise GrfError(err_msg)

    if x is not None:
        if conditioningMethod not in (1, 2):
            err_msg = f'{fname}: `conditioningMethod` invalid'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        if v is None:
            err_msg = f'{fname}: `x` is given (not `None`) but `v` is not given (`None`)'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        x = np.asarray(x, dtype='float').reshape(-1, 1) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

    mean_x = mean
    if mean is not None:
        if callable(mean):
            if x is not None:
                mean_x = mean(x[:, 0])
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            mean = mean(xi) # replace function 'mean' by its evaluation on the grid
        else:
            mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if mean.size == 1:
                if x is not None:
                    mean_x = mean
            elif mean.size == nx:
                # mean = mean.reshape(nx)
                if x is not None:
                    mean_x = img.Img_interp_func(img.Img(nx, 1, 1, sx, 1., 1., ox, 0., 0., nv=1, val=mean, logger=logger), iy=0, iz=0, logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `mean` is not valid'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

    var_x = var
    if var is not None:
        if callable(var):
            if x is not None:
                var_x = var(x[:, 0])
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            var = var(xi) # replace function 'var' by its evaluation on the grid
        else:
            var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if var.size == 1:
                if x is not None:
                    var_x = var
            elif var.size == nx:
                # var = var.reshape(nx)
                if x is not None:
                    var_x = img.Img_interp_func(img.Img(nx, 1, 1, sx, 1., 1., ox, 0., 0., nv=1, val=var, logger=logger), iy=0, iz=0, logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `var` is not valid'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

    if x is None:
        # No data: kriging return the mean and the standard deviation...
        krig = np.zeros(nx)
        if mean is not None:
            krig[...] = mean
        if computeKrigSD:
            krigSD = np.zeros(nx)
            if var is not None:
                krigSD[...] = np.sqrt(var)
            else:
                krigSD[...] = np.sqrt(cov_func(0.))
            return krig, krigSD
        else:
            return krig

    if aggregate_data_op_kwargs is None:
        aggregate_data_op_kwargs = {}

    if aggregate_data_op == 'krige':
        if cov_range is None:
            # cov_model is directly the covariance function
            err_msg = f"{fname}: `cov_model` must be a model (not directly a function) when `aggregate_data_op='{aggregate_data_op}'` is used"
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        # Get grid cell with at least one data point:
        # x_agg: 2D array, each row contains the coordinates of the center of such cell
        try:
            im_tmp = img.imageFromPoints(
                    x, values=None, varname=None,
                    nx=nx, sx=sx, ox=ox,
                    indicator_var=True, 
                    count_var=False,
                    logger=logger)
        except Exception as exc:
            err_msg = f'{fname}: cannot set image from points'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg) from exc

        ind_agg = np.where(im_tmp.val[0])
        if len(ind_agg[0]) == 0:
            err_msg = f'{fname}: no data point in grid'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        x_agg = im_tmp.xx()[ind_agg].reshape(-1, 1)
        # x_agg = im_tmp.xx()[*ind_agg].reshape(-1, 1)
        ind_agg = ind_agg[2:] # remove index along z and y axes
        del(im_tmp)
        # Compute
        # - kriging estimate (v_agg) and kriging std (v_agg_std) at x_agg,
        # - or nreal simulation(s) (v_agg) at x_agg
        if mean is not None and mean.size > 1:
            mean_x_agg = mean[ind_agg]
            # mean_x_agg = mean[*ind_agg]
        else:
            mean_x_agg = mean
        if var is not None and var.size > 1:
            var_x_agg = var[ind_agg]
            # var_x_agg = var[*ind_agg]
        else:
            var_x_agg = var
        try:
            v_agg, v_agg_std = gcm.krige(
                    x, v, x_agg, cov_model, method='simple_kriging',
                    mean_x=mean_x, mean_xu=mean_x_agg,
                    var_x=var_x, var_xu=var_x_agg,
                    verbose=0, logger=logger,
                    **aggregate_data_op_kwargs)
        except Exception as exc:
            err_msg = f"{fname}: aggratating data points in grid failed (`aggregate_data_op='{aggregate_data_op}'`)"
            if logger: logger.error(err_msg)
            raise GrfError(err_msg) from exc

        xx_agg = x_agg[:, 0]
        # yy_agg = 0.5*np.ones_like(xx_agg)
        # zz_agg = 0.5*np.ones_like(xx_agg)
    else:
        # Aggregate data on grid cell by using the given operation
        xx = x[:, 0]
        yy = 0.5*np.ones_like(xx)
        zz = 0.5*np.ones_like(xx)
        try:
            xx_agg, yy_agg, zz_agg, v_agg = img.aggregateDataPointsWrtGrid(
                    xx, yy, zz, v,
                    nx, 1, 1, sx, 1.0, 1.0, ox, 0.0, 0.0,
                    op=aggregate_data_op, 
                    logger=logger,
                    **aggregate_data_op_kwargs)
        except Exception as exc:
            err_msg = f"{fname}: aggratating data points in grid failed (`aggregate_data_op='{aggregate_data_op}'`)"
            if logger: logger.error(err_msg)
            raise GrfError(err_msg) from exc

        if len(xx_agg) == 0:
            err_msg = f'{fname}: no data point in grid'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

    if extensionMin is None:
        # default extensionMin
        if cov_range is not None: # known range
            # ... based on range of covariance model
            extensionMin = extension_min(rangeFactorForExtensionMin*cov_range, nx, s=sx)
        else:
            # ... based on dimension
            extensionMin = dimension - 1

    Nmin = nx + extensionMin

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: Computing circulant embedding...')
        else:
            print(f'{fname}: Computing circulant embedding...')

    # Circulant embedding of the covariance matrix
    # --------------------------------------------
    # The embedding matrix is a circulant matrix of size N x N, computed from
    # the covariance function.
    # To take a maximal benefit of Fast Fourier Transform (FFT) for computing DFT,
    # we choose
    #    N = 2^g (a power of 2), with N >= Nmin, N >= 2
    g = int(max(np.ceil(np.log2(Nmin)), 1.0))
    N = int(2**g)

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: embedding dimension: {N}')
        else:
            print(f'{fname}: embedding dimension: {N}')

    # ccirc: coefficient of the embedding matrix (first line), vector of size N
    L = int (N/2)
    h = np.arange(-L, L, dtype=float) * sx # [-L ... 0 ... L-1] * sx
    ccirc = cov_func(h)

    del(h)

    # ...shift first L index to the end of the axis, i.e.:
    #    [-L ... 0 ... L-1] -> [0 ... L-1 -L ... -1]
    ind = np.arange(L)
    ccirc = ccirc[np.hstack((ind+L, ind))]

    del(ind)

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: Computing FFT of circulant matrix...')
        else:
            print(f'{fname}: Computing FFT of circulant matrix...')

    # Compute the Discrete Fourier Transform (DFT) of ccric, via FFT
    # --------------------------------------------------------------
    # The DFT coefficients
    #   lam = DFT(ccirc) = (lam(0),lam(1),...,lam(N-1))
    # are the eigen values of the embedding matrix.
    # We have:
    #   a) lam are real coefficients, because the embedding matrix is symmetric
    #   b) lam(k) = lam(N-k), k=1,...,N-1, because the coefficients ccirc are real
    lam = np.real(np.fft.fft(ccirc))
    # ...note that the imaginary parts are equal to 0

    # -------------------------------------
    # If some DFT coefficients are negative, then set them to zero
    # and update them to fit the marginals distribution (approximate embedding)
    if np.min(lam) < 0:
        lam = np.sum(lam)/np.sum(np.maximum(lam, 0.)) * np.maximum(lam, 0.)

    # Take the square root of the (updated) DFT coefficients
    # ------------------------------------------------------
    lamSqrt = np.sqrt(lam)

    # For specified variance
    # ----------------------
    # Compute updating factor
    if var is not None:
        varUpdate = np.sqrt(var/cov_func(0.))

    # Kriging
    # -------
    # Let
    #    A: index of conditioning nodes
    #    B: index of non-conditioning nodes
    #    Zobs: vector of values at conditioning nodes
    # and
    #        +         +
    #        | rAA rAB |
    #    r = |         |
    #        | rBA rBB |
    #        +         +
    # the covariance matrix, where index A (resp. B) refers to
    # conditioning (resp. non-conditioning) index in the grid.
    #
    # Then, the kriging estimates are
    #     mean + rBA * rAA^(-1) * (v - mean)
    # and the kriging standard deviation
    #    diag(rBB - rBA * rAA^(-1) * rAB)

    # Compute the part rAA of the covariance matrix
    # Note: if a variance var is specified, then the matrix r should be updated
    # by the following operation:
    #    diag((var/cov_func(0))^1/2) * r * diag((var/cov_func(0))^1/2)
    # which is accounting in the computation of kriging estimates and standard
    # deviation below

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: Computing covariance matrix (rAA) for conditioning locations...')
        else:
            print(f'{fname}: Computing covariance matrix (rAA) for conditioning locations...')

    # Compute
    #    indc: node index of conditioning node,
    #          rounded to lower index if between two grid node and index is positive
    indc_f = (xx_agg-origin)/spacing
    indc = indc_f.astype(int)
    indc = indc - 1 * np.all((indc == indc_f, indc > 0), axis=0)

    nc = len(xx_agg)

    # rAA
    rAA = np.zeros((nc, nc))

    diagEntry = ccirc[0] + measureErrVar
    for i in range(nc):
        rAA[i,i] = diagEntry
        for j in range(i+1, nc):
            rAA[i,j] = ccirc[np.mod(indc[j]-indc[i], N)]
            rAA[j,i] = rAA[i,j]

    # Test if rAA is almost singular...
    if 1./np.linalg.cond(rAA) < tolInvKappa:
        err_msg = f'{fname}: conditioning issue: condition number of matrix rAA is too big'
        if logger: logger.error(err_msg)
        raise GrfError(err_msg)

    # Compute:
    #    indnc: node index of non-conditioning node (nearest node)
    indnc = np.asarray(np.setdiff1d(np.arange(nx), indc), dtype=int)
    nnc = len(indnc)

    if mean is None:
        # Set mean for kriging
        mean = np.array([np.mean(v)])

    # Initialize
    krig = np.zeros(nx)
    if computeKrigSD:
        krigSD = np.zeros(nx)

    if mean.size == 1:
        v_agg = v_agg - mean
    else:
        v_agg = v_agg - mean[indc]

    if var is not None and var.size > 1:
        v_agg = 1./varUpdate[indc] * v_agg

    if conditioningMethod == 1:
        # Method ConditioningA
        # --------------------
        if verbose > 1:
            if logger:
                logger.info(f'{fname}: Computing covariance matrix (rBA) for non-conditioning / conditioning locations...')
            else:
                print(f'{fname}: Computing covariance matrix (rBA) for non-conditioning / conditioning locations...')

        # Compute the parts rBA of the covariance matrix (see above)
        # rBA
        rBA = np.zeros((nnc, nc))
        for j in range(nc):
            k = np.mod(indc[j] - indnc, N)
            rBA[:,j] = ccirc[k]

        del(ccirc)

        if verbose > 1:
            if logger:
                logger.info(f'{fname}: Computing rBA * rAA^(-1)...')
            else:
                print(f'{fname}: Computing rBA * rAA^(-1)...')

        # compute rBA * rAA^(-1)
        rBArAAinv = np.dot(rBA, np.linalg.inv(rAA))

        del(rAA)
        if not computeKrigSD:
            del(rBA)

        # Compute kriging estimates
        if verbose > 1:
            if logger:
                logger.info(f'{fname}: computing kriging estimates...')
            else:
                print(f'{fname}: computing kriging estimates...')

        krig[indnc] = np.dot(rBArAAinv, v_agg)
        krig[indc] = v_agg

        if computeKrigSD:
            # Compute kriging standard deviation
            if verbose > 1:
                if logger:
                    logger.info(f'{fname}: computing kriging standard deviation ...')
                else:
                    print(f'{fname}: computing kriging standard deviation ...')

            for j in range(nnc):
                krigSD[indnc[j]] = np.dot(rBArAAinv[j,:], rBA[j,:])
            krigSD[indnc] = np.sqrt(np.maximum(diagEntry - krigSD[indnc], 0.))

            del(rBA)

    elif conditioningMethod == 2:
        # Method ConditioningB
        # --------------------
        if not computeKrigSD:
            del(ccirc)

        if verbose > 1:
            if logger:
                logger.info(f'{fname}: Computing index in the embedding grid for non-conditioning / conditioning locations...')
            else:
                print(f'{fname}: Computing index in the embedding grid for non-conditioning / conditioning locations...')

        # Compute index in the embedding grid for indc and indnc
        # (to allow use of fft)
        indcEmb = indc
        indncEmb = indnc

        # Compute kriging estimates
        if verbose > 1:
            if logger:
                logger.info(f'{fname}: computing kriging estimates...')
            else:
                print(f'{fname}: computing kriging estimates...')

        # Compute
        #    u = rAA^(-1) * v_agg, and then
        #    Z = rBA * u via the circulant embedding of the covariance matrix
        uEmb = np.zeros(N)
        uEmb[indcEmb] = np.linalg.solve(rAA, v_agg)
        Z = np.fft.ifft(lam * np.fft.fft(uEmb))
        # ...note that Im(Z) = 0
        krig[indnc] = np.real(Z[indncEmb])
        krig[indc] = v_agg

        if computeKrigSD:
            # Compute kriging standard deviation
            if verbose > 1:
                if logger:
                    logger.info(f'{fname}: computing kriging standard deviation ...')
                else:
                    print(f'{fname}: computing kriging standard deviation ...')

            for j in range(nnc):
                u = ccirc[np.mod(indc - indnc[j], N)] # j-th row of rBA
                krigSD[indnc[j]] = np.dot(u,np.linalg.solve(rAA, u))

            del(ccirc)

            krigSD[indnc] = np.sqrt(np.maximum(diagEntry - krigSD[indnc], 0.))

    if aggregate_data_op == 'krige' and computeKrigSD:
        # Set kriging standard deviation at grid cell containing a data
        krigSD[indc] = v_agg_std

    # ... update if non-stationary covariance is specified
    if var is not None:
        if var.size > 1:
            krig = varUpdate * krig
        if computeKrigSD:
            krigSD = varUpdate * krigSD

    krig = krig + mean

    if computeKrigSD:
        return krig, krigSD
    else:
        return krig
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def grf2D(
        cov_model,
        dimension, spacing=(1.0, 1.0), origin=(0.0, 0.0),
        x=None, v=None,
        aggregate_data_op=None,
        aggregate_data_op_kwargs=None,
        mean=None, var=None,
        nreal=1,
        extensionMin=None, rangeFactorForExtensionMin=1.0,
        crop=True,
        method=3, conditioningMethod=2,
        measureErrVar=0.0, tolInvKappa=1.e-10,
        verbose=1,
        printInfo=None,
        logger=None):
    """
    Generates Gaussian Random Fields (GRF) in 2D via Fast Fourier Transform (FFT).

    In brief, the GRFs

    - are generated using the given covariance model (`cov_model`),
    - may have a specified mean (`mean`) and variance (`var`), which can be non stationary,
    - may be conditioned to location(s) `x` with value(s) `v`.

    Parameters
    ----------
    cov_model : :class:`geone.covModel.CovModel2D`, or :class:`geone.covModel.CovModel1D`, or function (`callable`)
        covariance model in 2D, or covariance model in 1D interpreted as an omni-
        directional covariance model, or directly a function of covariance (taking
        2D lag vector(s) as argument)

    dimension : 2-tuple of ints
        `dimension=(nx, ny)`, number of cells in the 2D simulation grid along
        each axis

    spacing : 2-tuple of floats, default: (1.0, 1.0)
        `spacing=(sx, sy)`, cell size along each axis

    origin : 2-tuple of floats, default: (0.0, 0.0)
        `origin=(ox, oy)`, origin of the 2D simulation grid (lower-left corner)

    x : 2D array of floats of shape (n, 2), optional
        data points locations, with n the number of data points, each row of `x`
        is the float coordinates of one data point; note: if n=1, a 1D array of
        shape (2,) is accepted

    v : 1D array of floats of shape (n,), optional
        data values at `x` (`v[i]` is the data value at `x[i]`)

    aggregate_data_op : str {'sgs', 'krige', 'min', 'max', 'mean', 'quantile', 'most_freq', 'random'}, optional
        operation used to aggregate data points falling in the same grid cells

        - if `aggregate_data_op='sgs'`: function :func:`geone.covModel.sgs` is used \
        with the covariance model `cov_model` given in arguments
        - if `aggregate_data_op='krige'`: function :func:`geone.covModel.krige` is used \
        with the covariance model `cov_model` given in arguments
        - if `aggregate_data_op='most_freq'`: most frequent value is selected \
        (smallest one if more than one value with the maximal frequence)
        - if `aggregate_data_op='random'`: value from a random point is selected
        - otherwise: the function `numpy.<aggregate_data_op>` is used with the \
        additional parameters given by `aggregate_data_op_kwargs`, note that, e.g. \
        `aggregate_data_op='quantile'` requires the additional parameter \
        `q=<quantile_to_compute>`

        Note: if `aggregate_data_op='sgs'` or `aggregate_data_op='random'`, the
        aggregation is done for each realization (simulation), i.e. each simulation
        on the grid starts with a new set of values in conditioning grid cells;
        if `aggregate_data_op='sgs'` or `aggregate_data_op='krige'`, then
        `cov_model` must be a covariance model and not directly the covariance
        function

        By default (`None`): `aggregate_data_op='sgs'` is used

    aggregate_data_op_kwargs : dict, optional
        keyword arguments to be passed to `geone.covModel.sgs`,
        `geone.covModel.krige`, or `numpy.<aggregate_data_op>`, according to
        the parameter `aggregate_data_op`

    mean : function (`callable`), or array-like of floats, or float, optional
        kriging mean value:

        - if a function: function of two arguments (xi, yi) that returns the mean \
        at location (xi, yi)
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), mean values at grid cells (for \
        non-stationary mean)
        - if a float: same mean value at every grid cell

        By default (`None`): the mean of data value (`v`) (0.0 if no data) is
        considered at every grid cell

    var : function (`callable`), or array-like of floats, or float, optional
        kriging variance value:

        - if a function: function of two arguments (xi, yi) that returns the \
        variance at location (xi, yi)
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), variance values at grid cells (for \
        non-stationary variance)
        - if a float: same variance value at every grid cell

        By default (`None`): not used (use of covariance model only)

    nreal : int, default: 1
        number of realization(s)

    extensionMin : sequence of 2 ints, optional
        minimal extension in cells along each axis (see note 1 below)

        By default (`None`): minimal extension is automatically computed:

        - based on the range of the covariance model, if `cov_model` is given as \
        an instance of :class:`geone.covModel.CovModel1D` (or \
        :class:`geone.covModel.CovModel2D`)
        - set to (`nx-1`, `ny-1`), if `cov_model` is given as a function \
        (`callable`)

    rangeFactorForExtensionMin : float, default: 1.0
        factor by which the ranges of the covariance model are multiplied before
        computing the default minimal extension, if `cov_model` is given as
        an instance of :class:`geone.covModel.CovModel1D` (or
        :class:`geone.covModel.CovModel2D`) and if `extensionMin=None`
        (not used otherwise)

    crop : bool, default: True
        indicates if the extended generated field (simulation) will be cropped to
        original dimension; note that `crop=False` is not valid with conditioning
        or non-stationary mean or non-stationary variance

    method : int, default: 3
        indicates which method is used to generate unconditional simulations;
        for each method the Discrete Fourier Transform (DFT) "lam" of the
        circulant embedding of the covariance matrix is used, and periodic and
        stationary GRFs are generated

        - `method=1` (method A): generate one GRF Z as follows:
            - generate one real gaussian white noise W
            - apply fft (or fft inverse) on W to get X
            - multiply X by "lam" (term by term)
            - apply fft inverse (or fft) to get Z
        - `method=2` (method B, not implemented!): generate one GRF Z as follows:
           - generate directly X (from method A)
           - multiply X by lam (term by term)
           - apply fft inverse (or fft) to get Z
        - `method=3` (method C, default): generate two independent GRFs Z1, Z2 as follows:
           - generate two independant real gaussian white noises W1, W2 and set \
           W = W1 + i * W2
           - apply fft (or fft inverse) on W to get X
           - multiply X by "lam" (term by term)
           - apply fft inverse (or fft) to get Z, and set Z1 = Re(Z), Z2 = Im(Z); \
           note: if `nreal` is odd, the last field is generated using method A

    conditioningMethod : int, default: 2
        indicates which method is used to update the simulations to account for
        conditioning data; let

        * A: index of conditioning cells
        * B: index of non-conditioning cells
        * Zobs: vector of values of the unconditional simulation Z at conditioning cells
        * :math:`r = \\left(\\begin{array}{cc} r_{AA} & r_{AB}\\\\r_{BA} & r_{BB}\\end{array}\\right)` \
        the covariance matrix, where index A (resp. B) refers to conditioning \
        (resp. non-conditioning) index in the grid;

        an unconditional simulation Z is updated into a conditional simulation ZCond as
        follows; let

        * ZCond[A] = Zobs
        * ZCond[B] = Z[B] + rBA * rAA^(-1) * (Zobs - Z[A])

        i.e. the update consists in adding the kriging estimates of the residues
        to an unconditional simulation

        * `conditioningMethod=1` (method CondtioningA): the matrix M = rBA * rAA^(-1) \
        is explicitly computed (warning: could require large amount of memory), \
        then all the simulations are updated by a sum and a multiplication by the \
        matrix M
        * `conditioningMethod=2` (method CondtioningB, default): for each simulation, \
        the linear system rAA * x = Zobs - Z[A] is solved and then, the multiplication \
        by rBA is done via fft

        Note: parameter `conditioningMethod` is used only for conditional simulation

    measureErrVar : float, default: 0.0
        measurement error variance; the error on conditioning data is assumed to
        follow the distrubution N(0, `measureErrVar` * I); i.e.
        rAA + `measureErrVar` * I is considered instead of rAA for stabilizing the
        linear system for this matrix;
        note: parameter `measureErrVar` is used only for conditional simulation

    tolInvKappa : float, default: 1.e-10
        the simulation is stopped if the inverse of the condition number of rAA
        is above `tolInvKappa`;
        note: parameter `tolInvKappa` is used only for conditional simulation

    verbose : int, default: 1
        verbose mode, higher implies more printing (info):

        - 0: no display
        - 1: warnings
        - 2: warnings + basic info
        - 3 (or >2): all information

        note that if an error occurred, it is raised

    printInfo : bool, optional
        deprecated, use `verbose` instead;

        - if `printInfo=False`, `verbose` is set to 1 (overwritten)
        - if `printInfo=True`, `verbose` is set to 3 (overwritten)
        - if `printInfo=None` (default): not used

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    grf : 3D array of shape (`nreal`, n2, n1)
        GRF realizations, with

        * n1 = nx (= dimension[0]), n2 = ny (= dimension[1]), if `crop=True`,
        * but n1 >= nx, n2 >= ny if `crop=False`

        `grf[i, iy, ix]`: value of the i-th realisation at grid cell of index
        ix (resp. iy) along x (resp. y) axis

    Notes
    -----
    1. For reproducing covariance model, the dimension of GRF should be large
    enough; let K an integer such that K*`spacing` is greater or equal to the
    correlation range, then:

    - correlation accross opposite border should be removed by extending \
    the domain sufficiently, i.e.

        `extensionMin` >= K - 1

    - two cells could not be correlated simultaneously regarding both \
    distances between them (with respect to the periodic grid), i.e. one \
    should have

        `dimension+extensionMin` >= 2*K - 1.

    To sum up, `extensionMin` should be chosen such that

        `dimension+extensionMin` >= max(`dimension`, K) + K - 1

    i.e.

        `extensionMin` >= max(K-1, 2*K-`dimension`-1)

    2. For large data set:

    - `conditioningMethod` should be set to 2 for using FFT

    - `measureErrVar` can be set to a small positive value to stabilize the \
    covariance matrix for conditioning locations (solving linear system).

    3. Some mathematical details:

    Discrete Fourier Transform (DFT) of an array x of dim N1 x N2 is given by

        c = DFT(x) = F * x

    where F is the the (N1*N2) x (N1*N2) matrix with coefficients

        F(j,k) = [exp( -i*2*pi*(j^t*k)/(N1*N2) )], j=(j1,j2), k=(k1,k2) in G,

    and

        G = {n=(n1,n2), 0 <= n1 <= N1-1, 0 <= n2 <= N2-1}

    denotes the indices grid and where we use the bijection

        (n1,n2) in G -> n1 + n2 * N1 in {0,...,N1*N2-1},

    between the multiple-indices and the single indices.

    With N = N1*N2, we have

        F^(-1) = 1/N * F^(*)

    where ^(*) denotes the conjugate transpose.

    Let

        Q = 1/N^(1/2) * F

    Then Q is unitary, i.e. Q^(-1) = Q^(*)

    Then, we have

        DFT = F = N^(1/2) * Q,

        DFT^(-1) = 1/N * F^(*) = 1/N^(1/2) * Q^(*)

    Using `numpy` package:

        numpy.fft.fft2() = DFT()

        numpy.fft.ifft2() = DFT^(-1)()
    """
    fname = 'grf2D'

    # Set verbose mode according to printInfo (if given)
    if printInfo is not None:
        if printInfo:
            verbose = 3
        else:
            verbose = 1

    # Check first argument and get covariance function
    if cov_model.__class__.__name__ == 'function':
        # covariance function is given
        cov_func = cov_model
        cov_range = None # unknown range
    elif isinstance(cov_model, gcm.CovModel2D):
        # Prevent calculation if covariance model is not stationary
        if not cov_model.is_stationary():
            err_msg = f'{fname}: `cov_model` is not stationary: {fname} cannot be applied (use `geone.geosclassicinterface` package)'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        cov_func = cov_model.func() # covariance function
        cov_range = cov_model.rxy()
    elif isinstance(cov_model, gcm.CovModel1D):
        cov_model = gcm.covModel1D_to_covModel2D(cov_model) # convert model 1D in 2D
        # -> cov_model will not be modified at exit
        # Prevent calculation if covariance model is not stationary
        if not cov_model.is_stationary():
            err_msg = f'{fname}: `cov_model` is not stationary: {fname} cannot be applied (use `geone.geosclassicinterface` package)'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        cov_func = cov_model.func() # covariance function
        cov_range = cov_model.rxy()
    else:
        err_msg = f'{fname}: `cov_model` invalid'
        if logger: logger.error(err_msg)
        raise GrfError(err_msg)

    # aggregate_data_op (default)
    if aggregate_data_op is None:
        aggregate_data_op = 'sgs'

    # Number of realization(s)
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose > 0:
            if logger:
                logger.warning(f'{fname}: `nreal` <= 0: `None` is returned')
            else:
                print(f'{fname}: WARNING: `nreal` <= 0: `None` is returned')
        return None

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: Preliminary computation...')
        else:
            print(f'{fname}: Preliminary computation...')

    #### Preliminary computation ####
    nx, ny = dimension
    sx, sy = spacing
    ox, oy = origin

    nxy = nx*ny

    if method not in (1, 2, 3):
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise GrfError(err_msg)

    if method == 2:
        err_msg = f'{fname}: `method=2` not implemented'
        if logger: logger.error(err_msg)
        raise GrfError(err_msg)

    if x is None and v is not None:
        err_msg = f'{fname}: `x` is not given (`None`) but `v` is given (not `None`)'
        if logger: logger.error(err_msg)
        raise GrfError(err_msg)

    if x is not None:
        if conditioningMethod not in (1, 2):
            err_msg = f'{fname}: `conditioningMethod` invalid'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        if v is None:
            err_msg = f'{fname}: `x` is given (not `None`) but `v` is not given (`None`)'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        x = np.asarray(x, dtype='float').reshape(-1, 2) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

    mean_x = mean
    if mean is not None:
        if callable(mean):
            if x is not None:
                mean_x = mean(x[:, 0], x[:, 1])
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            yyi, xxi = np.meshgrid(yi, xi, indexing='ij')
            mean = mean(xxi, yyi) # replace function 'mean' by its evaluation on the grid
        else:
            mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if mean.size == 1:
                if x is not None:
                    mean_x = mean
            elif mean.size == nxy:
                mean = mean.reshape(ny, nx)
                if x is not None:
                    mean_x = img.Img_interp_func(img.Img(nx, ny, 1, sx, sy, 1., ox, oy, 0., nv=1, val=mean, logger=logger), iz=0, logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `mean` is not valid'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

    var_x = var
    if var is not None:
        if callable(var):
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            yyi, xxi = np.meshgrid(yi, xi, indexing='ij')
            var = var(xxi, yyi) # replace function 'var' by its evaluation on the grid
        else:
            var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if var.size == 1:
                if x is not None:
                    var_x = var
            elif var.size == nxy:
                var = var.reshape(ny, nx)
                if x is not None:
                    var_x = img.Img_interp_func(img.Img(nx, ny, 1, sx, sy, 1., ox, oy, 0., nv=1, val=var, logger=logger), iz=0, logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `var` is not valid'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

    # data point set from x, v
    if x is not None:
        if aggregate_data_op_kwargs is None:
            aggregate_data_op_kwargs = {}
        if aggregate_data_op == 'krige' or aggregate_data_op == 'sgs':
            if cov_range is None:
                # cov_model is directly the covariance function
                err_msg = f"{fname}: `cov_model` must be a model (not directly a function) when `aggregate_data_op='{aggregate_data_op}'` is used"
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

            # Get grid cell with at least one data point:
            # x_agg: 2D array, each row contains the coordinates of the center of such cell
            try:
                im_tmp = img.imageFromPoints(
                        x, values=None, varname=None,
                        nx=nx, ny=ny, sx=sx, sy=sy, ox=ox, oy=oy,
                        indicator_var=True, 
                        count_var=False,
                        logger=logger)
            except Exception as exc:
                err_msg = f'{fname}: cannot set image from points'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg) from exc

            ind_agg = np.where(im_tmp.val[0])
            if len(ind_agg[0]) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

            x_agg = np.array((im_tmp.xx()[ind_agg].reshape(-1), im_tmp.yy()[ind_agg].reshape(-1))).T
            # x_agg = np.array((im_tmp.xx()[*ind_agg].reshape(-1), im_tmp.yy()[*ind_agg].reshape(-1))).T
            ind_agg = ind_agg[1:] # remove index along z axis
            del(im_tmp)
            # Compute
            # - kriging estimate (v_agg) and kriging std (v_agg_std) at x_agg,
            # - or nreal simulation(s) (v_agg) at x_agg
            if mean is not None and mean.size > 1:
                mean_x_agg = mean[ind_agg]
                # mean_x_agg = mean[*ind_agg]
            else:
                mean_x_agg = mean
            if var is not None and var.size > 1:
                var_x_agg = var[ind_agg]
                # var_x_agg = var[*ind_agg]
            else:
                var_x_agg = var
            if aggregate_data_op == 'krige':
                try:
                    v_agg, v_agg_std = gcm.krige(
                            x, v, x_agg, cov_model, method='simple_kriging',
                            mean_x=mean_x, mean_xu=mean_x_agg,
                            var_x=var_x, var_xu=var_x_agg,
                            verbose=0, logger=logger,
                            **aggregate_data_op_kwargs)
                except Exception as exc:
                    err_msg = f"{fname}: aggratating data points in grid failed (`aggregate_data_op='{aggregate_data_op}'`)"
                    if logger: logger.error(err_msg)
                    raise GrfError(err_msg) from exc

                # all real (same values)
                v_agg = np.tile(v_agg, nreal).reshape(nreal, -1)
            else:
                try:
                    v_agg = gcm.sgs(
                            x, v, x_agg, cov_model, method='simple_kriging',
                            mean_x=mean_x, mean_xu=mean_x_agg,
                            var_x=var_x, var_xu=var_x_agg,
                            nreal=nreal, seed=None,
                            verbose=0, logger=logger,
                            **aggregate_data_op_kwargs)
                except Exception as exc:
                    err_msg = f"{fname}: aggratating data points in grid failed (`aggregate_data_op='{aggregate_data_op}'`)"
                    if logger: logger.error(err_msg)
                    raise GrfError(err_msg) from exc

            xx_agg, yy_agg = x_agg.T
            # zz_agg = 0.5*np.ones_like(xx_agg)
        elif aggregate_data_op == 'random':
            # Aggregate data on grid cell by taking random point
            xx, yy = x.T
            zz = 0.5*np.ones_like(xx)
            # first realization of v_agg
            try:
                xx_agg, yy_agg, zz_agg, v_agg, i_inv = img.aggregateDataPointsWrtGrid(
                        xx, yy, zz, v,
                        nx, ny, nz, sx, sy, sz, ox, oy, oz,
                        op=aggregate_data_op, 
                        return_inverse=True,
                        logger=logger,
                        **aggregate_data_op_kwargs)
            except Exception as exc:
                err_msg = f"{fname}: aggratating data points in grid failed (`aggregate_data_op='{aggregate_data_op}'`)"
                if logger: logger.error(err_msg)
                raise GrfError(err_msg) from exc

            if len(xx_agg) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

            # next realizations of v_agg
            v_agg = np.vstack((v_agg, np.zeros((nreal-1, v_agg.size))))
            for i in range(1, nreal):
                v_agg[i] = [v[np.random.choice(np.where(i_inv==j)[0])] for j in range(len(xx_agg))]
        else:
            # Aggregate data on grid cell by using the given operation
            xx, yy = x.T
            zz = 0.5*np.ones_like(xx)
            try:
                xx_agg, yy_agg, zz_agg, v_agg = img.aggregateDataPointsWrtGrid(
                        xx, yy, zz, v,
                        nx, ny, 1, sx, sy, 1.0, ox, oy, 0.0,
                        op=aggregate_data_op, 
                        logger=logger,
                        **aggregate_data_op_kwargs)
            except Exception as exc:
                err_msg = f"{fname}: aggratating data points in grid failed (`aggregate_data_op='{aggregate_data_op}'`)"
                if logger: logger.error(err_msg)
                raise GrfError(err_msg) from exc

            if len(xx_agg) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

            # all real (same values)
            v_agg = np.tile(v_agg, nreal).reshape(nreal, -1)

    if not crop:
        if x is not None: # conditional simulation
            err_msg = f'{fname}: `crop=False` cannot be used with conditional simulation'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        if mean is not None and mean.size > 1:
            err_msg = f'{fname}: `crop=False` cannot be used with non-stationary mean'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        if var is not None and var.size > 1:
            err_msg = f'{fname}: `crop=False` cannot be used with non-stationary variance'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

    if extensionMin is None:
        # default extensionMin
        if cov_range is not None: # known range
            # ... based on range of covariance model
            extensionMin = [extension_min(rangeFactorForExtensionMin*r, n, s) for r, n, s in zip(cov_range, dimension, spacing)]
        else:
            # ... based on dimension
            extensionMin = [nx-1, ny-1]

    N1min = nx + extensionMin[0]
    N2min = ny + extensionMin[1]

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: Computing circulant embedding...')
        else:
            print(f'{fname}: Computing circulant embedding...')

    # Circulant embedding of the covariance matrix
    # --------------------------------------------
    # The embedding matrix is a (N1,N2)-nested block circulant matrix, computed from
    # the covariance function.
    # To take a maximal benefit of Fast Fourier Transform (FFT) for computing DFT,
    # we choose
    #     N1 = 2^g1 (a power of 2), with N1 >= N1min, N1 >= 2
    #     N2 = 2^g2 (a power of 2), with N2 >= N2min, N2 >= 2
    g1 = int(max(np.ceil(np.log2(N1min)), 1.0))
    g2 = int(max(np.ceil(np.log2(N2min)), 1.0))
    N1 = int(2**g1)
    N2 = int(2**g2)

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: embedding dimension: {N1} x {N2}')
        else:
            print(f'{fname}: embedding dimension: {N1} x {N2}')

    N = N1*N2

    # ccirc: coefficient of the embedding matrix (N2, N1) array
    L1 = int (N1/2)
    L2 = int (N2/2)
    h1 = np.arange(-L1, L1, dtype=float) * sx # [-L1 ... 0 ... L1-1] * sx
    h2 = np.arange(-L2, L2, dtype=float) * sy # [-L2 ... 0 ... L2-1] * sy

    hh = np.meshgrid(h1, h2)
    ccirc = cov_func(np.hstack((hh[0].reshape(-1,1), hh[1].reshape(-1,1))))
    ccirc.resize(N2, N1)

    del(h1, h2, hh)

    # ...shift first L1 index to the end of the axis 1:
    ind = np.arange(L1)
    ccirc = ccirc[:, np.hstack((ind+L1, ind))]
    # ...shift first L2 index to the end of the axis 0:
    ind = np.arange(L2)
    ccirc = ccirc[np.hstack((ind+L2, ind)), :]

    del(ind)

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: Computing FFT of circulant matrix...')
        else:
            print(f'{fname}: Computing FFT of circulant matrix...')

    # Compute the Discrete Fourier Transform (DFT) of ccric, via FFT
    # --------------------------------------------------------------
    # The (2-dimensional) DFT coefficients
    #   lam = DFT(ccirc) = {lam(k1,k2), 0<=k1<=N1-1, 0<=k2<=N2-1}
    # are the eigen values of the embedding matrix.
    # We have:
    #   a) lam are real coefficients, because the embedding matrix is symmetric
    #   b) lam(k1,k2) = lam(N1-k1,N2-k2), 1<=k1<=N1-1, 1<=k2<=N2-1, because the coefficients ccirc are real
    lam = np.real(np.fft.fft2(ccirc))
    # ...note that the imaginary parts are equal to 0

    # Eventual use of approximate embedding
    # -------------------------------------
    # If some DFT coefficients are negative, then set them to zero
    # and update them to fit the marginals distribution (approximate embedding)
    if np.min(lam) < 0:
        lam = np.sum(lam)/np.sum(np.maximum(lam, 0.)) * np.maximum(lam, 0.)

    # Take the square root of the (updated) DFT coefficients
    # ------------------------------------------------------
    lamSqrt = np.sqrt(lam)

    if x is None or conditioningMethod == 1:
        del(lam)

    # For specified variance
    # ----------------------
    # Compute updating factor
    if var is not None:
        varUpdate = np.sqrt(var/cov_func(np.zeros(2)))

    # Dealing with conditioning
    # -------------------------
    if x is not None:
        if verbose > 1:
            if logger:
                logger.info(f'{fname}: Treatment of conditioning data...')
            else:
                print(f'{fname}: Treatment of conditioning data...')
        # Compute the part rAA of the covariance matrix
        #        +         +
        #        | rAA rAB |
        #    r = |         |
        #        | rBA rBB |
        #        +         +
        # where index A (resp. B) refers to
        # conditioning (resp. non-conditioning) index in the grid.

        if verbose > 1:
            if logger:
                logger.info(f'{fname}: Computing covariance matrix (rAA) for conditioning locations...')
            else:
                print(f'{fname}: Computing covariance matrix (rAA) for conditioning locations...')

        # Compute
        #    indc: node index of conditioning node,
        #          rounded to lower index if between two grid node and index is positive
        indc_f = (np.array((xx_agg, yy_agg)).T-origin)/spacing
        indc = indc_f.astype(int)
        indc = indc - 1 * np.all((indc == indc_f, indc > 0), axis=0)
        ix, iy = indc[:, 0], indc[:, 1]

        indc = ix + iy * nx # single-indices

        nc = len(xx_agg)

        # rAA
        rAA = np.zeros((nc, nc))

        diagEntry = ccirc[0, 0] + measureErrVar
        for i in range(nc):
            rAA[i,i] = diagEntry
            for j in range(i+1, nc):
                rAA[i,j] = ccirc[np.mod(iy[j]-iy[i], N2), np.mod(ix[j]-ix[i], N1)]
                rAA[j,i] = rAA[i,j]

        # Test if rAA is almost singular...
        if 1./np.linalg.cond(rAA) < tolInvKappa:
            err_msg = f'{fname}: conditioning issue: condition number of matrix rAA is too big'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        # Compute:
        #    indnc: node index of non-conditioning node (nearest node)
        indnc = np.asarray(np.setdiff1d(np.arange(nxy), indc), dtype=int)
        nnc = len(indnc)

        ky = np.floor_divide(indnc, nx)
        kx = np.mod(indnc, nx)

        if conditioningMethod == 1:
            # Method ConditioningA
            # --------------------
            if verbose > 1:
                if logger:
                    logger.info(f'{fname}: Computing covariance matrix (rBA) for non-conditioning / conditioning locations...')
                else:
                    print(f'{fname}: Computing covariance matrix (rBA) for non-conditioning / conditioning locations...')

            # Compute the parts rBA of the covariance matrix (see above)
            # rBA
            rBA = np.zeros((nnc, nc))
            for j in range(nc):
                rBA[:,j] = ccirc[np.mod(iy[j] - ky, N2), np.mod(ix[j] - kx, N1)]

            if verbose > 1:
                if logger:
                    logger.info(f'{fname}: Computing rBA * rAA^(-1)...')
                else:
                    print(f'{fname}: Computing rBA * rAA^(-1)...')

            # compute rBA * rAA^(-1)
            rBArAAinv = np.dot(rBA, np.linalg.inv(rAA))

            del(rAA, rBA)

            # If a variance var is specified, then the matrix r should be updated
            # by the following operation:
            #    diag((var/cov_func(0))^1/2) * r * diag((var/cov_func(0))^1/2)
            # Hence, if a non-stationary variance is specified,
            # the matrix rBA * rAA^(-1) should be consequently updated
            # by multiplying its columns by 1/varUpdate[indc] and its rows by varUpdate[indnc]
            if var is not None and var.size > 1:
                rBArAAinv = np.transpose(varUpdate.reshape(-1)[indnc] * np.transpose(1./varUpdate.reshape(-1)[indc] * rBArAAinv))

        elif conditioningMethod == 2:
            # Method ConditioningB
            # --------------------
            if verbose > 1:
                if logger:
                    logger.info(f'{fname}: Computing index in the embedding grid for non-conditioning / conditioning locations...')
                else:
                    print(f'{fname}: Computing index in the embedding grid for non-conditioning / conditioning locations...')

            # Compute index in the embedding grid for indc and indnc
            # (to allow use of fft)
            indcEmb =  iy * N1 + ix
            indncEmb = ky * N1 + kx

        del(ix, iy, kx, ky)

        if mean is None:
            # Set mean for grf
            mean = np.array([np.mean(v)])

    else: # x is None (unconditional)
        if mean is None:
            # Set mean for grf
            mean = np.array([0.0])

    del(ccirc)
    #### End of preliminary computation ####

    # Unconditional simulation
    # ========================
    # Method A: Generating one real GRF Z
    # --------
    # 1. Generate a real gaussian white noise W ~ N(0,1) on G (2D grid)
    # 2. Compute Z = Q^(*) D Q * W
    #    [OR: Z = Q D Q^(*) * W], where
    #       Q is normalized DFT matrix
    #       D = diag(lamSqrt)
    #    i.e:
    #       Z = DFT^(-1)(D * DFT(W))
    #       [OR: Z = DFT(D * DFT^(-1)(W))]
    #
    # Method B: Generating one real GRF Z
    # --------
    # Not implemented
    #
    # Method C: Generating two independent real GRFs Z1, Z2
    # --------
    # (If nreal is odd, the last realization is generated using method A.)
    # 1. Generate two independent real gaussian white noises W1,W2 ~ N(0,1) on G (2D grid)
    #    and let W = W1 + i * W2 (complex value)
    # 2. Compute Z = Q^(*) D * W
    #    [OR: Z = Q D * W], where
    #       Q is normalized DFT matrix
    #       D = diag(lamSqrt)
    #    i.e:
    #       Z = N^(1/2) * DFT^(-1)(D * W)
    #       [OR: Z = 1/N^(1/2) * DFT(D * W)]
    #    Then the real and imaginary parts of Z are two independent GRFs
    if crop:
        grfNx, grfNy = nx, ny
    else:
        grfNx, grfNy = N1, N2

    grf = np.zeros((nreal, grfNy, grfNx))

    if method == 1:
        # Method A
        # --------
        for i in range(nreal):
            if verbose > 2:
                if logger:
                    logger.info(f'{fname}: unconditional simulation {i+1:4d} of {nreal:4d}...')
                else:
                    print(f'{fname}: unconditional simulation {i+1:4d} of {nreal:4d}...')

            W = np.random.normal(size=(N2, N1))

            Z = np.fft.ifft2(lamSqrt * np.fft.fft2(W))
            # ...note that Im(Z) = 0
            grf[i] = np.real(Z[0:grfNy, 0:grfNx])

    elif method == 2:
        # Method B
        # --------
        err_msg = f'{fname}: (unconditional simulation) `method=2` not implemented'
        if logger: logger.error(err_msg)
        raise GrfError(err_msg)

    elif method == 3:
        # Method C
        # --------
        for i in np.arange(0, nreal-1, 2):
            if verbose > 2:
                if logger:
                    logger.info(f'{fname}: unconditional simulation {i+1:4d}-{i+2:4d} of {nreal:4d}...')
                else:
                    print(f'{fname}: unconditional simulation {i+1:4d}-{i+2:4d} of {nreal:4d}...')

            W = np.array(np.random.normal(size=(N2, N1)), dtype=complex)
            W.imag = np.random.normal(size=(N2, N1))
            Z = np.sqrt(N) * np.fft.ifft2(lamSqrt * W)
            #  Z = 1/np.sqrt(N) * np.fft.fft2(lamSqrt * W)] # see above: [OR:...]

            grf[i] = np.real(Z[0:grfNy, 0:grfNx])
            grf[i+1] = np.imag(Z[0:grfNy, 0:grfNx])

        if np.mod(nreal, 2) == 1:
            if verbose > 2:
                if logger:
                    logger.info(f'{fname}: unconditional simulation {nreal:4d} of {nreal:4d}...')
                else:
                    print(f'{fname}: unconditional simulation {nreal:4d} of {nreal:4d}...')

            W = np.random.normal(size=(N2, N1))

            Z = np.fft.ifft2(lamSqrt * np.fft.fft2(W))
            # ...note that Im(Z) = 0
            grf[nreal-1] = np.real(Z[0:grfNy, 0:grfNx])

    if var is not None:
        grf = varUpdate * grf

    grf = mean + grf

    # Conditional simulation
    # ----------------------
    # Let
    #    A: index of conditioning nodes
    #    B: index of non-conditioning nodes
    #    Zobs: vector of values at conditioning nodes
    # and
    #        +         +
    #        | rAA rAB |
    #    r = |         |
    #        | rBA rBB |
    #        +         +
    # the covariance matrix, where index A (resp. B) refers to
    # conditioning (resp. non-conditioning) index in the grid.
    #
    # Then, from an unconditional simulation Z, we retrieve a conditional
    # simulation ZCond as follows.
    # Let
    #    ZCond[A] = Zobs
    #    ZCond[B] = Z[B] + rBA * rAA^(-1) * (Zobs - Z[A])
    if x is not None:
        # We work with single indices...
        grf.resize(nreal, grfNx*grfNy)

        if conditioningMethod == 1:
            # Method ConditioningA
            # --------------------
            if verbose > 1:
                if logger:
                    logger.info(f'{fname}: updating conditional simulations...')
                else:
                    print(f'{fname}: updating conditional simulations...')

            # Update all simulations at a time,
            # use the matrix rBA * rAA^(-1) already computed
            grf[:,indnc] = grf[:,indnc] + np.transpose(np.dot(rBArAAinv, np.transpose(v_agg - grf[:,indc])))
            grf[:,indc] = v_agg

        elif conditioningMethod == 2:
            # Method ConditioningB
            # --------------------
            # Update each simulation successively as follows:
            #    - solve rAA * x = Zobs - z[A]
            #    - do the multiplication rBA * x via the circulant embedding of the
            #      covariance matrix (using fft)
            rAAinvResiduEmb = np.zeros(N2*N1)

            for i in range(nreal):
                if verbose > 2:
                    if logger:
                        logger.info(f'{fname}: updating conditional simulation {i+1:4d} of {nreal:4d}...')
                    else:
                        print(f'{fname}: updating conditional simulation {i+1:4d} of {nreal:4d}...')

                # Compute residue
                residu = v_agg[i] - grf[i, indc]
                # ... update if non-stationary variance is specified
                if var is not None and var.size > 1:
                    residu = 1./varUpdate.reshape(-1)[indc] * residu

                # Compute
                #    x = rAA^(-1) * residu, and then
                #    Z = rBA * x via the circulant embedding of the covariance matrix
                rAAinvResiduEmb[indcEmb] = np.linalg.solve(rAA, residu)
                Z = np.fft.ifft2(lam * np.fft.fft2(rAAinvResiduEmb.reshape(N2, N1)))
                # ...note that Im(Z) = 0
                Z = np.real(Z.reshape(-1)[indncEmb])

                # ... update if non-stationary covariance is specified
                if var is not None and var.size > 1:
                    Z = varUpdate.reshape(-1)[indnc] * Z

                grf[i, indnc] = grf[i, indnc] + Z
                grf[i, indc] = v_agg[i]

        # Reshape grf as initially
        grf.resize(nreal, grfNy, grfNx)

    return grf
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def krige2D(
        cov_model,
        dimension, spacing=(1.0, 1.0), origin=(0.0, 0.0),
        x=None, v=None,
        aggregate_data_op=None,
        aggregate_data_op_kwargs=None,
        mean=None, var=None,
        extensionMin=None, rangeFactorForExtensionMin=1.0,
        conditioningMethod=1, # note: set conditioningMethod=2 if unable to allocate memory
        measureErrVar=0.0, tolInvKappa=1.e-10,
        computeKrigSD=True,
        verbose=1,
        printInfo=None,
        logger=None):
    """
    Computes kriging estimates and standard deviations in 2D via FFT.

    It is a simple kriging

    - of value(s) `v` at location(s) `x`,
    - based on the given covariance model (`cov_model`),
    - it may account for a specified mean (`mean`) and variance (`var`), which can be non stationary.

    Parameters
    ----------
    cov_model : :class:`geone.covModel.CovModel2D`, or :class:`geone.covModel.CovModel1D`, or function (`callable`)
        covariance model in 2D, or covariance model in 1D interpreted as an omni-
        directional covariance model, or directly a function of covariance (taking
        2D lag vector(s) as argument)

    dimension : 2-tuple of ints
        `dimension=(nx, ny)`, number of cells in the 2D simulation grid along
        each axis

    spacing : 2-tuple of floats, default: (1.0, 1.0)
        `spacing=(sx, sy)`, cell size along each axis

    origin : 2-tuple of floats, default: (0.0, 0.0)
        `origin=(ox, oy)`, origin of the 2D simulation grid (lower-left corner)

    x : 2D array of floats of shape (n, 2), optional
        data points locations, with n the number of data points, each row of `x`
        is the float coordinates of one data point; note: if n=1, a 1D array of
        shape (2,) is accepted

    v : 1D array of floats of shape (n,), optional
        data values at `x` (`v[i]` is the data value at `x[i]`)

    aggregate_data_op : str {'krige', 'min', 'max', 'mean', 'quantile', 'most_freq'}, optional
        operation used to aggregate data points falling in the same grid cells

        - if `aggregate_data_op='krige'`: function :func:`geone.covModel.krige` is used \
        with the covariance model `cov_model` given in arguments
        - if `aggregate_data_op='most_freq'`: most frequent value is selected \
        (smallest one if more than one value with the maximal frequence)
        - otherwise: the function `numpy.<aggregate_data_op>` is used with the \
        additional parameters given by `aggregate_data_op_kwargs`, note that, e.g. \
        `aggregate_data_op='quantile'` requires the additional parameter \
        `q=<quantile_to_compute>`

        Note: if `aggregate_data_op='krige'`, then `cov_model` must be a
        covariance model and not directly the covariance function

        By default (`None`): `aggregate_data_op='krige'` is used

    aggregate_data_op_kwargs : dict, optional
        keyword arguments to be passed to `geone.covModel.krige`,
        `geone.covModel.krige`, or `numpy.<aggregate_data_op>`, according to
        the parameter `aggregate_data_op`

    mean : function (`callable`), or array-like of floats, or float, optional
        kriging mean value:

        - if a function: function of two arguments (xi, yi) that returns the mean \
        at location (xi, yi)
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), mean values at grid cells (for \
        non-stationary mean)
        - if a float: same mean value at every grid cell

        By default (`None`): the mean of data value (`v`) (0.0 if no data) is
        considered at every grid cell

    var : function (`callable`), or array-like of floats, or float, optional
        kriging variance value:

        - if a function: function of two arguments (xi, yi) that returns the \
        variance at location (xi, yi)
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), variance values at grid cells (for \
        non-stationary variance)
        - if a float: same variance value at every grid cell

        By default (`None`): not used (use of covariance model only)

    extensionMin : sequence of 2 ints, optional
        minimal extension in cells along each axis (see note 1 below)

        By default (`None`): minimal extension is automatically computed:

        - based on the range of the covariance model, if `cov_model` is given as \
        an instance of :class:`geone.covModel.CovModel1D` (or \
        :class:`geone.covModel.CovModel2D`)
        - set to (`nx-1`, `ny-1`), if `cov_model` is given as a function \
        (`callable`)

    rangeFactorForExtensionMin : float, default: 1.0
        factor by which the ranges of the covariance model are multiplied before
        computing the default minimal extension, if `cov_model` is given as
        an instance of :class:`geone.covModel.CovModel1D` (or
        :class:`geone.covModel.CovModel2D`) and if `extensionMin=None`
        (not used otherwise)

    conditioningMethod : int, default: 1
        indicates which method is used to update the simulations to account for
        conditioning data; let

        * A: index of conditioning cells
        * B: index of non-conditioning cells
        * :math:`r = \\left(\\begin{array}{cc} r_{AA} & r_{AB}\\\\r_{BA} & r_{BB}\\end{array}\\right)` \
        the covariance matrix, where index A (resp. B) refers to conditioning \
        (resp. non-conditioning) index in the grid;

        then, thre kriging estimates and kriging variances are

        * krig[B]    = mean + rBA * rAA^(-1) * (v - mean)
        * krigVar[B] = diag(rBB - rBA * rAA^(-1) * rAB)

        and the computation is done according to `conditioningMethod`:

        * `conditioningMethod=1` (method CondtioningA, default): the matrices \
        rBA, RAA^(-1) are explicitly computed (warning: could require large \
        amount of memory)
        * `conditioningMethod=2` (method CondtioningB): for kriging estimates, \
        the linear system rAA * y = (v - mean) is solved, and then mean + rBA*y is \
        computed; for kriging variances, for each column u[j] of rAB, the linear \
        system rAA * y = u[j] is solved, and then rBB[j,j] - y^t*y is computed

        Note: set `conditioningMethod=2` if unable to allocate memory

    measureErrVar : float, default: 0.0
        measurement error variance; the error on conditioning data is assumed to
        follow the distrubution N(0, `measureErrVar` * I); i.e.
        rAA + `measureErrVar` * I is considered instead of rAA for stabilizing the
        linear system for this matrix

    tolInvKappa : float, default: 1.e-10
        the computation is stopped if the inverse of the condition number of rAA
        is above `tolInvKappa`

    computeKrigSD : bool, default: True
        indicates if the kriging standard deviations are computed

    verbose : int, default: 1
        verbose mode, higher implies more printing (info):

        - 0: no display
        - 1: warnings
        - 2: warnings + basic info
        - 3 (or >2): all information

        note that if an error occurred, it is raised

    printInfo : bool, optional
        deprecated, use `verbose` instead;

        - if `printInfo=False`, `verbose` is set to 1 (overwritten)
        - if `printInfo=True`, `verbose` is set to 3 (overwritten)
        - if `printInfo=None` (default): not used

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    krig : 2D array of shape (ny, nx)
        kriging estimates, with (nx, ny) (= dimension);
        `krig[iy, ix]`: value at grid cell of index ix (resp. iy) along x (resp. y)
        axis

    krigSD : 2D array of shape (ny, nx), optional
        kriging standard deviations, with (nx, ny) (= dimension);
        `krigSD[iy, ix]`: value at grid cell of index ix (resp. iy) along x (resp. y)
        axis; returned if `computeKrigSD=True`

    Notes
    -----
    1. For reproducing covariance model, the dimension of GRF should be large
    enough; let K an integer such that K*`spacing` is greater or equal to the
    correlation range, then:

    - correlation accross opposite border should be removed by extending \
    the domain sufficiently, i.e.

        `extensionMin` >= K - 1

    - two cells could not be correlated simultaneously regarding both \
    distances between them (with respect to the periodic grid), i.e. one \
    should have

        `dimension+extensionMin` >= 2*K - 1.

    To sum up, `extensionMin` should be chosen such that

        `dimension+extensionMin` >= max(`dimension`, K) + K - 1

    i.e.

        `extensionMin` >= max(K-1, 2*K-`dimension`-1)

    2. For large data set:

    - `conditioningMethod` should be set to 2 for using FFT

    - `measureErrVar` can be set to a small positive value to stabilize the \
    covariance matrix for conditioning locations (solving linear system).

    3. Some mathematical details:

    Discrete Fourier Transform (DFT) of an array x of dim N1 x N2 is given by

        c = DFT(x) = F * x

    where F is the the (N1*N2) x (N1*N2) matrix with coefficients

        F(j,k) = [exp( -i*2*pi*(j^t*k)/(N1*N2) )], j=(j1,j2), k=(k1,k2) in G,

    and

        G = {n=(n1,n2), 0 <= n1 <= N1-1, 0 <= n2 <= N2-1}

    denotes the indices grid and where we use the bijection

        (n1,n2) in G -> n1 + n2 * N1 in {0,...,N1*N2-1},

    between the multiple-indices and the single indices.

    With N = N1*N2, we have

        F^(-1) = 1/N * F^(*)

    where ^(*) denotes the conjugate transpose.

    Let

        Q = 1/N^(1/2) * F

    Then Q is unitary, i.e. Q^(-1) = Q^(*)

    Then, we have

        DFT = F = N^(1/2) * Q,

        DFT^(-1) = 1/N * F^(*) = 1/N^(1/2) * Q^(*)

    Using `numpy` package:

        numpy.fft.fft2() = DFT()

        numpy.fft.ifft2() = DFT^(-1)()
    """
    fname = 'krige2D'

    # Set verbose mode according to printInfo (if given)
    if printInfo is not None:
        if printInfo:
            verbose = 3
        else:
            verbose = 1

    # Check third argument and get covariance function
    if cov_model.__class__.__name__ == 'function':
        # covariance function is given
        cov_func = cov_model
        cov_range = None # unknown range
    elif isinstance(cov_model, gcm.CovModel2D):
        # Prevent calculation if covariance model is not stationary
        if not cov_model.is_stationary():
            err_msg = f'{fname}: `cov_model` is not stationary: {fname} cannot be applied (use `geone.geosclassicinterface` package)'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        cov_func = cov_model.func() # covariance function
        cov_range = cov_model.rxy()
    elif isinstance(cov_model, gcm.CovModel1D):
        cov_model = gcm.covModel1D_to_covModel2D(cov_model) # convert model 1D in 2D
        # -> cov_model will not be modified at exit
        # Prevent calculation if covariance model is not stationary
        if not cov_model.is_stationary():
            err_msg = f'{fname}: `cov_model` is not stationary: {fname} cannot be applied (use `geone.geosclassicinterface` package)'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        cov_func = cov_model.func() # covariance function
        cov_range = cov_model.rxy()
    else:
        err_msg = f'{fname}: `cov_model` invalid'
        if logger: logger.error(err_msg)
        raise GrfError(err_msg)

    # aggregate_data_op (default)
    if aggregate_data_op is None:
        aggregate_data_op = 'krige'

    nx, ny = dimension
    sx, sy = spacing
    ox, oy = origin

    nxy = nx*ny

    if x is None and v is not None:
        err_msg = f'{fname}: `x` is not given (`None`) but `v` is given (not `None`)'
        if logger: logger.error(err_msg)
        raise GrfError(err_msg)

    if x is not None:
        if conditioningMethod not in (1, 2):
            err_msg = f'{fname}: `conditioningMethod` invalid'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        if v is None:
            err_msg = f'{fname}: `x` is given (not `None`) but `v` is not given (`None`)'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        x = np.asarray(x, dtype='float').reshape(-1, 2) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

    mean_x = mean
    if mean is not None:
        if callable(mean):
            if x is not None:
                mean_x = mean(x[:, 0], x[:, 1])
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            yyi, xxi = np.meshgrid(yi, xi, indexing='ij')
            mean = mean(xxi, yyi) # replace function 'mean' by its evaluation on the grid
        else:
            mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if mean.size == 1:
                if x is not None:
                    mean_x = mean
            elif mean.size == nxy:
                mean = mean.reshape(ny, nx)
                if x is not None:
                    mean_x = img.Img_interp_func(img.Img(nx, ny, 1, sx, sy, 1., ox, oy, 0., nv=1, val=mean, logger=logger), iz=0, logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `mean` is not valid'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

    var_x = var
    if var is not None:
        if callable(var):
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            yyi, xxi = np.meshgrid(yi, xi, indexing='ij')
            var = var(xxi, yyi) # replace function 'var' by its evaluation on the grid
        else:
            var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if var.size == 1:
                if x is not None:
                    var_x = var
            elif var.size == nxy:
                var = var.reshape(ny, nx)
                if x is not None:
                    var_x = img.Img_interp_func(img.Img(nx, ny, 1, sx, sy, 1., ox, oy, 0., nv=1, val=var, logger=logger), iz=0, logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `var` is not valid'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

    if x is None:
        # No data: kriging return the mean and the standard deviation...
        krig = np.zeros((ny, nx))
        if mean is not None:
            krig[...] = mean
        if computeKrigSD:
            krigSD = np.zeros((ny, nx))
            if var is not None:
                krigSD[...] = np.sqrt(var)
            else:
                krigSD[...] = np.sqrt(cov_func(np.zeros(2)))
            return krig, krigSD
        else:
            return krig

    if aggregate_data_op_kwargs is None:
        aggregate_data_op_kwargs = {}

    if aggregate_data_op == 'krige':
        if cov_range is None:
            # cov_model is directly the covariance function
            err_msg = f"{fname}: `cov_model` must be a model (not directly a function) when `aggregate_data_op='{aggregate_data_op}'` is used"
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        # Get grid cell with at least one data point:
        # x_agg: 2D array, each row contains the coordinates of the center of such cell
        try:
            im_tmp = img.imageFromPoints(
                    x, values=None, varname=None,
                    nx=nx, ny=ny, sx=sx, sy=sy, ox=ox, oy=oy,
                    indicator_var=True, 
                    count_var=False,
                    logger=logger)
        except Exception as exc:
            err_msg = f'{fname}: cannot set image from points'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg) from exc

        ind_agg = np.where(im_tmp.val[0])
        if len(ind_agg[0]) == 0:
            err_msg = f'{fname}: no data point in grid'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        x_agg = np.array((im_tmp.xx()[ind_agg].reshape(-1), im_tmp.yy()[ind_agg].reshape(-1))).T
        # x_agg = np.array((im_tmp.xx()[*ind_agg].reshape(-1), im_tmp.yy()[*ind_agg].reshape(-1))).T
        ind_agg = ind_agg[1:] # remove index along z axis
        del(im_tmp)
        # Compute
        # - kriging estimate (v_agg) and kriging std (v_agg_std) at x_agg,
        # - or nreal simulation(s) (v_agg) at x_agg
        if mean is not None and mean.size > 1:
            mean_x_agg = mean[ind_agg]
            # mean_x_agg = mean[*ind_agg]
        else:
            mean_x_agg = mean
        if var is not None and var.size > 1:
            var_x_agg = var[ind_agg]
            # var_x_agg = var[*ind_agg]
        else:
            var_x_agg = var
        try:
            v_agg, v_agg_std = gcm.krige(
                    x, v, x_agg, cov_model, method='simple_kriging',
                    mean_x=mean_x, mean_xu=mean_x_agg,
                    var_x=var_x, var_xu=var_x_agg,
                    verbose=0, logger=logger,
                    **aggregate_data_op_kwargs)
        except Exception as exc:
            err_msg = f"{fname}: aggratating data points in grid failed (`aggregate_data_op='{aggregate_data_op}'`)"
            if logger: logger.error(err_msg)
            raise GrfError(err_msg) from exc

        xx_agg, yy_agg = x_agg.T
        # zz_agg = 0.5*np.ones_like(xx_agg)
    else:
        # Aggregate data on grid cell by using the given operation
        xx, yy = x.T
        zz = 0.5*np.ones_like(xx)
        try:
            xx_agg, yy_agg, zz_agg, v_agg = img.aggregateDataPointsWrtGrid(
                    xx, yy, zz, v,
                    nx, ny, 1, sx, sy, 1.0, ox, oy, 0.0,
                    op=aggregate_data_op, 
                    logger=logger,
                    **aggregate_data_op_kwargs)
        except Exception as exc:
            err_msg = f"{fname}: aggratating data points in grid failed (`aggregate_data_op='{aggregate_data_op}'`)"
            if logger: logger.error(err_msg)
            raise GrfError(err_msg) from exc

        if len(xx_agg) == 0:
            err_msg = f'{fname}: no data point in grid'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

    if extensionMin is None:
        # default extensionMin
        if cov_range is not None: # known range
            # ... based on range of covariance model
            extensionMin = [extension_min(rangeFactorForExtensionMin*r, n, s) for r, n, s in zip(cov_range, dimension, spacing)]
        else:
            # ... based on dimension
            extensionMin = [nx-1, ny-1]

    N1min = nx + extensionMin[0]
    N2min = ny + extensionMin[1]

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: Computing circulant embedding...')
        else:
            print(f'{fname}: Computing circulant embedding...')

    # Circulant embedding of the covariance matrix
    # --------------------------------------------
    # The embedding matrix is a (N1,N2)-nested block circulant matrix, computed from
    # the covariance function.
    # To take a maximal benefit of Fast Fourier Transform (FFT) for computing DFT,
    # we choose
    #     N1 = 2^g1 (a power of 2), with N1 >= N1min, N1 >= 2
    #     N2 = 2^g2 (a power of 2), with N2 >= N2min, N2 >= 2
    g1 = int(max(np.ceil(np.log2(N1min)), 1.0))
    g2 = int(max(np.ceil(np.log2(N2min)), 1.0))
    N1 = int(2**g1)
    N2 = int(2**g2)

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: embedding dimension: {N1} x {N2}')
        else:
            print(f'{fname}: embedding dimension: {N1} x {N2}')

    N = N1*N2

    # ccirc: coefficient of the embedding matrix (N2, N1) array
    L1 = int (N1/2)
    L2 = int (N2/2)
    h1 = np.arange(-L1, L1, dtype=float) * sx # [-L1 ... 0 ... L1-1] * sx
    h2 = np.arange(-L2, L2, dtype=float) * sy # [-L2 ... 0 ... L2-1] * sy

    hh = np.meshgrid(h1, h2)
    ccirc = cov_func(np.hstack((hh[0].reshape(-1,1), hh[1].reshape(-1,1))))
    ccirc.resize(N2, N1)

    del(h1, h2, hh)

    # ...shift first L1 index to the end of the axis 1:
    ind = np.arange(L1)
    ccirc = ccirc[:, np.hstack((ind+L1, ind))]
    # ...shift first L2 index to the end of the axis 0:
    ind = np.arange(L2)
    ccirc = ccirc[np.hstack((ind+L2, ind)), :]

    del(ind)

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: Computing FFT of circulant matrix...')
        else:
            print(f'{fname}: Computing FFT of circulant matrix...')

    # Compute the Discrete Fourier Transform (DFT) of ccric, via FFT
    # --------------------------------------------------------------
    # The (2-dimensional) DFT coefficients
    #   lam = DFT(ccirc) = {lam(k1,k2), 0<=k1<=N1-1, 0<=k2<=N2-1}
    # are the eigen values of the embedding matrix.
    # We have:
    #   a) lam are real coefficients, because the embedding matrix is symmetric
    #   b) lam(k1,k2) = lam(N1-k1,N2-k2), 1<=k1<=N1-1, 1<=k2<=N2-1, because the coefficients ccirc are real
    lam = np.real(np.fft.fft2(ccirc))
    # ...note that the imaginary parts are equal to 0

    # Eventual use of approximate embedding
    # -------------------------------------
    # If some DFT coefficients are negative, then set them to zero
    # and update them to fit the marginals distribution (approximate embedding)
    if np.min(lam) < 0:
        lam = np.sum(lam)/np.sum(np.maximum(lam, 0.)) * np.maximum(lam, 0.)

    # Take the square root of the (updated) DFT coefficients
    # ------------------------------------------------------
    lamSqrt = np.sqrt(lam)

    # For specified variance
    # ----------------------
    # Compute updating factor
    if var is not None:
        varUpdate = np.sqrt(var/cov_func(np.zeros(2)))

    # Kriging
    # -------
    # Let
    #    A: index of conditioning nodes
    #    B: index of non-conditioning nodes
    #    Zobs: vector of values at conditioning nodes
    # and
    #        +         +
    #        | rAA rAB |
    #    r = |         |
    #        | rBA rBB |
    #        +         +
    # the covariance matrix, where index A (resp. B) refers to
    # conditioning (resp. non-conditioning) index in the grid.
    #
    # Then, the kriging estimates are
    #     mean + rBA * rAA^(-1) * (v - mean)
    # and the kriging standard deviation
    #    diag(rBB - rBA * rAA^(-1) * rAB)

    # Compute the part rAA of the covariance matrix
    # Note: if a variance var is specified, then the matrix r should be updated
    # by the following operation:
    #    diag((var/cov_func(0))^1/2) * r * diag((var/cov_func(0))^1/2)
    # which is accounting in the computation of kriging estimates and standard
    # deviation below

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: Computing covariance matrix (rAA) for conditioning locations...')
        else:
            print(f'{fname}: Computing covariance matrix (rAA) for conditioning locations...')

    # Compute
    #    indc: node index of conditioning node,
    #          rounded to lower index if between two grid node and index is positive
    indc_f = (np.array((xx_agg, yy_agg)).T-origin)/spacing
    indc = indc_f.astype(int)
    indc = indc - 1 * np.all((indc == indc_f, indc > 0), axis=0)
    ix, iy = indc[:, 0], indc[:, 1]

    indc = ix + iy * nx # single-indices

    nc = len(xx_agg)

    # rAA
    rAA = np.zeros((nc, nc))

    diagEntry = ccirc[0, 0] + measureErrVar
    for i in range(nc):
        rAA[i,i] = diagEntry
        for j in range(i+1, nc):
            rAA[i,j] = ccirc[np.mod(iy[j]-iy[i], N2), np.mod(ix[j]-ix[i], N1)]
            rAA[j,i] = rAA[i,j]

    # Test if rAA is almost singular...
    if 1./np.linalg.cond(rAA) < tolInvKappa:
        err_msg = f'{fname}: conditioning issue: condition number of matrix rAA is too big'
        if logger: logger.error(err_msg)
        raise GrfError(err_msg)

    # Compute:
    #    indnc: node index of non-conditioning node (nearest node)
    indnc = np.asarray(np.setdiff1d(np.arange(nxy), indc), dtype=int)
    nnc = len(indnc)

    ky = np.floor_divide(indnc, nx)
    kx = np.mod(indnc, nx)

    if mean is None:
        # Set mean for kriging
        mean = np.array([np.mean(v)])

    # Initialize
    krig = np.zeros(ny*nx)
    if computeKrigSD:
        krigSD = np.zeros(ny*nx)

    if mean.size == 1:
        v_agg = v_agg - mean
    else:
        v_agg = v_agg - mean.reshape(-1)[indc]

    if var is not None and var.size > 1:
        v_agg = 1./varUpdate.reshape(-1)[indc] * v_agg

    if conditioningMethod == 1:
        # Method ConditioningA
        # --------------------
        if verbose > 1:
            if logger:
                logger.info(f'{fname}: Computing covariance matrix (rBA) for non-conditioning / conditioning locations...')
            else:
                print(f'{fname}: Computing covariance matrix (rBA) for non-conditioning / conditioning locations...')

        # Compute the parts rBA of the covariance matrix (see above)
        # rBA
        rBA = np.zeros((nnc, nc))
        for j in range(nc):
            rBA[:,j] = ccirc[np.mod(iy[j] - ky, N2), np.mod(ix[j] - kx, N1)]

        del(ix, iy, kx, ky)
        del(ccirc)

        if verbose > 1:
            if logger:
                logger.info(f'{fname}: Computing rBA * rAA^(-1)...')
            else:
                print(f'{fname}: Computing rBA * rAA^(-1)...')

        # compute rBA * rAA^(-1)
        rBArAAinv = np.dot(rBA, np.linalg.inv(rAA))

        del(rAA)
        if not computeKrigSD:
            del(rBA)

        # Compute kriging estimates
        if verbose > 1:
            if logger:
                logger.info(f'{fname}: computing kriging estimates...')
            else:
                print(f'{fname}: computing kriging estimates...')

        krig[indnc] = np.dot(rBArAAinv, v_agg)
        krig[indc] = v_agg

        if computeKrigSD:
            # Compute kriging standard deviation
            if verbose > 1:
                if logger:
                    logger.info(f'{fname}: computing kriging standard deviation ...')
                else:
                    print(f'{fname}: computing kriging standard deviation ...')

            for j in range(nnc):
                krigSD[indnc[j]] = np.dot(rBArAAinv[j,:], rBA[j,:])
            krigSD[indnc] = np.sqrt(np.maximum(diagEntry - krigSD[indnc], 0.))

            del(rBA)

    elif conditioningMethod == 2:
        # Method ConditioningB
        # --------------------
        if not computeKrigSD:
            del(ccirc)

        if verbose > 1:
            if logger:
                logger.info(f'{fname}: Computing index in the embedding grid for non-conditioning / conditioning locations...')
            else:
                print(f'{fname}: Computing index in the embedding grid for non-conditioning / conditioning locations...')

        # Compute index in the embedding grid for indc and indnc
        # (to allow use of fft)
        indcEmb =  iy * N1 + ix
        indncEmb = ky * N1 + kx

        # Compute kriging estimates
        if verbose > 1:
            if logger:
                logger.info(f'{fname}: computing kriging estimates...')
            else:
                print(f'{fname}: computing kriging estimates...')

        # Compute
        #    u = rAA^(-1) * v_agg, and then
        #    Z = rBA * u via the circulant embedding of the covariance matrix
        uEmb = np.zeros(N2*N1)
        uEmb[indcEmb] = np.linalg.solve(rAA, v_agg)
        Z = np.fft.ifft2(lam * np.fft.fft2(uEmb.reshape(N2, N1)))
        # ...note that Im(Z) = 0
        krig[indnc] = np.real(Z.reshape(-1)[indncEmb])
        krig[indc] = v_agg

        if computeKrigSD:
            # Compute kriging standard deviation
            if verbose > 1:
                if logger:
                    logger.info(f'{fname}: computing kriging standard deviation ...')
                else:
                    print(f'{fname}: computing kriging standard deviation ...')

            for j in range(nnc):
                u = ccirc[np.mod(iy - ky[j], N2), np.mod(ix - kx[j], N1)] # j-th row of rBA
                krigSD[indnc[j]] = np.dot(u,np.linalg.solve(rAA, u))

            del(ccirc)

            krigSD[indnc] = np.sqrt(np.maximum(diagEntry - krigSD[indnc], 0.))

        del(ix, iy, kx, ky)

    if aggregate_data_op == 'krige' and computeKrigSD:
        # Set kriging standard deviation at grid cell containing a data
        krigSD[indc] = v_agg_std

    # ... update if non-stationary covariance is specified
    if var is not None:
        if var.size > 1:
            krig = varUpdate.reshape(-1) * krig
        if computeKrigSD:
            krigSD = varUpdate.reshape(-1) * krigSD

    krig.resize(ny, nx)
    if computeKrigSD:
        krigSD.resize(ny, nx)

    krig = krig + mean

    if computeKrigSD:
        return krig, krigSD
    else:
        return krig
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def grf3D(
        cov_model,
        dimension, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
        x=None, v=None,
        aggregate_data_op=None,
        aggregate_data_op_kwargs=None,
        mean=None, var=None,
        nreal=1,
        extensionMin=None, rangeFactorForExtensionMin=1.0,
        crop=True,
        method=3, conditioningMethod=2,
        measureErrVar=0.0, tolInvKappa=1.e-10,
        verbose=1,
        printInfo=None,
        logger=None):
    """
    Generates Gaussian Random Fields (GRF) in 3D via Fast Fourier Transform (FFT).

    In brief, the GRFs

    - are generated using the given covariance model (`cov_model`),
    - may have a specified mean (`mean`) and variance (`var`), which can be non stationary,
    - may be conditioned to location(s) `x` with value(s) `v`.

    Parameters
    ----------
    cov_model : :class:`geone.covModel.CovModel3D`, or :class:`geone.covModel.CovModel1D`, or function (`callable`)
        covariance model in 3D, or covariance model in 1D interpreted as an omni-
        directional covariance model, or directly a function of covariance (taking
        3D lag vector(s) as argument)

    dimension : 3-tuple of ints
        `dimension=(nx, ny, nz)`, number of cells in the 3D simulation grid along
        each axis

    spacing : 3-tuple of floats, default: (1.0,1.0, 1.0)
        `spacing=(sx, sy, sz)`, cell size along each axis

    origin : 3-tuple of floats, default: (0.0, 0.0, 0.0)
        `origin=(ox, oy, oz)`, origin of the 3D simulation grid (bottom-lower-left
        corner)

    x : 2D array of floats of shape (n, 3), optional
        data points locations, with n the number of data points, each row of `x`
        is the float coordinates of one data point; note: if n=1, a 1D array of
        shape (3,) is accepted

    v : 1D array of floats of shape (n,), optional
        data values at `x` (`v[i]` is the data value at `x[i]`)

    aggregate_data_op : str {'sgs', 'krige', 'min', 'max', 'mean', 'quantile', 'most_freq', 'random'}, optional
        operation used to aggregate data points falling in the same grid cells

        - if `aggregate_data_op='sgs'`: function :func:`geone.covModel.sgs` is used \
        with the covariance model `cov_model` given in arguments
        - if `aggregate_data_op='krige'`: function :func:`geone.covModel.krige` is used \
        with the covariance model `cov_model` given in arguments
        - if `aggregate_data_op='most_freq'`: most frequent value is selected \
        (smallest one if more than one value with the maximal frequence)
        - if `aggregate_data_op='random'`: value from a random point is selected
        - otherwise: the function `numpy.<aggregate_data_op>` is used with the \
        additional parameters given by `aggregate_data_op_kwargs`, note that, e.g. \
        `aggregate_data_op='quantile'` requires the additional parameter \
        `q=<quantile_to_compute>`

        Note: if `aggregate_data_op='sgs'` or `aggregate_data_op='random'`, the
        aggregation is done for each realization (simulation), i.e. each simulation
        on the grid starts with a new set of values in conditioning grid cells;
        if `aggregate_data_op='sgs'` or `aggregate_data_op='krige'`, then
        `cov_model` must be a covariance model and not directly the covariance
        function

        By default (`None`): `aggregate_data_op='sgs'` is used

    aggregate_data_op_kwargs : dict, optional
        keyword arguments to be passed to `geone.covModel.sgs`,
        `geone.covModel.krige`, or `numpy.<aggregate_data_op>`, according to
        the parameter `aggregate_data_op`

    mean : function (`callable`), or array-like of floats, or float, optional
        kriging mean value:

        - if a function: function of three arguments (xi, yi, zi) that returns \
        the mean at location (xi, yi, zi)
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), mean values at grid cells (for \
        non-stationary mean)
        - if a float: same mean value at every grid cell

        By default (`None`): the mean of data value (`v`) (0.0 if no data) is
        considered at every grid cell

    var : function (`callable`), or array-like of floats, or float, optional
        kriging variance value:

        - if a function: function of three arguments (xi, yi, yi) that returns \
        the variance at location (xi, yi, zi)
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), variance values at grid cells (for \
        non-stationary variance)
        - if a float: same variance value at every grid cell

        By default (`None`): not used (use of covariance model only)

    nreal : int, default: 1
        number of realization(s)

    extensionMin : sequence of 3 ints, optional
        minimal extension in cells along each axis (see note 1 below)

        By default (`None`): minimal extension is automatically computed:

        - based on the range of the covariance model, if `cov_model` is given as \
        an instance of :class:`geone.covModel.CovModel1D` (or \
        :class:`geone.covModel.CovModel3D`)
        - set to (`nx-1`, `ny-1`, `nz-1`), if `cov_model` is given as a function \
        (`callable`)

    rangeFactorForExtensionMin : float, default: 1.0
        factor by which the ranges of the covariance model are multiplied before
        computing the default minimal extension, if `cov_model` is given as
        an instance of :class:`geone.covModel.CovModel1D` (or
        :class:`geone.covModel.CovModel3D`) and if `extensionMin=None`
        (not used otherwise)

    crop : bool, default: True
        indicates if the extended generated field (simulation) will be cropped to
        original dimension; note that `crop=False` is not valid with conditioning
        or non-stationary mean or non-stationary variance

    method : int, default: 3
        indicates which method is used to generate unconditional simulations;
        for each method the Discrete Fourier Transform (DFT) "lam" of the
        circulant embedding of the covariance matrix is used, and periodic and
        stationary GRFs are generated

        - `method=1` (method A): generate one GRF Z as follows:
            - generate one real gaussian white noise W
            - apply fft (or fft inverse) on W to get X
            - multiply X by "lam" (term by term)
            - apply fft inverse (or fft) to get Z
        - `method=2` (method B, not implemented!): generate one GRF Z as follows:
           - generate directly X (from method A)
           - multiply X by lam (term by term)
           - apply fft inverse (or fft) to get Z
        - `method=3` (method C, default): generate two independent GRFs Z1, Z2 as follows:
           - generate two independant real gaussian white noises W1, W2 and set \
           W = W1 + i * W2
           - apply fft (or fft inverse) on W to get X
           - multiply X by "lam" (term by term)
           - apply fft inverse (or fft) to get Z, and set Z1 = Re(Z), Z2 = Im(Z); \
           note: if `nreal` is odd, the last field is generated using method A

    conditioningMethod : int, default: 2
        indicates which method is used to update the simulations to account for
        conditioning data; let

        * A: index of conditioning cells
        * B: index of non-conditioning cells
        * Zobs: vector of values of the unconditional simulation Z at conditioning cells
        * :math:`r = \\left(\\begin{array}{cc} r_{AA} & r_{AB}\\\\r_{BA} & r_{BB}\\end{array}\\right)` \
        the covariance matrix, where index A (resp. B) refers to conditioning \
        (resp. non-conditioning) index in the grid;

        an unconditional simulation Z is updated into a conditional simulation ZCond as
        follows; let

        * ZCond[A] = Zobs
        * ZCond[B] = Z[B] + rBA * rAA^(-1) * (Zobs - Z[A])

        i.e. the update consists in adding the kriging estimates of the residues
        to an unconditional simulation

        * `conditioningMethod=1` (method CondtioningA): the matrix M = rBA * rAA^(-1) \
        is explicitly computed (warning: could require large amount of memory), \
        then all the simulations are updated by a sum and a multiplication by the \
        matrix M
        * `conditioningMethod=2` (method CondtioningB, default): for each simulation, \
        the linear system rAA * x = Zobs - Z[A] is solved and then, the multiplication \
        by rBA is done via fft

        Note: parameter `conditioningMethod` is used only for conditional simulation

    measureErrVar : float, default: 0.0
        measurement error variance; the error on conditioning data is assumed to
        follow the distrubution N(0, `measureErrVar` * I); i.e.
        rAA + `measureErrVar` * I is considered instead of rAA for stabilizing the
        linear system for this matrix;
        note: parameter `measureErrVar` is used only for conditional simulation

    tolInvKappa : float, default: 1.e-10
        the simulation is stopped if the inverse of the condition number of rAA
        is above `tolInvKappa`;
        note: parameter `tolInvKappa` is used only for conditional simulation

    verbose : int, default: 1
        verbose mode, higher implies more printing (info):

        - 0: no display
        - 1: warnings
        - 2: warnings + basic info
        - 3 (or >2): all information

        note that if an error occurred, it is raised

    printInfo : bool, optional
        deprecated, use `verbose` instead;

        - if `printInfo=False`, `verbose` is set to 1 (overwritten)
        - if `printInfo=True`, `verbose` is set to 3 (overwritten)
        - if `printInfo=None` (default): not used

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    grf : 4D array of shape (`nreal`, n3, n2, n1)
        GRF realizations, with

        * n1 = nx (= dimension[0]), n2 = ny (= dimension[1]), n3 = nz (= dimension[2]), if `crop=True`,
        * but n1 >= nx, n2 >= ny, n3 >= nz if `crop=False`;

        `grf[i, iz, iy, ix]`: value of the i-th realisation at grid cell of index
        ix (resp. iy, iz) along x (resp. y, z) axis

    Notes
    -----
    1. For reproducing covariance model, the dimension of GRF should be large
    enough; let K an integer such that K*`spacing` is greater or equal to the
    correlation range, then:

    - correlation accross opposite border should be removed by extending \
    the domain sufficiently, i.e.

        `extensionMin` >= K - 1

    - two cells could not be correlated simultaneously regarding both \
    distances between them (with respect to the periodic grid), i.e. one \
    should have

        `dimension+extensionMin` >= 2*K - 1.

    To sum up, `extensionMin` should be chosen such that

        `dimension+extensionMin` >= max(`dimension`, K) + K - 1

    i.e.

        `extensionMin` >= max(K-1, 2*K-`dimension`-1)

    2. For large data set:

    - `conditioningMethod` should be set to 2 for using FFT

    - `measureErrVar` can be set to a small positive value to stabilize the \
    covariance matrix for conditioning locations (solving linear system).

    3. Some mathematical details:

    Discrete Fourier Transform (DFT) of an array x of dim N1 x N2 x N3 is
    given by

        c = DFT(x) = F * x

    where F is the the (N1*N2*N3) x (N1*N2*N3) matrix with coefficients

        F(j,k) = [exp( -i*2*pi*(j^t*k)/(N1*N2*N3) )], j=(j1,j2,j3), k=(k1,k2,k3) in G,

    and

        G = {n=(n1,n2,n3), 0 <= n1 <= N1-1, 0 <= n2 <= N2-1, 0 <= n3 <= N3-1}

    denotes the indices grid and where we use the bijection

        (n1,n2,n3) in G -> n1 + n2 * N1 + n3 * N1 * N2 in {0,...,N1*N2*N3-1},

    between the multiple-indices and the single indices.

    With N = N1*N2*N3, we have

        F^(-1) = 1/N * F^(*)

    where ^(*) denotes the conjugate transpose.

    Let

        Q = 1/N^(1/2) * F

    Then Q is unitary, i.e. Q^(-1) = Q^(*)

    Then, we have

        DFT = F = N^(1/2) * Q,

        DFT^(-1) = 1/N * F^(*) = 1/N^(1/2) * Q^(*)

    Using `numpy` package:

        numpy.fft.fftn() = DFT()

        numpy.fft.ifftn() = DFT^(-1)()
    """
    fname = 'grf3D'

    # Set verbose mode according to printInfo (if given)
    if printInfo is not None:
        if printInfo:
            verbose = 3
        else:
            verbose = 1

    # Check first argument and get covariance function
    if cov_model.__class__.__name__ == 'function':
        # covariance function is given
        cov_func = cov_model
        cov_range = None # unknown range
    elif isinstance(cov_model, gcm.CovModel3D):
        # Prevent calculation if covariance model is not stationary
        if not cov_model.is_stationary():
            err_msg = f'{fname}: `cov_model` is not stationary: {fname} cannot be applied (use `geone.geosclassicinterface` package)'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        cov_func = cov_model.func() # covariance function
        cov_range = cov_model.rxyz()
    elif isinstance(cov_model, gcm.CovModel1D):
        cov_model = gcm.covModel1D_to_covModel3D(cov_model) # convert model 1D in 3D
        # -> cov_model will not be modified at exit
        # Prevent calculation if covariance model is not stationary
        if not cov_model.is_stationary():
            err_msg = f'{fname}: `cov_model` is not stationary: {fname} cannot be applied (use `geone.geosclassicinterface` package)'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        cov_func = cov_model.func() # covariance function
        cov_range = cov_model.rxyz()
    else:
        err_msg = f'{fname}: `cov_model` invalid'
        if logger: logger.error(err_msg)
        raise GrfError(err_msg)

    # aggregate_data_op (default)
    if aggregate_data_op is None:
        aggregate_data_op = 'sgs'

    # Number of realization(s)
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose > 0:
            if logger:
                logger.warning(f'{fname}: `nreal` <= 0: `None` is returned')
            else:
                print(f'{fname}: WARNING: `nreal` <= 0: `None` is returned')
        return None

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: Preliminary computation...')
        else:
            print(f'{fname}: Preliminary computation...')

    #### Preliminary computation ####
    nx, ny, nz = dimension
    sx, sy, sz = spacing
    ox, oy, oz = origin

    nxy = nx*ny
    nxyz = nxy * nz

    if method not in (1, 2, 3):
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise GrfError(err_msg)

    if method == 2:
        err_msg = f'{fname}: `method=2` not implemented'
        if logger: logger.error(err_msg)
        raise GrfError(err_msg)

    if x is None and v is not None:
        err_msg = f'{fname}: `x` is not given (`None`) but `v` is given (not `None`)'
        if logger: logger.error(err_msg)
        raise GrfError(err_msg)

    if x is not None:
        if conditioningMethod not in (1, 2):
            err_msg = f'{fname}: `conditioningMethod` invalid'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        if v is None:
            err_msg = f'{fname}: `x` is given (not `None`) but `v` is not given (`None`)'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        x = np.asarray(x, dtype='float').reshape(-1, 3) # cast in 3-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

    mean_x = mean
    if mean is not None:
        if callable(mean):
            if x is not None:
                mean_x = mean(x[:, 0], x[:, 1], x[:, 2])
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            zi = oz + sz*(0.5+np.arange(nz)) # z-coordinate of cell center
            zzi, yyi, xxi = np.meshgrid(zi, yi, xi, indexing='ij')
            mean = mean(xxi, yyi, zzi) # replace function 'mean' by its evaluation on the grid
        else:
            mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if mean.size == 1:
                if x is not None:
                    mean_x = mean
            elif mean.size == nxyz:
                mean = mean.reshape(nz, ny, nx)
                if x is not None:
                    mean_x = img.Img_interp_func(img.Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv=1, val=mean, logger=logger), logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `mean` is not valid'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

    var_x = var
    if var is not None:
        if callable(var):
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            zi = oz + sz*(0.5+np.arange(nz)) # z-coordinate of cell center
            zzi, yyi, xxi = np.meshgrid(zi, yi, xi, indexing='ij')
            var = var(xxi, yyi, zzi) # replace function 'var' by its evaluation on the grid
        else:
            var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if var.size == 1:
                if x is not None:
                    var_x = var
            elif var.size == nxyz:
                var = var.reshape(nz, ny, nx)
                if x is not None:
                    var_x = img.Img_interp_func(img.Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv=1, val=var, logger=logger), logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `var` is not valid'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

    # data point set from x, v
    if x is not None:
        if aggregate_data_op_kwargs is None:
            aggregate_data_op_kwargs = {}
        if aggregate_data_op == 'krige' or aggregate_data_op == 'sgs':
            if cov_range is None:
                # cov_model is directly the covariance function
                err_msg = f"{fname}: `cov_model` must be a model (not directly a function) when `aggregate_data_op='{aggregate_data_op}'` is used"
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

            # Get grid cell with at least one data point:
            # x_agg: 2D array, each row contains the coordinates of the center of such cell
            try:
                im_tmp = img.imageFromPoints(
                        x, values=None, varname=None,
                        nx=nx, ny=ny, nz=nz, sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz,
                        indicator_var=True, 
                        count_var=False,
                        logger=logger)
            except Exception as exc:
                err_msg = f'{fname}: cannot set image from points'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg) from exc

            ind_agg = np.where(im_tmp.val[0])
            if len(ind_agg[0]) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

            x_agg = np.array((im_tmp.xx()[ind_agg].reshape(-1), im_tmp.yy()[ind_agg].reshape(-1), im_tmp.zz()[ind_agg].reshape(-1))).T
            # x_agg = np.array((im_tmp.xx()[*ind_agg].reshape(-1), im_tmp.yy()[*ind_agg].reshape(-1), im_tmp.zz()[*ind_agg].reshape(-1))).T
            del(im_tmp)
            # Compute
            # - kriging estimate (v_agg) and kriging std (v_agg_std) at x_agg,
            # - or nreal simulation(s) (v_agg) at x_agg
            if mean is not None and mean.size > 1:
                mean_x_agg = mean[ind_agg]
                # mean_x_agg = mean[*ind_agg]
            else:
                mean_x_agg = mean
            if var is not None and var.size > 1:
                var_x_agg = var[ind_agg]
                # var_x_agg = var[*ind_agg]
            else:
                var_x_agg = var
            if aggregate_data_op == 'krige':
                try:
                    v_agg, v_agg_std = gcm.krige(
                            x, v, x_agg, cov_model, method='simple_kriging',
                            mean_x=mean_x, mean_xu=mean_x_agg,
                            var_x=var_x, var_xu=var_x_agg,
                            verbose=0, logger=logger,
                            **aggregate_data_op_kwargs)
                except Exception as exc:
                    err_msg = f"{fname}: aggratating data points in grid failed (`aggregate_data_op='{aggregate_data_op}'`)"
                    if logger: logger.error(err_msg)
                    raise GrfError(err_msg) from exc

                # all real (same values)
                v_agg = np.tile(v_agg, nreal).reshape(nreal, -1)
            else:
                try:
                    v_agg = gcm.sgs(
                            x, v, x_agg, cov_model, method='simple_kriging',
                            mean_x=mean_x, mean_xu=mean_x_agg,
                            var_x=var_x, var_xu=var_x_agg,
                            nreal=nreal, seed=None,
                            verbose=0, logger=logger,
                            **aggregate_data_op_kwargs)
                except Exception as exc:
                    err_msg = f"{fname}: aggratating data points in grid failed (`aggregate_data_op='{aggregate_data_op}'`)"
                    if logger: logger.error(err_msg)
                    raise GrfError(err_msg) from exc

            xx_agg, yy_agg, zz_agg = x_agg.T
        elif aggregate_data_op == 'random':
            # Aggregate data on grid cell by taking random point
            xx, yy, zz = x.T
            # first realization of v_agg
            try:
                xx_agg, yy_agg, zz_agg, v_agg, i_inv = img.aggregateDataPointsWrtGrid(
                        xx, yy, zz, v,
                        nx, ny, nz, sx, sy, sz, ox, oy, oz,
                        op=aggregate_data_op, 
                        return_inverse=True,
                        logger=logger,
                        **aggregate_data_op_kwargs)
            except Exception as exc:
                err_msg = f"{fname}: aggratating data points in grid failed (`aggregate_data_op='{aggregate_data_op}'`)"
                if logger: logger.error(err_msg)
                raise GrfError(err_msg) from exc

            if len(xx_agg) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

            # next realizations of v_agg
            v_agg = np.vstack((v_agg, np.zeros((nreal-1, v_agg.size))))
            for i in range(1, nreal):
                v_agg[i] = [v[np.random.choice(np.where(i_inv==j)[0])] for j in range(len(xx_agg))]
        else:
            # Aggregate data on grid cell by using the given operation
            xx, yy, zz = x.T
            try:
                xx_agg, yy_agg, zz_agg, v_agg = img.aggregateDataPointsWrtGrid(
                        xx, yy, zz, v,
                        nx, ny, nz, sx, sy, sz, ox, oy, oz,
                        op=aggregate_data_op, 
                        logger=logger,
                        **aggregate_data_op_kwargs)
            except Exception as exc:
                err_msg = f"{fname}: aggratating data points in grid failed (`aggregate_data_op='{aggregate_data_op}'`)"
                if logger: logger.error(err_msg)
                raise GrfError(err_msg) from exc

            if len(xx_agg) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

            # all real (same values)
            v_agg = np.tile(v_agg, nreal).reshape(nreal, -1)

    if not crop:
        if x is not None: # conditional simulation
            err_msg = f'{fname}: `crop=False` cannot be used with conditional simulation'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        if mean is not None and mean.size > 1:
            err_msg = f'{fname}: `crop=False` cannot be used with non-stationary mean'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        if var is not None and var.size > 1:
            err_msg = f'{fname}: `crop=False` cannot be used with non-stationary variance'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

    if extensionMin is None:
        # default extensionMin
        if cov_range is not None: # known range
            # ... based on range of covariance model
            extensionMin = [extension_min(rangeFactorForExtensionMin*r, n, s) for r, n, s in zip(cov_range, dimension, spacing)]
        else:
            # ... based on dimension
            extensionMin = [nx-1, ny-1, nz-1] # default

    N1min = nx + extensionMin[0]
    N2min = ny + extensionMin[1]
    N3min = nz + extensionMin[2]

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: Computing circulant embedding...')
        else:
            print(f'{fname}: Computing circulant embedding...')

    # Circulant embedding of the covariance matrix
    # --------------------------------------------
    # The embedding matrix is a (N1,N2,N3)-nested block circulant matrix, computed from
    # the covariance function.
    # To take a maximal benefit of Fast Fourier Transform (FFT) for computing DFT,
    # we choose
    #     N1 = 2^g1 (a power of 2), with N1 >= N1min, N1 >= 2
    #     N2 = 2^g2 (a power of 2), with N2 >= N2min, N2 >= 2
    #     N3 = 2^g3 (a power of 2), with N3 >= N3min, N3 >= 2
    g1 = int(max(np.ceil(np.log2(N1min)), 1.0))
    g2 = int(max(np.ceil(np.log2(N2min)), 1.0))
    g3 = int(max(np.ceil(np.log2(N3min)), 1.0))
    N1 = int(2**g1)
    N2 = int(2**g2)
    N3 = int(2**g3)

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: embedding dimension: {N1} x {N2} x {N3}')
        else:
            print(f'{fname}: embedding dimension: {N1} x {N2} x {N3}')

    N12 = N1*N2
    N = N12 * N3

    # ccirc: coefficient of the embedding matrix, (N3, N2, N1) array
    L1 = int (N1/2)
    L2 = int (N2/2)
    L3 = int (N3/2)
    h1 = np.arange(-L1, L1, dtype=float) * sx # [-L1 ... 0 ... L1-1] * sx
    h2 = np.arange(-L2, L2, dtype=float) * sy # [-L2 ... 0 ... L2-1] * sy
    h3 = np.arange(-L3, L3, dtype=float) * sz # [-L3 ... 0 ... L3-1] * sz

    hh = np.meshgrid(h2, h3, h1) # as this! hh[i]: (N3, N2, N1) array
                                 # hh[0]: y-coord, hh[1]: z-coord, hh[2]: x-coord
    ccirc = cov_func(np.hstack((hh[2].reshape(-1,1), hh[0].reshape(-1,1), hh[1].reshape(-1,1))))
    ccirc.resize(N3, N2, N1)

    del(h1, h2, h3, hh)

    # ...shift first L1 index to the end of the axis 2:
    ind = np.arange(L1)
    ccirc = ccirc[:,:, np.hstack((ind+L1, ind))]
    # ...shift first L2 index to the end of the axis 1:
    ind = np.arange(L2)
    ccirc = ccirc[:, np.hstack((ind+L2, ind)), :]
    # ...shift first L3 index to the end of the axis 0:
    ind = np.arange(L3)
    ccirc = ccirc[np.hstack((ind+L3, ind)), :,:]

    del(ind)

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: Computing FFT of circulant matrix...')
        else:
            print(f'{fname}: Computing FFT of circulant matrix...')

    # Compute the Discrete Fourier Transform (DFT) of ccric, via FFT
    # --------------------------------------------------------------
    # The (3-dimensional) DFT coefficients
    #   lam = DFT(ccirc) = {lam(k1,k2,k3), 0<=k1<=N1-1, 0<=k2<=N2-1, 0<=k3<=N3-1}
    # are the eigen values of the embedding matrix.
    # We have:
    #   a) lam are real coefficients, because the embedding matrix is symmetric
    #   b) lam(k1,k2,k3) = lam(N1-k1,N2-k2,N3-k3), 1<=k1<=N1-1, 1<=k2<=N2-1, 1<=k3<=N3-1, because the coefficients ccirc are real
    lam = np.real(np.fft.fftn(ccirc))
    # ...note that the imaginary parts are equal to 0

    # Eventual use of approximate embedding
    # -------------------------------------
    # If some DFT coefficients are negative, then set them to zero
    # and update them to fit the marginals distribution (approximate embedding)
    if np.min(lam) < 0:
        lam = np.sum(lam)/np.sum(np.maximum(lam, 0.)) * np.maximum(lam, 0.)

    # Take the square root of the (updated) DFT coefficients
    # ------------------------------------------------------
    lamSqrt = np.sqrt(lam)

    if x is None or conditioningMethod == 1:
        del(lam)

    # For specified variance
    # ----------------------
    # Compute updating factor
    if var is not None:
        varUpdate = np.sqrt(var/cov_func(np.zeros(3)))

    # Dealing with conditioning
    # -------------------------
    if x is not None:
        if verbose > 1:
            if logger:
                logger.info(f'{fname}: Treatment of conditioning data...')
            else:
                print(f'{fname}: Treatment of conditioning data...')
        # Compute the part rAA of the covariance matrix
        #        +         +
        #        | rAA rAB |
        #    r = |         |
        #        | rBA rBB |
        #        +         +
        # where index A (resp. B) refers to
        # conditioning (resp. non-conditioning) index in the grid.

        if verbose > 1:
            if logger:
                logger.info(f'{fname}: Computing covariance matrix (rAA) for conditioning locations...')
            else:
                print(f'{fname}: Computing covariance matrix (rAA) for conditioning locations...')

        # Compute
        #    indc: node index of conditioning node,
        #          rounded to lower index if between two grid node and index is positive
        indc_f = (np.array((xx_agg, yy_agg, zz_agg)).T-origin)/spacing
        indc = indc_f.astype(int)
        indc = indc - 1 * np.all((indc == indc_f, indc > 0), axis=0)
        ix, iy, iz = indc[:, 0], indc[:, 1], indc[:, 2]

        indc = ix + iy * nx + iz * nxy # single-indices

        nc = len(xx_agg)

        # rAA
        rAA = np.zeros((nc, nc))

        diagEntry = ccirc[0, 0, 0] + measureErrVar
        for i in range(nc):
            rAA[i,i] = diagEntry
            for j in range(i+1, nc):
                rAA[i,j] = ccirc[np.mod(iz[j]-iz[i], N3), np.mod(iy[j]-iy[i], N2), np.mod(ix[j]-ix[i], N1)]
                rAA[j,i] = rAA[i,j]

        # Test if rAA is almost singular...
        if 1./np.linalg.cond(rAA) < tolInvKappa:
            err_msg = f'{fname}: conditioning issue: condition number of matrix rAA is too big'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        # Compute:
        #    indnc: node index of non-conditioning node (nearest node)
        indnc = np.asarray(np.setdiff1d(np.arange(nxyz), indc), dtype=int)
        nnc = len(indnc)

        kz = np.floor_divide(indnc, nxy)
        kk = np.mod(indnc, nxy)
        ky = np.floor_divide(kk, nx)
        kx = np.mod(kk, nx)
        del(kk)

        if conditioningMethod == 1:
            # Method ConditioningA
            # --------------------
            if verbose > 1:
                if logger:
                    logger.info(f'{fname}: Computing covariance matrix (rBA) for non-conditioning / conditioning locations...')
                else:
                    print(f'{fname}: Computing covariance matrix (rBA) for non-conditioning / conditioning locations...')

            # Compute the parts rBA of the covariance matrix (see above)
            # rBA
            rBA = np.zeros((nnc, nc))
            for j in range(nc):
                rBA[:,j] = ccirc[np.mod(iz[j] - kz, N3), np.mod(iy[j] - ky, N2), np.mod(ix[j] - kx, N1)]

            if verbose > 1:
                if logger:
                    logger.info(f'{fname}: Computing rBA * rAA^(-1)...')
                else:
                    print(f'{fname}: Computing rBA * rAA^(-1)...')

            # compute rBA * rAA^(-1)
            rBArAAinv = np.dot(rBA, np.linalg.inv(rAA))

            del(rAA, rBA)

            # If a variance var is specified, then the matrix r should be updated
            # by the following operation:
            #    diag((var/cov_func(0))^1/2) * r * diag((var/cov_func(0))^1/2)
            # Hence, if a non-stationary variance is specified,
            # the matrix rBA * rAA^(-1) should be consequently updated
            # by multiplying its columns by 1/varUpdate[indc] and its rows by varUpdate[indnc]
            if var is not None and var.size > 1:
                rBArAAinv = np.transpose(varUpdate.reshape(-1)[indnc] * np.transpose(1./varUpdate.reshape(-1)[indc] * rBArAAinv))

        elif conditioningMethod == 2:
            # Method ConditioningB
            # --------------------
            if verbose > 1:
                if logger:
                    logger.info(f'{fname}: Computing index in the embedding grid for non-conditioning / conditioning locations...')
                else:
                    print(f'{fname}: Computing index in the embedding grid for non-conditioning / conditioning locations...')

            # Compute index in the embedding grid for indc and indnc
            # (to allow use of fft)
            indcEmb =  iz * N12 + iy * N1 + ix
            indncEmb = kz * N12 + ky * N1 + kx

        del(ix, iy, iz, kx, ky, kz)

        if mean is None:
            # Set mean for grf
            mean = np.array([np.mean(v)])

    else: # x is None (unconditional)
        if mean is None:
            # Set mean for grf
            mean = np.array([0.0])

    del(ccirc)
    #### End of preliminary computation ####

    # Unconditional simulation
    # ========================
    # Method A: Generating one real GRF Z
    # --------
    # 1. Generate a real gaussian white noise W ~ N(0,1) on G (3D grid)
    # 2. Compute Z = Q^(*) D Q * W
    #    [OR: Z = Q D Q^(*) * W], where
    #       Q is normalized DFT matrix
    #       D = diag(lamSqrt)
    #    i.e:
    #       Z = DFT^(-1)(D * DFT(W))
    #       [OR: Z = DFT(D * DFT^(-1)(W))]
    #
    # Method B: Generating one real GRF Z
    # --------
    # Not implemented
    #
    # Method C: Generating two independent real GRFs Z1, Z2
    # --------
    # (If nreal is odd, the last realization is generated using method A.)
    # 1. Generate two independent real gaussian white noises W1,W2 ~ N(0,1) on G (3D grid)
    #    and let W = W1 + i * W2 (complex value)
    # 2. Compute Z = Q^(*) D * W
    #    [OR: Z = Q D * W], where
    #       Q is normalized DFT matrix
    #       D = diag(lamSqrt)
    #    i.e:
    #       Z = N^(1/2) * DFT^(-1)(D * W)
    #       [OR: Z = 1/N^(1/2) * DFT(D * W)]
    #    Then the real and imaginary parts of Z are two independent GRFs
    if crop:
        grfNx, grfNy, grfNz = nx, ny, nz
    else:
        grfNx, grfNy, grfNz = N1, N2, N3

    grf = np.zeros((nreal, grfNz, grfNy, grfNx))

    if method == 1:
        # Method A
        # --------
        for i in range(nreal):
            if verbose > 2:
                if logger:
                    logger.info(f'{fname}: unconditional simulation {i+1:4d} of {nreal:4d}...')
                else:
                    print(f'{fname}: unconditional simulation {i+1:4d} of {nreal:4d}...')

            W = np.random.normal(size=(N3, N2, N1))

            Z = np.fft.ifftn(lamSqrt * np.fft.fftn(W))
            # ...note that Im(Z) = 0
            grf[i] = np.real(Z[0:grfNz, 0:grfNy, 0:grfNx])

    elif method == 2:
        # Method B
        # --------
        err_msg = f'{fname}: (unconditional simulation) `method=2` not implemented'
        if logger: logger.error(err_msg)
        raise GrfError(err_msg)

    elif method == 3:
        # Method C
        # --------
        for i in np.arange(0, nreal-1, 2):
            if verbose > 2:
                if logger:
                    logger.info(f'{fname}: unconditional simulation {i+1:4d}-{i+2:4d} of {nreal:4d}...')
                else:
                    print(f'{fname}: unconditional simulation {i+1:4d}-{i+2:4d} of {nreal:4d}...')

            W = np.array(np.random.normal(size=(N3, N2, N1)), dtype=complex)
            W.imag = np.random.normal(size=(N3, N2, N1))
            Z = np.sqrt(N) * np.fft.ifftn(lamSqrt * W)
            #  Z = 1/np.sqrt(N) * np.fft.fftn(lamSqrt * W)] # see above: [OR:...]

            grf[i] = np.real(Z[0:grfNz, 0:grfNy, 0:grfNx])
            grf[i+1] = np.imag(Z[0:grfNz, 0:grfNy, 0:grfNx])

        if np.mod(nreal, 2) == 1:
            if verbose > 2:
                if logger:
                    logger.info(f'{fname}: unconditional simulation {nreal:4d} of {nreal:4d}...')
                else:
                    print(f'{fname}: unconditional simulation {nreal:4d} of {nreal:4d}...')

            W = np.random.normal(size=(N3, N2, N1))

            Z = np.fft.ifftn(lamSqrt * np.fft.fftn(W))
            # ...note that Im(Z) = 0
            grf[nreal-1] = np.real(Z[0:grfNz, 0:grfNy, 0:grfNx])

    if var is not None:
        grf = varUpdate * grf

    grf = mean + grf

    # Conditional simulation
    # ----------------------
    # Let
    #    A: index of conditioning nodes
    #    B: index of non-conditioning nodes
    #    Zobs: vector of values at conditioning nodes
    # and
    #        +         +
    #        | rAA rAB |
    #    r = |         |
    #        | rBA rBB |
    #        +         +
    # the covariance matrix, where index A (resp. B) refers to
    # conditioning (resp. non-conditioning) index in the grid.
    #
    # Then, from an unconditional simulation Z, we retrieve a conditional
    # simulation ZCond as follows.
    # Let
    #    ZCond[A] = Zobs
    #    ZCond[B] = Z[B] + rBA * rAA^(-1) * (Zobs - Z[A])
    if x is not None:
        # We work with single indices...
        grf.resize(nreal, grfNx*grfNy*grfNz)

        if conditioningMethod == 1:
            # Method ConditioningA
            # --------------------
            if verbose > 1:
                if logger:
                    logger.info(f'{fname}: updating conditional simulations...')
                else:
                    print(f'{fname}: updating conditional simulations...')

            # Update all simulations at a time,
            # use the matrix rBA * rAA^(-1) already computed
            grf[:,indnc] = grf[:,indnc] + np.transpose(np.dot(rBArAAinv, np.transpose(v_agg - grf[:,indc])))
            grf[:,indc] = v_agg

        elif conditioningMethod == 2:
            # Method ConditioningB
            # --------------------
            # Update each simulation successively as follows:
            #    - solve rAA * x = Zobs - z[A]
            #    - do the multiplication rBA * x via the circulant embedding of the
            #      covariance matrix (using fft)
            rAAinvResiduEmb = np.zeros(N3*N2*N1)

            for i in range(nreal):
                if verbose > 2:
                    if logger:
                        logger.info(f'{fname}: updating conditional simulation {i+1:4d} of {nreal:4d}...')
                    else:
                        print(f'{fname}: updating conditional simulation {i+1:4d} of {nreal:4d}...')

                # Compute residue
                residu = v_agg[i] - grf[i, indc]
                # ... update if non-stationary variance is specified
                if var is not None and var.size > 1:
                    residu = 1./varUpdate.reshape(-1)[indc] * residu

                # Compute
                #    x = rAA^(-1) * residu, and then
                #    Z = rBA * x via the circulant embedding of the covariance matrix
                rAAinvResiduEmb[indcEmb] = np.linalg.solve(rAA, residu)
                Z = np.fft.ifftn(lam * np.fft.fftn(rAAinvResiduEmb.reshape(N3, N2, N1)))
                # ...note that Im(Z) = 0
                Z = np.real(Z.reshape(-1)[indncEmb])

                # ... update if non-stationary covariance is specified
                if var is not None and var.size > 1:
                    Z = varUpdate.reshape(-1)[indnc] * Z

                grf[i, indnc] = grf[i, indnc] + Z
                grf[i, indc] = v_agg[i]

        # Reshape grf as initially
        grf.resize(nreal, grfNz, grfNy, grfNx)

    return grf
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def krige3D(
        cov_model,
        dimension, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
        x=None, v=None,
        aggregate_data_op=None,
        aggregate_data_op_kwargs=None,
        mean=None, var=None,
        extensionMin=None, rangeFactorForExtensionMin=1.0,
        conditioningMethod=1, # note: set conditioningMethod=2 if unable to allocate memory
        measureErrVar=0.0, tolInvKappa=1.e-10,
        computeKrigSD=True,
        verbose=1,
        printInfo=None,
        logger=None):
    """
    Computes kriging estimates and standard deviations in 3D via FFT.

    It is a simple kriging

    - of value(s) `v` at location(s) `x`,
    - based on the given covariance model (`cov_model`),
    - it may account for a specified mean (`mean`) and variance (`var`), which can be non stationary.

    Parameters
    ----------
    cov_model : :class:`geone.covModel.CovModel3D`, or :class:`geone.covModel.CovModel1D`, or function (`callable`)
        covariance model in 3D, or covariance model in 1D interpreted as an omni-
        directional covariance model, or directly a function of covariance (taking
        3D lag vector(s) as argument)

    dimension : 3-tuple of ints
        `dimension=(nx, ny, nz)`, number of cells in the 3D simulation grid along
        each axis

    spacing : 3-tuple of floats, default: (1.0,1.0, 1.0)
        `spacing=(sx, sy, sz)`, cell size along each axis

    origin : 3-tuple of floats, default: (0.0, 0.0, 0.0)
        `origin=(ox, oy, oz)`, origin of the 3D simulation grid (bottom-lower-left
        corner)

    x : 2D array of floats of shape (n, 3), optional
        data points locations, with n the number of data points, each row of `x`
        is the float coordinates of one data point; note: if n=1, a 1D array of
        shape (3,) is accepted

    v : 1D array of floats of shape (n,), optional
        data values at `x` (`v[i]` is the data value at `x[i]`)

    aggregate_data_op : str {'krige', 'min', 'max', 'mean', 'quantile', 'most_freq'}, optional
        operation used to aggregate data points falling in the same grid cells

        - if `aggregate_data_op='krige'`: function :func:`geone.covModel.krige` is used \
        with the covariance model `cov_model` given in arguments
        - if `aggregate_data_op='most_freq'`: most frequent value is selected \
        (smallest one if more than one value with the maximal frequence)
        - otherwise: the function `numpy.<aggregate_data_op>` is used with the \
        additional parameters given by `aggregate_data_op_kwargs`, note that, e.g. \
        `aggregate_data_op='quantile'` requires the additional parameter \
        `q=<quantile_to_compute>`

        Note: if `aggregate_data_op='krige'`, then `cov_model` must be a
        covariance model and not directly the covariance function

        By default (`None`): `aggregate_data_op='krige'` is used

    aggregate_data_op_kwargs : dict, optional
        keyword arguments to be passed to `geone.covModel.krige`,
        `geone.covModel.krige`, or `numpy.<aggregate_data_op>`, according to
        the parameter `aggregate_data_op`

    aggregate_data_op_kwargs : dict, optional
        keyword arguments to be passed to `geone.covModel.krige` or
        `numpy.<aggregate_data_op>`, according to the parameter
        `aggregate_data_op`

    mean : function (`callable`), or array-like of floats, or float, optional
        kriging mean value:

        - if a function: function of three arguments (xi, yi, zi) that returns \
        the mean at location (xi, yi, zi)
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), mean values at grid cells (for \
        non-stationary mean)
        - if a float: same mean value at every grid cell

        By default (`None`): the mean of data value (`v`) (0.0 if no data) is
        considered at every grid cell

    var : function (`callable`), or array-like of floats, or float, optional
        kriging variance value:

        - if a function: function of three arguments (xi, yi, yi) that returns \
        the variance at location (xi, yi, zi)
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), variance values at grid cells (for \
        non-stationary variance)
        - if a float: same variance value at every grid cell

        By default (`None`): not used (use of covariance model only)

    extensionMin : sequence of 3 ints, optional
        minimal extension in cells along each axis (see note 1 below)

        By default (`None`): minimal extension is automatically computed:

        - based on the range of the covariance model, if `cov_model` is given as \
        an instance of :class:`geone.covModel.CovModel1D` (or \
        :class:`geone.covModel.CovModel3D`)
        - set to (`nx-1`, `ny-1`, `nz-1`), if `cov_model` is given as a function \
        (`callable`)

    rangeFactorForExtensionMin : float, default: 1.0
        factor by which the ranges of the covariance model are multiplied before
        computing the default minimal extension, if `cov_model` is given as
        an instance of :class:`geone.covModel.CovModel1D` (or
        :class:`geone.covModel.CovModel3D`) and if `extensionMin=None`
        (not used otherwise)

    conditioningMethod : int, default: 1
        indicates which method is used to update the simulations to account for
        conditioning data; let

        * A: index of conditioning cells
        * B: index of non-conditioning cells
        * :math:`r = \\left(\\begin{array}{cc} r_{AA} & r_{AB}\\\\r_{BA} & r_{BB}\\end{array}\\right)` \
        the covariance matrix, where index A (resp. B) refers to conditioning \
        (resp. non-conditioning) index in the grid;

        then, thre kriging estimates and kriging variances are

        * krig[B]    = mean + rBA * rAA^(-1) * (v - mean)
        * krigVar[B] = diag(rBB - rBA * rAA^(-1) * rAB)

        and the computation is done according to `conditioningMethod`:

        * `conditioningMethod=1` (method CondtioningA, default): the matrices \
        rBA, RAA^(-1) are explicitly computed (warning: could require large \
        amount of memory)
        * `conditioningMethod=2` (method CondtioningB): for kriging estimates, \
        the linear system rAA * y = (v - mean) is solved, and then mean + rBA*y is \
        computed; for kriging variances, for each column u[j] of rAB, the linear \
        system rAA * y = u[j] is solved, and then rBB[j,j] - y^t*y is computed

        Note: set `conditioningMethod=2` if unable to allocate memory

    measureErrVar : float, default: 0.0
        measurement error variance; the error on conditioning data is assumed to
        follow the distrubution N(0, `measureErrVar` * I); i.e.
        rAA + `measureErrVar` * I is considered instead of rAA for stabilizing the
        linear system for this matrix

    tolInvKappa : float, default: 1.e-10
        the computation is stopped if the inverse of the condition number of rAA
        is above `tolInvKappa`

    computeKrigSD : bool, default: True
        indicates if the kriging standard deviations are computed

    verbose : int, default: 1
        verbose mode, higher implies more printing (info):

        - 0: no display
        - 1: warnings
        - 2: warnings + basic info
        - 3 (or >2): all information

        note that if an error occurred, it is raised

    printInfo : bool, optional
        deprecated, use `verbose` instead;

        - if `printInfo=False`, `verbose` is set to 1 (overwritten)
        - if `printInfo=True`, `verbose` is set to 3 (overwritten)
        - if `printInfo=None` (default): not used

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    krig : 3D array of shape (nz, ny, nx)
        kriging estimates, with (nx, ny, nz) (= dimension);
        `krig[iz, iy, ix]`: value at grid cell of index ix (resp. iy, iz) along x
        (resp. y, z) axis

    krigSD : 3D array of shape (nz, ny, nx), optional
        kriging standard deviations, with (nx, ny, nz) (= dimension);
        `krigSD[iz, iy, ix]`: value at grid cell of index ix (resp. iy, iz) along x
        (resp. y, z) axis; returned if `computeKrigSD=True`

    Notes
    -----
    1. For reproducing covariance model, the dimension of GRF should be large
    enough; let K an integer such that K*`spacing` is greater or equal to the
    correlation range, then:

    - correlation accross opposite border should be removed by extending \
    the domain sufficiently, i.e.

        `extensionMin` >= K - 1

    - two cells could not be correlated simultaneously regarding both \
    distances between them (with respect to the periodic grid), i.e. one \
    should have

        `dimension+extensionMin` >= 2*K - 1.

    To sum up, `extensionMin` should be chosen such that

        `dimension+extensionMin` >= max(`dimension`, K) + K - 1

    i.e.

        `extensionMin` >= max(K-1, 2*K-`dimension`-1)

    2. For large data set:

    - `conditioningMethod` should be set to 2 for using FFT

    - `measureErrVar` can be set to a small positive value to stabilize the \
    covariance matrix for conditioning locations (solving linear system).

    3. Some mathematical details:

    Discrete Fourier Transform (DFT) of an array x of dim N1 x N2 x N3 is
    given by

        c = DFT(x) = F * x

    where F is the the (N1*N2*N3) x (N1*N2*N3) matrix with coefficients

        F(j,k) = [exp( -i*2*pi*(j^t*k)/(N1*N2*N3) )], j=(j1,j2,j3), k=(k1,k2,k3) in G,

    and

        G = {n=(n1,n2,n3), 0 <= n1 <= N1-1, 0 <= n2 <= N2-1, 0 <= n3 <= N3-1}

    denotes the indices grid and where we use the bijection

        (n1,n2,n3) in G -> n1 + n2 * N1 + n3 * N1 * N2 in {0,...,N1*N2*N3-1},

    between the multiple-indices and the single indices.

    With N = N1*N2*N3, we have

        F^(-1) = 1/N * F^(*)

    where ^(*) denotes the conjugate transpose.

    Let

        Q = 1/N^(1/2) * F

    Then Q is unitary, i.e. Q^(-1) = Q^(*)

    Then, we have

        DFT = F = N^(1/2) * Q,

        DFT^(-1) = 1/N * F^(*) = 1/N^(1/2) * Q^(*)

    Using `numpy` package:

        numpy.fft.fftn() = DFT()

        numpy.fft.ifftn() = DFT^(-1)()
    """
    fname = 'krige3D'

    # Set verbose mode according to printInfo (if given)
    if printInfo is not None:
        if printInfo:
            verbose = 3
        else:
            verbose = 1

    # Check third argument and get covariance function
    if cov_model.__class__.__name__ == 'function':
        # covariance function is given
        cov_func = cov_model
        cov_range = None # unknown range
    elif isinstance(cov_model, gcm.CovModel3D):
        # Prevent calculation if covariance model is not stationary
        if not cov_model.is_stationary():
            err_msg = f'{fname}: `cov_model` is not stationary: {fname} cannot be applied (use `geone.geosclassicinterface` package)'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        cov_func = cov_model.func() # covariance function
        cov_range = cov_model.rxyz()
    elif isinstance(cov_model, gcm.CovModel1D):
        cov_model = gcm.covModel1D_to_covModel3D(cov_model) # convert model 1D in 3D
        # -> cov_model will not be modified at exit
        # Prevent calculation if covariance model is not stationary
        if not cov_model.is_stationary():
            err_msg = f'{fname}: `cov_model` is not stationary: {fname} cannot be applied (use `geone.geosclassicinterface` package)'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        cov_func = cov_model.func() # covariance function
        cov_range = cov_model.rxyz()
    else:
        err_msg = f'{fname}: `cov_model` invalid'
        if logger: logger.error(err_msg)
        raise GrfError(err_msg)

    # aggregate_data_op (default)
    if aggregate_data_op is None:
        aggregate_data_op = 'krige'

    nx, ny, nz = dimension
    sx, sy, sz = spacing
    ox, oy, oz = origin

    nxy = nx*ny
    nxyz = nxy * nz

    if x is None and v is not None:
        err_msg = f'{fname}: `x` is not given (`None`) but `v` is given (not `None`)'
        if logger: logger.error(err_msg)
        raise GrfError(err_msg)

    if x is not None:
        if conditioningMethod not in (1, 2):
            err_msg = f'{fname}: `conditioningMethod` invalid'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        if v is None:
            err_msg = f'{fname}: `x` is given (not `None`) but `v` is not given (`None`)'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        x = np.asarray(x, dtype='float').reshape(-1, 3) # cast in 3-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

    mean_x = mean
    if mean is not None:
        if callable(mean):
            if x is not None:
                mean_x = mean(x[:, 0], x[:, 1], x[:, 2])
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            zi = oz + sz*(0.5+np.arange(nz)) # z-coordinate of cell center
            zzi, yyi, xxi = np.meshgrid(zi, yi, xi, indexing='ij')
            mean = mean(xxi, yyi, zzi) # replace function 'mean' by its evaluation on the grid
        else:
            mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if mean.size == 1:
                if x is not None:
                    mean_x = mean
            elif mean.size == nxyz:
                mean = mean.reshape(nz, ny, nx)
                if x is not None:
                    mean_x = img.Img_interp_func(img.Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv=1, val=mean, logger=logger), logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `mean` is not valid'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

    var_x = var
    if var is not None:
        if callable(var):
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            zi = oz + sz*(0.5+np.arange(nz)) # z-coordinate of cell center
            zzi, yyi, xxi = np.meshgrid(zi, yi, xi, indexing='ij')
            var = var(xxi, yyi, zzi) # replace function 'var' by its evaluation on the grid
        else:
            var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if var.size == 1:
                if x is not None:
                    var_x = var
            elif var.size == nxyz:
                var = var.reshape(nz, ny, nx)
                if x is not None:
                    var_x = img.Img_interp_func(img.Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv=1, val=var, logger=logger), logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `var` is not valid'
                if logger: logger.error(err_msg)
                raise GrfError(err_msg)

    if x is None:
        # No data: kriging return the mean and the standard deviation...
        krig = np.zeros((nz, ny, nx))
        if mean is not None:
            krig[...] = mean
        if computeKrigSD:
            krigSD = np.zeros((nz, ny, nx))
            if var is not None:
                krigSD[...] = np.sqrt(var)
            else:
                krigSD[...] = np.sqrt(cov_func(np.zeros(3)))
            return krig, krigSD
        else:
            return krig

    if aggregate_data_op_kwargs is None:
        aggregate_data_op_kwargs = {}

    if aggregate_data_op == 'krige':
        if cov_range is None:
            # cov_model is directly the covariance function
            err_msg = f"{fname}: `cov_model` must be a model (not directly a function) when `aggregate_data_op='{aggregate_data_op}'` is used"
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        # Get grid cell with at least one data point:
        # x_agg: 2D array, each row contains the coordinates of the center of such cell
        try:
            im_tmp = img.imageFromPoints(
                    x, values=None, varname=None,
                    nx=nx, ny=ny, nz=nz, sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz,
                    indicator_var=True, 
                    count_var=False,
                    logger=logger)
        except Exception as exc:
            err_msg = f'{fname}: cannot set image from points'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg) from exc

        ind_agg = np.where(im_tmp.val[0])
        if len(ind_agg[0]) == 0:
            err_msg = f'{fname}: no data point in grid'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

        x_agg = np.array((im_tmp.xx()[ind_agg].reshape(-1), im_tmp.yy()[ind_agg].reshape(-1), im_tmp.zz()[ind_agg].reshape(-1))).T
        # x_agg = np.array((im_tmp.xx()[*ind_agg].reshape(-1), im_tmp.yy()[*ind_agg].reshape(-1), im_tmp.zz()[*ind_agg].reshape(-1))).T
        del(im_tmp)
        # Compute
        # - kriging estimate (v_agg) and kriging std (v_agg_std) at x_agg,
        # - or nreal simulation(s) (v_agg) at x_agg
        if mean is not None and mean.size > 1:
            mean_x_agg = mean[ind_agg]
            # mean_x_agg = mean[*ind_agg]
        else:
            mean_x_agg = mean
        if var is not None and var.size > 1:
            var_x_agg = var[ind_agg]
            # var_x_agg = var[*ind_agg]
        else:
            var_x_agg = var
        try:
            v_agg, v_agg_std = gcm.krige(
                    x, v, x_agg, cov_model, method='simple_kriging',
                    mean_x=mean_x, mean_xu=mean_x_agg,
                    var_x=var_x, var_xu=var_x_agg,
                    verbose=0, logger=logger,
                    **aggregate_data_op_kwargs)
        except Exception as exc:
            err_msg = f"{fname}: aggratating data points in grid failed (`aggregate_data_op='{aggregate_data_op}'`)"
            if logger: logger.error(err_msg)
            raise GrfError(err_msg) from exc

        xx_agg, yy_agg, zz_agg = x_agg.T
    else:
        # Aggregate data on grid cell by using the given operation
        xx, yy, zz = x.T
        try:
            xx_agg, yy_agg, zz_agg, v_agg = img.aggregateDataPointsWrtGrid(
                    xx, yy, zz, v,
                    nx, ny, nz, sx, sy, sz, ox, oy, oz,
                    op=aggregate_data_op, 
                    logger=logger,
                    **aggregate_data_op_kwargs)
        except Exception as exc:
            err_msg = f"{fname}: aggratating data points in grid failed (`aggregate_data_op='{aggregate_data_op}'`)"
            if logger: logger.error(err_msg)
            raise GrfError(err_msg) from exc

        if len(xx_agg) == 0:
            err_msg = f'{fname}: no data point in grid'
            if logger: logger.error(err_msg)
            raise GrfError(err_msg)

    if extensionMin is None:
        # default extensionMin
        if cov_range is not None: # known range
            # ... based on range of covariance model
            extensionMin = [extension_min(rangeFactorForExtensionMin*r, n, s) for r, n, s in zip(cov_range, dimension, spacing)]
        else:
            # ... based on dimension
            extensionMin = [nx-1, ny-1, nz-1] # default

    N1min = nx + extensionMin[0]
    N2min = ny + extensionMin[1]
    N3min = nz + extensionMin[2]

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: Computing circulant embedding...')
        else:
            print(f'{fname}: Computing circulant embedding...')

    # Circulant embedding of the covariance matrix
    # --------------------------------------------
    # The embedding matrix is a (N1,N2,N3)-nested block circulant matrix, computed from
    # the covariance function.
    # To take a maximal benefit of Fast Fourier Transform (FFT) for computing DFT,
    # we choose
    #     N1 = 2^g1 (a power of 2), with N1 >= N1min, N1 >= 2
    #     N2 = 2^g2 (a power of 2), with N2 >= N2min, N2 >= 2
    #     N3 = 2^g3 (a power of 2), with N3 >= N3min, N3 >= 2
    g1 = int(max(np.ceil(np.log2(N1min)), 1.0))
    g2 = int(max(np.ceil(np.log2(N2min)), 1.0))
    g3 = int(max(np.ceil(np.log2(N3min)), 1.0))
    N1 = int(2**g1)
    N2 = int(2**g2)
    N3 = int(2**g3)

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: embedding dimension: {N1} x {N2} x {N3}')
        else:
            print(f'{fname}: embedding dimension: {N1} x {N2} x {N3}')

    N12 = N1*N2
    N = N12 * N3

    # ccirc: coefficient of the embedding matrix, (N3, N2, N1) array
    L1 = int (N1/2)
    L2 = int (N2/2)
    L3 = int (N3/2)
    h1 = np.arange(-L1, L1, dtype=float) * sx # [-L1 ... 0 ... L1-1] * sx
    h2 = np.arange(-L2, L2, dtype=float) * sy # [-L2 ... 0 ... L2-1] * sy
    h3 = np.arange(-L3, L3, dtype=float) * sz # [-L3 ... 0 ... L3-1] * sz

    hh = np.meshgrid(h2, h3, h1) # as this! hh[i]: (N3, N2, N1) array
                                 # hh[0]: y-coord, hh[1]: z-coord, hh[2]: x-coord
    ccirc = cov_func(np.hstack((hh[2].reshape(-1,1), hh[0].reshape(-1,1), hh[1].reshape(-1,1))))
    ccirc.resize(N3, N2, N1)

    del(h1, h2, h3, hh)

    # ...shift first L1 index to the end of the axis 2:
    ind = np.arange(L1)
    ccirc = ccirc[:,:, np.hstack((ind+L1, ind))]
    # ...shift first L2 index to the end of the axis 1:
    ind = np.arange(L2)
    ccirc = ccirc[:, np.hstack((ind+L2, ind)), :]
    # ...shift first L3 index to the end of the axis 0:
    ind = np.arange(L3)
    ccirc = ccirc[np.hstack((ind+L3, ind)), :,:]

    del(ind)

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: Computing FFT of circulant matrix...')
        else:
            print(f'{fname}: Computing FFT of circulant matrix...')

    # Compute the Discrete Fourier Transform (DFT) of ccric, via FFT
    # --------------------------------------------------------------
    # The (3-dimensional) DFT coefficients
    #   lam = DFT(ccirc) = {lam(k1,k2,k3), 0<=k1<=N1-1, 0<=k2<=N2-1, 0<=k3<=N3-1}
    # are the eigen values of the embedding matrix.
    # We have:
    #   a) lam are real coefficients, because the embedding matrix is symmetric
    #   b) lam(k1,k2,k3) = lam(N1-k1,N2-k2,N3-k3), 1<=k1<=N1-1, 1<=k2<=N2-1, 1<=k3<=N3-1, because the coefficients ccirc are real
    lam = np.real(np.fft.fftn(ccirc))
    # ...note that the imaginary parts are equal to 0

    # Eventual use of approximate embedding
    # -------------------------------------
    # If some DFT coefficients are negative, then set them to zero
    # and update them to fit the marginals distribution (approximate embedding)
    if np.min(lam) < 0:
        lam = np.sum(lam)/np.sum(np.maximum(lam, 0.)) * np.maximum(lam, 0.)

    # Take the square root of the (updated) DFT coefficients
    # ------------------------------------------------------
    lamSqrt = np.sqrt(lam)

    # For specified variance
    # ----------------------
    # Compute updating factor
    if var is not None:
        varUpdate = np.sqrt(var/cov_func(np.zeros(3)))

    # Kriging
    # -------
    # Let
    #    A: index of conditioning nodes
    #    B: index of non-conditioning nodes
    #    Zobs: vector of values at conditioning nodes
    # and
    #        +         +
    #        | rAA rAB |
    #    r = |         |
    #        | rBA rBB |
    #        +         +
    # the covariance matrix, where index A (resp. B) refers to
    # conditioning (resp. non-conditioning) index in the grid.
    #
    # Then, the kriging estimates are
    #     mean + rBA * rAA^(-1) * (v - mean)
    # and the kriging standard deviation
    #    diag(rBB - rBA * rAA^(-1) * rAB)

    # Compute the part rAA of the covariance matrix
    # Note: if a variance var is specified, then the matrix r should be updated
    # by the following operation:
    #    diag((var/cov_func(0))^1/2) * r * diag((var/cov_func(0))^1/2)
    # which is accounting in the computation of kriging estimates and standard
    # deviation below

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: Computing covariance matrix (rAA) for conditioning locations...')
        else:
            print(f'{fname}: Computing covariance matrix (rAA) for conditioning locations...')

    # Compute
    #    indc: node index of conditioning node,
    #          rounded to lower index if between two grid node and index is positive
    indc_f = (np.array((xx_agg, yy_agg, zz_agg)).T-origin)/spacing
    indc = indc_f.astype(int)
    indc = indc - 1 * np.all((indc == indc_f, indc > 0), axis=0)
    ix, iy, iz = indc[:, 0], indc[:, 1], indc[:, 2]

    indc = ix + iy * nx + iz * nxy # single-indices

    nc = len(xx_agg)

    # rAA
    rAA = np.zeros((nc, nc))

    diagEntry = ccirc[0, 0, 0] + measureErrVar
    for i in range(nc):
        rAA[i,i] = diagEntry
        for j in range(i+1, nc):
            rAA[i,j] = ccirc[np.mod(iz[j]-iz[i], N3), np.mod(iy[j]-iy[i], N2), np.mod(ix[j]-ix[i], N1)]
            rAA[j,i] = rAA[i,j]

    # Test if rAA is almost singular...
    if 1./np.linalg.cond(rAA) < tolInvKappa:
        err_msg = f'{fname}: conditioning issue: condition number of matrix rAA is too big'
        if logger: logger.error(err_msg)
        raise GrfError(err_msg)

    # Compute:
    #    indnc: node index of non-conditioning node (nearest node)
    indnc = np.asarray(np.setdiff1d(np.arange(nxyz), indc), dtype=int)
    nnc = len(indnc)

    kz = np.floor_divide(indnc, nxy)
    kk = np.mod(indnc, nxy)
    ky = np.floor_divide(kk, nx)
    kx = np.mod(kk, nx)
    del(kk)

    if mean is None:
        # Set mean for kriging
        mean = np.array([np.mean(v)])

    # Initialize
    krig = np.zeros(nz*ny*nx)
    if computeKrigSD:
        krigSD = np.zeros(nz*ny*nx)

    if mean.size == 1:
        v_agg = v_agg - mean
    else:
        v_agg = v_agg - mean.reshape(-1)[indc]

    if var is not None and var.size > 1:
        v_agg = 1./varUpdate.reshape(-1)[indc] * v_agg

    if conditioningMethod == 1:
        # Method ConditioningA
        # --------------------
        if verbose > 1:
            if logger:
                logger.info(f'{fname}: Computing covariance matrix (rBA) for non-conditioning / conditioning locations...')
            else:
                print(f'{fname}: Computing covariance matrix (rBA) for non-conditioning / conditioning locations...')

        # Compute the parts rBA of the covariance matrix (see above)
        # rBA
        rBA = np.zeros((nnc, nc))
        for j in range(nc):
            rBA[:,j] = ccirc[np.mod(iz[j] - kz, N3), np.mod(iy[j] - ky, N2), np.mod(ix[j] - kx, N1)]

        del(ix, iy, iz, kx, ky, kz)
        del(ccirc)

        if verbose > 1:
            if logger:
                logger.info(f'{fname}: Computing rBA * rAA^(-1)...')
            else:
                print(f'{fname}: Computing rBA * rAA^(-1)...')

        # compute rBA * rAA^(-1)
        rBArAAinv = np.dot(rBA, np.linalg.inv(rAA))

        del(rAA)
        if not computeKrigSD:
            del(rBA)

        # Compute kriging estimates
        if verbose > 1:
            if logger:
                logger.info(f'{fname}: computing kriging estimates...')
            else:
                print(f'{fname}: computing kriging estimates...')

        krig[indnc] = np.dot(rBArAAinv, v_agg)
        krig[indc] = v_agg

        if computeKrigSD:
            # Compute kriging standard deviation
            if verbose > 1:
                if logger:
                    logger.info(f'{fname}: computing kriging standard deviation ...')
                else:
                    print(f'{fname}: computing kriging standard deviation ...')

            for j in range(nnc):
                krigSD[indnc[j]] = np.dot(rBArAAinv[j,:], rBA[j,:])
            krigSD[indnc] = np.sqrt(np.maximum(diagEntry - krigSD[indnc], 0.))

            del(rBA)

    elif conditioningMethod == 2:
        # Method ConditioningB
        # --------------------
        if not computeKrigSD:
            del(ccirc)

        if verbose > 1:
            if logger:
                logger.info(f'{fname}: Computing index in the embedding grid for non-conditioning / conditioning locations...')
            else:
                print(f'{fname}: Computing index in the embedding grid for non-conditioning / conditioning locations...')

        # Compute index in the embedding grid for indc and indnc
        # (to allow use of fft)
        indcEmb =  iz * N12 + iy * N1 + ix
        indncEmb = kz * N12 + ky * N1 + kx

        # Compute kriging estimates
        if verbose > 1:
            if logger:
                logger.info(f'{fname}: computing kriging estimates...')
            else:
                print(f'{fname}: computing kriging estimates...')

        # Compute
        #    u = rAA^(-1) * v_agg, and then
        #    Z = rBA * u via the circulant embedding of the covariance matrix
        uEmb = np.zeros(N3*N2*N1)
        uEmb[indcEmb] = np.linalg.solve(rAA, v_agg)
        Z = np.fft.ifftn(lam * np.fft.fftn(uEmb.reshape(N3, N2, N1)))
        # ...note that Im(Z) = 0
        krig[indnc] = np.real(Z.reshape(-1)[indncEmb])
        krig[indc] = v_agg

        if computeKrigSD:
            # Compute kriging standard deviation
            if verbose > 1:
                if logger:
                    logger.info(f'{fname}: computing kriging standard deviation ...')
                else:
                    print(f'{fname}: computing kriging standard deviation ...')

            for j in range(nnc):
                u = ccirc[np.mod(iz - kz[j], N3), np.mod(iy - ky[j], N2), np.mod(ix - kx[j], N1)] # j-th row of rBA
                krigSD[indnc[j]] = np.dot(u,np.linalg.solve(rAA, u))

            del(ccirc)

            krigSD[indnc] = np.sqrt(np.maximum(diagEntry - krigSD[indnc], 0.))

        del(ix, iy, iz, kx, ky, kz)

    if aggregate_data_op == 'krige' and computeKrigSD:
        # Set kriging standard deviation at grid cell containing a data
        krigSD[indc] = v_agg_std

    # ... update if non-stationary covariance is specified
    if var is not None:
        if var.size > 1:
            krig = varUpdate.reshape(-1) * krig
        if computeKrigSD:
            krigSD = varUpdate.reshape(-1) * krigSD

    krig.resize(nz, ny, nx)
    if computeKrigSD:
        krigSD.resize(nz, ny, nx)

    krig = krig + mean

    if computeKrigSD:
        return krig, krigSD
    else:
        return krig
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Module 'geone.grf' example:")

    import time
    import matplotlib.pyplot as plt
    import pyvista as pv

    from geone import img
    from geone import imgplot as imgplt
    from geone import imgplot3d as imgplt3
    from geone import covModel as gcm

    ########## 1D case ##########
    # Define grid
    nx = 2000
    sx = 0.5
    ox = 0.0
    # Define covariance model
    cov_model1 = gcm.CovModel1D(elem=[
                    ('gaussian', {'w':8.95, 'r':100}), # elementary contribution
                    ('nugget', {'w':0.05})             # elementary contribution
                    ], name='')

    # Define mean and variance of GRF
    mean = 10.
    # mean = np.linspace(5, 15, nx)
    var = None
    # var = np.linspace(1, 200, nx)

    # Define hard data
    x = [10., 50., 400., 800.]
    v = [ 8.,  9.,   8.,  12.]
    # x, v = None, None

    # Set number of realizations
    nreal = 2000

    # Set seed
    np.random.seed(123)

    # Generate GRF
    t1 = time.time()
    grf1 = grf1D(cov_model1, nx, sx, origin=ox,
                 nreal=nreal, mean=mean, var=var,
                 x=x, v=v,
                 method=3, conditioningMethod=2 ) # grf1: (nreal,nx) array
    t2 = time.time()

    time_case1D = t2-t1
    nreal_case1D = nreal
    infogrid_case1D = f'grid: {nx} cells'
    # print(f'Elapsed time: {time_case1D} sec')

    grf1_mean = np.mean(grf1, axis=0) # mean along axis 0
    grf1_std = np.std(grf1, axis=0) # standard deviation along axis 0

    if x is not None:
        # Kriging
        t1 = time.time()
        krig1, krig1_std = krige1D(x, v, cov_model1, nx, sx, origin=ox,
                               mean=mean, var=var,
                               conditioningMethod=2)
        t2 = time.time()
        time_krig_case1D = t2-t1
        #print(f'Elapsed time for kriging: {time_krig_case1D} sec')

        peak_to_peak_mean1 = np.ptp(grf1_mean - krig1)
        peak_to_peak_std1  = np.ptp(grf1_std - krig1_std)
        krig1D_done = True
    else:
        krig1D_done = False

    # Display
    # -------
    # xg: center of grid points
    xg = ox + sx * (0.5 + np.arange(nx))

    # === 4 real and mean and sd of all real
    fig, ax = plt.subplots(figsize=(20,10))
    for i in range(4):
        plt.plot(xg, grf1[i], label=f'real #{i+1}')

    plt.plot(xg, grf1_mean, c='black', ls='dashed', label=f'mean ({nreal} real)')
    plt.fill_between(xg, grf1_mean - grf1_std, grf1_mean + grf1_std, color='gray', alpha=0.5, label=f'mean +/- sd ({nreal} real)')

    if x is not None:
        plt.plot(x, v,'+k', markersize=10)

    plt.legend()
    plt.title('GRF1D')

    # fig.show()
    plt.show()

    if x is not None:
        # === 4 real and kriging estimates and sd
        fig, ax = plt.subplots(figsize=(20,10))
        for i in range(4):
            plt.plot(xg, grf1[i], label=f'real #{i+1}')

        plt.plot(xg, krig1, c='black', ls='dashed', label='kriging')
        plt.fill_between(xg, krig1 - krig1_std, krig1 + krig1_std, color='gray', alpha=0.5, label='kriging +/- sd')

        plt.plot(x,v,'+k', markersize=10)
        plt.legend()
        plt.title('GRF1D AND KRIGE1D')

        # fig.show()
        plt.show()

        # === comparison of mean and sd of all real, with kriging estimates and sd
        fig, ax = plt.subplots(figsize=(20,10))
        plt.plot(xg, grf1_mean - krig1, c='black', label='grf1_mean - krig')
        plt.plot(xg, grf1_std - krig1_std, c='red', label='grf1_std - krig1_std')

        plt.axhline(y=0)
        for xx in x:
            plt.axvline(x=xx)

        plt.legend()
        plt.title(f'GRF1D and KRIGE1D / nreal={nreal}')

        # fig.show()
        plt.show()

        del(krig1, krig1_std)

    del (grf1, grf1_mean, grf1_std)

    ########## 2D case ##########
    # Define grid
    nx, ny = 231, 249
    sx, sy = 1., 1.
    ox, oy = 0., 0.

    dimension = [nx, ny]
    spacing = [sx, sy]
    origin = [ox, oy]

    # Define covariance model
    cov_model2 = gcm.CovModel2D(elem=[
                    ('gaussian', {'w':8.5, 'r':[150, 40]}), # elementary contribution
                    ('nugget', {'w':0.5})                   # elementary contribution
                    ], alpha=-30., name='')

    # Define mean and variance of GRF
    mean = 10.
    # mean = sum(np.meshgrid(np.linspace(2, 8, nx), np.linspace(2, 8, ny)))
    var = None
    # var = sum(np.meshgrid(np.linspace(2, 100, nx), np.linspace(2, 100, ny)))

    # Define hard data
    x = np.array([[ 10.,  20.], # 1st point
                  [ 50.,  40.], # 2nd point
                  [ 20., 150.], # 3rd point
                  [200., 210.]]) # 4th point
    v = [ 8.,  9.,   8.,  12.] # values
    # x, v = None, None

    # Set number of realizations
    nreal = 1000

    # Set seed
    np.random.seed(123)

    # Generate GRF
    t1 = time.time()
    grf2 = grf2D(cov_model2, dimension, spacing, origin=origin,
                nreal=nreal, mean=mean, var = var,
                x=x, v=v,
                method=3, conditioningMethod=2) # grf2: (nreal,ny,nx) array
    t2 = time.time()
    nreal_case2D = nreal
    time_case2D = t2-t1
    infogrid_case2D = 'grid: {nx*ny} cells ({nx} x {ny})'
    # print(f'Elapsed time: {time_case2D} sec')

    # Fill an image (Img class) (for display, see below)
    im2 = img.Img(nx, ny, 1, sx, sy, 1., ox, oy, 0., nv=nreal, val=grf2)
    del(grf2)

    # Compute mean and standard deviation over the realizations
    im2_mean = img.imageContStat(im2, op='mean') # pixel-wise mean
    im2_std = img.imageContStat(im2, op='std')   # pixel-wise standard deviation
    # grf2_mean = np.mean(grf.reshape(nreal, -1), axis=0).reshape(ny, nx)
    # grf2_std = np.std(grf.reshape(nreal, -1), axis=0).reshape(ny, nx)

    if x is not None:
        # Kriging
        t1 = time.time()
        krig2, krig2_std = krige2D(x, v, cov_model2, dimension, spacing, origin=origin,
                                mean=mean, var=var,
                                conditioningMethod=2)
        t2 = time.time()
        time_krig_case2D = t2-t1
        # print(f'Elapsed time for kriging: {time_krig_case2D} sec')

        # Fill an image (Img class) (for display, see below)
        im2_krig = img.Img(nx, ny, 1, sx, sy, 1., ox, oy, 0., nv=2, val=np.array((krig2, krig2_std)))
        del(krig2, krig2_std)

        peak_to_peak_mean2 = np.ptp(im2_mean.val[0] - im2_krig.val[0])
        peak_to_peak_std2  = np.ptp(im2_mean.val[0] - im2_krig.val[1])
        krig2D_done = True
    else:
        krig2D_done = False

    # Display (using geone.imgplot)
    # -------
    # === 4 real and mean and standard deviation of all real
    #     and kriging estimates and standard deviation (if conditional)
    if x is not None:
        nc = 4
    else:
        nc = 3

    fig, ax = plt.subplots(2, nc, figsize=(24,12))
    # 4 first real ...
    pnum = [1, 2, nc+1, nc+2]
    for i in range(4):
        plt.subplot(2, nc, pnum[i])
        imgplt.drawImage2D(im2, iv=i)
        if x is not None:
            plt.plot(x[:,0],x[:,1],'+k', markersize=10)

        plt.title(f'GRF2D {cov_model2.name}: real #{i+1}')

    # mean of all real
    plt.subplot(2, nc, 3)
    imgplt.drawImage2D(im2_mean)
    if x is not None:
        plt.plot(x[:,0],x[:,1],'+k', markersize=10)

    plt.title(f'Mean over {nreal} real')

    # standard deviation of all real
    plt.subplot(2, nc, nc+3)
    imgplt.drawImage2D(im2_std, cmap='viridis')
    if x is not None:
        plt.plot(x[:,0],x[:,1],'+k', markersize=10)

    plt.title(f'St. dev. over {nreal} real')

    if x is not None:
        # kriging estimates
        plt.subplot(2, nc, 4)
        imgplt.drawImage2D(im2_krig, iv=0)
        plt.plot(x[:,0],x[:,1],'+k', markersize=10)
        plt.title('Kriging estimates')

        # kriging standard deviation
        plt.subplot(2, nc, nc+4)
        imgplt.drawImage2D(im2_krig, iv=1, cmap='viridis')
        plt.plot(x[:,0],x[:,1],'+k', markersize=10)
        plt.title('Kriging st. dev.')

    plt.suptitle('GRF2D and KRIGE2D')

    # fig.show()
    plt.show()

    if x is not None:
        # === comparison of mean and st. dev. of all real, with kriging estimates and st. dev.
        fig, ax = plt.subplots(1,2,figsize=(15,5))

        # grf mean - kriging estimates
        im_tmp = img.copyImg(im2_mean)
        im_tmp.val[0] = im_tmp.val[0] - im2_krig.val[0]
        plt.subplot(1,2,1)
        imgplt.drawImage2D(im_tmp, cmap='viridis')
        plt.plot(x[:,0],x[:,1],'+k', markersize=10)
        plt.title(f'grf mean - kriging estimates / nreal={nreal}')

        # grf st. dev. - kriging st. dev.
        im_tmp = img.copyImg(im2_std)
        im_tmp.val[0] = im_tmp.val[0] - im2_krig.val[1]
        plt.subplot(1,2,2)
        imgplt.drawImage2D(im_tmp, cmap='viridis')
        plt.plot(x[:,0],x[:,1],'+k', markersize=10)
        plt.title(f'grf st. dev. - kriging st. dev. / nreal={nreal}')

        plt.suptitle('GRF2D and KRIGE2D: comparisons')
        # fig.show()
        plt.show()

        del(im2_krig)

    del(im2, im2_mean, im2_std)

    ########## 3D case ##########
    # Define grid
    nx, ny, nz = 85, 56, 34
    sx, sy, sz = 1., 1., 1.
    ox, oy, oz = 0., 0., 0.

    dimension = [nx, ny, nz]
    spacing = [sx, sy, sz]
    origin = [ox, oy, oz]

    # Define covariance model
    cov_model3 = gcm.CovModel3D(elem=[
                    ('gaussian', {'w':8.5, 'r':[40, 20, 10]}), # elementary contribution
                    ('nugget', {'w':0.5})                      # elementary contribution
                    ], alpha=-30., beta=-40., gamma=20., name='')

    # Define mean and variance of GRF
    mean = 10.
    # mean = sum(np.meshgrid(np.linspace(2, 10, ny), np.linspace(2, 8, nz), np.repeat(0, nx))) # as this!!!
    var = None
    # var = sum(np.meshgrid(np.linspace(2, 400, ny), np.repeat(0, nz), np.linspace(2, 100, nx))) # as this!!!

    # Define hard data
    x = np.array([[ 10.5,  20.5,  3.5], # 1st point
                  [ 40.5,  10.5, 10.5], # 2nd point
                  [ 30.5,  40.5, 20.5], # 3rd point
                  [ 30.5,  30.5, 30.5]]) # 4th point
    v = [ -3.,  2.,   5.,  -1.] # values
    # x, v = None, None

    # Set number of realizations
    nreal = 500

    # Set seed
    np.random.seed(123)

    # Generate GRF
    t1 = time.time()
    grf3 = grf3D(cov_model3, dimension, spacing, origin=origin,
                nreal=nreal, mean=mean, var=var,
                x=x, v=v,
                method=3, conditioningMethod=2) # grf: (nreal,nz,ny,nx) array
    t2 = time.time()
    nreal_case3D = nreal
    time_case3D = t2-t1
    infogrid_case3D = f'grid: {nx*ny*nz} cells ({nx} x {ny} x {nz})'
    # print(f'Elapsed time: {time_case3D} sec')

    # Fill an image (Img class) (for display, see below)
    im3 = img.Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv=nreal, val=grf3)
    del(grf3)

    # Compute mean and standard deviation over the realizations
    im3_mean = img.imageContStat(im3, op='mean') # pixel-wise mean
    im3_std = img.imageContStat(im3, op='std')   # pixel-wise standard deviation
    # grf3_mean = np.mean(grf.reshape(nreal, -1), axis=0).reshape(nz, ny, nx)
    # grf3_std = np.std(grf.reshape(nreal, -1), axis=0).reshape(nz, ny, nx)

    if x is not None:
        # Kriging
        t1 = time.time()
        krig3, krig3_std = krige3D(x, v, cov_model3, dimension, spacing, origin=origin,
                               mean=mean, var=var,
                               conditioningMethod=2)
        t2 = time.time()
        time_krig_case3D = t2-t1
        # print(f'Elapsed time for kriging: {time_krig_case3D} sec')

        # Fill an image (Img class) (for display, see below)
        im3_krig = img.Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv=2, val=np.array((krig3, krig3_std)))
        del(krig3, krig3_std)

        peak_to_peak_mean3 = np.ptp(im3_mean.val[0] - im3_krig.val[0])
        peak_to_peak_std3  = np.ptp(im3_mean.val[0] - im3_krig.val[1])
        krig3D_done = True
    else:
        krig3D_done = False

    # Display (using geone.imgplot3d)
    # -------
    # === Show first real - volume in 3D
    imgplt3.drawImage3D_volume(im3, iv=0,
        text='GRF3D: real #1',
        cmap='nipy_spectral', scalar_bar_kwargs={'vertical':True, 'title':None})

    # === Show first real - (out) surface in 3D
    imgplt3.drawImage3D_surface(im3, iv=0,
        text='GRF3D: real #1',
        cmap='nipy_spectral', scalar_bar_kwargs={'vertical':True, 'title':None})

    # === Show first real - slices in 3D block
    # ... slices orthogonal to axes and going through the center of image
    cx = im3.ox + 0.5 * im3.nx * im3.sx
    cy = im3.oy + 0.5 * im3.ny * im3.sy
    cz = im3.oz + 0.5 * im3.nz * im3.sz # center of image (cx, cy, cz)
    imgplt3.drawImage3D_slice(im3, iv=0,
        slice_normal_x=cx,
        slice_normal_y=cy,
        slice_normal_z=cz,
        text='GRF3D: real #1',
        cmap='nipy_spectral', scalar_bar_kwargs={'vertical':True, 'title':None})

    # === Show first real - slices in 3D block
    # ... slices orthogonal to axes and going through the first data point
    #     + display the data points
    cmap = plt.get_cmap('nipy_spectral') # color map
    cmin=im3.vmin()[0] # min value for real 0
    cmax=im3.vmax()[0] # max value for real 0
    data_points_col = [cmap((vv-cmin)/(cmax-cmin)) for vv in v] # color for data points according to their value

    pp = pv.Plotter()
    imgplt3.drawImage3D_slice(im3, iv=0, plotter=pp,
        slice_normal_x=x[0,0],
        slice_normal_y=x[0,1],
        slice_normal_z=x[0,2],
        show_bounds=True,
        text='GRF3D: real #1',
        cmap=cmap, cmin=cmin, cmax=cmax, scalar_bar_kwargs={'vertical':True, 'title':None}) # specify color map and cmin, cmax
    data_points = pv.PolyData(x)
    data_points['colors'] = data_points_col
    pp.add_mesh(data_points, cmap=cmap, rgb=True,
        point_size=20., render_points_as_spheres=True)
    pp.show()

    # === Show first real - slices in 3D block
    # ... slices orthogonal to axes supporting the ranges according to rotation
    #     defined in the covariance model and going through the center of image
    mrot = cov_model3.mrot()
    imgplt3.drawImage3D_slice(im3, iv=0,
        slice_normal_custom=[[mrot[:,0], (cx, cy, cz)], [mrot[:,1], (cx, cy, cz)], [mrot[:,2], (cx, cy, cz)]],
        text='GRF3D: real #1',
        cmap='nipy_spectral', scalar_bar_kwargs={'vertical':True, 'title':None})

    # === Show two first reals, mean and st. dev. over real,
    #     and kriging estimates and standard deviation (if conditional)
    #     - volume in 3D
    if x is not None:
        nc = 3
    else:
        nc = 2

    pp = pv.Plotter(shape=(2, nc))
    # 2 first reals
    for i in (0, 1):
        pp.subplot(i, 0)
        imgplt3.drawImage3D_volume(im3, iv=i, plotter=pp,
            text=f'GRF3D: real #{i+1}',
            cmap='nipy_spectral', scalar_bar_kwargs={'vertical':True, 'title':None})

    # mean of all real
    pp.subplot(0, 1)
    imgplt3.drawImage3D_volume(im3_mean, plotter=pp,
        text=f'GRF3D: mean over {nreal} real',
        cmap='nipy_spectral', scalar_bar_kwargs={'vertical':True, 'title':None})

    # standard deviation of all real
    pp.subplot(1, 1)
    imgplt3.drawImage3D_volume(im3_std, plotter=pp,
        text=f'GRF3D: st. dev. over {nreal} real',
        cmap='viridis', scalar_bar_kwargs={'vertical':True, 'title':None})

    if x is not None:
        # kriging estimates
        pp.subplot(0, 2)
        imgplt3.drawImage3D_volume(im3_krig, iv=0, plotter=pp,
            text='GRF3D: kriging estimates',
            cmap='nipy_spectral', scalar_bar_kwargs={'vertical':True, 'title':None})

        # kriging standard deviation
        pp.subplot(1, 2)
        imgplt3.drawImage3D_volume(im3_krig, iv=1, plotter=pp,
            text='GRF3D: kriging st. dev.',
            cmap='viridis', scalar_bar_kwargs={'vertical':True, 'title':None})

    pp.link_views()
    pp.show()

    # === Show two first reals, mean and st. dev. over real,
    #     and kriging estimates and standard deviation (if conditional)
    #     - slices in 3D block
    # ... slices orthogonal to axes and going through the center of image
    if x is not None:
        nc = 3
    else:
        nc = 2

    pp = pv.Plotter(shape=(2, nc))
    # 2 first reals
    for i in (0, 1):
        pp.subplot(i, 0)
        imgplt3.drawImage3D_slice(im3, iv=i, plotter=pp,
            slice_normal_x=cx,
            slice_normal_y=cy,
            slice_normal_z=cz,
            text=f'GRF3D: real #{i+1}',
            cmap='nipy_spectral', scalar_bar_kwargs={'vertical':True, 'title':None})

    # mean of all real
    pp.subplot(0, 1)
    imgplt3.drawImage3D_slice(im3_mean, plotter=pp,
        slice_normal_x=cx,
        slice_normal_y=cy,
        slice_normal_z=cz,
        text=f'GRF3D: mean over {nreal} real',
        cmap='nipy_spectral', scalar_bar_kwargs={'vertical':True, 'title':None})

    # mean of all real
    pp.subplot(1, 1)
    imgplt3.drawImage3D_slice(im3_std, plotter=pp,
        slice_normal_x=cx,
        slice_normal_y=cy,
        slice_normal_z=cz,
        text=f'GRF3D: st. dev. over {nreal} real',
        cmap='viridis', scalar_bar_kwargs={'vertical':True, 'title':None})

    if x is not None:
        # kriging estimates
        pp.subplot(0, 2)
        imgplt3.drawImage3D_slice(im3_krig, iv=0, plotter=pp,
            slice_normal_x=cx,
            slice_normal_y=cy,
            slice_normal_z=cz,
            text='GRF3D: kriging estimates',
            cmap='nipy_spectral', scalar_bar_kwargs={'vertical':True, 'title':None})

        # kriging standard deviation
        pp.subplot(1, 2)
        imgplt3.drawImage3D_slice(im3_krig, iv=1, plotter=pp,
            slice_normal_x=cx,
            slice_normal_y=cy,
            slice_normal_z=cz,
            text='GRF3D: kriging st. dev.',
            cmap='viridis', scalar_bar_kwargs={'vertical':True, 'title':None})

    pp.link_views()
    pp.show()

    if x is not None:
        # === Show comparison of mean and st. dev. of all real, with kriging estimates and st. dev.
        #     - volume in 3D
        pp = pv.Plotter(shape=(1, 2))
        # grf mean - kriging estimates
        im_tmp = img.copyImg(im3_mean)
        im_tmp.val[0] = im_tmp.val[0] - im3_krig.val[0]
        pp.subplot(0, 0)
        imgplt3.drawImage3D_volume(im_tmp, plotter=pp,
            text=f'GRF3D: grf mean - kriging estimates / nreal={nreal}',
            cmap='viridis', scalar_bar_kwargs={'vertical':True, 'title':None})

        # grf st. dev. - kriging st. dev.
        im_tmp = img.copyImg(im3_std)
        im_tmp.val[0] = im_tmp.val[0] - im3_krig.val[1]
        pp.subplot(0, 1)
        imgplt3.drawImage3D_volume(im_tmp, plotter=pp,
            text=f'GRF3D: grf st. dev. - kriging st. dev. / nreal={nreal}',
            cmap='viridis', scalar_bar_kwargs={'vertical':True, 'title':None})

        pp.link_views()
        pp.show()

        del(im3_krig)

    del(im3, im3_mean, im3_std)

    ######### Print info: elapsed time, peak to peak for "mean of real - krig est." and "std. of real - krig std." ##########
    print('Case 1D\n-------')
    print('   Simulation - elapsed time: {:5.2f} sec  ({} real, {})'.format(time_case1D, nreal_case1D, infogrid_case1D))
    print('   Kriging    - elapsed time: {:5.2f} sec'.format(time_krig_case1D))
    if krig1D_done:
        print('   Peak to peak for "grf1_mean - krig1"    : {}'.format(peak_to_peak_mean1))
        print('   Peak to peak for "grf1_std  - krig1_std": {}'.format(peak_to_peak_std1))
    print('\n')
    print('Case 2D\n-------')
    print('   Simulation - elapsed time: {:5.2f} sec  ({} real, {})'.format(time_case2D, nreal_case2D, infogrid_case2D))
    print('   Kriging    - elapsed time: {:5.2f} sec'.format(time_krig_case2D))
    if krig2D_done:
        print('   Peak to peak for "grf2_mean - krig2"    : {}'.format(peak_to_peak_mean2))
        print('   Peak to peak for "grf2_std  - krig2_std": {}'.format(peak_to_peak_std2))
    print('\n')
    print('Case 3D\n-------')
    print('   Simulation - elapsed time: {:5.2f} sec  ({} real, {})'.format(time_case3D, nreal_case3D, infogrid_case3D))
    print('   Kriging    - elapsed time: {:5.2f} sec'.format(time_krig_case3D))
    if krig3D_done:
        print('   Peak to peak for "grf3_mean - krig3"    : {}'.format(peak_to_peak_mean3))
        print('   Peak to peak for "grf3_std  - krig3_std": {}'.format(peak_to_peak_std3))

    ######### END ##########
    a = input("Press enter to continue...")
