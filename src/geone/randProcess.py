#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# Python module:  'randProcess.py'
# author:         Julien Straubhaar
# date:           may-2022
# -------------------------------------------------------------------------

"""
Module for miscellaneous algorithms based on random processes.
"""

import numpy as np
import scipy

# ============================================================================
class RandProcessError(Exception):
    """
    Custom exception related to `randProcess` module.
    """
    pass
# ============================================================================

# ----------------------------------------------------------------------------
def acceptRejectSampler(
        n, xmin, xmax, f,
        c=None, 
        g=None, g_rvs=None,
        return_accept_ratio=False,
        max_trial=None,
        verbose=0, 
        show_progress=None,
        opt_kwargs=None,
        logger=None):
    """
    Generates samples according to a given density function.

    This function generates `n` points (which can be multi-variate) in a
    box-shape domain of lower bound(s) `xmin` and upper bound(s) `xmax`,
    according to a density proportional to the function `f`, based on the
    accept-reject algorithm.

    Let `g_rvs` a function returning random variates sample(s) from an
    instrumental distribution with density proportional to `g`, and `c` a
    constant such that `c*g(x) >= f(x)` for any `x` (in `[xmin, xmax[` (can be
    multi-dimensional), i.e. `x[i]` in `[xmin[i], xmax[i][` for any i). Let `fd`
    (resp. `gd`) the density function proportional to `f` (resp. `g`); the
    alogrithm consists in the following steps to generate samples `x ~ fd`:

    - generate `y ~ gd` (using `g_rvs`)
    - generate `u ~ Unif([0,1])`
    - if `u < f(y)/c*g(y)`, then accept `x` (reject `x` otherwise)

    The default instrumental distribution (if both `g` and `g_rvs` set to `None`)
    is the uniform distribution (`g=1`). If the domain (`[xmin, xmax[`) is
    infinite, the instrumental distribution (`g`, and `g_rvs`) and `c` must be
    specified.

    Parameters
    ----------
    n : int
        number of sample points

    xmin : float (or int), or array-like of shape(m,)
        lower bound of each coordinate (m is the space dimension);
        note: component(s) can be set to `-np.inf`

    xmax : float (or int), or array-like of shape(m,)
        upper bound of each coordinate (m is the space dimension)
        note: component(s) can be set to `np.inf`

    f : function (`callable`)
        function proportional to target density, `f(x)` returns the target
        density (times a constant) at `x`; with `x` array_like, the last
        axis of `x` denotes the components of the points where the function is
        evaluated

    c : float (or int), optional
        constant such that (not checked)) `c*g(x) >= f(x)` for all x in
        [xmin, xmax[, with `g(x)=1` if `g` is not specified (`g=None`);
        by default (`c=None`), the domain (`[xmin, xmax[`) must be finite and
        `c` is automatically computed (using the function
        `scipy.optimize.differential_evolution`)

    g : function (callable), optional
        function proportional to the instrumental density on `[xmin, xmax[`,
        `g(x)` returns the instrumental density (times a constant) at `x`;
        with `x` array_like, the last axis of `x` denotes the components of the
        points where the function is evaluated;
        by default (`g=None`), the domain (`[xmin, xmax[`) must be finite and
        the instrumental distribution considered is uniform (constant
        function `g=1` is considered)

    g_rvs : function (`callable`), optional
        function returning samples from the instrumental distribution with
        density proportional to `g` on `[xmin, xmax[` (restricted on this
        domain if needed); `g_rvs` must have the keyword arguments `size`
        (the number of sample(s) to draw);
        by default (`None`), uniform instrumental distribution is considered
        (see `g`);
        note: both `g` and `g_rvs` must be specified (or both set to `None`)

    return_accept_ratio : bool, default: False
        indicates if the acceptance ratio is returned

    verbose : int, default: 0
        verbose mode, higher implies more printing (info)

    show_progress : bool, optional
        deprecated, use `verbose` instead;

        - if `show_progress=False`, `verbose` is set to 1 (overwritten)
        - if `show_progress=True`, `verbose` is set to 2 (overwritten)
        - if `show_progress=None` (default): not used

    opt_kwargs : dict, optional
        keyword arguments to be passed to `scipy.optimize.differential_evolution`
        (do not set `'bounds'` key, bounds are set according to `xmin`, `xmax`)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    x : 2d-array of shape (n, m), or 1d-array of shape (n,)
        samples according to the target density proportional to `f on the
        domain `[xmin, max[`, `x[i]` is the i-th sample point;
        notes:

        - if dimension m >= 2: `x` is a 2d-array of shape (n, m)
        - if diemnsion is 1: `x` is an array of shape (n,)

    t : float, optional
        acceptance ratio, returned if `return_accept_ratio=True`, i.e.
        `t = n/ntot` where `ntot` is the number of points draws in the
        instrumental distribution
    """
    fname = 'acceptRejectSampler'

    # Set verbose mode according to show_progress (if given)
    if show_progress is not None:
        if show_progress:
            verbose = 2
        else:
            verbose = 1

    xmin = np.atleast_1d(xmin)
    xmax = np.atleast_1d(xmax)

    if xmin.ndim != xmax.ndim or np.any(np.isnan(xmin)) or np.any(np.isnan(xmax)) or np.any(xmin >= xmax):
        err_msg = f'{fname}: `xmin`, `xmax` invalid'
        if logger: logger.error(err_msg)
        raise RandProcessError(err_msg)

    lx = xmax - xmin
    dim = len(xmin)

    if np.any(np.isinf(lx)):
        dom_finite = False
    else:
        dom_finite = True

    if n <= 0:
        x = np.zeros((n, dim))
        if dim == 1:
            x = x.reshape(-1)
        if return_accept_ratio:
            return x, 1.0
        else:
            return x

    # Set g, g_rvs
    if (g is None and g_rvs is not None) or (g is not None and g_rvs is None):
        err_msg = f'{fname}: `g` and `g_rvs` should both be specified'
        if logger: logger.error(err_msg)
        raise RandProcessError(err_msg)

    if g is None:
        if not dom_finite:
            err_msg = f'{fname}: `g` and `g_rvs` must be specified when infinite domain is considered'
            if logger: logger.error(err_msg)
            raise RandProcessError(err_msg)

        # g
        g = lambda x: 1.0
        # g_rvs
        if dim == 1:
            def g_rvs(size=1):
                return xmin[0] + scipy.stats.uniform.rvs(size=size) * lx[0]
        else:
            def g_rvs(size=1):
                return xmin + scipy.stats.uniform.rvs(size=(size,dim)) * lx

    if c is None:
        if not dom_finite:
            err_msg = f'{fname}: `c` must be specified when infinite domain is considered'
            if logger: logger.error(err_msg)
            raise RandProcessError(err_msg)

        h = lambda x: -f(x)/g(x)
        # Compute the min of h(x) with the function scipy.optimize.differential_evolution
        if opt_kwargs is None:
            opt_kwargs = {}
        res = scipy.optimize.differential_evolution(h, bounds=list(zip(xmin, xmax)), **opt_kwargs)
        if not res.success:
            err_msg = f'{fname}: `scipy.optimize.differential_evolution` failed {res.message})'
            if logger: logger.error(err_msg)
            raise RandProcessError(err_msg)

        # -> res.x realizes the minimum of h(x)
        # -> res.fun is the minimum of h(x)
        # Set c such that c > f(x)/g(x) for all x in the domain
        c = -res.fun + 1.e-3 # add small number to ensure the inequality

    # Apply accept-reject algo
    naccept = 0
    ntot = 0
    x = []
    if max_trial is None:
        max_trial = np.inf
    if verbose > 1:
        progress = 0
        progressOld = -1
    while naccept < n:
        nn = n - naccept
        ntot = ntot+nn
        xnew = g_rvs(size=nn)
        ind = np.all((xnew >= xmin, xnew < xmax), axis=0)
        if dim > 1:
            ind = np.all(ind, axis=-1)
        xnew = xnew[ind]
        nn = len(xnew)
        if nn == 0:
            continue
        u = np.random.random(size=nn)
        xnew = xnew[u < (f(xnew)/(c*g(xnew))).reshape(nn)]
        nn = len(xnew)
        if nn == 0:
            continue
        x.extend(xnew)
        naccept = naccept+nn
        if verbose > 1:
            progress = int(100*naccept/n)
            if progress > progressOld:
                if logger:
                    logger.info(f'{fname}: A-R algo, progress: {progress:3d} %')
                else:
                    print(f'{fname}: A-R algo, progress: {progress:3d} %')
                progressOld = progress
        if ntot >= max_trial:
            break

    x = np.asarray(x)

    if naccept < n and verbose > 0:
        if logger:
            logger.warning(f'{fname}: sample size is only {naccept}! (increase `max_trial`)')
        else:
            print(f'{fname}: WARNING: sample size is only {naccept}! (increase `max_trial`)')

    if return_accept_ratio:
        accept_ratio = naccept/ntot
        return x, accept_ratio
    else:
        return x
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def poissonPointProcess(mu, xmin=0.0, xmax=1.0, ninterval=None, logger=None):
    """
    Generates random points following a Poisson point process.

    Random points are in `[xmin, xmax[` (can be multi-dimensional).

    Parameters
    ----------
    mu : function (`callable`), or ndarray of floats, or float
        intensity of the Poisson process, i.e. the mean number of points per
        unitary volume:

        - if a function: (non-homogeneous Poisson point process) \
        `mu(x)` returns the intensity at `x`; with `x` array_like, the last \
        axis of `x` denotes the components of the points where the function is \
        evaluated
        - if a ndarray: (non-homogeneous Poisson point process) \
        `mu[i_n, ..., i_0]` is the intensity on the box \
        `[xmin[j]+i_j*(xmax[j]-xmin[j])/mu.shape[n-j]]`, j = 0,..., n
        - if a float: homogeneous Poisson point process

    xmin : float (or int), or array-like of shape(m,)
        lower bound of each coordinate

    xmax : float (or int), or array-like of shape(m,)
        upper bound of each coordinate

    ninterval : int, or array-like of ints of shape (m,), optional
        used only if `mu` is a function (callable);
        `ninterval` contains the number of interval(s) in which the domain
        `[xmin, xmax[` is subdivided along each axis

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    pts : 2D array of shape (npts, m)
        each row is a random point in the domain `[xmin, xmax[`, the number of
        points (`npts`) follows a Poisson law of the given intensity (`mu`) and
        m is the dimension of the domain
    """
    fname = 'poissonPointProcess'

    xmin = np.atleast_1d(xmin)
    xmax = np.atleast_1d(xmax)

    if xmin.ndim != xmax.ndim or xmin.ndim != 1:
        err_msg = f'{fname}: `xmin`, `xmax` not valid (dimension or shape)'
        if logger: logger.error(err_msg)
        raise RandProcessError(err_msg)

    if np.any(xmin >= xmax):
        err_msg = f'{fname}: `xmin`, `xmax` not valid ((component of) xmin less than or equal to xmax)'
        if logger: logger.error(err_msg)
        raise RandProcessError(err_msg)

    # dimension
    dim = len(xmin)

    if callable(mu):
        if ninterval is None:
            err_msg = f'{fname}: `ninterval` must be specified when a function is passed for the intensity (`mu`)'
            if logger: logger.error(err_msg)
            raise RandProcessError(err_msg)

        ninterval = np.asarray(ninterval, dtype=int)  # possibly 0-dimensional
        if ninterval.size == 1:
            ninterval = ninterval.flat[0] * np.ones(dim)
        elif ninterval.size != dim:
            err_msg = f'{fname}: `ninterval` does not have an acceptable size'
            if logger: logger.error(err_msg)
            raise RandProcessError(err_msg)

        if np.any(ninterval < 1):
            err_msg = f'{fname}: `ninterval` has negative or zero value'
            if logger: logger.error(err_msg)
            raise RandProcessError(err_msg)

    elif isinstance(mu, np.ndarray):
        if mu.ndim != dim:
            err_msg = f'{fname}: inconsistent number of dimension for the ndarray `mu`'
            if logger: logger.error(err_msg)
            raise RandProcessError(err_msg)

        ninterval = mu.shape[::-1]

    else: # mu is a float
        mu = np.atleast_1d(mu)
        for i in range(dim-1):
            mu = mu[np.newaxis,...]
        # mu is a ndarray with dim dimension of shape (1,...,1) --> grid with one cell

        ninterval = mu.shape

    # spacing of the grid cell along each axis
    spa = [(b-a)/n for a, b, n in zip(xmin, xmax, ninterval)]
    # cell volume
    vol_cell = np.prod(spa)
    # cell center along each axis
    x_cell_center = [a + (0.5 + np.arange(n)) * s for a, n, s in zip(xmin, ninterval, spa)]
    # center of each grid cell
    xx_cell_center = np.meshgrid(*x_cell_center[::-1], indexing='ij')[::-1]
    xx_cell_center = np.array([xx.reshape(-1) for xx in xx_cell_center]).T # shape: ncell x dim

    # Poisson parameter (intensity) for each grid cell
    if callable(mu):
        mu_cell = mu(xx_cell_center)*vol_cell
    else:
        mu_cell = mu.reshape(-1) * vol_cell

    # Generate number of points in each grid cell (Poisson)
    npts_cell = np.array([scipy.stats.poisson.rvs(m) for m in mu_cell])

    # Generate random points (uniformly) in each cell
    pts = np.array([np.hstack(
            [a + spa[i] * (np.random.random(size=npts) - 0.5) for a, npts in zip(xx_cell_center[:,i], npts_cell)]
        ) for i in range(dim)]).T

    return pts
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def chentsov1D(
        n_mean,
        dimension, 
        spacing=1.0, 
        origin=0.0,
        direction_origin=None,
        p_min=None, 
        p_max=None,
        nreal=1,
        verbose=0,
        logger=None):
    """
    Generates a Chentsov's simulation in 1D.

    The domain of simulation is `[xmin, xmax]`, with `nx` cells along x axis,
    each cell having a length of `dx`, the left side is the origin:

    - along x axis:
        - `nx = dimension`
        - `dx = spacing`
        - `xmin = origin`
        - `xmax = origin + nx*dx`

    The simulation consists in:

    1. Drawing random hyper-plane (i.e. points in 1D) in the space
    [`p_min`, `p_max`] following a Poisson point process with intensity:

    * mu = `n_mean` / vol([`p_min`, `p_max`]);

    the points are given in the parametrized form: p;
    then, for each point p, and with direction_origin = x0
    (the center of the simulation domain by default), the hyper-plane
    (point)

    * {x : x-x0 = p} (i.e. the point x0 + p)

    is considered

    2. Each hyper-plane (point x0+p) splits the space (R) in two parts
    (two half lines); the value = +1 is set to one part (chosen
    randomly) and the value -1 is set to the other part. Denoting V_i
    the value over the space (R) associated to the i-th hyper-plane
    (point), the value assigned to a grid cell of center x is set to

    * Z(x) = 0.5 * sum_{i} (V_i(x) - V_i(x0))

    It corresponds to the number of hyper-planes (points) cut by the
    segment [x0, x].

    Parameters
    ----------
    n_mean : float
        mean number of hyper-plane drawn (via Poisson process)

    dimension : int
        `dimension=nx`, number of cells in the 1D simulation grid

    spacing : float, default: 1.0
        `spacing=dx`, cell size

    origin : float, default: 0.0
        `origin=ox`, origin of the 1D simulation grid (left border)

    direction_origin : float, optional
        origin from which the "points" are drawn in the Poisson process
        (see above);
        by default (`None`): the center of the 1D simulation domain is used

    p_min : float, optional
        minimal value for p (see above);
        by default (`None`): `p_min` is set automatically to "minus half of the
        length of the 1D simulation domain"

    p_max : float, optional
        maximal value for p (see above);
        by default (`None`): `p_max` is set automatically to "plus half of the
        length of the 1D simulation domain

    nreal : int, default: 1
        number of realization(s)

    verbose : int, default: 0
        verbose mode, higher implies more printing (info)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    sim : 2D array of floats of shape (nreal, nx)
        simulations of Z (see above);
        `sim[i, j]`: value of the i-th realisation at grid cell of index j

    n : 1D array of shape (nreal,)
        numbers of hyper-planes (points) drawn, `n[i]` is the number of
        hyper-planes for the i-th realization
    """
    fname = 'chentsov1D'

    # Number of realization(s)
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose > 0:
            if logger:
                logger.warning(f'{fname}: `nreal` <= 0: `None`, `None` is returned')
            else:
                print(f'{fname}: WARNING: `nreal` <= 0: `None`, `None` is returned')
        return None, None

    nx = dimension
    dx = spacing
    ox = origin

    if direction_origin is None:
        direction_origin = ox+0.5*nx*dx

    if p_min is None or p_max is None:
        d = 0.5*nx*dx
        if p_min is None:
            p_min = -d
        if p_max is None:
            p_max = d

    if p_min >= p_max:
        err_msg = f'{fname}: `p_min` is greater than or equal to `p_max`'
        if logger: logger.error(err_msg)
        raise RandProcessError(err_msg)

    # center of each grid cell of the simulation domain
    xc = ox + (0.5 + np.arange(nx)) * dx

    # Volume of [p_min, p_max]
    vol_poisson_domain = (p_max - p_min)

    # Set intensity of Poisson process
    mu = n_mean / vol_poisson_domain

    # Initialization
    z = np.zeros((nreal, nx))
    n = np.zeros(nreal, dtype='int')

    for k in range(nreal):
        # Draw points via Poisson process
        try:
            pts = poissonPointProcess(mu, p_min, p_max, logger=logger)
        except Exception as exc:
            err_msg = f'{fname}: Poisson point process failed'
            if logger: logger.error(err_msg)
            raise RandProcessError(err_msg) from exc

        n[k] = pts.shape[0]

        # Defines values of Z in each grid cell
        random_sign = (-1)**np.random.randint(2, size=n[k])
        for i in range(n[k]):
            z[k] = z[k] + (np.sign((xc-direction_origin)-pts[i])+np.sign(pts[i]))*random_sign[i]

    z = 0.5*z

    return z, n
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def chentsov2D(
        n_mean,
        dimension, 
        spacing=(1.0, 1.0), 
        origin=(0.0, 0.0),
        direction_origin=None,
        phi_min=0.0, 
        phi_max=np.pi,
        p_min=None, 
        p_max=None,
        nreal=1,
        verbose=0,
        logger=None):
    """
    Generates a Chentsov's simulation in 2D.

    The domain of simulation is `[xmin, xmax]` x `[ymin x ymax]`,
    with `nx` and `ny` cells along x axis and y axis respectively, each cell
    being a box of size `dx` x `dy`, the lower-left corner is the origin:

    - along x axis:
        - `nx = dimension[0]`
        - `dx = spacing[0]`
        - `xmin = origin[0]`
        - `xmax = origin[0] + nx*dx`

    - along y axis:
        - `ny = dimension[1]`
        - `dy = spacing[1]`
        - `ymin = origin[1]`
        - `ymax = origin[1] + ny*dy`

    The simulation consists in:

    1. Drawing random hyper-plane (i.e. lines in 2D):
    considering the space S x [`p_min`, `p_max`], where S is a part of
    the circle of radius 1 in the plane (by default: half circle),
    parametrized via

    * phi -> (cos(phi), sin(phi)), with phi in [`phi_min`, `phi_max`],

    some points are drawn randomly in S x [`p_min`, `p_max`] following a
    Poisson point process with intensity

    * mu = `n_mean` / vol(S x [`p_min`, `p_max`])

    the points are given in the parametrized form: (phi, p);
    then, for each point (phi, p), and with direction_origin = (x0, y0)
    (the center of the simulation domain by default), the hyper-plane
    (line)

    * {(x, y) : dot([x-x0, y-y0], [cos(phi), sin(phi)]) = p}

    (i.e. point (x, y) s.t. the orthogonal projection of (x-x0, y-y0)
    onto the direction (cos(phi), sin(phi)) is equal to p) is considered

    2. Each hyper-plane (line) splits the space (R^2) in two parts (two half
    planes); the value = +1 is set to one part (chosen randomly) and the
    value -1 is set to the other part. Denoting V_i the value over the
    space (R^2) associated to the i-th hyper-plane (line), the value
    assigned to a grid cell of center (x, y) is set to

    * Z(x, y) = 0.5 * sum_{i} (V_i(x, y) - V_i(x0, y0))

    It corresponds to the number of hyper-planes cut by the segment
    [(x0, y0), (x, y)].

    Parameters
    ----------
    n_mean : float
        mean number of hyper-plane drawn (via Poisson process)

    dimension : 2-tuple of ints
        `dimension=(nx, ny)`, number of cells in the 2D simulation grid along
        each axis

    spacing : 2-tuple of floats, default: (1.0, 1.0)
        `spacing=(dx, dy)`, cell size along each axis

    origin : 2-tuple of floats, default: (0.0, 0.0)
        `origin=(ox, oy)`, origin of the 2D simulation grid (lower-left corner)

    direction_origin : sequence of 2 floats, optional
        origin from which the directions are drawn in the Poisson process
        (see above);
        by default (`None`): the center of the 2D simulation domain is used

    phi_min : float, default: 0.0
        minimal angle for the parametrization of S (part of circle) defining
        the direction (see above)

    phi_max : float, default: `numpy.pi`
        maximal angle for the parametrization of S (part of circle) defining
        the direction (see above)

    p_min : float, optional
        minimal value for orthogonal projection (see above);
        by default (`None`): `p_min` is set automatically to "minus half of the
        diagonal of the 2D simulation domain"

    p_max : float, optional
        maximal value for orthogonal projection (see above);
        by default (`None`): `p_min` is set automatically to "plus half of the
        diagonal of the 2D simulation domain"

    nreal : int, default: 1
        number of realization(s)

    verbose : int, default: 0
        verbose mode, higher implies more printing (info)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    sim : 3D array of floats of shape (nreal, ny, nx)
        simulations of Z (see above);
        `sim[i, iy, ix]`: value of the i-th realisation at grid cell of index ix
        (resp. iy) along x (resp. y) axis

    n : 1D array of shape (nreal,)
        numbers of hyper-planes (lines) drawn, `n[i]` is the number of
        hyper-planes for the i-th realization
    """
    fname = 'chentsov2D'

    # Number of realization(s)
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose > 0:
            if logger:
                logger.warning(f'{fname}: `nreal` <= 0: `None`, `None` is returned')
            else:
                print(f'{fname}: WARNING: `nreal` <= 0: `None`, `None` is returned')
        return None, None

    nx, ny = dimension
    dx, dy = spacing
    ox, oy = origin

    if direction_origin is None:
        direction_origin = [ox+0.5*nx*dx, oy+0.5*ny*dy]

    if p_min is None or p_max is None:
        d = 0.5*np.sqrt((nx*dx)**2+(ny*dy)**2)
        if p_min is None:
            p_min = -d
        if p_max is None:
            p_max = d

    if p_min >= p_max:
        err_msg = f'{fname}: `p_min` is greater than or equal to `p_max`'
        if logger: logger.error(err_msg)
        raise RandProcessError(err_msg)

    if phi_min >= phi_max:
        err_msg = f'{fname}: `phi_min` is greater than or equal to `phi_max`'
        if logger: logger.error(err_msg)
        raise RandProcessError(err_msg)

    # center of each grid cell of the simulation domain
    yc, xc = np.meshgrid(oy + (0.5 + np.arange(ny)) * dy, ox + (0.5 + np.arange(nx)) * dx, indexing='ij')
    xyc = np.array([xc.reshape(-1), yc.reshape(-1)]).T # shape: ncell x 2

    # Volume of S x [p_min, p_max], (S being parametrized by phi in [phi_min, phi_max])
    vol_poisson_domain = (phi_max - phi_min) * (p_max - p_min)

    # Defines lines by random points in [phi_min, phi_max] x [p_min, p_max]
    # if callable(mu):
    #     def mu_intensity(x):
    #         return mu(x) / vol_poisson_domain
    # else:
    #     mu_intensity = mu / vol_poisson_domain

    # Set intensity of Poisson process
    mu = n_mean / vol_poisson_domain

    # Initialization
    z = np.zeros((nreal, nx*ny))
    n = np.zeros(nreal, dtype='int')

    for k in range(nreal):
        # Draw points via Poisson process
        try:
            pts = poissonPointProcess(mu, [phi_min, p_min], [phi_max, p_max], logger=logger)
        except Exception as exc:
            err_msg = f'{fname}: Poisson point process failed'
            if logger: logger.error(err_msg)
            raise RandProcessError(err_msg) from exc

        n[k] = pts.shape[0]

        # Defines values of Z in each grid cell
        random_sign = (-1)**np.random.randint(2, size=n[k])
        # Equivalent method below (4/ is better!)
        # 1/
        # vp = np.sum([np.sign((xyc-direction_origin).dot(np.array([np.cos(a), np.sin(a)]))-p)*rs for a, p, rs in zip(pts[:,0], pts[:,1], random_sign)], axis=0)
        # v0 = np.sum([np.sign(-p)*rs for p, rs in zip(pts[:,1], random_sign)])
        # z = 0.5 *(vp - v0)
        # 2/
        # z = 0.5*np.sum([(np.sign((xyc-direction_origin).dot(np.array([np.cos(a), np.sin(a)]))-p)+np.sign(p))*rs for a, p, rs in zip(pts[:,0], pts[:,1], random_sign)], axis=0)
        # 3/
        # z = 0.5*np.sum([(np.sign((xyc-direction_origin).dot(np.array([np.cos(pts[i,0]), np.sin(pts[i,0])]))-pts[i,1])+np.sign(pts[i,1]))*random_sign[i] for i in range(n[k])], axis=0)
        # 4/
        for i in range(n[k]):
            z[k] = z[k] + (np.sign((xyc-direction_origin).dot(np.array([np.cos(pts[i,0]), np.sin(pts[i,0])]))-pts[i,1])+np.sign(pts[i,1]))*random_sign[i]

    z = 0.5*z

    return z.reshape(nreal, ny, nx), n
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def chentsov3D(
        n_mean,
        dimension, 
        spacing=(1.0, 1.0, 1.0), 
        origin=(0.0, 0.0, 0.0),
        direction_origin=None,
        phi_min=0.0, 
        phi_max=2.0*np.pi,
        theta_min=0.0, 
        theta_max=0.5*np.pi,
        p_min=None, 
        p_max=None,
        ninterval_theta=100,
        nreal=1,
        verbose=0,
        logger=None):
    """
    Generates a Chentsov's simulation in 3D.

    The domain of simulation is
    `[xmin, xmax]` x `[ymin x ymax]` x `[zmin x zmax]`,
    with `nx`, `ny`, `nz` cells along x axis, y axis, z axis respectively, each
    cell being a box of size `dx` x `dy` x `dy`, the bottom-lower-left corner is
    the origin:

    - along x axis:
        - `nx = dimension[0]`
        - `dx = spacing[0]`
        - `xmin = origin[0]`
        - `xmax = origin[0] + nx*dx`

    - along y axis:
        - `ny = dimension[1]`
        - `dy = spacing[1]`
        - `ymin = origin[1]`
        - `ymax = origin[1] + ny*dy`

    - along z axis:
        - `nz = dimension[0]`
        - `dz = spacing[0]`
        - `zmin = origin[0]`
        - `zmax = origin[0] + nz*dz`.

    The simulation consists in:

    1. Drawing random hyper-plane (i.e. planes in 3D):
    considering the space S x [`p_min`, `p_max`], where S is a part of
    the sphere of radius 1 in the 3D space (by default: half sphere),
    parametrized via

    * (phi, theta) -> (cos(phi)cos(theta), sin(phi)cos(theta), sin(theta)), \
    with phi in [`phi_min`, `phi_max`], theta in [`theta_min`, `theta_max`]

    some points are drawn randomly in S x [`p_min`, `p_max`] following a
    Poisson point process with intensity

    * mu = `n_mean` / vol(S x [`p_min`, `p_max`]);

    the points are given in the parametrized form: (phi, theta, p);
    then, for each point (phi, theta, p), and with
    direction_origin = (x0, y0, z0) (the center of the simulation domain
    by default), the hyper-plane (plane)

    * {(x, y, z) : dot([x-x0, y-y0, z-z0], [cos(phi)cos(theta), sin(phi)cos(theta), sin(theta)]) = p}

    (i.e. point (x, y, z) s.t. the orthogonal projection of
    (x-x0, y-y0, z-z0) onto the direction
    (cos(phi)cos(theta), sin(phi)cos(theta), sin(theta)) is equal to p)
    is considered;

    2. Each hyper-plane (plane) splits the space (R^3) in two parts;
    the value = +1 is set to one part (chosen randomly) and the value -1
    is set to the other part. Denoting V_i the value over the space (R^3)
    associated to the i-th hyper-plane (plane), the value assigned to a
    grid cell of center (x, y) is set to

    * Z(x, y) = 0.5 * sum_{i} (V_i(x, y) - V_i(x0, y0))

    It corresponds to the number of hyper-planes (planes) cut by the
    segment [(x0, y0, z0), (x, y, z)].

    Parameters
    ----------
    n_mean : float
        mean number of hyper-plane drawn (via Poisson process)

    dimension : 3-tuple of ints
        `dimension=(nx, ny, nz)`, number of cells in the 3D simulation grid along
        each axis

    spacing : 3-tuple of floats, default: (1.0,1.0, 1.0)
        `spacing=(dx, dy, dz)`, cell size along each axis

    origin : 3-tuple of floats, default: (0.0, 0.0, 0.0)
        `origin=(ox, oy, oz)`, origin of the 3D simulation grid (bottom-lower-left
        corner)

    direction_origin : sequence of 3 floats, optional
        origin from which the directions are drawn in the Poisson process
        (see above);
        by default (`None`): the center of the 3D simulation domain is used

    phi_min : float, default: 0.0
        minimal angle for the parametrization of S (part of circle) defining
        the direction (see above)

    phi_max : float, default: `numpy.pi`
        maximal angle for the parametrization of S (part of circle) defining
        the direction (see above)

    theta_min : float, default: 0.0
        minimal angle for the parametrization of S (part of circle) defining
        the direction (see above)

    theta_max : float, default: `0.5*numpy.pi`
        maximal angle for the parametrization of S (part of circle) defining
        the direction (see above)

    p_min : float, optional
        minimal value for orthogonal projection (see above);
        by default (`None`): `p_min` is set automatically to "minus half of the
        diagonal of the 3D simulation domain"

    p_max : float, optional
        maximal value for orthogonal projection (see above);
        by default (`None`): `p_min` is set automatically to "plus half of the
        diagonal of the 3D simulation domain"

    ninterval_theta : int, default: 100
        number of sub-intervals in which the interval `[theta_min, theta_max]`
        is subdivided for applying the Poisson process

    nreal : int, default: 1
        number of realization(s)

    verbose : int, default: 0
        verbose mode, higher implies more printing (info)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    sim : 4D array of floats of shape (nreal, nz, ny, nx)
        simulations of Z (see above);
        `sim[i, iz, iy, ix]`: value of the i-th realisation at grid cell of
        index ix (resp. iy, iz) along x (resp. y, z) axis

    n : 1D array of shape (nreal,)
        numbers of hyper-planes (planes) drawn, `n[i]` is the number of
        hyper-planes for the i-th realization
    """
    fname = 'chentsov3D'

    # Number of realization(s)
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose > 0:
            if logger:
                logger.warning(f'{fname}: `nreal` <= 0: `None`, `None` is returned')
            else:
                print(f'{fname}: WARNING: `nreal` <= 0: `None`, `None` is returned')
        return None, None

    nx, ny, nz = dimension
    dx, dy, dz = spacing
    ox, oy, oz = origin

    if direction_origin is None:
        direction_origin = [ox+0.5*nx*dx, oy+0.5*ny*dy, oz+0.5*nz*dz]

    if p_min is None or p_max is None:
        d = 0.5*np.sqrt((nx*dx)**2+(ny*dy)**2+(nz*dz)**2)
        if p_min is None:
            p_min = -d
        if p_max is None:
            p_max = d

    if p_min >= p_max:
        err_msg = f'{fname}: `p_min` is greater than or equal to `p_max`'
        if logger: logger.error(err_msg)
        raise RandProcessError(err_msg)

    if phi_min >= phi_max:
        err_msg = f'{fname}: `phi_min` is greater than or equal to `phi_max`'
        if logger: logger.error(err_msg)
        raise RandProcessError(err_msg)

    if theta_min >= theta_max:
        err_msg = f'{fname}: `theta_min` is greater than or equal to `theta_max`'
        if logger: logger.error(err_msg)
        raise RandProcessError(err_msg)

    # center of each grid cell of the simulation domain
    zc, yc, xc = np.meshgrid(oz + (0.5 + np.arange(nz)) * dz, oy + (0.5 + np.arange(ny)) * dy, ox + (0.5 + np.arange(nx)) * dx, indexing='ij')
    xyzc = np.array([xc.reshape(-1), yc.reshape(-1), zc.reshape(-1)]).T # shape: ncell x 3

    # Volume of S x [p_min, p_max], (S being parametrized by phi in [phi_min, phi_max], and theta in [theta_min, theta_max])
    vol_poisson_domain = (phi_max - phi_min) * (np.sin(theta_max) - np.sin(theta_min)) * (p_max - p_min)

    # Set intensity of Poisson process as a function accounting for jacobian of the parametrization of S
    def mu(x):
        return n_mean * np.cos(x[:, 1])/ vol_poisson_domain # x = (phi, theta), cos(x[:, 1] = cos(theta)

    # Initialization
    z = np.zeros((nreal, nx*ny*nz))
    n = np.zeros(nreal, dtype='int')

    for k in range(nreal):
        # Draw points via Poisson process
        try:
            pts = poissonPointProcess(mu, [phi_min, theta_min, p_min], [phi_max, theta_max, p_max], ninterval=[1, ninterval_theta, 1], logger=logger)
        except Exception as exc:
            err_msg = f'{fname}: Poisson point process failed'
            if logger: logger.error(err_msg)
            raise RandProcessError(err_msg) from exc

        n[k] = pts.shape[0]

        # Defines values of Z in each grid cell
        random_sign = (-1)**np.random.randint(2, size=n[k])
        # 4/
        for i in range(n[k]):
            z[k] = z[k] + (np.sign((xyzc-direction_origin).dot(np.array([np.cos(pts[i,0])*np.cos(pts[i,1]), np.sin(pts[i,0])*np.cos(pts[i,1]), np.sin(pts[i,1])]))-pts[i,2])+np.sign(pts[i,2]))*random_sign[i]
    z = 0.5*z

    return z.reshape(nreal, nz, ny, nx), n
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Module 'geone.randProcess'.")

#####  OLD BELOW #####
# # ----------------------------------------------------------------------------
# def acceptRejectSampler(n, xmin, xmax, f, c=None, g=None, g_rvs=None,
#                         return_accept_ratio=False,
#                         max_trial=None, show_progress=False):
#     """
#     Generates samples according to a given density function.
#
#     This function generates `n` points in a box-shape domain of lower bound(s)
#     `xmin` and upper bound(s) `xmax`, according to a density proportional to the
#     function `f` are generated, based on the accept-reject algorithm.
#
#     Let `g_rvs` a function returning random variates sample(s) from an
#     instrumental distribution with density proportional to `g`, and `c` a
#     constant such that `c*g(x) >= f(x)` for any `x` (in `[xmin, xmax[` (can be
#     multi-dimensional), i.e. `x[i]` in `[xmin[i], xmax[i][` for any i). Let `fd`
#     (resp. `gd`) the density function proportional to `f` (resp. `g`); the
#     alogrithm consists in the following steps to generate samples `x ~ fd`:
#     - generate `y ~ gd` (using `g_rvs`)
#     - generate `u ~ Unif([0,1])`
#     - if `u < f(y)/c*g(y)`, then accept `x` (reject `x` otherwise)
#
#     If the instrumental distribution is not specified (both `g` and `g_rvs` set
#     to `None`), then:
#     - the uniform distribution if the domain `[xmin, xmax[` is finite
#     - the multi-normal distribution, centered at a point maximizing `f`, with
#     a variance 1 (covariance matrix I), if the domain `[xmin, xmax[` is infinite
#
#     Parameters
#     ----------
#     n : int
#         number of sample points
#     xmin : float (or int), or array-like of shape(m,)
#         lower bound of each coordinate (m is the space dimension);
#         note: component(s) can be set to `-np.inf`
#     xmax : float (or int), or array-like of shape(m,)
#         upper bound of each coordinate (m is the space dimension)
#         note: component(s) can be set to `np.inf`
#     f : function (callable)
#         function proportional to target density, `f(x)` returns the target
#         density (times a constant) at `x`; with `x` array_like, the last
#         axis of `x` denotes the components of the points where the function is
#         evaluated
#     c : float (or int), optional
#         constant such that (not checked)) `c*g(x) >= f(x)` for all x in
#         [xmin, xmax[, with `g(x)=1` if `g` is not specified (`g=None`);
#         by default (`c=None`), `c` is automatically computed (using the function
#         `scipy.optimize.minimize`)
#     g : function (callable), optional
#         function proportional to the instrumental density on `[xmin, xmax[`,
#         `g(x)` returns the instrumental density (times a constant) at `x`;
#         with `x` array_like, the last axis of `x` denotes the components of the
#         points where the function is evaluated;
#         by default (`g=None`): the instrumental distribution considered is
#         - uniform (constant function `g=1` is considered), if the domain
#         `[xmin, xmax[` is finite
#         - (multi-)normal density of variance 1 (covariance matrix I), centered
#         at a point maximizing `f`, otherwise;
#     g_rvs : function (callable), optional
#         function returning samples from the instrumental distribution with
#         density proportional to `g` on `[xmin, xmax[` (restricted on this
#         domain if needed); `g_rvs` must have the keyword arguments `size`
#         (the number of sample(s) to draw)
#         by default: uniform or non-correlated multi-normal instrumental
#         distribution is considered (see `g`);
#         note: both `g` and `g_rvs` must be specified (or both set to `None`)
#     return_accept_ratio : bool, default: False
#         indicates if the acceptance ratio is returned
#     show_progress : bool, default: False
#         indicates if progress is displayed (True) or not (False)
#
#     Returns
#     -------
#     x : 2d-array of shape (n, m), or 1d-array of shape (n,)
#         samples according to the target density proportional to `f on the
#         domain `[xmin, max[`, `x[i]` is the i-th sample point;
#         notes:
#         - if dimension m >= 2: `x` is a 2d-array of shape (n, m)
#         - if diemnsion is 1: `x` is an array of shape (n,)
#     t : float, optional
#         acceptance ratio, returned if `return_accept_ratio=True`, i.e.
#         `t = n/ntot` where `ntot` is the number of points draws in the
#         instrumental distribution
#     """
#     fname = 'acceptRejectSampler'
#
#     xmin = np.atleast_1d(xmin)
#     xmax = np.atleast_1d(xmax)
#
#     if xmin.ndim != xmax.ndim or np.any(np.isnan(xmin)) or np.any(np.isnan(xmax)) or np.any(xmin >= xmax):
#         print(f'ERROR ({fname}): `xmin`, `xmax` not valid')
#         return None
#
#     lx = xmax - xmin
#     dim = len(xmin)
#
#     x = np.zeros((n, dim)) # initialize random samples (one sample by row)
#     if n <= 0:
#         if return_accept_ratio:
#             return x, 1.0
#         else:
#             return x
#
#     # Set g, g_rvs, and c
#     if (g is None and g_rvs is not None) or (g is not None and g_rvs is None):
#         print(f'ERROR ({fname}): `g` and `g_rvs` should be both specified')
#         return None
#
#     mu = None # not necessarily used
#     if c is None:
#         if g is None:
#             h = lambda x: -f(x)
#         else:
#             h = lambda x: -f(x)/g(x)
#         # Compute the min of h(x) with the function scipy.optimize.minimize
#         # x0: initial guess (random)
#         x0 = xmin + np.random.random(size=dim)*lx
#         for i, binf in enumerate(np.isinf(x0)):
#             if binf:
#                 x0[i] = min(xmax[i], max(xmin[i], 0.0))
#         res = scipy.optimize.minimize(h, x0, bounds=list(zip(xmin, xmax)))
#         if not res.success:
#             print(f'ERROR ({fname}): `scipy.optimize.minimize` failed {res.message})')
#             return None
#         # -> res.x realizes the minimum of h(x)
#         # -> res.fun is the minimum of h(x)
#         mu = res.x
#         # Set c such that c > f(x)/g(x) for all x in the domain
#         c = -res.fun + 1.e-3 # add small number to ensure the inequality
#
#     if g is None:
#         if np.any((np.isinf(xmin), np.isinf(xmax))):
#             if mu is None:
#                 # Compute the min of h(x) = (f(x)-c)**2 with the function scipy.optimize.minimize
#                 h = lambda x: (f(x)-c)**2
#                 # x0: initial guess (random)
#                 x0 = xmin + np.random.random(size=dim)*lx
#                 for i, binf in enumerate(np.isinf(x0)):
#                     if binf:
#                         x0[i] = min(xmax[i], max(xmin[i], 0.0))
#                 res = scipy.optimize.minimize(h, x0, bounds=list(zip(xmin, xmax)))
#                 if not res.success:
#                     print(f'ERROR ({fname}): `scipy.optimize.minimize` failed {res.message})')
#                     return None
#                 # -> res.x is the minimum of h(x)
#             mu = res.x
#             # Set instrumental pdf proportional to: g = exp(-1/2 sum_i((x[i]-mu[i])**2))
#             # Set g, g_rvs
#             g = lambda x: np.exp(-0.5*np.sum((np.atleast_2d(x)-mu)**2), axis=1)
#             g_rvs = scipy.stats.multivariate_normal(mean=mu).rvs
#
#             # Update c
#             # Compute the min of h(x) = -f(x)/g(x) with the function scipy.optimize.minimize
#             h = lambda x: -f(x)/g(x)
#             # x0: initial guess
#             x0 = mu
#             res = scipy.optimize.minimize(h, x0, bounds=list(zip(xmin, xmax)))
#             if not res.success:
#                 print(f'ERROR ({fname}): `scipy.optimize.minimize` failed {res.message})')
#                 return None
#             # -> res.fun is the minimum of h(x)
#
#             # Set c such that c > f(x)/g(x) for all x in the domain
#             c = -res.fun + 1.e-3 # add small number to ensure the inequality
#
#         else:
#             # Set instrumental pdf proportional to: g = 1 (uniform distribution)
#             # Set g, g_rvs
#             g = lambda x: 1.0
#             def g_rvs(size=1):
#                 return xmin + scipy.stats.uniform.rvs(size=(size,dim)) * lx
#
#     # Apply accept-reject algo
#     naccept = 0
#     ntot = 0
#     x = []
#     if max_trial is None:
#         max_trial = np.inf
#     if show_progress:
#         progress = 0
#         progressOld = -1
#     while naccept < n:
#         nn = n - naccept
#         ntot = ntot+nn
#         xnew = g_rvs(size=nn)
#         # print('1', xnew)
#         # print('1b', np.all(np.all((xnew >= xmin, xnew < xmax), axis=0), axis=-1))
#         xnew = xnew[np.all(np.all((xnew >= xmin, xnew < xmax), axis=0), axis=-1)]
#         # print('2', xnew)
#         nn = len(xnew)
#         if nn == 0:
#             continue
#         u = np.random.random(size=nn)
#         xnew = xnew[u < (f(xnew)/(c*g(xnew))).reshape(nn)]
#         # print('3', xnew)
#         nn = len(xnew)
#         if nn == 0:
#             continue
#         x.extend(xnew)
#         naccept = naccept+nn
#         if show_progress:
#             progress = int(100*naccept/n)
#             if progress > progressOld:
#                 print(f'A-R algo, progress: {progress:3d} %')
#                 progressOld = progress
#         if ntot >= max_trial:
#             ok = False
#             break
#
#     x = np.asarray(x)
#     if dim == 1:
#         x = x.reshape(-1)
#
#     # # Apply accept-reject algo
#     # naccept = 0
#     # ntot = 0
#     # if max_trial is None:
#     #     max_trial = np.inf
#     # if show_progress:
#     #     progress = 0
#     #     progressOld = -1
#     # while naccept < n:
#     #     ntot = ntot+1
#     #     xnew = g_rvs()
#     #     if np.any((xnew < xmin, xnew >= xmax)):
#     #         continue
#     #     u = np.random.random()
#     #     if u < f(xnew)/(c*g(xnew)):
#     #         x[naccept] = xnew
#     #         naccept = naccept+1
#     #         if show_progress:
#     #             progress = int(100*naccept/n)
#     #             if progress > progressOld:
#     #                 print(f'A-R algo, progress: {progress:3d} %')
#     #                 progressOld = progress
#     #     if ntot >= max_trial:
#     #         ok = False
#     #         break
#
#     if naccept < n:
#         print(f'WARNING: sample size is only {naccept}! (increase `max_trial`)')
#
#     if return_accept_ratio:
#         accept_ratio = naccept/ntot
#         return x, accept_ratio
#     else:
#         return x
# # ----------------------------------------------------------------------------
