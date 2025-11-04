#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# Python module:  'tools.py'
# author:         Julien Straubhaar
# date:           nov-2023
# -------------------------------------------------------------------------

"""
Module for miscellaneous tools.
"""

import sys
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

from geone import img

# -----------------------------------------------------------------------------
def add_path_by_drawing(
        path_list,
        close=False,
        show_instructions=True,
        last_point_marker='o',
        last_point_color='red',
        **kwargs):
    """
    Add paths in a list, by interatively drawing on a plot.

    The first argument of the function is a list that is updated when the
    function is called by appending one (or more) path(s) (see notes below).
    A path is a 2D array of floats with two columns, each row is (the x and y
    coordinates of) a point. A path is interactively determined on the plot on
    the current axis (get with `matplotlib.pyplot.gca()`), with the following
    rules:

    - left click: add the next point (or first one)
    - right click: remove the last point

    When pressing a key:

    - key n/N: terminate the current path, and start a new path
    - key ENTER (or other): terminate the current path and exits

    Parameters
    ----------
    path_list : list
        list of paths, that will be updated by appending the path interactively
        drawn in the current axis (one can start with `path_list = []`)

    close : bool, default: False
        if `True`: when a path is terminated, the first points of the path is
        replicated at the end of the path to form a close line / path

    show_instructions : bool, default: True
        if `True`: instructions are printed in the standard output

    last_point_marker : "matplotlib marker", default: 'o'
        marker used for highlighting the last clicked point

    last_point_color : "matplotlib color", default: 'red'
        color used for highlighting the last clicked point

    kwargs : dict
        keyword arguments passed to `matplotlib.pyplot.plot` to plot the path(s)

    Notes
    -----
    * The function does not return anything. The first argument, `path_list` is \
    updated, with the path(s) drawn interactively, for example `path_list[-1]` \
    is the last path, a 2D array, where `path_list[-1][i]` is the i-th point \
    (1D array of two floats) of that path
    * An interactive maplotlib backend must be used, so that this function works \
    properly
    """
    # fname = 'add_path_by_drawing'

    ax = plt.gca()
    obj_drawn = []
    x, y = [], []

    set_default_color = False
    if 'color' not in kwargs.keys() and 'c' not in kwargs.keys():
        set_default_color = True
        col = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ncol = len(col)
        col_ind = [0]
        kwargs['color'] = col[col_ind[0]] # default color for lines (segments)
    # if 'color' not in kwargs.keys() and 'c' not in kwargs.keys():
    #     kwargs['color'] = 'tab:blue' # default color for lines (segments)

    if show_instructions:
        instruct0 = '\n'.join(['   Left click: add next point', '   Right click: remove last point', '   Key n/N: select a new line', '   key ENTER (or other): finish (quit)'])
        print('\n'.join(['Draw path', instruct0]))
        sys.stdout.flush()

    def on_click(event):
        if not event.inaxes:
            return
        if event.button is MouseButton.LEFT:
            if len(x):
                # remove last point from obj_drawn
                ax.lines[-1].remove()
            # add clicked point
            x.append(event.xdata)
            y.append(event.ydata)
            if len(x) > 1:
                # add line (segment) to obj_drawn
                obj_drawn.append(plt.plot(x[-2:], y[-2:], **kwargs))
            # add point to obj_drawn
            obj_drawn.append(plt.plot(x[-1], y[-1], marker=last_point_marker, color=last_point_color))
        if event.button is MouseButton.RIGHT:
            if len(x):
                # remove last clicked point
                del x[-1], y[-1]
                # remove last point from obj_drawn
                ax.lines[-1].remove()
            if len(x):
                # point(s) are still in the line
                # remove last line (segment) from obj_drawn
                ax.lines[-1].remove()
                # add last point to obj_drawn
                obj_drawn.append(plt.plot(x[-1], y[-1], marker=last_point_marker, color=last_point_color))

    def on_key(event):
        if len(x):
            # remove last point from obj_drawn
            ax.lines[-1].remove()
            if close:
                # close the line
                if len(x):
                    x.append(x[0])
                    y.append(y[0])
                    # add line (closing segment) to obj_drawn
                    obj_drawn.append(plt.plot(x[-2:], y[-2:], **kwargs))
        # add path to path_list
        if len(x):
            path_list.append(np.array((x, y)).T)
        if event.key.lower() == 'n':
            if show_instructions:
                print('\n'.join(['Draw next path', instruct0]))
                sys.stdout.flush()
            x.clear() # Do not use: x = [], because x is global for this function!
            y.clear()
            if set_default_color:
                # Set color for next line
                col_ind[0] = (col_ind[0]+1)%ncol
                kwargs['color'] = col[col_ind[0]] # default color for lines (segments)
            return

        plt.disconnect(cid_click)
        plt.disconnect(cid_key)
        return
        # if
        # path_list.append(np.array((x, y)).T)
        # cid_click = plt.connect('button_press_event', on_click)


    cid_click = plt.connect('button_press_event', on_click)
    cid_key = plt.connect('key_press_event', on_key)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def is_in_polygon(x, vertices, wrap=None, return_sum_of_angles=False, **kwargs):
    """
    Checks if point(s) is (are) in a polygon given by its vertices forming a close line.

    To check if a point is in the polygon, the method consists in computing
    the vectors from the given point to the vertices of the polygon and
    the sum of signed angles between two successives vectors (and the last
    and first one). Then, the point is in the polygon if and only if this sum
    is equal to +/- 2 pi.

    Note that if the sum of angles is +2 pi (resp. -2 pi), then the vertices form
    a close line counterclockwise (resp. clockwise); this can be checked by
    specifying a point `x` in the polygon and `return_sum_of_angles=True`.

    Parameters
    ----------
    x : 2D array-like or 1D array-like
        point(s) coordinates, each row `x[i]` (if 2D array-like) (or `x` if
        1D array-like) contains the two coordinates of one point

    vertices : 2D array
        vertices of a polygon in 2D, each row of `vertices` contains the two
        coordinates of one vertex; the segments of the polygon are obtained by
        linking two successive vertices (as well as the last one with the first
        one, if `wrap=True` (see below)), so that the vertices form a close line
        (clockwise or counterclockwise)

    wrap : bool, optional
        - if `True`: last and first vertices has to be linked to form a close line
        - if `False`: last and first vertices should be the same ones (i.e. the \
        vertices form a close line);

        by default (`None`): `wrap` is automatically computed

    return_sum_of_angles : bool, default: False
        if `True`, the sum of angles (computed by the method) is returned

    kwargs :
        keyword arguments passed to function `numpy.isclose`

    Returns
    -------
    out : 1D array of bools, or bool
        indicates for each point in `x` if it is inside (True) or outside (False)
        the polygon;
        note: if `x` is of shape (m, 2), then `out` is a 1D array of shape (m, ),
        and if `x` is of shape (2, ) (one point), `out` is bool

    sum_of_angles : 1D array of floats, or float
        returned if `return_sum_of_angles=True`; for each point in `x`, the sum
        of angles computed by the method is returned (`nan` if the sum is not
        computed);
        note: `sum_of_angles` is an array of same shape as `out` or a float
    """
    # fname = 'is_in_polygon'

    # Set wrap (and adjust vertices) if needed
    if wrap is None:
        wrap = ~np.isclose(np.sqrt(((vertices[-1] - vertices[0])**2).sum()), 0.0)
    if not wrap:
        # remove last vertice (should be equal to the first one)
        vertices = np.delete(vertices, -1, axis=0)

    # Array of shifted indices (on vertices)
    ind = np.hstack((np.arange(1, vertices.shape[0]), [0]))

    # Initialization
    xx = np.atleast_2d(x)
    res = np.zeros(xx.shape[0], dtype='bool')
    if return_sum_of_angles:
        res_sum = np.full((xx.shape[0],), np.nan)

    xmin, ymin = vertices.min(axis=0)
    xmax, ymax = vertices.max(axis=0)

    for j, xj in enumerate(xx):
        if xj[0] < xmin or xj[0] > xmax or xj[1] < ymin or xj[1] > ymax:
            continue

        # Set a, b, c: edge of the triangles to compute angle between a and b
        a = xj - vertices

        # a_norm2: square norm of a
        a_norm2 = (a**2).sum(axis=1)

        b = a[ind]
        b_norm2 = a_norm2[ind]
        ab_norm = np.sqrt(a_norm2*b_norm2)

        if np.any(np.isclose(ab_norm, 0, **kwargs)):
            # xj on the border of the polygon
            continue

        # Compute the sum of angles using the theorem of cosine
        c = b - a
        c_norm2 = (c**2).sum(axis=1)
        sign = 2*(a[:,0]*b[:,1] - a[:,1]*b[:,0] > 0) - 1
        sum_angles = np.sum(sign*np.arccos(np.minimum(1.0, np.maximum(-1.0, (a_norm2 + b_norm2 - c_norm2)/(2.0*ab_norm)))))

        res[j] = np.isclose(np.abs(sum_angles), 2*np.pi)
        if return_sum_of_angles:
            res_sum[j] = sum_angles

    if np.asarray(x).ndim == 1:
        res = res[0]
        if return_sum_of_angles:
            res_sum = res_sum[0]

    if return_sum_of_angles:
        return res, res_sum

    return res
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def is_in_polygon_mp(x, vertices, wrap=None, return_sum_of_angles=False, nproc=-1, **kwargs):
    """
    Computes the same as the function :func:`is_in_polygon`, using multiprocessing.

    All the parameters except `nproc` are the same as those of the function
    :func:`is_in_polygon`.

    The number of processes used (in parallel) is n, and determined by the
    parameter `nproc` (int, optional) as follows:

    - if `nproc > 0`: n = `nproc`,
    - if `nproc <= 0`: n = max(nmax+`nproc`, 1), where nmax is the total \
    number of cpu(s) of the system (retrieved by `multiprocessing.cpu_count()`), \
    i.e. all cpus except `-nproc` is used (but at least one)

    See function :func:`is_in_polygon`.
    """
    # fname = 'is_in_polygon_mp'

    # Set wrap (and adjust vertices) if needed
    if wrap is None:
        wrap = ~np.isclose(np.sqrt(((vertices[-1] - vertices[0])**2).sum()), 0.0)
    if not wrap:
        # remove last vertice (should be equal to the first one)
        vertices = np.delete(vertices, -1, axis=0)

    # Set wrap key in keywords arguments
    kwargs['wrap'] = True

    # Set return_sum_of_angles key in keywords arguments
    kwargs['return_sum_of_angles'] = return_sum_of_angles

    # Initialization
    xx = np.atleast_2d(x)

    # Set number of processes (n)
    if nproc > 0:
        n = nproc
    else:
        n = min(multiprocessing.cpu_count()+nproc, 1)

    # Set index for distributing tasks
    q, r = np.divmod(xx.shape[0], n)
    ids_proc = [i*q + min(i, r) for i in range(n+1)]

    # Set pool of n workers
    pool = multiprocessing.Pool(n)
    out_pool = []
    for i in range(n):
        # Set i-th process
        out_pool.append(pool.apply_async(is_in_polygon, args=(xx[ids_proc[i]:ids_proc[i+1]], vertices), kwds=kwargs))

    # Properly end working process
    pool.close() # Prevents any more tasks from being submitted to the pool,
    pool.join()  # then, wait for the worker processes to exit.

    # Get result from each process
    if return_sum_of_angles:
        res = []
        res_sum = []
        for w in out_pool:
            r, s = w.get()
            res.append(r)
            res_sum.append(s)
        res = np.hstack(res)
        res_sum = np.hstack(res_sum)
    else:
        res = np.hstack([w.get() for w in out_pool])

    if np.asarray(x).ndim == 1:
        res = res[0]
        if return_sum_of_angles:
            res_sum = res_sum[0]

    if return_sum_of_angles:
        return res, res_sum

    return res
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def rasterize_polygon_2d(
        vertices,
        nx=None, ny=None,
        sx=None, sy=None,
        ox=None, oy=None,
        xmin_ext=0.0, xmax_ext=0.0,
        ymin_ext=0.0, ymax_ext=0.0,
        wrap=None,
        logger=None,
        **kwargs):
    """
    Rasterizes a polygon (close line) in a 2D grid.

    This function returns an image with one variable indicating for each cell
    if it is inside (1) or outside (0) the polygon defined by the given
    vertices.

    The grid geometry of the output image is set by the given parameters or
    computed from the vertices, as in function :func:`geone.img.imageFromPoints`,
    i.e. for the x axis (similar for y):

    - `ox` (origin), `nx` (number of cells) and `sx` (resolution, cell size)
    - or only `nx`: `ox` and `sx` automatically computed
    - or only `sx`: `ox` and `nx` automatically computed

    In the two last cases, the parameters `xmin_ext`, `xmax_ext`, are used and
    the approximate limit of the grid along x axis is set to x0, x1, where

    - x0: min x coordinate of the vertices minus `xmin_ext`
    - x1: max x coordinate of the vertices plus `xmax_ext`

    Parameters
    ----------
    vertices : 2D array
        vertices of a polygon in 2D, each row of `vertices` contains the two
        coordinates of one vertex; the segments of the polygon are obtained by
        linking two successive vertices (as well as the last one with the first
        one, if `wrap=True` (see below)), so that the vertices form a close line
        (clockwise or counterclockwise)

    nx : int, optional
        number of grid cells along x axis; see above for possible inputs

    ny : int, optional
        number of grid cells along y axis; see above for possible inputs

    sx : float, optional
        cell size along x axis; see above for possible inputs

    sy : float, optional
        cell size along y axis; see above for possible inputs

    ox : float, optional
        origin of the grid along x axis (x coordinate of cell border);
        see above for possible

    oy : float, optional
        origin of the grid along y axis (y coordinate of cell border);
        see above for possible

        Note: `(ox, oy)` is the "lower-left" corner of the grid

    xmin_ext : float, default: 0.0
        extension beyond the min x coordinate of the vertices (see above)

    xmax_ext : float, default: 0.0
        extension beyond the max x coordinate of the vertices (see above)

    ymin_ext : float, default: 0.0
        extension beyond the min y coordinate of the vertices (see above)

    ymax_ext : float, default: 0.0
        extension beyond the max y coordinate of the vertices (see above)

    wrap : bool, optional
        - if `True`: last and first vertices has to be linked to form a close line
        - if `False`: last and first vertices should be the same ones (i.e. the \
        vertices form a close line);

        by default (`None`): `wrap` is automatically computed

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs:
        keyword arguments passed to function :func:`is_in_polygon`

    Returns
    -------
    im : :class:`geone.img.Img`
        output image (see above);
        note: the image grid is defined in 3D with `nz=1`, `sz=1.0`, `oz=-0.5`
    """
    # fname = 'rasterize_polygon_2d'

    # Define grid geometry (image with no variable)
    im = img.imageFromPoints(vertices,
                             nx=nx, ny=ny, sx=sx, sy=sy, ox=ox, oy=oy,
                             xmin_ext=xmin_ext, xmax_ext=xmax_ext,
                             ymin_ext=ymin_ext, ymax_ext=ymax_ext, 
                             logger=logger)

    # Rasterize: for each cell, check if its center is within the grid
    v = np.asarray(is_in_polygon(np.array((im.xx().reshape(-1), im.yy().reshape(-1))).T, vertices, wrap=wrap, **kwargs)).astype('float')
    im.append_var(v, varname='in', logger=logger)

    return im
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def rasterize_polygon_2d_mp(
        vertices,
        nx=None, ny=None,
        sx=None, sy=None,
        ox=None, oy=None,
        xmin_ext=0.0, xmax_ext=0.0,
        ymin_ext=0.0, ymax_ext=0.0,
        wrap=None,
        nproc=-1,
        logger=None,
        **kwargs):
    """
    Computes the same as the function :func:`rasterize_polygon_2d`, using multiprocessing.

    All the parameters except `nproc` are the same as those of the function
    :func:`rasterize_polygon_2d`.

    The number of processes used (in parallel) is n, and determined by the
    parameter `nproc` (int, optional) as follows:

    - if `nproc > 0`: n = `nproc`,
    - if `nproc <= 0`: n = max(nmax+`nproc`, 1), where nmax is the total \
    number of cpu(s) of the system (retrieved by `multiprocessing.cpu_count()`), \
    i.e. all cpus except `-nproc` is used (but at least one).

    See function :func:`rasterize_polygon_2d`.
    """
    # fname = 'rasterize_polygon_2d_mp'

    # Define grid geometry (image with no variable)
    im = img.imageFromPoints(vertices,
                             nx=nx, ny=ny, sx=sx, sy=sy, ox=ox, oy=oy,
                             xmin_ext=xmin_ext, xmax_ext=xmax_ext,
                             ymin_ext=ymin_ext, ymax_ext=ymax_ext,
                             logger=logger)

    # Rasterize: for each cell, check if its center is within the grid
    v = np.asarray(is_in_polygon_mp(np.array((im.xx().reshape(-1), im.yy().reshape(-1))).T, vertices, wrap=wrap, nproc=nproc, **kwargs)).astype('float')
    im.append_var(v, varname='in', logger=logger)

    return im
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def curv_coord_2d_from_center_line(
        x, cl_position, im_cl_dist,
        cl_u=None,
        gradx=None,
        grady=None,
        dg=None,
        gradtol=1.e-5,
        path_len_max=10000,
        return_path=False,
        verbose=1,
        logger=None):
    """
    Computes curvilinear coordinates in 2D from a center line, for points given in standard coordinates.

    This functions allows to change coordinates system in 2D. For a point in 2D,
    let the coordinates

    - u = (u1, u2) (in 2D) in curvilinear system
    - x = (x1, x2) (in 2D) in standard system

    The curvilinear coordinates system (u) is defined according to a center line
    in a 2D grid as follows:

    - considering the distance map (geone image `im_cl_dist`) of L2 distance \
    to the center line (`cl_position`)
    - the path from x to the point I on the center line is computed, descending \
    the gradient (`gradx`, `grady`) of the distance map
    - u = (u1, u2) is defined as:
        - u1: the distance along the center line to the point I,
        - u2: +/-the value of the distance map at x, with
            * sign + for point "at left" of the center line and,
            * sign - for point "at right" of the center line.

    Parameters
    ----------
    x : 2D array-like or 1D array-like
        points coordinates in standard system (should be in the grid of the image
        `im_cl_dist`, see below), each row `x[i]` (if 2D array-like) (or `x` if
        1D array-like) contains the two coordinates of one point

    cl_position : 2D array of shape (n, 2)
        position of the points of the center line (in standard system);
        note: the distance between two successive points of the center line gives
        the resolution of the u1 coordinate

    im_cl_dist : :class:`geone.img.Img`
        image of the distance to the center line:

        - its grid is the "support" of standard coordinate system and it \
        should contain all the points `x`
        - the center line (`cl_position`) should "separate" the image grid \
        in two (disconnected) regions
        - this image can be computed using the function \
        `geone.geosclassicinterface.imgDistanceImage`

    cl_u : 1D array-like of length n, optional
        distance along the center line (automatically computed if not given
        (`None`)), used for determining u1 coordinate

    gradx, grady : 2D array-like, optional
        gradient of the distance to the centerline, array of shape
        `(im_cl_dist.ny, im_cl_dist.nx)` (automatically computed if not given
        (`None`));
        `gradx[iy, ix], grady[iy, ix]`: gradient in grid cell of index `iy`, `ix`
        along x and y axes respectively

    dg : float, optional
        step (length) used for descending the gradient map; by default (`None`):
        `dg` is set to the minimal distance between two successive points in
        `cl_position`, (i.e. minimal difference between two succesive values in
        `cl_u`)

    gradtol : float, default: 1.e-5
        tolerance for the gradient magnitude (if the magintude of the gradient
        is below `gradtol`, it is considered as zero vector)

    path_len_max : int, default: 10000
        maximal length of the path from the initial point(s) (`x`) to the center
        line

    return_path : bool, default: False
        indicates if the path(s) from the initial point(s) (`x`) to the point(s)
        I of the centerline (descending the gradient of the distance map) is
        (are) retrieved

    verbose : int, default: 1
        verbose mode, integer >=0, higher implies more display

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    u : 2D array or 1D array (same shape as `x`)
        coordinates in curvilinear system of the input point(s) `x` (see above)

    x_path : list of 2D arrays (or 2D array), optional
        path(s) from the initial point(s) to the point(s) I of the centerline
        (descending the gradient of the distance map):

        - `x_path[i]` : 2D array of floats with 2 columns
            * `xpath[i][j]` is the coordinates (in standard system) of the \
            j-th point of the path from `x[i]` to the center line

        note: if `x` is reduced to one point and given as 1D array-like, then
        `x_path` is a 2D array of floats with 2 columns containing the path from
        `x` to the center line;
        returned if `returned_path=True`
    """
    fname = 'curv_coord_2d_from_center_line'

    if cl_u is None:
        cl_u = np.insert(np.cumsum(np.sqrt(((cl_position[1:,:] - cl_position[:-1,:])**2).sum(axis=1))), 0, 0.0)

    if gradx is None or grady is None:
        # Gradient of distance map
        grady, gradx = np.gradient(im_cl_dist.val[0,0])
        gradx = gradx / im_cl_dist.sx
        grady = grady / im_cl_dist.sy

    if dg is None:
        dg = np.diff(cl_u).min()

    x = np.asarray(x)
    x_ndim = x.ndim
    x = np.atleast_2d(x)
    u = np.zeros_like(x)

    if return_path:
        x_path = []

    for i in range(x.shape[0]):
        # Treat the i-th point
        x_cur = x[i]
        if return_path:
            xi_path = [x_cur]

        u2 = 0.0
        d_prev = np.inf

        for j in range(path_len_max):
            # index in the grid (and interpolation factor) of the "current" point
            tx = max(0, min((x_cur[0] - im_cl_dist.ox)/im_cl_dist.sx - 0.5, im_cl_dist.nx - 1.00001))
            ix = int(tx)
            tx = tx - int(tx)

            ty = max(0, min((x_cur[1] - im_cl_dist.oy)/im_cl_dist.sy - 0.5, im_cl_dist.ny - 1.00001))
            iy = int(ty)
            ty = ty - int(ty)

            # "current" distance to the center line
            d_cur = (1.-ty)*((1.-tx)*im_cl_dist.val[0, 0, iy, ix] + tx*im_cl_dist.val[0, 0, iy, ix+1]) + ty*((1.-tx)*im_cl_dist.val[0, 0, iy+1, ix] + tx*im_cl_dist.val[0, 0, iy+1, ix+1])

            # if np.abs(d_cur) < dg or d_cur*d_prev < 0:
            if d_cur < dg or d_cur > d_prev:
                break

            # "current" gradient
            gradx_cur = (1.-ty)*((1.-tx)*gradx[iy, ix] + tx*gradx[iy, ix+1]) + ty*((1.-tx)*gradx[iy+1, ix] + tx*gradx[iy+1, ix+1])
            grady_cur = (1.-ty)*((1.-tx)*grady[iy, ix] + tx*grady[iy, ix+1]) + ty*((1.-tx)*grady[iy+1, ix] + tx*grady[iy+1, ix+1])

            gradl = np.sqrt(gradx_cur**2 + grady_cur**2)
            if gradl < gradtol:
                break

            # compute next point descending the gradient
            # x_cur = x_cur - np.sign(d_cur) * dg*np.array([gradx_cur, grady_cur])/gradl
            x_cur = x_cur - dg*np.array([gradx_cur, grady_cur])/gradl
            if return_path:
                xi_path.append(x_cur)

            u2 = u2 + dg
            d_prev = d_cur

        # Finalize the computation of coordinate (u1, u2)
        d_cur = ((cl_position[:-1,:] - x_cur)**2).sum(axis=1)
        try:
            k = np.where(d_cur == d_cur.min())[0][0]
        except:
            # k = len(cl_position[-2])
            k = len(cl_position) - 2
            if verbose > 0:
                if logger:
                    logger.warning(f'{fname}: closest point on center line not found (last segment selected)')
                else:
                    print(f'{fname}: WARNING: closest point on center line not found (last segment selected)')
        
        u1 = cl_u[k]

        s = np.sign(np.linalg.det(np.vstack((cl_position[k+1] - cl_position[k], x[i] - x_cur))))
        u2 = s*u2

        u[i] = np.array([u1, u2])

        if return_path:
            x_path.append(np.asarray(xi_path))

    if return_path:
        if x_ndim == 1:
            u = u[0]
            x_path = x_path[0]
        return u, x_path
    else:
        if x_ndim == 1:
            u = u[0]
        return u
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
def curv_coord_2d_from_center_line_mp(
        x, cl_position, im_cl_dist,
        cl_u=None,
        gradx=None,
        grady=None,
        dg=None,
        gradtol=1.e-5,
        path_len_max=10000,
        return_path=False,
        nproc=-1,
        logger=None):
    """
    Computes the same as the function :func:`curv_coord_2d_from_center_line`, using multiprocessing.

    All the parameters except `nproc` are the same as those of the function
    :func:`curv_coord_2d_from_center_line`.

    The number of processes used (in parallel) is n, and determined by the
    parameter `nproc` (int, optional) as follows:

    - if `nproc > 0`: n = `nproc`,
    - if `nproc <= 0`: n = max(nmax+`nproc`, 1), where nmax is the total \
    number of cpu(s) of the system (retrieved by `multiprocessing.cpu_count()`), \
    i.e. all cpus except `-nproc` is used (but at least one).

    See function :func:`curv_coord_2d_from_center_line`.
    """
    # fname = 'curv_coord_2d_from_center_line_mp'

    # Initialization
    xx = np.atleast_2d(x)

    # Set number of processes (n)
    if nproc > 0:
        n = nproc
    else:
        n = min(multiprocessing.cpu_count()+nproc, 1)

    # Set index for distributing tasks
    q, r = np.divmod(xx.shape[0], n)
    ids_proc = [i*q + min(i, r) for i in range(n+1)]

    kwargs = dict(cl_u=cl_u, gradx=gradx, grady=grady, dg=dg, gradtol=gradtol, path_len_max=path_len_max, return_path=return_path, verbose=0, logger=logger)
    # Set pool of n workers
    pool = multiprocessing.Pool(n)
    out_pool = []
    for i in range(n):
        # Set i-th process
        out_pool.append(pool.apply_async(curv_coord_2d_from_center_line, args=(xx[ids_proc[i]:ids_proc[i+1]], cl_position, im_cl_dist), kwds=kwargs))

    # Properly end working process
    pool.close() # Prevents any more tasks from being submitted to the pool,
    pool.join()  # then, wait for the worker processes to exit.

    # Get result from each process
    if return_path:
        u = []
        x_path = []
        for p in out_pool:
            u_p, x_path_p = p.get()
            u.extend(u_p)
            x_path.extend(x_path_p)
        if np.asarray(x).ndim == 1:
            u = u[0]
            x_path = x_path[0]
        return u, x_path
    else:
        u = np.vstack([p.get() for p in out_pool])
        if np.asarray(x).ndim == 1:
            u = u[0]
        return u
# -----------------------------------------------------------------------------

##### OLD BELOW #####
# # -----------------------------------------------------------------------------
# def sector_angle(x, line, **kwargs):
#     """
#     Checks if point(s) (`x`) is (are) in a polygon given by its vertices
#     `vertices` forming a close line.
#
#     To check if a point is in the polygon, the method consists in computing
#     the vectors from the given point to the vertices of the polygon and
#     the sum of signed angles between two successives vectors (and the last
#     and first one). The point is in the polygon if and only if this sum is
#     equal to +/- 2 pi.
#
#     :param x:       (1d-array of 2 floats, or 2d-array of shape (m, 2))
#                         one or several points in 2D
#     :param vertices:(2d-array of shape (n,2)) vertices of a polygon in 2D,
#                         the segments of the polygon are obtained by linking
#                         two successive vertices (and the last one with the
#                         first one, if `wrap` is True (see below) or automatically
#                         checked if it is needed), so that the vertices form a
#                         close line (clockwise or counterclockwise)
#     :param wrap:    (bool)
#                         - if True, last and first vertices has to be linked
#                             to form a close line,
#                         - if False, last and first vertices should be the same
#                             ones (i.e. the vertices form a close line)
#                         - if None, automatically computed
#     :param kwargs:  keyword arguments passed to `numpy.isclose` (see below)
#
#     :return:        (bool or 1d-array of bool of shape (m,)) True / False
#                         value(s) indicating if the point(s) `x` is (are)
#                         inside / outside the polygon
#     """
#
#     # Initialization
#     xx = np.atleast_2d(x)
#     res = np.zeros(xx.shape[0], dtype='bool')
#
#     for j, xj in enumerate(xx):
#         # Set a, b, c: edge of the triangles to compute angle between a and b
#         v = xj - vertices
#
#         # v_norm2: square norm of v
#         v_norm2 = (a**2).sum(axis=1)
#
#         a = v[:-1]
#         b = v[1:]
# #        b_norm2 = v_norm2[1:]
#         ab_norm = np.sqrt(v_norm2[:-1]*v_norm2[1:])
#
#         ind = np.isclose(ab_norm, 0, **kwargs)
#         # Compute the sum of angles using the theorem of cosine
#         c = b - a
#         c_norm2 = (c**2).sum(axis=1)
#         sign = 2*(a[:,0]*b[:,1] - a[:,1]*b[:,0] > 0) - 1
#         sum_angles = np.sum(sign[ind]*np.arccos(np.minimum(1.0, np.maximum(-1.0, (a_norm2[ind] + b_norm2[ind] - c_norm2[ind])/(2.0*ab_norm[ind])))))
#         res[i] = sum_angles > 0
#
#     return res
# # -----------------------------------------------------------------------------
