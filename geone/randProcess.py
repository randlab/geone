#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python module:  'randProcess.py'
author:         Julien Straubhaar
date:           may-2022

Module for miscellaneous algorithms based on random processes.
"""

import numpy as np
import scipy.stats as stats

# ----------------------------------------------------------------------------
def poissonPointProcess(mu, xmin=0.0, xmax=1.0, ninterval=None):
    """
    Generates random points in [xmin, xmax[ (can be multi-dimensional) following
    a Poisson point process.

    :param mu:      (float or ndarray or func) intensity of the Poisson process,
                        i.e. the mean number of points per unitary volume:
                        - if mu is a float:
                            homogeneous Poisson point process
                        - if mu is an ndarray:
                            (non-homogeneous Poisson point process)
                            mu[i_n, ..., i_0] is the intensity on the box
                            [xmin[j]+i_j*(xmax[j]-xmin[j])/mu.shape[n-j]],
                            j=0,...,n
                        - if mu is a function:
                            (non-homogeneous Poisson point process)
                            mu(x): returns the intensity at `x`,
                            with `x` array_like, the last axis of `x` denoting
                            the components
    :param xmin:    (float or 1d-array of floats of shape(m,)) lower bound of
                        each coordinate
    :param xmax:    (float or 1d-array of floats of shape(m,)) upper bound of
                        each coordinate
    :param ninterval:
                    (None, int or 1d-array of ints of shape(m,))
                        used only if `mu` is a function, `ninterval` contains
                        the number of interval(s) in which the domain
                        [xmin, xmax[ is subdivided along each axis
    :return pts:    (2d-array of shape(npts, m)) each row is a random point in
                        the domain, the number of points following a Poisson law
                        of the given intensity (`mu`) and m the dimension of the
                        domain
    """
    xmin = np.atleast_1d(xmin)
    xmax = np.atleast_1d(xmax)

    if xmin.ndim != xmax.ndim or xmin.ndim != 1:
        print("ERROR (poissonPointProcess): xmin, xmax not valid (dimension or shape)")
        return None

    if np.any(xmin >= xmax):
        print("ERROR (poissonPointProcess): xmin, xmax not valid ((component of) xmin less than or equal to xmax)")
        return None

    # dimension
    dim = len(xmin)

    if callable(mu):
        if ninterval is None:
            print("ERROR (poissonPointProcess): ninterval must be specified when a function is passed for the intensity (mu)")
            return None

        ninterval = np.asarray(ninterval, dtype=int)  # possibly 0-dimensional
        if ninterval.size == 1:
            ninterval = ninterval.flat[0] * np.ones(dim)
        elif ninterval.size != dim:
            print ('ERROR (poissonPointProcess): ninterval does not have an acceptable size')
            return None

        if np.any(ninterval < 1):
            print ('ERROR (poissonPointProcess): ninterval has negative or zero value')
            return None

    elif isinstance(mu, np.ndarray):
        if mu.ndim != dim:
            print ('ERROR (poissonPointProcess): inconsistent number of dimension for the ndarray `mu`')
            return None

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
    npts_cell = np.array([stats.poisson.rvs(m) for m in mu_cell])

    # Generate random points (uniformly) in each cell
    pts = np.array([np.hstack(
            [a + spa[i] * (np.random.random(size=npts) - 0.5) for a, npts in zip(xx_cell_center[:,i], npts_cell)]
        ) for i in range(dim)]).T

    return pts
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def chentsov1D(n_mean,
               dimension, spacing=1.0, origin=0.0,
               direction_origin=None,
               p_min=None, p_max=None,
               nreal=1):
    """
    Generates a Chentsov's simulation in 1D.
    The domain of simulation is [xmin, xmax], with nx cells along x-axis, each
    cell having a length of dx, the left side is the origin:
        - along x-axis:
            nx = dimension
            dx = spacing
            xmin = origin
            xmax = origin + nx*dx
    The simulation consists in
        1. Drawing random hyper-plane (i.e. points in 1D) in the space
            [p_min, p_max] following a Poisson point process with intensity
                mu = n_mean / vol([p_min, p_max]);
            the points are given in the parametrized form: p;
            then, for each point p, and with direction_origin = x0
            (the center of the simulation domain by default), the hyper-plane
            (point)
                {x : x-x0 = p} (i.e. the point x0 + p)
            is considered;
        2. Each hyper-plane (point x0+p) splits the space (R) in two parts
            (two half lines); the value = +1 is set to one part (chosen
            randomly) and the value -1 is set to the other part. Denoting V_i
            the value over the space (R) associated to the i-th hyper-plane
            (point), the value assigned to a grid cell of center x is set to
                Z(x) = 0.5 * sum_{i} (V_i(x) - V_i(x0))
            It corresponds to the number of hyper-planes (points) cut by the
            segment [x0, x].

    :param n_mean:      (float) mean number of hyper-plane drawn (via Poisson
                            process)
    :param dimension:   (int) nx, number of cells in the 1D simulation domain
    :param spacing:     (float) dx, spacing between two adjacent cells in the 1D
                            simulation domain
    :param origin:      (float) ox, origin of the 1D simulation domain
    :param direction_origin:
                        (float or None) origin from which the "points" are drawn
                            in the Poisson process (see above); by default (None),
                            the center of the 1D simulation domain is used
    :param p_min:       (float) minimal value for p (see above)
                            if p_min is None, p_min is set automatically to
                                - half of the length of the 1D simulation domain
    :param p_max:       (float) maximal value for p (see above)
                            if p_max is None, p_max is set automatically to
                                + half of the length of the 1D simulation domain
    :param nreal:       (int) number of realizations

    :return sim, n:     sim:    (2-dimensional array of dim nreal x nx) nreal
                                    simulation of Z (see above), sim[i] is the
                                    i-th realization
                        n:      (1-dimensional array of dim nreal) numbers of
                                    hyper-planes (points) drawn, n[i] is the the
                                    number of hyper-planes for the i-th
                                    realization
    """
    # Number of realization(s)
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        print('CHENTSOV1D: nreal <= 0: nothing to do!')

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
        print ("ERROR (CHENTSOV1D): 'p_min' is greater than or equal to 'p_max'")
        return None

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
        pts = poissonPointProcess(mu, p_min, p_max)
        n[k] = pts.shape[0]

        # Defines values of Z in each grid cell
        random_sign = (-1)**np.random.randint(2, size=n[k])
        for i in range(n[k]):
            z[k] = z[k] + (np.sign((xc-direction_origin)-pts[i])+np.sign(pts[i]))*random_sign[i]

    z = 0.5*z

    return z, n
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def chentsov2D(n_mean,
               dimension, spacing=(1.0, 1.0), origin=(0.0, 0.0),
               direction_origin=None,
               phi_min=0.0, phi_max=np.pi,
               p_min=None, p_max=None,
               nreal=1):
    """
    Generates a Chentsov's simulation in 2D.
    The domain of simulation is [xmin, xmax] x [ymin x ymax], with nx and ny
    cells along x-axis and y-axis respectively, each cell being a box of size
    dx x dy, the bottom left corner is the origin:
        - along x-axis:
            nx = dimension[0]
            dx = spacing[0]
            xmin = origin[0]
            xmax = origin[0] + nx*dx
        - along y-axis:
            ny = dimension[1]
            dy = spacing[1]
            ymin = origin[1]
            ymax = origin[1] + ny*dy
    The simulation consists in
        1. Drawing random hyper-plane (i.e. lines in 2D):
            considering the space S x [p_min, p_max], where S is a part of
            the circle of radius 1 in the plane (by default: half circle),
            parametrized via
                phi -> (cos(phi), sin(phi)), with phi in [phi_min, phi_max],
            some points are drawn randomly in S x [p_min, p_max] following a
            Poisson point process with intensity
                mu = n_mean / vol(S x [p_min, p_max]);
            the points are given in the parametrized form: (phi, p);
            then, for each point (phi, p), and with direction_origin = (x0, y0)
            (the center of the simulation domain by default), the hyper-plane
            (line)
                {(x, y) : dot([x-x0, y-y0], [cos(phi), sin(phi)]) = p}
            (i.e. point (x, y) s.t. the orthogonal projection of (x-x0, y-y0)
            onto the direction (cos(phi), sin(phi)) is equal to p) is considered;
        2. Each hyper-plane (line) splits the space (R^2) in two parts (two half
            planes); the value = +1 is set to one part (chosen randomly) and the
            value -1 is set to the other part. Denoting V_i the value over the
            space (R^2) associated to the i-th hyper-plane (line), the value
            assigned to a grid cell of center (x, y) is set to
                Z(x, y) = 0.5 * sum_{i} (V_i(x, y) - V_i(x0, y0))
            It corresponds to the number of hyper-planes cut by the segment
            [(x0, y0), (x, y)].

    :param n_mean:      (float) mean number of hyper-plane drawn (via Poisson
                            process)
    :param dimension:   (sequence of 2 ints) [nx, ny], number of cells in the 2D
                            simulation domain in x-, y-axis direction
    :param spacing:     (sequence of 2 floats) [dx, dy], spacing between
                            two adjacent cells in the 2D simulation domain in
                            x-, y-axis direction
    :param origin:      (sequence of 2 floats) [ox, oy], origin of the 2D
                            simulation domain
    :param direction_origin:
                        (sequence of 2 floats or None) origin from which the
                            directions are drawn in the Poisson process (see
                            above); by default (None), the center of the 2D
                            simulation domain is used
    :param phi_min:     (float) minimal angle for the parametrization of S (part
                            of circle) defining the direction (see above)
    :param phi_max:     (float) maximal angle for the parametrization of S (part
                            of circle) defining the direction (see above)
    :param p_min:       (float) minimal value for orthogonal projection (see
                            above); if p_min is None, p_min is set automatically
                            to
                                - half of the diagonal of the 2D simulation domain
    :param p_max:       (float) maximal value for orthogonal projection (see
                            above); if p_max is None, p_max is set automatically
                            to
                                + half of the diagonal of the 2D simulation domain
    :param nreal:       (int) number of realizations
    :return sim, n:
                        sim:    (3-dimensional array of dim nreal x ny x nx)
                                    nreal simulation of Z (see above), sim[i] is
                                    the i-th realization
                        n:      (1-dimensional array of dim nreal) numbers of
                                    hyper-planes (lines) drawn, n[i] is the the
                                    number of hyper-planes for the i-th
                                    realization
    """
    # Number of realization(s)
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        print('CHENTSOV2D: nreal <= 0: nothing to do!')

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
        print ("ERROR (CHENTSOV2D): 'p_min' is greater than or equal to 'p_max'")
        return None

    if phi_min >= phi_max:
        print ("ERROR (CHENTSOV2D): 'phi_min' is greater than or equal to 'phi_max'")
        return None

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
        pts = poissonPointProcess(mu, [phi_min, p_min], [phi_max, p_max])
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
def chentsov3D(n_mean,
               dimension, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
               direction_origin=None,
               phi_min=0.0, phi_max=2.0*np.pi,
               theta_min=0.0, theta_max=0.5*np.pi,
               p_min=None, p_max=None,
               ninterval_theta=100,
               nreal=1):
    """
    Generates a Chentsov's simulation in 3D.
    The domain of simulation is [xmin, xmax] x [ymin x ymax] x [zmin x zmax],
    with nx, ny, nz cells along x-axis, y-axis, z-axis respectively, each cell
    being a box of size dx x dy x dy, the bottom left down corner is the origin:
        - along x-axis:
            nx = dimension[0]
            dx = spacing[0]
            xmin = origin[0]
            xmax = origin[0] + nx*dx
        - along y-axis:
            ny = dimension[1]
            dy = spacing[1]
            ymin = origin[1]
            ymax = origin[1] + ny*dy
        - along z-axis:
            nz = dimension[0]
            dz = spacing[0]
            zmin = origin[0]
            zmax = origin[0] + nz*dz
    The simulation consists in
        1. Drawing random hyper-plane (i.e. planes in 3D):
            considering the space S x [p_min, p_max], where S is a part of
            the sphere of radius 1 in the 3D space (by default: half sphere),
            parametrized via
                (phi, theta) -> (cos(phi)cos(theta), sin(phi)cos(theta), sin(theta)),
                    with phi in [phi_min, phi_max], theta in [theta_min, theta_max]
            some points are drawn randomly in S x [p_min, p_max] following a
            Poisson point process with intensity
                mu = n_mean / vol(S x [p_min, p_max]);
            the points are given in the parametrized form: (phi, theta, p);
            then, for each point (phi, theta, p), and with
            direction_origin = (x0, y0, z0) (the center of the simulation domain
            by default), the hyper-plane (plane)
                {(x, y, z) : dot([x-x0, y-y0, z-z0], [cos(phi)cos(theta), sin(phi)cos(theta), sin(theta)]) = p}
            (i.e. point (x, y, z) s.t. the orthogonal projection of
            (x-x0, y-y0, z-z0) onto the direction
            (cos(phi)cos(theta), sin(phi)cos(theta), sin(theta)) is equal to p)
            is considered;
        2. Each hyper-plane (plane) splits the space (R^3) in two parts;
            the value = +1 is set to one part (chosen randomly) and the value -1
            is set to the other part. Denoting V_i the value over the space (R^3)
            associated to the i-th hyper-plane (plane), the value assigned to a
            grid cell of center (x, y) is set to
                Z(x, y) = 0.5 * sum_{i} (V_i(x, y) - V_i(x0, y0))
            It corresponds to the number of hyper-planes (planes) cut by the
            segment [(x0, y0, z0), (x, y, z)].

    :param n:           (float) mean number of hyper-plane drawn (via Poisson
                            process)
    :param dimension:   (sequence of 3 ints) [nx, ny, nz], number of cells in
                            the 3D simulation domain in x-, y-, z-axis direction
    :param spacing:     (sequence of 3 floats) [dx, dy, dz], spacing between
                            two adjacent cells in the 3D simulation domain in
                            x-, y-, z-axis direction
    :param origin:      (sequence of 3 floats) [ox, oy, oz], origin of the 3D
                            simulation domain
    :param direction_origin:
                        (sequence of 3 floats or None) origin from which the
                            directions are drawn in the Poisson process (see
                            above); by default (None), the center of the 3D
                            simulation domain is used
    :param phi_min:     (float) minimal angle for the parametrization of S (part
                            of sphere) defining the direction (see above)
    :param phi_max:     (float) maximal angle for the parametrization of S (part
                            of sphere) defining the direction (see above)
    :param theta_min:   (float) minimal angle for the parametrization of S (part
                            of sphere) defining the direction (see above)
    :param theta_max:   (float) maximal angle for the parametrization of S (part
                            of sphere) defining the direction (see above)
    :param p_min:       (float) minimal value for orthogonal projection (see
                            above); if p_min is None, p_min is set automatically
                            to
                                - half of the diagonal of the 2D simulation domain
    :param p_max:       (float) maximal value for orthogonal projection (see 
                            above); if p_max is None, p_max is set automatically
                            to
                                + half of the diagonal of the 2D simulation domain
    :param ninterval_theta:
                        (int) number of sub-intervals in which the interval
                            [theta_min, theta_max] is subdivided for applying the
                            Poisson process
    :param nreal:       (int) number of realizations
    :return sim, n:
                        sim:    (4-dimensional array of dim nreal x nz x ny x nx)
                                    nreal simulation of Z (see above), sim[i] is
                                    the i-th realization
                        n:      (1-dimensional array of dim nreal) numbers of
                                    hyper-planes (planes) drawn, n[i] is the
                                    number of hyper-planes for the i-th
                                    realization
    """
    # Number of realization(s)
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        print('CHENTSOV3D: nreal <= 0: nothing to do!')

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
        print ("ERROR (CHENTSOV3D): 'p_min' is greater than or equal to 'p_max'")
        return None

    if phi_min >= phi_max:
        print ("ERROR (CHENTSOV3D): 'phi_min' is greater than or equal to 'phi_max'")
        return None

    if theta_min >= theta_max:
        print ("ERROR (CHENTSOV3D): 'theta_min' is greater than or equal to 'theta_max'")
        return None

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
        pts = poissonPointProcess(mu, [phi_min, theta_min, p_min], [phi_max, theta_max, p_max], ninterval=[1, ninterval_theta, 1])
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
