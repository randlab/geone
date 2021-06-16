# ----------------------------------------------------------------------------
def sgs2D(cov_model, dimension, spacing, origin=[0., 0.],
          search_radius=None, nreal=1, mean=0, var=None,
          x=None, v=None,
          printInfo=True):

#def sgs(x, v, xu, cov_model, nneighbors=None):
    """
    Ordinary kriging - interpolates at locations xu the values v measured at locations x.
    Covariance model given should be:
        - in same dimension as dimension of locations x, xu
        - in 1D, it is then used as an omni-directional covariance model
    (see below).

    :param x:       (2-dimensional array of shape (n, d)) coordinates
                        of the data points (n: number of points, d: dimension)
                        Note: for data in 1D, it can be a 1-dimensional array of shape (n,)
    :param v:       (1-dimensional array of shape (n,)) values at data points

    :param xu:      (2-dimensional array of shape (nu, d)) coordinates
                        of the points where the interpolation has to be done
                        (nu: number of points, d: dimension same as for x),
                        called unknown points
                        Note: for data in 1D, it can be a 1-dimensional array of shape (nu,)

    :param cov_model:   covariance model:
                            - in same dimension as dimension of points (d), i.e.:
                                - CovModel1D class if data in 1D (d=1)
                                - CovModel2D class if data in 2D (d=2)
                                - CovModel3D class if data in 3D (d=3)
                            - or CovModel1D whatever dimension of points (d):
                                - used as an omni-directional covariance model

    :return:        (vu, vu_std) with:
                        vu:     (1-dimensional array of shape (nu,)) kriged values (estimates) at points xu
                        vu_std: (1-dimensional array of shape (nu,)) kriged standard deviation at points xu
    """
    # Check first argument and get covariance function
    if cov_model.__class__.__name__ == 'function':
        # covariance function is given
        cov_func = cov_model
        range_known = False
    elif cov_model.__class__.__name__ == 'CovModel2D':
        cov_func = cov_model.func() # covariance function
        range_known = True
    else:
        print("ERROR: 'cov_model' (first argument) is not valid")
        return None

    # Number of realization(s)
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if printInfo:
            print('SGS2D: nreal = 0: nothing to do!')
        return None

    if search_radius is None:
        # default search_radius
        if range_known:
            search_radius = cov_model.r()
        else:
            print("ERROR (SGS2D): 'search_radius' should be given")
            return None

    # if printInfo:
    #     print('SGS1D: Preliminary computation...')

    #### Preliminary computation ####
    nx, ny = dimension
    dx, dy = spacing
    # ox, oy = origin

    nxy = nx*ny

    sgs = np.nan * np.zeros((nreal, ny, nx))

    if x is not None:
        x = np.asarray(x).reshape(-1, 2) # cast in 2-dimensional array if needed
        v = np.asarray(v).reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            print("ERROR (SGS1D): length of 'v' is not valid")
            return None

    mean = np.asarray(mean).reshape(-1) # cast in 1-dimensional array if needed

    if mean.size != 1:
        if mean.size != nxy:
            print('ERROR (GRF2D): number of entry for "mean"...')
            return None
        mean = np.asarray(mean).reshape(ny, nx) # cast in 2-dimensional array of same shape as grid

    if var is not None:
        var = np.asarray(var).reshape(-1) # cast in 1-dimensional array if needed
        if var.size != 1:
            if var.size != nxy:
                print('ERROR (GRF2D): number of entry for "var"...')
                return None
            var = np.asarray(var).reshape(ny, nx) # cast in 2-dimensional array of same shape as grid

    srx, sry = search_radius

    srxi, sryi = [np.int(np.floor(r/d)) for r, d in zip(search_radius, spacing)]

    tmp = np.arange(1, srxi + 1) * dx
    search_mask_x = np.hstack((tmp[::-1], [0.], tmp))
    tmp = np.arange(1, sryi + 1) * dy
    search_mask_y = np.hstack((tmp[::-1], [0.], tmp))

    search_mask = np.sum(np.array(np.meshgrid(search_mask_y**2 / sry**2, search_mask_x / srx**2, indexing='ij')), axis=0) <= 1

    # Compute
    #    indc: node index of conditioning node (nearest node)
    indc = np.asarray(np.floor((x-origin)/spacing), dtype=int) # multiple-indices: size n x 2

    ixc, iyc = indc[:, 0], indc[:, 1]

    if sum(ixc < 0) or sum(ixc >= nx):
        print('ERROR (SGS2D): a conditioning point is out of the grid (x-direction)')
        return None
    if sum(iyc < 0) or sum(iyc >= ny):
        print('ERROR (SGS2D): a conditioning point is out of the grid (y-direction)')
        return None

    if len(np.unique(ixc + iyc*nx)) != len(x):
        print('ERROR (SGS2D): more than one conditioning point in a same grid cell')
        return None

    ixc = ixc + srxi
    iyc = iyc + sryi
    for ir in range(nreal):
        # Integrate conditioning data into grid
        sgs_cur = np.nan * np.ones((ny + 2*sryi, nx + 2*srxi))
        #sgs_cur[ixc, iyc] = v
        path = np.arange(nxy, dtype='int')
        np.random.shuffle(path)
        path_iy = path // nx + sryi
        path_ix = path % nx + srxi
        for ix, iy in zip(path_ix, path_iy):
            sgs_cur_sub = sgs_cur[iy-sryi:iy+sryi, ix-srxi:ix+srxi]

    print('done')
    return None
#     x
#
#     # Number of data points
#     n = x.shape[0]
#     # Number of unknown points
#     nu = xu.shape[0]
#
#
#     ind = np.arange(nu)
#
#     # Covariance function
#     cov_func = cov_model.func() # covariance function
#     if omni_dir:
#         # covariance model in 1D is used
#         cov0 = cov_func(0.) # covariance function at origin (lag=0)
#     else:
#         cov0 = cov_func(np.zeros(d)) # covariance function at origin (lag=0)
#
#     # Fill matrix of ordinary kriging system (matOK)
#     nOK = n+1 # order of the matrix
#     matOK = np.ones((nOK, nOK))
#     for i in range(n-1):
#         # lag between x[i] and x[j], j=i+1, ..., n-1
#         h = x[(i+1):] - x[i]
#         if omni_dir:
#             # compute norm of lag
#             h = np.sqrt(np.sum(h**2, axis=1))
#         cov_h = cov_func(h)
#         matOK[i, (i+1):-1] = cov_h
#         matOK[(i+1):-1, i] = cov_h
#         matOK[i,i] = cov0
#     matOK[-2,-2] = cov0
#     matOK[-1,-1] = 0.0
#
#     # Right hand side of the ordinary kriging system (b):
#     #   b is a matrix of dimension nOK x nu
#     b = np.ones((nOK, nu))
#     for i in range(n):
#         # lag between x[i] and every xu
#         h = xu - x[i]
#         if omni_dir:
#             # compute norm of lag
#             h = np.sqrt(np.sum(h**2, axis=1))
#         b[i,:] = cov_func(h)
#
#     # Solve the kriging system
#     w = np.linalg.solve(matOK,b) # w: matrix of dimension nOK x nu
#
#     # Kriged values at unknown points
#     vu = v.dot(w[:-1,:])
#
#     # Kriged standard deviation at unknown points
#     vu_std = np.sqrt(np.maximum(0, cov0 - np.array([np.dot(w[:,i], b[:,i]) for i in range(nu)])))
#
#     return (vu, vu_std)
# # ----------------------------------------------------------------------------
