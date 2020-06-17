#!/usr/bin/python3
#-*- coding: utf-8 -*-

"""
Python module:  'grf.py'
author:         Julien Straubhaar
date:           jan-2018

Module for gaussian random fields (GRF) simulations in 1D, 2D and 3D.
"""

import numpy as np

# ----------------------------------------------------------------------------
def grf1D(covFun, dimension, spacing, origin=0.,
          nreal=1, mean=0, var=None,
          x=None, v=None,
          extensionMin=None, crop=True,
          method=3, conditioningMethod=2,
          measureErrVar=0., tolInvKappa=1.e-10,
          printInfo=True):
    """
    Generates gaussian random fields (GRF) in 1D via FFT.

    The GRFs:
        - are generated using the covariance function (covFun),
        - have specified mean (mean) and variance (var), which can be non stationary
        - are conditioned to location x with value v
    Notes:
    1) For reproducing covariance model, the dimension of GRF should be large
       enough; let K an integer such that K*spacing is greater or equal to the
       correlation range, then
        - correlation accross opposite border should be removed by extending
          the domain sufficiently, i.e.
              extensionMin >= K - 1
        - two nodes could not be correlated simultaneously regarding both distances
          between them (with respect to the periodic grid), i.e. one should have
              dimension+extensionMin >= 2*K - 1,
          To sum up, extensionMin should be chosen such that
              dimension+extensionMin >= max(dimension, K) + K - 1
          i.e.
              extensionMin >= max(K-1,2*K-dimension-1)
    2) For large conditional simulations with large data set:
        - conditioningMethod should be set to 2 for using FFT in conditioning step
        - measureErrVar could be set to a small positive value to stabilize
          the covariance matrix for conditioning locations (solving linear system)

    :param covFun:      (function) covariance function f(h), where
                            h: (1-dimensional array or float) 1D-lag(s)
    :param dimension:   (int) nx, number of cells
    :param spacing:     (float) dx, spacing between two adjacent cells
    :param origin:      (float) ox, origin of the 1D field
                            - used for localizing the conditioning points
    :param nreal:       (int) number of realizations
    :param mean:        (float or ndarray) mean of the GRF:
                            - scalar for stationary mean
                            - ndarray for non stationary mean, must contain
                                nx values (reshaped if needed)
    :param var:         (float or ndarray or None) variance of the GRF,
                            if not None: variance of GRF is updated
                            depending on the specified variance and the covariance
                            function, otherwise: only the covariance function is
                            used
                                - scalar for stationary variance
                                - array for non stationary variance, must contain
                                    nx values (reshaped if needed)
    :param x:           (1-dimensional array or float or None) coordinate of
                            conditioning points (None for unconditional GRF)
    :param v:           (1-dimensional array or float or None) value at
                            conditioning points (same type as x)
    :param extensionMin: (int) minimal extension in nodes for embedding (see above)

    :param crop:        (bool) indicates if the extended generated field will
                            be cropped to original dimension; note that no cropping
                            is not valid with conditioning or non stationary mean
                            or variance
    :param method:      (int) indicates which method is used to generate
                            unconditional simulations; for each method the DFT "lam"
                            of the circulant embedding of the covariance matrix is
                            used, and periodic and stationary GRFs are generated;
                            possible values:
                                1: method A:
                                   generate one GRF Z as follows:
                                   - generate one real gaussian white noise W
                                   - apply fft (or fft inverse) on W to get X
                                   - multiply X by lam (term by term)
                                   - apply fft inverse (or fft) to get Z
                                2: method B:
                                   generate one GRF Z as follows:
                                   - generate directly X (of method A)
                                   - multiply X by lam (term by term)
                                   - apply fft inverse (or fft) to get Z
                                3: method C:
                                   generate two independent GRFs Z1, Z2 as follows:
                                   - generate two independant real gaussian white
                                     noises W1, W2 and set W = W1 + i * W2
                                   - apply fft (or fft inverse) on W to get X
                                   - multiply X by lam (term by term)
                                   - apply fft inverse (or fft) to get Z,
                                     and set Z1 = Re(Z), Z2 = Im(Z)
                                   note: if nreal is odd, the last field is
                                         generated using method A
    :param conditioningMethod:
                        (int) indicates which method is used to update simulation
                            for accounting conditioning data.
                            Let
                                A: index of conditioning nodes
                                B: index of non-conditioning nodes
                                Zobs: vector of values of the unconditional
                                      simulation Z at conditioning nodes
                            and
                                    +         +
                                    | rAA rAB |
                                r = |         |
                                    | rBA rBB |
                                    +         +
                            the covariance matrix, where index A (resp. B) refers
                            to conditioning (resp. non-conditioning) index in the
                            grid. Then, an unconditional simulation Z is updated
                            into a conditional simulation ZCond as follows:
                            Let
                                ZCond[A] = Zobs
                                ZCond[B] = Z[B] + rBA * rAA^(-1) * (Zobs - Z[A])
                            (that is the update consists in adding the kriging
                            estimates of the residues to the unconditional
                            simulation); possible values for conditioningMethod:
                                1: method CondtioningA:
                                   the matrix M = rBA * rAA^(-1) is explicitly
                                   computed (warning: could require large amount
                                   of memory), then all the simulations are updated
                                   by a sum and a multiplication by the matrix M
                                2: method ConditioningB:
                                   for each simulation: the linear system
                                        rAA * x = Zobs - Z[A]
                                   is solved and then, the multiplication by rBA
                                   is done via fft
    :param measureErrVar:
                        (float >=0) measurement error variance; we assume that
                            the error on conditioining data follows the distrubution
                            N(0,measureErrVar*I); i.e. rAA + measureErrVar*I is
                            considered instead of rAA for stabilizing the linear
                            system for this matrix.
                            (Ignored if x is None, i.e. unconditional simulations)
    :param tolInvKappa: (float >0) used only for conditioning, the simulation is
                            stopped if the inverse of the condition number of rAA
                            is above tolInvKappa
    :param printInfo:   (bool) indicates if some info is printed in stdout

    :return grf:    (2-dimensional array of dim nreal x n) nreal GRFs
                        with n = nx if crop = True, and n >= nx otherwise;
                        grf[i] is the i-th realization

    NOTES:
        Discrete Fourier Transform (DFT) of a vector x of length N is given by
            c = DFT(x) = F * x
        where F is the N x N matrix with coefficients
            F(j,k) = [exp(-i*2*pi*j*k/N)], 0 <= j,k <= N-1
        We have
            F^(-1) = 1/N * F^(*)
        where ^(*) denotes the conjugate transpose
        Let
            Q = 1/N^(1/2) * F
        Then Q is unitary, i.e. Q^(-1) = Q^(*)
        Then, we have
            DFT = F = N^(1/2) * Q
            DFT^(-1) = 1/N * F^(*) = 1/N^(1/2) * Q^(*)

        Using numpy package in python3, we have
            numpy.fft.fft() = DFT
            numpy.fft.ifft() = DFT^(-1)
    """

    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if printInfo:
            print('GRF1D: nreal = 0: nothing to do!')
        return()

    if printInfo:
        print('GRF1D: Preliminary computation...')

    #### Preliminary computation ####
    nx = dimension
    dx = spacing
    # ox = origin

    if method not in (1, 2, 3):
        print('ERROR (GRF1D): invalid method')
        return

    if x is not None:
        if conditioningMethod not in (1, 2):
            print('ERROR (GRF1D): invalid method for conditioning')
            return
        x = np.asarray(x).reshape(-1) # cast in 1-dimensional array if needed
        v = np.asarray(v).reshape(-1) # cast in 1-dimensional array if needed

    mean = np.asarray(mean).reshape(-1) # cast in 1-dimensional array if needed

    if mean.size not in (1, nx):
        print('ERROR (GRF1D): number of entry for "mean"...')
        return

    if var is not None:
        var = np.asarray(var).reshape(-1) # cast in 1-dimensional array if needed
        if var.size not in (1, nx):
            print('ERROR (GRF1D): number of entry for "var"...')
            return

    if not crop:
        if x is not None: # conditional simulation
            print('ERROR (GRF1D): "no crop" is not valid with conditional simulation')
            return

        if mean.size > 1:
            print('ERROR (GRF1D): "no crop" is not valid with non stationary mean')
            return

        if var is not None and var.size > 1:
            print('ERROR (GRF1D): "no crop" is not valid with non stationary variance')
            return

    if extensionMin is None:
        extensionMin = dimension - 1 # default

    Nmin = nx + extensionMin

    if printInfo:
        print('GRF1D: Computing circulant embedding...')

    # Circulant embedding of the covariance matrix
    # --------------------------------------------
    # The embedding matrix is a circulant matrix of size N x N, computed from
    # the covariance function.
    # To take a maximal benefit of Fast Fourier Transform (FFT) for computing DFT,
    # we choose
    #    N = 2^g (a power of 2), with N >= Nmin
    g = int(np.ceil(np.log2(Nmin)))
    N = int(2**g)

    if printInfo:
        print('GRF1D: Embedding dimension: {}'.format(N))

    # ccirc: coefficient of the embedding matrix (first line), vector of size N
    L = int (N/2)
    h = np.arange(-L, L, dtype=float) * dx # [-L ... 0 ... L-1] * dx
    ccirc = covFun(h)

    del(h)

    # ...shift first L index to the end of the axis, i.e.:
    #    [-L ... 0 ... L-1] -> [0 ... L-1 -L ... -1]
    ind = np.arange(L)
    ccirc = ccirc[np.hstack((ind+L, ind))]

    del(ind)

    if printInfo:
        print('GRF1D: Computing FFT of circulant matrix...')

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
        varUpdate = np.sqrt(var/covFun(0.))

    # Dealing with conditioning
    # -------------------------
    if x is not None:
        if printInfo:
            print('GRF1D: Treatment of conditioning data...')
        # Compute the part rAA of the covariance matrix
        #        +         +
        #        | rAA rAB |
        #    r = |         |
        #        | rBA rBB |
        #        +         +
        # where index A (resp. B) refers to
        # conditioning (resp. non-conditioning) index in the grid.

        if printInfo:
            print('GRF1D: Computing covariance matrix (rAA) for conditioning locations...')

        # Compute
        #    indc: node index of conditioning node (nearest node)
        indc = np.asarray(np.floor((x-origin)/spacing), dtype=int)
        if sum(indc < 0) > 0 or sum(indc >= nx):
            print('ERROR (GRF1D): a conditioning point is out of the grid')
            return

        if len(np.unique(indc)) != len(x):
            print('ERROR (GRF1D): more than one conditioning point in a same grid cell')

        nc = len(x)

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
            print('ERROR (GRF1D): conditioning issue: condition number of matrix rAA is too big')
            return

        # Compute:
        #    indnc: node index of non-conditioning node (nearest node)
        indnc = np.asarray(np.setdiff1d(np.arange(nx), indc), dtype=int)
        nnc = len(indnc)

        if conditioningMethod == 1:
            # Method ConditioningA
            # --------------------
            if printInfo:
                print('GRF1D: Computing covariance matrix (rBA) for non-conditioning / conditioning locations...')

            # Compute the parts rBA of the covariance matrix (see above)
            # rBA
            rBA = np.zeros((nnc, nc))
            for j in range(nc):
                k = np.mod(indc[j] - indnc, N)
                rBA[:,j] = ccirc[k]

            if printInfo:
                print('GRF1D: Computing rBA * rAA^(-1)...')

            # compute rBA * rAA^(-1)
            rBArAAinv = np.dot(rBA, np.linalg.inv(rAA))

            del(rAA, rBA)

            # If a variance var is specified, then the matrix r should be updated
            # by the following operation:
            #    diag((var/covFun(0))^1/2) * r * diag((var/covFun(0))^1/2)
            # Hence, if a non stationary variance is specified,
            # the matrix rBA * rAA^(-1) should be consequently updated
            # by multiplying its columns by 1/varUpdate[indc] and its rows by varUpdate[indnc]
            if var is not None and var.size > 1:
                rBArAAinv = np.transpose(varUpdate[indnc] * np.transpose(1./varUpdate[indc] * rBArAAinv))

        elif conditioningMethod == 2:
            # Method ConditioningB
            # --------------------
            if printInfo:
                print('GRF1D: Computing index in the embedding grid for non-conditioning / conditioning locations...')

            # Compute index in the embedding grid for indc and indnc
            # (to allow use of fft)
            indcEmb = indc
            indncEmb = indnc

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
            if printInfo:
                print('GRF1D: Unconditional simulation {:4d} of {:4d}...'.format(i+1, nreal))

            W = np.random.normal(size=N)

            Z = np.fft.ifft(lamSqrt * np.fft.fft(W))
            # ...note that Im(Z) = 0
            grf[i] = np.real(Z[0:grfNx])

    elif method == 2:
        # Method B
        # --------
        for i in range(nreal):
            if printInfo:
                print('GRF1D: Unconditional simulation {:4d} of {:4d}...'.format(i+1, nreal))

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
        if nreal > 1:
            for i in np.arange(0,nreal,2):
                if printInfo:
                    print('GRF1D: Unconditional simulation {:4d}-{:4d} of {:4d}...'.format(i+1, i+2, nreal))

                W = np.array(np.random.normal(size=N), dtype=complex)
                W.imag = np.random.normal(size=N)
                Z = np.sqrt(N) * np.fft.ifft(lamSqrt * W)
                #  Z = 1/sqrt(N) * np.fft.fft(lamSqrt * W)] # see above: [OR:...]

                grf[i] = np.real(Z[0:grfNx])
                grf[i+1] = np.imag(Z[0:grfNx])

        if np.mod(nreal, 2) == 1:
            if printInfo:
                print('GRF1D: Unconditional simulation {:4d} of {:4d}...'.format(nreal, nreal))

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
            if printInfo:
                print('GRF1D: Updating conditional simulations...')

            # Update all simulations at a time,
            # use the matrix rBA * rAA^(-1) already computed
            grf[:,indnc] = grf[:,indnc] + np.transpose(np.dot(rBArAAinv, np.transpose(v - grf[:,indc])))
            grf[:,indc] = v

        elif conditioningMethod == 2:
            # Method ConditioningB
            # --------------------
            # Update each simulation successively as follows:
            #    - solve rAA * x = Zobs - z[A]
            #    - do the multiplication rBA * x via the circulant embedding of the
            #      covariance matrix (using fft)
            rAAinvResiduEmb = np.zeros(N)

            for i in range(nreal):
                if printInfo:
                    print('GRF1D: Updating conditional simulation {:4d} of {:4d}...'.format(i+1, nreal))

                # Compute residue
                residu = v - grf[i,indc]
                # ... update if non stationary variance is specified
                if var is not None and var.size > 1:
                    residu = 1./varUpdate[indc] * residu

                # Compute
                #    x = rAA^(-1) * residu, and then
                #    Z = rBA * x via the circulant embedding of the covariance matrix
                rAAinvResiduEmb[indcEmb] = np.linalg.solve(rAA, residu)
                Z = np.fft.ifft(lam * np.fft.fft(rAAinvResiduEmb))
                # ...note that Im(Z) = 0
                Z = np.real(Z[indncEmb])

                # ... update if non stationary covariance is specified
                if var is not None and var.size > 1:
                    Z = varUpdate[indnc] * Z

                grf[i, indnc] = grf[i, indnc] + Z
                grf[i, indc] = v

    return (grf)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def krige1D(x, v, covFun, dimension, spacing, origin=0.,
            mean=0, var=None,
            extensionMin=None,
            conditioningMethod=2,
            measureErrVar=0., tolInvKappa=1.e-10,
            computeKrigSD=True,
            printInfo=True):
    """
    Computes kriging estimates and standard deviation in 1D via FFT.

    It is a simple kriging
        - of value v at location x,
        - based on the covariance function (covFun),
        - with a specified mean (mean) and variance (var), which can be non stationary

    Notes:
    1) For reproducing covariance model, the dimension of the field/domain should be large
       enough; let K an integer such that K*spacing is greater or equal to the
       correlation range, then
        - correlation accross opposite border should be removed by extending
          the domain sufficiently, i.e.
              extensionMin >= K - 1
        - two nodes could not be correlated simultaneously regarding both distances
          between them (with respect to the periodic grid), i.e. one should have
              dimension+extensionMin >= 2*K - 1,
          To sum up, extensionMin should be chosen such that
              dimension+extensionMin >= max(dimension, K) + K - 1
          i.e.
              extensionMin >= max(K-1,2*K-dimension-1)
    2) For large data set:
        - conditioningMethod should be set to 2 for using FFT
        - measureErrVar could be set to a small positive value to stabilize
          the covariance matrix (solving linear system)

    :param x:           (1-dimensional array of float) coordinate of data points
    :param v:           (1-dimensional array of float) value at data points
    :param covFun:      (function) covariance function f(h), where
                            h: (1-dimensional array or float) 1D-lag(s)
    :param dimension:   (int) nx, number of cells
    :param spacing:     (float) dx, spacing between two adjacent cells
    :param origin:      (float) ox, origin of the 1D field
                            - used for localizing the conditioning points
    :param mean:        (float or ndarray) mean of the variable:
                            - scalar for stationary mean
                            - ndarray for non stationary mean, must contain
                                nx values (reshaped if needed)
    :param var:         (float or ndarray or None) variance of the variable,
                            if not None: variance in the field is updated
                            depending on the specified variance and the covariance
                            function, otherwise: only the covariance function is
                            used
                                - scalar for stationary variance
                                - array for non stationary variance, must contain
                                    nx values (reshaped if needed)
    :param extensionMin: (int) minimal extension in nodes for embedding (see above)
    :param conditioningMethod:
                        (int) indicates which method is used to perform kriging.
                            Let
                                A: index of conditioning (data) nodes
                                B: index of non-conditioning nodes
                            and
                                    +         +
                                    | rAA rAB |
                                r = |         |
                                    | rBA rBB |
                                    +         +
                            the covariance matrix, where index A (resp. B) refers
                            to conditioning (resp. non-conditioning) index in the
                            grid. Then, thre kriging estimates and variance are
                                krig[B]    = mean + rBA * rAA^(-1) * (v - mean)
                                krigVar[B] = diag(rBB - rBA * rAA^(-1) * rAB)
                            The computation is done in a way depending on the
                            following possible values for conditioningMethod:
                                1: method CondtioningA:
                                   the matrices rBA, RAA^(-1) are explicitly
                                   computed (warning: could require large amount
                                   of memory), then all the simulations are updated
                                   by a sum and a multiplication by the matrix M
                                2: method ConditioningB:
                                   for kriging estimates:
                                       the linear system
                                         rAA * y = (v - mean)
                                       is solved, and then
                                         mean + rBA*y
                                       is computed
                                   for kriging variances:
                                       for each column u[j] of rAB, the linear
                                       system
                                         rAA * y = u[j]
                                       is solved, and then
                                         rBB[j,j] - y^t*y
                                       is computed
    :param measureErrVar:
                        (float >=0) measurement error variance; we assume that
                            the error on conditioining data follows the distrubution
                            N(0,measureErrVar*I); i.e. rAA + measureErrVar*I is
                            considered instead of rAA for stabilizing the linear
                            system for this matrix.
    :param tolInvKappa: (float >0) the function is stopped if the inverse of
                            the condition number of rAA is above tolInvKappa
    :param computeKrigSD:
                        (bool) indicates if the standard deviation of kriging is computed
    :param printInfo:   (bool) indicates if some info is printed in stdout

    :return ret:        two possible cases:
                            ret = [krig, krigSD] if computeKrigSD is equal to True
                            ret = krig           if computeKrigSD is equal to False
                        where
                            krig:   (1-dimensional array of dim nx)
                                        kriging estimates
                            krigSD: (1-dimensional array of dim nx)
                                        kriging standard deviation

    NOTES:
        Discrete Fourier Transform (DFT) of a vector x of length N is given by
            c = DFT(x) = F * x
        where F is the N x N matrix with coefficients
            F(j,k) = [exp(-i*2*pi*j*k/N)], 0 <= j,k <= N-1
        We have
            F^(-1) = 1/N * F^(*)
        where ^(*) denotes the conjugate transpose
        Let
            Q = 1/N^(1/2) * F
        Then Q is unitary, i.e. Q^(-1) = Q^(*)
        Then, we have
            DFT = F = N^(1/2) * Q
            DFT^(-1) = 1/N * F^(*) = 1/N^(1/2) * Q^(*)

        Using numpy package in python3, we have
            numpy.fft.fft() = DFT
            numpy.fft.ifft() = DFT^(-1)
    """

    if conditioningMethod not in (1, 2):
        print('ERROR (KRIGE1D): invalid method!')
        return

    nx = dimension
    dx = spacing
    # ox = origin

    x = np.asarray(x).reshape(-1) # cast in 1-dimensional array if needed
    v = np.asarray(v).reshape(-1) # cast in 1-dimensional array if needed

    mean = np.asarray(mean).reshape(-1) # cast in 1-dimensional array if needed

    if mean.size not in (1, nx):
        print('ERROR (KRIGE1D): number of entry for "mean"...')
        return

    if var is not None:
        var = np.asarray(var).reshape(-1) # cast in 1-dimensional array if needed
        if var.size not in (1, nx):
            print('ERROR (KRIGE1D): number of entry for "var"...')
            return

    if extensionMin is None:
        extensionMin = dimension - 1 # default

    Nmin = nx + extensionMin

    if printInfo:
        print('KRIGE1D: Computing circulant embedding...')

    # Circulant embedding of the covariance matrix
    # --------------------------------------------
    # The embedding matrix is a circulant matrix of size N x N, computed from
    # the covariance function.
    # To take a maximal benefit of Fast Fourier Transform (FFT) for computing DFT,
    # we choose
    #    N = 2^g (a power of 2), with N >= Nmin
    g = int(np.ceil(np.log2(Nmin)))
    N = int(2**g)

    if printInfo:
        print('KRIGE1D: Embedding dimension: {}'.format(N))

    # ccirc: coefficient of the embedding matrix (first line), vector of size N
    L = int (N/2)
    h = np.arange(-L, L, dtype=float) * dx # [-L ... 0 ... L-1] * dx
    ccirc = covFun(h)

    del(h)

    # ...shift first L index to the end of the axis, i.e.:
    #    [-L ... 0 ... L-1] -> [0 ... L-1 -L ... -1]
    ind = np.arange(L)
    ccirc = ccirc[np.hstack((ind+L, ind))]

    del(ind)

    if printInfo:
        print('KRIGE1D: Computing FFT of circulant matrix...')

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
        varUpdate = np.sqrt(var/covFun(0.))

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
    #    diag((var/covFun(0))^1/2) * r * diag((var/covFun(0))^1/2)
    # which is accounting in the computation of kriging estimates and standard
    # deviation below

    if printInfo:
        print('KRIGE1D: Computing covariance matrix (rAA) for conditioning locations...')

    # Compute
    #    indc: node index of conditioning node (nearest node)
    indc = np.asarray(np.floor((x-origin)/spacing), dtype=int)
    if sum(indc < 0) > 0 or sum(indc >= nx):
        print('ERROR (KRIGE1D): a conditioning point is out of the grid')
        return

    if len(np.unique(indc)) != len(x):
        print('ERROR (KRIGE1D): more than one conditioning point in a same grid cell')

    nc = len(x)

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
        print('ERROR (KRIGE1D): conditioning issue: condition number of matrix rAA is too big')
        return

    # Compute:
    #    indnc: node index of non-conditioning node (nearest node)
    indnc = np.asarray(np.setdiff1d(np.arange(nx), indc), dtype=int)
    nnc = len(indnc)

    # Initialize
    krig = np.zeros(nx)
    if computeKrigSD:
        krigSD = np.zeros(nx)

    if mean.size == 1:
        v = v - mean
    else:
        v = v - mean[indc]

    if var is not None and var.size > 1:
        v = 1./varUpdate[indc] * v

    if conditioningMethod == 1:
        # Method ConditioningA
        # --------------------
        if printInfo:
            print('KRIGE1D: Computing covariance matrix (rBA) for non-conditioning / conditioning locations...')

        # Compute the parts rBA of the covariance matrix (see above)
        # rBA
        rBA = np.zeros((nnc, nc))
        for j in range(nc):
            k = np.mod(indc[j] - indnc, N)
            rBA[:,j] = ccirc[k]

        del(ccirc)

        if printInfo:
            print('KRIGE1D: Computing rBA * rAA^(-1)...')

        # compute rBA * rAA^(-1)
        rBArAAinv = np.dot(rBA, np.linalg.inv(rAA))

        del(rAA)
        if not computeKrigSD:
            del(rBA)

        # Compute kriging estimates
        if printInfo:
            print('KRIGE1D: computing kriging estimates...')

        krig[indnc] = np.dot(rBArAAinv, v)
        krig[indc] = v

        if computeKrigSD:
            # Compute kriging standard deviation
            if printInfo:
                print('KRIGE1D: computing kriging standard deviation ...')

            krigSD[indnc] = np.sqrt(diagEntry - np.diag(np.dot(rBArAAinv, np.transpose(rBA))))

            del(rBA)

    elif conditioningMethod == 2:
        # Method ConditioningB
        # --------------------
        if not computeKrigSD:
            del(ccirc)

        if printInfo:
            print('KRIGE1D: Computing index in the embedding grid for non-conditioning / conditioning locations...')

        # Compute index in the embedding grid for indc and indnc
        # (to allow use of fft)
        indcEmb = indc
        indncEmb = indnc

        # Compute kriging estimates
        if printInfo:
            print('KRIGE1D: computing kriging estimates...')

        # Compute
        #    u = rAA^(-1) * v, and then
        #    Z = rBA * u via the circulant embedding of the covariance matrix
        uEmb = np.zeros(N)
        uEmb[indcEmb] = np.linalg.solve(rAA, v)
        Z = np.fft.ifft(lam * np.fft.fft(uEmb))
        # ...note that Im(Z) = 0
        krig[indnc] = np.real(Z[indncEmb])
        krig[indc] = v

        if computeKrigSD:
            # Compute kriging standard deviation
            if printInfo:
                print('KRIGE1D: computing kriging standard deviation ...')

            for j in range(nnc):
                u = ccirc[np.mod(indc - indnc[j], N)] # j-th row of rBA
                krigSD[indnc[j]] = np.dot(u,np.linalg.solve(rAA, u))

            del(ccirc)

            krigSD[indnc] = np.sqrt(diagEntry - krigSD[indnc])

    # ... update if non stationary covariance is specified
    if var is not None:
        if var.size > 1:
            krig = varUpdate * krig
        if computeKrigSD:
            krigSD = varUpdate * krigSD

    krig = krig + mean

    if computeKrigSD:
        return ([krig, krigSD])
    else:
        return (krig)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def grf2D(covFun, dimension, spacing, origin=[0., 0.],
          nreal=1, mean=0, var=None,
          x=None, v=None,
          extensionMin=None, crop=True,
          method=3, conditioningMethod=2,
          measureErrVar=0., tolInvKappa=1.e-10,
          printInfo=True):
    """
    Generates gaussian random fields (GRF) in 2D via FFT.

    The GRFs:
        - are generated using the covariance function (covFun),
        - have specified mean (mean) and variance (var), which can be non stationary
        - are conditioned to location x with value v
    Notes:
    1) For reproducing covariance model, the dimension of GRF should be large
       enough; let K an integer such that K*spacing is greater or equal to the
       correlation range, then
        - correlation accross opposite border should be removed by extending
          the domain sufficiently, i.e.
              extensionMin >= K - 1
        - two nodes could not be correlated simultaneously regarding both distances
          between them (with respect to the periodic grid), i.e. one should have
          i.e. one should have
              dimension+extensionMin >= 2*K - 1,
          To sum up, extensionMin should be chosen such that
              dimension+extensionMin >= max(dimension, K) + K - 1
          i.e.
              extensionMin >= max(K-1,2*K-dimension-1)
    2) For large conditional simulations with large data set:
        - conditioningMethod should be set to 2 for using FFT in conditioning step
        - measureErrVar could be set to a small positive value to stabilize
          the covariance matrix for conditioning locations (solving linear system)

    :param covFun:      (function) covariance function f(h), where
                            h:  (2-dimensional array of dim n x 2, or
                                1-dimensional array of dim 2) 2D-lag(s)
    :param dimension:   (sequence of 2 int) [nx, ny], number of cells
                            in x-, y-axis direction
    :param spacing:     (sequence of 2 float) [dx, dy], spacing between
                            two adjacent cells in x-, y-axis direction
    :param origin:      (sequence of 2 float) [ox, oy], origin of the 2D field
                            - used for localizing the conditioning points
    :param nreal:       (int) number of realizations
    :param mean:        (float or ndarray) mean of the GRF:
                            - scalar for stationary mean
                            - ndarray for non stationary mean, must contain
                                nx*ny values (reshaped if needed)
    :param var:         (float or ndarray or None) variance of the GRF,
                            if not None: variance of GRF is updated
                            depending on the specified variance and the covariance
                            function, otherwise: only the covariance function is
                            used
                                - scalar for stationary variance
                                - array for non stationary variance, must contain
                                    nx*ny values (reshaped if needed)
    :param x:           (2-dimensional array of dim n x 2, or
                            1-dimensional array of dim 2 or None) coordinate of
                            conditioning points (None for unconditional GRF)
    :param v:           (1-dimensional array or float or None) value at
                            conditioning points (length n)
    :param extensionMin: (sequence of 2 int) minimal extension in nodes in
                            x-, y-axis direction for embedding (see above)
    :param crop:        (bool) indicates if the extended generated field will
                            be cropped to original dimension; note that no cropping
                            is not valid with conditioning or non stationary mean
                            or variance
    :param method:      (int) indicates which method is used to generate
                            unconditional simulations; for each method the DFT "lam"
                            of the circulant embedding of the covariance matrix is
                            used, and periodic and stationary GRFs are generated;
                            possible values:
                                1: method A:
                                   generate one GRF Z as follows:
                                   - generate one real gaussian white noise W
                                   - apply fft (or fft inverse) on W to get X
                                   - multiply X by lam (term by term)
                                   - apply fft inverse (or fft) to get Z
                                2: method B: NOT IMPLEMENTED!!!
                                   generate one GRF Z as follows:
                                   - generate directly X (of method A)
                                   - multiply X by lam (term by term)
                                   - apply fft inverse (or fft) to get Z
                                3: method C:
                                   generate two independent GRFs Z1, Z2 as follows:
                                   - generate two independant real gaussian white
                                     noises W1, W2 and set W = W1 + i * W2
                                   - apply fft (or fft inverse) on W to get X
                                   - multiply X by lam (term by term)
                                   - apply fft inverse (or fft) to get Z,
                                     and set Z1 = Re(Z), Z2 = Im(Z)
                                   note: if nreal is odd, the last field is
                                         generated using method A
    :param conditioningMethod:
                        (int) indicates which method is used to update simulation
                            for accounting conditioning data.
                            Let
                                A: index of conditioning nodes
                                B: index of non-conditioning nodes
                                Zobs: vector of values of the unconditional
                                      simulation Z at conditioning nodes
                            and
                                    +         +
                                    | rAA rAB |
                                r = |         |
                                    | rBA rBB |
                                    +         +
                            the covariance matrix, where index A (resp. B) refers
                            to conditioning (resp. non-conditioning) index in the
                            grid. Then, an unconditional simulation Z is updated
                            into a conditional simulation ZCond as follows:
                            Let
                                ZCond[A] = Zobs
                                ZCond[B] = Z[B] + rBA * rAA^(-1) * (Zobs - Z[A])
                            (that is the update consists in adding the kriging
                            estimates of the residues to the unconditional
                            simulation); possible values for conditioningMethod:
                                1: method CondtioningA:
                                   the matrix M = rBA * rAA^(-1) is explicitly
                                   computed (warning: could require large amount
                                   of memory), then all the simulations are updated
                                   by a sum and a multiplication by the matrix M
                                2: method ConditioningB:
                                   for each simulation: the linear system
                                        rAA * x = Zobs - Z[A]
                                   is solved and then, the multiplication by rBA
                                   is done via fft
    :param measureErrVar:
                        (float >=0) measurement error variance; we assume that
                            the error on conditioining data follows the distrubution
                            N(0,measureErrVar*I); i.e. rAA + measureErrVar*I is
                            considered instead of rAA for stabilizing the linear
                            system for this matrix.
                            (Ignored if x is None, i.e. unconditional simulations)
    :param tolInvKappa: (float >0) used only for conditioning, the simulation is
                            stopped if the inverse of the condition number of rAA
                            is above tolInvKappa
    :param printInfo:   (bool) indicates if some info is printed in stdout

    :return grf:    (3-dimensional array of dim nreal x n2 x n1) nreal GRFs
                        with n1 = nx, n2 = ny if crop = True,
                        and n1 >= nx, n2 >= ny otherwise;
                        grf[i] is the i-th realization

    NOTES:
        Discrete Fourier Transform (DFT) of an array x of dim N1 x N2 is given by
            c = DFT(x) = F * x
        where F is the the (N1*N2) x (N1*N2) matrix with coefficients
            F(j,k) = [exp( -i*2*pi*(j^t*k)/(N1*N2) )], j=(j1,j2), k=(k1,k2) in G,
        and
            G = {n=(n1,n2), 0 <= n1 <= N1-1, 0 <= n2 <= N2-1}
        denotes the indices grid
        and where we use the bijection
            (n1,n2) in G -> n1 + n2 * N1 in {0,...,N1*N2-1},
        between the multiple-indices and the single indices

        With N = N1*N2, we have
            F^(-1) = 1/N * F^(*)
        where ^(*) denotes the conjugate transpose
        Let
            Q = 1/N^(1/2) * F
        Then Q is unitary, i.e. Q^(-1) = Q^(*)
        Then, we have
            DFT = F = N^(1/2) * Q
            DFT^(-1) = 1/N * F^(*) = 1/N^(1/2) * Q^(*)

        Using numpy package in python3, we have
            numpy.fft.fft2() = DFT
            numpy.fft.ifft2() = DFT^(-1)
    """

    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if printInfo:
            print('GRF2D: nreal = 0: nothing to do!')
        return()

    if printInfo:
        print('GRF2D: Preliminary computation...')

    #### Preliminary computation ####
    nx, ny = dimension
    dx, dy = spacing
    # ox, oy = origin

    nxy = nx*ny

    if method not in (1, 2, 3):
        print('ERROR (GRF2D): invalid method')
        return

    if method == 2:
        print('ERROR (GRF2D): Unconditional simulation: "method=2" not implemented...')
        return

    if x is not None:
        if conditioningMethod not in (1, 2):
            print('ERROR (GRF2D): invalid method for conditioning')
            return
        x = np.asarray(x).reshape(-1,2) # cast in 1-dimensional array if needed
        v = np.asarray(v).reshape(-1) # cast in 1-dimensional array if needed

    mean = np.asarray(mean).reshape(-1) # cast in 1-dimensional array if needed

    if mean.size != 1:
        if mean.size != nxy:
            print('ERROR (GRF2D): number of entry for "mean"...')
            return
        mean = np.asarray(mean).reshape(ny, nx) # cast in 2-dimensional array of same shape as grid

    if var is not None:
        var = np.asarray(var).reshape(-1) # cast in 1-dimensional array if needed
        if var.size != 1:
            if var.size != nxy:
                print('ERROR (GRF2D): number of entry for "var"...')
                return
            var = np.asarray(var).reshape(ny, nx) # cast in 2-dimensional array of same shape as grid

    if not crop:
        if x is not None: # conditional simulation
            print('ERROR (GRF2D): "no crop" is not valid with conditional simulation')
            return

        if mean.size > 1:
            print('ERROR (GRF2D): "no crop" is not valid with non stationary mean')
            return

        if var is not None and var.size > 1:
            print('ERROR (GRF2D): "no crop" is not valid with non stationary variance')
            return

    if extensionMin is None:
        extensionMin = [nx-1, ny-1] # default

    N1min = nx + extensionMin[0]
    N2min = ny + extensionMin[1]

    if printInfo:
        print('GRF2D: Computing circulant embedding...')

    # Circulant embedding of the covariance matrix
    # --------------------------------------------
    # The embedding matrix is a (N1,N2)-nested block circulant matrix, computed from
    # the covariance function.
    # To take a maximal benefit of Fast Fourier Transform (FFT) for computing DFT,
    # we choose
    #     N1 = 2^g1 (a power of 2), with N1 >= N1min
    #     N2 = 2^g2 (a power of 2), with N2 >= N2min
    g1 = int(np.ceil(np.log2(N1min)))
    g2 = int(np.ceil(np.log2(N2min)))
    N1 = int(2**g1)
    N2 = int(2**g2)

    if printInfo:
        print('GRF2D: Embedding dimension: {} x {}'.format(N1, N2))

    N = N1*N2

    # ccirc: coefficient of the embedding matrix (N2, N1) array
    L1 = int (N1/2)
    L2 = int (N2/2)
    h1 = np.arange(-L1, L1, dtype=float) * dx # [-L1 ... 0 ... L1-1] * dx
    h2 = np.arange(-L2, L2, dtype=float) * dy # [-L2 ... 0 ... L2-1] * dy

    hh = np.meshgrid(h1, h2)
    ccirc = covFun(np.hstack((hh[0].reshape(-1,1), hh[1].reshape(-1,1))))
    ccirc.resize(N2, N1)

    del(h1, h2, hh)

    # ...shift first L1 index to the end of the axis 1:
    ind = np.arange(L1)
    ccirc = ccirc[:, np.hstack((ind+L1, ind))]
    # ...shift first L2 index to the end of the axis 0:
    ind = np.arange(L2)
    ccirc = ccirc[np.hstack((ind+L2, ind)), :]

    del(ind)

    if printInfo:
        print('GRF2D: Computing FFT of circulant matrix...')

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
        varUpdate = np.sqrt(var/covFun(np.zeros(2)))

    # Dealing with conditioning
    # -------------------------
    if x is not None:
        if printInfo:
            print('GRF2D: Treatment of conditioning data...')
        # Compute the part rAA of the covariance matrix
        #        +         +
        #        | rAA rAB |
        #    r = |         |
        #        | rBA rBB |
        #        +         +
        # where index A (resp. B) refers to
        # conditioning (resp. non-conditioning) index in the grid.

        if printInfo:
            print('GRF2D: Computing covariance matrix (rAA) for conditioning locations...')

        # Compute
        #    indc: node index of conditioning node (nearest node)
        indc = np.asarray(np.floor((x-origin)/spacing), dtype=int) # multiple-indices: size n x 2

        ix, iy = indc[:, 0], indc[:, 1]

        if sum(ix < 0) > 0 or sum(ix >= nx):
            print('ERROR (GRF2D): a conditioning point is out of the grid (x-direction)')
            return
        if sum(iy < 0) > 0 or sum(iy >= ny):
            print('ERROR (GRF2D): a conditioning point is out of the grid (y-direction)')
            return

        indc = ix + iy * nx # single-indices

        if len(np.unique(indc)) != len(x):
            print('ERROR (GRF2D): more than one conditioning point in a same grid cell')

        nc = len(x)

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
            print('ERROR (GRF2D): conditioning issue: condition number of matrix rAA is too big')
            return

        # Compute:
        #    indnc: node index of non-conditioning node (nearest node)
        indnc = np.asarray(np.setdiff1d(np.arange(nxy), indc), dtype=int)
        nnc = len(indnc)

        ky = np.floor_divide(indnc, nx)
        kx = np.mod(indnc, nx)

        if conditioningMethod == 1:
            # Method ConditioningA
            # --------------------
            if printInfo:
                print('GRF2D: Computing covariance matrix (rBA) for non-conditioning / conditioning locations...')

            # Compute the parts rBA of the covariance matrix (see above)
            # rBA
            rBA = np.zeros((nnc, nc))
            for j in range(nc):
                rBA[:,j] = ccirc[np.mod(iy[j] - ky, N2), np.mod(ix[j] - kx, N1)]

            if printInfo:
                print('GRF2D: Computing rBA * rAA^(-1)...')

            # compute rBA * rAA^(-1)
            rBArAAinv = np.dot(rBA, np.linalg.inv(rAA))

            del(rAA, rBA)

            # If a variance var is specified, then the matrix r should be updated
            # by the following operation:
            #    diag((var/covFun(0))^1/2) * r * diag((var/covFun(0))^1/2)
            # Hence, if a non stationary variance is specified,
            # the matrix rBA * rAA^(-1) should be consequently updated
            # by multiplying its columns by 1/varUpdate[indc] and its rows by varUpdate[indnc]
            if var is not None and var.size > 1:
                rBArAAinv = np.transpose(varUpdate.reshape(-1)[indnc] * np.transpose(1./varUpdate.reshape(-1)[indc] * rBArAAinv))

        elif conditioningMethod == 2:
            # Method ConditioningB
            # --------------------
            if printInfo:
                print('GRF2D: Computing index in the embedding grid for non-conditioning / conditioning locations...')

            # Compute index in the embedding grid for indc and indnc
            # (to allow use of fft)
            indcEmb =  iy * N1 + ix
            indncEmb = ky * N1 + kx

        del(ix, iy, kx, ky)

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
            if printInfo:
                print('GRF2D: Unconditional simulation {:4d} of {:4d}...'.format(i+1, nreal))

            W = np.random.normal(size=(N2, N1))

            Z = np.fft.ifft2(lamSqrt * np.fft.fft2(W))
            # ...note that Im(Z) = 0
            grf[i] = np.real(Z[0:grfNy, 0:grfNx])

    elif method == 2:
        # Method B
        # --------
        print('ERROR (GRF2D): Unconditional simulation: "method=2" not implemented...')
        return

    elif method == 3:
        # Method C
        # --------
        if nreal > 1:
            for i in np.arange(0,nreal,2):
                if printInfo:
                    print('GRF2D: Unconditional simulation {:4d}-{:4d} of {:4d}...'.format(i+1, i+2, nreal))

                W = np.array(np.random.normal(size=(N2, N1)), dtype=complex)
                W.imag = np.random.normal(size=(N2, N1))
                Z = np.sqrt(N) * np.fft.ifft2(lamSqrt * W)
                #  Z = 1/np.sqrt(N) * np.fft.fft2(lamSqrt * W)] # see above: [OR:...]

                grf[i] = np.real(Z[0:grfNy, 0:grfNx])
                grf[i+1] = np.imag(Z[0:grfNy, 0:grfNx])

        if np.mod(nreal, 2) == 1:
            if printInfo:
                print('GRF2D: Unconditional simulation {:4d} of {:4d}...'.format(nreal, nreal))

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
            if printInfo:
                print('GRF2D: Updating conditional simulations...')

            # Update all simulations at a time,
            # use the matrix rBA * rAA^(-1) already computed
            grf[:,indnc] = grf[:,indnc] + np.transpose(np.dot(rBArAAinv, np.transpose(v - grf[:,indc])))
            grf[:,indc] = v

        elif conditioningMethod == 2:
            # Method ConditioningB
            # --------------------
            # Update each simulation successively as follows:
            #    - solve rAA * x = Zobs - z[A]
            #    - do the multiplication rBA * x via the circulant embedding of the
            #      covariance matrix (using fft)
            rAAinvResiduEmb = np.zeros(N2*N1)

            for i in range(nreal):
                if printInfo:
                    print('GRF2D: Updating conditional simulation {:4d} of {:4d}...'.format(i+1, nreal))

                # Compute residue
                residu = v - grf[i,indc]
                # ... update if non stationary variance is specified
                if var is not None and var.size > 1:
                    residu = 1./varUpdate.reshape(-1)[indc] * residu

                # Compute
                #    x = rAA^(-1) * residu, and then
                #    Z = rBA * x via the circulant embedding of the covariance matrix
                rAAinvResiduEmb[indcEmb] = np.linalg.solve(rAA, residu)
                Z = np.fft.ifft2(lam * np.fft.fft2(rAAinvResiduEmb.reshape(N2, N1)))
                # ...note that Im(Z) = 0
                Z = np.real(Z.reshape(-1)[indncEmb])

                # ... update if non stationary covariance is specified
                if var is not None and var.size > 1:
                    Z = varUpdate.reshape(-1)[indnc] * Z

                grf[i, indnc] = grf[i, indnc] + Z
                grf[i, indc] = v

        # Reshape grf as initially
        grf.resize(nreal, grfNy, grfNx)

    return (grf)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def krige2D(x, v, covFun, dimension, spacing, origin=[0., 0.],
            mean=0, var=None,
            extensionMin=None,
            conditioningMethod=2,
            measureErrVar=0., tolInvKappa=1.e-10,
            computeKrigSD=True,
            printInfo=True):
    """
    Computes kriging estimates and standard deviation in 2D via FFT.

    It is a simple kriging
        - of value v at location x,
        - based on the covariance function (covFun),
        - with a specified mean (mean) and variance (var), which can be non stationary
    Notes:
    1) For reproducing covariance model, the dimension of field/domain should be large
       enough; let K an integer such that K*spacing is greater or equal to the
       correlation range, then
        - correlation accross opposite border should be removed by extending
          the domain sufficiently, i.e.
              extensionMin >= K - 1
        - two nodes could not be correlated simultaneously regarding both distances
          between them (with respect to the periodic grid), i.e. one should have
          i.e. one should have
              dimension+extensionMin >= 2*K - 1,
          To sum up, extensionMin should be chosen such that
              dimension+extensionMin >= max(dimension, K) + K - 1
          i.e.
              extensionMin >= max(K-1,2*K-dimension-1)
    2) For large data set:
        - conditioningMethod should be set to 2 for using FFT
        - measureErrVar could be set to a small positive value to stabilize
          the covariance matrix (solving linear system)

    :param x:           (2-dimensional array array of dim n x 2) coordinate of data points
    :param v:           (1-dimensional array length n) value at data points
    :param covFun:      (function) covariance function f(h), where
                            h:  (2-dimensional array of dim n x 2, or
                                1-dimensional array of dim 2) 2D-lag(s)
    :param dimension:   (sequence of 2 int) [nx, ny], number of cells
                            in x-, y-axis direction
    :param spacing:     (sequence of 2 float) [dx, dy], spacing between
                            two adjacent cells in x-, y-axis direction
    :param origin:      (sequence of 2 float) [ox, oy], origin of the 2D field
                            - used for localizing the conditioning points
    :param nreal:       (int) number of realizations
    :param mean:        (float or ndarray) mean of the GRF:
                            - scalar for stationary mean
                            - ndarray for non stationary mean, must contain
                                nx*ny values (reshaped if needed)
    :param var:         (float or ndarray or None) variance of the GRF,
                            if not None: variance of GRF is updated
                            depending on the specified variance and the covariance
                            function, otherwise: only the covariance function is
                            used
                                - scalar for stationary variance
                                - array for non stationary variance, must contain
                                    nx*ny values (reshaped if needed)
    :param extensionMin: (sequence of 2 int) minimal extension in nodes in
                            x-, y-axis direction for embedding (see above)
    :param conditioningMethod:
                        (int) indicates which method is used to perform kriging.
                            Let
                                A: index of conditioning (data) nodes
                                B: index of non-conditioning nodes
                            and
                                    +         +
                                    | rAA rAB |
                                r = |         |
                                    | rBA rBB |
                                    +         +
                            the covariance matrix, where index A (resp. B) refers
                            to conditioning (resp. non-conditioning) index in the
                            grid. Then, thre kriging estimates and variance are
                                krig[B]    = mean + rBA * rAA^(-1) * (v - mean)
                                krigVar[B] = diag(rBB - rBA * rAA^(-1) * rAB)
                            The computation is done in a way depending on the
                            following possible values for conditioningMethod:
                                1: method CondtioningA:
                                   the matrices rBA, RAA^(-1) are explicitly
                                   computed (warning: could require large amount
                                   of memory), then all the simulations are updated
                                   by a sum and a multiplication by the matrix M
                                2: method ConditioningB:
                                   for kriging estimates:
                                       the linear system
                                         rAA * y = (v - mean)
                                       is solved, and then
                                         mean + rBA*y
                                       is computed
                                   for kriging variances:
                                       for each column u[j] of rAB, the linear
                                       system
                                         rAA * y = u[j]
                                       is solved, and then
                                         rBB[j,j] - y^t*y
                                       is computed
    :param measureErrVar:
                        (float >=0) measurement error variance; we assume that
                            the error on conditioining data follows the distrubution
                            N(0,measureErrVar*I); i.e. rAA + measureErrVar*I is
                            considered instead of rAA for stabilizing the linear
                            system for this matrix.
    :param tolInvKappa: (float >0) the function is stopped if the inverse of
                            the condition number of rAA is above tolInvKappa
    :param computeKrigSD:
                        (bool) indicates if the standard deviation of kriging is computed
    :param printInfo:   (bool) indicates if some info is printed in stdout

    :return ret:        two possible cases:
                            ret = [krig, krigSD] if computeKrigSD is equal to True
                            ret = krig           if computeKrigSD is equal to False
                        where
                            krig:   (2-dimensional array of dim ny x nx)
                                        kriging estimates
                            krigSD: (2-dimensional array of dim ny x nx)
                                        kriging standard deviation

    NOTES:
        Discrete Fourier Transform (DFT) of an array x of dim N1 x N2 is given by
            c = DFT(x) = F * x
        where F is the the (N1*N2) x (N1*N2) matrix with coefficients
            F(j,k) = [exp( -i*2*pi*(j^t*k)/(N1*N2) )], j=(j1,j2), k=(k1,k2) in G,
        and
            G = {n=(n1,n2), 0 <= n1 <= N1-1, 0 <= n2 <= N2-1}
        denotes the indices grid
        and where we use the bijection
            (n1,n2) in G -> n1 + n2 * N1 in {0,...,N1*N2-1},
        between the multiple-indices and the single indices

        With N = N1*N2, we have
            F^(-1) = 1/N * F^(*)
        where ^(*) denotes the conjugate transpose
        Let
            Q = 1/N^(1/2) * F
        Then Q is unitary, i.e. Q^(-1) = Q^(*)
        Then, we have
            DFT = F = N^(1/2) * Q
            DFT^(-1) = 1/N * F^(*) = 1/N^(1/2) * Q^(*)

        Using numpy package in python3, we have
            numpy.fft.fft2() = DFT
            numpy.fft.ifft2() = DFT^(-1)
    """

    if conditioningMethod not in (1, 2):
        print('ERROR (KRIGE2D): invalid method!')
        return

    nx, ny = dimension
    dx, dy = spacing
    # ox, oy = origin

    nxy = nx*ny

    x = np.asarray(x).reshape(-1,2) # cast in 1-dimensional array if needed
    v = np.asarray(v).reshape(-1) # cast in 1-dimensional array if needed

    mean = np.asarray(mean).reshape(-1) # cast in 1-dimensional array if needed

    if mean.size != 1:
        if mean.size != nxy:
            print('ERROR (KRIGE2D): number of entry for "mean"...')
            return
        mean = np.asarray(mean).reshape(ny, nx) # cast in 2-dimensional array of same shape as grid

    if var is not None:
        var = np.asarray(var).reshape(-1) # cast in 1-dimensional array if needed
        if var.size != 1:
            if var.size != nxy:
                print('ERROR (KRIGE2D): number of entry for "var"...')
                return
            var = np.asarray(var).reshape(ny, nx) # cast in 2-dimensional array of same shape as grid

    if extensionMin is None:
        extensionMin = [nx-1, ny-1] # default

    N1min = nx + extensionMin[0]
    N2min = ny + extensionMin[1]

    if printInfo:
        print('KRIGE2D: Computing circulant embedding...')

    # Circulant embedding of the covariance matrix
    # --------------------------------------------
    # The embedding matrix is a (N1,N2)-nested block circulant matrix, computed from
    # the covariance function.
    # To take a maximal benefit of Fast Fourier Transform (FFT) for computing DFT,
    # we choose
    #     N1 = 2^g1 (a power of 2), with N1 >= N1min
    #     N2 = 2^g2 (a power of 2), with N2 >= N2min
    g1 = int(np.ceil(np.log2(N1min)))
    g2 = int(np.ceil(np.log2(N2min)))
    N1 = int(2**g1)
    N2 = int(2**g2)

    if printInfo:
        print('KRIGE2D: Embedding dimension: {} x {}'.format(N1, N2))

    N = N1*N2

    # ccirc: coefficient of the embedding matrix (N2, N1) array
    L1 = int (N1/2)
    L2 = int (N2/2)
    h1 = np.arange(-L1, L1, dtype=float) * dx # [-L1 ... 0 ... L1-1] * dx
    h2 = np.arange(-L2, L2, dtype=float) * dy # [-L2 ... 0 ... L2-1] * dy

    hh = np.meshgrid(h1, h2)
    ccirc = covFun(np.hstack((hh[0].reshape(-1,1), hh[1].reshape(-1,1))))
    ccirc.resize(N2, N1)

    del(h1, h2, hh)

    # ...shift first L1 index to the end of the axis 1:
    ind = np.arange(L1)
    ccirc = ccirc[:, np.hstack((ind+L1, ind))]
    # ...shift first L2 index to the end of the axis 0:
    ind = np.arange(L2)
    ccirc = ccirc[np.hstack((ind+L2, ind)), :]

    del(ind)

    if printInfo:
        print('KRIGE2D: Computing FFT of circulant matrix...')

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
        varUpdate = np.sqrt(var/covFun(np.zeros(2)))

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
    #    diag((var/covFun(0))^1/2) * r * diag((var/covFun(0))^1/2)
    # which is accounting in the computation of kriging estimates and standard
    # deviation below

    if printInfo:
        print('KRIGE2D: Computing covariance matrix (rAA) for conditioning locations...')

    # Compute
    #    indc: node index of conditioning node (nearest node)
    indc = np.asarray(np.floor((x-origin)/spacing), dtype=int) # multiple-indices: size n x 2

    ix, iy = indc[:, 0], indc[:, 1]

    if sum(ix < 0) > 0 or sum(ix >= nx):
        print('ERROR (KRIGE2D): a conditioning point is out of the grid (x-direction)')
        return
    if sum(iy < 0) > 0 or sum(iy >= ny):
        print('ERROR (KRIGE2D): a conditioning point is out of the grid (y-direction)')
        return

    indc = ix + iy * nx # single-indices

    if len(np.unique(indc)) != len(x):
        print('ERROR (KRIGE2D): more than one conditioning point in a same grid cell')

    nc = len(x)

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
        print('ERROR (GRF2D): conditioning issue: condition number of matrix rAA is too big')
        return

    # Compute:
    #    indnc: node index of non-conditioning node (nearest node)
    indnc = np.asarray(np.setdiff1d(np.arange(nxy), indc), dtype=int)
    nnc = len(indnc)

    ky = np.floor_divide(indnc, nx)
    kx = np.mod(indnc, nx)

    # Initialize
    krig = np.zeros(ny*nx)
    if computeKrigSD:
        krigSD = np.zeros(ny*nx)

    if mean.size == 1:
        v = v - mean
    else:
        v = v - mean.reshape(-1)[indc]

    if var is not None and var.size > 1:
        v = 1./varUpdate.reshape(-1)[indc] * v

    if conditioningMethod == 1:
        # Method ConditioningA
        # --------------------
        if printInfo:
            print('KRIGE2D: Computing covariance matrix (rBA) for non-conditioning / conditioning locations...')

        # Compute the parts rBA of the covariance matrix (see above)
        # rBA
        rBA = np.zeros((nnc, nc))
        for j in range(nc):
            rBA[:,j] = ccirc[np.mod(iy[j] - ky, N2), np.mod(ix[j] - kx, N1)]

        del(ix, iy, kx, ky)
        del(ccirc)

        if printInfo:
            print('KRIGE2D: Computing rBA * rAA^(-1)...')

        # compute rBA * rAA^(-1)
        rBArAAinv = np.dot(rBA, np.linalg.inv(rAA))

        del(rAA)
        if not computeKrigSD:
            del(rBA)

        # Compute kriging estimates
        if printInfo:
            print('KRIGE2D: computing kriging estimates...')

        krig[indnc] = np.dot(rBArAAinv, v)
        krig[indc] = v

        if computeKrigSD:
            # Compute kriging standard deviation
            if printInfo:
                print('KRIGE2D: computing kriging standard deviation ...')

            krigSD[indnc] = np.sqrt(diagEntry - np.diag(np.dot(rBArAAinv, np.transpose(rBA))))

            del(rBA)

    elif conditioningMethod == 2:
        # Method ConditioningB
        # --------------------
        if not computeKrigSD:
            del(ccirc)

        if printInfo:
            print('KRIGE2D: Computing index in the embedding grid for non-conditioning / conditioning locations...')

        # Compute index in the embedding grid for indc and indnc
        # (to allow use of fft)
        indcEmb =  iy * N1 + ix
        indncEmb = ky * N1 + kx

        # Compute kriging estimates
        if printInfo:
            print('KRIGE2D: computing kriging estimates...')

        # Compute
        #    u = rAA^(-1) * v, and then
        #    Z = rBA * u via the circulant embedding of the covariance matrix
        uEmb = np.zeros(N2*N1)
        uEmb[indcEmb] = np.linalg.solve(rAA, v)
        Z = np.fft.ifft2(lam * np.fft.fft2(uEmb.reshape(N2, N1)))
        # ...note that Im(Z) = 0
        krig[indnc] = np.real(Z.reshape(-1)[indncEmb])
        krig[indc] = v

        if computeKrigSD:
            # Compute kriging standard deviation
            if printInfo:
                print('KRIGE2D: computing kriging standard deviation ...')

            for j in range(nnc):
                u = ccirc[np.mod(iy - ky[j], N2), np.mod(ix - kx[j], N1)] # j-th row of rBA
                krigSD[indnc[j]] = np.dot(u,np.linalg.solve(rAA, u))

            del(ccirc)

            krigSD[indnc] = np.sqrt(diagEntry - krigSD[indnc])

        del(ix, iy, kx, ky)

    # ... update if non stationary covariance is specified
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
        return ([krig, krigSD])
    else:
        return (krig)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def grf3D(covFun, dimension, spacing, origin=[0., 0., 0.],
          nreal=1, mean=0, var=None,
          x=None, v=None,
          extensionMin=None, crop=True,
          method=3, conditioningMethod=2,
          measureErrVar=0., tolInvKappa=1.e-10,
          printInfo=True):
    """
    Generates gaussian random fields (GRF) in 3D via FFT.

    The GRFs:
        - are generated using the covariance function (covFun),
        - have specified mean (mean) and variance (var), which can be non stationary
        - are conditioned to location x with value v
    Notes:
    1) For reproducing covariance model, the dimension of GRF should be large
       enough; let K an integer such that K*spacing is greater or equal to the
       correlation range, then
        - correlation accross opposite border should be removed by extending
          the domain sufficiently, i.e.
              extensionMin >= K - 1
        - two nodes could not be correlated simultaneously regarding both distances
          between them (with respect to the periodic grid), i.e. one should have
          i.e. one should have
              dimension+extensionMin >= 2*K - 1,
          To sum up, extensionMin should be chosen such that
              dimension+extensionMin >= max(dimension, K) + K - 1
          i.e.
              extensionMin >= max(K-1,2*K-dimension-1)
    2) For large conditional simulations with large data set:
        - conditioningMethod should be set to 2 for using FFT in conditioning step
        - measureErrVar could be set to a small positive value to stabilize
          the covariance matrix for conditioning locations (solving linear system)

    :param covFun:      (function) covariance function f(h), where
                            h:  (2-dimensional array of dim n x 3, or
                                1-dimensional array of dim 3) 2D-lag(s)
    :param dimension:   (sequence of 3 int) [nx, ny, nz], number of cells
                            in x-, y-, z-axis direction
    :param spacing:     (sequence of 3 float) [dx, dy, dz], spacing between
                            two adjacent cells in x-, y-, z-axis direction
    :param origin:      (sequence of 3 float) [ox, oy, oz], origin of the 2D field
                            - used for localizing the conditioning points
    :param nreal:       (int) number of realizations
    :param mean:        (float or ndarray) mean of the GRF:
                            - scalar for stationary mean
                            - ndarray for non stationary mean, must contain
                                nx*ny*nz values (reshaped if needed)
    :param var:         (float or ndarray or None) variance of the GRF,
                            if not None: variance of GRF is updated
                            depending on the specified variance and the covariance
                            function, otherwise: only the covariance function is
                            used
                                - scalar for stationary variance
                                - array for non stationary variance, must contain
                                    nx*ny*nz values (reshaped if needed)
    :param x:           (2-dimensional array of dim n x 3, or
                            1-dimensional array of dim 3 or None) coordinate of
                            conditioning points (None for unconditional GRF)
    :param v:           (1-dimensional array or float or None) value at
                            conditioning points (length n)
    :param extensionMin: (sequence of 3 int) minimal extension in nodes in
                            x-, y-, z-axis direction for embedding (see above)
    :param crop:        (bool) indicates if the extended generated field will
                            be cropped to original dimension; note that no cropping
                            is not valid with conditioning or non stationary mean
                            or variance
    :param method:      (int) indicates which method is used to generate
                            unconditional simulations; for each method the DFT "lam"
                            of the circulant embedding of the covariance matrix is
                            used, and periodic and stationary GRFs are generated;
                            possible values:
                                1: method A:
                                   generate one GRF Z as follows:
                                   - generate one real gaussian white noise W
                                   - apply fft (or fft inverse) on W to get X
                                   - multiply X by lam (term by term)
                                   - apply fft inverse (or fft) to get Z
                                2: method B: NOT IMPLEMENTED!!!
                                   generate one GRF Z as follows:
                                   - generate directly X (of method A)
                                   - multiply X by lam (term by term)
                                   - apply fft inverse (or fft) to get Z
                                3: method C:
                                   generate two independent GRFs Z1, Z2 as follows:
                                   - generate two independant real gaussian white
                                     noises W1, W2 and set W = W1 + i * W2
                                   - apply fft (or fft inverse) on W to get X
                                   - multiply X by lam (term by term)
                                   - apply fft inverse (or fft) to get Z,
                                     and set Z1 = Re(Z), Z2 = Im(Z)
                                   note: if nreal is odd, the last field is
                                         generated using method A
    :param conditioningMethod:
                        (int) indicates which method is used to update simulation
                            for accounting conditioning data.
                            Let
                                A: index of conditioning nodes
                                B: index of non-conditioning nodes
                                Zobs: vector of values of the unconditional
                                      simulation Z at conditioning nodes
                            and
                                    +         +
                                    | rAA rAB |
                                r = |         |
                                    | rBA rBB |
                                    +         +
                            the covariance matrix, where index A (resp. B) refers
                            to conditioning (resp. non-conditioning) index in the
                            grid. Then, an unconditional simulation Z is updated
                            into a conditional simulation ZCond as follows:
                            Let
                                ZCond[A] = Zobs
                                ZCond[B] = Z[B] + rBA * rAA^(-1) * (Zobs - Z[A])
                            (that is the update consists in adding the kriging
                            estimates of the residues to the unconditional
                            simulation); possible values for conditioningMethod:
                                1: method CondtioningA:
                                   the matrix M = rBA * rAA^(-1) is explicitly
                                   computed (warning: could require large amount
                                   of memory), then all the simulations are updated
                                   by a sum and a multiplication by the matrix M
                                2: method ConditioningB:
                                   for each simulation: the linear system
                                        rAA * x = Zobs - Z[A]
                                   is solved and then, the multiplication by rBA
                                   is done via fft
    :param measureErrVar:
                        (float >=0) measurement error variance; we assume that
                            the error on conditioining data follows the distrubution
                            N(0,measureErrVar*I); i.e. rAA + measureErrVar*I is
                            considered instead of rAA for stabilizing the linear
                            system for this matrix.
                            (Ignored if x is None, i.e. unconditional simulations)
    :param tolInvKappa: (float >0) used only for conditioning, the simulation is
                            stopped if the inverse of the condition number of rAA
                            is above tolInvKappa
    :param printInfo:   (bool) indicates if some info is printed in stdout

    :return grf:    (4-dimensional array of dim nreal x n3 x n2 x n1) nreal GRFs
                        with n1 = nx, n2 = ny, n3 = nz if crop = True,
                        and n1 >= nx, n2 >= ny, n3 >= nz otherwise;
                        grf[i] is the i-th realization

    NOTES:
        Discrete Fourier Transform (DFT) of an array x of dim N1 x N2 x N3 is given by
            c = DFT(x) = F * x
        where F is the the (N1*N2*N3) x (N1*N2*N3) matrix with coefficients
            F(j,k) = [exp( -i*2*pi*(j^t*k)/(N1*N2*N3) )], j=(j1,j2,j3), k=(k1,k2,k3) in G,
        and
            G = {n=(n1,n2,n3), 0 <= n1 <= N1-1, 0 <= n2 <= N2-1, 0 <= n3 <= N3-1}
        denotes the indices grid
        and where we use the bijection
            (n1,n2,n3) in G -> n1 + n2 * N1 + n3 * N1 * N2 in {0,...,N1*N2*N3-1},
        between the multiple-indices and the single indices

        With N = N1*N2*N3, we have
            F^(-1) = 1/N * F^(*)
        where ^(*) denotes the conjugate transpose
        Let
            Q = 1/N^(1/2) * F
        Then Q is unitary, i.e. Q^(-1) = Q^(*)
        Then, we have
            DFT = F = N^(1/2) * Q
            DFT^(-1) = 1/N * F^(*) = 1/N^(1/2) * Q^(*)

        Using numpy package in python3, we have
            numpy.fft.fftn() = DFT
            numpy.fft.ifftn() = DFT^(-1)
    """

    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if printInfo:
            print('GRF3D: nreal = 0: nothing to do!')
        return()

    if printInfo:
        print('GRF3D: Preliminary computation...')

    #### Preliminary computation ####
    nx, ny, nz = dimension
    dx, dy, dz = spacing
    # ox, oy, oz = origin

    nxy = nx*ny
    nxyz = nxy * nz

    if method not in (1, 2, 3):
        print('ERROR (GRF3D): invalid method')
        return

    if method == 2:
        print('ERROR (GRF3D): Unconditional simulation: "method=2" not implemented...')
        return

    if x is not None:
        if conditioningMethod not in (1, 2):
            print('ERROR (GRF3D): invalid method for conditioning')
            return
        x = np.asarray(x).reshape(-1,3) # cast in 1-dimensional array if needed
        v = np.asarray(v).reshape(-1) # cast in 1-dimensional array if needed

    mean = np.asarray(mean).reshape(-1) # cast in 1-dimensional array if needed

    if mean.size != 1:
        if mean.size != nxyz:
            print('ERROR (GRF3D): number of entry for "mean"...')
            return
        mean = np.asarray(mean).reshape(nz, ny, nx) # cast in 3-dimensional array of same shape as grid

    if var is not None:
        var = np.asarray(var).reshape(-1) # cast in 1-dimensional array if needed
        if var.size != 1:
            if var.size != nxyz:
                print('ERROR (GRF3D): number of entry for "var"...')
                return
            var = np.asarray(var).reshape(nz, ny, nx) # cast in 3-dimensional array of same shape as grid

    if not crop:
        if x is not None: # conditional simulation
            print('ERROR (GRF3D): "no crop" is not valid with conditional simulation')
            return

        if mean.size > 1:
            print('ERROR (GRF3D): "no crop" is not valid with non stationary mean')
            return

        if var is not None and var.size > 1:
            print('ERROR (GRF3D): "no crop" is not valid with non stationary variance')
            return

    if extensionMin is None:
        extensionMin = [nx-1, ny-1, nz-1] # default

    N1min = nx + extensionMin[0]
    N2min = ny + extensionMin[1]
    N3min = nz + extensionMin[2]

    if printInfo:
        print('GRF3D: Computing circulant embedding...')

    # Circulant embedding of the covariance matrix
    # --------------------------------------------
    # The embedding matrix is a (N1,N2,N3)-nested block circulant matrix, computed from
    # the covariance function.
    # To take a maximal benefit of Fast Fourier Transform (FFT) for computing DFT,
    # we choose
    #     N1 = 2^g1 (a power of 2), with N1 >= N1min
    #     N2 = 2^g2 (a power of 2), with N2 >= N2min
    #     N3 = 2^g3 (a power of 2), with N3 >= N3min
    g1 = int(np.ceil(np.log2(N1min)))
    g2 = int(np.ceil(np.log2(N2min)))
    g3 = int(np.ceil(np.log2(N3min)))
    N1 = int(2**g1)
    N2 = int(2**g2)
    N3 = int(2**g3)

    if printInfo:
        print('GRF3D: Embedding dimension: {} x {} x {}'.format(N1, N2, N3))

    N12 = N1*N2
    N = N12 * N3

    # ccirc: coefficient of the embedding matrix, (N3, N2, N1) array
    L1 = int (N1/2)
    L2 = int (N2/2)
    L3 = int (N3/2)
    h1 = np.arange(-L1, L1, dtype=float) * dx # [-L1 ... 0 ... L1-1] * dx
    h2 = np.arange(-L2, L2, dtype=float) * dy # [-L2 ... 0 ... L2-1] * dy
    h3 = np.arange(-L3, L3, dtype=float) * dz # [-L3 ... 0 ... L3-1] * dz

    hh = np.meshgrid(h2, h3, h1) # as this! hh[i]: (N3, N2, N1) array
                                 # hh[0]: y-coord, hh[1]: z-coord, hh[2]: x-coord
    ccirc = covFun(np.hstack((hh[2].reshape(-1,1), hh[0].reshape(-1,1), hh[1].reshape(-1,1))))
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

    if printInfo:
        print('GRF3D: Computing FFT of circulant matrix...')

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
        varUpdate = np.sqrt(var/covFun(np.zeros(3)))

    # Dealing with conditioning
    # -------------------------
    if x is not None:
        if printInfo:
            print('GRF3D: Treatment of conditioning data...')
        # Compute the part rAA of the covariance matrix
        #        +         +
        #        | rAA rAB |
        #    r = |         |
        #        | rBA rBB |
        #        +         +
        # where index A (resp. B) refers to
        # conditioning (resp. non-conditioning) index in the grid.

        if printInfo:
            print('GRF3D: Computing covariance matrix (rAA) for conditioning locations...')

        # Compute
        #    indc: node index of conditioning node (nearest node)
        indc = np.asarray(np.floor((x-origin)/spacing), dtype=int) # multiple-indices: size n x 3

        ix, iy, iz = indc[:, 0], indc[:, 1], indc[:, 2]

        if sum(ix < 0) > 0 or sum(ix >= nx):
            print('ERROR (GRF3D): a conditioning point is out of the grid (x-direction)')
            return
        if sum(iy < 0) > 0 or sum(iy >= ny):
            print('ERROR (GRF3D): a conditioning point is out of the grid (y-direction)')
            return
        if sum(iz < 0) > 0 or sum(iz >= nz):
            print('ERROR (GRF3D): a conditioning point is out of the grid (z-direction)')
            return

        indc = ix + iy * nx + iz * nxy # single-indices

        if len(np.unique(indc)) != len(x):
            print('ERROR (GRF3D): more than one conditioning point in a same grid cell')

        nc = len(x)

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
            print('ERROR (GRF3D): conditioning issue: condition number of matrix rAA is too big')
            return

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
            if printInfo:
                print('GRF3D: Computing covariance matrix (rBA) for non-conditioning / conditioning locations...')

            # Compute the parts rBA of the covariance matrix (see above)
            # rBA
            rBA = np.zeros((nnc, nc))
            for j in range(nc):
                rBA[:,j] = ccirc[np.mod(iz[j] - kz, N3), np.mod(iy[j] - ky, N2), np.mod(ix[j] - ky, N1)]

            if printInfo:
                print('GRF3D: Computing rBA * rAA^(-1)...')

            # compute rBA * rAA^(-1)
            rBArAAinv = np.dot(rBA, np.linalg.inv(rAA))

            del(rAA, rBA)

            # If a variance var is specified, then the matrix r should be updated
            # by the following operation:
            #    diag((var/covFun(0))^1/2) * r * diag((var/covFun(0))^1/2)
            # Hence, if a non stationary variance is specified,
            # the matrix rBA * rAA^(-1) should be consequently updated
            # by multiplying its columns by 1/varUpdate[indc] and its rows by varUpdate[indnc]
            if var is not None and var.size > 1:
                rBArAAinv = np.transpose(varUpdate.reshape(-1)[indnc] * np.transpose(1./varUpdate.reshape(-1)[indc] * rBArAAinv))

        elif conditioningMethod == 2:
            # Method ConditioningB
            # --------------------
            if printInfo:
                print('GRF3D: Computing index in the embedding grid for non-conditioning / conditioning locations...')

            # Compute index in the embedding grid for indc and indnc
            # (to allow use of fft)
            indcEmb =  iz * N12 + iy * N1 + ix
            indncEmb = kz * N12 + ky * N1 + kx

        del(ix, iy, kx, ky)

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
            if printInfo:
                print('GRF3D: Unconditional simulation {:4d} of {:4d}...'.format(i+1, nreal))

            W = np.random.normal(size=(N3, N2, N1))

            Z = np.fft.ifftn(lamSqrt * np.fft.fftn(W))
            # ...note that Im(Z) = 0
            grf[i] = np.real(Z[0:grfNz, 0:grfNy, 0:grfNx])

    elif method == 2:
        # Method B
        # --------
        print('ERROR (GRF3D): Unconditional simulation: "method=2" not implemented...')
        return

    elif method == 3:
        # Method C
        # --------
        if nreal > 1:
            for i in np.arange(0,nreal,2):
                if printInfo:
                    print('GRF3D: Unconditional simulation {:4d}-{:4d} of {:4d}...'.format(i+1, i+2, nreal))

                W = np.array(np.random.normal(size=(N3, N2, N1)), dtype=complex)
                W.imag = np.random.normal(size=(N3, N2, N1))
                Z = np.sqrt(N) * np.fft.ifftn(lamSqrt * W)
                #  Z = 1/np.sqrt(N) * np.fft.fftn(lamSqrt * W)] # see above: [OR:...]

                grf[i] = np.real(Z[0:grfNz, 0:grfNy, 0:grfNx])
                grf[i+1] = np.imag(Z[0:grfNz, 0:grfNy, 0:grfNx])

        if np.mod(nreal, 2) == 1:
            if printInfo:
                print('GRF3D: Unconditional simulation {:4d} of {:4d}...'.format(nreal, nreal))

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
            if printInfo:
                print('GRF3D: Updating conditional simulations...')

            # Update all simulations at a time,
            # use the matrix rBA * rAA^(-1) already computed
            grf[:,indnc] = grf[:,indnc] + np.transpose(np.dot(rBArAAinv, np.transpose(v - grf[:,indc])))
            grf[:,indc] = v

        elif conditioningMethod == 2:
            # Method ConditioningB
            # --------------------
            # Update each simulation successively as follows:
            #    - solve rAA * x = Zobs - z[A]
            #    - do the multiplication rBA * x via the circulant embedding of the
            #      covariance matrix (using fft)
            rAAinvResiduEmb = np.zeros(N3*N2*N1)

            for i in range(nreal):
                if printInfo:
                    print('GRF3D: Updating conditional simulation {:4d} of {:4d}...'.format(i+1, nreal))

                # Compute residue
                residu = v - grf[i,indc]
                # ... update if non stationary variance is specified
                if var is not None and var.size > 1:
                    residu = 1./varUpdate.reshape(-1)[indc] * residu

                # Compute
                #    x = rAA^(-1) * residu, and then
                #    Z = rBA * x via the circulant embedding of the covariance matrix
                rAAinvResiduEmb[indcEmb] = np.linalg.solve(rAA, residu)
                Z = np.fft.ifftn(lam * np.fft.fftn(rAAinvResiduEmb.reshape(N3, N2, N1)))
                # ...note that Im(Z) = 0
                Z = np.real(Z.reshape(-1)[indncEmb])

                # ... update if non stationary covariance is specified
                if var is not None and var.size > 1:
                    Z = varUpdate.reshape(-1)[indnc] * Z

                grf[i, indnc] = grf[i, indnc] + Z
                grf[i, indc] = v

        # Reshape grf as initially
        grf.resize(nreal, grfNz, grfNy, grfNx)

    return (grf)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def krige3D(x, v, covFun, dimension, spacing, origin=[0., 0., 0.],
            mean=0, var=None,
            extensionMin=None,
            conditioningMethod=2,
            measureErrVar=0., tolInvKappa=1.e-10,
            computeKrigSD=True,
            printInfo=True):
    """
    Computes kriging estimates and standard deviation in 3D via FFT.

    It is a simple kriging
        - of value v at location x,
        - based on the covariance function (covFun),
        - with a specified mean (mean) and variance (var), which can be non stationary
    Notes:
    1) For reproducing covariance model, the dimension of field/domain should be large
       enough; let K an integer such that K*spacing is greater or equal to the
       correlation range, then
        - correlation accross opposite border should be removed by extending
          the domain sufficiently, i.e.
              extensionMin >= K - 1
        - two nodes could not be correlated simultaneously regarding both distances
          between them (with respect to the periodic grid), i.e. one should have
          i.e. one should have
              dimension+extensionMin >= 2*K - 1,
          To sum up, extensionMin should be chosen such that
              dimension+extensionMin >= max(dimension, K) + K - 1
          i.e.
              extensionMin >= max(K-1,2*K-dimension-1)
    2) For large data set:
        - conditioningMethod should be set to 2 for using FFT
        - measureErrVar could be set to a small positive value to stabilize
          the covariance matrix (solving linear system)

    :param x:           (2-dimensional array array of dim n x 3) coordinate of data points
    :param v:           (1-dimensional array length n) value at data points
    :param covFun:      (function) covariance function f(h), where
                            h:  (2-dimensional array of dim n x 3, or
                                1-dimensional array of dim 3) 2D-lag(s)
    :param dimension:   (sequence of 3 int) [nx, ny, nz], number of cells
                            in x-, y-, z-axis direction
    :param spacing:     (sequence of 3 float) [dx, dy, dz], spacing between
                            two adjacent cells in x-, y-, z-axis direction
    :param origin:      (sequence of 3 float) [ox, oy, oz], origin of the 2D field
                            - used for localizing the conditioning points
    :param nreal:       (int) number of realizations
    :param mean:        (float or ndarray) mean of the GRF:
                            - scalar for stationary mean
                            - ndarray for non stationary mean, must contain
                                nx*ny*nz values (reshaped if needed)
    :param var:         (float or ndarray or None) variance of the GRF,
                            if not None: variance of GRF is updated
                            depending on the specified variance and the covariance
                            function, otherwise: only the covariance function is
                            used
                                - scalar for stationary variance
                                - array for non stationary variance, must contain
                                    nx*ny*nz values (reshaped if needed)
    :param extensionMin: (sequence of 3 int) minimal extension in nodes in
                            x-, y-, z-axis direction for embedding (see above)
    :param conditioningMethod:
                        (int) indicates which method is used to perform kriging.
                            Let
                                A: index of conditioning (data) nodes
                                B: index of non-conditioning nodes
                            and
                                    +         +
                                    | rAA rAB |
                                r = |         |
                                    | rBA rBB |
                                    +         +
                            the covariance matrix, where index A (resp. B) refers
                            to conditioning (resp. non-conditioning) index in the
                            grid. Then, thre kriging estimates and variance are
                                krig[B]    = mean + rBA * rAA^(-1) * (v - mean)
                                krigVar[B] = diag(rBB - rBA * rAA^(-1) * rAB)
                            The computation is done in a way depending on the
                            following possible values for conditioningMethod:
                                1: method CondtioningA:
                                   the matrices rBA, RAA^(-1) are explicitly
                                   computed (warning: could require large amount
                                   of memory), then all the simulations are updated
                                   by a sum and a multiplication by the matrix M
                                2: method ConditioningB:
                                   for kriging estimates:
                                       the linear system
                                         rAA * y = (v - mean)
                                       is solved, and then
                                         mean + rBA*y
                                       is computed
                                   for kriging variances:
                                       for each column u[j] of rAB, the linear
                                       system
                                         rAA * y = u[j]
                                       is solved, and then
                                         rBB[j,j] - y^t*y
                                       is computed
    :param measureErrVar:
                        (float >=0) measurement error variance; we assume that
                            the error on conditioining data follows the distrubution
                            N(0,measureErrVar*I); i.e. rAA + measureErrVar*I is
                            considered instead of rAA for stabilizing the linear
                            system for this matrix.
    :param tolInvKappa: (float >0) the function is stopped if the inverse of
                            the condition number of rAA is above tolInvKappa
    :param computeKrigSD:
                        (bool) indicates if the standard deviation of kriging is computed
    :param printInfo:   (bool) indicates if some info is printed in stdout

    :return ret:        two possible cases:
                            ret = [krig, krigSD] if computeKrigSD is equal to True
                            ret = krig           if computeKrigSD is equal to False
                        where
                            krig:   (3-dimensional array of dim nz x ny x nx)
                                        kriging estimates
                            krigSD: (3-dimensional array of dim nz x ny x nx)
                                        kriging standard deviation

    NOTES:
        Discrete Fourier Transform (DFT) of an array x of dim N1 x N2 x N3 is given by
            c = DFT(x) = F * x
        where F is the the (N1*N2*N3) x (N1*N2*N3) matrix with coefficients
            F(j,k) = [exp( -i*2*pi*(j^t*k)/(N1*N2*N3) )], j=(j1,j2,j3), k=(k1,k2,k3) in G,
        and
            G = {n=(n1,n2,n3), 0 <= n1 <= N1-1, 0 <= n2 <= N2-1, 0 <= n3 <= N3-1}
        denotes the indices grid
        and where we use the bijection
            (n1,n2,n3) in G -> n1 + n2 * N1 + n3 * N1 * N2 in {0,...,N1*N2*N3-1},
        between the multiple-indices and the single indices

        With N = N1*N2*N3, we have
            F^(-1) = 1/N * F^(*)
        where ^(*) denotes the conjugate transpose
        Let
            Q = 1/N^(1/2) * F
        Then Q is unitary, i.e. Q^(-1) = Q^(*)
        Then, we have
            DFT = F = N^(1/2) * Q
            DFT^(-1) = 1/N * F^(*) = 1/N^(1/2) * Q^(*)

        Using numpy package in python3, we have
            numpy.fft.fftn() = DFT
            numpy.fft.ifftn() = DFT^(-1)
    """

    if conditioningMethod not in (1, 2):
        print('ERROR (KRIGE3D): invalid method!')
        return

    nx, ny, nz = dimension
    dx, dy, dz = spacing
    # ox, oy, oz = origin

    nxy = nx*ny
    nxyz = nxy * nz

    x = np.asarray(x).reshape(-1,3) # cast in 1-dimensional array if needed
    v = np.asarray(v).reshape(-1) # cast in 1-dimensional array if needed

    mean = np.asarray(mean).reshape(-1) # cast in 1-dimensional array if needed

    if mean.size != 1:
        if mean.size != nxyz:
            print('ERROR (KRIGE3D): number of entry for "mean"...')
            return
        mean = np.asarray(mean).reshape(nz, ny, nx) # cast in 3-dimensional array of same shape as grid

    if var is not None:
        var = np.asarray(var).reshape(-1) # cast in 1-dimensional array if needed
        if var.size != 1:
            if var.size != nxyz:
                print('ERROR (KRIGE3D): number of entry for "var"...')
                return
            var = np.asarray(var).reshape(nz, ny, nx) # cast in 3-dimensional array of same shape as grid

    if extensionMin is None:
        extensionMin = [nx-1, ny-1, nz-1] # default

    N1min = nx + extensionMin[0]
    N2min = ny + extensionMin[1]
    N3min = nz + extensionMin[2]

    if printInfo:
        print('KRIGE3D: Computing circulant embedding...')

    # Circulant embedding of the covariance matrix
    # --------------------------------------------
    # The embedding matrix is a (N1,N2,N3)-nested block circulant matrix, computed from
    # the covariance function.
    # To take a maximal benefit of Fast Fourier Transform (FFT) for computing DFT,
    # we choose
    #     N1 = 2^g1 (a power of 2), with N1 >= N1min
    #     N2 = 2^g2 (a power of 2), with N2 >= N2min
    #     N3 = 2^g3 (a power of 2), with N3 >= N3min
    g1 = int(np.ceil(np.log2(N1min)))
    g2 = int(np.ceil(np.log2(N2min)))
    g3 = int(np.ceil(np.log2(N3min)))
    N1 = int(2**g1)
    N2 = int(2**g2)
    N3 = int(2**g3)

    if printInfo:
        print('KRIGE3D: Embedding dimension: {} x {} x {}'.format(N1, N2, N3))

    N12 = N1*N2
    N = N12 * N3

    # ccirc: coefficient of the embedding matrix, (N3, N2, N1) array
    L1 = int (N1/2)
    L2 = int (N2/2)
    L3 = int (N3/2)
    h1 = np.arange(-L1, L1, dtype=float) * dx # [-L1 ... 0 ... L1-1] * dx
    h2 = np.arange(-L2, L2, dtype=float) * dy # [-L2 ... 0 ... L2-1] * dy
    h3 = np.arange(-L3, L3, dtype=float) * dz # [-L3 ... 0 ... L3-1] * dz

    hh = np.meshgrid(h2, h3, h1) # as this! hh[i]: (N3, N2, N1) array
                                 # hh[0]: y-coord, hh[1]: z-coord, hh[2]: x-coord
    ccirc = covFun(np.hstack((hh[2].reshape(-1,1), hh[0].reshape(-1,1), hh[1].reshape(-1,1))))
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

    if printInfo:
        print('KRIGE3D: Computing FFT of circulant matrix...')

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
        varUpdate = np.sqrt(var/covFun(np.zeros(3)))

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
    #    diag((var/covFun(0))^1/2) * r * diag((var/covFun(0))^1/2)
    # which is accounting in the computation of kriging estimates and standard
    # deviation below

    if printInfo:
        print('KRIGE3D: Computing covariance matrix (rAA) for conditioning locations...')

    # Compute
    #    indc: node index of conditioning node (nearest node)
    indc = np.asarray(np.floor((x-origin)/spacing), dtype=int) # multiple-indices: size n x 3

    ix, iy, iz = indc[:, 0], indc[:, 1], indc[:, 2]

    if sum(ix < 0) > 0 or sum(ix >= nx):
        print('ERROR (KRIGE3D): a conditioning point is out of the grid (x-direction)')
        return
    if sum(iy < 0) > 0 or sum(iy >= ny):
        print('ERROR (KRIGE3D): a conditioning point is out of the grid (y-direction)')
        return
    if sum(iz < 0) > 0 or sum(iz >= nz):
        print('ERROR (KRIGE3D): a conditioning point is out of the grid (z-direction)')
        return

    indc = ix + iy * nx + iz * nxy # single-indices

    if len(np.unique(indc)) != len(x):
        print('ERROR (KRIGE3D): more than one conditioning point in a same grid cell')

    nc = len(x)

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
        print('ERROR (GRF3D): conditioning issue: condition number of matrix rAA is too big')
        return

    # Compute:
    #    indnc: node index of non-conditioning node (nearest node)
    indnc = np.asarray(np.setdiff1d(np.arange(nxyz), indc), dtype=int)
    nnc = len(indnc)

    kz = np.floor_divide(indnc, nxy)
    kk = np.mod(indnc, nxy)
    ky = np.floor_divide(kk, nx)
    kx = np.mod(kk, nx)
    del(kk)

    # Initialize
    krig = np.zeros(nz*ny*nx)
    if computeKrigSD:
        krigSD = np.zeros(nz*ny*nx)

    if mean.size == 1:
        v = v - mean
    else:
        v = v - mean.reshape(-1)[indc]

    if var is not None and var.size > 1:
        v = 1./varUpdate.reshape(-1)[indc] * v

    if conditioningMethod == 1:
        # Method ConditioningA
        # --------------------
        if printInfo:
            print('KRIGE3D: Computing covariance matrix (rBA) for non-conditioning / conditioning locations...')

        # Compute the parts rBA of the covariance matrix (see above)
        # rBA
        rBA = np.zeros((nnc, nc))
        for j in range(nc):
            rBA[:,j] = ccirc[np.mod(iz[j] - kz, N3), np.mod(iy[j] - ky, N2), np.mod(ix[j] - ky, N1)]

        del(ix, iy, kx, ky)
        del(ccirc)

        if printInfo:
            print('KRIGE3D: Computing rBA * rAA^(-1)...')

        # compute rBA * rAA^(-1)
        rBArAAinv = np.dot(rBA, np.linalg.inv(rAA))

        del(rAA)
        if not computeKrigSD:
            del(rBA)

        # Compute kriging estimates
        if printInfo:
            print('KRIGE3D: computing kriging estimates...')

        krig[indnc] = np.dot(rBArAAinv, v)
        krig[indc] = v

        if computeKrigSD:
            # Compute kriging standard deviation
            if printInfo:
                print('KRIGE3D: computing kriging standard deviation ...')

            krigSD[indnc] = np.sqrt(diagEntry - np.diag(np.dot(rBArAAinv, np.transpose(rBA))))

            del(rBA)

    elif conditioningMethod == 2:
        # Method ConditioningB
        # --------------------
        if not computeKrigSD:
            del(ccirc)

        if printInfo:
            print('KRIGE3D: Computing index in the embedding grid for non-conditioning / conditioning locations...')

        # Compute index in the embedding grid for indc and indnc
        # (to allow use of fft)
        indcEmb =  iz * N12 + iy * N1 + ix
        indncEmb = kz * N12 + ky * N1 + kx

        # Compute kriging estimates
        if printInfo:
            print('KRIGE3D: computing kriging estimates...')

        # Compute
        #    u = rAA^(-1) * v, and then
        #    Z = rBA * u via the circulant embedding of the covariance matrix
        uEmb = np.zeros(N3*N2*N1)
        uEmb[indcEmb] = np.linalg.solve(rAA, v)
        Z = np.fft.ifftn(lam * np.fft.fftn(uEmb.reshape(N3, N2, N1)))
        # ...note that Im(Z) = 0
        krig[indnc] = np.real(Z.reshape(-1)[indncEmb])
        krig[indc] = v

        if computeKrigSD:
            # Compute kriging standard deviation
            if printInfo:
                print('KRIGE3D: computing kriging standard deviation ...')

            for j in range(nnc):
                u = ccirc[np.mod(iz - kz[j], N3), np.mod(iy - ky[j], N2), np.mod(ix - kx[j], N1)] # j-th row of rBA
                krigSD[indnc[j]] = np.dot(u,np.linalg.solve(rAA, u))

            del(ccirc)

            krigSD[indnc] = np.sqrt(diagEntry - krigSD[indnc])

        del(ix, iy, kx, ky)

    # ... update if non stationary covariance is specified
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
        return ([krig, krigSD])
    else:
        return (krig)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def extension_min(r, n, s=1.):
    """
    Compute extension of the dimension in a direction so that a GRF reproduces
    the covariance model appropriately:

    :param r:   (float) range (max) along the considered direction
    :param n:   (int) dimension (number of cells) in the considered direction
    :param s:   (float) cell size in the considered direction

    :return:    (int) extension in number of cells that should be specified for
                    GRF simulation
    """
    k = int(np.ceil(r/s))
    return max(k-n, 0) + k - 1
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Module 'geone.grf' example:")

    import time

    import matplotlib.pyplot as plt
    import geone.covModel as gcm

    ########## 1D case ##########
    # Define grid
    nx = 2000
    dx = 0.5
    ox = 0.0

    # Define covariance model
    cov_model = gcm.CovModel1D(elem=[
                    ('gaussian', {'w':8.95, 'r':100}), # elementary contribution
                    ('nugget', {'w':0.05})             # elementary contribution
                    ], name='')

    # Get covariance function and range
    cov_fun = cov_model.func()

    # Define mean and variance of GRF
    mean = 10.
    # mean = np.linspace(5, 15, nx)
    var = None
    # var = np.linspace(1, 200, nx)

    # Define hard data
    x = [10., 50., 400., 800.]
    v = [ 8.,  9.,   8.,  12.]
    # x, v = None, None

    # Set minimal extension according to the size of the grid and the range
    extensionMin = extension_min(cov_model.r(), nx, s=dx)

    # Set number of realizations
    nreal = 2000

    # Set seed
    np.random.seed(123)

    # Generate GRF
    t1 = time.time()
    grf = grf1D(cov_fun, nx, dx, origin=ox,
                nreal=nreal, mean=mean, var=var,
                x=x, v=v,
                method=3, conditioningMethod=2,
                extensionMin=extensionMin) # grf: (nreal,nx) array
    t2 = time.time()
    time_case1D = t2-t1
    nreal_case1D = nreal
    infogrid_case1D = 'grid: {} cells'.format(nx)
    print('Elapsed time: {} sec'.format(time_case1D))

    # Compute mean and sd over the realizations
    grfMean = np.mean(grf, axis=0) # mean along axis 0
    grfSD = np.std(grf, axis=0) # standard deviation along axis 0

    if x is not None:
        # Kriging
        t1 = time.time()
        krig, krigSD = krige1D(x, v, cov_fun, nx, dx, origin=ox,
                               mean=mean, var=var,
                               conditioningMethod=2,
                               extensionMin=extensionMin)
        t2 = time.time()
        time_krig_case1D = t2-t1
        print('Elapsed time for kriging: {} sec'.format(time_krig_case1D))

    # Display
    # -------
    # xg: center of grid points
    xg = ox + 0.5 * dx + dx * np.arange(nx)

    # === 4 real and mean and sd of all real
    fig, ax = plt.subplots(figsize=(20,10))
    for i in range(4):
        plt.plot(xg, grf[i], label='real #{}'.format(i+1))

    plt.plot(xg, grfMean, c='black', ls='dashed', label='mean ({} real)'.format(nreal))
    plt.plot(xg, grfMean + grfSD, c='gray', label='mean +/- sd ({} real)'.format(nreal))
    plt.plot(xg, grfMean - grfSD, c='gray')

    if x is not None:
        plt.plot(x,v,'+k', markersize=10)
    plt.legend()
    plt.title('GRF1D')

    # fig.show()
    plt.show()

    if x is not None:
        # === 4 real and kriging estimates and sd
        fig, ax = plt.subplots(figsize=(20,10))
        for i in range(4):
            plt.plot(xg, grf[i], label='real #{}'.format(i+1))

        plt.plot(xg, krig, c='black', ls='dashed', label='kriging')
        plt.plot(xg, krig + krigSD, c='gray', label='kriging +/- sd')
        plt.plot(xg, krig - krigSD, c='gray')

        plt.plot(x,v,'+k', markersize=10)
        plt.legend()
        plt.title('GRF1D')

        # fig.show()
        plt.show()

        # === comparison of mean and sd of all real, with kriging estimates and sd
        fig, ax = plt.subplots(figsize=(20,10))
        plt.plot(xg, grfMean - krig, c='black', label='grfMean-krig')
        plt.plot(xg, grfSD - krigSD, c='red', label='grfSD-krigSD')

        plt.axhline(y=0)
        for xx in x:
            plt.axvline(x=xx)
        plt.legend()
        plt.title('GRF1D and KRIGE1D / nreal={}'.format(nreal))

        # fig.show()
        plt.show()

        print('Peak to peak for "grfMean - krig": {}'.format(np.ptp(grfMean-krig)))
        print('Peak to peak for "grfSD - krigSD": {}'.format(np.ptp(grfSD-krigSD)))

        del(krig, krigSD)

    del (grf, grfMean, grfSD)

    ########## 2D case ##########
    # Define grid
    nx, ny = 231, 249
    dx, dy = 1., 1.
    ox, oy = 0., 0.

    dimension = [nx, ny]
    spacing = [dx, dy]
    origin = [ox, oy]

    # Define covariance model
    cov_model = gcm.CovModel2D(elem=[
                    ('gaussian', {'w':8.5, 'r':[150, 40]}), # elementary contribution
                    ('nugget', {'w':0.5})                   # elementary contribution
                    ], alpha=-30, name='')

    # Get covariance function and range
    cov_fun = cov_model.func()

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

    # Set minimal extension according to the size of the grid and the range
    extensionMin = [extension_min(r, n, s) for r, n, s in zip(cov_model.rxy(), dimension, spacing)]

    # Set number of realizations
    nreal = 200

    # Set seed
    np.random.seed(123)

    # Generate GRF
    t1 = time.time()
    grf = grf2D(cov_fun, dimension, spacing, origin=origin,
                nreal=nreal, mean=mean, var = var,
                x=x, v=v,
                method=3, conditioningMethod=2,
                extensionMin=extensionMin) # grf: (nreal,ny,nx) array
    t2 = time.time()
    nreal_case2D = nreal
    time_case2D = t2-t1
    infogrid_case2D = 'grid: {} cells ({} x {})'.format(nx*ny, nx, ny)
    print('Elapsed time: {} sec'.format(time_case2D))

    # Compute mean and sd over the realizations
    grfMean = np.mean(grf.reshape(nreal, -1), axis=0).reshape(ny, nx)
    grfSD = np.std(grf.reshape(nreal, -1), axis=0).reshape(ny, nx)

    if x is not None:
        # Kriging
        t1 = time.time()
        krig, krigSD = krige2D(x, v, cov_fun, dimension, spacing, origin=origin,
                               mean=mean, var=var,
                               conditioningMethod=2,
                               extensionMin=extensionMin)
        t2 = time.time()
        time_krig_case2D = t2-t1
        print('Elapsed time for kriging: {} sec'.format(time_krig_case2D))

    # Display
    # -------
    # xg, yg: center of grid points
    xg = ox + 0.5 * dx + dx * np.arange(nx)
    yg = oy + 0.5 * dy + dy * np.arange(ny)
    xmin, xmax = ox, ox + nx * dx
    ymin, ymax = oy, oy + ny * dy

    # === 4 real and mean and sd of all real
    fig, ax = plt.subplots(2,3,figsize=(24,12))

    # 4 first real ...
    pnum = [1, 2, 4, 5]
    for i in range(4):
        plt.subplot(2,3,pnum[i])
        im_plot = plt.imshow(grf[i],
                             origin='lower', extent=[xmin,xmax,ymin,ymax],
                             interpolation='none')
        plt.colorbar()

        if x is not None:
            plt.plot(x[:,0],x[:,1],'+k', markersize=10)

        plt.title('GRF2D {}: real #{}'.format(cov_model.name, i+1))

    # mean...
    plt.subplot(2,3,3)
    im_plot = plt.imshow(grfMean,
                         origin='lower', extent=[xmin,xmax,ymin,ymax],
                         interpolation='none')
    plt.colorbar()

    if x is not None:
        plt.plot(x[:,0],x[:,1],'+k', markersize=10)

    plt.title('Mean over {} real'.format(nreal))

    # sd...
    plt.subplot(2,3,6)
    im_plot = plt.imshow(grfSD,
                         origin='lower', extent=[xmin,xmax,ymin,ymax],
                         interpolation='none')
    plt.colorbar()

    if x is not None:
        plt.plot(x[:,0],x[:,1],'+k', markersize=10)
    plt.title('SD over {} real'.format(nreal))

    # fig.show()
    plt.show()

    if x is not None:
        # === 4 real and kriging estimates and sd
        fig, ax = plt.subplots(2,3,figsize=(24,12))

        # 4 first real ...
        pnum = [1, 2, 4, 5]
        for i in range(4):
            plt.subplot(2,3,pnum[i])
            im_plot = plt.imshow(grf[i],
                                 origin='lower', extent=[xmin,xmax,ymin,ymax],
                                 interpolation='none')
            plt.colorbar()
            plt.plot(x[:,0],x[:,1],'+k', markersize=10)

            plt.title('GRF2D {}: real #{}'.format(cov_model.name, i+1))

        # kriging estimates...
        plt.subplot(2,3,3)
        im_plot = plt.imshow(krig,
                             origin='lower', extent=[xmin,xmax,ymin,ymax],
                             interpolation='none')
        plt.colorbar()
        plt.plot(x[:,0],x[:,1],'+k', markersize=10)

        plt.title('Kriging')

        # kiging sd...
        plt.subplot(2,3,6)
        im_plot = plt.imshow(krigSD,
                             origin='lower', extent=[xmin,xmax,ymin,ymax],
                             interpolation='none')
        plt.colorbar()
        plt.plot(x[:,0],x[:,1],'+k', markersize=10)

        plt.title('Kriging SD')

        # fig.show()
        plt.show()

        # === comparison of mean and sd of all real, with kriging estimates and sd
        fig, ax = plt.subplots(1,2,figsize=(15,5))

        # grfMean - krig
        plt.subplot(1,2,1)
        im_plot = plt.imshow(grfMean - krig,
                             origin='lower', extent=[xmin,xmax,ymin,ymax],
                             interpolation='none')
        plt.colorbar()
        plt.plot(x[:,0],x[:,1],'+k', markersize=10)

        plt.title('2D: grfMean - krig / nreal={}'.format(nreal))

        # grfMean - krig
        plt.subplot(1,2,2)
        im_plot = plt.imshow(grfSD - krigSD,
                             origin='lower', extent=[xmin,xmax,ymin,ymax],
                             interpolation='none')
        plt.colorbar()
        plt.plot(x[:,0],x[:,1],'+k', markersize=10)

        plt.title('2D: grfSD - krigSD / nreal={}'.format(nreal))

        # fig.show()
        plt.show()

        print('Peak to peak for "grfMean - krig": {}'.format(np.ptp(grfMean-krig)))
        print('Peak to peak for "grfSD - krigSD": {}'.format(np.ptp(grfSD-krigSD)))

        del(krig, krigSD)

    del (grf, grfMean, grfSD)

    ########## 3D case ##########
    # Define grid
    nx, ny, nz = 85, 56, 34
    dx, dy, dz = 1., 1., 1.
    ox, oy, oz = 0., 0., 0.

    dimension = [nx, ny, nz]
    spacing = [dx, dy, dy]
    origin = [ox, oy, oz]

    # Define covariance model
    cov_model = gcm.CovModel3D(elem=[
                    ('gaussian', {'w':8.5, 'r':[40, 20, 10]}), # elementary contribution
                    ('nugget', {'w':0.5})                      # elementary contribution
                    ], alpha=-30, beta=-45, gamma=20, name='')

    # Get covariance function and range
    cov_fun = cov_model.func()

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
    v = [ 8.,  9.,   8.,  12.] # values
    # x, v = None, None

    # Set minimal extension according to the size of the grid and the range
    extensionMin = [extension_min(r, n, s) for r, n, s in zip(cov_model.rxyz(), dimension, spacing)]

    # Set number of realizations
    nreal = 50

    # Set seed
    np.random.seed(123)

    # Generate GRF
    t1 = time.time()
    grf = grf3D(cov_fun, dimension, spacing, origin=origin,
                nreal=nreal, mean=mean, var=var,
                x=x, v=v,
                method=3, conditioningMethod=2,
                extensionMin=extensionMin) # grf: (nreal,nz,ny,nx) array
    t2 = time.time()
    nreal_case3D = nreal
    time_case3D = t2-t1
    infogrid_case3D = 'grid: {} cells ({} x {} x {})'.format(nx*ny*nz, nx, ny, nz)
    print('Elapsed time: {} sec'.format(time_case3D))

    # Compute mean and sd over the realizations
    grfMean = np.mean(grf.reshape(nreal, -1), axis=0).reshape(nz, ny, nx)
    grfSD = np.std(grf.reshape(nreal, -1), axis=0).reshape(nz, ny, nx)

    if x is not None:
        # Kriging
        t1 = time.time()
        krig, krigSD = krige3D(x, v, cov_fun, dimension, spacing, origin=origin,
                               mean=mean, var=var,
                               conditioningMethod=2,
                               extensionMin=extensionMin)
        t2 = time.time()
        time_krig_case3D = t2-t1
        print('Elapsed time for kriging: {} sec'.format(time_krig_case3D))

    # Display: slices going through the cell index ix0, iy0, iz0
    # -------
    ix0, iy0, iz0 = [30, 30, 30]
    # xg, yg, zg: center of grid points
    xg = ox + 0.5 * dx + dx * np.arange(nx)
    yg = oy + 0.5 * dy + dy * np.arange(ny)
    zg = oz + 0.5 * dz + dz * np.arange(nz)
    xmin, xmax = ox, ox + nx * dx
    ymin, ymax = oy, oy + ny * dy
    zmin, zmax = oz, oz + nz * dz

    # === first real and mean and sd of all real
    fig, ax = plt.subplots(3,3,figsize=(20,12))

    # first real ...
    # ... xy slice
    plt.subplot(3,3,1)
    im_plot = plt.imshow(grf[0,iz0,:,:],
                         origin='lower', extent=[xmin,xmax,ymin,ymax],
                         interpolation='none')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('GRF3D {}: real #{}, iz = {}'.format(cov_model.name, 1, iz0))

    # ... xz slice
    plt.subplot(3,3,2)
    im_plot = plt.imshow(grf[0,:,iy0,:],
                         origin='lower', extent=[xmin,xmax,zmin,zmax],
                         interpolation='none')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('GRF3D {}: real #{}, iy = {}'.format(cov_model.name, 1, iy0))

    # ... yz slice
    plt.subplot(3,3,3)
    im_plot = plt.imshow(grf[0,:,:,ix0],
                         origin='lower', extent=[ymin,ymax,zmin,zmax],
                         interpolation='none')
    plt.colorbar()
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('GRF3D {}: real #{}, ix = {}'.format(cov_model.name, 1, ix0))

    # mean...
    # ... xy slice
    plt.subplot(3,3,4)
    im_plot = plt.imshow(grfMean[iz0,:,:],
                         origin='lower', extent=[xmin,xmax,ymin,ymax],
                         interpolation='none')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Mean over {} real, iz = {}'.format(nreal, iz0))

    # ... xz slice
    plt.subplot(3,3,5)
    im_plot = plt.imshow(grfMean[:,iy0,:],
                         origin='lower', extent=[xmin,xmax,zmin,zmax],
                         interpolation='none')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Mean over {} real, iy = {}'.format(nreal, iy0))

    # ... yz slice
    plt.subplot(3,3,6)
    im_plot = plt.imshow(grfMean[:,:,ix0],
                         origin='lower', extent=[ymin,ymax,zmin,zmax],
                         interpolation='none')
    plt.colorbar()
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('Mean over {} real, ix = {}'.format(nreal, ix0))

    # sd...
    # ... xy slice
    plt.subplot(3,3,7)
    im_plot = plt.imshow(grfSD[iz0,:,:],
                         origin='lower', extent=[xmin,xmax,ymin,ymax],
                         interpolation='none')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('SD over {} real, iz = {}'.format(nreal, iz0))

    # ... xz slice
    plt.subplot(3,3,8)
    im_plot = plt.imshow(grfSD[:,iy0,:],
                         origin='lower', extent=[xmin,xmax,zmin,zmax],
                         interpolation='none')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('SD over {} real, iy = {}'.format(nreal, iy0))

    # ... yz slice
    plt.subplot(3,3,9)
    im_plot = plt.imshow(grfSD[:,:,ix0],
                         origin='lower', extent=[ymin,ymax,zmin,zmax],
                         interpolation='none')
    plt.colorbar()
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('SD over {} real, ix = {}'.format(nreal, ix0))

    # fig.show()
    plt.show()

    if x is not None:
        # === first real and kriging estimates and sd
        fig, ax = plt.subplots(3,3,figsize=(20,12))

        # first real ...
        # ... xy slice
        plt.subplot(3,3,1)
        im_plot = plt.imshow(grf[0,iz0,:,:],
                             origin='lower', extent=[xmin,xmax,ymin,ymax],
                             interpolation='none')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('GRF3D {}: real #{}, iz = {}'.format(cov_model.name, 1, iz0))

        # ... xz slice
        plt.subplot(3,3,2)
        im_plot = plt.imshow(grf[0,:,iy0,:],
                             origin='lower', extent=[xmin,xmax,zmin,zmax],
                             interpolation='none')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title('GRF3D {}: real #{}, iy = {}'.format(cov_model.name, 1, iy0))

        # ... yz slice
        plt.subplot(3,3,3)
        im_plot = plt.imshow(grf[0,:,:,ix0],
                             origin='lower', extent=[ymin,ymax,zmin,zmax],
                             interpolation='none')
        plt.colorbar()
        plt.xlabel('y')
        plt.ylabel('z')
        plt.title('GRF3D {}: real #{}, ix = {}'.format(cov_model.name, 1, ix0))

        # Kriging..
        # ... xy slice
        plt.subplot(3,3,4)
        im_plot = plt.imshow(krig[iz0,:,:],
                             origin='lower', extent=[xmin,xmax,ymin,ymax],
                             interpolation='none')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Kriging, iz = {}'.format(iz0))

        # ... xz slice
        plt.subplot(3,3,5)
        im_plot = plt.imshow(krig[:,iy0,:],
                             origin='lower', extent=[xmin,xmax,zmin,zmax],
                             interpolation='none')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title('Kriging, iy = {}'.format(iy0))

        # ... yz slice
        plt.subplot(3,3,6)
        im_plot = plt.imshow(krig[:,:,ix0],
                             origin='lower', extent=[ymin,ymax,zmin,zmax],
                             interpolation='none')
        plt.colorbar()
        plt.xlabel('y')
        plt.ylabel('z')
        plt.title('Kriging, ix = {}'.format(ix0))

        # Kriging sd...
        # ... xy slice
        plt.subplot(3,3,7)
        im_plot = plt.imshow(krigSD[iz0,:,:],
                             origin='lower', extent=[xmin,xmax,ymin,ymax],
                             interpolation='none')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Kriging SD, iz = {}'.format(iz0))

        # ... xz slice
        plt.subplot(3,3,8)
        im_plot = plt.imshow(krigSD[:,iy0,:],
                             origin='lower', extent=[xmin,xmax,zmin,zmax],
                             interpolation='none')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title('Kriging SD, iy = {}'.format(iy0))

        # ... yz slice
        plt.subplot(3,3,9)
        im_plot = plt.imshow(krigSD[:,:,ix0],
                             origin='lower', extent=[ymin,ymax,zmin,zmax],
                             interpolation='none')
        plt.colorbar()
        plt.xlabel('y')
        plt.ylabel('z')
        plt.title('Kriging SD, ix = {}'.format(ix0))

        # fig.show()
        plt.show()

        # === comparison of mean and sd of all real, with kriging estimates and sd
        fig, ax = plt.subplots(2,3,figsize=(20,10))

        # grfMean - krig ...
        # ... xy slice
        plt.subplot(2,3,1)
        im_plot = plt.imshow(grfMean[iz0,:,:]-krig[iz0,:,:],
                             origin='lower', extent=[xmin,xmax,ymin,ymax],
                             interpolation='none')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('3D: grfMean - krig / nreal={} / iz = {}'.format(nreal, iz0))

        # ... xz slice
        plt.subplot(2,3,2)
        im_plot = plt.imshow(grfMean[:,iy0,:]-krig[:,iy0,:],
                             origin='lower', extent=[xmin,xmax,zmin,zmax],
                             interpolation='none')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title('3D: grfMean - krig / nreal={} / iy = {}'.format(nreal, iy0))

        # ... yz slice
        plt.subplot(2,3,3)
        im_plot = plt.imshow(grfMean[:,:,ix0]-krig[:,:,ix0],
                             origin='lower', extent=[ymin,ymax,zmin,zmax],
                             interpolation='none')
        plt.colorbar()
        plt.xlabel('y')
        plt.ylabel('z')
        plt.title('3D: grfMean - krig / nreal={} / ix = {}'.format(nreal, ix0))

        # grfSD - krigSD ...
        # ... xy slice
        plt.subplot(2,3,4)
        im_plot = plt.imshow(grfSD[iz0,:,:]-krigSD[iz0,:,:],
                             origin='lower', extent=[xmin,xmax,ymin,ymax],
                             interpolation='none')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('3D: grfSD - krigSD / nreal={} / iz = {}'.format(nreal, iz0))

        # ... xz slice
        plt.subplot(2,3,5)
        im_plot = plt.imshow(grfSD[:,iy0,:]-krigSD[:,iy0,:],
                             origin='lower', extent=[xmin,xmax,zmin,zmax],
                             interpolation='none')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title('3D: grfSD - krigSD / nreal={} / iy = {}'.format(nreal, iy0))

        # ... yz slice
        plt.subplot(2,3,6)
        im_plot = plt.imshow(grfSD[:,:,ix0]-krigSD[:,:,ix0],
                             origin='lower', extent=[ymin,ymax,zmin,zmax],
                             interpolation='none')
        plt.colorbar()
        plt.xlabel('y')
        plt.ylabel('z')
        plt.title('3D: grfSD - krigSD / nreal={} / ix = {}'.format(nreal, ix0))

        # fig.show()
        plt.show()

        print('Peak to peak for "grfMean - krig": {}'.format(np.ptp(grfMean-krig)))
        print('Peak to peak for "grfSD - krigSD": {}'.format(np.ptp(grfSD-krigSD)))

        del(krig, krigSD)

    del (grf, grfMean, grfSD)

    ######### Elapsed time for all cases ##########
    print('Case 1D: elapsed time: {:5.2f} sec  ({} real, {})'.format(time_case1D, nreal_case1D, infogrid_case1D))
    print('Case 2D: elapsed time: {:5.2f} sec  ({} real, {})'.format(time_case2D, nreal_case2D, infogrid_case2D))
    print('Case 3D: elapsed time: {:5.2f} sec  ({} real, {})'.format(time_case3D, nreal_case3D, infogrid_case3D))
    print('Kriging Case 1D: elapsed time: {:5.2f} sec'.format(time_krig_case1D))
    print('Kriging Case 2D: elapsed time: {:5.2f} sec'.format(time_krig_case2D))
    print('Kriging Case 3D: elapsed time: {:5.2f} sec'.format(time_krig_case3D))

    ######### END ##########
    a = input("Press enter to continue...")
