#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# Python module:  'markovChain.py'
# author:         Julien Straubhaar
# date:           sep-2024
# -------------------------------------------------------------------------

"""
Module for simulation of Markov Chain on finite sets of states.
"""

import numpy as np

# ============================================================================
class MarkovChainError(Exception):
    """
    Custom exception related to `markovChain` module.
    """
    pass
# ============================================================================

# ----------------------------------------------------------------------------
def mc_kernel1(n, p, return_pinv=False):
    """
    Sets the following symmetric transition kernel of order n for a Markov chain:

    .. math::
        P = \\left(\\begin{array}{cccc}
            p                & \\frac{1-p}{n-1}  & \\ldots          & \\frac{1-p}{n-1}\\\\
            \\frac{1-p}{n-1}  & \\ddots           & \\ddots          & \\vdots\\\\
            \\vdots           & \\ddots           & \\ddots          & \\frac{1-p}{n-1}\\\\
            \\frac{1-p}{n-1}  & \\ldots           & \\frac{1-p}{n-1} & p
            \\end{array}\\right)

    where :math:`0\\leqslant p < 1` is a parameter.

    Parameters
    ----------
    n : int
        order of the kernel, number of states

    p : float
        number in the interval [0, 1[

    return_pinv : bool
        indicates if the invariant distribution is returned

    Returns
    -------
    kernel : 2d-array of shape (n, n)
        transition kernel (P) above

    pinv : (1d-array of shape (n,)
        invariant distibution of the kernel, that is

        - [1/n, ... , 1/n]

        returned if `return_pinv=True`
    """
    # fname = 'mc_kernel1'

    if p < 0 or p >= 1:
        return None
    r = (1.0 - p)/(n-1)
    x = [p] + (n-1)*[r]
    kernel = np.array([x[-i:] + x[0:-i] for i in range(n)])
    if return_pinv:
        pinv = 1/n * np.ones(n)
    if return_pinv:
        out = kernel, pinv
    else:
        out = kernel
    return out
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def mc_kernel2(n, p, return_pinv=False):
    """
    Sets the following transition kernel of order n for a Markov chain:

    .. math::
        P = \\left(\\begin{array}{cccccc}
            p              & 1-p     & 0              & \\ldots        & 0       & 0\\\\
            \\frac{1-p}{2} & p       & \\frac{1-p}{2} & \\ddots        &         & 0\\\\
            0              & \\ddots & \\ddots        & \\ddots        & \\ddots & \\vdots\\\\
            \\vdots        & \\ddots & \\ddots        & \\ddots        & \\ddots & 0\\\\
            0              &         & \\ddots        & \\frac{1-p}{2} &  p      & \\frac{1-p}{2}\\\\
            0              & 0       & \\ldots        &     0          &  1-p   & p
            \\end{array}\\right)

    where :math:`0\\leqslant p < 1` is a parameter.

    Parameters
    ----------
    n : int
        order of the kernel, number of states

    p : float
        number in the interval [0, 1[

    return_pinv : bool
        indicates if the invariant distribution is returned

    Returns
    -------
    kernel : 2d-array of shape (n, n)
        transition kernel (P) above

    pinv : (1d-array of shape (n,)
        invariant distibution of the kernel, that is

        - [1/(2(n-1)), 1/(n-1), ... , 1/(n-1), 1/(2(n-1))]

        returned if `return_pinv=True`
    """
    # fname = 'mc_kernel2'

    if p < 0 or p >= 1:
        return None
    r = (1.0 - p)/2.0
    xa = [p, 2*r] + (n-2)*[0]
    x = [r, p, r] + (n-3)*[0]
    xb = (n-2)*[0] + [2*r, p]
    kernel = np.array([xa] + [x[-i:] + x[0:-i] for i in range(n-2)] + [xb])
    if return_pinv:
        t = 1.0 / (2.0*(n-1))
        pinv = np.hstack((np.array([t]), 1/(n-1) * np.ones(n-2), np.array([t])))
    if return_pinv:
        out = kernel, pinv
    else:
        out = kernel
    return out
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def mc_kernel3(n, p, q, return_pinv=False):
    """
    Sets the following transition kernel of order n for a Markov chain:

    .. math::
        P = \\left(\\begin{array}{ccccc}
            p          & (1-p)q  & 0         & \\ldots     & (1-p)(1-q)\\\\
            (1-p)(1-q) & \\ddots & \\ddots   &             & 0\\\\
            0          & \\ddots & \\ddots   & \\ddots     & \\vdots\\\\
            \\vdots    & \\ddots & \\ddots   & \\ddots     & 0\\\\
            0          &         & \\ddots   & \\ddots     & (1-p)q\\\\
            (1-p)q     & 0       & \\ldots   & (1-p)(1-q)  & p
            \\end{array}\\right)

    where :math:`0\\leqslant p < 1` and :math:`0\\leqslant q \\leqslant 1` are
    two parameters.

    Parameters
    ----------
    n : int
        order of the kernel, number of states

    p : float
        number in the interval [0, 1[

    q : float
        number in the interval [0, 1]

    return_pinv : bool
        indicates if the invariant distribution is returned

    Returns
    -------
    kernel : 2d-array of shape (n, n)
        transition kernel (P) above

    pinv : (1d-array of shape (n,)
        invariant distibution of the kernel, that is

        - [1/n, ... , 1/n]

        returned if `return_pinv=True`
    """
    # fname = 'mc_kernel3'

    if p < 0 or p >= 1 or q < 0 or q > 1:
        return None
    r = (1.0 - p)*q
    s = (1.0 - p)*(1.0 - q)
    x = [p, r] + (n-3)*[0] + [s]
    kernel = np.array([x[-i:] + x[0:-i] for i in range(n)])
    if return_pinv:
        pinv = 1/n * np.ones(n)
    if return_pinv:
        out = kernel, pinv
    else:
        out = kernel
    return out
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def mc_kernel4(n, p, q, return_pinv=False):
    """
    Sets the following transition kernel of order n for a Markov chain:

    .. math::
        P = \\left(\\begin{array}{ccccc}
            q                & 0       & \\ldots & 0               & 1-q\\\\
            0                & \\ddots & \\ddots & \\vdots         & \\vdots\\\\
            \\vdots          & \\ddots & \\ddots & 0               & \\vdots\\\\
            0                & \\ldots &  0      & q               & 1-q\\\\
            \\frac{1-p}{n-1} & \\ldots & \\ldots &\\frac{1-p}{n-1} & p
            \\end{array}\\right)

    where :math:`0\\leqslant p, q < 1` are two parameters.

    Parameters
    ----------
    n : int
        order of the kernel, number of states

    p : float
        number in the interval [0, 1[

    q : float
        number in the interval [0, 1[

    return_pinv : bool
        indicates if the invariant distribution is returned

    Returns
    -------
    kernel : 2d-array of shape (n, n)
        transition kernel (P) above

    pinv : (1d-array of shape (n,)
        invariant distibution of the kernel, that is

        - 1/(2-p-q) * [(1-p)/(n-1), ... , (1-p)/(n-1), 1-q]

        returned if `return_pinv=True`
    """
    # fname = 'mc_kernel4'

    if p < 0 or p >= 1 or q < 0 or q >= 1:
        return None
    kernel = np.vstack((np.hstack((np.diag(q*np.ones(n-1)), (1-q)*np.ones((n-1, 1)))),[(n-1)*[(1-p)/(n-1)]+[p]]))
    if return_pinv:
        pinv = 1/(2-p-q)*np.array((n-1)*[(1-p)/(n-1)] + [1-q])
    if return_pinv:
        out = kernel, pinv
    else:
        out = kernel
    return out
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def compute_mc_pinv(kernel, logger=None):
    """
    Computes the invariant distribution of a Markov chain for a given kernel.

    Parameters
    ----------
    kernel : 2d-array of shape (n, n)
        transition kernel of a Markov chain on a set of states
        :math:`S=\\{0, \\ldots, n-1\\}`; the element at row `i` and column `j` is
        the probability to have the state of index `j` at the next step given
        the state `i` at the current step, i.e.

        - :math:`kernel[i][j] = P(X_{k+1}=j\\ \\vert\\ X_{k}=i)`

        where the sequence of random variables :math:`(X_k)` is a Markov chain
        on `S` defined by the kernel `kernel`.

        In particular, every element of `kernel` is positive or zero, and its
        rows sum to one.

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    pinv : 1d-array of shape (n,)
        invariant distribution of the Markov chain, i.e. `pinv` is an eigen
        vector of eigen value `1` of the transpose of `kernel`:
        :math:`pinv \\cdot kernel= pinv`; note:

        - `None` is returned if no eigen value is equal to 1
        - if more than one eigen value is equal to 1, only the first corresponding\
        eigen vector computed by `numpy.linalg.eig` is returned
    """
    fname = 'compute_mc_pinv'

    valt, st = np.linalg.eig(kernel.T)
    ind = np.where(np.isclose(valt, 1.0))[0]
    if len(ind) == 0:
        err_msg = f'{fname}: no invariant distribution found'
        if logger: logger.error(err_msg)
        raise MarkovChainError(err_msg)

    pinv = np.real(st[:,ind[0]]/np.sum(st[:,ind[0]]))
    return pinv
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def compute_mc_kernel_rev(kernel, pinv=None, logger=None):
    """
    Computes the reverse transition kernel of a Markov chain for a given kernel.

    Parameters
    ----------
    kernel : 2d-array of shape (n, n)
        transition kernel of a Markov chain on a set of states
        :math:`S=\\{0, \\ldots, n-1\\}`; the element at row `i` and column `j` is
        the probability to have the state of index `j` at the next step given
        the state `i` at the current step, i.e.

        - :math:`kernel[i][j] = P(X_{k+1}=j\\ \\vert\\ X_{k}=i)`

        where the sequence of random variables :math:`(X_k)` is a Markov chain
        on `S` defined by the kernel `kernel`.

        In particular, every element of `kernel` is positive or zero, and its
        rows sum to one.

    pinv : 1d-array of shape (n,), optional
        invariant distribution of the Markov chain;
        by default (`None`): `pinv` is automatically computed

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    kernel_rev : 2d-array of shape (n, n)
        reverse transition kernel of the Markov chain, i.e.

        - :math:`kernel\\_rev[i, j] = pinv[i]^{-1} \\cdot kernel[j, i] \\cdot pinv[j]`
    """
    fname = 'compute_mc_kernel_rev'
    
    if pinv is None:
        try:
            pinv = compute_mc_pinv(kernel, logger=logger)
        except Exception as exc:
            err_msg = f'{fname}: computing invariant distribution failed'
            if logger: logger.error(err_msg)
            raise MarkovChainError(err_msg) from exc

    if pinv is None or np.any(np.isclose(pinv, 0)):
        err_msg = f'{fname}: kernel not reversible'
        if logger: logger.error(err_msg)
        raise MarkovChainError(err_msg)

    kernel_rev = (kernel/pinv).T*pinv

    return kernel_rev
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def compute_mc_cov(kernel, pinv=None, nsteps=1, logger=None):
    """
    Computes covariances of indicators for a Markov chain, accross time steps.

    Parameters
    ----------
    kernel : 2d-array of shape (n, n)
        transition kernel of a Markov chain on a set of states
        :math:`S=\\{0, \\ldots, n-1\\}`; the element at row `i` and column `j` is
        the probability to have the state of index `j` at the next step given
        the state `i` at the current step, i.e.

        - :math:`kernel[i][j] = P(X_{k+1}=j\\ \\vert\\ X_{k}=i)`

        where the sequence of random variables :math:`(X_k)` is a Markov chain
        on `S` defined by the kernel `kernel`.

        In particular, every element of `kernel` is positive or zero, and its
        rows sum to one.

    pinv : 1d-array of shape (n,), optional
        invariant distribution of the Markov chain;
        by default (`None`): `pinv` is automatically computed

    nsteps : int, default: 1
        number of (time) steps for which the covariance is computed

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    mc_cov : 3d-array of shape (nstep, n, n)
        covariances of indicators for a Markov chain X built from the kernel
        and its invariant distribution:

        - :math:`mc\\_cov[l, i, j] = \\operatorname{Cov}(X_k=i, X_{k+l}=j) = pinv[i] \\cdot ((kernel^l)[i,j]- pinv[j])`

        for `l=0, \\ldots, nsteps-1`, and :math:`0\\leqslant i, j \\leqslant n-1`
    """
    fname = 'compute_mc_cov'

    # Diagonalization of kernel: kernel = s.dot(diag(val)).dot(sinv)
    val, s = np.linalg.eig(kernel)
    val = np.real(val) # take the real part (if needed)
    s = np.real(s)     # take the real part (if needed)
    sinv = np.linalg.inv(s)

    if pinv is None:
        try:
            pinv = compute_mc_pinv(kernel, logger=logger)
        except Exception as exc:
            err_msg = f'{fname}: computing invariant distribution failed'
            if logger: logger.error(err_msg)
            raise MarkovChainError(err_msg) from exc

    n = kernel.shape[0]
    mc_cov = np.zeros((nsteps, n, n))
    for m in range(nsteps):
        km = s.dot(np.diag(val**m)).dot(sinv)
        mc_cov[m, :, :] = np.repeat(pinv, n).reshape(n, n)*km - np.outer(pinv, pinv)

    return mc_cov
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def simulate_mc(
        kernel,
        nsteps,
        categVal=None,
        data_ind=None, data_val=None,
        pstep0=None,
        pinv=None,
        kernel_rev=None, kernel_pow=None,
        nreal=1,
        logger=None):
    """
    Generates (conditional) Markov chains for a given kernel.

    Parameters
    ----------
    kernel : 2d-array of shape (n, n)
        transition kernel of a Markov chain on a set of states
        :math:`S=\\{0, \\ldots, n-1\\}`; the element at row `i` and column `j` is
        the probability to have the state of index `j` at the next step given
        the state `i` at the current step, i.e.

        - :math:`kernel[i][j] = P(X_{k+1}=j\\ \\vert\\ X_{k}=i)`

        where the sequence of random variables :math:`(X_k)` is a Markov chain
        on `S` defined by the kernel `kernel`.

        In particular, every element of `kernel` is positive or zero, and its
        rows sum to one.

    nsteps : int
        length of each generated chain, number of time steps

    categVal :1d-array of shape (n,), optional
        values of categories (one value for each state `0, ..., n-1`);
        by default (`None`) : `categVal` is set to `[0, ..., n-1]`

    data_ind : sequence, optional
        index (time step) of conditioning locations; each index should be in
        `{0, 1, ..., nsteps-1}`

    data_val : sequence, optional
        values at conditioning locations (same length as `data_ind`); each value
        should be in `categVal`

    pstep0 : 1d-array of shape (n,), optional
        distribution for step 0 of the chain, for unconditional simulation
        (used only if `data_ind=None` and `data_val=None`)
        by default (`None`): `pinv` is used (see below)

    pinv : 1d-array of shape (n,), optional
        invariant distribution of the Markov chain;
        by default (`None`): `pinv` is automatically computed

    kernel_rev : 2d-array of shape (n, n), optional
        reverse transition kernel of the Markov chain;
        by default (`None`): `kernel_rev` is automatically computed;
        note: only used for conditional simulation

    kernel_pow : 3d-array of shape (m, n, n), optional
        pre-computed kernel raised to power 0, 1, ..., m-1:

        - :math:`kernel\\_pow[k] = kernel^k`

        note: only used for conditional simulation

    nreal : int, default: 1
        number of realization(s), number of generated chain(s)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    x : 3d-array of shape (nreal, nsteps)
        generated Markov chain (conditional to `data_ind, data_val` if present), `x[i]` is
        the i-th realization
    """
    fname = 'simulate_mc'

    # Number of categories (order of the kernel)
    n = kernel.shape[0]

    # Check category values
    if categVal is None:
        categVal = np.arange(n)
    else:
        categVal = np.asarray(categVal)
        if categVal.ndim != 1 or categVal.shape[0] != n:
            err_msg = f'{fname}: `categVal` invalid'
            if logger: logger.error(err_msg)
            raise MarkovChainError(err_msg)

        if len(np.unique(categVal)) != len(categVal):
            err_msg = f'{fname}: `categVal` contains duplicated values'
            if logger: logger.error(err_msg)
            raise MarkovChainError(err_msg)

    # Conditioning
    if (data_ind is None and data_val is not None) or (data_ind is not None and data_val is None):
        err_msg = f'{fname}: `data_ind` and `data_val` must both be specified'
        if logger: logger.error(err_msg)
        raise MarkovChainError(err_msg)

    if data_ind is None:
        nhd = 0
    else:
        data_ind = np.asarray(data_ind, dtype='int').reshape(-1) # cast in 1-dimensional array if needed
        data_val = np.asarray(data_val, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        nhd = len(data_ind)
        if len(data_ind) != len(data_val):
            err_msg = f'{fname}: length of `data_ind` and length of `data_val` differ'
            if logger: logger.error(err_msg)
            raise MarkovChainError(err_msg)

        # Check values
        if not np.all([xv in categVal for xv in data_val]):
            err_msg = f'{fname}: `data_val` contains an invalid value'
            if logger: logger.error(err_msg)
            raise MarkovChainError(err_msg)

        # Replace values by their index in categVal
        data_val = np.array([np.where(categVal==xv)[0][0] for xv in data_val], dtype='int')

    if nhd == 0:
        # Compute invariant distribution for kernel
        if pinv is None:
            try:
                pinv = compute_mc_pinv(kernel, logger=logger)
            except Exception as exc:
                err_msg = f'{fname}: computing invariant distribution failed'
                if logger: logger.error(err_msg)
                raise MarkovChainError(err_msg) from exc

        # Compute cdf for x0
        if pstep0 is not None:
            x0_cdf = np.cumsum(pstep0)
        else:
            x0_cdf = np.cumsum(pinv)

        # Compute conditional cdf from the transition kernel
        kernel_cdf = np.cumsum(kernel, axis=1)

        # Generate X
        x = []
        for _ in range(nreal):
            # Generate one chain (realization)
            xk = - np.ones(nsteps, dtype='int')
            u = np.random.random(size=nsteps)
            xk[0] = np.where(u[0] < x0_cdf)[0][0]
            for i in range(1, nsteps):
                xk[i] = np.where(u[i] < kernel_cdf[int(xk[i-1])])[0][0]
            # Append realization to x
            x.append(xk)

    else:
        inds = np.argsort(data_ind)
        if data_ind[inds[0]] > 0:
            # Compute reverse transition kernel
            if kernel_rev is None:
                try:
                    kernel_rev = compute_mc_kernel_rev(kernel, pinv=pinv, logger=logger)
                except Exception as exc:
                    err_msg = f'{fname}: computing reverse transition kernel failed'
                    if logger: logger.error(err_msg)
                    raise MarkovChainError(err_msg) from exc

            # Compute conditional cdf from the reverse transition kernel
            kernel_rev_cdf = np.cumsum(kernel_rev, axis=1)

        if data_ind[inds[-1]] < nsteps-1:
            # compute conditional cdf from the transition kernel
            kernel_cdf = np.cumsum(kernel, axis=1)

        if nhd > 1:
            # Compute list of kernel raise to power 0, 1, 2, ..., m-1 for further computation
            m = max(np.diff([data_ind[j] for j in inds])) + 1
            if kernel_pow is None:
                kernel_pow = np.zeros((m, n, n))
                kernel_pow[0] = np.eye(n)
                m0 = 1
            else:
                m0 = kernel_pow.shape[0]
                if m0 < m:
                    kernel_pow = np.concatenate((kernel_pow, np.zeros((m-m0, n, n))), axis=0)
            for i in range(m0, m):
                kernel_pow[i] = kernel_pow[i-1].dot(kernel)
            #
            # With k1 < k2 < k3, we have:
            #     Prob(x[k2]=i2 | x[k1]=i1, x[k3]=i3) = kernel^(k2-k1)[i1, i2] * kernel^(k3-k2)[i2, i3] / kernel^(k3-k1)[i1, i3]
            # Check validity of conditioning points:
            #    check that the denominator above is positive for each pair (k1, k3) of consecutive conditioning points, i.e.
            #       Prob(x[k1]=i1 | x[k3]=i3) = kernel^(k3-k1)[i1, i3] > 0
            if np.any(np.isclose([kernel_pow[data_ind[inds[i+1]]-data_ind[inds[i]], int(data_val[inds[i]]), int(data_val[inds[i+1]])] for i in range(nhd-1)], 0)):
            # if np.any([kernel_pow[data_ind[inds[i+1]]-data_ind[inds[i]], int(data_val[inds[i]]), int(data_val[inds[i+1]])] < 1.e-20 for i in range(nhd-1)]):
                err_msg = f'{fname}: invalid conditioning points wrt. kernel'
                if logger: logger.error(err_msg)
                raise MarkovChainError(err_msg)

        # Generate X
        x = []
        for _ in range(nreal):
            # Generate one chain (realization)
            # Initialization
            xk = - np.ones(nsteps, dtype='int')
            xk[data_ind] = data_val
            #
            # Random numbers in [0,1[
            u = np.random.random(size=nsteps)
            #
            # Simulate in reverse order the values before the first conditioning point (sorted)
            for i in range(data_ind[inds[0]]-1, -1, -1):
                xk[i] = np.where(u[i] < kernel_rev_cdf[int(xk[i+1])])[0][0]
            # Simulate the values between the pairs of consecutive conditioning points (sorted)
            # With k1 < k2 < k3, we have:
            #     Prob(x[k2]=i2 | x[k1]=i1, x[k3]=i3) = kernel^(k2-k1)[i1, i2] * kernel^(k3-k2)[i2, i3] / kernel^(k3-k1)[i1, i3]
            for ii in range(nhd-1):
                # Simulate the values between the ii-th and (ii+1)-th conditioning points (sorted)
                xend_ind = data_ind[inds[ii+1]]
                xend_val = int(data_val[inds[ii+1]])
                for i in range(data_ind[inds[ii]]+1, data_ind[inds[ii+1]]):
                    prob = np.array([kernel_pow[1, int(xk[i-1]), j]*kernel_pow[xend_ind-i, j, xend_val]/kernel_pow[xend_ind-i+1, int(xk[i-1]), xend_val] for j in range(n)])
                    cdf = np.cumsum(prob)
                    # if i==11:
                    #     print(k, i, kernel_pow[xend_ind-i+1, int(xk[i-1]), xend_val], prob, cdf)
                    xk[i] = np.where(u[i] < cdf)[0][0]
            # Simulate the values after the last conditioning point (sorted)
            for i in range(data_ind[inds[-1]]+1, nsteps):
                xk[i] = np.where(u[i] < kernel_cdf[int(xk[i-1])])[0][0]

            # Append realization to x
            x.append(xk)

    x = np.asarray(x)

    # Set original values
    x = categVal[x]

    return x
# ----------------------------------------------------------------------------
