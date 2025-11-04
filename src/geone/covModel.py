#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# Python module:  'covModel.py'
# authors:        Julien Straubhaar and Philippe Renard
# date:           2018-2024
# -------------------------------------------------------------------------

"""
Module for:

- definition of covariance / variogram models in 1D, 2D, and 3D \
(omni-directional or anisotropic)
- covariance / variogram analysis and fitting
- ordinary kriging
- cross-validation (leave-one-out (loo))
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.optimize
from scipy import stats
import pyvista as pv
import copy
import multiprocessing

from geone import img
from geone import imgplot as imgplt
from geone import imgplot3d as imgplt3

# ============================================================================
class CovModelError(Exception):
    """
    Custom exception related to `covModel` module.
    """
    pass
# ============================================================================

# ============================================================================
# Definition of 1D elementary covariance models:
#   - nugget, spherical, exponential, gaussian, triangular, cubic, sinus_cardinal
#       parameters: w, r
#   - gamma, exponential_generalized, power (non-stationary)
#       parameters: w, r, s (power)
#   - matern
#       parameters: w, r, nu
# ============================================================================
# ----------------------------------------------------------------------------
def cov_nug(h, w=1.0):
    """
    1D-nugget covariance model.

    Function `v = w * f(h)`, where

    * f(h) = 1, if h=0
    * f(h) = 0, otherwise

    Parameters
    ----------
    h : 1D array-like of floats, or float
        value(s) (lag(s)) where the covariance model is evaluated

    w : float, default: 1.0
        weight (sill), should be positive

    Returns
    -------
    v : 1D array of floats, or float
        evaluation of the covariance model at `h` (see above)
    """
    # fname = 'cov_nug'
    return w * np.asarray(h==0., dtype=float)

def cov_sph(h, w=1.0, r=1.0):
    """
    1D-spherical covariance model.

    Function `v = w * f(|h|/r)`, where

    * f(t) = 1 - 3/2 * t + 1/2 * t**3, if 0 <= t < 1
    * f(t) = 0,                        if t >= 1

    Parameters
    ----------
    h : 1D array-like of floats, or float
        value(s) (lag(s)) where the covariance model is evaluated

    w : float, default: 1.0
        weight (sill), should be positive

    r : float, default: 1.0
        range, should be positive

    Returns
    -------
    v : 1D array of floats, or float
        evaluation of the covariance model at `h` (see above)
    """
    # fname = 'cov_sph'
    t = np.minimum(np.abs(h)/r, 1.) # "parallel or element-wise minimum"
    return w * (1 - 0.5 * t * (3. - t**2))

def cov_exp(h, w=1.0, r=1.0):
    """
    1D-exponential covariance model.

    Function `v = w * f(|h|/r)`, where

    * f(t) = exp(-3*t)

    Parameters
    ----------
    h : 1D array-like of floats, or float
        value(s) (lag(s)) where the covariance model is evaluated

    w : float, default: 1.0
        weight (sill), should be positive

    r : float, default: 1.0
        range, should be positive

    Returns
    -------
    v : 1D array of floats, or float
        evaluation of the covariance model at `h` (see above)
    """
    # fname = 'cov_exp'
    return w * np.exp(-3. * np.abs(h)/r)

def cov_gau(h, w=1.0, r=1.0):
    """
    1D-gaussian covariance model.

    Function `v = w * f(|h|/r)`, where

    * f(t) = exp(-3*t**2)

    Parameters
    ----------
    h : 1D array-like of floats, or float
        value(s) (lag(s)) where the covariance model is evaluated

    w : float, default: 1.0
        weight (sill), should be positive

    r : float, default: 1.0
        range, should be positive

    Returns
    -------
    v : 1D array of floats, or float
        evaluation of the covariance model at `h` (see above)
    """
    # fname = 'cov_gau'
    return w * np.exp(-3. * (h/r)**2)

def cov_tri(h, w=1.0, r=1.0):
    """
    1D-triangular covariance model.

    Function `v = w * f(|h|/r)`, where

        * f(t) = 1 - t, if 0 <= t < 1
        * f(t) = 0,     if t >= 1

    Parameters
    ----------
    h : 1D array-like of floats, or float
        value(s) (lag(s)) where the covariance model is evaluated

    w : float, default: 1.0
        weight (sill), should be positive

    r : float, default: 1.0
        range, should be positive

    Returns
    -------
    v : 1D array of floats, or float
        evaluation of the covariance model at `h` (see above)
    """
    # fname = 'cov_tri'
    t = np.minimum(np.abs(h)/r, 1.) # "parallel or element-wise minimum"
    return w * (1.0 - t)

def cov_cub(h, w=1.0, r=1.0):
    """
    1D-cubic covariance model.

    Function `v = w * f(|h|/r)`, where

    * f(t) = 1 - 7 * t**2 + 35/4 * t**3 - 7/2 * t**5 + 3/4 * t**7, if 0 <= t < 1
    * f(t) = 0,                                                    if t >= 1

    Parameters
    ----------
    h : 1D array-like of floats, or float
        value(s) (lag(s)) where the covariance model is evaluated

    w : float, default: 1.0
        weight (sill), should be positive

    r : float, default: 1.0
        range, should be positive

    Returns
    -------
    v : 1D array of floats, or float
        evaluation of the covariance model at `h` (see above)
    """
    # fname = 'cov_cub'
    t = np.minimum(np.abs(h)/r, 1.) # "parallel or element-wise minimum"
    t2 = t**2
    return w * (1 + t2 * (-7. + t * (8.75 + t2 * (-3.5 + 0.75 * t2))))

def cov_sinc(h, w=1.0, r=1.0):
    """
    1D-sinus-cardinal covariance model.

    Function `v = w * f(|h|/r)`, where

    * f(t) = sin(pi*t)/(pi*t)

    Parameters
    ----------
    h : 1D array-like of floats, or float
        value(s) (lag(s)) where the covariance model is evaluated

    w : float, default: 1.0
        weight (sill), should be positive

    r : float, default: 1.0
        range, should be positive

    Returns
    -------
    v : 1D array of floats, or float
        evaluation of the covariance model at `h` (see above)
    """
    # fname = 'cov_sinc'
    # np.sinc(x) = np.sin(np.pi*x)/(np.pi*x)
    return w * np.sinc(h/r)

def cov_gamma(h, w=1.0, r=1.0, s=1.0):
    """
    1D-gamma covariance model.

    Function `v = w * f(|h|/r)`, where

    * f(t) = 1 / (1 + alpha*t)**s, with alpha = 20**(1/s) - 1

    Parameters
    ----------
    h : 1D array-like of floats, or float
        value(s) (lag(s)) where the covariance model is evaluated

    w : float, default: 1.0
        weight (sill), should be positive

    r : float, default: 1.0
        range, should be positive

    s : float, default: 1.0
        power

    Returns
    -------
    v : 1D array of floats, or float
        evaluation of the covariance model at `h` (see above)
    """
    # fname = 'cov_gamma'
    alpha = 20.0**(1.0/s) - 1.0
    return w / (1.0 + alpha * np.abs(h)/r)**s

def cov_pow(h, w=1.0, r=1.0, s=1.0):
    """
    1D-power covariance model.

    Function `v = w * f(|h|/r)`, where

    * f(t) = 1 - t**s

    Parameters
    ----------
    h : 1D array-like of floats, or float
        value(s) (lag(s)) where the covariance model is evaluated

    w : float, default: 1.0
        weight (sill), should be positive

    r : float, default: 1.0
        range, should be positive

    s : float, default: 1.0
        power

    Returns
    -------
    v : 1D array of floats, or float
        evaluation of the covariance model at `h` (see above)
    """
    # fname = 'cov_pow'
    return w * (1. - (np.abs(h)/r)**s)

def cov_exp_gen(h, w=1.0, r=1.0, s=1.0):
    """
    1D-exponential-generalized covariance model.

    Function `v = w * f(|h|/r)`, where

    * f(t) = exp(-3*t**s)

    Parameters
    ----------
    h : 1D array-like of floats, or float
        value(s) (lag(s)) where the covariance model is evaluated

    w : float, default: 1.0
        weight (sill), should be positive

    r : float, default: 1.0
        range, should be positive

    s : float, default: 1.0
        power

    Returns
    -------
    v : 1D array of floats, or float
        evaluation of the covariance model at `h` (see above)
    """
    # fname = 'cov_exp_gen'
    return w * np.exp(-3. * (np.abs(h)/r)**s)

def cov_matern(h, w=1.0, r=1.0, nu=0.5):
    """
    1D-Matern covariance model (the effective range depends on `nu`).

    Function

    * `v = w * 1.0/(2.0**(nu-1.0)*Gamma(nu)) * u**nu * K_{nu}(u)`

    where

    * `u = np.sqrt(2.0*nu)/r * |h|`
    * Gamma is the function gamma
    * `K_{nu}` is the modified Bessel function of the second kind of \
    parameter `nu`

    Parameters
    ----------
    h : 1D array-like of floats, or float
        value(s) (lag(s)) where the covariance model is evaluated

    w : float, default: 1.0
        weight (sill), should be positive

    r : float, default: 1.0
        parameter "r" (scale) of the Matern covariance, should be positive

    nu : float, default: 0.5
        parameter "nu" of the Matern covariance model

    Returns
    -------
    v : 1D array of floats, or float
        evaluation of the covariance model at `h` (see above)

    Notes
    -----
    1. `cov_matern(h, w, r, nu=0.5) = cov_exp(h, w, 3*r)`
    2. `cov_matern(h, w, r, nu)` tends to `cov_gau(h, w, np.sqrt(6)*r)` when `nu` \
    tends to infinity
    """
    # fname = 'cov_matern'
    v = np.zeros_like(h).astype(float) # be sure that the type is float to avoid truncation
    u = np.sqrt(2.0*nu)/r * np.abs(h)
    u1 = (0.5*u)**nu
    u2 = scipy.special.kv(nu, u)
    i1 = np.isinf(u1)
    i2 = np.isinf(u2)
    if isinstance(h, np.ndarray):
        # array
        ii = ~np.any((i1, i2), axis=0)
        v[ii] = w * 2.0/scipy.special.gamma(nu) * u1[ii] * u2[ii]
        v[i2] = w
    else:
        # one number (float or int)
        if i2:
            v = w
        elif not i1:
            v = w * 2.0/scipy.special.gamma(nu) * u1 * u2
    return v
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Utility functions for Matern covariance model
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def cov_matern_get_effective_range(nu, r):
    """
    Computes the effective range of a 1D-Matern covariance model.

    Parameters
    ----------
    nu : float
        parameter "nu" of the Matern covariance model

    r : float
        parameter "r" (scale) of the Matern covariance, should be positive

    Returns
    -------
    r_eff : float
        effective range of the 1D-Matern covariance model of parameters "nu"
        and "r"
    """
    # fname = 'cov_matern_get_effective_range'
    res = scipy.optimize.root_scalar(lambda h: cov_matern(h, w=1.0, r=r, nu=nu) - 0.05, bracket=[1.e-10*r, 4.0*r])
    return res.root
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def cov_matern_get_r_param(nu, r_eff):
    """
    Computes the parameter "r" (scale) of a 1D-Matern covariance model.

    Parameters
    ----------
    nu : float
        "nu" parameter of the Matern covariance model

    r_eff : float
        effective range, should be positive

    Returns
    -------
    r : float
        parameter "r" (scale) of the 1D-Matern covariance model of parameter "nu",
        such that its effective range is `r_eff`
    """
    # fname = 'cov_matern_get_r_param'
    res = scipy.optimize.minimize_scalar(lambda r: (cov_matern_get_effective_range(nu, r) - r_eff)**2, bracket=[1.e-10*r_eff, 4*r_eff])
    return res.x
# ----------------------------------------------------------------------------

# ============================================================================
# Definition of function to check an elementary covariance contribution
# (type and dictionary of parameters)
# ============================================================================
# ----------------------------------------------------------------------------
def check_elem_cov_model(elem, verbose=0):
    """
    Checks type and dictionary of parameters for an elementary covariance.

    This function validates the type and the dictionary of parameters for an
    elementary contribution in a covariance model in 1D, 2D, or 3D (classes
    :class:`CovModel1D`, :class:`CovModel2D`, :class:`CovModel3D`).

    Parameters
    ----------
    elem : 2-tuple
        elementary model (contributing to a covariance model), elem = (t, d)
        with

        * t : str
            type of elementary covariance model, can be

            - 'nugget'         (see function :func:`cov_nug`)
            - 'spherical'      (see function :func:`cov_sph`)
            - 'exponential'    (see function :func:`cov_exp`)
            - 'gaussian'       (see function :func:`cov_gau`)
            - 'triangular'     (see function :func:`cov_tri`)
            - 'cubic'          (see function :func:`cov_cub`)
            - 'sinus_cardinal' (see function :func:`cov_sinc`)
            - 'gamma'          (see function :func:`cov_gamma`)
            - 'power'          (see function :func:`cov_pow`)
            - 'exponential_generalized' (see function :func:`cov_exp_gen`)
            - 'matern'         (see function :func:`cov_matern`)

        * d : dict
            dictionary of required parameters to be passed to the elementary
            model `t`; parameters required according to `t`:

            - `t = 'nugget'`:
                - `w`, [sequence of] numerical value(s)

            - `t in ('spherical', 'exponential', 'gaussian', 'triangular', 'cubic', 'sinus_cardinal')`:
                - `w`, [sequence of] numerical value(s)
                - `r`, [sequence of] numerical value(s)

            - `t in ('spherical', 'exponential', 'gaussian', 'triangular', 'cubic', 'sinus_cardinal')`:
                - `w`, [sequence of] numerical value(s)
                - `r`, [sequence of] numerical value(s)

            - `t in ('gamma', 'power', 'exponential_generalized')`:
                - `w`, [sequence of] numerical value(s)
                - `r`, [sequence of] numerical value(s)
                - `s`, [sequence of] numerical value(s)

            - `t = matern`:
                - `w`, [sequence of] numerical value(s)
                - `r`, [sequence of] numerical value(s)
                - `nu`, [sequence of] numerical value(s)

    dim : int
        space dimension, 1, 2, or 3

    verbose : int, default: 0
        verbose mode, error message(s) are printed if `verbose>0`

    Returns
    -------
    ok : bool
        - True: covariance type and parameters are valid
        - False: otherwise

    err_mes_list : list
        list of error message (empty if `ok=True`)

    Notes
    -----
    Parameters above may be given as arrays (for non-stationary covariance).
    """
    fname = 'check_elem_cov_model'

    ok = True
    err_mes_list = []

    t, d = elem
    if t == 'nugget': # function `cov_nug`
        # Check required parameters
        if 'w' not in d.keys():
            err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: parameter 'w' is required")
            ok = False
        else:
            if np.any(np.asarray(d['w'])) < 0.0:
                err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: parameter 'w' must be >= 0.0")
                ok = False
        # Check that no other parameter is present
        for p in d.keys():
            if p not in ('w'):
                err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: unknown parameter `{p}`")
                ok = False

    elif t in ('spherical', 'exponential', 'gaussian', 'triangular', 'cubic', 'sinus_cardinal'):
        # Check required parameters
        if 'w' not in d.keys():
            err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: parameter 'w' is required")
            ok = False
        else:
            if np.any(np.asarray(d['w'])) < 0.0:
                err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: parameter 'w' must be >= 0.0")
                ok = False
        if 'r' not in d.keys():
            err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: parameter 'r' is required")
            ok = False
        # else: # no check, e.g. for a 2D cov model, r could be a list  [array([., ., .]), float]
        #     if np.asarray(d['r']).size % dim != 0:
        #         err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: parameter 'r' must be (a sequence of) {dim} floats (or an array of a multiple of {dim} floats)")
        #         ok = False
        #     if np.any(np.asarray(d['r'])) <= 0.0:
        #         err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: parameter 'r' must be > 0.0")
        # Check that no other parameter is present
        for p in d.keys():
            if p not in ('w', 'r'):
                err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: unknown parameter `{p}`")
                ok = False

    elif t in ('gamma', 'power', 'exponential_generalized'):
        # Check required parameters
        if 'w' not in d.keys():
            err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: parameter 'w' is required")
            ok = False
        else:
            if np.any(np.asarray(d['w'])) < 0.0:
                err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: parameter 'w' must be >= 0.0")
                ok = False
        if 'r' not in d.keys():
            err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: parameter 'r' is required")
            ok = False
        # else: # no check, e.g. for a 2D cov model, r could be a list  [array([., ., .]), float]
        #     if np.asarray(d['r']).size % dim != 0:
        #         err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: parameter 'r' must be (a sequence of) {dim} floats (or an array of a multiple of {dim} floats)")
        #         ok = False
        #     if np.any(np.asarray(d['r'])) <= 0.0:
        #         err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: parameter 'r' must be > 0.0")
        if 's' not in d.keys():
            err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: parameter 's' is required")
            ok = False
        else:
            if np.any(np.asarray(d['s'])) < 0.0:
                err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: parameter 's' must be >= 0.0")
                ok = False
        # Check that no other parameter is present
        for p in d.keys():
            if p not in ('w', 'r', 's'):
                err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: unknown parameter `{p}`")
                ok = False

    elif t == 'matern':
        # Check required parameters
        if 'w' not in d.keys():
            err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: parameter 'w' is required")
            ok = False
        else:
            if np.any(np.asarray(d['w'])) < 0.0:
                err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: parameter 'w' must be >= 0.0")
                ok = False
        if 'r' not in d.keys():
            err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: parameter 'r' is required")
            ok = False
        # else: # no check, e.g. for a 2D cov model, r could be a list  [array([., ., .]), float]
        #     if np.asarray(d['r']).size % dim != 0:
        #         err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: parameter 'r' must be (a sequence of) {dim} floats (or an array of a multiple of {dim} floats)")
        #         ok = False
        #     if np.any(np.asarray(d['r'])) <= 0.0:
        #         err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: parameter 'r' must be > 0.0")
        if 'nu' not in d.keys():
            err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: parameter 'nu' is required")
            ok = False
        else:
            if np.any(np.asarray(d['nu'])) < 0.0:
                err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: parameter 'nu' must be >= 0.0")
                ok = False
        # Check that no other parameter is present
        for p in d.keys():
            if p not in ('w', 'r', 'nu'):
                err_mes_list.append(f"ERROR ({fname}): covariance type `'{t}'`: unknown parameter `{p}`")
                ok = False
    else:
        err_mes_list.append(f"ERROR ({fname}): unknown covariance type `'{t}'`")
        ok = False

    if verbose > 0 and not ok:
        for s in err_mes_list:
            print(s)

    return ok, err_mes_list
# ----------------------------------------------------------------------------

# ============================================================================
# Definition of class for covariance models in 1D, 2D, 3D, as combination
# of elementary models and accounting for anisotropy and rotation
# ============================================================================
# ----------------------------------------------------------------------------
class CovModel1D(object):
    """
    Class defining a covariance model in 1D.

    A covariance model is defined as the sum of elementary covariance models.

    An elementary variogram model is defined as its weight parameter (`w`) minus
    the covariance elementary model, and a variogram model is defined as the sum
    of elementary variogram models.

    This class is callable, returning the evaluation of the model (covariance or
    variogram) at given point(s) (lag(s)).

    **Attributes**

    elem : 1D array-like
        sequence of elementary model(s) (contributing to the covariance model),
        each element of the sequence is a 2-tuple (t, d), where

        - t : str
            type of elementary covariance model, can be

            - 'nugget'         (see function :func:`cov_nug`)
            - 'spherical'      (see function :func:`cov_sph`)
            - 'exponential'    (see function :func:`cov_exp`)
            - 'gaussian'       (see function :func:`cov_gau`)
            - 'triangular'     (see function :func:`cov_tri`)
            - 'cubic'          (see function :func:`cov_cub`)
            - 'sinus_cardinal' (see function :func:`cov_sinc`)
            - 'gamma'          (see function :func:`cov_gamma`)
            - 'power'          (see function :func:`cov_pow`)
            - 'exponential_generalized' (see function :func:`cov_exp_gen`)
            - 'matern'         (see function :func:`cov_matern`)

        - d : dict
            dictionary of required parameters to be passed to the elementary
            model `t`

        e.g.

        - (t, d) = ('spherical', {'w':2.0, 'r':1.5})
        - (t, d) = ('power', {'w':2.0, 'r':1.5, 's':1.7})
        - (t, d) = ('matern', {'w':2.0, 'r':1.5, 'nu':1.5})

    name : str, optional
        name of the model


    **Private attributes (SHOULD NOT BE SET DIRECTLY)**

    _r : float
        (effective) range

    _sill : float
        sill (sum of weight of elementary contributions)

    _is_orientation_stationary : bool
        indicates if the covariance model has stationary orientation
        (always True for 1D covariance model)

    _is_weight_stationary : bool
        indicates if the covariance model has stationary weight

    _is_range_stationary : bool
        indicates if the covariance model has stationary range(s)

    _is_stationary : bool
        indicates if the covariance model is stationary

    Examples
    --------
    To define a covariance model (1D) that is the sum of the 2 following
    elementary models:

    - gaussian with a contribution (weight) of 10.0 and a range of 100.0,
    - nugget of (contribution, weight) 0.5

        >>> cov_model = CovModel1D(elem=[
            ('gaussian', {'w':10., 'r':100.0}), # elementary contribution
            ('nugget', {'w':0.5})               # elementary contribution
            ], name='gau+nug')                  # name (optional)

    **Methods**
    """
    #
    # Methods
    # -------
    # reset_private_attributes()
    #     Resets private attributes
    # multiply_w(factor, elem_ind=None)
    #     Multiplies parameter `w` of the (given) elementary contribution(s) by the given factor
    # multiply_r(factor, elem_ind=None)
    #     Multiplies parameter `r` of the (given) elementary contribution(s) by the given factor
    # is_orientation_stationary(recompute=False)
    #     Checks if the covariance model has stationary orientation
    # is_weight_stationary(recompute=False)
    #     Checks if the covariance model has stationary weight
    # is_range_stationary(recompute=False)
    #     Checks if the covariance model has stationary range
    # is_stationary(recompute=False)
    #     Checks if the covariance model is stationary
    # sill(recompute=False)
    #     Retrieves the sill of the covariance model
    # r(recompute=False)
    #     Retrieves the (effective) range of the covariance model
    # func()
    #     Returns the function f for the evaluation of the covariance model
    # vario_func()
    #     Returns the function f for the evaluation of the variogram model
    # plot_model(vario=False, hmin=0-0, hmax=None, npts=500, grid=True, show_xlabel=True, show_ylabel=True, **kwargs)
    #     Plots the covariance or variogram model (in the current figure axis).
    #
    def __init__(self,
                 elem=[],
                 name=None,
                 logger=None):
        """
        Inits an instance of the class.

        Parameters
        ----------
        elem : 1D array-like, default: []
            sequence of elementary model(s)

        name : str, optional
            name of the model

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'CovModel1D'

        self.elem = elem
        for el in self.elem:
            ok, err_mes_list = check_elem_cov_model(el, verbose=0)
            if not ok:
                err_msg = f'{fname}: elementary contribution not valid\n ... ' + '\n ... '.join(err_mes_list)
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

        if name is None:
            if len(elem) == 1:
                name = 'cov1D-' + elem[0][0]
            elif len(elem) > 1:
                name = 'cov1D-multi-contribution'
            else:
                name = 'cov1D-zero'
        self.name = name
        self._r = None  # initialize "internal" variable _r for effective range
        self._sill = None  # initialize "internal" variable _sill for sill (sum of weight(s))
        self._is_orientation_stationary = None # Will be always True for 1D covariance model
        self._is_weight_stationary = None
        self._is_range_stationary = None
        self._is_stationary = None

    # ------------------------------------------------------------------------
    # def __str__(self):
    #     return self.name
    def __repr__(self):
        # """
        # Returns the string that is displayed when the instance of the class is "printed".
        # """
        out = '*** CovModel1D object ***'
        out = out + '\n' + "name = '{0.name}'".format(self)
        out = out + '\n' + 'number of elementary contribution(s): {}'.format(len(self.elem))
        for i, el in enumerate(self.elem):
            out = out + '\n' + 'elementary contribution {}'.format(i)
            out = out + '\n' + '    type: {}'.format(el[0])
            out = out + '\n' + '    parameters:'
            #for j, (k, val) in enumerate(el[1].items()):
            for k, val in el[1].items():
                out = out + '\n' + '        {} = {}'.format(k, val)
        out = out + '\n' + '*****'
        return out
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def __call__(self, h, vario=False):
        """
        Evaluates the covariance model at given 1D lags (`h`).

        Parameters
        ----------
        h : 1D array-like of floats, or float
            point(s) (lag(s)) where the covariance model is evaluated

        vario : bool, default: False
            - if False: computes the covariance
            - if True: computes the variogram

        Returns
        -------
        y : 1D array
            evaluation of the covariance or variogram model at `h`;
            note: the result is casted to a 1D array if `h` is a float
        """
        if vario:
            return self.vario_func()(h)
        else:
            return self.func()(h)
    # ------------------------------------------------------------------------

    def reset_private_attributes(self):
        """
        Resets (sets to `None`) the "private" attributes (beginning with "_").
        """
        # fname = 'reset_private_attributes'

        self._r = None
        self._sill = None
        self._is_orientation_stationary = None
        self._is_weight_stationary = None
        self._is_range_stationary = None
        self._is_stationary = None

    def multiply_w(self, factor, elem_ind=None, logger=None):
        """
        Multiplies parameter `w` of the (given) elementary contribution(s) by the given factor.

        The covariance model is updated and relevant private attributes
        (beginning with "_") are reset.

        Parameters
        ----------
        factor : array of floats or float
            multiplier(s), if array, its shape must be compatible with the
            dimension of the grid on which the covariance model is used (for
            Gaussian interpolation or simulation)

        elem_ind : 1D array-like of ints, or int, optional
            indexe(s) of the elementary contribution (attribute `elem`) to be
            modified; by default (`None`): indexes of any elementary contribution
            are selected

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'multiply_w'

        if elem_ind is None:
            elem_ind = np.arange(len(self.elem))
        else:
            elem_ind = np.atleast_1d(elem_ind).reshape(-1)
            n = len(self.elem)
            if np.any((elem_ind > n - 1, elem_ind < -n)):
                err_msg = f'{fname}: `elem_ind` invalid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

        for i in elem_ind:
            if 'w' in self.elem[i][1].keys():
                self.elem[i][1]['w'] = factor * self.elem[i][1]['w']

        self._sill = None
        self._is_weight_stationary = None
        self._is_stationary = None

    def add_w(self, a, elem_ind=None, logger=None):
        """
        Add `a`to parameter `w` of the (given) elementary contribution(s).

        The covariance model is updated and relevant private attributes
        (beginning with "_") are reset.

        Parameters
        ----------
        a : array of floats or float
            term(s) to add, if array, its shape must be compatible with the
            dimension of the grid on which the covariance model is used (for
            Gaussian interpolation or simulation)

        elem_ind : 1D array-like of ints, or int, optional
            indexe(s) of the elementary contribution (attribute `elem`) to be
            modified; by default (`None`): indexes of any elementary contribution
            are selected

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'add_w'

        if elem_ind is None:
            elem_ind = np.arange(len(self.elem))
        else:
            elem_ind = np.atleast_1d(elem_ind).reshape(-1)
            n = len(self.elem)
            if np.any((elem_ind > n - 1, elem_ind < -n)):
                err_msg = f'{fname}: `elem_ind` invalid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

        for i in elem_ind:
            if 'w' in self.elem[i][1].keys():
                new_w = self.elem[i][1]['w'] + a
                if new_w < 0:
                    err_msg = f'{fname}: new `w` is negative'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)

                self.elem[i][1]['w'] = new_w

        self._sill = None
        self._is_weight_stationary = None
        self._is_stationary = None

    def multiply_r(self, factor, elem_ind=None, logger=None):
        """
        Multiplies parameter `r` of the (given) elementary contribution(s) by the given factor.

        The covariance model is updated and relevant private attributes
        (beginning with "_") are reset.

        Parameters
        ----------
        factor : array of floats or float
            multiplier(s), if array, its shape must be compatible with the
            dimension of the grid on which the covariance model is used (for
            Gaussian interpolation or simulation)

        elem_ind : 1D array-like of ints, or int, optional
            indexe(s) of the elementary contribution (attribute `elem`) to be
            modified; by default (`None`): indexes of any elementary contribution
            are selected

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'multiply_r'

        if elem_ind is None:
            elem_ind = np.arange(len(self.elem))
        else:
            elem_ind = np.atleast_1d(elem_ind).reshape(-1)
            n = len(self.elem)
            if np.any((elem_ind > n - 1, elem_ind < -n)):
                err_msg = f'{fname}: `elem_ind` invalid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

        for i in elem_ind:
            if 'r' in self.elem[i][1].keys():
                self.elem[i][1]['r'] = factor * self.elem[i][1]['r']

        self._r = None
        self._is_range_stationary = None
        self._is_stationary = None

    def is_orientation_stationary(self, recompute=False):
        """
        Checks if the covariance model has stationary orientation.

        (Always True for 1D covariance model.)

        Parameters
        ----------
        recompute : bool, default: False
            True to force (re-)computing

        Returns
        -------
        flag : bool
            boolean indicating if the orientation is stationary
            (private attribute `_is_orientation_stationary`)
        """
        # fname = 'is_orientation_stationary'

        self._is_orientation_stationary = True
        return self._is_orientation_stationary

    def is_weight_stationary(self, recompute=False):
        """
        Checks if the covariance model has stationary weight.

        Parameters
        ----------
        recompute : bool, default: False
            True to force (re-)computing

        Returns
        -------
        flag : bool
            boolean indicating if the weight (parameter `w`) of every elementary
            contribution is stationary (defined as a unique value)
            (private attribute `_is_weight_stationary`)
        """
        # fname = 'is_weight_stationary'
        if self._is_weight_stationary is None or recompute:
            self._is_weight_stationary = not np.any([np.size(el[1]['w']) > 1 for el in self.elem])
        return self._is_weight_stationary

    def is_range_stationary(self, recompute=False):
        """
        Checks if the covariance model has stationary range.

        Parameters
        ----------
        recompute : bool, default: False
            True to force (re-)computing

        Returns
        -------
        flag : bool
            boolean indicating if the range (parameter `r`) of every elementary
            contribution is stationary (defined as a unique value)
            (private attribute `_is_range_stationary`)
        """
        # fname = 'is_range_stationary'

        if self._is_range_stationary is None or recompute:
            self._is_range_stationary = True
            for el in self.elem:
                if 'r' in el[1].keys() and np.size(el[1]['r']) > 1:
                    self._is_range_stationary = False
                    break
        return self._is_range_stationary

    def is_stationary(self, recompute=False):
        """
        Checks if the covariance model is stationary.

        Parameters
        ----------
        recompute : bool, default: False
            True to force (re-)computing

        Returns
        -------
        flag : bool
            boolean indicating if all the parameters are stationary (defined as
            a unique value)
            (private attribute `_is_stationary`)
        """
        # fname = 'is_stationary'
        if self._is_stationary is None or recompute:
            self._is_stationary = self.is_orientation_stationary(recompute) and self.is_weight_stationary(recompute) and self.is_range_stationary(recompute)
            if self._is_stationary:
                for t, d in self.elem:
                    flag = True
                    for k, v in d.items():
                        if k in ('w', 'r'):
                            continue
                        if np.size(v) > 1:
                            flag = False
                            break
                    if not flag:
                        self._is_stationary = False
                        break
        return self._is_stationary

    def sill(self, recompute=False):
        """
        Retrieves the sill of the covariance model.

        Parameters
        ----------
        recompute : bool, default: False
            True to force (re-)computing

        Returns
        -------
        sill : float
            sill, sum of the weights of all elementary contributions
            (private attribute `_sill`)

        Notes
        -----
        Nothing is returned if the model has non-stationary weight
        (return `None`).
        """
        # fname = 'sill'

        if self._sill is None or recompute:
            # Prevent calculation if weight is not stationary
            if not self.is_weight_stationary(recompute):
                self._sill = None
                return self._sill

            self._sill = sum([d['w'] for t, d in self.elem if 'w' in d])

        return self._sill

    def r(self, recompute=False):
        """
        Retrieves the (effective) range of the covariance model.

        The effective range of the model is the maximum of the effective range
        of all elementary contributions; note that the "effective" range is the
        distance beyond which the covariance is zero or below 5% of the weight,
        and corresponds to the parameter `r` for most of elementary covariance
        models.

        Parameters
        ----------
        recompute : bool, default: False
            True to force (re-)computing

        Returns
        -------
        range : float
            (effective) range of the covariance model

        Notes
        -----
        Nothing is returned if the model has non-stationary range
        (return `None`).
        """
        # fname = 'r'

        if self._r is None or recompute:
            # Prevent calculation if range is not stationary
            if not self.is_range_stationary(recompute):
                self._r = None
                return self._r

            r = 0.
            for t, d in self.elem:
                if t in (
                        'spherical',
                        'exponential',
                        'gaussian',
                        'triangular',
                        'cubic',
                        'sinus_cardinal',
                        'gamma',
                        'power', # not really the range for this case
                        'exponential_generalized',
                        ):
                    r = max(r, d['r'])

                elif t == 'matern':
                    r = max(r, cov_matern_get_effective_range(d['nu'], d['r']))

            self._r = r

        return self._r

    def func(self):
        """
        Returns the function f for the evaluation of the covariance model.

        Returns
        -------
        f : function
            function with parameters (arguments):

            - h : 1D array-like of floats, or float
                point(s) (lag(s)) where the covariance model is evaluated

            that returns:

            - f(h) : 1D array
                evaluation of the covariance model at `h`;
                note: the result is casted to a 1D array if `h` is a float

        Notes
        -----
        No evaluation is done if the model is not stationary (return `None`).
        """
        # fname = 'func'

        # Prevent calculation if covariance model is not stationary
        if not self.is_stationary():
            return None
        def f(h):
            h = np.array(h).reshape(-1)  # cast to 1-dimensional array if needed
            s = np.zeros(len(h))
            for t, d in self.elem:
                if t == 'nugget':
                    s = s + cov_nug(h, **d)

                elif t == 'spherical':
                    s = s + cov_sph(h, **d)

                elif t == 'exponential':
                    s = s + cov_exp(h, **d)

                elif t == 'gaussian':
                    s = s + cov_gau(h, **d)

                elif t == 'triangular':
                    s = s + cov_tri(h, **d)

                elif t == 'cubic':
                    s = s + cov_cub(h, **d)

                elif t == 'sinus_cardinal':
                    s = s + cov_sinc(h, **d)

                elif t == 'gamma':
                    s = s + cov_gamma(h, **d)

                elif t == 'power':
                    s = s + cov_pow(h, **d)

                elif t == 'exponential_generalized':
                    s = s + cov_exp_gen(h, **d)

                elif t == 'matern':
                    s = s + cov_matern(h, **d)

            return s

        return f

    def vario_func(self):
        """
        Returns the function f for the evaluation of the variogram model.

        Returns
        -------
        f : function
            function with parameters (arguments):

            - h : 1D array-like of floats, or float
                point(s) (lag(s)) where the variogram model is evaluated

            that returns:

            - f(h) : 1D array
                evaluation of the variogram model at `h`;
                note: the result is casted to a 1D array if `h` is a float

        Notes
        -----
        No evaluation is done if the model is not stationary (return `None`).
        """
        # fname = 'vario_func'

        # Prevent calculation if covariance model is not stationary
        if not self.is_stationary():
            return None
        def f(h):
            h = np.array(h).reshape(-1)  # cast to 1-dimensional array if needed
            s = np.zeros(len(h))
            for t, d in self.elem:
                if t == 'nugget':
                    s = s + d['w'] - cov_nug(h, **d)

                elif t == 'spherical':
                    s = s + d['w'] - cov_sph(h, **d)

                elif t == 'exponential':
                    s = s + d['w'] - cov_exp(h, **d)

                elif t == 'gaussian':
                    s = s + d['w'] - cov_gau(h, **d)

                elif t == 'triangular':
                    s = s + d['w'] - cov_tri(h, **d)

                elif t == 'cubic':
                    s = s + d['w'] - cov_cub(h, **d)

                elif t == 'sinus_cardinal':
                    s = s + d['w'] - cov_sinc(h, **d)

                elif t == 'gamma':
                    s = s + d['w'] - cov_gamma(h, **d)

                elif t == 'power':
                    s = s + d['w'] - cov_pow(h, **d)

                elif t == 'exponential_generalized':
                    s = s + d['w'] - cov_exp_gen(h, **d)

                elif t == 'matern':
                    s = s + d['w'] - cov_matern(h, **d)

            return s

        return f

    def plot_model(
            self,
            vario=False,
            hmin=0.0,
            hmax=None,
            npts=500,
            show_xlabel=True,
            show_ylabel=True,
            grid=True,
            **kwargs):
        """
        Plots the covariance or variogram model f(h) (in the current figure axis).

        Parameters
        ----------
        vario : bool, default: False
            - if False: plots the covariance
            - if True: plots the variogram

        hmin : float, default: 0.0
            see `hmax`
        hmax : float, optional
            function is plotted for h in interval [`hmin`,` hmax`]; by default
            (`hmax=None`), `hmax` is set to 1.2 times the range of the model

        npts : int, default: 500
            number of points used in interval [`hmin`,` hmax`]

        show_xlabel : bool, default: True
            indicates if (default) label for abscissa is displayed

        show_ylabel : bool, default: True
            indicates if (default) label for ordinate is displayed

        grid : bool, default: True
            indicates if a grid is plotted

        kwargs : dict
            keyword arguments passed to the funtion `matplotlib.pyplot.plot`

        Notes
        -----
        No plot is displayed if the model is not stationary.
        """
        # fname = 'plot_model'

        # Prevent calculation if covariance model is not stationary
        if not self.is_stationary():
            return None
        # In kwargs:
        #   - add default 'label' if not given
        if 'label' not in kwargs.keys():
            if vario:
                kwargs['label'] = 'vario func'
            else:
                kwargs['label'] = 'cov func'

        # Set hmax if needed
        if hmax is None:
            hmax = 1.2*self.r()

        h = np.linspace(hmin, hmax, npts)
        g = self(h, vario=vario)
        # if vario:
        #     g = self.vario_func()(h)
        # else:
        #     g = self.func()(h)

        plt.plot(h, g, **kwargs)

        if show_xlabel:
            plt.xlabel('h')
        if show_ylabel:
            if vario:
                plt.ylabel(r'$\gamma(h)$')
            else:
                plt.ylabel(r'$cov(h)$')

        plt.grid(grid)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
class CovModel2D(object):
    """
    Class defining a covariance model in 2D.

    A covariance model is defined as the sum of elementary covariance models.

    An elementary variogram model is defined as its weight parameter (`w`) minus
    the covariance elementary model, and a variogram model is defined as the sum
    of elementary variogram models.

    This class is callable, returning the evaluation of the model (covariance or
    variogram) at given point(s) (lag(s)).

    **Attributes**

    elem : 1D array-like
        sequence of elementary model(s) (contributing to the covariance model),
        each element of the sequence is a 2-tuple (t, d), where

        - t : str
            type of elementary covariance model, can be

            - 'nugget'         (see function :func:`cov_nug`)
            - 'spherical'      (see function :func:`cov_sph`)
            - 'exponential'    (see function :func:`cov_exp`)
            - 'gaussian'       (see function :func:`cov_gau`)
            - 'triangular'     (see function :func:`cov_tri`)
            - 'cubic'          (see function :func:`cov_cub`)
            - 'sinus_cardinal' (see function :func:`cov_sinc`)
            - 'gamma'          (see function :func:`cov_gamma`)
            - 'power'          (see function :func:`cov_pow`)
            - 'exponential_generalized' (see function :func:`cov_exp_gen`)
            - 'matern'         (see function :func:`cov_matern`)

        - d : dict
            dictionary of required parameters to be passed to the elementary
            model `t`

        e.g.

        - (t, d) = ('spherical', {'w':2.0, 'r':[1.5, 2.5]})
        - (t, d) = ('power', {'w':2.0, 'r':[1.5, 2.5], 's':1.7})
        - (t, d) = ('matern', {'w':2.0, 'r':[1.5, 2.5], 'nu':1.5})

    alpha : float, default: 0.0
        azimuth angle in degrees; the system Ox'y', supporting the axes of the
        model (ranges), is obtained from the system Oxy by applying a rotation
        of angle `-alpha`.
        The 2x2 matrix m for changing the coordinates system from Ox'y' to Oxy is:

        .. math::
            m = \\left(\\begin{array}{cc}
                    \\cos\\alpha & \\sin\\alpha\\\\
                   -\\sin\\alpha & \\cos\\alpha
                \\end{array}\\right)

    name : str, optional
        name of the model

    **Private attributes (SHOULD NOT BE SET DIRECTLY)**

    _r : float
        maximal (effective) range, along the two axes

    _sill : float
        sill (sum of weight of elementary contributions)

    _mrot : 2D array of shape (2, 2)
        rotation matrix m (see above)

    _is_orientation_stationary : bool
        indicates if the covariance model has stationary orientation

    _is_weight_stationary : bool
        indicates if the covariance model has stationary weight

    _is_range_stationary : bool
        indicates if the covariance model has stationary range(s)

    _is_stationary : bool
        indicates if the covariance model is stationary

    Examples
    --------
    To define a covariance model (2D) that is the sum of the 2 following
    elementary models:

    - gaussian with a contribution (weight) of 10.0 and ranges of 150.0 and 50.0,
    - nugget of (contribution, weight) 0.5

    and in the system Ox'y' defined by the angle alpha=-30.0

        >>> cov_model = CovModel2D(elem=[
                ('gaussian', {'w':10.0, 'r':[150.0, 50.0]}), # elementary contribution
                ('nugget', {'w':0.5})                        # elementary contribution
                ], alpha=-30.0,                              # angle
                name='')                                     # name (optional)

    **Methods**
    """
    #
    # The 2x2 matrix m for changing the coordinates system from Ox'y' to Oxy is:
    #         +                         +
    #         |  cos(alpha)   sin(alpha)|
    #     m = | -sin(alpha)   cos(alpha)|
    #         +                         +
    #
    # Methods
    # -------
    # reset_private_attributes()
    #     Resets private attributes
    # set_alpha(alpha)
    #     Sets (modifies) the attribute `alpha`
    # multiply_w(factor, elem_ind=None)
    #     Multiplies parameter `w` of the (given) elementary contribution(s) by the given factor
    # multiply_r(sfactor, r_ind=None, elem_ind=None)
    #     Multiplies (given index(es) of) parameter `r` of the (given) elementary contribution(s) by the given factor
    # is_orientation_stationary(recompute=False)
    #     Checks if the covariance model has stationary orientation
    # is_weight_stationary(recompute=False)
    #     Checks if the covariance model has stationary weight
    # is_range_stationary(recompute=False)
    #     Checks if the covariance model has stationary range
    # is_stationary(recompute=False)
    #     Checks if the covariance model is stationary
    # sill(recompute=False)
    #     Retrieves the sill of the covariance model
    # mrot(recompute=False)
    #     Returns the 2x2 matrix of rotation defining the axes of the model
    # r12(recompute=False)
    #     Returns the (effective) ranges along x', y' axes supporting the model
    # rxy(recompute=False)
    #     Returns the (effective) ranges along x, y axes of the "original" coordinates system
    # func()
    #     Returns the function f for the evaluation of the covariance model
    # vario_func()
    #     Returns the function f for the evaluation of the variogram model
    # plot_mrot(color0='red', color1='green')
    #     Plots axes of system Oxy and Ox'y' (in the current figure axis)
    # plot_model(vario=False, plot_map=True, plot_curves=True, cmap='terrain', color0='red', color1='green', extent=None, ncell=(201, 201), h1min=0.0, h1max=None, h2min=0.0, h2max=None, n1=500, n2=500, grid=True, show_xlabel=True, show_ylabel=True, show_suptitle=True, figsize=None)
    #     Plots the covariance or variogram model
    # plot_model_one_curve(main_axis=1, vario=False, hmin=0.0, hmax=None, npts=500, grid=True, show_xlabel=True, show_ylabel=True, **kwargs)
    #     Plots the covariance or variogram curve along one main axis (in the current figure axis)
    #
    def __init__(self,
                 elem=[],
                 alpha=0.0,
                 name=None,
                 logger=None):
        """
        Inits an instance of the class.

        Parameters
        ----------
        elem : 1D array-like, default: []
            sequence of elementary model(s)

        alpha : float, default: 0.0
            azimuth angle in degrees

        name : str, optional
            name of the model
            
        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'CovModel2D'

        self.elem = elem
        for el in self.elem:
            ok, err_mes_list = check_elem_cov_model(el, verbose=0)
            if not ok:
                err_msg = f'{fname}: elementary contribution not valid\n ... ' + '\n ... '.join(err_mes_list)
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

        self.alpha = alpha
        if name is None:
            if len(elem) == 1:
                name = 'cov2D-' + elem[0][0]
            elif len(elem) > 1:
                name = 'cov2D-multi-contribution'
            else:
                name = 'cov2D-zero'

        self.name = name
        self._r = None  # initialize "internal" variable _r for effective range
        self._sill = None  # initialize "internal" variable _sill for sill (sum of weight(s))
        self._mrot = None  # initialize "internal" variable _mrot for rotation matrix
        self._is_orientation_stationary = None
        self._is_weight_stationary = None
        self._is_range_stationary = None
        self._is_stationary = None

    # ------------------------------------------------------------------------
    # def __str__(self):
    #     return self.name
    def __repr__(self):
        # """
        # Returns the string that is displayed when the instance of the class is "printed".
        # """
        out = '*** CovModel2D object ***'
        out = out + '\n' + "name = '{0.name}'".format(self)
        nelem = len(self.elem)
        out = out + '\n' + 'number of elementary contribution(s): {}'.format(len(self.elem))
        for i, el in enumerate(self.elem):
            out = out + '\n' + 'elementary contribution {}'.format(i)
            out = out + '\n' + '    type: {}'.format(el[0])
            out = out + '\n' + '    parameters:'
            # nparam = len(el[1])
            for j, (k, val) in enumerate(el[1].items()):
                out = out + '\n' + '        {} = {}'.format(k, val)
        out = out + '\n' + 'angle: alpha = {0.alpha} deg.'.format(self)
        out = out + '\n' + "    i.e.: the system Ox'y', supporting the axes of the model (ranges),"
        out = out + '\n' + '    is obtained from the system Oxy by applying a rotation of angle -alpha.'
        out = out + '\n' + '*****'
        return out
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def __call__(self, h, vario=False):
        """
        Evaluates the covariance model at given 2D lags (`h`).

        Parameters
        ----------
        h : 2D array-like of shape (n, 2) or 1D array-like of shape (2,)
            point(s) (lag(s)) where the covariance model is evaluated;
            if `h` is a 2D array, each row is a lag

        vario : bool, default: False
            - if False: computes the covariance
            - if True: computes the variogram

        Returns
        -------
        y : 1D array
            evaluation of the covariance or variogram model at `h`;
            note: the result is casted to a 1D array if `h` is a 1D array
        """
        if vario:
            return self.vario_func()(h)
        else:
            return self.func()(h)
    # ------------------------------------------------------------------------

    def reset_private_attributes(self):
        """
        Resets (sets to `None`) the "private" attributes (beginning with "_").
        """
        # fname = 'reset_private_attributes'

        self._r = None
        self._sill = None
        self._mrot = None
        self._is_orientation_stationary = None
        self._is_weight_stationary = None
        self._is_range_stationary = None
        self._is_stationary = None

    def set_alpha(self, alpha):
        """
        Sets (modifies) the attribute `alpha`.

        The covariance model is updated and relevant private attributes
        (beginning with "_") are reset.

        Parameters
        ----------
        alpha : array of float or float
            azimuth angle in degrees; if array, its shape must be compatible with
            the dimension of the grid on which the covariance model is used (for
            Gaussian interpolation or simulation)
        """
        # fname = 'set_alpha'

        self.alpha = alpha
        self._mrot = None
        self._is_orientation_stationary = None
        self._is_stationary = None

    def multiply_w(self, factor, elem_ind=None, logger=None):
        """
        Multiplies parameter `w` of the (given) elementary contribution(s) by the given factor.

        The covariance model is updated and relevant private attributes
        (beginning with "_") are reset.

        Parameters
        ----------
        factor : array of floats or float
            multiplier(s), if array, its shape must be compatible with the
            dimension of the grid on which the covariance model is used (for
            Gaussian interpolation or simulation)

        elem_ind : 1D array-like of ints, or int, optional
            indexe(s) of the elementary contribution (attribute `elem`) to be
            modified; by default (`None`): indexes of any elementary contribution
            are selected
            
        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'multiply_w'

        if elem_ind is None:
            elem_ind = np.arange(len(self.elem))
        else:
            elem_ind = np.atleast_1d(elem_ind).reshape(-1)
            n = len(self.elem)
            if np.any((elem_ind > n - 1, elem_ind < -n)):
                err_msg = f'{fname}: `elem_ind` invalid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

        for i in elem_ind:
            if 'w' in self.elem[i][1].keys():
                self.elem[i][1]['w'] = factor * self.elem[i][1]['w']

        self._sill = None
        self._is_weight_stationary = None
        self._is_stationary = None

    def add_w(self, a, elem_ind=None, logger=None):
        """
        Add `a`to parameter `w` of the (given) elementary contribution(s).

        The covariance model is updated and relevant private attributes
        (beginning with "_") are reset.

        Parameters
        ----------
        a : array of floats or float
            term(s) to add, if array, its shape must be compatible with the
            dimension of the grid on which the covariance model is used (for
            Gaussian interpolation or simulation)

        elem_ind : 1D array-like of ints, or int, optional
            indexe(s) of the elementary contribution (attribute `elem`) to be
            modified; by default (`None`): indexes of any elementary contribution
            are selected
            
        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'add_w'

        if elem_ind is None:
            elem_ind = np.arange(len(self.elem))
        else:
            elem_ind = np.atleast_1d(elem_ind).reshape(-1)
            n = len(self.elem)
            if np.any((elem_ind > n - 1, elem_ind < -n)):
                err_msg = f'{fname}: `elem_ind` invalid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

        for i in elem_ind:
            if 'w' in self.elem[i][1].keys():
                new_w = self.elem[i][1]['w'] + a
                if new_w < 0:
                    err_msg = f'{fname}: new `w` is negative'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)

                self.elem[i][1]['w'] = new_w

        self._sill = None
        self._is_weight_stationary = None
        self._is_stationary = None

    def multiply_r(self, factor, r_ind=None, elem_ind=None, logger=None):
        """
        Multiplies (given index(es) of) parameter `r` of the (given) elementary contribution(s) by the given factor.

        The covariance model is updated and relevant private attributes
        (beginning with "_") are reset.

        Parameters
        ----------
        factor : array of floats or float
            multiplier(s), if array, its shape must be compatible with the
            dimension of the grid on which the covariance model is used (for
            Gaussian interpolation or simulation)

        r_ind : int or sequence of ints, optional
            indexe(s) of the parameter `r` of elementary contribution to be
            modified; by default (`None`): `r_ind=(0, 1)` is used, i.e.
            parameter `r` along each axis is multiplied

        elem_ind : 1D array-like of ints, or int, optional
            indexe(s) of the elementary contribution (attribute `elem`) to be
            modified; by default (`None`): indexes of any elementary contribution
            are selected
            
        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'multiply_r'

        if r_ind is None:
            r_ind = (0, 1)
        else:
            r_ind = np.atleast_1d(r_ind).reshape(-1)
            if np.any((r_ind > 1, r_ind < -2)):
                err_msg = f'{fname}: `r_ind` invalid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

        if elem_ind is None:
            elem_ind = np.arange(len(self.elem))
        else:
            elem_ind = np.atleast_1d(elem_ind).reshape(-1)
            n = len(self.elem)
            if np.any((elem_ind > n - 1, elem_ind < -n)):
                err_msg = f'{fname}: `elem_ind` invalid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

        for i in elem_ind:
            if 'r' in self.elem[i][1].keys():
                for j in r_ind:
                    self.elem[i][1]['r'][j] = factor * self.elem[i][1]['r'][j]

        self._r = None
        self._is_range_stationary = None
        self._is_stationary = None

    def is_orientation_stationary(self, recompute=False):
        """
        Checks if the covariance model has stationary orientation.

        Parameters
        ----------
        recompute : bool, default: False
            True to force (re-)computing

        Returns
        -------
        flag : bool
            boolean indicating if the orientation is stationary, i.e. attritbute
            `alpha` is defined as a unique value
            (private attribute `_is_orientation_stationary`)
        """
        # fname = 'is_orientation_stationary'

        if self._is_orientation_stationary is None or recompute:
            self._is_orientation_stationary = np.size(self.alpha) == 1
        return self._is_orientation_stationary

    def is_weight_stationary(self, recompute=False):
        """
        Checks if the covariance model has stationary weight.

        Parameters
        ----------
        recompute : bool, default: False
            True to force (re-)computing

        Returns
        -------
        flag : bool
            boolean indicating if the weight (parameter `w`) of every elementary
            contribution is stationary (defined as a unique value)
            (private attribute `_is_weight_stationary`)
        """
        # fname = 'is_weight_stationary'

        if self._is_weight_stationary is None or recompute:
            self._is_weight_stationary = not np.any([np.size(el[1]['w']) > 1 for el in self.elem])
        return self._is_weight_stationary

    def is_range_stationary(self, recompute=False):
        """
        Checks if the covariance model has stationary ranges.

        Parameters
        ----------
        recompute : bool, default: False
            True to force (re-)computing

        Returns
        -------
        flag : bool
            boolean indicating if the range along each axis (parameter `r`)
            of every elementary contribution is stationary (`r[i]` defined as a
            unique value)
            (private attribute `_is_range_stationary`)
        """
        # fname = 'is_range_stationary'

        if self._is_range_stationary is None or recompute:
            self._is_range_stationary = True
            for el in self.elem:
                if 'r' in el[1].keys() and np.any([np.size(ri) > 1 for ri in el[1]['r']]):
                    self._is_range_stationary = False
                    break
        return self._is_range_stationary

    def is_stationary(self, recompute=False):
        """
        Checks if the covariance model is stationary.

        Parameters
        ----------
        recompute : bool, default: False
            True to force (re-)computing

        Returns
        -------
        flag : bool
            boolean indicating if all the parameters are stationary (defined as
            a unique value)
            (private attribute `_is_stationary`)
        """
        # fname = 'is_stationary'

        if self._is_stationary is None or recompute:
            self._is_stationary = self.is_orientation_stationary(recompute) and self.is_weight_stationary(recompute) and self.is_range_stationary(recompute)
            if self._is_stationary:
                for t, d in self.elem:
                    flag = True
                    for k, v in d.items():
                        if k in ('w', 'r'):
                            continue
                        if np.size(v) > 1:
                            flag = False
                            break
                    if not flag:
                        self._is_stationary = False
                        break
        return self._is_stationary

    def sill(self, recompute=False):
        """
        Retrieves the sill of the covariance model.

        Parameters
        ----------
        recompute : bool, default: False
            True to force (re-)computing

        Returns
        -------
        sill : float
            sill, sum of the weights of all elementary contributions
            (private attribute `_sill`)

        Notes
        -----
        Nothing is returned if the model has non-stationary weight
        (return `None`).
        """
        # fname = 'sill'

        if self._sill is None or recompute:
            # Prevent calculation if weight is not stationary
            if not self.is_weight_stationary(recompute):
                self._sill = None
                return self._sill

            self._sill = sum([d['w'] for t, d in self.elem if 'w' in d])

        return self._sill

    def mrot(self, recompute=False):
        """
        Returns the 2x2 matrix of rotation defining the axes of the model.

        The 2x2 matrix m is the matrix of changes of coordinate system,
        from Ox'y' to Oxy, where Ox' and Oy' are the axes supporting the ranges
        of the model.

        Parameters
        ----------
        recompute : bool, default: False
            True to force (re-)computing

        Returns
        -------
        mrot : 2D array of shape (2, 2)
            rotation matrix (private attribute _mrot)

        Notes
        -----
        Nothing is returned if the model has non-stationary orientation
        (return `None`).
        """
        # fname = 'mrot'

        if self._mrot is None or recompute:
            # Prevent calculation if orientation is not stationary
            if not self.is_orientation_stationary(recompute):
                self._mrot = None
                return self._mrot

            self._mrot = rotationMatrix2D(self.alpha)

        return self._mrot

    def r12(self, recompute=False):
        """
        Returns the (effective) ranges along x', y' axes supporting the model.

        The effective range of the model (in a given direction) is the maximum
        of the effective range of all elementary contributions; note that the
        "effective" range is the distance beyond which the covariance is zero or
        below 5% of the weight, and corresponds to the (components of the)
        parameter `r` for most of elementary covariance models.

        Parameters
        ----------
        recompute : bool, default: False
            True to force (re-)computing

        Returns
        -------
        range : 1D array of shape (2,)
            (effective) ranges along x', y' axes supporting the model
            (private attribute `_r`)

        Notes
        -----
        Nothing is returned if the model has non-stationary ranges
        (return `None`).
        """
        # fname = 'r12'

        if self._r is None or recompute:
            # Prevent calculation if range is not stationary
            if not self.is_range_stationary(recompute):
                self._r = None
                return self._r

            r = np.array([0., 0.])
            for t, d in self.elem:
                if t in (
                        'spherical',
                        'exponential',
                        'gaussian',
                        'triangular',
                        'cubic',
                        'sinus_cardinal',
                        'gamma',
                        'power', # not really the range for this case
                        'exponential_generalized',
                        ):
                    r = np.maximum(r, d['r']) # element-wise maximum

                elif t == 'matern':
                    for i, ri in enumerate(d['r']):
                        r[i] = max(r[i], cov_matern_get_effective_range(d['nu'], ri))

            self._r = r

        return self._r

    def rxy(self, recompute=False):
        """
        Returns the (effective) ranges along x, y axes of the "original" coordinates system.

        The effective range of the model (in a given direction) is the maximum
        of the effective range of all elementary contributions; note that the
        "effective" range is the distance beyond which the covariance is zero or
        below 5% of the weight, and corresponds to the (components of the)
        parameter `r` for most of elementary covariance models.

        Parameters
        ----------
        recompute : bool, default: False
            True to force (re-)computing

        Returns
        -------
        range : 1D array of shape (2,)
            (effective) ranges along x, y axes of the "original" coordinates
            system

        Notes
        -----
        Nothing is returned if the model has non-stationary ranges or non
        stationary orientation (return `None`).
        """
        # fname = 'rxy'

        # Prevent calculation if range or orientation is not stationary
        if not self.is_range_stationary(recompute) or not self.is_orientation_stationary(recompute):
            return None
        r12 = self.r12(recompute)
        m = np.abs(self.mrot(recompute))
        return np.maximum(r12[0] * m[:,0], r12[1] * m[:,1]) # element-wise maximum

    def func(self):
        """
        Returns the function f for the evaluation of the covariance model.

        Returns
        -------
        f : function
            function with parameters (arguments):

            - h : 2D array-like of shape (n, 2) or 1D array-like of shape (2,)
                point(s) (lag(s)) where the covariance model is evaluated;
                if `h` is a 2D array, each row is a lag

            that returns:

            - f(h) : 1D array
                evaluation of the covariance model at `h`;
                note: the result is casted to a 1D array if `h` is a 1D array

        Notes
        -----
        No evaluation is done if the model is not stationary (return `None`).
        """
        # fname = 'func'

        # Prevent calculation if covariance model is not stationary
        if not self.is_stationary():
            return None
        def f(h):
            h = np.array(h).reshape(-1,2)  # cast to 2-dimensional array with 2 columns if needed
            if self.alpha != 0:
                hnew = np.dot(h, self.mrot()).reshape(-1,2)
            else:
                hnew = h.reshape(-1,2)

            s = np.zeros(hnew.shape[0])

            for t, d in self.elem:
                # new dictionary from d (remove 'r' key)
                dnew = {key:val for key, val in d.items() if key != 'r'}
                if t == 'nugget':
                    s = s + cov_nug(np.sum(hnew != 0, axis=1), **dnew)

                elif t == 'spherical':
                    s = s + cov_sph(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'exponential':
                    s = s + cov_exp(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'gaussian':
                    s = s + cov_gau(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'triangular':
                    s = s + cov_tri(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'cubic':
                    s = s + cov_cub(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'sinus_cardinal':
                    s = s + cov_sinc(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'gamma':
                    s = s + cov_gamma(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'power':
                    s = s + cov_pow(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'exponential_generalized':
                    s = s + cov_exp_gen(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'matern':
                    s = s + cov_matern(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

            return s

        return f

    def vario_func(self):
        """
        Returns the function f for the evaluation of the variogram model.

        Returns
        -------
        f : function
            function with parameters (arguments):

            - h : 2D array-like of shape (n, 2) or 1D array-like of shape (2,)
                point(s) (lag(s)) where the variogram model is evaluated;
                if `h` is a 2D array, each row is a lag

            that returns:

            - f(h) : 1D array
                evaluation of the variogram model at `h`;
                note: the result is casted to a 1D array if `h` is a 1D array

        Notes
        -----
        No evaluation is done if the model is not stationary (return `None`).
        """
        # fname = 'vario_func'

        # Prevent calculation if covariance model is not stationary
        if not self.is_stationary():
            return None
        def f(h):
            h = np.array(h).reshape(-1,2)  # cast to 2-dimensional array with 2 columns if needed
            if self.alpha != 0:
                hnew = np.dot(h, self.mrot()).reshape(-1,2)
            else:
                hnew = h.reshape(-1,2)

            s = np.zeros(hnew.shape[0])

            for t, d in self.elem:
                # new dictionary from d (remove 'r' key)
                dnew = {key:val for key, val in d.items() if key != 'r'}
                if t == 'nugget':
                    s = s + d['w'] - cov_nug(np.sum(hnew != 0, axis=1), **dnew)

                elif t == 'spherical':
                    s = s + d['w'] - cov_sph(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'exponential':
                    s = s + d['w'] - cov_exp(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'gaussian':
                    s = s + d['w'] - cov_gau(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'triangular':
                    s = s + d['w'] - cov_tri(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'cubic':
                    s = s + d['w'] - cov_cub(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'sinus_cardinal':
                    s = s + d['w'] - cov_sinc(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'gamma':
                    s = s + d['w'] - cov_gamma(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'power':
                    s = s + d['w'] - cov_pow(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'exponential_generalized':
                    s = s + d['w'] - cov_exp_gen(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'matern':
                    s = s + d['w'] - cov_matern(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

            return s

        return f

    def plot_mrot(self, color0='red', color1='green'):
        """
        Plots axes of system Oxy and Ox'y' (in the current figure axis).

        Parameters
        ----------
        color0 : color, default: 'red'
            color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
            the 1st axis (x') supporting the covariance model

        color1 : color, default: 'green'
            color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
            the 2nd axis (y') supporting the covariance model

        Notes
        -----
        No plot is displayed if the model has non-stationary orientation.
        """
        # fname = 'plot_mrot'

        # Prevent calculation if orientation is not stationary
        if not self.is_orientation_stationary():
            return None
        mrot = self.mrot()
        # Plot system Oxy and Ox'y'
        # This:
        plt.arrow(*[0,0], *[0.9,0], color='k', head_width=0.05, head_length=0.1)
        plt.arrow(*[0,0], *[0,0.9], color='k', head_width=0.05, head_length=0.1)
        plt.text(*[1,0], "x", c='k', ha='left', va='top')
        plt.text(*[0,1], "y", c='k', ha='left', va='top')
        plt.arrow(*[0,0], *(0.9*mrot[:,0]), color=color0, head_width=0.05, head_length=0.1)
        plt.arrow(*[0,0], *(0.9*mrot[:,1]), color=color1, head_width=0.05, head_length=0.1)
        plt.text(*mrot[:,0], "x'", c=color0 , ha='right', va='bottom')
        plt.text(*mrot[:,1], "y'", c=color1 , ha='right', va='bottom')
        plt.text(0, 0, "O", c='k', ha='right', va='top')
        plt.xlim(min(min(mrot[0,:]), 0)-0.1, max(max(mrot[0,:]), 1)+0.1)
        plt.ylim(min(min(mrot[1,:]), 0)-0.1, max(max(mrot[1,:]), 1)+0.1)
        plt.gca().set_aspect('equal')
        plt.axis('off')
        # # Or that:
        # plt.arrow(*[0,0], *(0.9*mrot[:,0]), color=color0, head_width=0.05, head_length=0.1)
        # plt.arrow(*[0,0], *(0.9*mrot[:,1]), color=color1, head_width=0.05, head_length=0.1)
        # plt.text(*mrot[:,0], "x'", c=color0, ha='right', va='bottom')
        # plt.text(*mrot[:,1], "y'", c=color1, ha='right', va='bottom')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.xlim(min(min(mrot[0,:]), 0)-0.1, max(max(mrot[0,:]), 1)+0.1)
        # plt.ylim(min(min(mrot[1,:]), 0)-0.1, max(max(mrot[1,:]), 1)+0.1)
        # plt.gca().set_aspect('equal')
        # plt.gca().spines['left'].set_position('zero')
        # plt.gca().spines['left'].set_position('zero')
        # plt.gca().spines['right'].set_color('none')
        # plt.gca().spines['bottom'].set_position('zero')
        # plt.gca().spines['top'].set_color('none')

    def plot_model(
            self,
            vario=False,
            plot_map=True,
            plot_curves=True,
            cmap='terrain',
            color0='red',
            color1='green',
            extent=None,
            ncell=(201, 201),
            h1min=0.0,
            h1max=None,
            h2min=0.0,
            h2max=None,
            n1=500,
            n2=500,
            show_xlabel=True,
            show_ylabel=True,
            grid=True,
            show_suptitle=True,
            figsize=None,
            logger=None):
        """
        Plots the covariance or variogram model.

        The model can be displayed as
        - map of the function, and / or
        - curves along axis x' and axis y' supporting the model.

        If map (`plot_map=True`) and curves (`plot_curves=True`) are displayed,
        a new "1x2" figure is used, if only one of map or curves is displayed,
        the current axis is used.

        Parameters
        ----------
        vario : bool, default: False
            - if False: the covariance model is displayed
            - if True: the variogram model is displayed

        plot_map : bool, default: True
            indicates if (2D) map of the model is displayed

        plot_curves : bool, default: True
            indicates if curves of the model along x' and y' axes are displayed

        cmap : colormap
            color map (can be a string, in this case the color map
            `matplotlib.pyplot.get_cmap(cmap)`

        color0 : color, default: 'red'
            color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
            the curve along the 1st axis (x')

        color1 : color, default: 'green'
            color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
            the curve along the 2nd axis (y')

        extent : sequence of 4 floats, optional
            `extent=(hxmin, hxmax, hymin, hymax)` 4 floats defining the limit of
            the map; by default (`extent=None`), `hxmin`, `hymin` (resp. `hxmax`,
            `hymax`) are set the + (resp. -) 1.2 times max(r1, r2), where r1, r2
            are the ranges along the 1st, 2nd axis respectively

        ncell : sequence of 2 ints, default: (201, 201)
            `ncell=(nx, ny)` 2 ints defining the number of the cells  in the
            map along each direction (in "original" coordinates system)

        h1min : float, default: 0.0
            see `h1max`
        h1max : float, optional
            function (curve) is plotted for h in interval [`h1min`,` h1max`]
            along the 1st axis (x'); by default (`h1max=None`), `h1max` is set to
            1.2 times max(r1, r2), where r1, r2 are the ranges along the 1st and
            2nd axis respectively

        h2min : float, default: 0.0
            see `h2max`
        h2max : float, optional
            function (curve) is plotted for h in interval [`h2min`,` h2max`]
            along the 2nd axis (y'); by default (`h2max=None`), `h2max` is set to
            1.2 times max(r1, r2), where r1, r2 are the ranges along the 1st and
            2nd axis respectively

        n1 : int, default: 500
            number of points for the plot of the curve along the 1st axis, in
            interval [`h1min`,` h1max`]

        n2 : int, default: 500
            number of points for the plot of the curve along the 2nd axis, in
            interval [`h2min`,` h2max`]

        show_xlabel : bool, default: True
            indicates if (default) label for abscissa is displayed

        show_ylabel : bool, default: True
            indicates if (default) label for ordinate is displayed

        grid : bool, default: True
            indicates if a grid is plotted for the plot of the curves

        show_suptitle : bool, default: True
            indicates if (default) suptitle is displayed, if both map and curves
            are plotted (`plot_map=True` and `plot_curves=True`) (in a new "1x2"
            figure)

        figsize : 2-tuple, optional
            size of the new "1x2" figure, if both map and curves are plotted
            (`plot_map=True` and `plot_curves=True`)

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)

        Notes
        -----
        No plot is displayed if the model is not stationary.
        """
        # fname = 'plot_model'

        # Prevent calculation if covariance model is not stationary
        if not self.is_stationary():
            return None
        if not plot_map and not plot_curves:
            return None

        # Set hr to 1.2 * max of ranges, used as default in extent and h1max, h2max below
        r = max(self.r12())
        hr = 1.2 * r

        # Rotation matrix
        mrot = self.mrot()

        if plot_map:
            # Set extent if needed
            if extent is None:
                extent = [-hr, hr, -hr, hr]
            hxmin, hxmax, hymin, hymax = extent

            # Evaluate function on 2D mesh
            nx, ny = ncell
            sx, sy = (hxmax - hxmin) / nx, (hymax - hymin) / ny
            ox, oy = hxmin, hymin
            hx = ox + sx * (0.5 + np.arange(nx))
            hy = oy + sy * (0.5 + np.arange(ny))
            hhx, hhy = np.meshgrid(hx, hy)
            hh = np.hstack((hhx.reshape(-1,1), hhy.reshape(-1,1))) # 2D-lags: (n, 2) array
            gg = self(hh, vario=vario).reshape(ny, nx)
            # if vario:
            #     gg = self.vario_func()(hh).reshape(ny, nx)
            # else:
            #     gg = self.func()(hh).reshape(ny, nx)

            # Set image (Img class)
            im = img.Img(nx=nx, ny=ny, nz=1, sx=sx, sy=sy, sz=1.0, ox=ox, oy=oy, oz=0.0, nv=1, val=gg)

        if plot_curves:
            # Set h1max, h2max if needed
            if h1max is None:
                h1max = hr
            if h2max is None:
                h2max = hr

            # Evaluate function along axis x'
            h1 = np.linspace(h1min, h1max, n1)
            hh1 = np.hstack((h1.reshape(-1,1), np.zeros((len(h1),1)))) # (n1,2) array) 2D-lags along x' expressed in system Ox'y'
            g1 = self(hh1.dot(mrot.T), vario=vario) # hh1.dot(mrot.T): 2D-lags in system Oxy (what is taken by the function)
            # if vario:
            #     g1 = self.vario_func()(hh1.dot(mrot.T)) # hh1.dot(mrot.T): 2D-lags in system Oxy (what is taken by the function)
            # else:
            #     g1 = self.func()(hh1.dot(mrot.T)) # hh1.dot(mrot.T): 2D-lags in system Oxy (what is taken by the function)

            # Evaluate function along axis y'
            h2 = np.linspace(h2min, h2max, n2)
            hh2 = np.hstack((np.zeros((len(h2),1)), h2.reshape(-1,1))) # (n2,2) array) 2D-lags along y' expressed in system Ox'y'
            g2 = self(hh2.dot(mrot.T), vario=vario) # hh1.dot(mrot.T): 2D-lags in system Oxy (what is taken by the function)
            # if vario:
            #     g2 = self.vario_func()(hh2.dot(mrot.T)) # hh2.dot(mrot.T): 2D-lags in system Oxy (what is taken by the function)
            # else:
            #     g2 = self.func()(hh2.dot(mrot.T)) # hh2.dot(mrot.T): 2D-lags in system Oxy (what is taken by the function)

        # Plot...
        if plot_map and plot_curves:
            # Figure (new)
            _, ax = plt.subplots(1,2, figsize=figsize)
            plt.sca(ax[0])

        if plot_map:
            # Plot map and system Ox'y'
            # ... map
            imgplt.drawImage2D(im, cmap=cmap, logger=logger)
            # ... system Ox'y'
            hm1 = 0.9*min(hxmax, hymax)
            hm2 = 0.9*max(hxmax, hymax)
            plt.arrow(*[0,0], *(hm2*mrot[:,0]), color=color0)#, head_width=0.05, head_length=0.1)
            plt.arrow(*[0,0], *(hm2*mrot[:,1]), color=color1)#,    head_width=0.05, head_length=0.1)
            plt.text(*(hm1*mrot[:,0]), "x'", c=color0, ha='right', va='bottom')
            plt.text(*(hm1*mrot[:,1]), "y'", c=color1, ha='right', va='bottom')
            # plt.text(0, 0, "O", c='k', ha='right', va='top')
            # plt.gca().set_aspect('equal')
            plt.xlabel("x")
            plt.ylabel("y")

        if plot_map and plot_curves:
            plt.sca(ax[1])

        if plot_curves:
            # Plot curve along x'
            plt.plot(h1, g1, '-', c=color0, label="along x'")
            # Plot curve along y'
            plt.plot(h2, g2, '-', c=color1, label="along y'")

            if show_xlabel:
                plt.xlabel('h')
            if show_ylabel:
                if vario:
                    plt.ylabel(r'$\gamma(h)$')
                else:
                    plt.ylabel(r'$cov(h)$')

            plt.legend()
            plt.grid(grid)

        if plot_map and plot_curves and show_suptitle:
            if vario:
                s = [f'Model (vario): alpha={self.alpha}'] + [f'{el}' for el in self.elem]
            else:
                s = [f'Model (cov): alpha={self.alpha}'] + [f'{el}' for el in self.elem]
            plt.suptitle('\n'.join(s))
            # plt.show()

    def plot_model_one_curve(
            self,
            main_axis=1,
            vario=False,
            hmin=0.0,
            hmax=None,
            npts=500,
            show_xlabel=True,
            show_ylabel=True,
            grid=True,
            logger=None,
            **kwargs):
        """
        Plots the covariance or variogram curve along one main axis (in the current figure axis).

        Parameters
        ----------
        main_axis : int (1 or 2), default: 1
            - if `main_axis=1`, plots the curve along the 1st axis (x')
            - if `main_axis=2`, plots the curve along the 2nd axis (y')

        vario : bool, default: False
            - if False: the covariance model is displayed
            - if True: the variogram model is displayed

        hmin : float, default: 0.0
            see `hmax`
        hmax : float, optional
            function is plotted for h in interval [`hmin`,` hmax`] along the
            axis specified by `main_axis`; by default (`hmax=None`), `hmax` is
            set to 1.2 times the range of the model along the specified axis

        npts : int, default: 500
            number of points used in interval [`hmin`,` hmax`]

        show_xlabel : bool, default: True
            indicates if (default) label for abscissa is displayed

        show_ylabel : bool, default: True
            indicates if (default) label for ordinate is displayed

        grid : bool, default: True
            indicates if a grid is plotted
            
        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)

        kwargs : dict
            keyword arguments passed to the funtion `matplotlib.pyplot.plot`

        Notes
        -----
        No plot is displayed if the model is not stationary.
        """
        fname = 'plot_model_one_curve'

        # Prevent calculation if covariance model is not stationary
        if not self.is_stationary():
            return None
        if main_axis not in (1, 2):
            err_msg = f'{fname}: `main_axis` invalid (should be 1 or 2)'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

        # In kwargs:
        #   - add default 'label' if not given
        if 'label' not in kwargs.keys():
            if vario:
                kwargs['label'] = 'vario func'
            else:
                kwargs['label'] = 'cov func'

        # Set hmax if needed
        if hmax is None:
            hmax = 1.2*self.r12()[main_axis-1]

        # Rotation matrix
        mrot = self.mrot()

        # Evaluate function along selected axis
        h = np.linspace(hmin, hmax, npts)
        if main_axis == 1:
            hh = np.hstack((h.reshape(-1,1), np.zeros((len(h),1)))) # (npts,2) array of 2D-lags along x' expressed in system Ox'y'
        else:
            hh = np.hstack((np.zeros((len(h),1)), h.reshape(-1,1))) # (npts,2) array of 2D-lags along y' expressed in system Ox'y'
        g = self(hh.dot(mrot.T), vario=vario) # hh.dot(mrot.T): 2D-lags in system Oxy (what is taken by the function)
        # if vario:
        #     g = self.vario_func()(hh.dot(mrot.T)) # hh.dot(mrot.T): 2D-lags in system Oxy (what is taken by the function)
        # else:
        #     g = self.func()(hh.dot(mrot.T)) # hh.dot(mrot.T): 2D-lags in system Oxy (what is taken by the function)

        plt.plot(h, g, **kwargs)

        if show_xlabel:
            plt.xlabel('h')
        if show_ylabel:
            if vario:
                plt.ylabel(r'$\gamma(h)$')
            else:
                plt.ylabel(r'$cov(h)$')

        plt.grid(grid)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
class CovModel3D(object):
    """
    Class defining a covariance model in 3D.

    A covariance model is defined as the sum of elementary covariance models.

    An elementary variogram model is defined as its weight parameter (`w`) minus
    the covariance elementary model, and a variogram model is defined as the sum
    of elementary variogram models.

    This class is callable, returning the evaluation of the model (covariance or
    variogram) at given point(s) (lag(s)).

    **Attributes**

    elem : 1D array-like
        sequence of elementary model(s) (contributing to the covariance model),
        each element of the sequence is a 2-tuple (t, d), where

        - t : str
            type of elementary covariance model, can be

            - 'nugget'         (see function :func:`cov_nug`)
            - 'spherical'      (see function :func:`cov_sph`)
            - 'exponential'    (see function :func:`cov_exp`)
            - 'gaussian'       (see function :func:`cov_gau`)
            - 'triangular'     (see function :func:`cov_tri`)
            - 'cubic'          (see function :func:`cov_cub`)
            - 'sinus_cardinal' (see function :func:`cov_sinc`)
            - 'gamma'          (see function :func:`cov_gamma`)
            - 'power'          (see function :func:`cov_pow`)
            - 'exponential_generalized' (see function :func:`cov_exp_gen`)
            - 'matern'         (see function :func:`cov_matern`)

        - d : dict
            dictionary of required parameters to be passed to the elementary
            model `t`

        e.g.

        - (t, d) = ('spherical', {'w':2.0, 'r':[1.5, 2.5, 3.0]})
        - (t, d) = ('power', {'w':2.0, 'r':[1.5, 2.5, 3.0], 's':1.7})
        - (t, d) = ('matern', {'w':2.0, 'r':[1.5, 2.5, 3.0], 'nu':1.5})

    alpha, beta, gamma: floats, default: 0.0, 0.0, 0.0
        azimuth, dip and plunge angles in degrees; the system Ox'''y''''z''',
        supporting the axes of the model (ranges), is obtained from the system
        Oxyz as follows::

            # Oxyz      -- rotation of angle -alpha around Oz  --> Ox'y'z'
            # Ox'y'z'   -- rotation of angle -beta  around Ox' --> Ox''y''z''
            # Ox''y''z''-- rotation of angle -gamma around Oy''--> Ox'''y'''z'''

        The 3x3 matrix m for changing the coordinates system from Ox'''y'''z'''
        to Oxyz is:

        .. math::
            m = \\left(\\begin{array}{rrr}
                    \\cos\\alpha \\cdot \\cos\\gamma + \\sin\\alpha \\cdot \\sin\\beta \\cdot \\sin\\gamma &  \\sin\\alpha \\cdot \\cos\\beta & - \\cos\\alpha \\cdot \\sin\\gamma + \\sin\\alpha \\cdot \\sin\\beta \\cdot \\cos\\gamma \\\\ \
                  - \\sin\\alpha \\cdot \\cos\\gamma + \\cos\\alpha \\cdot \\sin\\beta \\cdot \\sin\\gamma &  \\cos\\alpha \\cdot \\cos\\beta &   \\sin\\alpha \\cdot \\sin\\gamma + \\cos\\alpha \\cdot \\sin\\beta \\cdot \\cos\\gamma \\\\ \
                                                                           \\cos\\beta \\cdot \\sin\\gamma &                    - \\sin\\beta &                                                          \\cos\\beta \\cdot \\cos\\gamma
                \\end{array}\\right)

    name : str, optional
        name of the model

    **Private attributes (SHOULD NOT BE SET DIRECTLY)**

    _r : float
        maximal (effective) range, along the three axes

    _sill : float
        sill (sum of weight of elementary contributions)

    _mrot : 2D array of shape (3, 3)
        rotation matrix m (see above)

    _is_orientation_stationary : bool
        indicates if the covariance model has stationary orientation

    _is_weight_stationary : bool
        indicates if the covariance model has stationary weight

    _is_range_stationary : bool
        indicates if the covariance model has stationary range(s)

    _is_stationary : bool
        indicates if the covariance model is stationary

    Examples
    --------
    To define a covariance model (3D) that is the sum of the 2 following
    elementary models:

    - gaussian with a contributtion of 9.5 and ranges of 40.0, 20.0 and 10.0,
    - nugget of (contribution, weight) 0.5

    and in the system Ox'''y'''z''' defined by the angle alpha=-30.0, beta=-40.0,
    gamma=20.0

        >>> cov_model = CovModel3D(elem=[
                ('gaussian', {'w':9.5, 'r':[40.0, 20.0, 10.0]}), # elementary contribution
                ('nugget', {'w':0.5})                            # elementary contribution
                ], alpha=-30.0, beta=-40.0, gamma=20.0,          # angles
                name='')                                         # name (optional)

    **Methods**
    """
    #
    # The 3x3 matrix m for changing the coordinates system from Ox'''y'''z'''
    # to Oxyz is:
    #         +                                                            +
    #         |  ca * cc + sa * sb * sc,  sa * cb, - ca * sc + sa * sb * cc|
    #     m = |- sa * cc + ca * sb * sc,  ca * cb,   sa * sc + ca * sb * cc|
    #         |                 cb * sc,     - sb,                  cb * cc|
    #         +                                                            +
    # where
    #     ca = cos(alpha), cb = cos(beta), cc = cos(gamma),
    #     sa = sin(alpha), sb = sin(beta), sc = sin(gamma)
    #
    # Methods
    # -------
    # reset_private_attributes()
    #     Resets private attributes
    # set_alpha(alpha)
    #     Sets (modifies) the attribute `alpha`
    # set_beta(beta)
    #     Sets (modifies) the attribute `beta`
    # set_gamma(gamma)
    #     Sets (modifies) the attribute `gamma`
    # multiply_w(factor, elem_ind=None)
    #     Multiplies parameter `w` of the (given) elementary contribution(s) by the given factor
    # multiply_r(sfactor, r_ind=None, elem_ind=None)
    #     Multiplies (given index(es) of) parameter `r` of the (given) elementary contribution(s) by the given factor
    # is_orientation_stationary(recompute=False)
    #     Checks if the covariance model has stationary orientation
    # is_weight_stationary(recompute=False)
    #     Checks if the covariance model has stationary weight
    # is_range_stationary(recompute=False)
    #     Checks if the covariance model has stationary range
    # is_stationary(recompute=False)
    #     Checks if the covariance model is stationary
    # sill(recompute=False)
    #     Retrieves the sill of the covariance model
    # mrot(recompute=False)
    #     Returns the 2x2 matrix of rotation defining the axes of the model
    # r123(recompute=False)
    #     Returns the (effective) ranges along x''', y''', z''' axes supporting the model
    # rxyz(recompute=False)
    #     Returns the (effective) ranges along x, y, z axes of the "original" coordinates system
    # func()
    #     Returns the function f for the evaluation of the covariance model
    # vario_func()
    #     Returns the function f for the evaluation of the variogram model
    # plot_mrot(self, color0='red', color1='green', color2='blue', set_3d_subplot=True, figsize=None)
    #     Plots axes of system Oxyz and Ox'''y'''z''' (in the current figure axis or a new figure)
    # plot_model3d_volume(plotter=None, vario=False, color0='red', color1='green', color2='blue', extent=None, ncell=(101, 101, 101), **kwargs)
    #     Plots the covariance or variogram model in 3D (volume)
    # plot_model3d_slice(plotter=None, vario=False, color0='red', color1='green', color2='blue', extent=None, ncell=(101, 101, 101), **kwargs)
    #     Plots the covariance or variogram model in 3D (slices)
    # plot_model_curves(plotter=None, vario=False, color0='red', color1='green', color2='blue', h1min=0.0, h1max=None, h2min=0.0, h2max=None, h3min=0.0, h3max=None, n1=500, n2=500, n3=500, grid=True, show_xlabel=True, show_ylabel=True)
    #     Plots the covariance or variogram model along the main axes x''', y''', z''' (in current figure axis)
    # plot_model_one_curve(main_axis=1, vario=False, hmin=0.0, hmax=None, npts=500, grid=True, show_xlabel=True, show_ylabel=True, **kwargs)
    #     Plots the covariance or variogram curve along one main axis (in the current figure axis)
    #
    def __init__(self,
                 elem=[],
                 alpha=0.0, 
                 beta=0.0, 
                 gamma=0.0,
                 name=None,
                 logger=None):
        """
        Inits an instance of the class.

        Parameters
        ----------
        elem : 1D array-like, default: []
            sequence of elementary model(s)

        alpha : float, default: 0.0
            azimuth angle in degrees

        beta : float, default: 0.0
            dip angle in degrees

        gamma : float, default: 0.0
            plunge angle in degrees

        name : str, optional
            name of the model
            
        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'CovModel3D'

        self.elem = elem
        for el in self.elem:
            ok, err_mes_list = check_elem_cov_model(el, verbose=0)
            if not ok:
                err_msg = f'{fname}: elementary contribution not valid\n ... ' + '\n ... '.join(err_mes_list)
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if name is None:
            if len(elem) == 1:
                name = 'cov3D-' + elem[0][0]
            elif len(elem) > 1:
                name = 'cov3D-multi-contribution'
            else:
                name = 'cov3D-zero'
        self.name = name
        self._r = None  # initialize "internal" variable _r for effective range
        self._sill = None  # initialize "internal" variable _sill for sill (sum of weight(s))
        self._mrot = None  # initialize "internal" variable _mrot for rotation matrix
        self._is_orientation_stationary = None
        self._is_weight_stationary = None
        self._is_range_stationary = None
        self._is_stationary = None

    # ------------------------------------------------------------------------
    # def __str__(self):
    #     return self.name
    def __repr__(self):
        # """
        # Returns the string that is displayed when the instance of the class is "printed".
        # """
        out = '*** CovModel3D object ***'
        out = out + '\n' + "name = '{0.name}'".format(self)
        nelem = len(self.elem)
        out = out + '\n' + 'number of elementary contribution(s): {}'.format(len(self.elem))
        for i, el in enumerate(self.elem):
            out = out + '\n' + 'elementary contribution {}'.format(i)
            out = out + '\n' + '    type: {}'.format(el[0])
            out = out + '\n' + '    parameters:'
            nparam = len(el[1])
            for j, (k, val) in enumerate(el[1].items()):
                out = out + '\n' + '        {} = {}'.format(k, val)
        out = out + '\n' + 'angles: alpha = {0.alpha}, beta = {0.beta}, gamma = {0.gamma} (in degrees)'.format(self)
        out = out + '\n' + "    i.e.: the system Ox'''y''''z''', supporting the axes of the model (ranges),"
        out = out + '\n' + "    is obtained from the system Oxyz as follows:"
        out = out + '\n' + "        Oxyz      -- rotation of angle -alpha around Oz  --> Ox'y'z'"
        out = out + '\n' + "        Ox'y'z'   -- rotation of angle -beta  around Ox' --> Ox''y''z''"
        out = out + '\n' + "        Ox''y''z''-- rotation of angle -gamma around Oy''--> Ox'''y'''z'''"
        out = out + '\n' + '*****'
        return out
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    def __call__(self, h, vario=False):
        """
        Evaluates the covariance model at given 3D lags (`h`).

        Parameters
        ----------
        h : 2D array-like of shape (n, 3) or 1D array-like of shape (3,)
            point(s) (lag(s)) where the covariance model is evaluated;
            if `h` is a 2D array, each row is a lag

        vario : bool, default: False
            - if False: computes the covariance
            - if True: computes the variogram

        Returns
        -------
        y : 1D array
            evaluation of the covariance or variogram model at `h`;
            note: the result is casted to a 1D array if `h` is a 1D array
        """
        if vario:
            return self.vario_func()(h)
        else:
            return self.func()(h)
    # ------------------------------------------------------------------------

    def reset_private_attributes(self):
        """
        Resets (sets to `None`) the "private" attributes (beginning with "_").
        """
        # fname = 'reset_private_attributes'

        self._r = None
        self._sill = None
        self._mrot = None
        self._is_orientation_stationary = None
        self._is_weight_stationary = None
        self._is_range_stationary = None
        self._is_stationary = None

    def set_alpha(self, alpha):
        """
        Sets (modifies) the attribute `alpha`.

        The covariance model is updated and relevant private attributes
        (beginning with "_") are reset.

        Parameters
        ----------
        alpha : array of float or float
            azimuth angle in degrees; if array, its shape must be compatible with
            the dimension of the grid on which the covariance model is used (for
            Gaussian interpolation or simulation)
        """
        # fname = 'set_alpha'

        self.alpha = alpha
        self._mrot = None
        self._is_orientation_stationary = None
        self._is_stationary = None

    def set_beta(self, beta):
        """
        Sets (modifies) the attribute `beta`.

        The covariance model is updated and relevant private attributes
        (beginning with "_") are reset.

        Parameters
        ----------
        beta : array of float or float
            dip angle in degrees; if array, its shape must be compatible with
            the dimension of the grid on which the covariance model is used (for
            Gaussian interpolation or simulation)
        """
        # fname = 'set_beta'

        self.beta = beta
        self._mrot = None
        self._is_orientation_stationary = None
        self._is_stationary = None

    def set_gamma(self, gamma):
        """
        Sets (modifies) the attribute `gamma`.

        The covariance model is updated and relevant private attributes
        (beginning with "_") are reset.

        Parameters
        ----------
        gamma : array of float or float
            plunge angle in degrees; if array, its shape must be compatible with
            the dimension of the grid on which the covariance model is used (for
            Gaussian interpolation or simulation)
        """
        # fname = 'set_gamma'

        self.gamma = gamma
        self._mrot = None
        self._is_orientation_stationary = None
        self._is_stationary = None

    def multiply_w(self, factor, elem_ind=None, logger=None):
        """
        Multiplies parameter `w` of the (given) elementary contribution(s) by the given factor.

        The covariance model is updated and relevant private attributes
        (beginning with "_") are reset.

        Parameters
        ----------
        factor : array of floats or float
            multiplier(s), if array, its shape must be compatible with the
            dimension of the grid on which the covariance model is used (for
            Gaussian interpolation or simulation)

        elem_ind : 1D array-like of ints, or int, optional
            indexe(s) of the elementary contribution (attribute `elem`) to be
            modified; by default (`None`): indexes of any elementary contribution
            are selected
            
        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'multiply_w'

        if elem_ind is None:
            elem_ind = np.arange(len(self.elem))
        else:
            elem_ind = np.atleast_1d(elem_ind).reshape(-1)
            n = len(self.elem)
            if np.any((elem_ind > n - 1, elem_ind < -n)):
                err_msg = f'{fname}: `elem_ind` invalid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

        for i in elem_ind:
            if 'w' in self.elem[i][1].keys():
                self.elem[i][1]['w'] = factor * self.elem[i][1]['w']

        self._sill = None
        self._is_weight_stationary = None
        self._is_stationary = None

    def add_w(self, a, elem_ind=None, logger=None):
        """
        Add `a`to parameter `w` of the (given) elementary contribution(s).

        The covariance model is updated and relevant private attributes
        (beginning with "_") are reset.

        Parameters
        ----------
        a : array of floats or float
            term(s) to add, if array, its shape must be compatible with the
            dimension of the grid on which the covariance model is used (for
            Gaussian interpolation or simulation)

        elem_ind : 1D array-like of ints, or int, optional
            indexe(s) of the elementary contribution (attribute `elem`) to be
            modified; by default (`None`): indexes of any elementary contribution
            are selected
            
        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'add_w'

        if elem_ind is None:
            elem_ind = np.arange(len(self.elem))
        else:
            elem_ind = np.atleast_1d(elem_ind).reshape(-1)
            n = len(self.elem)
            if np.any((elem_ind > n - 1, elem_ind < -n)):
                err_msg = f'{fname}: `elem_ind` invalid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

        for i in elem_ind:
            if 'w' in self.elem[i][1].keys():
                new_w = self.elem[i][1]['w'] + a
                if new_w < 0:
                    err_msg = f'{fname}: new `w` is negative'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)

                self.elem[i][1]['w'] = new_w

        self._sill = None
        self._is_weight_stationary = None
        self._is_stationary = None

    def multiply_r(self, factor, r_ind=None, elem_ind=None, logger=None):
        """
        Multiplies (given index(es) of) parameter `r` of the (given) elementary contribution(s) by the given factor.

        The covariance model is updated and relevant private attributes
        (beginning with "_") are reset.

        Parameters
        ----------
        factor : array of floats or float
            multiplier(s), if array, its shape must be compatible with the
            dimension of the grid on which the covariance model is used (for
            Gaussian interpolation or simulation)

        r_ind : int or sequence of ints, optional
            indexe(s) of the parameter `r` of elementary contribution to be
            modified; by default (`None`): `r_ind=(0, 1, 2)` is used, i.e.
            parameter `r` along each axis is multiplied

        elem_ind : 1D array-like of ints, or int, optional
            indexe(s) of the elementary contribution (attribute `elem`) to be
            modified; by default (`None`): indexes of any elementary contribution
            are selected
            
        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)
        """
        fname = 'multiply_r'

        if r_ind is None:
            r_ind = (0, 1, 2)
        else:
            r_ind = np.atleast_1d(r_ind).reshape(-1)
            if np.any((r_ind > 2, r_ind < -3)):
                err_msg = f'{fname}: `r_ind` invalid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

        if elem_ind is None:
            elem_ind = np.arange(len(self.elem))
        else:
            elem_ind = np.atleast_1d(elem_ind).reshape(-1)
            n = len(self.elem)
            if np.any((elem_ind > n - 1, elem_ind < -n)):
                err_msg = f'{fname}: `elem_ind` invalid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

        for i in elem_ind:
            if 'r' in self.elem[i][1].keys():
                for j in r_ind:
                    self.elem[i][1]['r'][j] = factor * self.elem[i][1]['r'][j]

        self._r = None
        self._is_range_stationary = None
        self._is_stationary = None

    def is_orientation_stationary(self, recompute=False):
        """
        Checks if the covariance model has stationary orientation.

        Parameters
        ----------
        recompute : bool, default: False
            True to force (re-)computing

        Returns
        -------
        flag : bool
            boolean indicating if the orientation is stationary, i.e. attritbutes
            `alpha`, `beta`, `gamma` are defined as a unique value
            (private attribute `_is_orientation_stationary`)
        """
        # fname = 'is_orientation_stationary'

        if self._is_orientation_stationary is None or recompute:
            self._is_orientation_stationary = np.size(self.alpha) == 1 and np.size(self.beta) == 1 and np.size(self.gamma) == 1
        return self._is_orientation_stationary

    def is_weight_stationary(self, recompute=False):
        """
        Checks if the covariance model has stationary weight.

        Parameters
        ----------
        recompute : bool, default: False
            True to force (re-)computing

        Returns
        -------
        flag : bool
            boolean indicating if the weight (parameter `w`) of every elementary
            contribution is stationary (defined as a unique value)
            (private attribute `_is_weight_stationary`)
        """
        # fname = is_weight_stationary

        if self._is_weight_stationary is None or recompute:
            self._is_weight_stationary = not np.any([np.size(el[1]['w']) > 1 for el in self.elem])
        return self._is_weight_stationary

    def is_range_stationary(self, recompute=False):
        """
        Checks if the covariance model has stationary ranges.

        Parameters
        ----------
        recompute : bool, default: False
            True to force (re-)computing

        Returns
        -------
        flag : bool
            boolean indicating if the range along each axis (parameter `r`)
            of every elementary contribution is stationary (`r[i]` defined as a
            unique value)
            (private attribute `_is_range_stationary`)
        """
        # fname = 'is_range_stationary'

        if self._is_range_stationary is None or recompute:
            self._is_range_stationary = True
            for el in self.elem:
                if 'r' in el[1].keys() and np.any([np.size(ri) > 1 for ri in el[1]['r']]):
                    self._is_range_stationary = False
                    break
        return self._is_range_stationary

    def is_stationary(self, recompute=False):
        """
        Checks if the covariance model is stationary.

        Parameters
        ----------
        recompute : bool, default: False
            True to force (re-)computing

        Returns
        -------
        flag : bool
            boolean indicating if all the parameters are stationary (defined as
            a unique value)
            (private attribute `_is_stationary`)
        """
        # fname = 'is_stationary'

        if self._is_stationary is None or recompute:
            self._is_stationary = self.is_orientation_stationary(recompute) and self.is_weight_stationary(recompute) and self.is_range_stationary(recompute)
            if self._is_stationary:
                for t, d in self.elem:
                    flag = True
                    for k, v in d.items():
                        if k in ('w', 'r'):
                            continue
                        if np.size(v) > 1:
                            flag = False
                            break
                    if not flag:
                        self._is_stationary = False
                        break
        return self._is_stationary

    def sill(self, recompute=False):
        """
        Retrieves the sill of the covariance model.

        Parameters
        ----------
        recompute : bool, default: False
            True to force (re-)computing

        Returns
        -------
        sill : float
            sill, sum of the weights of all elementary contributions
            (private attribute `_sill`)

        Notes
        -----
        Nothing is returned if the model has non-stationary weight
        (return `None`).
        """
        # fname = 'sill'

        if self._sill is None or recompute:
            # Prevent calculation if weight is not stationary
            if not self.is_weight_stationary(recompute):
                self._sill = None
                return self._sill

            self._sill = sum([d['w'] for t, d in self.elem if 'w' in d])

        return self._sill

    def mrot(self, recompute=False):
        """
        Returns the 3x3 matrix of rotation defining the axes of the model.

        The 3x3 matrix m is the matrix of changes of coordinate system,
        from Ox'''y'''z''' to Oxyz, where Ox''', Oy''' and Oz''' are the axes
        supporting the ranges of the model.

        Parameters
        ----------
        recompute : bool, default: False
            True to force (re-)computing

        Returns
        -------
        mrot : 2D array of shape (3, 3)
            rotation matrix (private attribute _mrot)

        Notes
        -----
        Nothing is returned if the model has non-stationary orientation
        (return `None`).
        """
        # fname = 'mrot'

        if self._mrot is None or recompute:
            # Prevent calculation if orientation is not stationary
            if not self.is_orientation_stationary(recompute):
                self._mrot = None
                return self._mrot

            self._mrot = rotationMatrix3D(self.alpha, self.beta, self.gamma)

        return self._mrot

    def r123(self, recompute=False):
        """
        Returns the (effective) ranges along x''', y''', z''' axes supporting the model.

        The effective range of the model (in a given direction) is the maximum
        of the effective range of all elementary contributions; note that the
        "effective" range is the distance beyond which the covariance is zero or
        below 5% of the weight, and corresponds to the (components of the)
        parameter `r` for most of elementary covariance models.

        Parameters
        ----------
        recompute : bool, default: False
            True to force (re-)computing

        Returns
        -------
        range : 1D array of shape (3,)
            (effective) ranges along x''', y''', z''' axes supporting the model
            (private attribute `_r`)

        Notes
        -----
        Nothing is returned if the model has non-stationary ranges
        (return `None`).
        """
        # fname = 'r123'

        if self._r is None or recompute:
            # Prevent calculation if range is not stationary
            if not self.is_range_stationary(recompute):
                self._r = None
                return self._r

            r = np.array([0., 0., 0.])
            for t, d in self.elem:
                if t in (
                        'spherical',
                        'exponential',
                        'gaussian',
                        'triangular',
                        'cubic',
                        'sinus_cardinal',
                        'gamma',
                        'power', # not really the range for this case
                        'exponential_generalized',
                        ):
                    r = np.maximum(r, d['r']) # element-wise maximum

                elif t == 'matern':
                    for i, ri in enumerate(d['r']):
                        r[i] = max(r[i], cov_matern_get_effective_range(d['nu'], ri))

            self._r = r

        return self._r

    def rxyz(self, recompute=False):
        """
        Returns the (effective) ranges along x, y, z axes of the "original" coordinates system.

        The effective range of the model (in a given direction) is the maximum
        of the effective range of all elementary contributions; note that the
        "effective" range is the distance beyond which the covariance is zero or
        below 5% of the weight, and corresponds to the (components of the)
        parameter `r` for most of elementary covariance models.

        Parameters
        ----------
        recompute : bool, default: False
            True to force (re-)computing

        Returns
        -------
        range : 1D array of shape (3,)
            (effective) ranges along x, y, z axes of the "original" coordinates
            system

        Notes
        -----
        Nothing is returned if the model has non-stationary ranges or non
        stationary orientation (return `None`).
        """
        # fname = 'rxyz'

        # Prevent calculation if range or orientation is not stationary
        if not self.is_range_stationary(recompute) or not self.is_orientation_stationary(recompute):
            return None
        r123 = self.r123(recompute)
        m = np.abs(self.mrot(recompute))
        return np.maximum(r123[0] * m[:,0], r123[1] * m[:,1], r123[2] * m[:,2]) # element-wise maximum

    def func(self):
        """
        Returns the function f for the evaluation of the covariance model.

        Returns
        -------
        f : function
            function with parameters (arguments):

            - h : 2D array-like of shape (n, 3) or 1D array-like of shape (3,)
                point(s) (lag(s)) where the covariance model is evaluated;
                if `h` is a 2D array, each row is a lag

            that returns:

            - f(h) : 1D array
                evaluation of the covariance model at `h`;
                note: the result is casted to a 1D array if `h` is a 1D array

        Notes
        -----
        No evaluation is done if the model is not stationary (return `None`).
        """
        # fname = 'func'

        # Prevent calculation if covariance model is not stationary
        if not self.is_stationary():
            return None
        def f(h):
            h = np.array(h).reshape(-1,3)  # cast to 2-dimensional array with 3 columns if needed
            if self.alpha != 0 or self.beta != 0 or self.gamma != 0:
                hnew = np.dot(h, self.mrot()).reshape(-1,3)
            else:
                hnew = h.reshape(-1,3)

            s = np.zeros(hnew.shape[0])

            for t, d in self.elem:
                # new dictionary from d (remove 'r' key)
                dnew = {key:val for key, val in d.items() if key != 'r'}
                if t == 'nugget':
                    s = s + cov_nug(np.sum(hnew != 0, axis=1), **dnew)

                elif t == 'spherical':
                    s = s + cov_sph(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'exponential':
                    s = s + cov_exp(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'gaussian':
                    s = s + cov_gau(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'triangular':
                    s = s + cov_tri(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'cubic':
                    s = s + cov_cub(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'sinus_cardinal':
                    s = s + cov_sinc(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'gamma':
                    s = s + cov_gamma(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'power':
                    s = s + cov_pow(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'exponential_generalized':
                    s = s + cov_exp_gen(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'matern':
                    s = s + cov_matern(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

            return s

        return f

    def vario_func(self):
        """
        Returns the function f for the evaluation of the variogram model.

        Returns
        -------
        f : function
            function with parameters (arguments):

            - h : 2D array-like of shape (n, 3) or 1D array-like of shape (3,)
                point(s) (lag(s)) where the variogram model is evaluated;
                if `h` is a 2D array, each row is a lag

            that returns:

            - f(h) : 1D array
                evaluation of the variogram model at `h`;
                note: the result is casted to a 1D array if `h` is a 1D array

        Notes
        -----
        No evaluation is done if the model is not stationary (return `None`).
        """
        # fname = 'vario_func'

        # Prevent calculation if covariance model is not stationary
        if not self.is_stationary():
            return None
        def f(h):
            h = np.array(h).reshape(-1,3)  # cast to 2-dimensional array with 3 columns if needed
            if self.alpha != 0 or self.beta != 0 or self.gamma != 0:
                hnew = np.dot(h, self.mrot()).reshape(-1,3)
            else:
                hnew = h.reshape(-1,3)

            s = np.zeros(hnew.shape[0])

            for t, d in self.elem:
                # new dictionary from d (remove 'r' key)
                dnew = {key:val for key, val in d.items() if key != 'r'}
                if t == 'nugget':
                    s = s + d['w'] - cov_nug(np.sum(hnew != 0, axis=1), **dnew)

                elif t == 'spherical':
                    s = s + d['w'] - cov_sph(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'exponential':
                    s = s + d['w'] - cov_exp(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'gaussian':
                    s = s + d['w'] - cov_gau(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'triangular':
                    s = s + d['w'] - cov_tri(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'cubic':
                    s = s + d['w'] - cov_cub(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'sinus_cardinal':
                    s = s + d['w'] - cov_sinc(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'gamma':
                    s = s + d['w'] - cov_gamma(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'power':
                    s = s + d['w'] - cov_pow(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'exponential_generalized':
                    s = s + d['w'] - cov_exp_gen(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'matern':
                    s = s + d['w'] - cov_matern(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

            return s

        return f

    def plot_mrot(self, color0='red', color1='green', color2='blue', set_3d_subplot=True, figsize=None):
        """
        Plots axes of system Oxyz and Ox'''y'''z''' (in the current figure axis or a new figure).

        Parameters
        ----------
        color0 : color, default: 'red'
            color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
            the 1st axis (x''') supporting the covariance model

        color1 : color, default: 'green'
            color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
            the 2nd axis (y''') supporting the covariance model

        color2 : color, default: 'blue'
            color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
            the 2nd axis (z''') supporting the covariance model

        set_3d_subplot : bool, default: True
            - if True: a new figure is created, with "projection 3d" subplot
            - if False: the current axis is used for the plot

        figsize : 2-tuple, optional
            size of the new "1x2" figure (if `set_3d_subplot=True`)

        Notes
        -----
        No plot is displayed if the model has non-stationary orientation.
        """
        # fname = 'plot_mrot'

        # Prevent calculation if orientation is not stationary
        if not self.is_orientation_stationary():
            return None
        mrot = self.mrot()

        if set_3d_subplot:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(projection='3d')
        else:
            ax = plt.gca()

        # Plot system Oxzy and Ox'''y'''z'''
        # This:
        ax.plot([0,1], [0,0], [0,0], color='k')
        ax.plot([0,0], [0,1], [0,0], color='k')
        ax.plot([0,0], [0,0], [0,1], color='k')
        ax.plot([0, mrot[0,0]], [0, mrot[1,0]], [0, mrot[2,0]], color=color0, label="x'''")
        ax.plot([0, mrot[0,1]], [0, mrot[1,1]], [0, mrot[2,1]], color=color1, label="y'''")
        ax.plot([0, mrot[0,2]], [0, mrot[1,2]], [0, mrot[2,2]], color=color2, label="z'''")
        ax.set_xticks([0,1])
        ax.set_yticks([0,1])
        ax.set_zticks([0,1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()
        # plt.sca(ax)
        # plt.title("System Ox'''y'''z'''")
        # plt.show()

    def plot_model3d_volume(
            self,
            plotter=None,
            vario=False,
            color0='red',
            color1='green',
            color2='blue',
            extent=None,
            ncell=(101, 101, 101),
            logger=None,
            **kwargs):
        """
        Plots the covariance or variogram model in 3D (volume).

        The plot is done using the function :func:`geone.imgplot3d.drawImage3D_volume`
        (based on `pyvista`).

        Parameters
        ----------
        plotter : :class:`pyvista.Plotter`, optional
            - if given (not `None`), add element to the plotter, a further call to \
            `plotter.show()` will be required to show the plot
            - if not given (`None`, default): a plotter is created and the plot \
            is shown

        vario : bool, default: False
            - if False: the covariance model is displayed
            - if True: the variogram model is displayed

        color0 : color, default: 'red'
            color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
            the 1st axis (x''') supporting the covariance model

        color1 : color, default: 'green'
            color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
            the 2nd axis (y''') supporting the covariance model

        color2 : color, default: 'blue'
            color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
            the 3rd axis (z''') supporting the covariance model

        extent : sequence of 6 floats, optional
            `extent=(hxmin, hxmax, hymin, hymax, hzmin, hzmax)` 6 floats defining
            the limit of the map; by default (`extent=None`), `hxmin`, `hymin`,
            `hzmin` (resp. `hxmax`, `hymax`, `hzmax`) are set the + (resp. -)
            1.2 times max(r1, r2, r3), where r1, r2, r3 are the ranges along
            the 1st, 2nd, 3rd axis respectively

        ncell : sequence of 3 ints, default: (101, 101, 101)
            `ncell=(nx, ny, nz)` 3 ints defining the number of the cells in the
            plot along each direction (in "original" coordinates system)

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)

        kwargs : dict
            keyword arguments passed to the funtion
            `geone.imgplot3d.drawImage3D_volume` (cmap, etc.)

        Notes
        -----
        No plot is displayed if the model is not stationary.
        """
        # fname = 'plot_model3d_volume'

        # Prevent calculation if covariance model is not stationary
        if not self.is_stationary():
            return None
        # Set extent if needed
        r = max(self.r123())
        hr = 1.2 * r

        if extent is None:
            extent = [-hr, hr, -hr, hr, -hr, hr]
        hxmin, hxmax, hymin, hymax, hzmin, hzmax = extent

        # Rotation matrix
        mrot = self.mrot()

        # Evaluate function on 3D mesh
        nx, ny, nz = ncell
        sx, sy, sz = (hxmax - hxmin) / nx, (hymax - hymin) / ny, (hzmax - hzmin) / nz
        ox, oy, oz = hxmin, hymin, hzmin
        hx = ox + sx * (0.5 + np.arange(nx))
        hy = oy + sy * (0.5 + np.arange(ny))
        hz = oz + sz * (0.5 + np.arange(nz))
        hhz, hhy, hhx = np.meshgrid(hz, hy, hx, indexing='ij')
        hh = np.hstack((hhx.reshape(-1,1), hhy.reshape(-1,1), hhz.reshape(-1,1))) # 3D-lags: (n, 3) array
        gg = self(hh, vario=vario).reshape(nz, ny, nx)
        # if vario:
        #     gg = self.vario_func()(hh).reshape(nz, ny, nx)
        # else:
        #     gg = self.func()(hh).reshape(nz, ny, nx)

        # Set image (Img class)
        im = img.Img(nx=nx, ny=ny, nz=nz, sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz, nv=1, val=gg)

        # In kwargs (for imgplt3d.drawImage3D_slice):
        #   - set color map 'cmap'
        #   - set 'show_bounds' to True
        #   - set 'scalar_bar_kwargs'
        if 'cmap' not in kwargs.keys():
            kwargs['cmap'] = 'terrain'
        if 'show_bounds' not in kwargs.keys():
            kwargs['show_bounds'] = True
        if 'scalar_bar_kwargs' not in kwargs.keys():
            if vario:
                title='vario'
            else:
                title='cov'
            kwargs['scalar_bar_kwargs'] = {'vertical':True, 'title':title, 'title_font_size':16, 'label_font_size':12}

        # Set plotter if not given
        plotter_show = False
        if plotter is None:
            plotter = pv.Plotter()
            plotter_show = True

        # plot slices in 3D
        imgplt3.drawImage3D_volume(im, plotter=plotter, logger=logger, **kwargs)
        # add main axis x''' (cyl1), y''' (cyl2), z''' (cyl3)
        height = min(hxmax-hxmin, hymax-hymin, hzmax-hzmin)
        radius = 0.005*height
        cyl1 = pv.Cylinder(center=(0.0, 0.0, 0.0), direction=mrot[:,0], radius=radius, height=height, resolution=100, capping=True)
        cyl2 = pv.Cylinder(center=(0.0, 0.0, 0.0), direction=mrot[:,1], radius=radius, height=height, resolution=100, capping=True)
        cyl3 = pv.Cylinder(center=(0.0, 0.0, 0.0), direction=mrot[:,2], radius=radius, height=height, resolution=100, capping=True)
        plotter.add_mesh(cyl1, color=color0)
        plotter.add_mesh(cyl2, color=color1)
        plotter.add_mesh(cyl3, color=color2)

        if plotter_show:
            plotter.show()

    def plot_model3d_slice(
            self,
            plotter=None,
            vario=False,
            color0='red',
            color1='green',
            color2='blue',
            extent=None,
            ncell=(101, 101, 101),
            logger=None,
            **kwargs):
        """
        Plots the covariance or variogram model in 3D (slices).

        The plot is done using the function :func:`geone.imgplot3d.drawImage3D_slice`
        (based on `pyvista`).

        If 'slice_normal_custom' is not an item of `kwargs`, it is added by
        setting slices (to be plotted), orthogonal to axes x''', y''', z''' and
        going through origin (point (0.0, 0.0, 0.0)).

        Parameters
        ----------
        plotter : :class:`pyvista.Plotter`, optional
            - if given (not `None`), add element to the plotter, a further call to \
            `plotter.show()` will be required to show the plot
            - if not given (`None`, default): a plotter is created and the plot \
            is shown

        vario : bool, default: False
            - if False: the covariance model is displayed
            - if True: the variogram model is displayed

        color0 : color, default: 'red'
            color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
            the 1st axis (x''') supporting the covariance model

        color1 : color, default: 'green'
            color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
            the 2nd axis (y''') supporting the covariance model

        color2 : color, default: 'blue'
            color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
            the 2nd axis (z''') supporting the covariance model

        extent : sequence of 6 floats, optional
            `extent=(hxmin, hxmax, hymin, hymax, hzmin, hzmax)` 6 floats defining
            the limit of the map; by default (`extent=None`), `hxmin`, `hymin`,
            `hzmin` (resp. `hxmax`, `hymax`, `hzmax`) are set the + (resp. -)
            1.2 times max(r1, r2, r3), where r1, r2, r3 are the ranges along
            the 1st, 2nd, 3rd axis respectively

        ncell : sequence of 3 ints, default: (101, 101, 101)
            `ncell=(nx, ny, nz)` 3 ints defining the number of the cells in the
            plot along each direction (in "original" coordinates system)

        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)

        kwargs : dict
            keyword arguments passed to the funtion
            `geone.imgplot3d.drawImage3D_slice` (cmap, etc.)

        Notes
        -----
        No plot is displayed if the model is not stationary.
        """
        # fname = 'plot_model3d_slice'

        # Prevent calculation if covariance model is not stationary
        if not self.is_stationary():
            return None
        # Set extent if needed
        r = max(self.r123())
        hr = 1.2 * r

        if extent is None:
            extent = [-hr, hr, -hr, hr, -hr, hr]
        hxmin, hxmax, hymin, hymax, hzmin, hzmax = extent

        # Rotation matrix
        mrot = self.mrot()

        # Evaluate function on 3D mesh
        nx, ny, nz = ncell
        sx, sy, sz = (hxmax - hxmin) / nx, (hymax - hymin) / ny, (hzmax - hzmin) / nz
        ox, oy, oz = hxmin, hymin, hzmin
        hx = ox + sx * (0.5 + np.arange(nx))
        hy = oy + sy * (0.5 + np.arange(ny))
        hz = oz + sz * (0.5 + np.arange(nz))
        hhz, hhy, hhx = np.meshgrid(hz, hy, hx, indexing='ij')
        hh = np.hstack((hhx.reshape(-1,1), hhy.reshape(-1,1), hhz.reshape(-1,1))) # 3D-lags: (n, 3) array
        gg = self(hh, vario=vario).reshape(nz, ny, nx)
        # if vario:
        #     gg = self.vario_func()(hh).reshape(nz, ny, nx)
        # else:
        #     gg = self.func()(hh).reshape(nz, ny, nx)

        # Set image (Img class)
        im = img.Img(nx=nx, ny=ny, nz=nz, sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz, nv=1, val=gg)

        # In kwargs (for imgplt3d.drawImage3D_slice):
        #   - add 'slice_normal_custom' (orthogonal to axes x''', y''', z''' and going through origin) if not given
        #   - set color map 'cmap'
        #   - set 'show_bounds' to True
        #   - set 'scalar_bar_kwargs'
        if 'slice_normal_custom' not in kwargs.keys():
            kwargs['slice_normal_custom'] = [[mrot[:,0], (0,0,0)], [mrot[:,1], (0,0,0)], [mrot[:,2], (0,0,0)]]
        if 'cmap' not in kwargs.keys():
            kwargs['cmap'] = 'terrain'
        if 'show_bounds' not in kwargs.keys():
            kwargs['show_bounds'] = True
        if 'scalar_bar_kwargs' not in kwargs.keys():
            if vario:
                title='vario'
            else:
                title='cov'
            kwargs['scalar_bar_kwargs'] = {'vertical':True, 'title':title, 'title_font_size':16, 'label_font_size':12}

        # Set plotter if not given
        plotter_show = False
        if plotter is None:
            plotter = pv.Plotter()
            plotter_show = True

        # plot slices in 3D
        imgplt3.drawImage3D_slice(im, plotter=plotter, logger=logger, **kwargs)
        # add main axis x''' (cyl1), y''' (cyl2), z''' (cyl3)
        height = min(hxmax-hxmin, hymax-hymin, hzmax-hzmin)
        radius = 0.005*height
        cyl1 = pv.Cylinder(center=(0.0, 0.0, 0.0), direction=mrot[:,0], radius=radius, height=height, resolution=100, capping=True)
        cyl2 = pv.Cylinder(center=(0.0, 0.0, 0.0), direction=mrot[:,1], radius=radius, height=height, resolution=100, capping=True)
        cyl3 = pv.Cylinder(center=(0.0, 0.0, 0.0), direction=mrot[:,2], radius=radius, height=height, resolution=100, capping=True)
        plotter.add_mesh(cyl1, color=color0)
        plotter.add_mesh(cyl2, color=color1)
        plotter.add_mesh(cyl3, color=color2)

        if plotter_show:
            plotter.show()

    def plot_model_curves(
            self,
            plotter=None,
            vario=False,
            color0='red',
            color1='green',
            color2='blue',
            h1min=0.0,
            h1max=None,
            h2min=0.0,
            h2max=None,
            h3min=0.0,
            h3max=None,
            n1=500,
            n2=500,
            n3=500,
            show_xlabel=True,
            show_ylabel=True,
            grid=True):
        """
        Plots the covariance or variogram model along the main axes x''', y''', z''' (in current figure axis).

        Parameters
        ----------
        vario : bool, default: False
            - if False: the covariance model is displayed
            - if True: the variogram model is displayed

        color0 : color, default: 'red'
            color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
            the curve along the 1st axis (x''')

        color1 : color, default: 'green'
            color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
            the curve along the 2nd axis (y''')

        color2 : color, default: 'blue'
            color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
            the curve along the 2nd axis (z''')

        h1min : float, default: 0.0
            see `h1max`
        h1max : float, optional
            function (curve) is plotted for h in interval [`h1min`,` h1max`]
            along the 1st axis (x'''); by default (`h1max=None`), `h1max` is set
            to 1.2 times max(r1, r2, r3), where r1, r2, r3 are the ranges along
            the 1st, 2nd, 3rd axis respectively

        h2min : float, default: 0.0
            see `h2max`
        h2max : float, optional
            function (curve) is plotted for h in interval [`h2min`,` h2max`]
            along the 2nd axis (y'''); by default (`h2max=None`), `h2max` is set
            to 1.2 times max(r1, r2, r3), where r1, r2, r3 are the ranges along
            the 1st, 2nd, 3rd axis respectively

        h3min : float, default: 0.0
            see `32max`
        h3max : float, optional
            function (curve) is plotted for h in interval [`h3min`,` h3max`]
            along the 3rd axis (z'''); by default (`h3max=None`), `h3max` is set
            to 1.2 times max(r1, r2, r3), where r1, r2, r3 are the ranges along
            the 1st, 2nd, 3rd axis respectively

        n1 : int, default: 500
            number of points for the plot of the curve along the 1st axis, in
            interval [`h1min`,` h1max`]

        n2 : int, default: 500
            number of points for the plot of the curve along the 2nd axis, in
            interval [`h2min`,` h2max`]

        n3 : int, default: 500
            number of points for the plot of the curve along the 3rd axis, in
            interval [`h3min`,` h3max`]

        show_xlabel : bool, default: True
            indicates if (default) label for abscissa is displayed

        show_ylabel : bool, default: True
            indicates if (default) label for ordinate is displayed

        grid : bool, default: True
            indicates if a grid is plotted for the plot of the curves

        Notes
        -----
        No plot is displayed if the model is not stationary.
        """
        # fname = 'plot_model_curves'

        # Prevent calculation if covariance model is not stationary
        if not self.is_stationary():
            return None
        # Set h1max, h2max, h3max if needed
        r = max(self.r123())
        hr = 1.2 * r

        # Set h1max, h2max if needed
        if h1max is None:
            h1max = hr
        if h2max is None:
            h2max = hr
        if h3max is None:
            h3max = hr

        # Rotation matrix
        mrot = self.mrot()

        # Evaluate function along axis x'''
        h1 = np.linspace(h1min, h1max, n1)
        hh1 = np.hstack((h1.reshape(-1,1), np.zeros((len(h1),1)), np.zeros((len(h1),1)))) # (n1,3) array) 3D-lags along x''' expressed in system Ox''y'''z''''
        g1 = self(hh1.dot(mrot.T), vario=vario) # hh1.dot(mrot.T): 3D-lags in system Oxyz (what is taken by the function)
        # if vario:
        #     g1 = self.vario_func()(hh1.dot(mrot.T)) # hh1.dot(mrot.T): 3D-lags in system Oxyz (what is taken by the function)
        # else:
        #     g1 = self.func()(hh1.dot(mrot.T)) # hh1.dot(mrot.T): 3D-lags in system Oxz (what is taken by the function)

        # Evaluate function along axis y'''
        h2 = np.linspace(h2min, h2max, n2)
        hh2 = np.hstack((np.zeros((len(h2),1)), h2.reshape(-1,1), np.zeros((len(h2),1)))) # (n1,3) array) 3D-lags along y''' expressed in system Ox''y'''z''''
        g2 = self(hh2.dot(mrot.T), vario=vario) # hh1.dot(mrot.T): 3D-lags in system Oxyz (what is taken by the function)
        # if vario:
        #     g2 = self.vario_func()(hh2.dot(mrot.T)) # hh2.dot(mrot.T): 3D-lags in system Oxyz (what is taken by the function)
        # else:
        #     g2 = self.func()(hh2.dot(mrot.T)) # hh2.dot(mrot.T): 3D-lags in system Oxz (what is taken by the function)

        # Evaluate function along axis z'''
        h3 = np.linspace(h3min, h3max, n3)
        hh3 = np.hstack((np.zeros((len(h3),1)), np.zeros((len(h3),1)), h3.reshape(-1,1))) # (n1,3) array) 3D-lags along z''' expressed in system Ox''y'''z''''
        g3 = self(hh3.dot(mrot.T), vario=vario) # hh1.dot(mrot.T): 3D-lags in system Oxyz (what is taken by the function)
        # if vario:
        #     g3 = self.vario_func()(hh3.dot(mrot.T)) # hh3.dot(mrot.T): 3D-lags in system Oxyz (what is taken by the function)
        # else:
        #     g3 = self.func()(hh3.dot(mrot.T)) # hh3.dot(mrot.T): 3D-lags in system Oxz (what is taken by the function)

        # Plot curve along x'''
        plt.plot(h1, g1, '-', c=color0, label="along x'''")
        # Plot curve along y'''
        plt.plot(h2, g2, '-', c=color1, label="along y'''")
        # Plot curve along z'''
        plt.plot(h3, g3, '-', c=color2, label="along z'''")

        if show_xlabel:
            plt.xlabel('h')
        if show_ylabel:
            if vario:
                plt.ylabel(r'$\gamma(h)$')
            else:
                plt.ylabel(r'$cov(h)$')

        plt.legend()
        plt.grid(grid)

    def plot_model_one_curve(
            self,
            main_axis=1,
            vario=False,
            hmin=0.0,
            hmax=None,
            npts=500,
            show_xlabel=True,
            show_ylabel=True,
            grid=True,
            logger=None,
            **kwargs):
        """
        Plots the covariance or variogram curve along one main axis (in the current figure axis).

        Parameters
        ----------
        main_axis : int (1 or 2 or 3), default: 1
            if `main_axis=1`, plots the curve along the 1st axis (x''')
            if `main_axis=2`, plots the curve along the 2nd axis (y''')
            if `main_axis=3`, plots the curve along the 3rd axis (z''')

        vario : bool, default: False
            - if False: the covariance model is displayed
            - if True: the variogram model is displayed

        hmin : float, default: 0.0
            see `hmax`
        hmax : float, optional
            function is plotted for h in interval [`hmin`,` hmax`] along the
            axis specified by `main_axis`; by default (`hmax=None`), `hmax` is
            set to 1.2 times the range of the model along the specified axis

        npts : int, default: 500
            number of points used in interval [`hmin`,` hmax`]

        show_xlabel : bool, default: True
            indicates if (default) label for abscissa is displayed

        show_ylabel : bool, default: True
            indicates if (default) label for ordinate is displayed

        grid : bool, default: True
            indicates if a grid is plotted
            
        logger : :class:`logging.Logger`, optional
            logger (see package `logging`)
            if specified, messages are written via `logger` (no print)

        kwargs : dict
            keyword arguments passed to the funtion `matplotlib.pyplot.plot`

        Notes
        -----
        No plot is displayed if the model is not stationary.
        """
        fname = 'plot_model_one_curve'

        # Prevent calculation if covariance model is not stationary
        if not self.is_stationary():
            return None
        if main_axis not in (1, 2, 3):
            err_msg = f'{fname}: `main_axis` invalid (should be 1, 2 or 3)'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

        # In kwargs:
        #   - add default 'label' if not given
        if 'label' not in kwargs.keys():
            if vario:
                kwargs['label'] = 'vario func'
            else:
                kwargs['label'] = 'cov func'

        # Set hmax if needed
        if hmax is None:
            hmax = 1.2*self.r123()[main_axis-1]

        # Rotation matrix
        mrot = self.mrot()

        # Evaluate function along selected axis
        h = np.linspace(hmin, hmax, npts)
        if main_axis == 1:
            hh = np.hstack((h.reshape(-1,1), np.zeros((len(h),1)), np.zeros((len(h),1)))) # (npts,3) array) 3D-lags along x''' expressed in system Ox''y'''z''''
        elif main_axis == 2:
            hh = np.hstack((np.zeros((len(h),1)), h.reshape(-1,1), np.zeros((len(h),1)))) # (npts,3) array) 3D-lags along y''' expressed in system Ox''y'''z''''
        else:
            hh = np.hstack((np.zeros((len(h),1)), np.zeros((len(h),1)), h.reshape(-1,1))) # (npts,3) array) 3D-lags along z''' expressed in system Ox''y'''z''''
        g = self(hh.dot(mrot.T), vario=vario) # hh.dot(mrot.T): 3D-lags in system Oxyz (what is taken by the function)
        # if vario:
        #     g = self.vario_func()(hh.dot(mrot.T)) # hh.dot(mrot.T): 3D-lags in system Oxyz (what is taken by the function)
        # else:
        #     g = self.func()(hh.dot(mrot.T)) # hh.dot(mrot.T): 3D-lags in system Oxyz (what is taken by the function)

        plt.plot(h, g, **kwargs)

        if show_xlabel:
            plt.xlabel('h')
        if show_ylabel:
            if vario:
                plt.ylabel(r'$\gamma(h)$')
            else:
                plt.ylabel(r'$cov(h)$')

        plt.grid(grid)
# ----------------------------------------------------------------------------

# ============================================================================
# Definition of functions to provide a copy of a covariance model
# ============================================================================
# ----------------------------------------------------------------------------
def copyCovModel(cov_model):
    """
    Returns a copy of a covariance model in 1D, 2D, or 3D.

    Parameters
    ----------
    cov_model : :class:`CovModel1D` or :class:`CovModel2D` or :class:`CovModel3D`
        covariance model in 1D, 2D, or 3D

    Returns
    -------
    cov_model_out : same type as cov_model
        copy of cov_model
    """
    # fname = 'copyCovModel'

    cov_model_out = copy.deepcopy(cov_model)
    return cov_model_out
# ----------------------------------------------------------------------------

# ============================================================================
# Definition of functions to convert covariance models
# ============================================================================
# ----------------------------------------------------------------------------
def covModel1D_to_covModel2D(cov_model_1d):
    """
    Converts a covariance model in 1D to an omni-directional covariance model
    in 2D.

    The elementary models of the 2D model are those of the 1D model
    (the attribute `alpha` of the 2D model is set to 0.0).

    Parameters
    ----------
    cov_model_1d : :class:`CovModel1D`
        covariance model in 1D

    Returns
    -------
    cov_model_2d : :class:`CovModel2D`
        covariance model in 2D, omni-directional (same range parameter `r` along
        each axis for every elementary models), defined from `cov_model_1d`
    """
    # fname = 'covModel1D_to_covModel2D'

    cov_model_2d = CovModel2D()
    cov_model_2d.elem = copy.deepcopy(cov_model_1d.elem)
    for el in cov_model_2d.elem:
        for k, val in el[1].items():
            if k == 'r':
                el[1]['r'] = [val, val]
    return cov_model_2d
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel1D_to_covModel3D(cov_model_1d):
    """
    Converts a covariance model in 1D to an omni-directional covariance model
    in 3D.

    The elementary models of the 3D model are those of the 1D model
    (the attributes `alpha`, `beta`, `gamma` of the 3D model are set to 0.0).

    Parameters
    ----------
    cov_model_1d : :class:`CovModel1D`
        covariance model in 1D

    Returns
    -------
    cov_model_3d : :class:`CovModel3D`
        covariance model in 3D, omni-directional (same range parameter `r` along
        each axis for every elementary models), defined from `cov_model_1d`
    """
    # fname = 'covModel1D_to_covModel3D'

    cov_model_3d = CovModel3D()
    cov_model_3d.elem = copy.deepcopy(cov_model_1d.elem)
    for el in cov_model_3d.elem:
        for k, val in el[1].items():
            if k == 'r':
                el[1]['r'] = [val, val, val]

    return cov_model_3d
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel2D_to_covModel3D(cov_model_2d, r_ind=(0, 0, 1), alpha=0.0, beta=0.0, gamma=0.0, logger=None):
    """
    Converts a covariance model in 2D to a covariance model in 3D.

    The elementary models of the 3D model are those of the 2D model.
    See parameters below for the ranges and angles `alpha`, `beta`, `gamma`.

    Parameters
    ----------
    cov_model_2d : :class:`CovModel2D`
        covariance model in 2D

    r_ind: 3-tuple of ints (with values 0 or 1), default: (0, 0, 1)
        indexes of range to be taken from the covariance model in 2D, for the 3
        axes of the covariance model in 3D, i.e. the parameter `r` of every
        elementary models is set to `(r[r_ind[0]], r[r_ind[1]], r[r_ind[2]])` for
        the 3D model, from the parameter `r`

    alpha : float, default: 0.0
        attribute `alpha` of the 3D model

    beta : float, default: 0.0
        attribute `beta` of the 3D model

    gamma : float, default: 0.0
        attribute `gamma` of the 3D model

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    cov_model_3d : :class:`CovModel3D`
        covariance model in 3D, defined from `cov_model_2d`
    """
    fname = 'covModel2D_to_covModel3D'

    r_ind = np.atleast_1d(r_ind).reshape(-1)
    if r_ind.size != 3:
        err_msg = f'{fname}: `r_ind` invalid (should be of length 3)'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    if np.any((r_ind > 1, r_ind < -2)):
        err_msg = f'{fname}: `r_ind` invalid  (index)'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    cov_model_3d = CovModel3D()
    cov_model_3d.elem = copy.deepcopy(cov_model_2d.elem)
    cov_model_3d.alpha = alpha
    cov_model_3d.beta = beta
    cov_model_3d.gamma = gamma
    for el in cov_model_3d.elem:
        for k, val in el[1].items():
            if k == 'r':
                el[1]['r'] = [val[r_ind[0]], val[r_ind[1]], val[r_ind[2]]]

    return cov_model_3d
# ----------------------------------------------------------------------------

# ============================================================================
# Definition of functions to get rotation matrix in 2D and 3D given angle(s)
# ============================================================================
# ----------------------------------------------------------------------------
def rotationMatrix2D(alpha=0.0):
    """
    Returns the 2x2 matrix of rotation defining axes of covariance model in 2D.

    The function returns the 2x2 matrix m for changing the coordinates system
    from Ox'y' to Oxy, where the system Ox'y' is obtained from the system Oxy by
    applying a rotation of angle -alpha,

    .. math::
        m = \\left(\\begin{array}{cc}
                \\cos\\alpha & \\sin\\alpha\\\\
               -\\sin\\alpha & \\cos\\alpha
            \\end{array}\\right)

    i.e. the columns of m are the new axes expressed in the system Oxy.

    Parameters
    ----------
    alpha : float, default: 0.0
        azimuth angle in degrees

    Returns
    -------
    mrot : 2D array of shape (2, 2)
        rotation matrix (see above)
    """
    # fname = 'rotationMatrix2D'
    #
    # The 2x2 matrix m for changing the coordinates system from Ox'y' to Oxy is:
    #         +                         +
    #         |  cos(alpha)   sin(alpha)|
    #     m = | -sin(alpha)   cos(alpha)|
    #         +                         +
    #
    a = alpha * np.pi/180.0
    ca, sa = np.cos(a), np.sin(a)
    m = np.array([[ca, sa], [-sa, ca]])
    return m
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def rotationMatrix3D(alpha=0.0, beta=0.0, gamma=0.0):
    """
    Returns the 3x3 matrix of rotation defining axes of covariance model in 3D.

    The function returns the 3x3 matrix m for changing the coordinates system
    from Ox'''y'''z''' to Oxyz, where the system Ox'''y''''z''' is obtained
    from the system Oxyz as follows::

        # Oxyz      -- rotation of angle -alpha around Oz  --> Ox'y'z'
        # Ox'y'z'   -- rotation of angle -beta  around Ox' --> Ox''y''z''
        # Ox''y''z''-- rotation of angle -gamma around Oy''--> Ox'''y'''z'''

    The matrix m is given by

    .. math::
        m = \\left(\\begin{array}{rrr}
                \\cos\\alpha \\cdot \\cos\\gamma + \\sin\\alpha \\cdot \\sin\\beta \\cdot \\sin\\gamma &  \\sin\\alpha \\cdot \\cos\\beta & - \\cos\\alpha \\cdot \\sin\\gamma + \\sin\\alpha \\cdot \\sin\\beta \\cdot \\cos\\gamma \\\\ \
              - \\sin\\alpha \\cdot \\cos\\gamma + \\cos\\alpha \\cdot \\sin\\beta \\cdot \\sin\\gamma &  \\cos\\alpha \\cdot \\cos\\beta &   \\sin\\alpha \\cdot \\sin\\gamma + \\cos\\alpha \\cdot \\sin\\beta \\cdot \\cos\\gamma \\\\ \
                                                                       \\cos\\beta \\cdot \\sin\\gamma &                    - \\sin\\beta &                                                          \\cos\\beta \\cdot \\cos\\gamma
            \\end{array}\\right)

    i.e. the columns of m are the new axes expressed in the system Oxyz.

    Parameters
    ----------
    alpha : float, default: 0.0
        azimuth angle in degrees

    beta : float, default: 0.0
        dip angle in degrees

    plunge : float, default: 0.0
        azimuth angle in degrees

    Returns
    -------
    mrot : 2D array of shape (3, 3)
        rotation matrix (see above)
    """
    # fname = 'rotationMatrix3D'
    #
    # The 3x3 matrix m for changing the coordinates system from Ox'''y'''z'''
    # to Oxyz is:
    #         +                                                            +
    #         |  ca * cc + sa * sb * sc,  sa * cb, - ca * sc + sa * sb * cc|
    #     m = |- sa * cc + ca * sb * sc,  ca * cb,   sa * sc + ca * sb * cc|
    #         |                 cb * sc,     - sb,                  cb * cc|
    #         +                                                            +
    # where
    #     ca = cos(alpha), cb = cos(beta), cc = cos(gamma),
    #     sa = sin(alpha), sb = sin(beta), sc = sin(gamma)
    #
    t = np.pi/180.0
    a = alpha * t
    b = beta * t
    c = gamma * t
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cc, sc = np.cos(c), np.sin(c)
    m = np.array(
            [[  ca * cc + sa * sb * sc,  sa * cb,  - ca * sc + sa * sb * cc],
             [- sa * cc + ca * sb * sc,  ca * cb,    sa * sc + ca * sb * cc],
             [                 cb * sc,     - sb,                   cb * cc]])
    return m
# ----------------------------------------------------------------------------

# ============================================================================
# Basic functions for plotting variogram cloud and experimental variogram (1D)
# ============================================================================
# ----------------------------------------------------------------------------
def plot_variogramCloud1D(
        h, g,
        decim=1.0,
        seed=None,
        show_xlabel=True,
        show_ylabel=True,
        grid=True,
        **kwargs):
    """
    Plots a variogram cloud (1D) (in the current figure axis).

    Parameters
    -------
    h : 1D array of floats
        see `g`

    g : 1D array of floats
        `h` and `g` (of same length) are the coordinates of the
        points (lag values and gamma values resp.) in the variogram cloud

    decim : float, default: 1.0
        the variogram cloud plotted after decimation by taking into account
        a proportion of `decim` points, randomly chosen; by default (`1`): all
        points are plotted

    seed : int, optional
        seed number used to initialize the random number generator (used if
        `decim<1`)

    show_xlabel : bool, default: True
        indicates if (default) label for abscissa is displayed

    show_ylabel : bool, default: True
        indicates if (default) label for ordinate is displayed

    grid : bool, default: True
        indicates if a grid is plotted

    kwargs : dict
        keyword arguments passed to the funtion `matplotlib.pyplot.plot`
    """
    # fname = 'plot_variogramCloud1D'

    # In kwargs:
    #   - add default 'label' if not given
    #   - set default 'marker' if not given
    #   - set default 'linestyle' (or 'ls') if not given
    if 'label' not in kwargs.keys():
        kwargs['label'] = 'vario cloud'
    if 'marker' not in kwargs.keys():
        kwargs['marker'] = '.'
    if 'linestyle' not in kwargs.keys() and 'ls' not in kwargs.keys():
        kwargs['linestyle'] = 'none'

    if decim < 1.0:
        n = np.round(decim*len(h))
        if n > 0:
            if seed is not None:
                np.random.seed(seed)
            # ind = np.sort(np.random.choice(len(h), size=n, replace=False))
            ind = np.random.choice(len(h), size=n, replace=False)
            hi = h[ind]
            gi = g[ind]
        else:
            hi = []
            gi = []
    else:
        hi = h
        gi = g

    plt.plot(hi, gi, **kwargs)
    if show_xlabel:
        plt.xlabel('h')
    if show_ylabel:
        plt.ylabel(r'$1/2(Z(x)-Z(x+h))^2$')
    plt.grid(grid)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def plot_variogramExp1D(
        hexp, gexp, cexp,
        show_count=True,
        show_ylabel=True,
        show_xlabel=True,
        grid=True,
        **kwargs):
    """
    Plots an experimental variogram (1D) (in the current figure axis).

    Parameters
    ----------
    hexp : 1D array of floats
        see `gexp`
    gexp : 1D array of floats
        `hexp` and `gexp` (of same length) are the coordinates of the
        points (lag values and gamma values resp.) in the experimental variogram

    cexp : 1D array of ints
        array of same length as `hexp`, `gexp`, counters, number of points
        (pairs of data points considered) comming from the variogram cloud that
        defines each point in the experimental variogram

    show_count : bool, default: True
        indicates if counters (`cexp`) are displayed on the plot

    show_xlabel : bool, default: True
        indicates if (default) label for abscissa is displayed

    show_ylabel : bool, default: True
        indicates if (default) label for ordinate is displayed

    grid : bool, default: True
        indicates if a grid is plotted

    kwargs : dict
        keyword arguments passed to the funtion `matplotlib.pyplot.plot`
    """
    # fname = 'plot_variogramExp1D'

    # In kwargs:
    #   - add default 'label' if not given
    #   - set default 'marker' if not given
    #   - set default 'linestyle' (or 'ls') if not given
    if 'label' not in kwargs.keys():
        kwargs['label'] = 'vario exp.'
    if 'marker' not in kwargs.keys():
        kwargs['marker'] = '.'
    if 'linestyle' not in kwargs.keys() and 'ls' not in kwargs.keys():
        kwargs['linestyle'] = 'dashed'

    plt.plot(hexp, gexp, **kwargs)
    if show_count:
        for i, c in enumerate(cexp):
            if c > 0:
                plt.text(hexp[i], gexp[i], str(c), ha='left', va='top')
    if show_xlabel:
        plt.xlabel('h')
    if show_ylabel:
        plt.ylabel(r'$1/2(Z(x)-Z(x+h))^2$')
    plt.grid(grid)
# ----------------------------------------------------------------------------

# ============================================================================
# Functions for variogram cloud, experimental variogram,
# and covariance model fitting (1D)
# ============================================================================
# ----------------------------------------------------------------------------
def variogramCloud1D(
        x, v,
        hmax=None,
        w_factor_loc_func=None,
        coord_factor_loc_func=None,
        loc_m=1,
        make_plot=True,
        logger=None,
        **kwargs):
    """
    Computes the omni-directional variogram cloud for a data set in 1D, 2D or 3D.

    From the pair of the i-th and j-th data points (i not equal to j), let

    .. math::
        \\begin{array}{rcl}
            h(i, j) &=& x_i-x_j\\\\[2mm]
            g(i, j) &=& \\frac{1}{2}(v_i - v_j)^2
        \\end{array}

    where :math:`x_i` and :math:`x_j` are the coordinates of the i-th and j-th
    data points and :math:`v_i` and :math:`v_j` the values at these points
    (:math:`v_i=Z(x_i)`, where :math:`Z` is the considered variable).
    The points  :math:`(\\Vert h(i, j)\\Vert, g(i, j))` such that
    :math:`\\Vert h(i, j)\\Vert` does not exceed `hmax` (see below) constitute
    the points of the variogram cloud.

    Moreover, the parameters `w_factor_loc_func` and `coord_factor_loc_func`
    allow to account for variogram locally varying in space with respect to
    weight and range resp., by multiplying "g" and "h" values resp.

    Parameters
    ----------
    x : 2D array of floats of shape (n, d)
        data points locations, with n the number of data points and d the space
        dimension (1, 2, or 3), each row of `x` is the coordinatates of one data
        point; note: for data in 1D (`d=1`), 1D array of shape `(n,)` is accepted
        for `n` data points

    v : 1D array of floats of shape (n,)
        data points values, with n the number of data points, `v[i]` is the data
        value at location `x[i]`

    hmax : float, optional
        maximal distance between a pair of data points to be integrated in the
        variogram cloud;
        note: `None` (default), `numpy.nan` are converted to `numpy.inf` (no
        restriction)

    w_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "weight" as function of a given
        location in 1D, 2D, or 3D (same dimension as the data set), i.e. "g"
        values (i.e. ordinate axis component in the variogram) are multiplied

    coord_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" as function of a given
        location in 1D, 2D, or 3D (same dimension as the data set), i.e. "h"
        values (i.e. abscissa axis component in the variogram) are multiplied
        (the condition wrt `hmax` is checked after)

    loc_m : int, default: 1
        integer (greater than or equal to 0) defining how the function(s)
        `*_loc_func` (above) are evaluated for a pair of two locations x1, x2
        (data point locations):

        - if `loc_m>0` the segment from x1 to x2 is divided in `loc_m` intervals \
        of same length and the mean of the evaluations of the function at the \
        (`loc_m` + 1) interval bounds is computed
        - if `loc_m=0`, the evaluation at x1 is considered

    make_plot : bool, default: True
        indicates if the variogram cloud is plotted (in the current figure axis,
        using the function `plot_variogramCloud1D`)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs : dict
        keyword arguments passed to the funtion `plot_variogramCloud1D`
        (if `make_plot=True`)

    Returns
    -------
    h : 1D array of floats
        see `g`

    g : 1D array of floats
        `h` and `g` (of same length) are the coordinates of the
        points (lag values and gamma values resp.) in the variogram cloud

    npair : int
        number of points (pairs of data points considered) in the variogram
        cloud (length of `h` or `g`)
    """
    fname = 'variogramCloud1D'

    x = np.asarray(x)
    # Get dimension (d) from x
    if x.ndim == 1:
        # x is a 1-dimensional array
        x = x.reshape(-1, 1)
        d = 1
    else:
        # x is a 2-dimensional array
        d = x.shape[1]

    # Number of data points
    n = x.shape[0]

    # Check length of v
    v = np.asarray(v).reshape(-1)
    if len(v) != n:
        err_msg = f'{fname}: length of `v` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # Set types of local transformations
    #    w_loc:     True / False: is local w (weight / sill) used ?
    #    coord_loc: True / False: is local coordinate (lag) used ?
    w_loc = False
    coord_loc = False
    if w_factor_loc_func is not None:
        w_loc = True
    if coord_factor_loc_func is not None:
        coord_loc = True

    if w_loc or coord_loc:
        transform_flag = True
    else:
        transform_flag = False

    # Compute variogram cloud
    if hmax is None or np.isnan(hmax):
        # Consider all pairs of points
        npair = int(0.5*(n-1)*n)
        h = np.zeros(npair)
        g = np.zeros(npair)
        j = 0
        if transform_flag:
            wf = 1.0 # default weight factor
            if loc_m > 0:
                for i in range(n-1):
                    jj = n-1-i
                    d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
                    dx = d/loc_m
                    ddx = np.asarray([x[i]+np.outer(np.arange(loc_m+1), dxk) for dxk in dx]) # 3-dimensional array (n-1-i) x (loc_m+1) x dim
                    htmp = np.sqrt(np.sum(d**2, axis=1))
                    if coord_loc:
                        htmp = np.mean(coord_factor_loc_func(ddx.reshape(-1, 1)).reshape(-1, loc_m+1), axis=1)*htmp
                        # htmp = np.asarray([np.mean(coord_factor_loc_func(ddxk)) for ddxk in ddx])*htmp
                    h[j:(j+jj)] = htmp
                    if w_loc:
                        wf = np.mean(w_factor_loc_func(ddx.reshape(-1, 1)).reshape(-1, loc_m+1), axis=1)
                        # wf = np.asarray([np.mean(w_factor_loc_func(ddxk)) for ddxk in ddx])
                    g[j:(j+jj)] = wf * 0.5*(v[i] - v[(i+1):])**2
                    j = j+jj
            else:
                for i in range(n-1):
                    jj = n-1-i
                    d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
                    htmp = np.sqrt(np.sum(d**2, axis=1))
                    if coord_loc:
                        htmp = coord_factor_loc_func(x[i])[0]*htmp
                    h[j:(j+jj)] = htmp
                    if w_loc:
                        wf = w_factor_loc_func(x[i])[0]
                    g[j:(j+jj)] = wf * 0.5*(v[i] - v[(i+1):])**2
                    j = j+jj
        else:
            for i in range(n-1):
                jj = n-1-i
                h[j:(j+jj)] = np.sqrt(np.sum((x[i] - x[(i+1):])**2, axis=1))
                g[j:(j+jj)] = 0.5*(v[i] - v[(i+1):])**2
                j = j+jj
    else:
        # Consider only pairs of points with a distance less than or equal to hmax
        h, g = [], []
        if transform_flag:
            wf = 1.0 # default weight factor
            if loc_m > 0:
                for i in range(n-1):
                    jj = n-1-i
                    d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
                    dx = d/loc_m
                    ddx = np.asarray([x[i]+np.outer(np.arange(loc_m+1), dxk) for dxk in dx]) # 3-dimensional array (n-1-i) x (loc_m+1) x dim
                    htmp = np.sqrt(np.sum(d**2, axis=1))
                    if coord_loc:
                        htmp = np.mean(coord_factor_loc_func(ddx.reshape(-1, 1)).reshape(-1, loc_m+1), axis=1)*htmp
                        # htmp = np.asarray([np.mean(coord_factor_loc_func(ddxk)) for ddxk in ddx])*htmp
                    ind = np.where(htmp <= hmax)[0]
                    if len(ind) > 0:
                        h.append(htmp[ind])
                        if w_loc:
                            wf = np.mean(w_factor_loc_func(ddx[ind].reshape(-1, 1)).reshape(-1, loc_m+1), axis=1)
                            # wf = np.asarray([np.mean(w_factor_loc_func(ddxk)) for ddxk in ddx[ind]])
                        g.append(wf * 0.5*(v[i] - v[i+1+ind])**2)
            else:
                for i in range(n-1):
                    jj = n-1-i
                    d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
                    htmp = np.sqrt(np.sum(d**2, axis=1))
                    if coord_loc:
                        htmp = coord_factor_loc_func(x[i])[0]*htmp
                    ind = np.where(htmp <= hmax)[0]
                    if len(ind) > 0:
                        h.append(htmp[ind])
                        if w_loc:
                            wf = w_factor_loc_func(x[i])[0]
                        g.append(wf * 0.5*(v[i] - v[i+1+ind])**2)
        else:
            hmax2 = hmax**2
            for i in range(n-1):
                htmp = np.sum((x[i] - x[(i+1):])**2, axis=1)
                ind = np.where(htmp <= hmax2)[0]
                h.append(np.sqrt(htmp[ind]))
                g.append(0.5*(v[i] - v[i+1+ind])**2)
        npair = len(h)
        if npair:
            h = np.hstack(h)
            g = np.hstack(g)

    if make_plot:
        plot_variogramCloud1D(h, g, **kwargs)
        plt.title(f'Variogram cloud ({npair} pts)')

    return h, g, npair
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def variogramExp1D(
        x, v,
        hmax=None,
        w_factor_loc_func=None,
        coord_factor_loc_func=None,
        loc_m=1,
        ncla=10,
        cla_center=None,
        cla_length=None,
        variogramCloud=None,
        make_plot=True,
        logger=None,
        **kwargs):
    """
    Computes the exprimental omni-directional variogram for a data set in 1D, 2D or 3D.

    The mean point in each class is retrieved from the variogram cloud (returned
    by the function `variogramCloud1D`); the i-th class is determined by its
    center `cla_center[i]` and its length `cla_length[i]`, and corresponds to the
    interval

        `]cla_center[i]-cla_length[i]/2, cla_center[i]+cla_length[i]/2]`

    along h (lag) axis (abscissa).

    Parameters
    ----------
    x : 2D array of floats of shape (n, d)
        data points locations, with n the number of data points and d the space
        dimension (1, 2, or 3), each row of x is the coordinatates of one data
        point; note: for data in 1D (`d=1`), 1D array of shape `(n,)` is accepted
        for `n` data points

    v : 1D array of floats of shape (n,)
        data points values, with n the number of data points, `v[i]` is the data
        value at location `x[i]`

    hmax : float, optional
        maximal distance between a pair of data points to be integrated in the
        variogram cloud;
        note: `None` (default), `numpy.nan` are converted to `numpy.inf` (no
        restriction)

    w_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "weight" as function of a given
        location in 1D, 2D, or 3D (same dimension as the data set), i.e. "g"
        values (i.e. ordinate axis component in the variogram) are multiplied

    coord_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" as function of a given
        location in 1D, 2D, or 3D (same dimension as the data set), i.e. "h"
        values (i.e. abscissa axis component in the variogram) are multiplied

    loc_m : int, default: 1
        integer (greater than or equal to 0) defining how the function(s)
        `*_loc_func` (above) are evaluated for a pair of two locations x1, x2
        (data point locations):

        - if `loc_m>0` the segment from x1 to x2 is divided in `loc_m` intervals \
        of same length and the mean of the evaluations of the function at the \
        (`loc_m` + 1) interval bounds is computed
        - if `loc_m=0`, the evaluation at x1 is considered

    ncla : int, default: 10
        number of classes, the parameter is used if `cla_center=None`, in that
        situation `ncla` classes are considered and the class centers are set to

        - `cla_center[i] = (i+0.5)*l, i=0,...,ncla-1`

        with l = H / ncla, H being the max of the distance between two points of
        the considered pairs (in the variogram cloud);
        if `cla_center` is specified (not `None`), the number of classes (`ncla`)
        is set to the length of the sequence `cla_center` (ignoring the value
        passed as argument)

    cla_center : 1D array-like of floats, optional
        sequence of floats, center of each class (in abscissa) in the experimental
        variogram; by default (`None`): `cla_center` is defined from `ncla` (see
        above)

    cla_length : 1D array-like of floats, or float, optional
        length of each class centered at `cla_center` (in abscissa) in the
        experimental variogram:

        - if `cla_length` is a sequence, it should be of length `ncla`
        - if `cla_length` is a float, the value is repeated `ncla` times
        - if `cla_length=None` (default), the minimum of difference between two \
        sucessive class centers (`np.inf` if one class) is used and repeated `ncla` \
        times

    variogramCloud : 3-tuple, optional
        `variogramCloud` = (h, g, npair) is a variogram cloud (already computed and
        returned by the function `variogramCloud1D` (npair not used)); in this case,
        `x`, `v`, `hmax`, `w_factor_loc_func`, `coord_factor_loc_func`, `loc_m`
        are not used

        By default (`None`): the variogram cloud is computed by using the
        function `variogramCloud1D`

    make_plot : bool, default: True
        indicates if the experimental variogram is plotted (in the current figure
        axis, using the function `plot_variogramExp1D`)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs : dict
        keyword arguments passed to the funtion `plot_variogramExp1D`
        (if `make_plot=True`)

    Returns
    -------
    hexp : 1D array of floats
        see `gexp`

    gexp : 1D array of floats
        `hexp` and `gexp` (of same length) are the coordinates of the
        points (lag values and gamma values resp.) in the experimental variogram

    cexp : 1D array of ints
        array of same length as `hexp`, `gexp`, counters, number of points
        (pairs of data points considered) comming from the variogram cloud that
        defines each point in the experimental variogram
    """
    fname = 'variogramExp1D'

    # Compute variogram cloud if needed (npair won't be used)
    if variogramCloud is None:
        try:
            h, g, npair = variogramCloud1D(
                    x, v, hmax=hmax,
                    w_factor_loc_func=w_factor_loc_func, 
                    coord_factor_loc_func=coord_factor_loc_func, 
                    loc_m=loc_m,
                    make_plot=False,
                    logger=logger)
        except Exception as exc:
            err_msg = f'{fname}: cannot compute variogram cloud (1D)'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg) from exc

    else:
        h, g, npair = variogramCloud

    if npair == 0:
        # print('No point in the variogram cloud (nothing is done).')
        # return None, None, None
        return np.empty(0), np.empty(0), np.empty(0, dtype=int)

    # Set classes
    if cla_center is not None:
        cla_center = np.asarray(cla_center, dtype='float').reshape(-1)
        ncla = len(cla_center)
    else:
        if ncla == 0:
            err_msg = f'{fname}: `ncla` invalid (must be greater than 0)'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

        length = np.max(h) / ncla
        cla_center = (np.arange(ncla, dtype='float') + 0.5) * length

    if cla_length is not None:
        cla_length = np.asarray(cla_length, dtype='float').reshape(-1)
        if len(cla_length) == 1:
            cla_length = np.repeat(cla_length, ncla)
        elif len(cla_length) != ncla:
            err_msg = f'{fname}: `cla_length` invalid'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    else:
        if ncla == 1:
            cla_length = np.array([np.inf], dtype='float')
        else:
            cla_length = np.repeat(np.min(np.diff(cla_center)), ncla)

    # Compute experimental variogram
    hexp = np.nan * np.ones(ncla)
    gexp = np.nan * np.ones(ncla)
    cexp = np.zeros(ncla, dtype='int')

    for i, (c, l) in enumerate(zip(cla_center, cla_length)):
        d = 0.5*l
        ind = np.all((h > c-d , h <= c+d), axis=0)
        hexp[i] = np.mean(h[ind])
        gexp[i] = np.mean(g[ind])
        cexp[i] = np.sum(ind)

    if make_plot:
        plot_variogramExp1D(hexp, gexp, cexp, **kwargs)
        plt.title('Experimental variogram')

    return hexp, gexp, cexp
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel1D_fit(
        x, v, cov_model,
        hmax=None,
        w_factor_loc_func=None,
        coord_factor_loc_func=None,
        loc_m=1,
        variogramCloud=None,
        make_plot=True,
        logger=None,
        **kwargs):
    """
    Fits a covariance model in 1D, from data in 1D, 2D, or 3D.

    If the input data is in 2D or 3D, an omni-directional model is fitted.

    The parameter `cov_model` is a covariance model in 1D where all the
    parameters to be fitted are set to `numpy.nan`. The fit is done according to
    the variogram cloud, by using the function `scipy.optimize.curve_fit`.

    Parameters
    ----------
    x : 2D array of floats of shape (n, d)
        data points locations, with n the number of data points and d the space
        dimension (1, 2, or 3), each row of x is the coordinatates of one data
        point; note: for data in 1D (`d=1`), 1D array of shape `(n,)` is accepted
        for `n` data points

    v : 1D array of floats of shape (n,)
        data points values, with n the number of data points, `v[i]` is the data
        value at location `x[i]`

    cov_model : :class:`CovModel1D`
        covariance model to otpimize (parameters set to `numpy.nan` are optimized)

    hmax : float, optional
        maximal distance between a pair of data points to be integrated in the
        variogram cloud;
        note: `None` (default), `numpy.nan` are converted to `numpy.inf` (no
        restriction)

    w_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "weight" as function of a given
        location in 1D, 2D, or 3D (same dimension as the data set), i.e. "g"
        values (i.e. ordinate axis component in the variogram) are multiplied

    coord_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" as function of a given
        location in 1D, 2D, or 3D (same dimension as the data set), i.e. "h"
        values (i.e. abscissa axis component in the variogram) are multiplied

    loc_m : int, default: 1
        integer (greater than or equal to 0) defining how the function(s)
        `*_loc_func` (above) are evaluated for a pair of two locations x1, x2
        (data point locations):

        - if `loc_m>0` the segment from x1 to x2 is divided in `loc_m` intervals \
        of same length and the mean of the evaluations of the function at the \
        (`loc_m` + 1) interval bounds is computed
        - if `loc_m=0`, the evaluation at x1 is considered

    variogramCloud : 3-tuple, optional
        `variogramCloud` =(h, g, npair) is a variogram cloud (already computed and
        returned by the function `variogramCloud1D` (npair not used)); in this case,
        `x`, `v`, `hmax`, `w_factor_loc_func`, `coord_factor_loc_func`, `loc_m`
        are not used

        By default (`None`): the variogram cloud is computed by using the
        function `variogramCloud1D`

    make_plot : bool, default: True
        indicates if the fitted covariance model is plotted (using the method
        `plot_model` with default parameters)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs : dict
        keyword arguments passed to the funtion `scipy.optimize.curve_fit`

    Returns
    -------
    cov_model_opt: :class:`CovModel1D`
        optimized covariance model

    popt: 1D array
        values of the optimal parameters, corresponding to the parameters of the
        input covariance model (`cov_model`) set to `numpy.nan`, in the order of
        appearance (vector of optimized parameters returned by
        `scipy.optimize.curve_fit`)

    Examples
    --------
    The following allows to fit a covariance model made up of a gaussian
    elementary model and a nugget effect (nugget elementary model), where the
    weight and range of the gaussian elementary model and the weight of the
    nugget effect are fitted (optimized) in intervals given by the keyword
    argument `bounds`. The arguments `x`, `v` are the data points and values,
    and the fitted covariance model is not plotted (`make_plot=False`)

        >>> # covariance model to optimize
        >>> cov_model_to_optimize = CovModel1D(elem=[
        >>>     ('gaussian', {'w':np.nan, 'r':np.nan}), # elementary contribution
        >>>     ('nugget', {'w':np.nan})                # elementary contribution
        >>>     ])
        >>> covModel1D_fit(x, v, cov_model_to_optimize,
        >>>                bounds=([ 0.0,   0.0,  0.0],  # lower bounds for parameters to fit
        >>>                        [10.0, 100.0, 10.0]), # upper bounds for parameters to fit
        >>>                make_plot=False)
     """
    fname = 'covModel1D_fit'

    # Check cov_model
    if not isinstance(cov_model, CovModel1D):
        err_msg = f'{fname}: `cov_model` is not a covariance model in 1D'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # if cov_model.__class__.__name__ != 'CovModel1D':
    #     err_msg = f'{fname}: `cov_model` is not a covariance model in 1D'
    #     if logger: logger.error(err_msg)
    #     raise CovModelError(err_msg)

    # Prevent calculation if covariance model is not stationary
    if not cov_model.is_stationary():
        err_msg = f'{fname}: `cov_model` is not stationary: fit cannot be applied'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # Work on a (deep) copy of cov_model
    cov_model_opt = copy.deepcopy(cov_model)

    # Get index of element and key of parameters to fit
    ielem_to_fit=[]
    key_to_fit=[]
    for i, el in enumerate(cov_model_opt.elem):
        for k, val in el[1].items():
            if np.isnan(val):
                ielem_to_fit.append(i)
                key_to_fit.append(k)

    nparam = len(ielem_to_fit)
    if nparam == 0:
        # print('No parameter to fit!')
        return cov_model_opt, np.array([])

    # Compute variogram cloud if needed (npair won't be used)
    if variogramCloud is None:
        try:
            h, g, npair = variogramCloud1D(
                    x, v, hmax=hmax,
                    w_factor_loc_func=w_factor_loc_func, 
                    coord_factor_loc_func=coord_factor_loc_func, 
                    loc_m=loc_m,
                    make_plot=False,
                    logger=logger) # npair won't be used
        except Exception as exc:
            err_msg = f'{fname}: cannot compute variogram cloud (1D)'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg) from exc

    else:
        h, g, npair = variogramCloud

    if npair == 0:
        err_msg = f'{fname}: no pair of points (in variogram cloud) for fitting'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)
        # print('No point to fit!')
        # return cov_model_opt, np.nan * np.ones(nparam)

    def func(d, *p):
        """
        Function whose p is the vector of parameters to optimize.

        Parameters
        ----------
        d : 1D array
            xdata, i.e. lags (h) from the variogram cloud where the current
            covariance model is evaluated

        p : 1D array
            current values of the parameters (floats) to optimize in the
            covariance model (parameters to optimized are identified with
            ielem_to_fit, key_to_fit, computed above)

        Returns
        -------
        v: 1D array
            evaluations of the current variogram model at `d`
        """
        for i, (iel, k) in enumerate(zip(ielem_to_fit, key_to_fit)):
            cov_model_opt.elem[iel][1][k] = p[i]
        return cov_model_opt(d, vario=True)

    # Optimize parameters with curve_fit: initial vector of parameters (p0) must be given
    #   because number of parameter to fit in function func is not known in its expression
    bounds = None
    if 'bounds' in kwargs.keys():
        bounds = kwargs['bounds']

    if 'p0' not in kwargs.keys():
        # add default p0 in kwargs
        p0 = np.ones(nparam)
        if bounds is not None:
            # adjust p0 to given bounds
            for i in range(nparam):
                if np.isinf(bounds[0][i]):
                    if np.isinf(bounds[1][i]):
                        p0[i] = 1.
                    else:
                        p0[i] = bounds[1][i]
                elif np.isinf(bounds[1][i]):
                    p0[i] = bounds[0][i]
                else:
                    p0[i] = 0.5*(bounds[0][i]+bounds[1][i])
        kwargs['p0'] = p0
    else:
        if len(kwargs['p0']) != nparam:
            err_msg = f'{fname}: length of `p0` and number of parameters to fit differ'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    # Fit with curve_fit
    try:
        popt, pcov = scipy.optimize.curve_fit(func, h, g, **kwargs)
    except:
        err_msg = f'{fname}: fitting covariance model failed'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    if make_plot:
        cov_model_opt.plot_model(vario=True, hmax=np.max(h), label='vario opt.')
        s = ['Vario opt.:'] + [f'{el}' for el in cov_model_opt.elem]
        # plt.title(textwrap.TextWrapper(width=50).fill(s))
        plt.title('\n'.join(s))

    return cov_model_opt, popt
# ----------------------------------------------------------------------------

# ============================================================================
# Functions for variogram cloud, experimental variogram,
# and covariance model fitting (2D)
# ============================================================================
# ----------------------------------------------------------------------------
def variogramCloud2D(
        x, v,
        alpha=0.0,
        tol_dist=None,
        tol_angle=None,
        hmax=None,
        alpha_loc_func=None,
        w_factor_loc_func=None,
        coord1_factor_loc_func=None,
        coord2_factor_loc_func=None,
        loc_m=1,
        make_plot=True,
        color0='red',
        color1='green',
        figsize=None,
        logger=None,
        **kwargs):
    """
    Computes the two directional variogram clouds (wrt. main axes) for a data set in 2D.

    From the pair of the i-th and j-th data points (i not equal to j), let

    .. math::
        \\begin{array}{rcl}
            h(i, j) &=& x_i-x_j \\\\[2mm]
            g(i, j) &=& \\frac{1}{2}(v_i - v_j)^2
        \\end{array}

    where :math:`x_i` and :math:`x_j` are the coordinates of the i-th and j-th
    data points and :math:`v_i` and :math:`v_j` the values at these points
    (:math:`v_i=Z(x_i)`, where :math:`Z` is the considered variable).
    The lag vector :math:`h(i, j)` is expressed along the two orthogonal main axes:
    h(i, j) = (h1(i, j), h2(i, j)). Let `tol_dist` = (tol_dist1, tol_dist2),
    `tol_angle` = (tol_angle1, tol_angle2), and `hmax` = (h1max, h2max)
    (see parameters below); if distance from h(i, j) to
    the 1st (resp. 2nd) main axis, i.e. \\|h2(i, j)\\| (resp. \\|h1(i, j)\\|) does not
    exceed tol_dist1 (resp. toldist2), and if the angle between the lag h(i, j)
    and the 1st (resp. 2nd) main axis does not exceed tol_angle1 (resp.
    tol_angle2), and if the distance along the 1st (resp. 2nd) axis, i.e.
    \\|h1(i, j)\\| (resp. \\|h2(i, j)\\|) does not exceed h1max (resp. h2max), then, the
    point (\\|h1(i, j)\\|, g(i, j)) (resp. (\\|h2(i, j)\\|, g(i, j))) is integrated in
    the directional variogram cloud along the 1st (resp. 2nd) main axis.

    Moreover, the parameter `alpha_loc_func` allows to account for main axes
    locally varying in space, and the parameters `w_factor_loc_func` and
    `coord1_factor_loc_func`, `coord2_factor_loc_func` allow to account for
    variogram locally varying in space with respect to weight and ranges along
    each main axis resp., by multiplying "g", "h1", "h2" values resp.

    Parameters
    ----------
    x : 2D array of floats of shape (n, 2)
        data points locations, with n the number of data points, each row of `x`
        is the coordinatates of one data point

    v : 1D array of floats of shape (n,)
        data points values, with n the number of data points, `v[i]` is the data
        value at location `x[i]`

    alpha : float, default: 0.0
        azimuth angle in degrees, defining the main axes
        (see :class:`CovModel2D`)

    tol_dist : sequence of 2 floats, or float, optional
        let `tol_dist` = (tol_dist1, tol_dist2); tol_dist1 (resp. tol_dist2) is the
        maximal distance to the 1st (resp. 2nd) main axis for the lag (vector
        between two data points), such that the pair is integrated in the
        variogram cloud along the 1st (resp. 2nd) main axis;
        note: if `tol_dist` is specified as a float or `None` (default), the
        entry is duplicated; if tol_dist1 (resp. tol_dist2) is `None`, then
        tol_dist1 (resp. tol_dist2) is set to 10% of h1max (resp. h2max) if
        h1max (resp. h2max) is finite, and set to 10.0 otherwise: see parameter
        `hmax` for the definition of h1max and h2max

    tol_angle : sequence of 2 floats, or float, optional
        let `tol_angle` = (tol_angle1, tol_angle2); tol_angle1 (resp. tol_angle2)
        is the maximal angle in degrees between the lag (vector between two data
        points) and the 1st (resp. 2nd) main axis, such that the pair is
        integrated in the variogram cloud along the 1st (resp. 2nd) main axis;
        note: if `tol_angle` is specified as a float, it is duplicated;
        by default (`None`): `tol_angle` is set to 45.0

    hmax : sequence of 2 floats, or float, optional
        let `hmax` = (h1max, h2max); h1max (resp. h2max) is the maximal distance
        between a pair of data points along the 1st (resp. 2nd) main axis, such
        that the pair is integrated in the variogram cloud along the 1st
        (resp. 2nd) main axis;
        note: if `hmax` is specified as a float or `None` (default), the entry is
        duplicated, and `None`, `numpy.nan` are converted to `numpy.inf` (no
        restriction)

    alpha_loc_func : function (`callable`), optional
        function returning azimuth angle, defining the main axes, as function of
        a given location in 2D, i.e. the main axes are defined locally

    w_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "weight" as function of a given
        location in 2D, i.e. "g" values (i.e. ordinate axis component in the two
        variograms) are multiplied

    coord1_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" along the 1st (local) main
        axis as function of a given location in 2D, i.e. "h1" values (i.e.
        abscissa axis component in the 1st variogram) are multiplied
        (the condition wrt h1max, see `hmax`, is checked after)

    coord2_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" along the 2nd (local) main
        axis as function of a given location in 2D, i.e. "h2" values (i.e.
        abscissa axis component in the 2nd variogram) are multiplied
        (the condition wrt h2max, see `hmax`, is checked after)

    loc_m : int, default: 1
        integer (greater than or equal to 0) defining how the function(s)
        `*_loc_func` (above) are evaluated for a pair of two locations x1, x2
        (data point locations):

        - if `loc_m>0` the segment from x1 to x2 is divided in `loc_m` intervals \
        of same length and the mean of the evaluations of the function at the \
        (`loc_m` + 1) interval bounds is computed
        - if `loc_m=0`, the evaluation at x1 is considered

    make_plot : bool, default: True
        indicates if the variogram clouds are plotted (in a new "2x2" figure)

    color0 : color, default: 'red'
        color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
        the plot of the variogram cloud along the 1st main axis
        (if `make_plot=True`)

    color1 : color, default: 'green'
        color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
        the plot of the variogram cloud along the 2nd main axis
        (if `make_plot=True`)

    figsize : 2-tuple, optional
        size of the new "2x2" figure (if `make_plot=True`)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs : dict
        keyword arguments passed to the funtion `plot_variogramCloud1D`
        (if `make_plot=True`)

    Returns
    -------
    (h0, g0, npair0) : 3-tuple
        h0, g0 : 1D arrays of floats of same length
            coordinates of the points in the variogram cloud along 1st main axis
            (see above)

        npair0 : int
            number of points (pairs of data points considered) in the variogram
            cloud along the 1st main axis

    (h1, g1, npair1) : 3-tuple
        h1, g1 : 1D arrays of floats of same length
            coordinates of the points in the variogram cloud along 2nd main axis
            (see above)

        npair1 : int
            number of points (pairs of data points considered) in the variogram
            cloud along the 2nd main axis
    """
    fname = 'variogramCloud2D'

    # Number of data points
    n = x.shape[0]

    # Check length of v
    if len(v) != n:
        err_msg = f'{fname}: length of `v` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # Set hmax as an array of shape (2,)
    hmax = np.atleast_1d(hmax).astype('float').reshape(-1) # None is converted to nan
    hmax[np.isnan(hmax)] = np.inf # convert nan to inf
    if hmax.size == 1:
        hmax = np.array([hmax[0], hmax[0]])
    elif hmax.size != 2:
        err_msg = f'{fname}: size of `hmax` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # Set tol_dist as an array of shape (2,)
    tol_dist = np.atleast_1d(tol_dist).astype('float').reshape(-1) # None is converted to nan
    if tol_dist.size == 1:
        tol_dist = np.array([tol_dist[0], tol_dist[0]])
    elif tol_dist.size != 2:
        err_msg = f'{fname}: size of `tol_dist` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    for i in range(2):
        if np.isnan(tol_dist[i]):
            if np.isinf(hmax[i]):
                tol_dist[i] = 10.0
            else:
                tol_dist[i] = 0.1 * hmax[i]

    # Set tol_angle as an array of shape (2,)
    tol_angle = np.atleast_1d(tol_angle).astype('float').reshape(-1) # None is converted to nan
    if tol_angle.size == 1:
        tol_angle = np.array([tol_angle[0], tol_angle[0]])
    elif tol_angle.size != 2:
        err_msg = f'{fname}: size of `tol_angle` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    tol_angle[np.isnan(tol_angle)] = 45.0

    if alpha != 0.0:
        rotate_coord_sys = True
        # Rotation matrix
        a = alpha * np.pi/180.
        ca, sa = np.cos(a), np.sin(a)
        mrot = np.array([[ca, sa], [-sa, ca]])
    else:
        rotate_coord_sys = False
        mrot = np.eye(2)

    # Set types of local transformations
    #    alpha_loc: True / False: is local angle alpha used ?
    #    w_loc:     True / False: is local w (weight / sill) used ?
    #    coord_loc: integer
    #               0: no transformation
    #               1: transformation for 1st coordinate only
    #               2: transformation for 2nd coordinate only
    #               3: distinct transformations for 1st and 2nd coordinates
    #               8: same transformation for 1st and 2nd coordinates
    alpha_loc = False
    w_loc = False
    coord_loc = 0
    if alpha_loc_func is not None:
        # factor to transform angle in degree into radian
        t_angle = np.pi/180.0
        alpha_loc = True

    if w_factor_loc_func is not None:
        w_loc = True

    if coord1_factor_loc_func is not None:
        coord_loc = coord_loc + 1
    if coord2_factor_loc_func is not None:
        coord_loc = coord_loc + 2
        if coord1_factor_loc_func == coord2_factor_loc_func:
            coord_loc = 8

    if alpha_loc or w_loc or coord_loc > 0:
        transform_flag = True
    else:
        transform_flag = False

    # Tolerance for slope compute from tol_angle
    tol_s = np.tan(tol_angle*np.pi/180)

    # Compute variogram clouds
    h0, g0, h1, g1 = [], [], [], []
    if transform_flag:
        wf = 1.0 # default weight factor
        if loc_m > 0:
            for i in range(n-1):
                d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
                dx = d/loc_m
                ddx = np.asarray([x[i]+np.outer(np.arange(loc_m+1), dxk) for dxk in dx]) # 3-dimensional array (n-1-i) x (loc_m+1) x dim
                if rotate_coord_sys:
                    # Rotate according to new system
                    d = d.dot(mrot)
                if alpha_loc:
                    a = t_angle * alpha_loc_func(ddx.reshape(-1, 2)).reshape(-1, loc_m+1) # 2-dimensional array (n-1-i) x (loc_m+1)
                    # a = t_angle * np.asarray([alpha_loc_func(ddxk) for ddxk in ddx])
                    cos_a, sin_a = np.mean(np.cos(a), axis=1), np.mean(np.sin(a), axis=1)
                    d = np.asarray([dk.dot(np.array([[cos_ak, sin_ak], [-sin_ak, cos_ak]])) for (cos_ak, sin_ak, dk) in zip (cos_a, sin_a, d)])
                if coord_loc == 1:
                    d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 2)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                    # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                elif coord_loc == 2:
                    d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 2)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                    # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                elif coord_loc == 3:
                    d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 2)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                    d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 2)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                    # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                    # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                elif coord_loc == 8:
                    d = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 2)).reshape(-1, loc_m+1), axis=1)*d.T).T
                    # d = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d.T).T
                d_abs = np.fabs(d)
                ind = np.where(np.all((d_abs[:, 0] <= hmax[0], d_abs[:, 1] <= tol_dist[0], d_abs[:, 1] <= tol_s[0]*d_abs[:, 0]), axis=0))[0]
                if len(ind) > 0:
                    h0.append(d_abs[ind, 0])
                    if w_loc:
                        wf = np.mean(w_factor_loc_func(ddx[ind].reshape(-1, 2)).reshape(-1, loc_m+1), axis=1)
                        # wf = np.asarray([np.mean(w_factor_loc_func(ddxk)) for ddxk in ddx[ind]])
                    g0.append(wf * 0.5*(v[i] - v[i+1+ind])**2)
                ind = np.where(np.all((d_abs[:, 1] <= hmax[1], d_abs[:, 0] <= tol_dist[1], d_abs[:, 0] <= tol_s[1]*d_abs[:, 1]), axis=0))[0]
                if len(ind) > 0:
                    h1.append(d_abs[ind, 1])
                    if w_loc:
                        wf = np.mean(w_factor_loc_func(ddx[ind].reshape(-1, 2)).reshape(-1, loc_m+1), axis=1)
                        # wf = np.asarray([np.mean(w_factor_loc_func(ddxk)) for ddxk in ddx[ind]])
                    g1.append(wf * 0.5*(v[i] - v[i+1+ind])**2)
        else:
            for i in range(n-1):
                d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
                if rotate_coord_sys:
                    # Rotate according to new system
                    d = d.dot(mrot)
                if alpha_loc:
                    a = t_angle * alpha_loc_func(x[i])[0]
                    cos_a, sin_a = np.cos(a), np.sin(a)
                    d = d.dot(np.array([[cos_a, sin_a], [-sin_a, cos_a]]))
                    # d = np.array([[cos_a, -sin_a], [sin_a, cos_a]]).dot(d.T).T
                if coord_loc == 1:
                    d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                elif coord_loc == 2:
                    d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                elif coord_loc == 3:
                    d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                    d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                elif coord_loc == 8:
                    d = coord1_factor_loc_func(x[i])[0]*d
                d_abs = np.fabs(d)
                ind = np.where(np.all((d_abs[:, 0] <= hmax[0], d_abs[:, 1] <= tol_dist[0], d_abs[:, 1] <= tol_s[0]*d_abs[:, 0]), axis=0))[0]
                if len(ind) > 0:
                    h0.append(d_abs[ind, 0])
                    if w_loc:
                        wf = w_factor_loc_func(x[i])[0]
                    g0.append(wf * 0.5*(v[i] - v[i+1+ind])**2)
                ind = np.where(np.all((d_abs[:, 1] <= hmax[1], d_abs[:, 0] <= tol_dist[1], d_abs[:, 0] <= tol_s[1]*d_abs[:, 1]), axis=0))[0]
                if len(ind) > 0:
                    h1.append(d_abs[ind, 1])
                    if w_loc:
                        wf = w_factor_loc_func(x[i])[0]
                    g1.append(wf * 0.5*(v[i] - v[i+1+ind])**2)
    else:
        if rotate_coord_sys:
            # Rotate according to new system
            x = x.dot(mrot)
        for i in range(n-1):
            d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
            d_abs = np.fabs(d)
            ind = np.where(np.all((d_abs[:, 0] <= hmax[0], d_abs[:, 1] <= tol_dist[0], d_abs[:, 1] <= tol_s[0]*d_abs[:, 0]), axis=0))[0]
            if len(ind) > 0:
                h0.append(d_abs[ind, 0])
                g0.append(0.5*(v[i] - v[i+1+ind])**2)
            ind = np.where(np.all((d_abs[:, 1] <= hmax[1], d_abs[:, 0] <= tol_dist[1], d_abs[:, 0] <= tol_s[1]*d_abs[:, 1]), axis=0))[0]
            if len(ind) > 0:
                h1.append(d_abs[ind, 1])
                g1.append(0.5*(v[i] - v[i+1+ind])**2)

    npair0 = len(h0)
    if npair0:
        h0 = np.hstack(h0)
        g0 = np.hstack(g0)
    npair1 = len(h1)
    if npair1:
        h1 = np.hstack(h1)
        g1 = np.hstack(g1)

    if make_plot:
        _, ax = plt.subplots(2,2, figsize=figsize)

        plt.sca(ax[0,0])
        # Plot system Oxy and Ox'y'
        # This:
        plt.arrow(*[0,0], *[0.9,0], color='k', head_width=0.05, head_length=0.1)
        plt.arrow(*[0,0], *[0,0.9], color='k', head_width=0.05, head_length=0.1)
        plt.text(*[1,0], "x", c='k', ha='left', va='top')
        plt.text(*[0,1], "y", c='k', ha='left', va='top')
        plt.arrow(*[0,0], *(0.9*mrot[:,0]), color=color0, head_width=0.05, head_length=0.1)
        plt.arrow(*[0,0], *(0.9*mrot[:,1]), color=color1, head_width=0.05, head_length=0.1)
        plt.text(*mrot[:,0], "x'", c=color0, ha='right', va='bottom')
        plt.text(*mrot[:,1], "y'", c=color1, ha='right', va='bottom')
        plt.text(0, 0, "O", c='k', ha='right', va='top')
        plt.xlim(min(min(mrot[0,:]), 0)-0.1, max(max(mrot[0,:]), 1)+0.1)
        plt.ylim(min(min(mrot[1,:]), 0)-0.1, max(max(mrot[1,:]), 1)+0.1)
        plt.gca().set_aspect('equal')
        plt.axis('off')
        # # Or that:
        # plt.arrow(*[0,0], *(0.9*mrot[:,0]), color=color0, head_width=0.05, head_length=0.1)
        # plt.arrow(*[0,0], *(0.9*mrot[:,1]), color=color1, head_width=0.05, head_length=0.1)
        # plt.text(*mrot[:,0], "x'", c=color0, ha='right', va='bottom')
        # plt.text(*mrot[:,1], "y'", c=color1, ha='right', va='bottom')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.xlim(min(min(mrot[0,:]), 0)-0.1, max(max(mrot[0,:]), 1)+0.1)
        # plt.ylim(min(min(mrot[1,:]), 0)-0.1, max(max(mrot[1,:]), 1)+0.1)
        # plt.gca().set_aspect('equal')
        # plt.gca().spines['left'].set_position('zero')
        # plt.gca().spines['left'].set_position('zero')
        # plt.gca().spines['right'].set_color('none')
        # plt.gca().spines['bottom'].set_position('zero')
        # plt.gca().spines['top'].set_color('none')
        # plt.title(f'Vario cloud: alpha= {alpha} deg.\ntol_dist ={tol_dist} deg. / tol_angle ={tol_angle} deg.')

        plt.sca(ax[0,1])
        # Plot both variogram clouds
        plot_variogramCloud1D(h0, g0, c=color0, alpha=0.5, label="along x'")
        plot_variogramCloud1D(h1, g1, c=color1, alpha=0.5, label="along y'")
        plt.legend()
        #plt.title(f'Total #points = {npair0 + npair1}')

        plt.sca(ax[1,0])
        # Plot variogram cloud along x'
        plot_variogramCloud1D(h0, g0, c=color0, **kwargs)
        plt.title(f"along x' ({npair0} pts)")

        plt.sca(ax[1,1])
        # Plot variogram cloud along y'
        plot_variogramCloud1D(h1, g1, c=color1, **kwargs)
        plt.title(f"along y' ({npair1} pts)")

        plt.suptitle('Vario cloud')
        #plt.suptitle(f'Vario cloud: alpha={alpha} deg.')
        # plt.suptitle(f'Vario cloud: alpha={alpha} deg.\ntol_dist={tol_dist} / tol_angle={tol_angle}deg.')
        # plt.show()

    return (h0, g0, npair0), (h1, g1, npair1)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def variogramExp2D(
        x, v,
        alpha=0.0,
        tol_dist=None,
        tol_angle=None,
        hmax=None,
        alpha_loc_func=None,
        w_factor_loc_func=None,
        coord1_factor_loc_func=None,
        coord2_factor_loc_func=None,
        loc_m=1,
        ncla=(10, 10),
        cla_center=(None, None),
        cla_length=(None, None),
        variogramCloud=None,
        make_plot=True,
        color0='red',
        color1='green',
        figsize=None,
        logger=None,
        **kwargs):
    """
    Computes the two experimental directional variograms (wrt. main axes) for a data set in 2D.

    For the experimental variogram along the 1st (resp. 2nd) main axis, the mean
    point in each class is retrieved from the 1st (resp. 2nd) variogram cloud
    (returned by the function `variogramCloud2D`); along the 1st axis (j=0) (resp.
    2nd axis (j=1)), the i-th class is determined by its center
    `cla_center[j][i]` and its length `cla_length[j][i]`, and corresponds to the
    interval

        `]cla_center[j][i]-cla_length[j][i]/2, cla_center[j][i]+cla_length[j][i]/2]`

    along h1 (resp. h2) (lag) axis (abscissa).

    Parameters
    ----------
    x : 2D array of floats of shape (n, 2)
        data points locations, with n the number of data points, each row of `x`
        is the coordinatates of one data point

    v : 1D array of floats of shape (n,)
        data points values, with n the number of data points, `v[i]` is the data
        value at location `x[i]`

    alpha : float, default: 0.0
        azimuth angle in degrees, defining the main axes
        (see :class:`CovModel2D`)

    tol_dist : sequence of 2 floats, or float, optional
        let `tol_dist` = (tol_dist1, tol_dist2); tol_dist1 (resp. tol_dist2) is the
        maximal distance to the 1st (resp. 2nd) main axis for the lag (vector
        between two data points), such that the pair is integrated in the
        variogram cloud along the 1st (resp. 2nd) main axis;
        note: if `tol_dist` is specified as a float or `None` (default), the
        entry is duplicated; if tol_dist1 (resp. tol_dist2) is `None`, then
        tol_dist1 is set to 10% of h1max (resp. h2max) if h1max (resp. h2max) is
        finite, and set to 10.0 otherwise: see parameter `hmax` for the
        definition of h1max and h2max

    tol_angle : sequence of 2 floats, or float, optional
        let `tol_angle` = (tol_angle1, tol_angle2); tol_angle1 is the maximal angle
        in degrees between the lag (vector between two data points) and the 1st
        (resp. 2nd) main axis, such that the pair is integrated in the variogram
        cloud along the 1st (resp. 2nd) main axis;
        note: if `tol_angle` is specified as a float, it is duplicated;
        by default (`None`): `tol_angle` is set to 45.0

    hmax : sequence of 2 floats, or float, optional
        let `hmax` = (h1max, h2max); h1max (resp. h2max) is the maximal distance
        between a pair of data points along the 1st (resp. 2nd) main axis, such
        that the pair is integrated in the variogram cloud along the 1st
        (resp. 2nd) main axis;
        note: if `hmax` is specified as a float or `None` (default), the entry is
        duplicated, and `None`, `numpy.nan` are converted to `numpy.inf` (no
        restriction)

    alpha_loc_func : function (`callable`), optional
        function returning azimuth angle, defining the main axes, as function of
        a given location in 2D, i.e. the main axes are defined locally

    w_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "weight" as function of a given
        location in 2D, i.e. "g" values (i.e. ordinate axis component in the two
        variograms) are multiplied

    coord1_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" along the 1st (local) main
        axis as function of a given location in 2D, i.e. "h1" values (i.e.
        abscissa axis component in the 1st variogram) are multiplied
        (the condition wrt h1max, see `hmax`, is checked after)

    coord2_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" along the 2nd (local) main
        axis as function of a given location in 2D, i.e. "h2" values (i.e.
        abscissa axis component in the 2nd variogram) are multiplied
        (the condition wrt h2max, see `hmax`, is checked after)

    loc_m : int, default: 1
        integer (greater than or equal to 0) defining how the function(s)
        `*_loc_func` (above) are evaluated for a pair of two locations x1, x2
        (data point locations):

        - if `loc_m>0` the segment from x1 to x2 is divided in `loc_m` intervals \
        of same length and the mean of the evaluations of the function at the \
        (`loc_m` + 1) interval bounds is computed
        - if `loc_m=0`, the evaluation at x1 is considered

    ncla : sequence of 2 ints, default: (10, 10)
        number of classes along each main axis, the parameter `ncla[j]` is used
        if `cla_center[j]=None`, in that situation `ncla[j]` classes are
        considered and the class centers are set to

        - `cla_center[j][i] = (i+0.5)*l, i=0,...,ncla[j]-1`

        with l = H / ncla[j], H being the max of the distance, along the
        corresponding main axis, between two points of the considered pairs
        (in the variogram cloud along the corresponding main axis);
        if `cla_center[j]` is specified (not `None`), the number of classes
        (`ncla[j]`) is set to the length of the sequence `cla_center[j]`
        (ignoring the value passed as argument)

    cla_center : sequence of length 2
        cla_center[j] : 1D array-like of floats, or `None` (default)
            center of each class (in abscissa) in the experimental variogram
            along the 1st (j=0) (resp. 2nd (j=1)) main axis; by default (`None`):
            `cla_center[j]` is defined from `ncla[j]` (see above)

    cla_length : sequence of length 2
        cla_length[j] : 1D array-like of floats, or float, or `None`
            length of each class centered at `cla_center[j]` (in abscissa) in the
            experimental variogram along the 1st (j=0) (resp. 2nd (j=1)) main
            axis:

            - if `cla_length[j]` is a sequence, it should be of length `ncla[j]`
            - if `cla_length[j]` is a float, the value is repeated `ncla[j]` times
            - if `cla_length[j]=None` (default), the minimum of difference between \
            two sucessive class centers along the corresponding main axis (`np.inf` \
            if one class) is used and repeated `ncla[j]` times

    variogramCloud : sequence of two 3-tuple, optional
        `variogramCloud` = ((h0, g0, npair0), (h1, g1, npair1))
        is variogram clouds (already computed and returned by the function
        `variogramCloud2D` (npair0, npair1 not used)) along the two main axes;
        in this case, `x`, `v`, `alpha`, `tol_dist`, `tol_angle`, `hmax`,
        `alpha_loc_func`, `w_factor_loc_func`, `coord1_factor_loc_func`,
        `coord2_factor_loc_func`, `loc_m` are not used

        By default (`None`): the variogram clouds are computed by using the
        function `variogramCloud2D`

    make_plot : bool, default: True
        indicates if the experimental variograms are plotted (in a new "2x2"
        figure)

    color0 : color, default: 'red'
        color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
        the plot of the experimental variogram along the 1st main axis
        (if `make_plot=True`)

    color1 : color, default: 'green'
        color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
        the plot of the experimental variogram along the 2nd main axis
        (if `make_plot=True`)

    figsize : 2-tuple, optional
        size of the new "2x2" figure (if `make_plot=True`)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs : dict
        keyword arguments passed to the funtion `plot_variogramExp1D`
        (if `make_plot=True`)

    Returns
    -------
    (hexp0, gexp0, cexp0) : 3-tuple
        hexp0, gexp0 : 1D arrays of floats of same length
            coordinates of the points of the experimental variogram along the
            1st main axis

        cexp0 : 1D array of ints
            array of same length as `hexp0`, `gexp0`, number of points (pairs of
            data points considered) in each class in the variogram cloud along
            the 1st main axis

    (hexp1, gexp1, cexp1) : 3-tuple
        hexp1, gexp1 : 1D arrays of floats of same length
            coordinates of the points of the experimental variogram along the
            2nd main axis

        cexp1 : 1D array of ints
            array of same length as `hexp1`, `gexp1`, number of points (pairs of
            data points considered) in each class in the variogram cloud along
            the 2nd main axis
    """
    fname = 'variogramExp2D'

    # Compute variogram clouds if needed
    if variogramCloud is None:
        try:
            vc = variogramCloud2D(
                    x, v, alpha=alpha, tol_dist=tol_dist, tol_angle=tol_angle, hmax=hmax,
                    alpha_loc_func=alpha_loc_func, 
                    w_factor_loc_func=w_factor_loc_func,
                    coord1_factor_loc_func=coord1_factor_loc_func, 
                    coord2_factor_loc_func=coord2_factor_loc_func, 
                    loc_m=loc_m,
                    make_plot=False,
                    logger=logger)
        except Exception as exc:
            err_msg = f'{fname}: cannot compute variogram cloud (2D)'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg) from exc

    else:
        vc = variogramCloud
    # -> vc[0] = (h0, g0, npair0) and vc[1] = (h1, g1, npair1)

    # Compute variogram experimental in each direction (using function variogramExp1D)
    ve = [None, None]
    for j in (0, 1):
        try:
            ve[j] = variogramExp1D(
                        None, None, 
                        hmax=None, w_factor_loc_func=None, coord_factor_loc_func=None, loc_m=loc_m,
                        ncla=ncla[j], cla_center=cla_center[j], cla_length=cla_length[j], 
                        variogramCloud=vc[j], make_plot=False, logger=logger)
        except Exception as exc:
            err_msg = f'{fname}: cannot compute experimental variogram in one direction'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg) from exc

    (hexp0, gexp0, cexp0), (hexp1, gexp1, cexp1) = ve

    if make_plot:
        # Rotation matrix
        a = alpha * np.pi/180.
        ca, sa = np.cos(a), np.sin(a)
        mrot = np.array([[ca, sa], [-sa, ca]])

        _, ax = plt.subplots(2,2, figsize=figsize)
        plt.sca(ax[0,0])
        # Plot system Oxy and Ox'y'
        # This:
        plt.arrow(*[0,0], *[0.9,0], color='k', head_width=0.05, head_length=0.1)
        plt.arrow(*[0,0], *[0,0.9], color='k', head_width=0.05, head_length=0.1)
        plt.text(*[1,0], "x", c='k', ha='left', va='top')
        plt.text(*[0,1], "y", c='k', ha='left', va='top')
        plt.arrow(*[0,0], *(0.9*mrot[:,0]), color=color0, head_width=0.05, head_length=0.1)
        plt.arrow(*[0,0], *(0.9*mrot[:,1]), color=color1, head_width=0.05, head_length=0.1)
        plt.text(*mrot[:,0], "x'", c=color0, ha='right', va='bottom')
        plt.text(*mrot[:,1], "y'", c=color1, ha='right', va='bottom')
        plt.text(0, 0, "O", c='k', ha='right', va='top')
        plt.xlim(min(min(mrot[0,:]), 0)-0.1, max(max(mrot[0,:]), 1)+0.1)
        plt.ylim(min(min(mrot[1,:]), 0)-0.1, max(max(mrot[1,:]), 1)+0.1)
        plt.gca().set_aspect('equal')
        plt.axis('off')
        # # Or that:
        # plt.arrow(*[0,0], *(0.9*mrot[:,0]), color=color0, head_width=0.05, head_length=0.1)
        # plt.arrow(*[0,0], *(0.9*mrot[:,1]), color=color1, head_width=0.05, head_length=0.1)
        # plt.text(*mrot[:,0], "x'", c=color0, ha='right', va='bottom')
        # plt.text(*mrot[:,1], "y'", c=color1, ha='right', va='bottom')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.xlim(min(min(mrot[0,:]), 0)-0.1, max(max(mrot[0,:]), 1)+0.1)
        # plt.ylim(min(min(mrot[1,:]), 0)-0.1, max(max(mrot[1,:]), 1)+0.1)
        # plt.gca().set_aspect('equal')
        # plt.gca().spines['left'].set_position('zero')
        # plt.gca().spines['left'].set_position('zero')
        # plt.gca().spines['right'].set_color('none')
        # plt.gca().spines['bottom'].set_position('zero')
        # plt.gca().spines['top'].set_color('none')

        plt.sca(ax[0,1])
        # Plot variogram exp along x' and along y'
        plot_variogramExp1D(hexp0, gexp0, cexp0, show_count=False, grid=True, c=color0, alpha=0.5, label="along x'")
        plot_variogramExp1D(hexp1, gexp1, cexp1, show_count=False, grid=True, c=color1, alpha=0.5, label="along y'")
        plt.legend()

        plt.sca(ax[1,0])
        # Plot variogram exp along x'
        plot_variogramExp1D(hexp0, gexp0, cexp0, color=color0, **kwargs)
        plt.title("along x'")

        plt.sca(ax[1,1])
        # Plot variogram exp along y'
        plot_variogramExp1D(hexp1, gexp1, cexp1, color=color1, **kwargs)
        plt.title("along y'")

        plt.suptitle(f'Vario exp.: alpha={alpha}deg.\ntol_dist={tol_dist} / tol_angle={tol_angle}deg.')
        # plt.show()

    return (hexp0, gexp0, cexp0), (hexp1, gexp1, cexp1)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def variogramExp2D_rose(
        x, v,
        r_max=None,
        r_ncla=10,
        phi_ncla=12,
        set_polar_subplot=True,
        figsize=None,
        logger=None,
        **kwargs):
    """
    Computes and shows an experimental variogram rose for a data set in 2D.

    The lags vectors between the pairs of data points are divided in classes
    according to length (radius) and angle from the x-axis counter-clockwise
    (warning: opposite sense to the sense given by angle in definition of a
    covariance model in 2D, see :class:`CovModel2D`).

    Parameters
    ----------
    x : 2D array of floats of shape (n, 2)
        data points locations, with n the number of data points, each row of `x`
        is the coordinatates of one data point

    v : 1D array of floats of shape (n,)
        data points values, with n the number of data points, `v[i]` is the data
        value at location `x[i]`

    r_max : float, optional
        maximal radius, i.e. maximal length of 2D lag vector between a pair of
        data points to be integrated in the variogram rose plot;
        note: `None` (default), `numpy.nan` are converted to `numpy.inf` (no
        restriction)

    r_ncla : int, default: 10
        number of classes for radius

    phi_ncla : int, default: 12
        number of classes for angle on an half of the whole disk

    set_polar_subplot : bool, default: True
        - if True: a new figure is created, with "polar" subplot
        - if False: the current axis is used for the plot

    figsize : 2-tuple, optional
        size of the figure (if `set_polar_subplot=True`)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs : dict
        keyword arguments passed to the funtion
        `matplotlib.pyplot.pcolormesh` (cmap, etc.)
    """
    fname = 'variogramExp2D_rose'

    # Number of data points
    n = x.shape[0]

    # Check length of v
    if len(v) != n:
        err_msg = f'{fname}: length of `v` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # Compute lag vector (h) and gamma value (g) for pair of points with distance less than or equal to hmax
    if r_max is None or np.isnan(r_max):
        # consider all pairs of points
        npair = int(0.5*(n-1)*n)
        h = np.zeros((npair, 2))
        g = np.zeros(npair)
        j = 0
        for i in range(n-1):
            jj = n-1-i
            h[j:(j+jj),:] = x[(i+1):, :] - x[i,:]
            g[j:(j+jj)] = 0.5*(v[i] - v[(i+1):])**2
            j = j+jj

    else:
        # consider only pairs of points with a distance less than or equal to hmax
        r_max2 = r_max**2
        h, g = [], []

        npair = 0
        for i in range(n-1):
            htmp = x[(i+1):, :] - x[i,:] # 2-dimensional array (n-1-i) x dim
            ind = np.where(np.sum(htmp**2, axis=1) <= r_max2)[0]
            h.append(htmp[ind])
            g.append(0.5*(v[i] - v[i+1+ind])**2)
            npair = npair + len(ind)

        if npair > 0:
            h = np.vstack(h)
            g = np.hstack(g)

    # Compute r, phi (radius and angle in complex plane) for each lag vector
    r = np.sqrt(np.sum(h*h, axis=1))
    phi = np.array([np.arctan2(hh[1], hh[0]) for hh in h])
    # or: phi = np.array([np.angle(np.complex(*hh)) for hh in h])
    # ... set each angle phi in [-np.pi/2, np.pi/2[ (symmetry of variogram)
    pi_half = 0.5*np.pi
    np.putmask(phi, phi < -pi_half, phi + np.pi)
    np.putmask(phi, phi >= pi_half, phi - np.pi)

    # Set classes for r and phi
    if r_max is None or np.isnan(r_max):
        r_max = np.max(r)

    r_cla = np.linspace(0., r_max, r_ncla+1)
    phi_cla = np.linspace(-pi_half, pi_half, phi_ncla+1)

    # Compute rose map
    gg = np.nan * np.ones((phi_ncla, r_ncla)) # initialize gamma values
    for ip in range(phi_ncla):
        pind = np.all((phi >= phi_cla[ip], phi < phi_cla[ip+1]), axis=0)
        for ir in range(r_ncla):
            rind = np.all((r >= r_cla[ir], r < r_cla[ir+1]), axis=0)
            gg[ip, ir] = np.mean(g[np.all((pind, rind), axis=0)])

    gg = np.vstack((gg, gg))
    rr, pp = np.meshgrid(r_cla, np.hstack((phi_cla[:-1],phi_cla+np.pi)))

    # Set default color map to 'terrain' if not given in kwargs
    if 'cmap' not in kwargs.keys():
        kwargs['cmap'] = 'terrain' #'nipy_spectral'

    if set_polar_subplot:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='polar')
    plt.pcolormesh(pp, rr, gg, **kwargs)
    plt.colorbar()
    plt.title('Vario rose (gamma value)')
    plt.grid()
    # plt.show()
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel2D_fit(
        x, v, cov_model,
        hmax=None,
        alpha_loc_func=None,
        w_factor_loc_func=None,
        coord1_factor_loc_func=None,
        coord2_factor_loc_func=None,
        loc_m=1,
        make_plot=True,
        figsize=None,
        verbose=0,
        logger=None,
        **kwargs):
    """
    Fits a covariance model in 2D (for data in 2D).

    The parameter `cov_model` is a covariance model in 2D where all the
    parameters to be fitted are set to `numpy.nan`. The fit is done according to
    the variogram cloud, by using the function `scipy.optimize.curve_fit`.

    Parameters
    ----------
    x : 2D array of floats of shape (n, 2)
        data points locations, with n the number of data points, each row of `x`
        is the coordinates of one data point

    v : 1D array of floats of shape (n,)
        data points values, with n the number of data points, `v[i]` is the data
        value at location `x[i]`

    cov_model : :class:`CovModel2D`
        covariance model to otpimize (parameters set to `numpy.nan` are optimized)

    hmax : sequence of 2 floats, or float, optional
        the pairs of data points with lag h (in rotated coordinates system if
        applied) satisfying

        .. math::
            (h[0]/hmax[0])^2 + (h[1]/hmax[1])^2 \\leqslant 1

        are taking into account in the variogram cloud
        note: if `hmax` is specified as a float or `None` (default), the entry is
        duplicated, and `None`, `numpy.nan` are converted to `numpy.inf` (no
        restriction)

    alpha_loc_func : function (`callable`), optional
        function returning azimuth angle, defining the main axes, as function of
        a given location in 2D, i.e. the main axes are defined locally

    w_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "weight" as function of a given
        location in 2D, i.e. "g" values (i.e. ordinate axis component in the two
        variograms) are multiplied

    coord1_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" along the 1st (local) main
        axis as function of a given location in 2D, i.e. "h1" values (i.e.
        abscissa axis component in the 1st variogram) are multiplied
        (the condition wrt `hmax`, is checked after)

    coord2_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" along the 2nd (local) main
        axis as function of a given location in 2D, i.e. "h2" values (i.e.
        abscissa axis component in the 2nd variogram) are multiplied
        (the condition wrt `hmax`, is checked after)

    loc_m : int, default: 1
        integer (greater than or equal to 0) defining how the function(s)
        `*_loc_func` (above) are evaluated for a pair of two locations x1, x2
        (data point locations):

        - if `loc_m>0` the segment from x1 to x2 is divided in `loc_m` intervals \
        of same length and the mean of the evaluations of the function at the \
        (`loc_m` + 1) interval bounds is computed
        - if `loc_m=0`, the evaluation at x1 is considered

    make_plot : bool, default: True
        indicates if the fitted covariance model is plotted (in a new "1x2"
        figure, using the method `plot_model` with default parameters)

    figsize : 2-tuple, optional
        size of the new "1x2" figure (if `make_plot=True`)

    verbose : int, default: 0
        verbose mode, higher implies more printing (info)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs : dict
        keyword arguments passed to the funtion `scipy.optimize.curve_fit`

    Returns
    -------
    cov_model_opt: :class:`CovModel2D`
        optimized covariance model

    popt: 1D array
        values of the optimal parameters, corresponding to the parameters of the
        input covariance model (`cov_model`) set to `numpy.nan`, in the order of
        appearance (vector of optimized parameters returned by
        `scipy.optimize.curve_fit`)

    Examples
    --------
    The following allows to fit a covariance model made up of a gaussian
    elementary model and a nugget effect (nugget elementary model), where the
    azimuth angle (defining the main axes), the weight and ranges of the gaussian
    elementary model and the weight of the nugget effect are fitted (optimized)
    in intervals given by the keyword argument `bounds`. The arguments `x`, `v`
    are the data points and values, and the fitted covariance model is not plotted
    (`make_plot=False`)

        >>> # covariance model to optimize
        >>> cov_model = CovModel2D(elem=[
        >>>     ('gaussian', {'w':np.nan, 'r':[np.nan, np.nan]}), # elem. contrib.
        >>>     ('nugget', {'w':np.nan})                          # elem. contrib.
        >>>     ], alpha=np.nan,                                  # azimuth angle
        >>>     name='')
        >>> covModel2D_fit(x, v, cov_model_to_optimize,
        >>>                bounds=([ 0.0,   0.0,   0.0,  0.0, -90.0],  # lower bounds
        >>>                        [10.0, 100.0, 100.0, 10.0,  90.0]), # upper bounds
        >>>                                                    # for parameters to fit
        >>>                make_plot=False)
    """
    fname = 'covModel2D_fit'

    # Check cov_model
    if not isinstance(cov_model, CovModel2D):
        err_msg = f'{fname}: `cov_model` is not a covariance model in 2D'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # if cov_model.__class__.__name__ != 'CovModel2D':
    #     err_msg = f'{fname}: `cov_model` is not a covariance model in 2D'
    #     if logger: logger.error(err_msg)
    #     raise CovModelError(err_msg)

    # Prevent calculation if covariance model is not stationary
    if not cov_model.is_stationary():
        err_msg = f'{fname}: `cov_model` is not stationary: fit cannot be applied'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # Work on a (deep) copy of cov_model
    cov_model_opt = copy.deepcopy(cov_model)

    # Get index of element, key of parameters and index of range to fit
    ielem_to_fit=[]
    key_to_fit=[]
    ir_to_fit=[] # if key is equal to 'r' (range), set the index of the range to fit, otherwise set np.nan
    for i, el in enumerate(cov_model_opt.elem):
        for k, val in el[1].items():
            if k == 'r':
                for j in (0, 1):
                    if np.isnan(val[j]):
                        ielem_to_fit.append(i)
                        key_to_fit.append(k)
                        ir_to_fit.append(j)
            elif np.isnan(val):
                ielem_to_fit.append(i)
                key_to_fit.append(k)
                ir_to_fit.append(np.nan)

    # Is angle alpha must be fit ?
    alpha_to_fit = np.isnan(cov_model_opt.alpha)

    nparam = len(ielem_to_fit) + int(alpha_to_fit)
    if nparam == 0:
        # print('No parameter to fit!')
        return cov_model_opt, np.array([])

    # Set hmax as an array of shape (2,)
    hmax = np.atleast_1d(hmax).astype('float').reshape(-1) # None is converted to nan
    hmax[np.isnan(hmax)] = np.inf # convert nan to inf
    if hmax.size == 1:
        hmax = np.array([hmax[0], hmax[0]])
    elif hmax.size != 2:
        err_msg = f'{fname}: size of `hmax` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    if alpha_to_fit and hmax[0] != hmax[1] and verbose > 0:
        if logger:
            logger.warning(f'{fname}: as angle is flagged for fitting, all the components of `hmax` should be equal')
        else:
            print(f'{fname}: WARNING: as angle is flagged for fitting, all the components of `hmax` should be equal')

    if not alpha_to_fit and cov_model_opt.alpha != 0.0:
        alpha_copy = cov_model_opt.alpha
        rotate_coord_sys = True
        mrot = cov_model_opt.mrot()
        cov_model_opt.alpha = 0.0 # set (temporarily) to 0.0
    else:
        rotate_coord_sys = False

    # Set types of local transformations
    #    alpha_loc: True / False: is local angle alpha used ?
    #    w_loc:     True / False: is local w (weight / sill) used ?
    #    coord_loc: integer
    #               0: no transformation
    #               1: transformation for 1st coordinate only
    #               2: transformation for 2nd coordinate only
    #               3: distinct transformations for 1st and 2nd coordinates
    #               8: same transformation for 1st and 2nd coordinates
    alpha_loc = False
    w_loc = False
    coord_loc = 0
    if alpha_loc_func is not None:
        # factor to transform angle in degree into radian
        t_angle = np.pi/180.0
        alpha_loc = True

    if w_factor_loc_func is not None:
        w_loc = True

    if coord1_factor_loc_func is not None:
        coord_loc = coord_loc + 1
    if coord2_factor_loc_func is not None:
        coord_loc = coord_loc + 2
        if coord1_factor_loc_func == coord2_factor_loc_func:
            coord_loc = 8

    if alpha_loc or w_loc or coord_loc > 0:
        transform_flag = True
    else:
        transform_flag = False

    # Compute lag vector (h) and gamma value (g) for pair of points with distance less than or equal to hmax
    n = x.shape[0] # number of points
    if np.all(np.isinf(hmax)):
        # Consider all pairs of points
        npair = int(0.5*(n-1)*n)
        h = np.zeros((npair, 2))
        g = np.zeros(npair)
        j = 0
        if transform_flag:
            wf = 1.0 # default weight factor
            if loc_m > 0:
                for i in range(n-1):
                    jj = n-1-i
                    d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
                    dx = d/loc_m
                    ddx = np.asarray([x[i]+np.outer(np.arange(loc_m+1), dxk) for dxk in dx]) # 3-dimensional array (n-1-i) x (loc_m+1) x dim
                    if rotate_coord_sys:
                        # Rotate according to new system
                        d = d.dot(mrot)
                    if alpha_loc:
                        a = t_angle * alpha_loc_func(ddx.reshape(-1, 2)).reshape(-1, loc_m+1) # 2-dimensional array (n-1-i) x (loc_m+1)
                        # a = t_angle * np.asarray([alpha_loc_func(ddxk) for ddxk in ddx])
                        cos_a, sin_a = np.mean(np.cos(a), axis=1), np.mean(np.sin(a), axis=1)
                        d = np.asarray([dk.dot(np.array([[cos_ak, sin_ak], [-sin_ak, cos_ak]])) for (cos_ak, sin_ak, dk) in zip (cos_a, sin_a, d)])
                    if coord_loc == 1:
                        d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 2)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                        # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                    elif coord_loc == 2:
                        d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 2)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                        # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                    elif coord_loc == 3:
                        d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 2)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                        d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 2)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                        # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                        # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                    elif coord_loc == 8:
                        d = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 2)).reshape(-1, loc_m+1), axis=1)*d.T).T
                        # d = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d.T).T
                    h[j:(j+jj),:] = d
                    if w_loc:
                        wf = np.mean(w_factor_loc_func(ddx.reshape(-1, 2)).reshape(-1, loc_m+1), axis=1)
                        # wf = np.asarray([np.mean(w_factor_loc_func(ddxk)) for ddxk in ddx])
                    g[j:(j+jj)] = wf * 0.5*(v[i] - v[(i+1):])**2
                    j = j+jj
            else:
                for i in range(n-1):
                    jj = n-1-i
                    d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
                    if rotate_coord_sys:
                        # Rotate according to new system
                        d = d.dot(mrot)
                    if alpha_loc:
                        a = t_angle * alpha_loc_func(x[i])[0]
                        cos_a, sin_a = np.cos(a), np.sin(a)
                        d = d.dot(np.array([[cos_a, sin_a], [-sin_a, cos_a]]))
                        # d = np.array([[cos_a, -sin_a], [sin_a, cos_a]]).dot(d.T).T
                    if coord_loc == 1:
                        d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                    elif coord_loc == 2:
                        d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                    elif coord_loc == 3:
                        d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                        d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                    elif coord_loc == 8:
                        d = coord1_factor_loc_func(x[i])[0]*d
                    h[j:(j+jj),:] = d
                    if w_loc:
                        wf = w_factor_loc_func(x[i])[0]
                    g[j:(j+jj)] = wf * 0.5*(v[i] - v[(i+1):])**2
                    j = j+jj
        else:
            if rotate_coord_sys:
                # Rotate according to new system
                x = x.dot(mrot)
            for i in range(n-1):
                jj = n-1-i
                d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
                h[j:(j+jj),:] = d
                g[j:(j+jj)] = 0.5*(v[i] - v[(i+1):])**2
                j = j+jj
    else:
        # Consider only pairs of points according to parameter hmax, i.e.
        #   pairs with lag h (in rotated coordinates system if applied) satisfying
        #   (h[0]/hmax[0])**2 + (h[1]/hmax[1])**2 <= 1
        h, g = [], []
        npair = 0
        if transform_flag:
            wf = 1.0 # default weight factor
            if loc_m > 0:
                for i in range(n-1):
                    d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
                    dx = d/loc_m
                    ddx = np.asarray([x[i]+np.outer(np.arange(loc_m+1), dxk) for dxk in dx]) # 3-dimensional array (n-1-i) x (loc_m+1) x dim
                    if rotate_coord_sys:
                        # Rotate according to new system
                        d = d.dot(mrot)
                    if alpha_loc:
                        a = t_angle * alpha_loc_func(ddx.reshape(-1, 2)).reshape(-1, loc_m+1) # 2-dimensional array (n-1-i) x (loc_m+1)
                        # a = t_angle * np.asarray([alpha_loc_func(ddxk) for ddxk in ddx])
                        cos_a, sin_a = np.mean(np.cos(a), axis=1), np.mean(np.sin(a), axis=1)
                        d = np.asarray([dk.dot(np.array([[cos_ak, sin_ak], [-sin_ak, cos_ak]])) for (cos_ak, sin_ak, dk) in zip (cos_a, sin_a, d)])
                    if coord_loc == 1:
                        d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 2)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                        # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                    elif coord_loc == 2:
                        d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 2)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                        # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                    elif coord_loc == 3:
                        d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 2)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                        d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 2)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                        # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                        # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                    elif coord_loc == 8:
                        d = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 2)).reshape(-1, loc_m+1), axis=1)*d.T).T
                        # d = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d.T).T
                    ind = np.where(np.sum((d/hmax)**2, axis=1) <= 1.0)[0]
                    if len(ind) == 0:
                        continue
                    h.append(d[ind])
                    if w_loc:
                        wf = np.mean(w_factor_loc_func(ddx[ind].reshape(-1, 2)).reshape(-1, loc_m+1), axis=1)
                        # wf = np.asarray([np.mean(w_factor_loc_func(ddxk)) for ddxk in ddx[ind]])
                    g.append(wf * 0.5*(v[i] - v[i+1+ind])**2)
                    npair = npair + len(ind)
            else:
                for i in range(n-1):
                    d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
                    if rotate_coord_sys:
                        # Rotate according to new system
                        d = d.dot(mrot)
                    if alpha_loc:
                        a = t_angle * alpha_loc_func(x[i])[0]
                        cos_a, sin_a = np.cos(a), np.sin(a)
                        d = d.dot(np.array([[cos_a, sin_a], [-sin_a, cos_a]]))
                        # d = np.array([[cos_a, -sin_a], [sin_a, cos_a]]).dot(d.T).T
                    if coord_loc == 1:
                        d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                    elif coord_loc == 2:
                        d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                    elif coord_loc == 3:
                        d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                        d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                    elif coord_loc == 8:
                        d = coord1_factor_loc_func(x[i])[0]*d
                    ind = np.where(np.sum((d/hmax)**2, axis=1) <= 1.0)[0]
                    if len(ind) == 0:
                        continue
                    h.append(d[ind])
                    if w_loc:
                        wf = w_factor_loc_func(x[i])[0]
                    g.append(wf * 0.5*(v[i] - v[i+1+ind])**2)
                    npair = npair + len(ind)
        else:
            if rotate_coord_sys:
                # Rotate according to new system
                x = x.dot(mrot)
            for i in range(n-1):
                d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
                ind = np.where(np.sum((d/hmax)**2, axis=1) <= 1.0)[0]
                if len(ind) == 0:
                    continue
                h.append(d[ind])
                g.append(0.5*(v[i] - v[i+1+ind])**2)
                npair = npair + len(ind)
        if npair > 0:
            h = np.vstack(h)
            g = np.hstack(g)

    if npair == 0:
        err_msg = f'{fname}: no pair of points (in variogram cloud) for fitting'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # Define the function to optimize in a format compatible with curve_fit from scipy.optimize
    def func(d, *p):
        """
        Function whose p is the vector of parameters to optimize.

        Parameters
        ----------
        d : 1D array
            xdata, i.e. lags (h) from the variogram cloud where the current
            covariance model is evaluated

        p : 1D array
            current values of the parameters (floats) to optimize in the
            covariance model (parameters to optimized are identified with
            ielem_to_fit, key_to_fit, computed above)

        Returns
        -------
        v: 1D array
            evaluations of the current variogram model at `d`
        """
        for i, (iel, k, j) in enumerate(zip(ielem_to_fit, key_to_fit, ir_to_fit)):
            if k == 'r':
                cov_model_opt.elem[iel][1]['r'][j] = p[i]
            else:
                cov_model_opt.elem[iel][1][k] = p[i]
        if alpha_to_fit:
            cov_model_opt.alpha = p[-1]
            cov_model_opt._mrot = None # reset attribute _mrot !
        return cov_model_opt(d, vario=True)

    # Optimize parameters with curve_fit: initial vector of parameters (p0) must be given
    #   because number of parameter to fit in function func is not known in its expression
    bounds = None
    if 'bounds' in kwargs.keys():
        bounds = kwargs['bounds']

    if 'p0' not in kwargs.keys():
        # add default p0 in kwargs
        p0 = np.ones(nparam)
        if bounds is not None:
            # adjust p0 to given bounds
            for i in range(nparam):
                if np.isinf(bounds[0][i]):
                    if np.isinf(bounds[1][i]):
                        p0[i] = 1.
                    else:
                        p0[i] = bounds[1][i]
                elif np.isinf(bounds[1][i]):
                    p0[i] = bounds[0][i]
                else:
                    p0[i] = 0.5*(bounds[0][i]+bounds[1][i])
        kwargs['p0'] = p0
    else:
        if len(kwargs['p0']) != nparam:
            err_msg = f'{fname}: length of `p0` and number of parameters to fit differ'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    # Fit with curve_fit
    try:
        popt, pcov = scipy.optimize.curve_fit(func, h, g, **kwargs)
        if rotate_coord_sys:
            # Restore alpha
            cov_model_opt.alpha = alpha_copy
    except:
        err_msg = f'{fname}: fitting covariance model failed'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    if make_plot:
        cov_model_opt.plot_model(vario=True, figsize=figsize)
        # suptitle already in function cov_model_opt.plot_model...
        # s = [f'Vario opt.: alpha={cov_model_opt.alpha}'] + [f'{el}' for el in cov_model_opt.elem]
        # # plt.suptitle(textwrap.TextWrapper(width=50).fill(s))
        # plt.suptitle('\n'.join(s))

    return cov_model_opt, popt
# ----------------------------------------------------------------------------

# ============================================================================
# Functions for variogram cloud, experimental variogram,
# and covariance model fitting (3D)
# ============================================================================
# ----------------------------------------------------------------------------
def variogramCloud3D(
        x, v,
        alpha=0.0,
        beta=0.0,
        gamma=0.0,
        tol_dist=None,
        tol_angle=None,
        hmax=None,
        alpha_loc_func=None,
        beta_loc_func=None,
        gamma_loc_func=None,
        w_factor_loc_func=None,
        coord1_factor_loc_func=None,
        coord2_factor_loc_func=None,
        coord3_factor_loc_func=None,
        loc_m=1,
        make_plot=True,
        color0='red',
        color1='green',
        color2='blue',
        figsize=None,
        logger=None,
        **kwargs):
    """
    Computes the three directional variogram clouds (wrt. main axes) for a data set in 3D.

    From the pair of the i-th and j-th data points (i not equal to j), let

    .. math::
        \\begin{array}{rcl}
            h(i, j) &=& x_i-x_j \\\\[2mm]
            g(i, j) &=& \\frac{1}{2}(v_i - v_j)^2
        \\end{array}

    where :math:`x_i` and :math:`x_j` are the coordinates of the i-th and j-th
    data points and :math:`v_i` and :math:`v_j` the values at these points
    (:math:`v_i=Z(x_i)`, where :math:`Z` is the considered variable).
    The lag vector h(i, j) is expressed along the three orthogonal main axes,
    h(i, j) = (h1(i, j), h2(i, j), h3(i, j)). Let
    `tol_dist` = (tol_dist1, tol_dist2, tol_dist3),
    `tol_angle` = (tol_angle1, tol_angle2, tol_angle3), and
    `hmax` = (h1max, h2max, h3max) (see parameters below); if distance from h(i, j)
    to the 1st (resp. 2nd, 3rd) main axis does not exceed tol_dist1 (resp.
    toldist2, toldist3), and if the angle between the lag h(i, j) and the 1st (resp.
    2nd, 3rd) main axis does not exceed tol_angle1 (resp. tol_angle2, tol_angle3),
    and if the distance along the 1st (resp. 2nd, 3rd) axis, i.e. \\|h1(i, j)\\|
    (resp. \\|h2(i, j)\\|, \\|h3(i,j)\\|) does not exceed h1max (resp. h2max, h3max), then,
    the point (\\|h1(i, j)\\|, g(i, j)) (resp. (\\|h2(i, j)\\|, g(i, j)),
    (\\|h3(i, j)\\|, g(i, j))) is integrated in the directional variogram cloud along
    the 1st (resp. 2nd, 3rd) main axis.

    Moreover, the parameters `alpha_loc_func`, `beta_loc_func`, `gamma_loc_func`
    allow to account for main axes locally varying in space, and the parameters
    `w_factor_loc_func` and `coord1_factor_loc_func`, `coord2_factor_loc_func`,
    `coord2_factor_loc_func` allow to account for variogram locally varying in
    space with respect to weight and ranges along each main axis resp., by
    multiplying "g", "h1", "h2", "h3" values resp.

    Parameters
    ----------
    x : 2D array of floats of shape (n, 3)
        data points locations, with n the number of data points, each row of `x`
        is the coordinatates of one data point

    v : 1D array of floats of shape (n,)
        data points values, with n the number of data points, `v[i]` is the data
        value at location `x[i]`

    alpha : float, default: 0.0
        azimuth angle in degrees (see :class:`CovModel3D`)

    beta : float, default: 0.0
        dip angle in degrees (see :class:`CovModel3D`)

    gamma : float, default: 0.0
        plunge angle in degrees (see :class:`CovModel3D`)

    tol_dist : sequence of 3 floats, or float, optional
        let `tol_dist` = (tol_dist1, tol_dist2, tol_dist3); tol_dist1 (resp.
        tol_dist2, tol_dist3) is the maximal distance to the 1st (resp. 2nd, 3rd)
        main axis for the lag (vector between two data points), such that the pair
        is integrated in the variogram cloud along the 1st (resp. 2nd, 3rd) main
        axis;
        note: if `tol_dist` is specified as a float or `None` (default), the
        entry is duplicated; if tol_dist1 (resp. tol_dist2, tol_dist3) is `None`,
        then tol_dist1 (resp. tol_dist2, tol_dist3) is set to 10% of h1max (resp.
        h2max, h3max) if h1max (resp. h2max, h3max) is finite, and set to 10.0
        otherwise: see parameter `hmax` for the definition of h1max, h2max and
        h3max

    tol_angle : sequence of 3 floats, or float, optional
        let `tol_angle` = (tol_angle1, tol_angle2, tol_angl3); tol_angle1 (resp.
        tol_angle2, tol_angle3) is the maximal angle in degrees between the lag
        (vector between two data points) and the 1st (resp. 2nd, 3rd) main axis,
        such that the pair is integrated in the variogram cloud along the 1st
        (resp. 2nd, 3rd) main axis;
        note: if `tol_angle` is specified as a float, it is duplicated;
        by default (`None`): `tol_angle` is set to 45.0

    hmax : sequence of 3 floats, or float, optional
        let `hmax` = (h1max, h2max, h3max); h1max (resp. h2max, h3max) is the
        maximal distance between a pair of data points along the 1st (resp. 2nd,
        3rd) main axis, such that the pair is integrated in the variogram cloud
        along the 1st (resp. 2nd, 3rd) main axis;
        note: if `hmax` is specified as a float or `None` (default), the entry is
        duplicated, and `None`, `numpy.nan` are converted to `numpy.inf` (no
        restriction)

    alpha_loc_func : function (`callable`), optional
        function returning azimuth angle, defining the main axes, as function of
        a given location in 3D, i.e. the main axes are defined locally

    beta_loc_func : function (`callable`), optional
        function returning dip angle, defining the main axes, as function of
        a given location in 3D, i.e. the main axes are defined locally

    gamma_loc_func : function (`callable`), optional
        function returning plunge angle, defining the main axes, as function of
        a given location in 3D, i.e. the main axes are defined locally

    w_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "weight" as function of a given
        location in 3D, i.e. "g" values (i.e. ordinate axis component in the two
        variograms) are multiplied

    coord1_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" along the 1st (local) main
        axis as function of a given location in 3D, i.e. "h1" values (i.e.
        abscissa axis component in the 1st variogram) are multiplied
        (the condition wrt h1max, see `hmax`, is checked after)

    coord2_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" along the 2nd (local) main
        axis as function of a given location in 3D, i.e. "h2" values (i.e.
        abscissa axis component in the 2nd variogram) are multiplied
        (the condition wrt h2max, see `hmax`, is checked after)

    coord3_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" along the 3rd (local) main
        axis as function of a given location in 3D, i.e. "h3" values (i.e.
        abscissa axis component in the 3rd variogram) are multiplied
        (the condition wrt h3max, see `hmax`, is checked after)

    loc_m : int, default: 1
        integer (greater than or equal to 0) defining how the function(s)
        `*_loc_func` (above) are evaluated for a pair of two locations x1, x2
        (data point locations):

        - if `loc_m>0` the segment from x1 to x2 is divided in `loc_m` intervals \
        of same length and the mean of the evaluations of the function at the \
        (`loc_m` + 1) interval bounds is computed
        - if `loc_m=0`, the evaluation at x1 is considered

    make_plot : bool, default: True
        indicates if the variogram clouds are plotted (in a new "2x2" figure)

    color0 : color, default: 'red'
        color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
        the plot of the variogram cloud along the 1st main axis
        (if `make_plot=True`)

    color1 : color, default: 'green'
        color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
        the plot of the variogram cloud along the 2nd main axis
        (if `make_plot=True`)

    color2 : color, default: 'blue'
        color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
        the plot of the variogram cloud along the 3rd main axis
        (if `make_plot=True`)

    figsize : 2-tuple, optional
        size of the new "2x2" figure (if `make_plot=True`)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs : dict
        keyword arguments passed to the funtion `plot_variogramCloud1D`
        (if `make_plot=True`)

    Returns
    -------
    (h0, g0, npair0) : 3-tuple
        h0, g0 : 1D arrays of floats of same length
            coordinates of the points in the variogram cloud along 1st main axis
            (see above)

        npair0 : int
            number of points (pairs of data points considered) in the variogram
            cloud along the 1st main axis

    (h1, g1, npair1) : 3-tuple
        h1, g1 : 1D arrays of floats of same length
            coordinates of the points in the variogram cloud along 2nd main axis
            (see above)

        npair1 : int
            number of points (pairs of data points considered) in the variogram
            cloud along the 2nd main axis

    (h2, g2, npair2) : 3-tuple
        h2, g2 : 1D arrays of floats of same length
            coordinates of the points in the variogram cloud along 3rd main axis
            (see above)

        npair2 : int
            number of points (pairs of data points considered) in the variogram
            cloud along the 3rd main axis
    """
    fname = 'variogramCloud3D'

    # Number of data points
    n = x.shape[0]

    # Check length of v
    if len(v) != n:
        err_msg = f'{fname}: length of `v` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # Set hmax as an array of shape (3,)
    hmax = np.atleast_1d(hmax).astype('float').reshape(-1) # None is converted to nan
    hmax[np.isnan(hmax)] = np.inf # convert nan to inf
    if hmax.size == 1:
        hmax = np.array([hmax[0], hmax[0], hmax[0]])
    elif hmax.size != 3:
        err_msg = f'{fname}: size of `hmax` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # Set tol_dist as an array of shape (3,)
    tol_dist = np.atleast_1d(tol_dist).astype('float').reshape(-1) # None is converted to nan
    if tol_dist.size == 1:
        tol_dist = np.array([tol_dist[0], tol_dist[0], tol_dist[0]])
    elif tol_dist.size != 3:
        err_msg = f'{fname}: size of `tol_dist` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    for i in range(3):
        if np.isnan(tol_dist[i]):
            if np.isinf(hmax[i]):
                tol_dist[i] = 10.0
            else:
                tol_dist[i] = 0.1 * hmax[i]

    # Set tol_angle as an array of shape (3,)
    tol_angle = np.atleast_1d(tol_angle).astype('float').reshape(-1) # None is converted to nan
    if tol_angle.size == 1:
        tol_angle = np.array([tol_angle[0], tol_angle[0], tol_angle[0]])
    elif tol_angle.size != 3:
        err_msg = f'{fname}: size of `tol_angle` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    tol_angle[np.isnan(tol_angle)] = 45.0

    if alpha != 0.0 or beta != 0.0 or gamma != 0.0:
        rotate_coord_sys = True
        # Rotation matrix
        a = alpha * np.pi/180.
        b = beta * np.pi/180.
        c = gamma * np.pi/180.
        ca, sa = np.cos(a), np.sin(a)
        cb, sb = np.cos(b), np.sin(b)
        cc, sc = np.cos(c), np.sin(c)
        mrot = np.array([[  ca * cc + sa * sb * sc,  sa * cb,  - ca * sc + sa * sb * cc],
                         [- sa * cc + ca * sb * sc,  ca * cb,    sa * sc + ca * sb * cc],
                         [                 cb * sc,     - sb,                  cb * cc ]])
    else:
        rotate_coord_sys = False

    # Set types of local transformations
    #    alpha_loc: True / False: is local angle alpha used ?
    #    beta_loc : True / False: is local angle beta used ?
    #    gamma_loc: True / False: is local angle gamma used ?
    #    rotation_loc: True / False: is local rotation used ?
    #    w_loc:     True / False: is local w (weight / sill) used ?
    #    coord_loc: integer
    #               0: no transformation
    #               1: transformation for 1st coordinate only
    #               2: transformation for 2nd coordinate only
    #               3: distinct transformations for 1st and 2nd coordinates, no transformation for 3rd coordinate
    #               4: transformation for 3rd coordinate only
    #               5: distinct transformations for 1st and 3rd coordinates, no transformation for 2nd coordinate
    #               6: distinct transformations for 2nd and 3rd coordinates, no transformation for 1st coordinate
    #               7: distinct transformations for 1st, 2nd and 3rd coordinates
    #               8: same transformation for 1st, 2nd and 3rd coordinates
    #               9: same transformation for 1st and 2nd coordinates, no transformation for 3rd coordinate
    #              10: same transformation for 1st and 3rd coordinates, no transformation for 2nd coordinate
    #              11: same transformation for 2nd and 3rd coordinates, no transformation for 1st coordinate
    #              12: same transformation for 1st and 2nd coordinates, distinct transformation for 3rd coordinate
    #              13: same transformation for 1st and 3rd coordinates, distinct transformation for 2nd coordinate
    #              14: same transformation for 2nd and 3rd coordinates, distinct transformation for 1st coordinate
    alpha_loc = False
    beta_loc = False
    gamma_loc = False
    rotation_loc = False
    w_loc = False
    coord_loc = 0
    if alpha_loc_func is not None:
        alpha_loc = True
        rotation_loc = True
    if beta_loc_func is not None:
        beta_loc = True
        rotation_loc = True
    if gamma_loc_func is not None:
        gamma_loc = True
        rotation_loc = True

    if rotation_loc:
        # factor to transform angle in degree into radian
        t_angle = np.pi/180.0

    if w_factor_loc_func is not None:
        w_loc = True

    if coord1_factor_loc_func is not None:
        coord_loc = coord_loc + 1
    if coord2_factor_loc_func is not None:
        coord_loc = coord_loc + 2
    if coord3_factor_loc_func is not None:
        coord_loc = coord_loc + 4
    if coord_loc == 3:
        if coord1_factor_loc_func == coord2_factor_loc_func:
            coord_loc = 9
    elif coord_loc == 5:
        if coord1_factor_loc_func == coord3_factor_loc_func:
            coord_loc = 10
    elif coord_loc == 6:
        if coord2_factor_loc_func == coord3_factor_loc_func:
            coord_loc = 11
    elif coord_loc == 7:
        if coord1_factor_loc_func == coord2_factor_loc_func and coord1_factor_loc_func == coord3_factor_loc_func:
            coord_loc = 8
        elif coord1_factor_loc_func == coord2_factor_loc_func:
            coord_loc = 12
        elif coord1_factor_loc_func == coord3_factor_loc_func:
            coord_loc = 13
        elif coord2_factor_loc_func == coord3_factor_loc_func:
            coord_loc = 14

    if alpha_loc or beta_loc or gamma_loc or w_loc or coord_loc > 0:
        transform_flag = True
    else:
        transform_flag = False

    # Tolerance for slope compute from tol_angle
    tol_s = np.tan(tol_angle*np.pi/180)

    # Compute variogram clouds
    h0, g0, h1, g1, h2, g2 = [], [], [], [], [], []
    if transform_flag:
        wf = 1.0 # default weight factor
        if loc_m > 0:
            for i in range(n-1):
                d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
                dx = d/loc_m
                ddx = np.asarray([x[i]+np.outer(np.arange(loc_m+1), dxk) for dxk in dx]) # 3-dimensional array (n-1-i) x (loc_m+1) x dim
                if rotate_coord_sys:
                    # Rotate according to new system
                    d = d.dot(mrot)
                if rotation_loc:
                    if alpha_loc:
                        a = t_angle * alpha_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1) # 2-dimensional array (n-1-i) x (loc_m+1)
                        # a = t_angle * np.asarray([alpha_loc_func(ddxk) for ddxk in ddx])
                        ca, sa = np.mean(np.cos(a), axis=1), np.mean(np.sin(a), axis=1)
                    else:
                        ca, sa = np.ones(d.shape[0]), np.zeros(d.shape[0])
                    if beta_loc:
                        a = t_angle * beta_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1) # 2-dimensional array (n-1-i) x (loc_m+1)
                        # a = t_angle * np.asarray([beta_loc_func(ddxk) for ddxk in ddx])
                        cb, sb = np.mean(np.cos(a), axis=1), np.mean(np.sin(a), axis=1)
                    else:
                        cb, sb = np.ones(d.shape[0]), np.zeros(d.shape[0])
                    if gamma_loc:
                        a = t_angle * gamma_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1) # 2-dimensional array (n-1-i) x (loc_m+1)
                        # a = t_angle * np.asarray([gamma_loc_func(ddxk) for ddxk in ddx])
                        cc, sc = np.mean(np.cos(a), axis=1), np.mean(np.sin(a), axis=1)
                    else:
                        cc, sc = np.ones(d.shape[0]), np.zeros(d.shape[0])
                    d = np.asarray([dk.dot(np.array(
                                    [[  cak * cck + sak * sbk * sck,  sak * cbk, - cak * sck + sak * sbk * cck],
                                     [- sak * cck + cak * sbk * sck,  cak * cbk,   sak * sck + cak * sbk * cck],
                                     [                    cbk * sck,      - sbk,                     cbk * cck]]))
                                 for (cak, sak, cbk, sbk, cck, sck, dk) in zip (ca, sa, cb, sb, cc, sc, d)])
                    # d = np.asarray([np.array(
                    #                 [[  cak * cck + sak * sbk * sck,  - sak * cck + cak * sbk * sck,  cbk * sck],
                    #                  [                    sak * cbk,                      cak * cbk,      - sbk],
                    #                  [- cak * sck + sak * sbk * cck,    sak * sck + cak * sbk * cck,  cbk * cck]]).dot(dk)
                    #              for (cak, sak, cbk, sbk, cck, sck, dk) in zip (ca, sa, cb, sb, cc, sc, d)])
                if coord_loc == 1:
                    d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                    # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                elif coord_loc == 2:
                    d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                    # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                elif coord_loc == 3:
                    d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                    d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                    # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                    # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                elif coord_loc == 4:
                    d[:, 2] = np.mean(coord3_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 2]
                    # d[:, 2] = np.asarray([np.mean(coord3_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 2]
                elif coord_loc == 5:
                    d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                    d[:, 2] = np.mean(coord3_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 2]
                    # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                    # d[:, 2] = np.asarray([np.mean(coord3_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 2]
                elif coord_loc == 6:
                    d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                    d[:, 2] = np.mean(coord3_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 2]
                    # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                    # d[:, 2] = np.asarray([np.mean(coord3_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 2]
                elif coord_loc == 7:
                    d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                    d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                    d[:, 2] = np.mean(coord3_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 2]
                    # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                    # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                    # d[:, 2] = np.asarray([np.mean(coord3_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 2]
                elif coord_loc == 8:
                    d = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d.T).T
                    # d = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d.T).T
                elif coord_loc == 9:
                    d[:, (0, 1)] = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (0, 1)].T).T
                    # d[:, (0, 1)] = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (0, 1)].T).T
                elif coord_loc == 10:
                    d[:, (0, 2)] = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (0, 2)].T).T
                    # d[:, (0, 2)] = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (0, 2)].T).T
                elif coord_loc == 11:
                    d[:, (1, 2)] = (np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (1, 2)].T).T
                    # d[:, (1, 2)] = (np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (1, 2)].T).T
                elif coord_loc == 12:
                    d[:, (0, 1)] = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (0, 1)].T).T
                    d[:, 2] = np.mean(coord3_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 2]
                    # d[:, (0, 1)] = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (0, 1)].T).T
                    # d[:, 2] = np.asarray([np.mean(coord3_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 2]
                elif coord_loc == 13:
                    d[:, (0, 2)] = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (0, 2)].T).T
                    d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                    # d[:, (0, 2)] = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (0, 2)].T).T
                    # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                elif coord_loc == 14:
                    d[:, (1, 2)] = (np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (1, 2)].T).T
                    d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                    # d[:, (1, 2)] = (np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (1, 2)].T).T
                    # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                d_abs = np.fabs(d)
                # di: distance to axis i (in new system)
                d0 = np.sqrt(d[:, 1]**2 + d[:, 2]**2)
                d1 = np.sqrt(d[:, 0]**2 + d[:, 2]**2)
                d2 = np.sqrt(d[:, 0]**2 + d[:, 1]**2)
                ind = np.where(np.all((d_abs[:, 0] <= hmax[0], d0 <= tol_dist[0], d0 <= tol_s[0]*d_abs[:, 0]), axis=0))[0]
                if len(ind) > 0:
                    h0.append(d_abs[ind, 0])
                    if w_loc:
                        wf = np.mean(w_factor_loc_func(ddx[ind].reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)
                        # wf = np.asarray([np.mean(w_factor_loc_func(ddxk)) for ddxk in ddx[ind]])
                    g0.append(wf * 0.5*(v[i] - v[i+1+ind])**2)
                ind = np.where(np.all((d_abs[:, 1] <= hmax[1], d1 <= tol_dist[1], d1 <= tol_s[1]*d_abs[:, 1]), axis=0))[0]
                if len(ind) > 0:
                    h1.append(d_abs[ind, 1])
                    if w_loc:
                        wf = np.mean(w_factor_loc_func(ddx[ind].reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)
                        # wf = np.asarray([np.mean(w_factor_loc_func(ddxk)) for ddxk in ddx[ind]])
                    g1.append(wf * 0.5*(v[i] - v[i+1+ind])**2)
                ind = np.where(np.all((d_abs[:, 2] <= hmax[2], d2 <= tol_dist[2], d2 <= tol_s[2]*d_abs[:, 2]), axis=0))[0]
                if len(ind) > 0:
                    h2.append(d_abs[ind, 2])
                    if w_loc:
                        wf = np.mean(w_factor_loc_func(ddx[ind].reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)
                        # wf = np.asarray([np.mean(w_factor_loc_func(ddxk)) for ddxk in ddx[ind]])
                    g2.append(wf * 0.5*(v[i] - v[i+1+ind])**2)
        else:
            for i in range(n-1):
                d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
                if rotate_coord_sys:
                    # Rotate according to new system
                    d = d.dot(mrot)
                if rotation_loc:
                    if alpha_loc:
                        a = t_angle * alpha_loc_func(x[i])[0]
                        ca, sa = np.cos(a), np.sin(a)
                    else:
                        ca, sa = np.ones(d.shape[0]), np.zeros(d.shape[0])
                    if beta_loc:
                        a = t_angle * beta_loc_func(x[i])[0]
                        cb, sb = np.cos(a), np.sin(a)
                    else:
                        cb, sb = np.ones(d.shape[0]), np.zeros(d.shape[0])
                    if gamma_loc:
                        a = t_angle * gamma_loc_func(x[i])[0]
                        cc, sc = np.cos(a), np.sin(a)
                    else:
                        cc, sc = np.ones(d.shape[0]), np.zeros(d.shape[0])
                    d = d.dot(np.array(
                            [[  ca * cc + sa * sb * sc,  sa * cb, - ca * sc + sa * sb * cc],
                             [- sa * cc + ca * sb * sc,  ca * cb,   sa * sc + ca * sb * cc],
                             [                 cb * sc,     - sb,                  cb * cc]]))
                if coord_loc == 1:
                    d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                elif coord_loc == 2:
                    d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                elif coord_loc == 3:
                    d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                    d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                elif coord_loc == 4:
                    d[:, 2] = coord3_factor_loc_func(x[i])[0]*d[:, 2]
                elif coord_loc == 5:
                    d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                    d[:, 2] = coord3_factor_loc_func(x[i])[0]*d[:, 2]
                elif coord_loc == 6:
                    d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                    d[:, 2] = coord3_factor_loc_func(x[i])[0]*d[:, 2]
                elif coord_loc == 7:
                    d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                    d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                    d[:, 2] = coord3_factor_loc_func(x[i])[0]*d[:, 2]
                elif coord_loc == 8:
                    d = coord1_factor_loc_func(x[i])[0]*d
                elif coord_loc == 9:
                    d[:, (0, 1)] = coord1_factor_loc_func(x[i])[0]*d[:, (0, 1)]
                elif coord_loc == 10:
                    d[:, (0, 2)] = coord1_factor_loc_func(x[i])[0]*d[:, (0, 2)]
                elif coord_loc == 11:
                    d[:, (1, 2)] = coord2_factor_loc_func(x[i])[0]*d[:, (1, 2)]
                elif coord_loc == 12:
                    d[:, (0, 1)] = coord1_factor_loc_func(x[i])[0]*d[:, (0, 1)]
                    d[:, 2] = coord3_factor_loc_func(x[i])[0]*d[:, 2]
                elif coord_loc == 13:
                    d[:, (0, 2)] = coord1_factor_loc_func(x[i])[0]*d[:, (0, 2)]
                    d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                elif coord_loc == 14:
                    d[:, (1, 2)] = coord2_factor_loc_func(x[i])[0]*d[:, (1, 2)]
                    d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                d_abs = np.fabs(d)
                # di: distance to axis i (in new system)
                d0 = np.sqrt(d[:, 1]**2 + d[:, 2]**2)
                d1 = np.sqrt(d[:, 0]**2 + d[:, 2]**2)
                d2 = np.sqrt(d[:, 0]**2 + d[:, 1]**2)
                ind = np.where(np.all((d_abs[:, 0] <= hmax[0], d0 <= tol_dist[0], d0 <= tol_s[0]*d_abs[:, 0]), axis=0))[0]
                if len(ind) > 0:
                    h0.append(d_abs[ind, 0])
                    if w_loc:
                        wf = w_factor_loc_func(x[i])[0]
                    g0.append(wf * 0.5*(v[i] - v[i+1+ind])**2)
                ind = np.where(np.all((d_abs[:, 1] <= hmax[1], d1 <= tol_dist[1], d1 <= tol_s[1]*d_abs[:, 1]), axis=0))[0]
                if len(ind) > 0:
                    h1.append(d_abs[ind, 1])
                    if w_loc:
                        wf = w_factor_loc_func(x[i])[0]
                    g1.append(wf * 0.5*(v[i] - v[i+1+ind])**2)
                ind = np.where(np.all((d_abs[:, 2] <= hmax[2], d2 <= tol_dist[2], d2 <= tol_s[2]*d_abs[:, 2]), axis=0))[0]
                if len(ind) > 0:
                    h2.append(d_abs[ind, 2])
                    if w_loc:
                        wf = w_factor_loc_func(x[i])[0]
                    g2.append(wf * 0.5*(v[i] - v[i+1+ind])**2)
    else:
        if rotate_coord_sys:
            # Rotate according to new system
            x = x.dot(mrot)
        for i in range(n-1):
            d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
            d_abs = np.fabs(d)
            # di: distance to axis i (in new system)
            d0 = np.sqrt(d[:, 1]**2 + d[:, 2]**2)
            d1 = np.sqrt(d[:, 0]**2 + d[:, 2]**2)
            d2 = np.sqrt(d[:, 0]**2 + d[:, 1]**2)
            ind = np.where(np.all((d_abs[:, 0] <= hmax[0], d0 <= tol_dist[0], d0 <= tol_s[0]*d_abs[:, 0]), axis=0))[0]
            if len(ind) > 0:
                h0.append(d_abs[ind, 0])
                g0.append(0.5*(v[i] - v[i+1+ind])**2)
            ind = np.where(np.all((d_abs[:, 1] <= hmax[1], d1 <= tol_dist[1], d1 <= tol_s[1]*d_abs[:, 1]), axis=0))[0]
            if len(ind) > 0:
                h1.append(d_abs[ind, 1])
                g1.append(0.5*(v[i] - v[i+1+ind])**2)
            ind = np.where(np.all((d_abs[:, 2] <= hmax[2], d2 <= tol_dist[2], d2 <= tol_s[2]*d_abs[:, 2]), axis=0))[0]
            if len(ind) > 0:
                h2.append(d_abs[ind, 2])
                g2.append(0.5*(v[i] - v[i+1+ind])**2)

    npair0 = len(h0)
    if npair0:
        h0 = np.hstack(h0)
        g0 = np.hstack(g0)
    npair1 = len(h1)
    if npair1:
        h1 = np.hstack(h1)
        g1 = np.hstack(g1)
    npair2 = len(h2)
    if npair2:
        h2 = np.hstack(h2)
        g2 = np.hstack(g2)

    if make_plot:
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(2,2,1, projection='3d')
        ax2 = fig.add_subplot(2,2,2)
        ax3 = fig.add_subplot(2,2,3)
        ax4 = fig.add_subplot(2,2,4)

        # Plot system Oxzy and Ox'y'z'
        # This:
        ax1.plot([0,1], [0,0], [0,0], color='k')
        ax1.plot([0,0], [0,1], [0,0], color='k')
        ax1.plot([0,0], [0,0], [0,1], color='k')
        ax1.plot([0, mrot[0,0]], [0, mrot[1,0]], [0, mrot[2,0]], color=color0, label="x'''")
        ax1.plot([0, mrot[0,1]], [0, mrot[1,1]], [0, mrot[2,1]], color=color1, label="y'''")
        ax1.plot([0, mrot[0,2]], [0, mrot[1,2]], [0, mrot[2,2]], color=color2, label="z'''")
        ax1.set_xticks([0,1])
        ax1.set_yticks([0,1])
        ax1.set_zticks([0,1])
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.legend()
        plt.sca(ax1)
        plt.title("System Ox'''y'''z'''")

        plt.sca(ax2)
        # Plot variogram cloud along x'''
        plot_variogramCloud1D(h0, g0, c=color0, **kwargs)
        plt.title(f"along x''' ({npair0} pts)")

        plt.sca(ax3)
        # Plot variogram cloud along y'''
        plot_variogramCloud1D(h1, g1, c=color1, **kwargs)
        plt.title(f"along y''' ({npair1} pts)")

        plt.sca(ax4)
        # Plot variogram cloud along z'''
        plot_variogramCloud1D(h2, g2, c=color2, **kwargs)
        plt.title(f"along z''' ({npair2} pts)")

        plt.suptitle(f'Vario cloud: alpha={alpha}deg. beta={beta}deg. gamma={gamma}deg.\ntol_dist={tol_dist} / tol_angle={tol_angle}deg.')
        # plt.show()

    return (h0, g0, npair0), (h1, g1, npair1), (h2, g2, npair2)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def variogramExp3D(
        x, v,
        alpha=0.0,
        beta=0.0,
        gamma=0.0,
        tol_dist=None,
        tol_angle=None,
        hmax=None,
        alpha_loc_func=None,
        beta_loc_func=None,
        gamma_loc_func=None,
        w_factor_loc_func=None,
        coord1_factor_loc_func=None,
        coord2_factor_loc_func=None,
        coord3_factor_loc_func=None,
        loc_m=1,
        ncla=(10, 10, 10),
        cla_center=(None, None, None),
        cla_length=(None, None, None),
        variogramCloud=None,
        make_plot=True,
        color0='red',
        color1='green',
        color2='blue',
        figsize=None, 
        logger=None,
        **kwargs):
    """
    Computes the three experimental directional variograms (wrt. main axes) for a data set in 3D.

    For the experimental variogram along the 1st (resp. 2nd, 3rd) main axis, the
    mean point in each class is retrieved from the 1st (resp. 2nd, 3rd) variogram
    cloud (returned by the function `variogramCloud3D`); along the 1st axis (j=0)
    (resp. 2nd axis (j=1), 3rd axis (j=2)), the i-th class is determined by its
    center `cla_center[j][i]` and its length `cla_length[j][i]`, and corresponds
    to the interval

        `]cla_center[j][i]-cla_length[j][i]/2, cla_center[j][i]+cla_length[j][i]/2]`

    along h1 (resp. h2, h3) (lag) axis (abscissa).

    Parameters
    ----------
    x : 2D array of floats of shape (n, 3)
        data points locations, with n the number of data points, each row of `x`
        is the coordinatates of one data point

    v : 1D array of floats of shape (n,)
        data points values, with n the number of data points, `v[i]` is the data
        value at location `x[i]`

    alpha : float, default: 0.0
        azimuth angle in degrees (see :class:`CovModel3D`)

    beta : float, default: 0.0
        dip angle in degrees (see :class:`CovModel3D`)

    gamma : float, default: 0.0
        plunge angle in degrees (see :class:`CovModel3D`)

    tol_dist : sequence of 3 floats, or float, optional
        let `tol_dist` = (tol_dist1, tol_dist2, tol_dist3); tol_dist1 (resp.
        tol_dist2, tol_dist3) is the maximal distance to the 1st (resp. 2nd, 3rd)
        main axis for the lag (vector between two data points), such that the pair
        is integrated in the variogram cloud along the 1st (resp. 2nd, 3rd) main
        axis;
        note: if `tol_dist` is specified as a float or `None` (default), the
        entry is duplicated; if tol_dist1 (resp. tol_dist2, tol_dist3) is `None`,
        then tol_dist1 (resp. tol_dist2, tol_dist3) is set to 10% of h1max (resp.
        h2max, h3max) if h1max (resp. h2max, h3max) is finite, and set to 10.0
        otherwise: see parameter `hmax` for the definition of h1max, h2max and
        h3max

    tol_angle : sequence of 3 floats, or float, optional
        let `tol_angle` = (tol_angle1, tol_angle2, tol_angl3); tol_angle1 (resp.
        tol_angle2, tol_angle3) is the maximal angle in degrees between the lag
        (vector between two data points) and the 1st (resp. 2nd, 3rd) main axis,
        such that the pair is integrated in the variogram cloud along the 1st
        (resp. 2nd, 3rd) main axis;
        note: if `tol_angle` is specified as a float, it is duplicated;
        by default (`None`): `tol_angle` is set to 45.0

    hmax : sequence of 3 floats, or float, optional
        let `hmax` = (h1max, h2max, h3max); h1max (resp. h2max, h3max) is the
        maximal distance between a pair of data points along the 1st (resp. 2nd,
        3rd) main axis, such that the pair is integrated in the variogram cloud
        along the 1st (resp. 2nd, 3rd) main axis;
        note: if `hmax` is specified as a float or `None` (default), the entry is
        duplicated, and `None`, `numpy.nan` are converted to `numpy.inf` (no
        restriction)

    alpha_loc_func : function (`callable`), optional
        function returning azimuth angle, defining the main axes, as function of
        a given location in 3D, i.e. the main axes are defined locally

    beta_loc_func : function (`callable`), optional
        function returning dip angle, defining the main axes, as function of
        a given location in 3D, i.e. the main axes are defined locally

    gamma_loc_func : function (`callable`), optional
        function returning plunge angle, defining the main axes, as function of
        a given location in 3D, i.e. the main axes are defined locally

    w_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "weight" as function of a given
        location in 3D, i.e. "g" values (i.e. ordinate axis component in the two
        variograms) are multiplied

    coord1_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" along the 1st (local) main
        axis as function of a given location in 3D, i.e. "h1" values (i.e.
        abscissa axis component in the 1st variogram) are multiplied
        (the condition wrt h1max, see `hmax`, is checked after)

    coord2_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" along the 2nd (local) main
        axis as function of a given location in 3D, i.e. "h2" values (i.e.
        abscissa axis component in the 2nd variogram) are multiplied
        (the condition wrt h2max, see `hmax`, is checked after)

    coord3_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" along the 3rd (local) main
        axis as function of a given location in 3D, i.e. "h3" values (i.e.
        abscissa axis component in the 3rd variogram) are multiplied
        (the condition wrt h3max, see `hmax`, is checked after)

    loc_m : int, default: 1
        integer (greater than or equal to 0) defining how the function(s)
        `*_loc_func` (above) are evaluated for a pair of two locations x1, x2
        (data point locations):

        - if `loc_m>0` the segment from x1 to x2 is divided in `loc_m` intervals \
        of same length and the mean of the evaluations of the function at the \
        (`loc_m` + 1) interval bounds is computed
        - if `loc_m=0`, the evaluation at x1 is considered

    ncla : sequence of 3 ints, default: (10, 10, 10)
        number of classes along each main axis, the parameter `ncla[j]` is used
        if `cla_center[j]=None`, in that situation `ncla[j]` classes are
        considered and the class centers are set to

        - `cla_center[j][i] = (i+0.5)*l, i=0,...,ncla[j]-1`

        with l = H / ncla[j], H being the max of the distance, along the
        corresponding main axis, between two points of the considered pairs
        (in the variogram cloud along the corresponding main axis);
        if `cla_center[j]` is specified (not `None`), the number of classes
        (`ncla[j]`) is set to the length of the sequence `cla_center[j]`
        (ignoring the value passed as argument)

    cla_center : sequence of length 3
        cla_center[j] : 1D array-like of floats, or `None` (default)
            center of each class (in abscissa) in the experimental variogram
            along the 1st (j=0) (resp. 2nd (j=1),) main axis; by default (`None`):
            `cla_center[j]` is defined from `ncla[j]` (see above)

    cla_length : sequence of length 3
        cla_length[j] : 1D array-like of floats, or float, or `None`
            length of each class centered at `cla_center[j]` (in abscissa) in the
            experimental variogram along the 1st (j=0) (resp. 2nd (j=1),
            3rd (j=2)) main axis:

            - if `cla_length[j]` is a sequence, it should be of length `ncla[j]`
            - if `cla_length[j]` is a float, the value is repeated `ncla[j]` times
            - if `cla_length[j]=None` (default), the minimum of difference between \
            two sucessive class centers along the corresponding main axis (`np.inf` \
            if one class) is used and repeated `ncla[j]` times

    variogramCloud : sequence of three 3-tuple, optional
        `variogramCloud` = ((h0, g0, npair0), (h1, g1, npair1), (h2, g2, npair2))
        is variogram clouds (already computed and returned by the function
        `variogramCloud3D` (npair0, npair1, npair2 not used)) along the three
        main axes;
        in this case, `x`, `v`, `alpha`, `tol_dist`, `tol_angle`, `hmax`,
        `alpha_loc_func`, `w_factor_loc_func`, `coord1_factor_loc_func`,
        `coord2_factor_loc_func`, `coord3_factor_loc_func`, `loc_m` are not used

        By default (`None`): the variogram clouds are computed by using the
        function `variogramCloud3D`

    make_plot : bool, default: True
        indicates if the experimental variograms are plotted (in a new "2x3"
        figure)

    color0 : color, default: 'red'
        color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
        the plot of the experimental variogram along the 1st main axis
        (if `make_plot=True`)

    color1 : color, default: 'green'
        color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
        the plot of the experimental variogram along the 2nd main axis
        (if `make_plot=True`)

    color2 : color, default: 'blue'
        color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
        the plot of the variogram cloud along the 3rd main axis
        (if `make_plot=True`)

    figsize : 2-tuple, optional
        size of the new "2x3" figure (if `make_plot=True`)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs : dict
        keyword arguments passed to the funtion `plot_variogramExp1D`
        (if `make_plot=True`)

    Returns
    -------
    (hexp0, gexp0, cexp0) : 3-tuple
        hexp0, gexp0 : 1D arrays of floats of same length
            coordinates of the points of the experimental variogram along the
            1st main axis

        cexp0 : 1D array of ints
            array of same length as `hexp0`, `gexp0`, number of points (pairs of
            data points considered) in each class in the variogram cloud along
            the 1st main axis

    (hexp1, gexp1, cexp1) : 3-tuple
        hexp1, gexp1 : 1D arrays of floats of same length
            coordinates of the points of the experimental variogram along the
            2nd main axis

        cexp1 : 1D array of ints
            array of same length as `hexp1`, `gexp1`, number of points (pairs of
            data points considered) in each class in the variogram cloud along
            the 2nd main axis

    (hexp2, gexp2, cexp2) : 3-tuple
        hexp2, gexp2 : 1D arrays of floats of same length
            coordinates of the points of the experimental variogram along the
            3rd main axis

        cexp2 : 1D array of ints
            array of same length as `hexp2`, `gexp2`, number of points (pairs of
            data points considered) in each class in the variogram cloud along
            the 3rd main axis
    """
    fname = 'variogramExp3D'

    # Compute variogram clouds if needed
    if variogramCloud is None:
        try:
            vc = variogramCloud3D(
                    x, v, alpha=alpha, beta=beta, gamma=gamma, tol_dist=tol_dist, tol_angle=tol_angle, hmax=hmax,
                    alpha_loc_func=alpha_loc_func, 
                    beta_loc_func=beta_loc_func, 
                    gamma_loc_func=gamma_loc_func,
                    w_factor_loc_func=w_factor_loc_func,
                    coord1_factor_loc_func=coord1_factor_loc_func, 
                    coord2_factor_loc_func=coord2_factor_loc_func, 
                    coord3_factor_loc_func=coord3_factor_loc_func, 
                    loc_m=loc_m,
                    make_plot=False,
                    logger=logger)
        except Exception as exc:
            err_msg = f'{fname}: cannot compute variogram cloud (3D)'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg) from exc

    else:
        vc = variogramCloud
    # -> vc[0] = (h0, g0, npair0) and vc[1] = (h1, g1, npair1) and vc[2] = (h2, g2, npair2)

    # Compute variogram experimental in each direction (using function variogramExp1D)
    ve = [None, None, None]
    for j in (0, 1, 2):
        try:
            ve[j] = variogramExp1D(
                        None, None, 
                        hmax=None, w_factor_loc_func=None, coord_factor_loc_func=None, loc_m=loc_m,
                        ncla=ncla[j], cla_center=cla_center[j], cla_length=cla_length[j], variogramCloud=vc[j], 
                        make_plot=False, logger=logger)
        except Exception as exc:
            err_msg = f'{fname}: cannot compute experimental variogram in one direction'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg) from exc

    (hexp0, gexp0, cexp0), (hexp1, gexp1, cexp1), (hexp2, gexp2, cexp2) = ve

    if make_plot:
        # Rotation matrix
        a = alpha * np.pi/180.
        b = beta * np.pi/180.
        c = gamma * np.pi/180.
        ca, sa = np.cos(a), np.sin(a)
        cb, sb = np.cos(b), np.sin(b)
        cc, sc = np.cos(c), np.sin(c)

        mrot = np.array([[  ca * cc + sa * sb * sc,  sa * cb,  - ca * sc + sa * sb * cc],
                         [- sa * cc + ca * sb * sc,  ca * cb,    sa * sc + ca * sb * cc],
                         [                 cb * sc,     - sb,                   cb * cc]])

        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(2,3,1, projection='3d')
        # subplot(2,3,2) is empty
        ax2 = fig.add_subplot(2,3,3)
        ax3 = fig.add_subplot(2,3,4)
        ax4 = fig.add_subplot(2,3,5)
        ax5 = fig.add_subplot(2,3,6)

        # Plot system Oxzy and Ox'y'z'
        # This:
        ax1.plot([0,1], [0,0], [0,0], color='k')
        ax1.plot([0,0], [0,1], [0,0], color='k')
        ax1.plot([0,0], [0,0], [0,1], color='k')
        ax1.plot([0, mrot[0,0]], [0, mrot[1,0]], [0, mrot[2,0]], color=color0, label="x'''")
        ax1.plot([0, mrot[0,1]], [0, mrot[1,1]], [0, mrot[2,1]], color=color1, label="y'''")
        ax1.plot([0, mrot[0,2]], [0, mrot[1,2]], [0, mrot[2,2]], color=color2, label="z'''")
        ax1.set_xticks([0,1])
        ax1.set_yticks([0,1])
        ax1.set_zticks([0,1])
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.legend()
        plt.sca(ax1)
        plt.title("System Ox'''y'''z'''")

        plt.sca(ax2)
        # Plot variogram exp along x''', along y''' and along z'''
        plot_variogramExp1D(hexp0, gexp0, cexp0, show_count=False, grid=True, c=color0, alpha=0.5, label="along x'''")
        plot_variogramExp1D(hexp1, gexp1, cexp1, show_count=False, grid=True, c=color1, alpha=0.5, label="along y'''")
        plot_variogramExp1D(hexp2, gexp2, cexp2, show_count=False, grid=True, c=color2, alpha=0.5, label="along z'''")
        plt.legend()

        plt.sca(ax3)
        # Plot variogram exp along x'''
        plot_variogramExp1D(hexp0, gexp0, cexp0, c=color0, **kwargs)
        plt.title("along x'''")

        plt.sca(ax4)
        # Plot variogram exp along y'''
        plot_variogramExp1D(hexp1, gexp1, cexp1, c=color1, **kwargs)
        plt.title("along y'''")

        plt.sca(ax5)
        # Plot variogram exp along z'''
        plot_variogramExp1D(hexp2, gexp2, cexp2, c=color2, **kwargs)
        plt.title("along z'''")

        plt.suptitle(f'Vario exp.: alpha={alpha}deg. beta={beta}deg. gamma={gamma}deg.')
        # plt.suptitle(f'Vario exp.: alpha={alpha}deg. beta={beta}deg. gamma={gamma}deg.\ntol_dist={tol_dist} / tol_angle={tol_angle}deg.')
        # plt.show()

    return (hexp0, gexp0, cexp0), (hexp1, gexp1, cexp1), (hexp2, gexp2, cexp2)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def variogramCloud3D_omni_wrt_2_first_axes(
        x, v,
        alpha=0.0,
        beta=0.0,
        gamma=0.0,
        tol_dist=None,
        tol_angle=None,
        hmax=None,
        alpha_loc_func=None,
        beta_loc_func=None,
        gamma_loc_func=None,
        w_factor_loc_func=None,
        coord1_factor_loc_func=None,
        coord2_factor_loc_func=None,
        coord3_factor_loc_func=None,
        loc_m=1,
        make_plot=True,
        color01='orange',
        color2='blue',
        figsize=None,
        logger=None,
        **kwargs):
    """
    Computes two variogram clouds for a data set in 3D.

    The computed variogram clouds are:

    - the omni-directional variogram cloud wrt. the first 2 axes (i.e with any direction \
    parallel to the plane spanned by the first 2 axes)
    - the directional variogram cloud wrt. the 3rd main axis.

    From the pair of the i-th and j-th data points (i not equal to j), let

    .. math::
        \\begin{array}{rcl}
            h(i, j) &=& x_i-x_j \\\\[2mm]
            g(i, j) &=& \\frac{1}{2}(v_i - v_j)^2
        \\end{array}

    where :math:`x_i` and :math:`x_j` are the coordinates of the i-th and j-th
    data points and :math:`v_i` and :math:`v_j` the values at these points
    (:math:`v_i=Z(x_i)`, where :math:`Z` is the considered variable).
    The lag vector h(i, j) is expressed along the three orthogonal main axes,
    h(i, j) = (h1(i, j), h2(i, j), h3(i, j)). Let
    `tol_dist` = (tol_dist12, tol_dist3),
    `tol_angle` = (tol_angle12, tol_angle3), and
    `hmax` = (h12max, h3max) (see parameters below); if distance from h(i, j)
    to the plane spanned by the first two main axes (resp. to the 3rd main axis)
    does not exceed tol_dist12 (resp. toldist3), and if the angle between the
    lag h(i, j) and the the plane spanned by the first two main axes (resp. the
    3rd main axis) does not exceed tol_angle12 (resp. tol_angle3),
    and if the distance :math:`\\sqrt{h1(i, j)^2+h2(i, j)^2}` in the plane spanned
    by the first two main axes (resp. the distance \\|h3(i,j)\\| along the 3rd main
    axis) does not exceed h12max (resp. h3max), then, the point
    (:math:`\\sqrt{h1(i, j)^2+h2(i, j)^2}`, g(i, j)) (resp. (\\|h3(i, j)\\|, g(i, j)))
    is integrated in the omni-directional variogram cloud wrt. the first two main axes
    (resp. the directional variogram cloud wrt. the 3rd main axis).

    Moreover, the parameters `alpha_loc_func`, `beta_loc_func`, `gamma_loc_func`
    allow to account for main axes locally varying in space, and the parameters
    `w_factor_loc_func` and `coord1_factor_loc_func`, `coord2_factor_loc_func`,
    `coord2_factor_loc_func` allow to account for variogram locally varying in
    space with respect to weight and ranges along each main axis resp., by
    multiplying "g", "h1", "h2", "h3" values resp.

    Parameters
    ----------
    x : 2D array of floats of shape (n, 3)
        data points locations, with n the number of data points, each row of `x`
        is the coordinatates of one data point

    v : 1D array of floats of shape (n,)
        data points values, with n the number of data points, `v[i]` is the data
        value at location `x[i]`

    alpha : float, default: 0.0
        azimuth angle in degrees (see :class:`CovModel3D`)

    beta : float, default: 0.0
        dip angle in degrees (see :class:`CovModel3D`)

    gamma : float, default: 0.0
        plunge angle in degrees (see :class:`CovModel3D`)

    tol_dist : sequence of 2 floats, or float, optional
        let `tol_dist` = (tol_dist12, tol_dist3); tol_dist12 (resp. tol_dist3) is
        the maximal distance to the plane spanned by the first two main axes (resp.
        to the 3rd main axis) for the lag (vector between two data points), such
        that the pair is integrated in
        the omni-directional variogram cloud wrt. the first two main axes
        (resp. the directional variogram cloud wrt. the 3rd main axis);
        note: if `tol_dist` is specified as a float or `None` (default), the
        entry is duplicated; if tol_dist12 (resp. tol_dist3) is `None`,
        then tol_dist12 (resp. tol_dist3) is set to 10% of h12max (resp.
        h3max) if h12max (resp. h3max) is finite, and set to 10.0
        otherwise: see parameter `hmax` for the definition of h12max and h3max

    tol_angle : sequence of 2 floats, or float, optional
        let `tol_angle` = (tol_angle12, tol_angl3); tol_angle12 (resp. tol_angle3)
        is the maximal angle in degrees between the lag (vector between two data
        points) and the plane spanned by the first two main axes (resp.
        the 3rd main axis), such that the pair is integrated in
        the omni-directional variogram cloud wrt. the first two main axes
        (resp. the directional variogram cloud wrt. the 3rd main axis);
        note: if `tol_angle` is specified as a float, it is duplicated;
        by default (`None`): `tol_angle` is set to 45.0

    hmax : sequence of 2 floats, or float, optional
        let `hmax` = (h12max, h3max); h12max (resp. h3max) is the
        maximal distance between a pair of data points in the plane
        spanned by the first two main axes (resp. along the 3rd) main axis,
        such that the pair in
        the omni-directional variogram cloud wrt. the first two main axes
        (resp. the directional variogram cloud wrt. the 3rd main axis);
        note: if `hmax` is specified as a float or `None` (default), the entry is
        duplicated, and `None`, `numpy.nan` are converted to `numpy.inf` (no
        restriction)

    alpha_loc_func : function (`callable`), optional
        function returning azimuth angle, defining the main axes, as function of
        a given location in 3D, i.e. the main axes are defined locally

    beta_loc_func : function (`callable`), optional
        function returning dip angle, defining the main axes, as function of
        a given location in 3D, i.e. the main axes are defined locally

    gamma_loc_func : function (`callable`), optional
        function returning plunge angle, defining the main axes, as function of
        a given location in 3D, i.e. the main axes are defined locally

    w_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "weight" as function of a given
        location in 3D, i.e. "g" values (i.e. ordinate axis component in the two
        variograms) are multiplied

    coord1_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" along the 1st (local) main
        axis as function of a given location in 3D, i.e. "h1" values (i.e.
        abscissa axis component in the 1st variogram) are multiplied
        (the condition wrt h1max, see `hmax`, is checked after)

    coord2_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" along the 2nd (local) main
        axis as function of a given location in 3D, i.e. "h2" values (i.e.
        abscissa axis component in the 2nd variogram) are multiplied
        (the condition wrt h2max, see `hmax`, is checked after)

    coord3_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" along the 3rd (local) main
        axis as function of a given location in 3D, i.e. "h3" values (i.e.
        abscissa axis component in the 3rd variogram) are multiplied
        (the condition wrt h3max, see `hmax`, is checked after)

    loc_m : int, default: 1
        integer (greater than or equal to 0) defining how the function(s)
        `*_loc_func` (above) are evaluated for a pair of two locations x1, x2
        (data point locations):

        - if `loc_m>0` the segment from x1 to x2 is divided in `loc_m` intervals \
        of same length and the mean of the evaluations of the function at the \
        (`loc_m` + 1) interval bounds is computed
        - if `loc_m=0`, the evaluation at x1 is considered

    make_plot : bool, default: True
        indicates if the variogram clouds are plotted (in a new "2x2" figure)

    color01 : color, default: 'orange'
        color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
        the plot of the omni-directional variogram cloud wrt. the first 2 axes
        (if `make_plot=True`)

    color2 : color, default: 'blue'
        color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
        the plot of the variogram cloud along the 3rd main axis
        (if `make_plot=True`)

    figsize : 2-tuple, optional
        size of the new "2x2" figure (if `make_plot=True`)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs : dict
        keyword arguments passed to the funtion `plot_variogramCloud1D`
        (if `make_plot=True`)

    Returns
    -------
    (h01, g01, npair01) : 3-tuple
        h01, g01 : 1D arrays of floats of same length
            coordinates of the points in the omni-directional variogram cloud
            wrt. the first 2 axes (see above)

        npair01 : int
            number of points (pairs of data points considered) in the
            omni-directional variogram cloud wrt. the first 2 axes

    (h2, g2, npair2) : 3-tuple
        h2, g2 : 1D arrays of floats of same length
            coordinates of the points in the variogram cloud along 3rd main axis
            (see above)

        npair2 : int
            number of points (pairs of data points considered) in the variogram
            cloud along the 3rd main axis
    """
    fname = 'variogramCloud3D_omni_wrt_2_first_axes'

    # Number of data points
    n = x.shape[0]

    # Check length of v
    if len(v) != n:
        err_msg = f'{fname}: length of `v` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # Set hmax as an array of shape (2,)
    hmax = np.atleast_1d(hmax).astype('float').reshape(-1) # None is converted to nan
    hmax[np.isnan(hmax)] = np.inf # convert nan to inf
    if hmax.size == 1:
        hmax = np.array([hmax[0], hmax[0]])
    elif hmax.size != 2:
        err_msg = f'{fname}: size of `hmax` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # Set tol_dist as an array of shape (2,)
    tol_dist = np.atleast_1d(tol_dist).astype('float').reshape(-1) # None is converted to nan
    if tol_dist.size == 1:
        tol_dist = np.array([tol_dist[0], tol_dist[0]])
    elif tol_dist.size != 2:
        err_msg = f'{fname}: size of `tol_dist` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    for i in range(2):
        if np.isnan(tol_dist[i]):
            if np.isinf(hmax[i]):
                tol_dist[i] = 10.0
            else:
                tol_dist[i] = 0.1 * hmax[i]

    # Set tol_angle as an array of shape (2,)
    tol_angle = np.atleast_1d(tol_angle).astype('float').reshape(-1) # None is converted to nan
    if tol_angle.size == 1:
        tol_angle = np.array([tol_angle[0], tol_angle[0]])
    elif tol_angle.size != 2:
        err_msg = f'{fname}: size of `tol_angle` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    tol_angle[np.isnan(tol_angle)] = 45.0

    if alpha != 0.0 or beta != 0.0 or gamma != 0.0:
        rotate_coord_sys = True
        # Rotation matrix
        a = alpha * np.pi/180.
        b = beta * np.pi/180.
        c = gamma * np.pi/180.
        ca, sa = np.cos(a), np.sin(a)
        cb, sb = np.cos(b), np.sin(b)
        cc, sc = np.cos(c), np.sin(c)
        mrot = np.array([[  ca * cc + sa * sb * sc,  sa * cb,  - ca * sc + sa * sb * cc],
                         [- sa * cc + ca * sb * sc,  ca * cb,    sa * sc + ca * sb * cc],
                         [                 cb * sc,     - sb,                  cb * cc ]])
    else:
        rotate_coord_sys = False
        mrot = np.eye(3)

    # Set types of local transformations
    #    alpha_loc: True / False: is local angle alpha used ?
    #    beta_loc : True / False: is local angle beta used ?
    #    gamma_loc: True / False: is local angle gamma used ?
    #    rotation_loc: True / False: is local rotation used ?
    #    w_loc:     True / False: is local w (weight / sill) used ?
    #    coord_loc: integer
    #               0: no transformation
    #               1: transformation for 1st coordinate only
    #               2: transformation for 2nd coordinate only
    #               3: distinct transformations for 1st and 2nd coordinates, no transformation for 3rd coordinate
    #               4: transformation for 3rd coordinate only
    #               5: distinct transformations for 1st and 3rd coordinates, no transformation for 2nd coordinate
    #               6: distinct transformations for 2nd and 3rd coordinates, no transformation for 1st coordinate
    #               7: distinct transformations for 1st, 2nd and 3rd coordinates
    #               8: same transformation for 1st, 2nd and 3rd coordinates
    #               9: same transformation for 1st and 2nd coordinates, no transformation for 3rd coordinate
    #              10: same transformation for 1st and 3rd coordinates, no transformation for 2nd coordinate
    #              11: same transformation for 2nd and 3rd coordinates, no transformation for 1st coordinate
    #              12: same transformation for 1st and 2nd coordinates, distinct transformation for 3rd coordinate
    #              13: same transformation for 1st and 3rd coordinates, distinct transformation for 2nd coordinate
    #              14: same transformation for 2nd and 3rd coordinates, distinct transformation for 1st coordinate
    alpha_loc = False
    beta_loc = False
    gamma_loc = False
    rotation_loc = False
    w_loc = False
    coord_loc = 0
    if alpha_loc_func is not None:
        alpha_loc = True
        rotation_loc = True
    if beta_loc_func is not None:
        beta_loc = True
        rotation_loc = True
    if gamma_loc_func is not None:
        gamma_loc = True
        rotation_loc = True

    if rotation_loc:
        # factor to transform angle in degree into radian
        t_angle = np.pi/180.0

    if w_factor_loc_func is not None:
        w_loc = True

    if coord1_factor_loc_func is not None:
        coord_loc = coord_loc + 1
    if coord2_factor_loc_func is not None:
        coord_loc = coord_loc + 2
    if coord3_factor_loc_func is not None:
        coord_loc = coord_loc + 4
    if coord_loc == 3:
        if coord1_factor_loc_func == coord2_factor_loc_func:
            coord_loc = 9
    elif coord_loc == 5:
        if coord1_factor_loc_func == coord3_factor_loc_func:
            coord_loc = 10
    elif coord_loc == 6:
        if coord2_factor_loc_func == coord3_factor_loc_func:
            coord_loc = 11
    elif coord_loc == 7:
        if coord1_factor_loc_func == coord2_factor_loc_func and coord1_factor_loc_func == coord3_factor_loc_func:
            coord_loc = 8
        elif coord1_factor_loc_func == coord2_factor_loc_func:
            coord_loc = 12
        elif coord1_factor_loc_func == coord3_factor_loc_func:
            coord_loc = 13
        elif coord2_factor_loc_func == coord3_factor_loc_func:
            coord_loc = 14

    if alpha_loc or beta_loc or gamma_loc or w_loc or coord_loc > 0:
        transform_flag = True
    else:
        transform_flag = False

    # Tolerance for slope compute from tol_angle
    tol_s = np.tan(tol_angle*np.pi/180)

    # Compute variogram clouds
    h01, g01, h2, g2 = [], [], [], []
    if transform_flag:
        wf = 1.0 # default weight factor
        if loc_m > 0:
            for i in range(n-1):
                d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
                dx = d/loc_m
                ddx = np.asarray([x[i]+np.outer(np.arange(loc_m+1), dxk) for dxk in dx]) # 3-dimensional array (n-1-i) x (loc_m+1) x dim
                if rotate_coord_sys:
                    # Rotate according to new system
                    d = d.dot(mrot)
                if rotation_loc:
                    if alpha_loc:
                        a = t_angle * alpha_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1) # 2-dimensional array (n-1-i) x (loc_m+1)
                        # a = t_angle * np.asarray([alpha_loc_func(ddxk) for ddxk in ddx])
                        ca, sa = np.mean(np.cos(a), axis=1), np.mean(np.sin(a), axis=1)
                    else:
                        ca, sa = np.ones(d.shape[0]), np.zeros(d.shape[0])
                    if beta_loc:
                        a = t_angle * beta_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1) # 2-dimensional array (n-1-i) x (loc_m+1)
                        # a = t_angle * np.asarray([beta_loc_func(ddxk) for ddxk in ddx])
                        cb, sb = np.mean(np.cos(a), axis=1), np.mean(np.sin(a), axis=1)
                    else:
                        cb, sb = np.ones(d.shape[0]), np.zeros(d.shape[0])
                    if gamma_loc:
                        a = t_angle * gamma_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1) # 2-dimensional array (n-1-i) x (loc_m+1)
                        # a = t_angle * np.asarray([gamma_loc_func(ddxk) for ddxk in ddx])
                        cc, sc = np.mean(np.cos(a), axis=1), np.mean(np.sin(a), axis=1)
                    else:
                        cc, sc = np.ones(d.shape[0]), np.zeros(d.shape[0])
                    d = np.asarray([dk.dot(np.array(
                                    [[  cak * cck + sak * sbk * sck,  sak * cbk, - cak * sck + sak * sbk * cck],
                                     [- sak * cck + cak * sbk * sck,  cak * cbk,   sak * sck + cak * sbk * cck],
                                     [                    cbk * sck,      - sbk,                     cbk * cck]]))
                                 for (cak, sak, cbk, sbk, cck, sck, dk) in zip (ca, sa, cb, sb, cc, sc, d)])
                    # d = np.asarray([np.array(
                    #                 [[  cak * cck + sak * sbk * sck,  - sak * cck + cak * sbk * sck,  cbk * sck],
                    #                  [                    sak * cbk,                      cak * cbk,      - sbk],
                    #                  [- cak * sck + sak * sbk * cck,    sak * sck + cak * sbk * cck,  cbk * cck]]).dot(dk)
                    #              for (cak, sak, cbk, sbk, cck, sck, dk) in zip (ca, sa, cb, sb, cc, sc, d)])
                if coord_loc == 1:
                    d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                    # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                elif coord_loc == 2:
                    d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                    # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                elif coord_loc == 3:
                    d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                    d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                    # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                    # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                elif coord_loc == 4:
                    d[:, 2] = np.mean(coord3_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 2]
                    # d[:, 2] = np.asarray([np.mean(coord3_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 2]
                elif coord_loc == 5:
                    d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                    d[:, 2] = np.mean(coord3_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 2]
                    # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                    # d[:, 2] = np.asarray([np.mean(coord3_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 2]
                elif coord_loc == 6:
                    d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                    d[:, 2] = np.mean(coord3_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 2]
                    # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                    # d[:, 2] = np.asarray([np.mean(coord3_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 2]
                elif coord_loc == 7:
                    d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                    d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                    d[:, 2] = np.mean(coord3_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 2]
                    # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                    # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                    # d[:, 2] = np.asarray([np.mean(coord3_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 2]
                elif coord_loc == 8:
                    d = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d.T).T
                    # d = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d.T).T
                elif coord_loc == 9:
                    d[:, (0, 1)] = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (0, 1)].T).T
                    # d[:, (0, 1)] = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (0, 1)].T).T
                elif coord_loc == 10:
                    d[:, (0, 2)] = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (0, 2)].T).T
                    # d[:, (0, 2)] = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (0, 2)].T).T
                elif coord_loc == 11:
                    d[:, (1, 2)] = (np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (1, 2)].T).T
                    # d[:, (1, 2)] = (np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (1, 2)].T).T
                elif coord_loc == 12:
                    d[:, (0, 1)] = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (0, 1)].T).T
                    d[:, 2] = np.mean(coord3_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 2]
                    # d[:, (0, 1)] = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (0, 1)].T).T
                    # d[:, 2] = np.asarray([np.mean(coord3_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 2]
                elif coord_loc == 13:
                    d[:, (0, 2)] = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (0, 2)].T).T
                    d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                    # d[:, (0, 2)] = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (0, 2)].T).T
                    # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                elif coord_loc == 14:
                    d[:, (1, 2)] = (np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (1, 2)].T).T
                    d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                    # d[:, (1, 2)] = (np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (1, 2)].T).T
                    # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                d_abs = np.fabs(d)
                # d01: distance to plane spanned by axes 0 and 1 (in new system)
                # d2: distance to axis 2 (in new system)
                d01 = d_abs[:, 2]
                d2 = np.sqrt(d[:, 0]**2 + d[:, 1]**2)
                ind = np.where(np.all((d2 <= hmax[0], d01 <= tol_dist[0], d01 <= tol_s[0]*d2), axis=0))[0]
                if len(ind) > 0:
                    h01.append(d2[ind])
                    if w_loc:
                        wf = np.mean(w_factor_loc_func(ddx[ind].reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)
                        # wf = np.asarray([np.mean(w_factor_loc_func(ddxk)) for ddxk in ddx[ind]])
                    g01.append(wf * 0.5*(v[i] - v[i+1+ind])**2)
                ind = np.where(np.all((d01 <= hmax[1], d2 <= tol_dist[1], d2 <= tol_s[1]*d01), axis=0))[0]
                if len(ind) > 0:
                    h2.append(d01[ind])
                    if w_loc:
                        wf = np.mean(w_factor_loc_func(ddx[ind].reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)
                        # wf = np.asarray([np.mean(w_factor_loc_func(ddxk)) for ddxk in ddx[ind]])
                    g2.append(wf * 0.5*(v[i] - v[i+1+ind])**2)
        else:
            for i in range(n-1):
                d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
                if rotate_coord_sys:
                    # Rotate according to new system
                    d = d.dot(mrot)
                if rotation_loc:
                    if alpha_loc:
                        a = t_angle * alpha_loc_func(x[i])[0]
                        ca, sa = np.cos(a), np.sin(a)
                    else:
                        ca, sa = np.ones(d.shape[0]), np.zeros(d.shape[0])
                    if beta_loc:
                        a = t_angle * beta_loc_func(x[i])[0]
                        cb, sb = np.cos(a), np.sin(a)
                    else:
                        cb, sb = np.ones(d.shape[0]), np.zeros(d.shape[0])
                    if gamma_loc:
                        a = t_angle * gamma_loc_func(x[i])[0]
                        cc, sc = np.cos(a), np.sin(a)
                    else:
                        cc, sc = np.ones(d.shape[0]), np.zeros(d.shape[0])
                    d = d.dot(np.array(
                            [[  ca * cc + sa * sb * sc,  sa * cb, - ca * sc + sa * sb * cc],
                             [- sa * cc + ca * sb * sc,  ca * cb,   sa * sc + ca * sb * cc],
                             [                 cb * sc,     - sb,                  cb * cc]]))
                if coord_loc == 1:
                    d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                elif coord_loc == 2:
                    d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                elif coord_loc == 3:
                    d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                    d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                elif coord_loc == 4:
                    d[:, 2] = coord3_factor_loc_func(x[i])[0]*d[:, 2]
                elif coord_loc == 5:
                    d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                    d[:, 2] = coord3_factor_loc_func(x[i])[0]*d[:, 2]
                elif coord_loc == 6:
                    d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                    d[:, 2] = coord3_factor_loc_func(x[i])[0]*d[:, 2]
                elif coord_loc == 7:
                    d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                    d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                    d[:, 2] = coord3_factor_loc_func(x[i])[0]*d[:, 2]
                elif coord_loc == 8:
                    d = coord1_factor_loc_func(x[i])[0]*d
                elif coord_loc == 9:
                    d[:, (0, 1)] = coord1_factor_loc_func(x[i])[0]*d[:, (0, 1)]
                elif coord_loc == 10:
                    d[:, (0, 2)] = coord1_factor_loc_func(x[i])[0]*d[:, (0, 2)]
                elif coord_loc == 11:
                    d[:, (1, 2)] = coord2_factor_loc_func(x[i])[0]*d[:, (1, 2)]
                elif coord_loc == 12:
                    d[:, (0, 1)] = coord1_factor_loc_func(x[i])[0]*d[:, (0, 1)]
                    d[:, 2] = coord3_factor_loc_func(x[i])[0]*d[:, 2]
                elif coord_loc == 13:
                    d[:, (0, 2)] = coord1_factor_loc_func(x[i])[0]*d[:, (0, 2)]
                    d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                elif coord_loc == 14:
                    d[:, (1, 2)] = coord2_factor_loc_func(x[i])[0]*d[:, (1, 2)]
                    d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                d_abs = np.fabs(d)
                # d01: distance to plane spanned by axes 0 and 1 (in new system)
                # d2: distance to axis 2 (in new system)
                d01 = d_abs[:, 2]
                d2 = np.sqrt(d[:, 0]**2 + d[:, 1]**2)
                ind = np.where(np.all((d2 <= hmax[0], d01 <= tol_dist[0], d01 <= tol_s[0]*d2), axis=0))[0]
                if len(ind) > 0:
                    h01.append(d2[ind])
                    if w_loc:
                        wf = w_factor_loc_func(x[i])[0]
                    g01.append(wf * 0.5*(v[i] - v[i+1+ind])**2)
                ind = np.where(np.all((d01 <= hmax[1], d2 <= tol_dist[1], d2 <= tol_s[1]*d01), axis=0))[0]
                if len(ind) > 0:
                    h2.append(d01[ind])
                    if w_loc:
                        wf = w_factor_loc_func(x[i])[0]
                    g2.append(wf * 0.5*(v[i] - v[i+1+ind])**2)
    else:
        if rotate_coord_sys:
            # Rotate according to new system
            x = x.dot(mrot)
        for i in range(n-1):
            d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
            d_abs = np.fabs(d)
            # d01: distance to plane spanned by axes 0 and 1 (in new system)
            # d2: distance to axis 2 (in new system)
            d01 = d_abs[:, 2]
            d2 = np.sqrt(d[:, 0]**2 + d[:, 1]**2)
            ind = np.where(np.all((d2 <= hmax[0], d01 <= tol_dist[0], d01 <= tol_s[0]*d2), axis=0))[0]
            if len(ind) > 0:
                h01.append(d2[ind])
                g01.append(0.5*(v[i] - v[i+1+ind])**2)
            ind = np.where(np.all((d01 <= hmax[1], d2 <= tol_dist[1], d2 <= tol_s[1]*d01), axis=0))[0]
            if len(ind) > 0:
                h2.append(d01[ind])
                g2.append(0.5*(v[i] - v[i+1+ind])**2)

    npair01 = len(h01)
    if npair01:
        h01 = np.hstack(h01)
        g01 = np.hstack(g01)
    npair2 = len(h2)
    if npair2:
        h2 = np.hstack(h2)
        g2 = np.hstack(g2)

    if make_plot:
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(2,2,1, projection='3d')
        ax2 = fig.add_subplot(2,2,2)
        ax3 = fig.add_subplot(2,2,3)
        ax4 = fig.add_subplot(2,2,4)

        # Plot system Oxzy and Ox'y'z'
        # This:
        ax1.plot([0,1], [0,0], [0,0], color='k')
        ax1.plot([0,0], [0,1], [0,0], color='k')
        ax1.plot([0,0], [0,0], [0,1], color='k')
        ax1.plot([0, mrot[0,0]], [0, mrot[1,0]], [0, mrot[2,0]], color=color01, label="x'''")
        ax1.plot([0, mrot[0,1]], [0, mrot[1,1]], [0, mrot[2,1]], color=color01, label="y'''")
        ax1.plot([0, mrot[0,2]], [0, mrot[1,2]], [0, mrot[2,2]], color=color2, label="z'''")
        ax1.set_xticks([0,1])
        ax1.set_yticks([0,1])
        ax1.set_zticks([0,1])
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.legend()
        plt.sca(ax1)
        plt.title("System Ox'''y'''z'''")

        plt.sca(ax2)
        # Plot both variogram clouds
        plot_variogramCloud1D(h01, g01, c=color01, alpha=0.5, label="in x'''y'''")
        plot_variogramCloud1D(h2,  g2,  c=color2,  alpha=0.5, label="along z'''")
        plt.legend()

        plt.sca(ax3)
        # Plot variogram cloud in x'''y'''
        plot_variogramCloud1D(h01, g01, c=color01, **kwargs)
        plt.title(f"in x'''y''' ({npair01} pts)")

        plt.sca(ax4)
        # Plot variogram cloud along z'''
        plot_variogramCloud1D(h2, g2, c=color2, **kwargs)
        plt.title(f"along z''' ({npair2} pts)")

        plt.suptitle(f'Vario cloud: alpha={alpha}deg. beta={beta}deg. gamma={gamma}deg.\ntol_dist={tol_dist} / tol_angle={tol_angle}deg.')
        # plt.show()

    return (h01, g01, npair01), (h2, g2, npair2)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def variogramExp3D_omni_wrt_2_first_axes(
        x, v,
        alpha=0.0,
        beta=0.0,
        gamma=0.0,
        tol_dist=None,
        tol_angle=None,
        hmax=None,
        alpha_loc_func=None,
        beta_loc_func=None,
        gamma_loc_func=None,
        w_factor_loc_func=None,
        coord1_factor_loc_func=None,
        coord2_factor_loc_func=None,
        coord3_factor_loc_func=None,
        loc_m=1,
        ncla=(10, 10),
        cla_center=(None, None),
        cla_length=(None, None),
        variogramCloud=None,
        make_plot=True,
        color01='orange',
        color2='blue',
        figsize=None, 
        logger=None,
        **kwargs):
    """
    Computes two experimental variograms for a data set in 3D.

    The computed experimental variograms are:

    - the experimental omni-directional variogram wrt. the first 2 axes (i.e with \
    any direction parallel to the plane spanned by the first 2 axes)
    - the experimental directional variogram wrt. the 3rd main axis

    For both experimental variograms, the mean point in each class is retrieved
    from corresponding variogram cloud (returned by the function
    `variogramCloud3D_omni_wrt_2_first_axes`); for the experimental omni-directional
    variograma (j=0) (resp. the experimental directional variogram (j=1)),
    the i-th class is determined by its
    center `cla_center[j][i]` and its length `cla_length[j][i]`, and corresponds
    to the interval

        `]cla_center[j][i]-cla_length[j][i]/2, cla_center[j][i]+cla_length[j][i]/2]`

    (lag) axis (abscissa).

    Parameters
    ----------
    x : 2D array of floats of shape (n, 3)
        data points locations, with n the number of data points, each row of `x`
        is the coordinatates of one data point

    v : 1D array of floats of shape (n,)
        data points values, with n the number of data points, `v[i]` is the data
        value at location `x[i]`

    alpha : float, default: 0.0
        azimuth angle in degrees (see :class:`CovModel3D`)

    beta : float, default: 0.0
        dip angle in degrees (see :class:`CovModel3D`)

    gamma : float, default: 0.0
        plunge angle in degrees (see :class:`CovModel3D`)

    tol_dist : sequence of 2 floats, or float, optional
        let `tol_dist` = (tol_dist12, tol_dist3); tol_dist12 (resp. tol_dist3) is
        the maximal distance to the plane spanned by the first two main axes (resp.
        to the 3rd main axis) for the lag (vector between two data points), such
        that the pair is integrated in
        the omni-directional variogram cloud wrt. the first two main axes
        (resp. the directional variogram cloud wrt. the 3rd main axis);
        note: if `tol_dist` is specified as a float or `None` (default), the
        entry is duplicated; if tol_dist12 (resp. tol_dist3) is `None`,
        then tol_dist12 (resp. tol_dist3) is set to 10% of h12max (resp.
        h3max) if h12max (resp. h3max) is finite, and set to 10.0
        otherwise: see parameter `hmax` for the definition of h12max and h3max

    tol_angle : sequence of 2 floats, or float, optional
        let `tol_angle` = (tol_angle12, tol_angl3); tol_angle12 (resp. tol_angle3)
        is the maximal angle in degrees between the lag (vector between two data
        points) and the plane spanned by the first two main axes (resp.
        the 3rd main axis), such that the pair is integrated in
        the omni-directional variogram cloud wrt. the first two main axes
        (resp. the directional variogram cloud wrt. the 3rd main axis);
        note: if `tol_angle` is specified as a float, it is duplicated;
        by default (`None`): `tol_angle` is set to 45.0

    hmax : sequence of 2 floats, or float, optional
        let `hmax` = (h12max, h3max); h12max (resp. h3max) is the
        maximal distance between a pair of data points in the plane
        spanned by the first two main axes (resp. along the 3rd) main axis,
        such that the pair in
        the omni-directional variogram cloud wrt. the first two main axes
        (resp. the directional variogram cloud wrt. the 3rd main axis);
        note: if `hmax` is specified as a float or `None` (default), the entry is
        duplicated, and `None`, `numpy.nan` are converted to `numpy.inf` (no
        restriction)

    alpha_loc_func : function (`callable`), optional
        function returning azimuth angle, defining the main axes, as function of
        a given location in 3D, i.e. the main axes are defined locally

    beta_loc_func : function (`callable`), optional
        function returning dip angle, defining the main axes, as function of
        a given location in 3D, i.e. the main axes are defined locally

    gamma_loc_func : function (`callable`), optional
        function returning plunge angle, defining the main axes, as function of
        a given location in 3D, i.e. the main axes are defined locally

    w_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "weight" as function of a given
        location in 3D, i.e. "g" values (i.e. ordinate axis component in the two
        variograms) are multiplied

    coord1_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" along the 1st (local) main
        axis as function of a given location in 3D, i.e. "h1" values (i.e.
        abscissa axis component in the 1st variogram) are multiplied
        (the condition wrt h1max, see `hmax`, is checked after)

    coord2_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" along the 2nd (local) main
        axis as function of a given location in 3D, i.e. "h2" values (i.e.
        abscissa axis component in the 2nd variogram) are multiplied
        (the condition wrt h2max, see `hmax`, is checked after)

    coord3_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" along the 3rd (local) main
        axis as function of a given location in 3D, i.e. "h3" values (i.e.
        abscissa axis component in the 3rd variogram) are multiplied
        (the condition wrt h3max, see `hmax`, is checked after)

    loc_m : int, default: 1
        integer (greater than or equal to 0) defining how the function(s)
        `*_loc_func` (above) are evaluated for a pair of two locations x1, x2
        (data point locations):

        - if `loc_m>0` the segment from x1 to x2 is divided in `loc_m` intervals \
        of same length and the mean of the evaluations of the function at the \
        (`loc_m` + 1) interval bounds is computed
        - if `loc_m=0`, the evaluation at x1 is considered

    ncla : sequence of 2 ints, default: (10, 10)
        number of classes for each experimental variogram, the parameter `ncla[j]`
        is used if `cla_center[j]=None`, in that situation `ncla[j]` classes are
        considered and the class centers are set to

        - `cla_center[j][i] = (i+0.5)*l, i=0,...,ncla[j]-1`

        with l = H / ncla[j], H being the max of the distance, between two points
        of the considered pairs in the corresponding variogram cloud;
        if `cla_center[j]` is specified (not `None`), the number of classes
        (`ncla[j]`) is set to the length of the sequence `cla_center[j]`
        (ignoring the value passed as argument)

    cla_center : sequence of length 2
        cla_center[j] : 1D array-like of floats, or `None` (default)
            center of each class (in abscissa) for each experimental variogram;
            by default (`None`):
            `cla_center[j]` is defined from `ncla[j]` (see above)

    cla_length : sequence of length 2
        cla_length[j] : 1D array-like of floats, or float, or `None`
            length of each class centered at `cla_center[j]` (in abscissa) for
            each experimental variogram:

            - if `cla_length[j]` is a sequence, it should be of length `ncla[j]`
            - if `cla_length[j]` is a float, the value is repeated `ncla[j]` times
            - if `cla_length[j]=None` (default), the minimum of difference between \
            two sucessive class centers (`np.inf` if one class) is used and \
            repeated `ncla[j]` times

    variogramCloud : sequence of two 3-tuple, optional
        `variogramCloud` = ((h01, g01, npair01), (h2, g2, npair2))
        is variogram clouds (already computed and returned by the function
        `variogramCloud3D_omni_wrt_2_first_axes` (npair01, npair2 not used));
        in this case, `x`, `v`, `alpha`, `tol_dist`, `tol_angle`, `hmax`,
        `alpha_loc_func`, `w_factor_loc_func`, `coord1_factor_loc_func`,
        `coord2_factor_loc_func`, `coord3_factor_loc_func`, `loc_m` are not used

        By default (`None`): the variogram clouds are computed by using the
        function `variogramCloud3D_omni_wrt_2_first_axes`

    make_plot : bool, default: True
        indicates if the experimental variograms are plotted (in a new "2x2"
        figure)

    color01 : color, default: 'orange'
        color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
        the plot of the experimental variogram, omni-directional wrt. the first 2
        main axes (if `make_plot=True`)

    color2 : color, default: 'blue'
        color (3-tuple (RGB code), 4-tuple (RGBA code) or str), used for
        the plot of the variogram cloud along the 3rd main axis
        (if `make_plot=True`)

    figsize : 2-tuple, optional
        size of the new "2x2" figure (if `make_plot=True`)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs : dict
        keyword arguments passed to the funtion `plot_variogramExp1D`
        (if `make_plot=True`)

    Returns
    -------
    (hexp01, gexp01, cexp01) : 3-tuple
        hexp0, gexp0 : 1D arrays of floats of same length
            coordinates of the points of the experimental variogram,
            omni-directional wrt. the first 2 main axes

        cexp01 : 1D array of ints
            array of same length as `hexp01`, `gexp01`, number of points (pairs of
            data points considered) in each class in the variogram,
            omni-directional wrt. the first 2 main axes

    (hexp2, gexp2, cexp2) : 3-tuple
        hexp2, gexp2 : 1D arrays of floats of same length
            coordinates of the points of the experimental variogram along the
            3rd main axis

        cexp2 : 1D array of ints
            array of same length as `hexp2`, `gexp2`, number of points (pairs of
            data points considered) in each class in the variogram cloud along
            the 3rd main axis
    """
    fname = 'variogramExp3D_omni_wrt_2_first_axes'

    # Compute variogram clouds if needed
    if variogramCloud is None:
        try:
            vc = variogramCloud3D_omni_wrt_2_first_axes(
                    x, v, alpha=alpha, beta=beta, gamma=gamma, tol_dist=tol_dist, tol_angle=tol_angle, hmax=hmax,
                    alpha_loc_func=alpha_loc_func, 
                    beta_loc_func=beta_loc_func, 
                    gamma_loc_func=gamma_loc_func,
                    w_factor_loc_func=w_factor_loc_func,
                    coord1_factor_loc_func=coord1_factor_loc_func, 
                    coord2_factor_loc_func=coord2_factor_loc_func, 
                    coord3_factor_loc_func=coord3_factor_loc_func, 
                    loc_m=loc_m,
                    make_plot=False,
                    logger=logger)
        except Exception as exc:
            err_msg = f'{fname}: cannot compute variogram cloud (3D)'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg) from exc

    else:
        vc = variogramCloud
    # -> vc[0] = (h01, g01, npair01) and vc[1] = (h2, g2, npair2)

    # Compute variogram experimental in each direction (using function variogramExp1D)
    ve = [None, None]
    for j in (0, 1):
        try:
            ve[j] = variogramExp1D(
                        None, None, 
                        hmax=None, w_factor_loc_func=None, coord_factor_loc_func=None, loc_m=loc_m,
                        ncla=ncla[j], cla_center=cla_center[j], cla_length=cla_length[j], variogramCloud=vc[j], 
                        make_plot=False, logger=logger)
        except Exception as exc:
            err_msg = f'{fname}: cannot compute experimental variogram in one direction'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg) from exc

    (hexp01, gexp01, cexp01), (hexp2, gexp2, cexp2) = ve

    if make_plot:
        # Rotation matrix
        a = alpha * np.pi/180.
        b = beta * np.pi/180.
        c = gamma * np.pi/180.
        ca, sa = np.cos(a), np.sin(a)
        cb, sb = np.cos(b), np.sin(b)
        cc, sc = np.cos(c), np.sin(c)

        mrot = np.array([[  ca * cc + sa * sb * sc,  sa * cb,  - ca * sc + sa * sb * cc],
                         [- sa * cc + ca * sb * sc,  ca * cb,    sa * sc + ca * sb * cc],
                         [                 cb * sc,     - sb,                   cb * cc]])

        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(2,2,1, projection='3d')
        ax2 = fig.add_subplot(2,2,2)
        ax3 = fig.add_subplot(2,2,3)
        ax4 = fig.add_subplot(2,2,4)

        # Plot system Oxzy and Ox'y'z'
        # This:
        ax1.plot([0,1], [0,0], [0,0], color='k')
        ax1.plot([0,0], [0,1], [0,0], color='k')
        ax1.plot([0,0], [0,0], [0,1], color='k')
        ax1.plot([0, mrot[0,0]], [0, mrot[1,0]], [0, mrot[2,0]], color=color01, label="x'''")
        ax1.plot([0, mrot[0,1]], [0, mrot[1,1]], [0, mrot[2,1]], color=color01, label="y'''")
        ax1.plot([0, mrot[0,2]], [0, mrot[1,2]], [0, mrot[2,2]], color=color2, label="z'''")
        ax1.set_xticks([0,1])
        ax1.set_yticks([0,1])
        ax1.set_zticks([0,1])
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        ax1.legend()
        plt.sca(ax1)
        plt.title("System Ox'''y'''z'''")

        plt.sca(ax2)
        # Plot variogram exp in x'''y''' and along z'''
        plot_variogramExp1D(hexp01, gexp01, cexp01, show_count=False, grid=True, c=color01, alpha=0.5, label="in x'''y'''")
        plot_variogramExp1D(hexp2,  gexp2,  cexp2,  show_count=False, grid=True, c=color2,  alpha=0.5, label="along z'''")
        plt.legend()

        plt.sca(ax3)
        # Plot variogram exp in x'''y'''
        plot_variogramExp1D(hexp01, gexp01, cexp01, c=color01, **kwargs)
        plt.title("in x'''y'''")

        plt.sca(ax4)
        # Plot variogram exp along z'''
        plot_variogramExp1D(hexp2, gexp2, cexp2, c=color2, **kwargs)
        plt.title("along z'''")

        plt.suptitle(f'Vario exp.: alpha={alpha}deg. beta={beta}deg. gamma={gamma}deg.')
        # plt.suptitle(f'Vario exp.: alpha={alpha}deg. beta={beta}deg. gamma={gamma}deg.\ntol_dist={tol_dist} / tol_angle={tol_angle}deg.')
        # plt.show()

    return (hexp01, gexp01, cexp01), (hexp2, gexp2, cexp2)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel3D_fit(
        x, v, cov_model,
        hmax=None,
        link_range12=False,
        alpha_loc_func=None,
        beta_loc_func=None,
        gamma_loc_func=None,
        w_factor_loc_func=None,
        coord1_factor_loc_func=None,
        coord2_factor_loc_func=None,
        coord3_factor_loc_func=None,
        loc_m=1,
        make_plot=True,
        verbose=0,
        logger=None,
        **kwargs):
    """
    Fits a covariance model in 3D (for data in 3D).

    The parameter `cov_model` is a covariance model in 3D where all the
    parameters to be fitted are set to `numpy.nan`. The fit is done according to
    the variogram cloud, by using the function `scipy.optimize.curve_fit`.

    Parameters
    ----------
    x : 2D array of floats of shape (n, 3)
        data points locations, with n the number of data points, each row of `x`
        is the coordinatates of one data point

    v : 1D array of floats of shape (n,)
        data points values, with n the number of data points, `v[i]` is the data
        value at location `x[i]`

    cov_model : :class:`CovModel3D`
        covariance model to otpimize (parameters set to `numpy.nan` are optimized)

    hmax : sequence of 3 floats, or float, optional
        the pairs of data points with lag h (in rotated coordinates system if
        applied) satisfying

        .. math::
            (h[0]/hmax[0])^2 + (h[1]/hmax[1])^2 + (h[2]/hmax[2])^2 \\leqslant 1

        are taking into account in the variogram cloud
        note: if `hmax` is specified as a float or `None` (default), the entry is
        duplicated, and `None`, `numpy.nan` are converted to `numpy.inf` (no
        restriction)

    link_range12 : bool, default: False
        - if `True`: ranges along the first two main axes are "linked", i.e. must \
        have the same value; in particular, both ranges along the first two main axes \
        must be set to the same value or be set for optimization, and `hmax[0]` \
        must be equal to `hmax[1]`
        - if `False`: ranges along the first two main axes are independent

    alpha_loc_func : function (`callable`), optional
        function returning azimuth angle, defining the main axes, as function of
        a given location in 3D, i.e. the main axes are defined locally

    beta_loc_func : function (`callable`), optional
        function returning dip angle, defining the main axes, as function of
        a given location in 3D, i.e. the main axes are defined locally

    gamma_loc_func : function (`callable`), optional
        function returning plunge angle, defining the main axes, as function of
        a given location in 3D, i.e. the main axes are defined locally

    w_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "weight" as function of a given
        location in 3D, i.e. "g" values (i.e. ordinate axis component in the two
        variograms) are multiplied

    coord1_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" along the 1st (local) main
        axis as function of a given location in 3D, i.e. "h1" values (i.e.
        abscissa axis component in the 1st variogram) are multiplied
        (the condition wrt `hmax`, is checked after)

    coord2_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" along the 2nd (local) main
        axis as function of a given location in 3D, i.e. "h2" values (i.e.
        abscissa axis component in the 2nd variogram) are multiplied
        (the condition wrt `hmax`, is checked after)

    coord3_factor_loc_func : function (`callable`), optional
        function returning a multiplier for the "lag" along the 3rd (local) main
        axis as function of a given location in 3D, i.e. "h3" values (i.e.
        abscissa axis component in the 3rd variogram) are multiplied
        (the condition wrt `hmax`, is checked after)

    loc_m : int, default: 1
        integer (greater than or equal to 0) defining how the function(s)
        `*_loc_func` (above) are evaluated for a pair of two locations x1, x2
        (data point locations):

        - if `loc_m>0` the segment from x1 to x2 is divided in `loc_m` intervals \
        of same length and the mean of the evaluations of the function at the \
        (`loc_m` + 1) interval bounds is computed
        - if `loc_m=0`, the evaluation at x1 is considered

    make_plot : bool, default: True
        indicates if the fitted covariance model is plotted (in a new "1x2"
        figure, using the method `plot_model` with default parameters)

    figsize : 2-tuple, optional
        size of the new "1x2" figure (if `make_plot=True`)

    verbose : int, default: 0
        verbose mode, higher implies more printing (info)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    kwargs : dict
        keyword arguments passed to the funtion `scipy.optimize.curve_fit`

    Returns
    -------
    cov_model_opt: :class:`CovModel3D`
        optimized covariance model
    popt: 1D array
        values of the optimal parameters, corresponding to the parameters of the
        input covariance model (`cov_model`) set to `numpy.nan`, in the order of
        appearance (vector of optimized parameters returned by
        `scipy.optimize.curve_fit`)

    Examples
    --------
    The following allows to fit a covariance model made up of a gaussian
    elementary model and a nugget effect (nugget elementary model), where the
    azimuth angle (defining the main axes), the weight and ranges of the gaussian
    elementary model and the weight of the nugget effect are fitted (optimized)
    in intervals given by the keyword argument `bounds`. The arguments `x`, `v`
    are the data points and values, and the fitted covariance model is not plotted
    (`make_plot=False`)

        >>> # covariance model to optimize
        >>> cov_model = CovModel3D(elem=[
        >>>     ('gaussian', {'w':np.nan, 'r':[np.nan, np.nan, np.nan]}), # el. contrib.
        >>>     ('nugget', {'w':np.nan})                                  # el. contrib.
        >>>     ], alpha=np.nan, beta=0.0, gamma=0.0,      # azimuth, dip, plunge angles
        >>>     name='')
        >>> covModel3D_fit(x, v, cov_model_to_optimize,
        >>>                bounds=([ 0.0,   0.0,   0.0,   0.0,  0.0, -90.0],  # lower b.
        >>>                        [10.0, 100.0, 100.0, 100.0, 10.0,  90.0]), # upper b.
        >>>                                                      # for parameters to fit
        >>>                make_plot=False)
    """
    fname = 'covModel3D_fit'

    # Check cov_model
    if not isinstance(cov_model, CovModel3D):
        err_msg = f'{fname}: `cov_model` is not a covariance model in 3D'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)
    # if cov_model.__class__.__name__ != 'CovModel3D':
    #     err_msg = f'{fname}: `cov_model` is not a covariance model in 1D'
    #     if logger: logger.error(err_msg)
    #     raise CovModelError(err_msg)

    # Prevent calculation if covariance model is not stationary
    if not cov_model.is_stationary():
        err_msg = f'{fname}: `cov_model` is not stationary: fit cannot be applied'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # Work on a (deep) copy of cov_model
    cov_model_opt = copy.deepcopy(cov_model)

    # Get index of element, key of parameters and index of range to fit
    ielem_to_fit=[]
    key_to_fit=[]
    ir_to_fit=[] # if key is equal to 'r' (range), set the index of the range to fit, otherwise set np.nan
    for i, el in enumerate(cov_model_opt.elem):
        for k, val in el[1].items():
            if k == 'r':
                if link_range12:
                    if np.isnan(val[0]):
                        if not np.isnan(val[1]):
                            err_msg = f"{fname}: with `link_range12=True`, range ('r') along the first two main axes must both be defined (to the same value) or both to be optimized"
                            if logger: logger.error(err_msg)
                            raise CovModelError(err_msg)

                        ielem_to_fit.append(i)
                        key_to_fit.append(k)
                        ir_to_fit.append(0)
                    else:
                        if np.isnan(val[1]):
                            err_msg = f"{fname}: with `link_range12=True`, range ('r') along the first two main axes must both be defined (to the same value) or both to be optimized"
                            if logger: logger.error(err_msg)
                            raise CovModelError(err_msg)

                        if val[0] != val[1]:
                            err_msg = f"{fname}: with `link_range12=True`, range ('r') defined along the first two main axes must have the same value"
                            if logger: logger.error(err_msg)
                            raise CovModelError(err_msg)

                    if np.isnan(val[2]):
                        ielem_to_fit.append(i)
                        key_to_fit.append(k)
                        ir_to_fit.append(2)
                else:
                    for j in (0, 1, 2):
                        if np.isnan(val[j]):
                            ielem_to_fit.append(i)
                            key_to_fit.append(k)
                            ir_to_fit.append(j)
            elif np.isnan(val):
                ielem_to_fit.append(i)
                key_to_fit.append(k)
                ir_to_fit.append(np.nan)

    # Is angle alpha, beta, gamma must be fit ?
    alpha_to_fit = np.isnan(cov_model_opt.alpha)
    beta_to_fit  = np.isnan(cov_model_opt.beta)
    gamma_to_fit = np.isnan(cov_model_opt.gamma)

    nparam = len(ielem_to_fit) + int(alpha_to_fit) + int(beta_to_fit) + int(gamma_to_fit)
    if nparam == 0:
        # print('No parameter to fit!')
        return cov_model_opt, np.array([])

    # Set hmax as an array of shape (2,)
    hmax = np.atleast_1d(hmax).astype('float').reshape(-1) # None are converted to nan
    hmax[np.isnan(hmax)] = np.inf # convert nan to inf
    if hmax.size == 1:
        hmax = np.array([hmax[0], hmax[0], hmax[0]])
    elif hmax.size != 3:
        err_msg = f'{fname}: size of `hmax` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    if link_range12 and hmax[0] != hmax[1]:
        err_msg = f'{fname}: with `link_range12=True`, the first two entries of `hmax` must be the same ones'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    a = alpha_to_fit + beta_to_fit + gamma_to_fit
    if a == 1:
        if alpha_to_fit:
            if hmax[0] != hmax[1] and verbose > 0:
                if logger:
                    logger.warning(f'{fname}: as alpha angle (only) is flagged for fitting, `hmax[0]` and `hmax[1]` should be equal')
                else:
                    print(f'{fname}: WARNING: as alpha angle (only) is flagged for fitting, `hmax[0]` and `hmax[1]` should be equal')
        elif beta_to_fit:
            if hmax[1] != hmax[2] and verbose > 0:
                if logger:
                    logger.warning(f'{fname}: as beta angle (only) is flagged for fitting, `hmax[1]` and `hmax[2]` should be equal')
                else:
                    print(f'{fname}: WARNING: as beta angle (only) is flagged for fitting, `hmax[1]` and `hmax[2]` should be equal')
        elif gamma_to_fit:
            if hmax[0] != hmax[2] and verbose > 0:
                if logger:
                    logger.warning(f'{fname}: as beta angle (only) is flagged for fitting, `hmax[0]` and `hmax[2]` should be equal')
                else:
                    print(f'{fname}: WARNING: as beta angle (only) is flagged for fitting, `hmax[0]` and `hmax[2]` should be equal')
    elif a > 0:
        if hmax[0] != hmax[2] or hmax[0] != hmax[1] or hmax[1] != hmax[2] and verbose > 0:
            if logger:
                logger.warning(f'{fname}: as (at least two) angles are flagged for fitting, all the components of `hmax` should be equal')
            else:
                print(f'{fname}: WARNING: as (at least two) angles are flagged for fitting, all the components of `hmax` should be equal')

    if not alpha_to_fit and not beta_to_fit and not gamma_to_fit \
            and (cov_model_opt.alpha != 0.0 or cov_model_opt.beta != 0.0 or cov_model_opt.gamma != 0.0):
        alpha_copy = cov_model_opt.alpha
        beta_copy = cov_model_opt.beta
        gamma_copy = cov_model_opt.gamma
        rotate_coord_sys = True
        mrot = cov_model_opt.mrot()
        cov_model_opt.alpha = 0.0 # set (temporarily) to 0.0
        cov_model_opt.beta = 0.0 # set (temporarily) to 0.0
        cov_model_opt.gamma = 0.0 # set (temporarily) to 0.0
    else:
        rotate_coord_sys = False

    # Set types of local transformations
    #    alpha_loc: True / False: is local angle alpha used ?
    #    beta_loc : True / False: is local angle beta used ?
    #    gamma_loc: True / False: is local angle gamma used ?
    #    rotation_loc: True / False: is local rotation used ?
    #    w_loc:     True / False: is local w (weight / sill) used ?
    #    coord_loc: integer
    #               0: no transformation
    #               1: transformation for 1st coordinate only
    #               2: transformation for 2nd coordinate only
    #               3: distinct transformations for 1st and 2nd coordinates, no transformation for 3rd coordinate
    #               4: transformation for 3rd coordinate only
    #               5: distinct transformations for 1st and 3rd coordinates, no transformation for 2nd coordinate
    #               6: distinct transformations for 2nd and 3rd coordinates, no transformation for 1st coordinate
    #               7: distinct transformations for 1st, 2nd and 3rd coordinates
    #               8: same transformation for 1st, 2nd and 3rd coordinates
    #               9: same transformation for 1st and 2nd coordinates, no transformation for 3rd coordinate
    #              10: same transformation for 1st and 3rd coordinates, no transformation for 2nd coordinate
    #              11: same transformation for 2nd and 3rd coordinates, no transformation for 1st coordinate
    #              12: same transformation for 1st and 2nd coordinates, distinct transformation for 3rd coordinate
    #              13: same transformation for 1st and 3rd coordinates, distinct transformation for 2nd coordinate
    #              14: same transformation for 2nd and 3rd coordinates, distinct transformation for 1st coordinate
    alpha_loc = False
    beta_loc = False
    gamma_loc = False
    rotation_loc = False
    w_loc = False
    coord_loc = 0
    if alpha_loc_func is not None:
        alpha_loc = True
        rotation_loc = True
    if beta_loc_func is not None:
        beta_loc = True
        rotation_loc = True
    if gamma_loc_func is not None:
        gamma_loc = True
        rotation_loc = True

    if rotation_loc:
        # factor to transform angle in degree into radian
        t_angle = np.pi/180.0

    if w_factor_loc_func is not None:
        w_loc = True

    if coord1_factor_loc_func is not None:
        coord_loc = coord_loc + 1
    if coord2_factor_loc_func is not None:
        coord_loc = coord_loc + 2
    if coord3_factor_loc_func is not None:
        coord_loc = coord_loc + 4
    if coord_loc == 3:
        if coord1_factor_loc_func == coord2_factor_loc_func:
            coord_loc = 9
    elif coord_loc == 5:
        if coord1_factor_loc_func == coord3_factor_loc_func:
            coord_loc = 10
    elif coord_loc == 6:
        if coord2_factor_loc_func == coord3_factor_loc_func:
            coord_loc = 11
    elif coord_loc == 7:
        if coord1_factor_loc_func == coord2_factor_loc_func and coord1_factor_loc_func == coord3_factor_loc_func:
            coord_loc = 8
        elif coord1_factor_loc_func == coord2_factor_loc_func:
            coord_loc = 12
        elif coord1_factor_loc_func == coord3_factor_loc_func:
            coord_loc = 13
        elif coord2_factor_loc_func == coord3_factor_loc_func:
            coord_loc = 14

    if alpha_loc or beta_loc or gamma_loc or w_loc or coord_loc > 0:
        transform_flag = True
    else:
        transform_flag = False

    # Compute lag vector (h) and gamma value (g) for pair of points with distance less than or equal to hmax
    n = x.shape[0] # number of points
    if np.all(np.isinf(hmax)):
        # Consider all pairs of points
        npair = int(0.5*(n-1)*n)
        h = np.zeros((npair, 3))
        g = np.zeros(npair)
        j = 0
        if transform_flag:
            wf = 1.0 # default weight factor
            if loc_m > 0:
                for i in range(n-1):
                    jj = n-1-i
                    d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
                    dx = d/loc_m
                    ddx = np.asarray([x[i]+np.outer(np.arange(loc_m+1), dxk) for dxk in dx]) # 3-dimensional array (n-1-i) x (loc_m+1) x dim
                    if rotate_coord_sys:
                        # Rotate according to new system
                        d = d.dot(mrot)
                    if rotation_loc:
                        if alpha_loc:
                            a = t_angle * alpha_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1) # 2-dimensional array (n-1-i) x (loc_m+1)
                            # a = t_angle * np.asarray([alpha_loc_func(ddxk) for ddxk in ddx])
                            ca, sa = np.mean(np.cos(a), axis=1), np.mean(np.sin(a), axis=1)
                        else:
                            ca, sa = np.ones(d.shape[0]), np.zeros(d.shape[0])
                        if beta_loc:
                            a = t_angle * beta_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1) # 2-dimensional array (n-1-i) x (loc_m+1)
                            # a = t_angle * np.asarray([beta_loc_func(ddxk) for ddxk in ddx])
                            cb, sb = np.mean(np.cos(a), axis=1), np.mean(np.sin(a), axis=1)
                        else:
                            cb, sb = np.ones(d.shape[0]), np.zeros(d.shape[0])
                        if gamma_loc:
                            a = t_angle * gamma_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1) # 2-dimensional array (n-1-i) x (loc_m+1)
                            # a = t_angle * np.asarray([gamma_loc_func(ddxk) for ddxk in ddx])
                            cc, sc = np.mean(np.cos(a), axis=1), np.mean(np.sin(a), axis=1)
                        else:
                            cc, sc = np.ones(d.shape[0]), np.zeros(d.shape[0])
                        d = np.asarray([dk.dot(np.array(
                                        [[  cak * cck + sak * sbk * sck,  sak * cbk, - cak * sck + sak * sbk * cck],
                                         [- sak * cck + cak * sbk * sck,  cak * cbk,   sak * sck + cak * sbk * cck],
                                         [                    cbk * sck,      - sbk,                     cbk * cck]]))
                                     for (cak, sak, cbk, sbk, cck, sck, dk) in zip (ca, sa, cb, sb, cc, sc, d)])
                        # d = np.asarray([np.array(
                        #                 [[  cak * cck + sak * sbk * sck,  - sak * cck + cak * sbk * sck,  cbk * sck],
                        #                  [                    sak * cbk,                      cak * cbk,      - sbk],
                        #                  [- cak * sck + sak * sbk * cck,    sak * sck + cak * sbk * cck,  cbk * cck]]).dot(dk)
                        #              for (cak, sak, cbk, sbk, cck, sck, dk) in zip (ca, sa, cb, sb, cc, sc, d)])
                    if coord_loc == 1:
                        d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                        # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                    elif coord_loc == 2:
                        d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                        # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                    elif coord_loc == 3:
                        d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                        d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                        # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                        # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                    elif coord_loc == 4:
                        d[:, 2] = np.mean(coord3_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 2]
                        # d[:, 2] = np.asarray([np.mean(coord3_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 2]
                    elif coord_loc == 5:
                        d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                        d[:, 2] = np.mean(coord3_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 2]
                        # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                        # d[:, 2] = np.asarray([np.mean(coord3_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 2]
                    elif coord_loc == 6:
                        d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                        d[:, 2] = np.mean(coord3_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 2]
                        # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                        # d[:, 2] = np.asarray([np.mean(coord3_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 2]
                    elif coord_loc == 7:
                        d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                        d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                        d[:, 2] = np.mean(coord3_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 2]
                        # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                        # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                        # d[:, 2] = np.asarray([np.mean(coord3_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 2]
                    elif coord_loc == 8:
                        d = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d.T).T
                        # d = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d.T).T
                    elif coord_loc == 9:
                        d[:, (0, 1)] = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (0, 1)].T).T
                        # d[:, (0, 1)] = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (0, 1)].T).T
                    elif coord_loc == 10:
                        d[:, (0, 2)] = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (0, 2)].T).T
                        # d[:, (0, 2)] = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (0, 2)].T).T
                    elif coord_loc == 11:
                        d[:, (1, 2)] = (np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (1, 2)].T).T
                        # d[:, (1, 2)] = (np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (1, 2)].T).T
                    elif coord_loc == 12:
                        d[:, (0, 1)] = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (0, 1)].T).T
                        d[:, 2] = np.mean(coord3_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 2]
                        # d[:, (0, 1)] = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (0, 1)].T).T
                        # d[:, 2] = np.asarray([np.mean(coord3_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 2]
                    elif coord_loc == 13:
                        d[:, (0, 2)] = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (0, 2)].T).T
                        d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                        # d[:, (0, 2)] = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (0, 2)].T).T
                        # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                    elif coord_loc == 14:
                        d[:, (1, 2)] = (np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (1, 2)].T).T
                        d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                        # d[:, (1, 2)] = (np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (1, 2)].T).T
                        # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                    h[j:(j+jj),:] = d
                    if w_loc:
                        wf = np.mean(w_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)
                        # wf = np.asarray([np.mean(w_factor_loc_func(ddxk)) for ddxk in ddx])
                    g[j:(j+jj)] = wf * 0.5*(v[i] - v[(i+1):])**2
                    j = j+jj
            else:
                for i in range(n-1):
                    jj = n-1-i
                    d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
                    if rotate_coord_sys:
                        # Rotate according to new system
                        d = d.dot(mrot)
                    if rotation_loc:
                        if alpha_loc:
                            a = t_angle * alpha_loc_func(x[i])[0]
                            ca, sa = np.cos(a), np.sin(a)
                        else:
                            ca, sa = np.ones(d.shape[0]), np.zeros(d.shape[0])
                        if beta_loc:
                            a = t_angle * beta_loc_func(x[i])[0]
                            cb, sb = np.cos(a), np.sin(a)
                        else:
                            cb, sb = np.ones(d.shape[0]), np.zeros(d.shape[0])
                        if gamma_loc:
                            a = t_angle * gamma_loc_func(x[i])[0]
                            cc, sc = np.cos(a), np.sin(a)
                        else:
                            cc, sc = np.ones(d.shape[0]), np.zeros(d.shape[0])
                        d = d.dot(np.array(
                                [[  ca * cc + sa * sb * sc,  sa * cb, - ca * sc + sa * sb * cc],
                                 [- sa * cc + ca * sb * sc,  ca * cb,   sa * sc + ca * sb * cc],
                                 [                 cb * sc,     - sb,                  cb * cc]]))
                    if coord_loc == 1:
                        d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                    elif coord_loc == 2:
                        d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                    elif coord_loc == 3:
                        d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                        d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                    elif coord_loc == 4:
                        d[:, 2] = coord3_factor_loc_func(x[i])[0]*d[:, 2]
                    elif coord_loc == 5:
                        d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                        d[:, 2] = coord3_factor_loc_func(x[i])[0]*d[:, 2]
                    elif coord_loc == 6:
                        d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                        d[:, 2] = coord3_factor_loc_func(x[i])[0]*d[:, 2]
                    elif coord_loc == 7:
                        d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                        d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                        d[:, 2] = coord3_factor_loc_func(x[i])[0]*d[:, 2]
                    elif coord_loc == 8:
                        d = coord1_factor_loc_func(x[i])[0]*d
                    elif coord_loc == 9:
                        d[:, (0, 1)] = coord1_factor_loc_func(x[i])[0]*d[:, (0, 1)]
                    elif coord_loc == 10:
                        d[:, (0, 2)] = coord1_factor_loc_func(x[i])[0]*d[:, (0, 2)]
                    elif coord_loc == 11:
                        d[:, (1, 2)] = coord2_factor_loc_func(x[i])[0]*d[:, (1, 2)]
                    elif coord_loc == 12:
                        d[:, (0, 1)] = coord1_factor_loc_func(x[i])[0]*d[:, (0, 1)]
                        d[:, 2] = coord3_factor_loc_func(x[i])[0]*d[:, 2]
                    elif coord_loc == 13:
                        d[:, (0, 2)] = coord1_factor_loc_func(x[i])[0]*d[:, (0, 2)]
                        d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                    elif coord_loc == 14:
                        d[:, (1, 2)] = coord2_factor_loc_func(x[i])[0]*d[:, (1, 2)]
                        d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                    h[j:(j+jj),:] = d
                    if w_loc:
                        wf = w_factor_loc_func(x[i])[0]
                    g[j:(j+jj)] = wf * 0.5*(v[i] - v[(i+1):])**2
                    j = j+jj
        else:
            if rotate_coord_sys:
                # Rotate according to new system
                x = x.dot(mrot)
            for i in range(n-1):
                jj = n-1-i
                d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
                h[j:(j+jj),:] = d
                g[j:(j+jj)] = 0.5*(v[i] - v[(i+1):])**2
                j = j+jj
    else:
        # Consider only pairs of points according to parameter hmax, i.e.
        #   pairs with lag h (in rotated coordinates system if applied) satisfying
        #   (h[0]/hmax[0])**2 + (h[1]/hmax[1])**2 + (h[2]/hmax[2])**2 <= 1
        h, g = [], []
        npair = 0
        if transform_flag:
            wf = 1.0 # default weight factor
            if loc_m > 0:
                for i in range(n-1):
                    jj = n-1-i
                    d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
                    dx = d/loc_m
                    ddx = np.asarray([x[i]+np.outer(np.arange(loc_m+1), dxk) for dxk in dx]) # 3-dimensional array (n-1-i) x (loc_m+1) x dim
                    if rotate_coord_sys:
                        # Rotate according to new system
                        d = d.dot(mrot)
                    if rotation_loc:
                        if alpha_loc:
                            a = t_angle * alpha_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1) # 2-dimensional array (n-1-i) x (loc_m+1)
                            # a = t_angle * np.asarray([alpha_loc_func(ddxk) for ddxk in ddx])
                            ca, sa = np.mean(np.cos(a), axis=1), np.mean(np.sin(a), axis=1)
                        else:
                            ca, sa = np.ones(d.shape[0]), np.zeros(d.shape[0])
                        if beta_loc:
                            a = t_angle * beta_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1) # 2-dimensional array (n-1-i) x (loc_m+1)
                            # a = t_angle * np.asarray([beta_loc_func(ddxk) for ddxk in ddx])
                            cb, sb = np.mean(np.cos(a), axis=1), np.mean(np.sin(a), axis=1)
                        else:
                            cb, sb = np.ones(d.shape[0]), np.zeros(d.shape[0])
                        if gamma_loc:
                            a = t_angle * gamma_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1) # 2-dimensional array (n-1-i) x (loc_m+1)
                            # a = t_angle * np.asarray([gamma_loc_func(ddxk) for ddxk in ddx])
                            cc, sc = np.mean(np.cos(a), axis=1), np.mean(np.sin(a), axis=1)
                        else:
                            cc, sc = np.ones(d.shape[0]), np.zeros(d.shape[0])
                        d = np.asarray([dk.dot(np.array(
                                        [[  cak * cck + sak * sbk * sck,  sak * cbk, - cak * sck + sak * sbk * cck],
                                         [- sak * cck + cak * sbk * sck,  cak * cbk,   sak * sck + cak * sbk * cck],
                                         [                    cbk * sck,      - sbk,                     cbk * cck]]))
                                     for (cak, sak, cbk, sbk, cck, sck, dk) in zip (ca, sa, cb, sb, cc, sc, d)])
                        # d = np.asarray([np.array(
                        #                 [[  cak * cck + sak * sbk * sck,  - sak * cck + cak * sbk * sck,  cbk * sck],
                        #                  [                    sak * cbk,                      cak * cbk,      - sbk],
                        #                  [- cak * sck + sak * sbk * cck,    sak * sck + cak * sbk * cck,  cbk * cck]]).dot(dk)
                        #              for (cak, sak, cbk, sbk, cck, sck, dk) in zip (ca, sa, cb, sb, cc, sc, d)])
                    if coord_loc == 1:
                        d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                        # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                    elif coord_loc == 2:
                        d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                        # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                    elif coord_loc == 3:
                        d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                        d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                        # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                        # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                    elif coord_loc == 4:
                        d[:, 2] = np.mean(coord3_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 2]
                        # d[:, 2] = np.asarray([np.mean(coord3_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 2]
                    elif coord_loc == 5:
                        d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                        d[:, 2] = np.mean(coord3_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 2]
                        # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                        # d[:, 2] = np.asarray([np.mean(coord3_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 2]
                    elif coord_loc == 6:
                        d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                        d[:, 2] = np.mean(coord3_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 2]
                        # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                        # d[:, 2] = np.asarray([np.mean(coord3_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 2]
                    elif coord_loc == 7:
                        d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                        d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                        d[:, 2] = np.mean(coord3_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 2]
                        # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                        # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                        # d[:, 2] = np.asarray([np.mean(coord3_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 2]
                    elif coord_loc == 8:
                        d = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d.T).T
                        # d = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d.T).T
                    elif coord_loc == 9:
                        d[:, (0, 1)] = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (0, 1)].T).T
                        # d[:, (0, 1)] = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (0, 1)].T).T
                    elif coord_loc == 10:
                        d[:, (0, 2)] = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (0, 2)].T).T
                        # d[:, (0, 2)] = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (0, 2)].T).T
                    elif coord_loc == 11:
                        d[:, (1, 2)] = (np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (1, 2)].T).T
                        # d[:, (1, 2)] = (np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (1, 2)].T).T
                    elif coord_loc == 12:
                        d[:, (0, 1)] = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (0, 1)].T).T
                        d[:, 2] = np.mean(coord3_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 2]
                        # d[:, (0, 1)] = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (0, 1)].T).T
                        # d[:, 2] = np.asarray([np.mean(coord3_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 2]
                    elif coord_loc == 13:
                        d[:, (0, 2)] = (np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (0, 2)].T).T
                        d[:, 1] = np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 1]
                        # d[:, (0, 2)] = (np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (0, 2)].T).T
                        # d[:, 1] = np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 1]
                    elif coord_loc == 14:
                        d[:, (1, 2)] = (np.mean(coord2_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, (1, 2)].T).T
                        d[:, 0] = np.mean(coord1_factor_loc_func(ddx.reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)*d[:, 0]
                        # d[:, (1, 2)] = (np.asarray([np.mean(coord2_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, (1, 2)].T).T
                        # d[:, 0] = np.asarray([np.mean(coord1_factor_loc_func(ddxk)) for ddxk in ddx])*d[:, 0]
                    ind = np.where(np.sum((d/hmax)**2, axis=1) <= 1.0)[0]
                    if len(ind) == 0:
                        continue
                    h.append(d[ind])
                    if w_loc:
                        wf = np.mean(w_factor_loc_func(ddx[ind].reshape(-1, 3)).reshape(-1, loc_m+1), axis=1)
                        # wf = np.asarray([np.mean(w_factor_loc_func(ddxk)) for ddxk in ddx[ind]])
                    g.append(wf * 0.5*(v[i] - v[i+1+ind])**2)
                    npair = npair + len(ind)
            else:
                for i in range(n-1):
                    jj = n-1-i
                    d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
                    if rotate_coord_sys:
                        # Rotate according to new system
                        d = d.dot(mrot)
                    if rotation_loc:
                        if alpha_loc:
                            a = t_angle * alpha_loc_func(x[i])[0]
                            ca, sa = np.cos(a), np.sin(a)
                        else:
                            ca, sa = np.ones(d.shape[0]), np.zeros(d.shape[0])
                        if beta_loc:
                            a = t_angle * beta_loc_func(x[i])[0]
                            cb, sb = np.cos(a), np.sin(a)
                        else:
                            cb, sb = np.ones(d.shape[0]), np.zeros(d.shape[0])
                        if gamma_loc:
                            a = t_angle * gamma_loc_func(x[i])[0]
                            cc, sc = np.cos(a), np.sin(a)
                        else:
                            cc, sc = np.ones(d.shape[0]), np.zeros(d.shape[0])
                        d = d.dot(np.array(
                                [[  ca * cc + sa * sb * sc,  sa * cb, - ca * sc + sa * sb * cc],
                                 [- sa * cc + ca * sb * sc,  ca * cb,   sa * sc + ca * sb * cc],
                                 [                 cb * sc,     - sb,                  cb * cc]]))
                    if coord_loc == 1:
                        d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                    elif coord_loc == 2:
                        d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                    elif coord_loc == 3:
                        d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                        d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                    elif coord_loc == 4:
                        d[:, 2] = coord3_factor_loc_func(x[i])[0]*d[:, 2]
                    elif coord_loc == 5:
                        d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                        d[:, 2] = coord3_factor_loc_func(x[i])[0]*d[:, 2]
                    elif coord_loc == 6:
                        d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                        d[:, 2] = coord3_factor_loc_func(x[i])[0]*d[:, 2]
                    elif coord_loc == 7:
                        d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                        d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                        d[:, 2] = coord3_factor_loc_func(x[i])[0]*d[:, 2]
                    elif coord_loc == 8:
                        d = coord1_factor_loc_func(x[i])[0]*d
                    elif coord_loc == 9:
                        d[:, (0, 1)] = coord1_factor_loc_func(x[i])[0]*d[:, (0, 1)]
                    elif coord_loc == 10:
                        d[:, (0, 2)] = coord1_factor_loc_func(x[i])[0]*d[:, (0, 2)]
                    elif coord_loc == 11:
                        d[:, (1, 2)] = coord2_factor_loc_func(x[i])[0]*d[:, (1, 2)]
                    elif coord_loc == 12:
                        d[:, (0, 1)] = coord1_factor_loc_func(x[i])[0]*d[:, (0, 1)]
                        d[:, 2] = coord3_factor_loc_func(x[i])[0]*d[:, 2]
                    elif coord_loc == 13:
                        d[:, (0, 2)] = coord1_factor_loc_func(x[i])[0]*d[:, (0, 2)]
                        d[:, 1] = coord2_factor_loc_func(x[i])[0]*d[:, 1]
                    elif coord_loc == 14:
                        d[:, (1, 2)] = coord2_factor_loc_func(x[i])[0]*d[:, (1, 2)]
                        d[:, 0] = coord1_factor_loc_func(x[i])[0]*d[:, 0]
                    ind = np.where(np.sum((d/hmax)**2, axis=1) <= 1.0)[0]
                    if len(ind) == 0:
                        continue
                    h.append(d[ind])
                    if w_loc:
                        wf = w_factor_loc_func(x[i])[0]
                    g.append(wf * 0.5*(v[i] - v[i+1+ind])**2)
                    npair = npair + len(ind)
        else:
            if rotate_coord_sys:
                # Rotate according to new system
                x = x.dot(mrot)
            for i in range(n-1):
                d = x[(i+1):] - x[i] # 2-dimensional array (n-1-i) x dim
                ind = np.where(np.sum((d/hmax)**2, axis=1) <= 1.0)[0]
                if len(ind) == 0:
                    continue
                h.append(d[ind])
                g.append(0.5*(v[i] - v[i+1+ind])**2)
                npair = npair + len(ind)
        if npair > 0:
            h = np.vstack(h)
            g = np.hstack(g)

    if npair == 0:
        err_msg = f'{fname}: no pair of points (in variogram cloud) for fitting'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # Define the function to optimize in a format compatible with curve_fit from scipy.optimize
    def func(d, *p):
        """
        Function whose p is the vector of parameters to optimize.

        Parameters
        ----------
        d : 1D array
            xdata, i.e. lags (h) from the variogram cloud where the current
            covariance model is evaluated

        p : 1D array
            current values of the parameters (floats) to optimize in the
            covariance model (parameters to optimized are identified with
            ielem_to_fit, key_to_fit, computed above)

        Returns
        -------
        v: 1D array
            evaluations of the current variogram model at `d`
        """
        for i, (iel, k, j) in enumerate(zip(ielem_to_fit, key_to_fit, ir_to_fit)):
            if k == 'r':
                cov_model_opt.elem[iel][1]['r'][j] = p[i]
                if link_range12 and j == 0:
                    cov_model_opt.elem[iel][1]['r'][1] = p[i]
            else:
                cov_model_opt.elem[iel][1][k] = p[i]
        if alpha_to_fit:
            cov_model_opt.alpha = p[-1-int(beta_to_fit)-int(gamma_to_fit)]
            cov_model_opt._mrot = None # reset attribute _mrot !
        if beta_to_fit:
            cov_model_opt.beta = p[-1-int(gamma_to_fit)]
            cov_model_opt._mrot = None # reset attribute _mrot !
        if gamma_to_fit:
            cov_model_opt.gamma = p[-1]
            cov_model_opt._mrot = None # reset attribute _mrot !
        return cov_model_opt(d, vario=True)

    # Optimize parameters with curve_fit: initial vector of parameters (p0) must be given
    #   because number of parameter to fit in function func is not known in its expression
    bounds = None
    if 'bounds' in kwargs.keys():
        bounds = kwargs['bounds']

    if 'p0' not in kwargs.keys():
        # add default p0 in kwargs
        p0 = np.ones(nparam)
        if bounds is not None:
            # adjust p0 to given bounds
            for i in range(nparam):
                if np.isinf(bounds[0][i]):
                    if np.isinf(bounds[1][i]):
                        p0[i] = 1.
                    else:
                        p0[i] = bounds[1][i]
                elif np.isinf(bounds[1][i]):
                    p0[i] = bounds[0][i]
                else:
                    p0[i] = 0.5*(bounds[0][i]+bounds[1][i])
        kwargs['p0'] = p0
    else:
        if len(kwargs['p0']) != nparam:
            err_msg = f'{fname}: length of `p0` and number of parameters to fit differ'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    # Fit with curve_fit
    try:
        popt, pcov = scipy.optimize.curve_fit(func, h, g, **kwargs)
        if rotate_coord_sys:
            # Restore alpha, beta, gamma
            cov_model_opt.alpha = alpha_copy
            cov_model_opt.beta = beta_copy
            cov_model_opt.gamma = gamma_copy
    except:
        err_msg = f'{fname}: fitting covariance model failed'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    if make_plot:
        # plt.suptitle(textwrap.TextWrapper(width=50).fill(s))
        s = [f'Vario opt.: alpha={cov_model_opt.alpha}, beta={cov_model_opt.beta}, gamma={cov_model_opt.gamma}'] + [f'{el}' for el in cov_model_opt.elem]
        cov_model_opt.plot_model3d_volume(vario=True, text='\n'.join(s), text_kwargs={'font_size':12})

    return cov_model_opt, popt
# ----------------------------------------------------------------------------

# ============================================================================
# Utility function
# ============================================================================
# ----------------------------------------------------------------------------
def values_to_mean_and_err_std(v, v_min=np.nan, v_max=np.nan, p=0.95, def_shift=1.e-5, logger=None):
    """
    Computes a central value and an error standard deviation from an ensemble of values.

    Given an ensemble of values `v`, a lower bound `v_min` and/or an upper 
    bound `v_max`, the central value `v_mean` is set to:
    
    - the mean of values `v` that are within the bounds
    - or, if no value is within the bounds: the mean of the two bounds if \
    both bounds are given, or the value of the unique given bound plus \
    (resp. minus) `def_shift` if only the lower bound (`v_min`) (resp. \
    upper bound (`v_max`)) is given, or zero if no bound is given

    Then, the error standard deviation `v_err_std` is set such that
    the Gaussian distribution of mean `v_mean` and standard deviation 
    `v_err_std` has a probability `p/2` at least to be between `v_mean` and 
    the lower bound (`v_min`), resp. between `v_mean` and the upper bound
    `v_max`. If no bound is given, `v_err_std` is set to the standard deviation 
    of the values `v`.
    
    Parameters
    ----------
    v : 1D array-like of floats
        values
    
    v_min : float, default: numpy.nan
        lower bound (if not given or `numpy.nan`: no lower bound)
    
    v_max : float, default: numpy.nan
        upper bound (if not given or `numpy.nan`: no upper bound)
    
    p : float, default: 0.95
        probability used to set the output error standard deviation 
        (see above)

    def_shift : float, default: 1.e-5
        shift used to set the output central value if no value is within
        the bounds and only one bound is given (see above)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)
    
    Returns
    -------
    v_mean : float
        central value (see above)
    
    v_err_std : float
        error standard deviation (see above)
    """
    fname = 'values_to_mean_and_err_std'

    if v is None:
        err_msg = f'{fname}: `v` is None'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
    if len(v) == 0:
        err_msg = f'{fname}: `v` has no value'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    if v_min is None or np.isnan(v_min):
        v_min = -np.inf

    if v_max is None or np.isnan(v_max):
        v_max = np.inf    
    
    if v_min >= v_max:
        err_msg = f'{fname}: `v_min` should be less than `v_max`'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)
    
    t = scipy.stats.norm.ppf((1.0+p)/2.0)

    if np.isinf(v_min):
        if np.isinf(v_max):
            # no lower bound, no upper bound
            v_mean = np.mean(v)
            v_err_std = np.std(v)

        else:
            # only upper bound
            ind = v <= v_max
            if len(ind):
                v_mean = np.mean(v[ind])
            else:
                v_mean = v_max - def_shift
            v_err_std = (v_max - v_mean) / t

    else:
        if np.isinf(v_max):
            # only lower bound
            ind = v >= v_min
            if len(ind):
                v_mean = np.mean(v[ind])
            else:
                v_mean = v_min + def_shift
            v_err_std = (v_mean - v_min) / t

        else:
            # lower bound and upper bound
            ind = np.all((v >= v_min, v <= v_max), axis=0)
            if len(ind):
                v_mean = np.mean(v[ind])
            else:
                v_mean = 0.5 *(v_min + v_max)
            v_err_std = min(v_mean - v_min, v_max - v_mean) / t

    return v_mean, v_err_std
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def eval_at_points_1D(f, pts, nx, sx, ox, return_float_if_unique=True, logger=None):
    """
    Evaluates `f` at points - 1D.

    Parameters
    ----------
    f : function (callable), or array-like of floats, or float
        function or value(s) in grid in input:

        - if a function: function of 1 argument that returns a value for each location \
        given by their coordinate in argument
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), values at grid cells; note the shape of the \
        array is `(nx,)`
        - if a float: same value at every grid cell
            
    pts : 2D array of floats of shape (n, 1)
        points at which the input fuction or value(s) has to be interpolated

    nx : int
        number of grid cells along x axis

    sx : float
        cell size along x axis

    ox : float
        origin of the grid along x axis (x coordinate of cell border)

        Note: `(ox, )` is the "bottom-lower-left" corner of the grid
    
    return_float_if_unique : bool, default: True
        if `True` and if all output values are identical, then a float 
        (the unique output value) is returned; 
        otherwise: an array of shape `(n,)` is returned

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)
    
    Returns
    -------
    f_pts : array of floats, or float
        array of output values at point `pts` of shape `(n,)`;
        if `return_float_if_unique=True` and if all output values are identical, 
        then a float (the unique output value) is returned
    """
    fname = 'eval_at_points_1D'

    if f is None:
        f_pts = None

    elif callable(f):
        f_pts = f(pts[:, 0])
        if return_float_if_unique and np.allclose(f_pts, f_pts[0]):
            f_pts = f_pts[0]

    else:
        f_grid = np.asarray(f, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if f_grid.size == 1:
            if return_float_if_unique:
                f_pts = f_grid[0]
            else:
                f_pts = f_grid[0] * np.ones(pts.shape[0])
        
        elif f_grid.size == nx:
            f_pts = img.Img_interp_func(img.Img(nx, 1, 1, sx, 1.0, 1.0, ox, 0.0, 0.0, nv=1, val=f_grid, logger=logger), iy=0, iz=0, logger=logger)(pts)
            if return_float_if_unique and np.allclose(f_pts, f_pts[0]):
                f_pts = f_pts[0]
        else:
            err_msg = f'{fname}: size of `f` is not valid'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
            
    return f_pts
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def eval_at_points_2D(f, pts, nx, ny, sx, sy, ox, oy, return_float_if_unique=True, logger=None):
    """
    Evaluates `f` at points - 2D.

    Parameters
    ----------
    f : function (callable), or array-like of floats, or float
        function or value(s) in grid in input:

        - if a function: function of 2 arguments that returns a value for each location \
        given by their coordinates in argument
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), values at grid cells; note the shape of the \
        array is `(ny, nx)`
        - if a float: same value at every grid cell
            
    pts : 2D array of floats of shape (n, 2)
        points at which the input fuction or value(s) has to be interpolated

    nx : int
        number of grid cells along x axis

    ny : int
        number of grid cells along y axis

    sx : float
        cell size along x axis

    sy : float
        cell size along y axis

    ox : float
        origin of the grid along x axis (x coordinate of cell border)

    oy : float
        origin of the grid along y axis (y coordinate of cell border)

        Note: `(ox, oy)` is the "bottom-lower-left" corner of the grid
    
    return_float_if_unique : bool, default: True
        if `True` and if all output values are identical, then a float 
        (the unique output value) is returned; 
        otherwise: an array of shape `(n,)` is returned
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    f_pts : array of floats, or float
        array of output values at point `pts` of shape `(n,)`;
        if `return_float_if_unique=True` and if all output values are identical, 
        then a float (the unique output value) is returned
    """
    fname = 'eval_at_points_2D'

    if f is None:
        f_pts = None

    elif callable(f):
        f_pts = f(*pts.T)
        if return_float_if_unique and np.allclose(f_pts, f_pts[0]):
            f_pts = f_pts[0]

    else:
        f_grid = np.asarray(f, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if f_grid.size == 1:
            if return_float_if_unique:
                f_pts = f_grid[0]
            else:
                f_pts = f_grid[0] * np.ones(pts.shape[0])
        
        elif f_grid.size == nx*ny:
            f_pts = img.Img_interp_func(img.Img(nx, ny, 1, sx, sy, 1.0, ox, oy, 0.0, nv=1, val=f_grid, logger=logger), iz=0, logger=logger)(pts)
            if return_float_if_unique and np.allclose(f_pts, f_pts[0]):
                f_pts = f_pts[0]
        else:
            err_msg = f'{fname}: size of `f` is not valid'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
            
    return f_pts
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def eval_at_points_3D(f, pts, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=None):
    """
    Evaluates `f` at points - 3D.

    Parameters
    ----------
    f : function (callable), or array-like of floats, or float
        function or value(s) in grid in input:

        - if a function: function of 3 arguments that returns a value for each location \
        given by their coordinates in argument
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), values at grid cells; note the shape of the \
        array is `(nz, ny, nx)`
        - if a float: same value at every grid cell
            
    pts : 2D array of floats of shape (n, 3)
        points at which the input fuction or value(s) has to be interpolated

    nx : int
        number of grid cells along x axis

    ny : int
        number of grid cells along y axis

    nz : int
        number of grid cells along z axis

    sx : float
        cell size along x axis

    sy : float
        cell size along y axis

    sz : float
        cell size along z axis

    ox : float
        origin of the grid along x axis (x coordinate of cell border)

    oy : float
        origin of the grid along y axis (y coordinate of cell border)

    oz : float
        origin of the grid along z axis (z coordinate of cell border)

        Note: `(ox, oy, oz)` is the "bottom-lower-left" corner of the grid
    
    return_float_if_unique : bool, default: True
        if `True` and if all output values are identical, then a float 
        (the unique output value) is returned; 
        otherwise: an array of shape `(n,)` is returned
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    f_pts : array of floats, or float
        array of output values at point `pts` of shape `(n,)`;
        if `return_float_if_unique=True` and if all output values are identical, 
        then a float (the unique output value) is returned
    """
    fname = 'eval_at_points_3D'

    if f is None:
        f_pts = None

    elif callable(f):
        f_pts = f(*pts.T)
        if return_float_if_unique and np.allclose(f_pts, f_pts[0]):
            f_pts = f_pts[0]

    else:
        f_grid = np.asarray(f, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if f_grid.size == 1:
            if return_float_if_unique:
                f_pts = f_grid[0]
            else:
                f_pts = f_grid[0] * np.ones(pts.shape[0])
        
        elif f_grid.size == nx*ny*nz:
            f_pts = img.Img_interp_func(img.Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv=1, val=f_grid, logger=logger), logger=logger)(pts)
            if return_float_if_unique and np.allclose(f_pts, f_pts[0]):
                f_pts = f_pts[0]
        else:
            err_msg = f'{fname}: size of `f` is not valid'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
            
    return f_pts
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def eval_in_grid_1D(f, nx, sx, ox, return_float_if_unique=True, logger=None):
    """
    Evaluates `f` in grid - 1D.

    Parameters
    ----------
    f : function (callable), or array-like of floats, or float
        function or value(s) in grid in input:

        - if a function: function of 1 argument that returns a value for each location \
        given by their coordinate in argument
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), values at grid cells; note the shape of the \
        array is `(nx,)`
        - if a float: same value at every grid cell
            
    nx : int
        number of grid cells along x axis

    sx : float
        cell size along x axis

    ox : float
        origin of the grid along x axis (x coordinate of cell border)

        Note: `(ox, )` is the "bottom-lower-left" corner of the grid
       
    return_float_if_unique : bool, default: True
        if `True` and if all output values are identical, then a float 
        (the unique output value) is returned; 
        otherwise: an array of shape `(nx,)` is returned
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    f_grid : array of floats, or float
        array of output values in the grid (cell centers), of shape `(nx,)`,
        `f_grid[ix]` is the value at the grid cell of index `ix` along x-axis;
        if `return_float_if_unique=True` and if all output values are identical, 
        then a float (the unique output value) is returned
    """
    fname = 'eval_in_grid_1D'

    if f is None:
        f_grid = None

    elif callable(f):
        xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
        f_grid = f(xi)
        if return_float_if_unique and np.allclose(f_grid, f_grid[0]):
            f_grid = f_grid[0]

    else:
        f_grid = np.asarray(f, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if f_grid.size == 1:
            if return_float_if_unique:
                f_grid = f_grid[0]
            else:
                f_grid = f_grid[0] * np.ones(nx)
        elif f_grid.size == nx:
            f_grid = f_grid.reshape(-1)
            if return_float_if_unique and np.allclose(f_grid, f_grid[0]):
                f_grid = f_grid[0]
        else:
            err_msg = f'{fname}: size of `f` is not valid'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
            
    return f_grid
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def eval_in_grid_2D(f, nx, ny, sx, sy, ox, oy, return_float_if_unique=True, logger=None):
    """
    Evaluates `f` in grid - 2D.

    Parameters
    ----------
    f : function (callable), or array-like of floats, or float
        function or value(s) in grid in input:

        - if a function: function of 2 arguments that returns a value for each location \
        given by their coordinates in argument
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), values at grid cells; note the shape of the \
        array is `(ny, nx)`
        - if a float: same value at every grid cell
            
    nx : int
        number of grid cells along x axis

    ny : int
        number of grid cells along y axis

    sx : float
        cell size along x axis

    sy : float
        cell size along y axis

    ox : float
        origin of the grid along x axis (x coordinate of cell border)

    oy : float
        origin of the grid along y axis (y coordinate of cell border)

        Note: `(ox, oy)` is the "bottom-lower-left" corner of the grid
    
    return_float_if_unique : bool, default: True
        if `True` and if all output values are identical, then a float 
        (the unique output value) is returned; 
        otherwise: an array of shape `(ny, nx)` is returned
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    f_grid : array of floats, or float
        array of output values in the grid (cell centers), of shape `(ny, nx)`,
        `f_grid[iy, ix]` is the value at the grid cell of 
        indices `ix` along x-axis, `iy` along y-axis;
        if `return_float_if_unique=True` and if all output values are identical, 
        then a float (the unique output value) is returned
    """
    fname = 'eval_in_grid_2D'

    if f is None:
        f_grid = None

    elif callable(f):
        xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
        yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
        yyi, xxi = np.meshgrid(yi, xi, indexing='ij')
        f_grid = f(xxi, yyi)
        if return_float_if_unique and np.allclose(f_grid, f_grid[0, 0]):
            f_grid = f_grid[0, 0]

    else:
        f_grid = np.asarray(f, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if f_grid.size == 1:
            if return_float_if_unique:
                f_grid = f_grid[0]
            else:
                f_grid = f_grid[0] * np.ones((ny, nx))
        elif f_grid.size == nx*ny:
            f_grid = f_grid.reshape(ny, nx)
            if return_float_if_unique and np.allclose(f_grid, f_grid[0, 0]):
                f_grid = f_grid[0, 0]
        else:
            err_msg = f'{fname}: size of `f` is not valid'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
            
    return f_grid
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def eval_in_grid_3D(f, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=None):
    """
    Evaluates `f` in grid - 3D.

    Parameters
    ----------
    f : function (callable), or array-like of floats, or float
        function or value(s) in grid in input:

        - if a function: function of 3 arguments that returns a value for each location \
        given by their coordinates in argument
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), values at grid cells; note the shape of the \
        array is `(nz, ny, nx)`
        - if a float: same value at every grid cell
   
    nx : int
        number of grid cells along x axis

    ny : int
        number of grid cells along y axis

    nz : int
        number of grid cells along z axis

    sx : float
        cell size along x axis

    sy : float
        cell size along y axis

    sz : float
        cell size along z axis

    ox : float
        origin of the grid along x axis (x coordinate of cell border)

    oy : float
        origin of the grid along y axis (y coordinate of cell border)

    oz : float
        origin of the grid along z axis (z coordinate of cell border)

        Note: `(ox, oy, oz)` is the "bottom-lower-left" corner of the grid
    
    return_float_if_unique : bool, default: True
        if `True` and if all output values are identical, then a float 
        (the unique output value) is returned; 
        otherwise: an array of shape `(nz, ny, nx)` is returned
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    f_grid : array of floats, or float
        array of output values in the grid (cell centers), of shape `(nz, ny, nx)`,
        `f_grid[iz, iy, ix]` is the value at the grid cell of 
        indices `ix` along x-axis, `iy` along y-axis, `iz` along z-axis;
        if `return_float_if_unique=True` and if all output values are identical, 
        then a float (the unique output value) is returned
    """
    fname = 'eval_in_grid_3D'

    if f is None:
        f_grid = None

    elif callable(f):
        xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
        yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
        zi = oz + sz*(0.5+np.arange(nz)) # z-coordinate of cell center
        zzi, yyi, xxi = np.meshgrid(zi, yi, xi, indexing='ij')
        f_grid = f(xxi, yyi, zzi)
        if return_float_if_unique and np.allclose(f_grid, f_grid[0, 0, 0]):
            f_grid = f_grid[0, 0, 0]

    else:
        f_grid = np.asarray(f, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if f_grid.size == 1:
            if return_float_if_unique:
                f_grid = f_grid[0]
            else:
                f_grid = f_grid[0] * np.ones((nz, ny, nx))
        elif f_grid.size == nx*ny*nz:
            f_grid = f_grid.reshape(nz, ny, nx)
            if return_float_if_unique and np.allclose(f_grid, f_grid[0, 0, 0]):
                f_grid = f_grid[0, 0, 0]
        else:
            err_msg = f'{fname}: size of `f` is not valid'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
            
    return f_grid
# ----------------------------------------------------------------------------

# ============================================================================
# Simple and ordinary kriging and cross validation by leave-one-out (loo)
# ============================================================================
# ----------------------------------------------------------------------------
def krige(
        x, v, xu, cov_model,
        v_err_std=0.0,
        method='ordinary_kriging',
        mean_x=None,
        mean_xu=None,
        var_x=None,
        var_xu=None,
        alpha_xu=None,
        beta_xu=None,
        gamma_xu=None,
        cov_model_non_stationarity_xu_list=None,
        use_unique_neighborhood=False,
        searchRadius=None,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        pid=None,
        verbose=0,
        logger=None):
    """
    Interpolates data by kriging at given location(s).

    This function performs kriging interpolation at locations `xu` of the values
    `v` measured at locations `x`.

    Parameters
    ----------
    x : 2D array of floats of shape (n, d)
        data points locations, with n the number of data points and d the space
        dimension (1, 2, or 3), each row of `x` is the coordinatates of one data
        point; note: for data in 1D (`d=1`), 1D array of shape `(n,)` is accepted
        for `n` data points

    v : 1D array of floats of shape (n,)
        data points values, with n the number of data points, `v[i]` is the data
        value at location `x[i]`

    xu : 2D array of floats of shape (nu, d)
        points locations where the interpolation has to be done, with nu the
        number of points and d the space dimension (1, 2, or 3, same as for `x`),
        each row of `xu` is the coordinatates of one point;
        note: for data in 1D (`d=1`), 1D array of shape `(nu,)` is accepted 
        for `nu` points

    cov_model : :class:`CovModel1D` or :class:`CovModel2D` or :class:`CovModel3D`
        covariance model in 1D, 2D, or 3D, in same dimension as dimension of
        points (d), i.e.:

        - :class:`CovModel1D` for data in 1D (d=1)
        - :class:`CovModel2D` for data in 2D (d=2)
        - :class:`CovModel3D` for data in 3D (d=3)

        or

        - :class:`CovModel1D` interpreted as an omni-directional covariance model \
        whatever dimension of points (d);

        note: the covariance model must be stationary, however, non stationarity is
        handled: 

        - local rotation by specifying `alpha_xu` (in 2D or 3D), `beta_xu` (in 3D), `gamma_xu` (in 3D)
        - other non-stationarities by specifying `cov_model_non_stationarity_xu_list` (see below)

    v_err_std : 1D array of floats of shape (n,), or float, default: 0.0
        standard deviation of error at data points, with n the number of data points; 
        if `v_err_std` is a float, the same value is used for all data points; 
        this means that at location x[i], the data value is considered as in a Gaussian 
        distribution of mean `v[i]` and standard deviation `v_err_std[i]`

    method : str {'simple_kriging', 'ordinary_kriging'}, default: 'ordinary_kriging'
        type of kriging;
        note: if `method='ordinary_kriging'`, the parameters
        `mean_x`, `mean_xu`, `var_x`, `var_xu` are not used

    mean_x : 1D array-like of floats, or float, optional
        kriging mean value at data points `x`

        - if `mean_x` is a float, the same value is considered for any point
        - if `mean_x=None` (default): the mean of data values, i.e. mean of `v`, \
        is considered for any point

        note: if `method=ordinary_kriging`, parameter `mean_x` is ignored

    mean_xu : 1D array-like of floats, or float, optional
        kriging mean value at points `xu`

        - if `mean_xu` is a float, the same value is considered for any point
        - if `mean_xu=None` (default): the value `mean_x` (assumed to be a single \
        float) is considered for any point

        note: if `method=ordinary_kriging`, parameter `mean_xu` is ignored

    var_x : 1D array-like of floats, or float, optional
        kriging variance value at data points `x`

        - if `var_x` is a float, the same value is considered for any point
        - if `var_x=None` (default): not used  (use of covariance model only)

        note: if `method=ordinary_kriging`, parameter `var_x` is ignored

    var_xu : 1D array-like of floats, or float, optional
        kriging variance value at points `xu`

        - if `var_xu` is a float, the same value is considered for any point
        - if `var_xu=None` (default): not used  (use of covariance model only)

        note: if `method=ordinary_kriging`, parameter `var_xu` is ignored

    alpha_xu : 1D array-like of floats, or float, optional
        azimuth angle in degrees at points `xu`

        - if `alpha_xu` is a float, the same value is considered for any point
        - if `alpha_xu=None` (default): `alpha_xu=0.0` is used for any point

        note: `alpha_xu` is ignored if the covariance model is in 1D

    beta_xu : 1D array-like of floats, or float, optional
        dip angle in degrees at points `xu`

        - if `beta_xu` is a float, the same value is considered for any point
        - if `beta_xu=None` (default): `beta_xu=0.0` is used for any point

        note: `beta_xu` is ignored if the covariance model is in 1D or 2D

    gamma_xu : 1D array-like of floats, or float, optional
        dip angle in degrees at points `xu`

        - if `gamma_xu` is a float, the same value is considered for any point
        - if `gamma_xu=None` (default): `gamma_xu=0.0` is used for any point

        note: `gamma_xu` is ignored if the covariance model is in 1D or 2D

    cov_model_non_stationarity_xu_list : list, optional
        list to set non-stationarities in covariance model; each entry must be
        a tuple (or list) `cm_ns` of length 2 or 3 with:

        - `cm_ns[0]`: str: the name of the method of `cov_model` to be applied
        - `cm_ns[1]`: 1D array-like of floats, or float: \
        used to set the main parameter passed to the method:
            - if array-like: its size must be equal to `nu`, \
            (the array is reshaped if needed), values at points `xu`
            - if a float: same value at all points `xu`
        - `cm_ns[2]`: dict, optional: keyworkds arguments to be passed to the method

        Examples (with the parameter `arg` is set from `val`)
        
        - `('multiply_w', val)` will apply `cov_model.multiply_w(arg)`; 
            this multipies the weight contribution of every elementary contribution of the 
            covariance model
        - `('multiply_w', val, {'elem_ind':0})` will apply `cov_model.multiply_w(arg, elem_ind=0)`;
            this multipies the weight contribution of the elementary contribution of index 0 of the 
            covariance model
        - `('multiply_r', val)` will apply `cov_model.multiply_r(arg)`;
            this multipies the range in all direction of every elementary contribution of the
            covariance model
        - `('multiply_r', val, {'r_ind':0})` will apply `cov_model.multiply_r(arg, r_ind=0)`;
            this multipies the range in the first main direction (index 0) of every elementary 
            contribution of the covariance model
        - `('multiply_r', val, {'r_ind':0, 'elem_ind':0})` will apply `cov_model.multiply_r(arg, r_ind=0, elem_ind=0)`;
            this multipies the range in the first main direction (index 0) of the elementary 
            contribution of index 0 of the covariance model

    use_unique_neighborhood : bool, default: False
        indicates if a unique neighborhood is used:

        - if True: all data points are taken into account, and the kriging matrix \
        is computed once; the parameters `searchRadius`, `searchRadiusRelative`, \
        `nneighborMax` are not used, \
        and  `alpha_xu`, `beta_xu`, `gamma_xu` must be `None` or constant \
        and any non-stationarity set in `cov_model_non_stationarity_xu_list` must be constant \
        (i.e. `cm_ns[1]` must be constant for each entry `cm_ns` in `cov_model_non_stationarity_xu_list`)
        - if False: only data points within a search neighborhood (ellipsoid) are \
        taken into account according to `searchRadius`, `searchRadiusRelative`, `nneighborMax`

    searchRadius : float, optional
        if specified, i.e. not `None`: radius of the search neighborhood (ellipsoid
        with same radii along each axis), i.e. the data points at distance to the 
        estimated point greater than `searchRadius` are not taken into account 
        in the kriging system; if `searchRadius` is not `None`, then 
        `searchRadiusRelative` is not used;
        by default (`searchRadius=None`): `searchRadiusRelative` is used to 
        define the search ellipsoid;

    searchRadiusRelative : float, default: 1.2
        used only if `searchRadius` is `None`;
        indicates how the search ellipsoid is limited (should be positive): let
        r_i be the ranges of the covariance model along its main axes, when
        estimating/simulating a cell x, a cell y is taken into account iff it is
        within the ellipsoid centered at x of half axes equal to
        `searchRadiusRelative` * r_i; 
        (note that the distances to the central node are computed in the axes 
        sytem supporting the covariance model and accounting for anisotropy given 
        by the ranges)

    nneighborMax : int, default: 12
        maximal number of neighbors (data points) taken into account in the
        kriging system; the data points the closest to the estimated points are
        taken into account;
        note: if `nneighborMax=None` or `nneighborMax<0`, then `nneighborMax` is
        set to the number of data points

    pid : int, optional
        process id of the caller (used with multiprocessing)

    verbose : int, default: 0
        verbose mode, higher implies more printing (info)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    vu : 1D array of shape (nu,)
        kriging estimates at points `xu`;
        note: `vu[j]=numpy.nan` if there is no data point in the neighborhood
        of `xu[j]` (see parameters `searchRadius`, `searchRadiusRelative`, `nneighborMax`)

    vu_std : 1D array of shape (nu,)
        kriging standard deviations at points `xu`;
        note: `vu_std[j]=numpy.nan` if there is no data point in the neighborhood
        of `xu[j]` (see parameters `searchRadius`, `searchRadiusRelative`, `nneighborMax`)
    """
    fname = 'krige'
    if pid is not None:
        fname = f'{fname} [pid={pid}]'

    # Prevent calculation if covariance model is not stationary
    if not cov_model.is_stationary():
        err_msg = f'{fname}: `cov_model` is not stationary: {fname} cannot be applied (use the other paramters for non-stationary covariance)'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # Get dimension from x (d) and number of data points (n) and size of v
    if x is None:
        n = 0
        d = 0
        if v is not None:
            err_msg = f'{fname}: `x` is None but `v` is not None'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
    
    else:
        x = np.asarray(x)
        # Get dimension (d) from x
        if x.ndim == 1:
            # x is a 1-dimensional array
            x = x.reshape(-1, 1)
            d = 1
        else:
            # x is a 2-dimensional array
            d = x.shape[1]

        n = x.shape[0]

        # Check size of v
        if v is None:
            err_msg = f'{fname}: `x` is not None but `v` is None'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

        v = np.asarray(v).reshape(-1)
        if v.size != n:
            err_msg = f'{fname}: size of `v` is not valid'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    # Set variance of data error (from standard deviation)
    if v_err_std is None:
        v_err_std = 0.0
    v_err_var = np.asarray(v_err_std, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
    if v_err_var.size == 1:
        v_err_var = v_err_var[0] * np.ones(n)
    elif v_err_var.size != n:
        err_msg = f'{fname}: size of `v_err_std` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    v_err_var = v_err_var * v_err_var

    # Get dimension from xu (du) and number of unknown points (nu)
    if xu is None:
        nu = 0
        du = 0
    else:
        xu = np.asarray(xu)
        # Get dimension (du) from xu
        if xu.ndim == 1:
            # xu is a 1-dimensional array
            xu = xu.reshape(-1, 1)
            du = 1
        else:
            # xu is a 2-dimensional array
            du = xu.shape[1]

        nu = xu.shape[0]

    if nu == 0:
        vu, vu_std = np.array([], dtype='float'), np.array([], dtype='float')
        return vu, vu_std

    # Here: nu > 0

    if n > 0:
        # Check dimension of x and xu
        if d != du:
            err_msg = f'{fname}: `x` and `xu` do not have the same dimension'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    else: # n == 0
        d = du # set d to du if no data point

    # Check that all data points (locations) are distinct
    for i in range(1, n):
        if np.any(np.isclose(np.sum((x[:i]-x[i])**2, axis=1), 0.0)):
            err_msg = f'{fname}: `x` contains duplicated entries'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    # Check dimension of cov_model and set if used as omni-directional model
    if isinstance(cov_model, CovModel1D):
        omni_dir = True
    else:
        if cov_model.__class__.__name__ != f'CovModel{d}D':
            err_msg = f'{fname}: `cov_model` dimension is incompatible with dimension of points'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

        omni_dir = False

    # Covariance function and value at 0
    cov_func = cov_model.func() # covariance function
    if omni_dir:
        # covariance model in 1D is used
        cov0 = cov_func(0.)[0] # covariance function at origin (lag=0)
    else:
        cov0 = cov_func(np.zeros(d))[0] # covariance function at origin (lag=0)

    # Method and mean, var
    if method == 'simple_kriging':
        ordinary_kriging = False
        if n == 0:
            mean_x = np.array([], dtype='float')
        else:
            if mean_x is None:
                mean_x = np.mean(v) * np.ones(n)
            else:
                mean_x = np.asarray(mean_x, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                if mean_x.size == 1:
                    mean_x = mean_x * np.ones(n)
                elif mean_x.size != n:
                    err_msg = f'{fname}: size of `mean_x` is not valid'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)
                
        if mean_xu is None:
            if n == 0:
                mean_xu = np.zeros(nu)
            else:
                mean_xu = np.mean(v) * np.ones(nu)
        else:
            mean_xu = np.asarray(mean_xu, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if mean_xu.size == 1:
                mean_xu = mean_xu * np.ones(nu)
            elif mean_xu.size != nu:
                err_msg = f'{fname}: size of `mean_xu` is not valid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

        if n == 0:
            if var_xu is None:
                var_x = None
            else:
                var_x = np.array([], dtype='float')

        if (var_x is None and var_xu is not None) or (var_x is not None and var_xu is None):
            err_msg = f'{fname}: `var_x` and `var_xu` must both be specified'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

        if var_x is not None:
            var_x = np.asarray(var_x, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if var_x.size == 1:
                var_x = var_x * np.ones(n)
            elif var_x.size != n:
                err_msg = f'{fname}: size of `var_x` is not valid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

            varUpdate_x = np.sqrt(var_x/cov0)

        if var_xu is not None:
            var_xu = np.asarray(var_xu, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if var_xu.size == 1:
                var_xu = var_xu * np.ones(nu)
            elif var_xu.size != nu:
                err_msg = f'{fname}: size of `var_xu` is not valid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

            varUpdate_xu = np.sqrt(var_xu/cov0)

    elif method == 'ordinary_kriging':
        if verbose > 0:
            if mean_x is not None:
                if logger:
                    logger.warning(f"{fname}: `mean_x` is ignored with `method='ordinary_kriging'`")
                else:
                    print(f"{fname}: WARNING: `mean_x` is ignored with `method='ordinary_kriging'`")
            if mean_xu is not None:
                if logger:
                    logger.warning(f"{fname}: `mean_xu` is ignored with `method='ordinary_kriging'`")
                else:
                    print(f"{fname}: WARNING: `mean_xu` is ignored with `method='ordinary_kriging'`")
            if var_x is not None:
                if logger:
                    logger.warning(f"{fname}: `var_x` is ignored with `method='ordinary_kriging'`")
                else:
                    print(f"{fname}: WARNING: `var_x` is ignored with `method='ordinary_kriging'`")
            if var_xu is not None:
                if logger:
                    logger.warning(f"{fname}: `var_xu` is ignored with `method='ordinary_kriging'`")
                else:
                    print(f"{fname}: WARNING: `var_xu` is ignored with `method='ordinary_kriging'`")

        ordinary_kriging = True
        mean_x, mean_xu, var_x, var_xu = None, None, None, None

    else:
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    if n == 0:
        # Treat the special case with no data point
        if method == 'simple_kriging':
            vu = mean_xu

            if var_xu is not None:
                if cov_model_non_stationarity_xu_list is not None:
                    if not isinstance(cov_model_non_stationarity_xu_list, list):
                        err_msg = f'{fname}: `cov_model_non_stationarity_xu_list` must be a list if not `None`'
                        if logger: logger.error(err_msg)
                        raise CovModelError(err_msg)
                    for i, cm_ns_xu in enumerate(cov_model_non_stationarity_xu_list):
                        if not hasattr(cm_ns_xu, '__len__') or len(cm_ns_xu) not in (2, 3):
                            err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_xu_list` not valid: should be a tuple or list of length 2 or 3 containing: method name (str), val[, kwds (dict)]'
                            if logger: logger.error(err_msg)
                            raise CovModelError(err_msg)
                        method_name = cm_ns_xu[0]
                        if method_name == 'multiply_w':
                            err_msg = f'{fname}: covariance model with non stationary sill (method `multiply_w`) cannot be used if `var_xu` is not `None`'
                            if logger: logger.error(err_msg)
                            raise CovModelError(err_msg)
                
                vu_std = np.sqrt(cov0) * varUpdate_xu

            elif cov_model_non_stationarity_xu_list is not None:
                if not isinstance(cov_model_non_stationarity_xu_list, list):
                    err_msg = f'{fname}: `cov_model_non_stationarity_xu_list` must be a list if not `None`'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)
                
                # WORK ON A COPY OF COVARIANCE MODEL (IN CASE IT IS ADAPTED)!
                cov_model = copyCovModel(cov_model) 
                cov_model_has_changed = False
                adapt_cov_model_ind = []
                for i, cm_ns_xu in enumerate(cov_model_non_stationarity_xu_list):
                    if not hasattr(cm_ns_xu, '__len__') or len(cm_ns_xu) not in (2, 3):
                        err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_xu_list` not valid: should be a tuple or list of length 2 or 3 containing: method name (str), val[, kwds (dict)]'
                        if logger: logger.error(err_msg)
                        raise CovModelError(err_msg)
                    if cm_ns_xu[0] != 'multiply_w':
                        continue
                    val = np.asarray(cm_ns_xu[1], dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                    if val.size != 1:
                        if val.size != nu:
                            err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_xu_list`, `(method_name, val[, kwds])`: size of `val` not valid'
                            if logger: logger.error(err_msg)
                            raise CovModelError(err_msg)
                        if np.allclose(val, val[0]):
                            val = val[:1] # of size 1
                        else:
                            adapt_cov_model_ind.append(i)
                    if val.size == 1:
                        if len(cm_ns_xu) == 3:
                            kwds = cm_ns_xu[2]
                        else:
                            kwds = {}
                        try:
                            cov_model.multiply_w(val[0], logger=logger, **kwds)
                        except:
                            err_msg = f'{fname}: cannot apply non-stationarity (at points) for covariance model (check parameter `cov_model_non_stationarity_xu_list`)'
                            if logger: logger.error(err_msg)
                            raise CovModelError(err_msg)
                        cov_model_has_changed = True

                if cov_model_has_changed:
                    # Update - Covariance function and value at 0
                    cov_func = cov_model.func() # covariance function
                    if omni_dir:
                        # covariance model in 1D is used
                        cov0 = cov_func(0.)[0] # covariance function at origin (lag=0)
                    else:
                        cov0 = cov_func(np.zeros(d))[0] # covariance function at origin (lag=0)

                if len(adapt_cov_model_ind):
                    cov_model_base = copyCovModel(cov_model)
                    vu_std = np.zeros(nu)
                    for k in range(nu):
                        cov_model = copyCovModel(cov_model_base)
                        for i in adapt_cov_model_ind:
                            cm_ns_xu = cov_model_non_stationarity_xu_list[i]
                            val = cm_ns_xu[1]
                            if len(cm_ns_xu) == 3:
                                kwds = cm_ns_xu[2]
                            else:
                                kwds = {}
                            try:
                                cov_model.multiply_w(val[k], logger=logger, **kwds)
                            except:
                                err_msg = f'{fname}: cannot apply non-stationarity (at one point) for covariance model (check parameter `cov_model_non_stationarity_xu_list`)'
                                if logger: logger.error(err_msg)
                                raise CovModelError(err_msg)

                        cov_func = cov_model.func() # update covariance function
                        if omni_dir:
                            # covariance model in 1D is used
                            cov0 = cov_func(0.)[0] # covariance function at origin (lag=0)
                        else:
                            cov0 = cov_func(np.zeros(d))[0] # covariance function at origin (lag=0)
                        
                        vu_std[k] = np.sqrt(cov0)
                
                else:
                    vu_std = np.sqrt(cov0) * np.ones(nu)

            else:
                vu_std = np.sqrt(cov0) * np.ones(nu)

        else: #if method == 'ordinary_kriging':               
            vu = np.zeros(nu)

            if cov_model_non_stationarity_xu_list is not None:
                if not isinstance(cov_model_non_stationarity_xu_list, list):
                    err_msg = f'{fname}: `cov_model_non_stationarity_xu_list` must be a list if not `None`'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)
                
                # WORK ON A COPY OF COVARIANCE MODEL (IN CASE IT IS ADAPTED)!
                cov_model = copyCovModel(cov_model) 
                cov_model_has_changed = False
                adapt_cov_model_ind = []
                for i, cm_ns_xu in enumerate(cov_model_non_stationarity_xu_list):
                    if not hasattr(cm_ns_xu, '__len__') or len(cm_ns_xu) not in (2, 3):
                        err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_xu_list` not valid: should be a tuple or list of length 2 or 3 containing: method name (str), val[, kwds (dict)]'
                        if logger: logger.error(err_msg)
                        raise CovModelError(err_msg)
                    if cm_ns_xu[0] != 'multiply_w':
                        continue
                    val = np.asarray(cm_ns_xu[1], dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                    if val.size != 1:
                        if val.size != nu:
                            err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_xu_list`, `(method_name, val[, kwds])`: size of `val` not valid'
                            if logger: logger.error(err_msg)
                            raise CovModelError(err_msg)
                        if np.allclose(val, val[0]):
                            val = val[:1] # of size 1
                        else:
                            adapt_cov_model_ind.append(i)
                    if val.size == 1:
                        if len(cm_ns_xu) == 3:
                            kwds = cm_ns_xu[2]
                        else:
                            kwds = {}
                        try:
                            cov_model.multiply_w(val[0], logger=logger, **kwds)
                        except:
                            err_msg = f'{fname}: cannot apply non-stationarity (at points) for covariance model (check parameter `cov_model_non_stationarity_xu_list`)'
                            if logger: logger.error(err_msg)
                            raise CovModelError(err_msg)
                        cov_model_has_changed = True

                if cov_model_has_changed:
                    # Update - Covariance function and value at 0
                    cov_func = cov_model.func() # covariance function
                    if omni_dir:
                        # covariance model in 1D is used
                        cov0 = cov_func(0.)[0] # covariance function at origin (lag=0)
                    else:
                        cov0 = cov_func(np.zeros(d))[0] # covariance function at origin (lag=0)

                if len(adapt_cov_model_ind):
                    cov_model_base = copyCovModel(cov_model)
                    vu_std = np.zeros(nu)
                    for k in range(nu):
                        cov_model = copyCovModel(cov_model_base)
                        for i in adapt_cov_model_ind:
                            cm_ns_xu = cov_model_non_stationarity_xu_list[i]
                            val = cm_ns_xu[1]
                            if len(cm_ns_xu) == 3:
                                kwds = cm_ns_xu[2]
                            else:
                                kwds = {}
                            try:
                                cov_model.multiply_w(val[k], logger=logger, **kwds)
                            except:
                                err_msg = f'{fname}: cannot apply non-stationarity (at one point) for covariance model (check parameter `cov_model_non_stationarity_xu_list`)'
                                if logger: logger.error(err_msg)
                                raise CovModelError(err_msg)

                        cov_func = cov_model.func() # update covariance function
                        if omni_dir:
                            # covariance model in 1D is used
                            cov0 = cov_func(0.)[0] # covariance function at origin (lag=0)
                        else:
                            cov0 = cov_func(np.zeros(d))[0] # covariance function at origin (lag=0)
                        
                        vu_std[k] = np.sqrt(cov0)
                
                else:
                    vu_std = np.sqrt(cov0) * np.ones(nu)

            else:
                vu_std = np.sqrt(cov0) * np.ones(nu)

        return vu, vu_std

    # Here: n > 0 and nu > 0

    # WORK ON A COPY OF COVARIANCE MODEL (IN CASE IT IS ADAPTED)!
    cov_model = copyCovModel(cov_model) 
    cov_model_has_changed = False

    # Rotation given by alpha_xu, beta_xu, gamma_xu
    if alpha_xu is not None:
        if omni_dir:
            err_msg = f'{fname}: `alpha_xu` cannot be used with 1D covariance model'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    if beta_xu is not None:
        if omni_dir or d < 3:
            err_msg = f'{fname}: `beta_xu` cannot be used with 1D or 2D covariance model'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
    
    if gamma_xu is not None:
        if omni_dir or d < 3:
            err_msg = f'{fname}: `gamma_xu` cannot be used with 1D or 2D covariance model'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    if omni_dir:
        rot = False
    else:
        if d == 2:
            # 2D - check only alpha
            if alpha_xu is None:
                rot = False
            else:
                alpha_xu = np.asarray(alpha_xu, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                if alpha_xu.size == 1:
                    if alpha_xu[0] == 0.0:
                        rot = False
                    else:
                        rot_mat = rotationMatrix2D(alpha_xu[0]) # rot_mat : rotation matrix for any xu[i]
                        rot = True
                        rot_mat_unique = True
                elif alpha_xu.size == nu:
                    if np.allclose(alpha_xu, alpha_xu[0]):
                        if np.isclose(alpha_xu[0], 0.0):
                            rot = False
                        else:
                            rot_mat = rotationMatrix2D(alpha_xu[0]) # rot_mat : rotation matrix for any xu[i]
                            rot = True
                            rot_mat_unique = True
                    else:
                        rot_mat = rotationMatrix2D(alpha_xu).transpose(2, 0, 1) # rot_mat[i] : rotation matrix for xu[i]
                        rot = True
                        rot_mat_unique = False
                else:
                    err_msg = f'{fname}: size of `alpha_xu` is not valid'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)

        else: # d == 3
            # 3D
            if alpha_xu is None and beta_xu is None and gamma_xu is None:
                rot = False
            else:
                if alpha_xu is not None:
                    alpha_xu = np.asarray(alpha_xu, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                    if alpha_xu.size == 1:
                        alpha_xu = alpha_xu * np.ones(nu)
                    elif alpha_xu.size != nu:
                        err_msg = f'{fname}: size of `alpha_xu` is not valid'
                        if logger: logger.error(err_msg)
                        raise CovModelError(err_msg)
                else:
                    alpha_xu = np.zeros(nu)

                if beta_xu is not None:
                    beta_xu = np.asarray(beta_xu, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                    if beta_xu.size == 1:
                        beta_xu = beta_xu * np.ones(nu)
                    elif beta_xu.size != nu:
                        err_msg = f'{fname}: size of `beta_xu` is not valid'
                        if logger: logger.error(err_msg)
                        raise CovModelError(err_msg)
                else:
                    beta_xu = np.zeros(nu)

                if gamma_xu is not None:
                    gamma_xu = np.asarray(gamma_xu, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                    if gamma_xu.size == 1:
                        gamma_xu = gamma_xu * np.ones(nu)
                    elif gamma_xu.size != nu:
                        err_msg = f'{fname}: size of `gamma_xu` is not valid'
                        if logger: logger.error(err_msg)
                        raise CovModelError(err_msg)
                else:
                    gamma_xu = np.zeros(nu)

                # if np.unique(np.array((alpha_xu, beta_xu, gamma_xu)).T, axis=0).shape[0] == 1:
                if np.allclose(np.vstack((alpha_xu, beta_xu, gamma_xu)).T, np.array([alpha_xu[0], beta_xu[0], gamma_xu[0]])):
                    # if alpha_xu[0] == 0.0 and beta_xu[0] == 0.0 and gamma_xu[0] == 0.0:
                    if np.isclose(alpha_xu[0], 0.0) and np.isclose(beta_xu[0], 0.0) and np.isclose(gamma_xu[0], 0.0):
                        rot = False
                    else:
                        rot_mat = rotationMatrix3D(alpha_xu[0], beta_xu[0], gamma_xu[0]) # rot_mat : rotation matrix for any xu[i]
                        rot = True
                        rot_mat_unique = True
                else:
                    rot_mat = rotationMatrix3D(alpha_xu, beta_xu, gamma_xu).transpose(2, 0, 1) # rot_mat[i] : rotation matrix for xu[i]
                    rot = True
                    rot_mat_unique = False

    if rot:
        if d == 2:
            cov_model.set_alpha(0.0)
            cov_model_has_changed = True
        elif d == 3:
            cov_model.set_alpha(0.0)
            cov_model.set_beta(0.0)
            cov_model.set_gamma(0.0)
            cov_model_has_changed = True
        if rot_mat_unique:
            # apply rotation to data points x and points xu
            x = x.dot(rot_mat)
            xu = xu.dot(rot_mat)
            rot = False # no need rotation further

    # here: rot = True means that local rotation are applied

    # Prepare non-stationarities for integration in covariance model
    adapt_cov_model_ind = []
    recompute_cov0 = False
    recompute_dmax_ax = False
    if cov_model_non_stationarity_xu_list is not None:
        if not isinstance(cov_model_non_stationarity_xu_list, list):
            err_msg = f'{fname}: `cov_model_non_stationarity_xu_list` must be a list if not `None`'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
        for i, cm_ns_xu in enumerate(cov_model_non_stationarity_xu_list):
            if not hasattr(cm_ns_xu, '__len__') or len(cm_ns_xu) not in (2, 3):
                err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_xu_list` not valid: should be a tuple or list of length 2 or 3 containing: method name (str), val[, kwds (dict)]'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)
            val = np.asarray(cm_ns_xu[1], dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if val.size != 1:
                if val.size != nu:
                    err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_xu_list`, `(method_name, val[, kwds])`: size of `val` not valid'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)
                if np.allclose(val, val[0]):
                    val = val[:1] # of size 1
                else:
                    adapt_cov_model_ind.append(i)
                    if cm_ns_xu[0] == 'multiply_w':
                        recompute_cov0 = True
                    elif cm_ns_xu[0] == 'multiply_r':
                        recompute_dmax_ax = True
            if val.size == 1:
                method_name = cm_ns_xu[0]
                if len(cm_ns_xu) == 3:
                    kwds = cm_ns_xu[2]
                else:
                    kwds = {}
                try:
                    eval(f'cov_model.{method_name}')(val[0], logger=logger, **kwds)
                except:
                    err_msg = f'{fname}: cannot apply non-stationarity (at points) for covariance model (check parameter `cov_model_non_stationarity_xu_list`)'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)
                cov_model_has_changed = True

    adapt_cov_model = len(adapt_cov_model_ind) > 0

    if cov_model_has_changed:
        # Update - Covariance function and value at 0
        cov_func = cov_model.func() # covariance function
        if omni_dir:
            # covariance model in 1D is used
            cov0 = cov_func(0.)[0] # covariance function at origin (lag=0)
        else:
            cov0 = cov_func(np.zeros(d))[0] # covariance function at origin (lag=0)
    
    # Do kriging
    if use_unique_neighborhood:
        if rot:
            err_msg = f'{fname}: unique search neighborhood cannot be used with local rotation'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

        if adapt_cov_model:
            err_msg = f'{fname}: unique search neighborhood cannot be used with non-stationary for covariance'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

        # Set kriging matrix (mat) of order nmat
        if ordinary_kriging:
            nmat = n+1
            mat = np.ones((nmat, nmat))
            mat[-2,-2] = cov0 + v_err_var[n-1]
            mat[-1,-1] = 0.0
        else:
            nmat = n
            mat = np.ones((nmat, nmat))
            mat[-1,-1] = cov0 + v_err_var[n-1]
        for i in range(n-1):
            # lag between x[i] and x[j], j=i+1, ..., n-1
            h = x[(i+1):] - x[i]
            if omni_dir:
                # compute norm of lag
                h = np.sqrt(np.sum(h**2, axis=1))
            cov_h = cov_func(h)
            mat[i, (i+1):n] = cov_h
            mat[(i+1):n, i] = cov_h
            mat[i, i] = cov0 + v_err_var[i]

        # Set right hand side of all kriging systems (b),
        #   - matrix of dimension nmat x nu
        b = np.ones((nmat, nu))
        for i in range(n):
            # lag between x[i] and every xu
            h = xu - x[i]
            if omni_dir:
                # compute norm of lag
                h = np.sqrt(np.sum(h**2, axis=1))
            b[i,:] = cov_func(h)

        # Solve all kriging systems
        w = np.linalg.solve(mat, b) # w: matrix of dimension nmat x nu

        # Kriged values at unknown points
        if mean_x is not None:
            # simple kriging
            if var_x is not None:
                vu = mean_xu + varUpdate_xu*(1.0/varUpdate_x*(v-mean_x)).dot(w)
                # vu = mean_xu + varUpdate_xu*(1.0/varUpdate_x*(v-mean_x)).dot(w[:n,:])
            else:
                vu = mean_xu + (v-mean_x).dot(w)
                # vu = mean_xu + (v-mean_x).dot(w[:n,:])
        else:
            # ordinary kriging
            vu = v.dot(w[:n,:])

        # Kriged standard deviation at unknown points
        vu_std = np.sqrt(np.maximum(0.0, cov0 - np.array([np.dot(w[:,i], b[:,i]) for i in range(nu)])))

    else:
        # Limited search neighborhood
        if searchRadius is not None:
            if searchRadius <= 0.0:
                err_msg = f'{fname}: search radius (isotropic neighborhood) not valid (negative)'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)
            
            if omni_dir:
                dmax_ax = np.array([searchRadius], dtype='float')
            else:
                dmax_ax = searchRadius * np.ones(d)

            recompute_dmax_ax = False

        else:
            # use searchRadiusRelative
            if searchRadiusRelative <= 0.0:
                err_msg = f'{fname}: search radius relative (factor) not valid (negative)'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)
            
            if omni_dir:
                dmax_ax = np.array([cov_model.r()])
            elif d == 2:
                dmax_ax = cov_model.r12()
            elif d == 3:
                dmax_ax = cov_model.r123()

            dmax_ax = searchRadiusRelative * dmax_ax

        dmax_ax_inv2 = 1.0 / (dmax_ax * dmax_ax)

        # Preparation if covariance model will be adapted
        if adapt_cov_model:
            cov_model_base = copyCovModel(cov_model)

        # Maximum number of neighbors
        if nneighborMax is None or nneighborMax > n or nneighborMax < 0:
            nneighborMax = n

        mat = np.ones((nneighborMax+1, nneighborMax+1)) # allocate kriging matrix
        b = np.ones(nneighborMax+1) # allocate second member

        # Allocate memory for output
        vu = np.zeros(nu)
        vu_std = np.zeros(nu)

        if verbose > 0:
            progress_old = 0

        for j, x0 in enumerate(xu):
            if verbose > 0:
                progress = int(j/nu*100.0)
                if progress > progress_old:
                    if logger:
                        logger.info(f'{fname}: {progress:3d}%')
                    else:
                        print(f'{fname}: {progress:3d}%')
                    progress_old = progress

            if adapt_cov_model:
                cov_model = copyCovModel(cov_model_base)
                for i in adapt_cov_model_ind:
                    cm_ns_xu = cov_model_non_stationarity_xu_list[i]
                    method_name = cm_ns_xu[0]
                    val = cm_ns_xu[1]
                    if len(cm_ns_xu) == 3:
                        kwds = cm_ns_xu[2]
                    else:
                        kwds = {}
                    try:
                        eval(f'cov_model.{method_name}')(val[j], logger=logger, **kwds)
                    except:
                        err_msg = f'{fname}: cannot apply non-stationarity (at one point) for covariance model (check parameter `cov_model_non_stationarity_xu_list`)'
                        if logger: logger.error(err_msg)
                        raise CovModelError(err_msg)

                cov_func = cov_model.func() # update covariance function
                if recompute_cov0:
                    if omni_dir:
                        # covariance model in 1D is used
                        cov0 = cov_func(0.)[0] # covariance function at origin (lag=0)
                    else:
                        cov0 = cov_func(np.zeros(d))[0] # covariance function at origin (lag=0)

                if recompute_dmax_ax:
                    if omni_dir:
                        dmax_ax = np.array([cov_model.r()])
                    elif d == 2:
                        dmax_ax = cov_model.r12()
                    elif d == 3:
                        dmax_ax = cov_model.r123()

                    dmax_ax = searchRadiusRelative * dmax_ax
                    dmax_ax_inv2 = 1.0 / (dmax_ax * dmax_ax)

            h = x0 - x
            if rot:
                h = h.dot(rot_mat[j])
            d2 = np.sum(dmax_ax_inv2 * h**2, axis=1)
            ind = np.where(d2 < 1.0)[0]
            if len(ind) > nneighborMax:
                ind_s = np.argsort(d2[ind])
                ind = ind[ind_s[:nneighborMax]]
            nn = len(ind)
            if nn == 0:
                # vu[j] = np.nan
                # vu_std[j] = np.nan
                # Kriged values at xu[j]
                if mean_x is not None:
                    vu[j] = mean_xu[j]
                else:
                    vu[j] = np.mean(v)

                # Kriged standard deviation at xu[j]
                vu_std[j] = np.sqrt(cov0)

            else:
                xneigh = x[ind]
                vneigh = v[ind]
                v_err_var_neigh = v_err_var[ind]

                # Set right hand side of the kriging system (b)
                h = h[ind]
                if omni_dir:
                    # compute norm of lag
                    h = np.sqrt(np.sum(h**2, axis=1))
                b[:nn] = cov_func(h)

                if ordinary_kriging:
                    nmat = nn+1
                else:
                    nmat = nn

                # Set kriging matrix (mat) of order nmat
                for i in range(nn-1):
                    # lag between xneigh[i] and xneigh[j], j=i+1, ..., nn-1
                    h = xneigh[(i+1):] - xneigh[i]
                    if omni_dir:
                        # compute norm of lag
                        h = np.sqrt(np.sum(h**2, axis=1))
                    elif rot:
                        h = h.dot(rot_mat[j])
                    cov_h = cov_func(h)
                    mat[i, (i+1):nn] = cov_h
                    mat[(i+1):nn, i] = cov_h
                    mat[i, i] = cov0 + v_err_var_neigh[i]

                mat[nn-1,nn-1] = cov0 + v_err_var_neigh[nn-1]

                if ordinary_kriging:
                    mat[:, nn] = 1.0
                    mat[nn, :] = 1.0
                    mat[nn,nn] = 0.0
                    b[nn] = 1.0

                # Solve the kriging system
                w = np.linalg.solve(mat[:nmat,:nmat], b[:nmat])

                # Kriged values at xu[j]
                if mean_x is not None:
                    # simple kriging
                    if var_x is not None:
                        vu[j] = mean_xu[j] + varUpdate_xu[j]*(1.0/varUpdate_x[ind]*(vneigh-mean_x[ind])).dot(w)
                    else:
                        vu[j] = mean_xu[j] + (vneigh-mean_x[ind]).dot(w)
                else:
                    # ordinary kriging
                    vu[j] = vneigh.dot(w[:nn])

                # Kriged standard deviation at xu[j]
                vu_std[j] = np.sqrt(max(0, cov0 - np.dot(w, b[:nmat])))

    if var_x is not None:
        vu_std = varUpdate_xu * vu_std

    if verbose > 0:
        if logger:
            logger.info(f'{fname}: {100:3d}%')
        else:
            print(f'{fname}: {100:3d}%')

    return vu, vu_std
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def cross_valid_loo(
        x, v, cov_model,
        v_err_std=0.0,
        significance=0.05,
        dmin=None,
        mean_x=None,
        var_x=None,
        alpha_x=None,
        beta_x=None,
        gamma_x=None,
        cov_model_non_stationarity_x_list=None,
        interpolator=krige,
        interpolator_kwargs=None,
        print_result=True,
        make_plot=True,
        figsize=None,
        nbins=None,
        logger=None):
    """
    Performs cross-validation by leave-one-out error.

    A covariance model is tested (cross-validation) on a data set, by appliying
    an interpolator on each data point (ignoring its value).

    Let vm[i] and vsd[i] be respectively the mean and standard deviation at point
    x[i] (accounting for the other points in x and the given covariance model),
    obtained by kriging. This mean that the value at x[i] should follow a normal
    law :math:`\\mathcal{N}(vm[i], vsd[i]^2)`.

    Continuous Rank Probability Score (CRPS) are computed for each data point.
    Let F[i] be the cumulative distribution function (CDF) of N(vm[i], vsd[i]^2),
    the prediction distribution at point x[i], where the true value is v[i].
    The CRPS at x[i] is defined as

    .. math::
        crps[i] = - \\int_{-\\infty}^{+\\infty}(F[i](y) - \\mathbf{1}(y>v[i]))^2 dy

    and is equal to

    .. math::
        crps[i] = vsd[i] \\left[1/\\sqrt{\\pi} - 2\\varphi(w[i]) - w[i](2\\Phi(w[i])-1)\\right]

    where :math:`w[i] = (v[i] - vm[i])/vsd[i]`, and :math:`\\varphi`, :math:`\\Phi`
    are respectively the pdf and cdf of the standard nomral distribution
    (:math:`\\mathcal{N}(0,1)`); see reference:

    - Tilmann Gneiting, Adrian E. Raftery (2012), Strictly Proper Scoring Rules, \
    Prediction, and Estimation, pp. 359-378, \
    `doi:10.1198/016214506000001437 <https://dx.doi.org/10.1198/016214506000001437>`_

    Note that

    - the normalized error e[i] (see below) verifies: w[i] = -e[i]
    - CRPS is negative, the larger, the better; moreover, the unbiased prediction \
    at x[i] with a standard deviation of std(v) (st. dev. of the data values) \
    would be vm[i] = v[i], and vsd[i] = std(v), which gives a "default" crps of \
    :math:`crps_{def} = std(v) (1-\\sqrt{2})/\\sqrt{\\pi}`

    Furthermore, if the model is right, the normalized error between the true
    value v[i] and the vm[i], i.e.

    .. math::
        e[i] = (vm[i] - v[i])/vsd[i]

    should follows a standard normal distribution (:math:`\\mathcal{N}(0,1)`).

    Then:

    - the mean of the nomalized error should follow (according to the central \
    limit theorem, CLT) a normal law :math:`\\mathcal{N}(0,1/n)`, where n is \
    the number of samples, i.e. the number of points in x (see test 1 below)
    - assuming the nomrmalized error independant, the sum of their square should \
    follow a chi-square distribution with n degrees of freedom (see test 2 below)

    Two statisic tests are performed:

    - 1. normal law test for mean of normalized error, merrn: \
    the test computes the p-value: :math:`P(\\vert Z\\vert \\geqslant \\vert merrn\\vert)`, \
    where :math:`Z\\sim\\mathcal{N}(0,1/n)`
    - 2. chi-square test for sum of squares of normalized error, sserrn: \
    the test computes the p-value: :math:`P(X \\geqslant sserrn)`, where \
    :math:`X\\sim\\chi^2_n` (chi-square with n degrees of freedom);

    A low p-value (e.g. below a significance level alpha=0.05) means that the
    model should be rejected (falsely rejected with probability alpha):
    the smaller the p-value, the more evidence there is to reject the model;
    each test computes:

    - the p-value `pvalue`
    - the result (success/failure): the boolean `success=(pvalue > significance)` \
    which means that the model should be rejected when `success=False` with \
    respect to the specified significance level

    Parameters
    ----------
    x : 2D array of floats of shape (n, d)
        data points locations, with n the number of data points and d the space
        dimension (1, 2, or 3), each row of `x` is the coordinatates of one data
        point; note: for data in 1D (`d=1`), 1D array of shape `(n,)` is accepted
        for `n` data points

    v : 1D array of floats of shape (n,)
        data points values, with n the number of data points, `v[i]` is the data
        value at location `x[i]`

    cov_model : :class:`CovModel1D` or :class:`CovModel2D` or :class:`CovModel3D`
        covariance model in 1D, 2D, or 3D, in same dimension as dimension of
        points (d), i.e.:

        - :class:`CovModel1D` for data in 1D (d=1)
        - :class:`CovModel2D` for data in 2D (d=2)
        - :class:`CovModel3D` for data in 3D (d=3)

        or

        - :class:`CovModel1D` interpreted as an omni-directional covariance model \
        whatever dimension of points (d);

    v_err_std : 1D array of floats of shape (n,), or float, default: 0.0
        standard deviation of error at data points, with n the number of data points; 
        if `v_err_std` is a float, the same value is used for all data points; 
        this means that at location x[i], the data value is considered as in a Gaussian 
        distribution of mean `v[i]` and standard deviation `v_err_std[i]`

    dmin : float, optional
        minimal distance between the data point to be estimated and the other
        ones, i.e. when estimating the value at `x[i]`, all points `x[j]` at a
        distance less than `dmin` are not taking into account;
        note: this means that the cross-validation is no longer "leave-one-out"
        since more than one point may be ignored

    mean_x : 1D array-like of floats, or float, optional
        kriging mean value at data points `x`

        - if `mean_x` is a float, the same value is considered for any point;
        - if `mean_x=None` (default): the mean of data values, i.e. mean of `v`, \
        is considered for any point;

        note: parameter `mean_x` is ignored if ordinary kriging is used as
        interpolator

    var_x : 1D array-like of floats, or float, optional
        kriging variance value at data points `x`

        - if `var_x` is a float, the same value is considered for any point
        - if `var_x=None` (default): not used  (use of covariance model only)

        note: parameter `var_x` is ignored if ordinary kriging is used as
        interpolator

    alpha_x : 1D array-like of floats, or float, optional
        azimuth angle in degrees at points `x`

        - if `alpha_x` is a float, the same value is considered for any point
        - if `alpha_x=None` (default): `alpha_x=0.0` is used for any point

        note: `alpha_x` is ignored if the covariance model is in 1D

    beta_x : 1D array-like of floats, or float, optional
        dip angle in degrees at points `x`
        - if `beta_x` is a float, the same value is considered for any point
        - if `beta_x=None` (default): `beta_x=0.0` is used for any point

        note: `beta_x` is ignored if the covariance model is in 1D or 2D

    gamma_x : 1D array-like of floats, or float, optional
        dip angle in degrees at points `x`

        - if `gamma_x` is a float, the same value is considered for any point
        - if `gamma_x=None` (default): `gamma_x=0.0` is used for any point

        note: `gamma_x` is ignored if the covariance model is in 1D or 2D

    cov_model_non_stationarity_x_list : list, optional
        list to set non-stationarities in covariance model; each entry must be
        a tuple (or list) `cm_ns` of length 2 or 3 with:

        - `cm_ns[0]`: str: the name of the method of `cov_model` to be applied
        - `cm_ns[1]`: 1D array-like of floats, or float: \
        used to set the main parameter passed to the method:
            - if array-like: its size must be equal to `nu`, \
            (the array is reshaped if needed), values at points `x`
            - if a float: same value at all points `x`
        - `cm_ns[2]`: dict, optional: keyworkds arguments to be passed to the method

        Examples (with the parameter `arg` is set from `val`)
        
        - `('multiply_w', val)` will apply `cov_model.multiply_w(arg)`; 
            this multipies the weight contribution of every elementary contribution of the 
            covariance model
        - `('multiply_w', val, {'elem_ind':0})` will apply `cov_model.multiply_w(arg, elem_ind=0)`;
            this multipies the weight contribution of the elementary contribution of index 0 of the 
            covariance model
        - `('multiply_r', val)` will apply `cov_model.multiply_r(arg)`;
            this multipies the range in all direction of every elementary contribution of the
            covariance model
        - `('multiply_r', val, {'r_ind':0})` will apply `cov_model.multiply_r(arg, r_ind=0)`;
            this multipies the range in the first main direction (index 0) of every elementary 
            contribution of the covariance model
        - `('multiply_r', val, {'r_ind':0, 'elem_ind':0})` will apply `cov_model.multiply_r(arg, r_ind=0, elem_ind=0)`;
            this multipies the range in the first main direction (index 0) of the elementary 
            contribution of index 0 of the covariance model

    significance : float, default: 0.05
        significance level for the two statisic tests, a float between 0 and 1,
        defining the success/failure of the two statistic tests (see above)

    interpolator : function (`callable`), default: krige
        function used to do the interpolations

    interpolator_kwargs : dict, optional
        keyword arguments passed to `interpolator`;
        e.g. with `interpolator=krige`:

        - `interpolator_kwargs={'method':'ordinary_kriging'}`,
        - `interpolator_kwargs={'method':'simple_kriging', 'use_unique_neighborhood':True}`

    print_result : bool, default: True
        indicates if the results (mean CRPS, and the 2 statistic tests) are
        printed, as well as some indicators

    make_plot : bool, default: True
        indicates if a plot of the results is displayed (in a new "1x2" figure)

    figsize : 2-tuple, optional
        size of the new "1x2" figure (if `make_plot=True`)

    nbins : int, optional
        number of bins in plotted histogram

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    v_est : 1D array of shape (n,)
        estimates at data points `x` by the interpolation

    v_std : 1D array of shape (n,)
        standard deviations at data points `x` by the interpolation

    crps : 1D array of shape (n,)
        CRPS of the prediction distribution (normal) at data points `x`

    crps_def : float
        "default" crps, according to unbiased prediction with the standard
        deviation of the data values (see above)

    pvalue : 1D array of two floats
        - pvalue[0]: p-value for the statistic test 1 \
        (normal law test for mean of normalized error)
        - pvalue[1]: p-value for the statistic test 2 \
        (chi-square test for sum of squares of normalized error)

    success : 1D array of two bools
        - `success[i] = (pvalue[i] > significance)`, success (True) \
        or failure (False) of the corresponding statistical test

        if one False is obtained, it means that the model should be
        rejected
    """
    fname = 'cross_valid_loo'

    # Prevent calculation if covariance model is not stationary
    if not cov_model.is_stationary():
        err_msg = f'{fname}: `cov_model` is not stationary: cross validation cannot be applied'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    x = np.asarray(x)
    # Get dimension (d) from x
    if x.ndim == 1:
        # x is a 1-dimensional array
        x = x.reshape(-1, 1)
        d = 1
    else:
        # x is a 2-dimensional array
        d = x.shape[1]

    # Number of data points
    n = x.shape[0]

    # Check size of v
    v = np.asarray(v).reshape(-1)
    if v.size != n:
        err_msg = f'{fname}: size of `v` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # Check dimension of cov_model and set if used as omni-directional model
    if isinstance(cov_model, CovModel1D):
        omni_dir = True
    else:
        if cov_model.__class__.__name__ != f'CovModel{d}D':
            err_msg = f'{fname}: `cov_model` dimension is incompatible with dimension of points'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

        omni_dir = False

    # Leave-one-out (loo) cross validation dictionary of parameters of the interpolator
    if interpolator_kwargs is None:
        interpolator_kwargs = {}

    # Preparation of integration in keyword arguments of the function krige
    if interpolator == krige:
        adapt_kwds_v_err_std = False
        v_err_std = np.asarray(v_err_std, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if v_err_std.size == 1:
            interpolator_kwargs['v_err_std'] = v_err_std[0]
        elif v_err_std.size != n:
            err_msg = f'{fname}: size of `v_err_std` is not valid'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)        
        else:
            adapt_kwds_v_err_std = True

        adapt_kwds_mean_x = False
        if mean_x is not None:
            mean_x = np.asarray(mean_x, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if mean_x.size == 1:
                interpolator_kwargs['mean_x'] = mean_x[0]
                interpolator_kwargs['mean_xu'] = mean_x[0]
            elif mean_x.size != n:
                err_msg = f'{fname}: size of `mean_x` is not valid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)
            else:
                adapt_kwds_mean_x = True

        adapt_kwds_var_x = False
        if var_x is not None:
            var_x = np.asarray(var_x, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if var_x.size == 1:
                interpolator_kwargs['var_x'] = var_x[0]
                interpolator_kwargs['var_xu'] = var_x[0]
            elif var_x.size != n:
                err_msg = f'{fname}: size of `var_x` is not valid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)
            else:
                adapt_kwds_var_x = True

        adapt_kwds_alpha_x = False
        if alpha_x is not None:
            alpha_x = np.asarray(alpha_x, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if alpha_x.size == 1:
                interpolator_kwargs['alpha_xu'] = alpha_x[0]
            elif alpha_x.size != n:
                err_msg = f'{fname}: size of `alpha_x` is not valid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)
            else:
                adapt_kwds_alpha_x = True

        adapt_kwds_beta_x = False
        if beta_x is not None:
            beta_x = np.asarray(beta_x, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if beta_x.size == 1:
                interpolator_kwargs['beta_xu'] = beta_x[0]
            elif beta_x.size != n:
                err_msg = f'{fname}: size of `beta_x` is not valid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)
            else:
                adapt_kwds_beta_x = True

        adapt_kwds_gamma_x = False
        if gamma_x is not None:
            gamma_x = np.asarray(gamma_x, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if gamma_x.size == 1:
                interpolator_kwargs['gamma_xu'] = gamma_x[0]
            elif gamma_x.size != n:
                err_msg = f'{fname}: size of `gamma_x` is not valid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)
            else:
                adapt_kwds_gamma_x = True

        adapt_kwds_cov_model_non_stationarity_x_list = False
        if cov_model_non_stationarity_x_list is not None:
            interpolator_kwargs['cov_model_non_stationarity_xu_list'] = []
            adapt_kwds_cov_model_non_stationarity_x_list_ind = []
            if not isinstance(cov_model_non_stationarity_x_list, list):
                err_msg = f'{fname}: `cov_model_non_stationarity_x_list` must be a list if not `None`'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)
            for i, cm_ns_x in enumerate(cov_model_non_stationarity_x_list):
                if not hasattr(cm_ns_x, '__len__') or len(cm_ns_x) not in (2, 3):
                    err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_x_list` not valid: should be a tuple or list of length 2 or 3 containing: method name (str), val[, kwds (dict)]'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)
                val = np.asarray(cm_ns_x[1], dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                if val.size != 1:
                    if val.size != n:
                        err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_x_list`, `(method_name, val[, kwds])`: size of `val` not valid'
                        if logger: logger.error(err_msg)
                        raise CovModelError(err_msg)
                    if np.allclose(val, val[0]):
                        val = val[:1] # of size 1
                    else:
                        adapt_kwds_cov_model_non_stationarity_x_list_ind.append(i)
                if len(cm_ns_x) == 3:
                    interpolator_kwargs['cov_model_non_stationarity_xu_list'].append([cm_ns_x[0], val[0], cm_ns_x[2]])
                else:
                    interpolator_kwargs['cov_model_non_stationarity_xu_list'].append([cm_ns_x[0], val[0]])
    
            adapt_kwds_cov_model_non_stationarity_x_list = len(adapt_kwds_cov_model_non_stationarity_x_list_ind) > 0

    adapt_kwds =       adapt_kwds_v_err_std \
                    or adapt_kwds_mean_x \
                    or adapt_kwds_var_x \
                    or adapt_kwds_alpha_x \
                    or adapt_kwds_beta_x \
                    or adapt_kwds_gamma_x \
                    or adapt_kwds_cov_model_non_stationarity_x_list

    # Do loo cross-validation
    v_est, v_std = np.zeros(n), np.zeros(n)
    ind = np.arange(n)
    if dmin is not None and dmin > 0.0:
        dmin2 = dmin**2
        if adapt_kwds:
            # adapt_kwds = True
            for i in range(n):
                indx = np.sum((x-x[i])**2, axis=1) >= dmin2
                if np.all(~indx):
                    err_msg = f'{fname}: `dmin` is too large: no more point for evaluation'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)

                # adapt kwds
                if adapt_kwds_v_err_std:
                    interpolator_kwargs['v_err_std'] = v_err_std[indx]
                if adapt_kwds_mean_x:
                    interpolator_kwargs['mean_x'] = mean_x[indx]
                    interpolator_kwargs['mean_xu'] = mean_x[i]
                if adapt_kwds_var_x:
                    interpolator_kwargs['var_x'] = var_x[indx]
                    interpolator_kwargs['var_xu'] = var_x[i]
                if adapt_kwds_alpha_x:
                    interpolator_kwargs['alpha_xu'] = alpha_x[i]
                if adapt_kwds_beta_x:
                    interpolator_kwargs['beta_xu'] = beta_x[i]
                if adapt_kwds_gamma_x:
                    interpolator_kwargs['gamma_xu'] = gamma_x[i]
                if adapt_kwds_cov_model_non_stationarity_x_list:
                    for k in adapt_kwds_cov_model_non_stationarity_x_list_ind:
                        interpolator_kwargs['cov_model_non_stationarity_xu_list'][k][1] = cov_model_non_stationarity_x_list[k][1][i]

                # interpolation
                v_est[i], v_std[i] = interpolator(x[indx], v[indx], np.array(x[i]).reshape(-1, d), cov_model, **interpolator_kwargs)

        else:
            # adapt_kwds = False
            for i in range(n):
                indx = np.sum((x-x[i])**2, axis=1) >= dmin2
                if np.all(~indx):
                    err_msg = f'{fname}: `dmin` is too large: no more point for evaluation'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)

                # interpolation
                v_est[i], v_std[i] = interpolator(x[indx], v[indx], np.array(x[i]).reshape(-1, d), cov_model, **interpolator_kwargs)
    
    else:
        if adapt_kwds:
            # adapt_kwds = True
            for i in range(n):
                indx = np.delete(ind, i)

                # adapt kwds
                if adapt_kwds_v_err_std:
                    interpolator_kwargs['v_err_std'] = v_err_std[indx]
                if adapt_kwds_mean_x:
                    interpolator_kwargs['mean_x'] = mean_x[indx]
                    interpolator_kwargs['mean_xu'] = mean_x[i]
                if adapt_kwds_var_x:
                    interpolator_kwargs['var_x'] = var_x[indx]
                    interpolator_kwargs['var_xu'] = var_x[i]
                if adapt_kwds_alpha_x:
                    interpolator_kwargs['alpha_xu'] = alpha_x[i]
                if adapt_kwds_beta_x:
                    interpolator_kwargs['beta_xu'] = beta_x[i]
                if adapt_kwds_gamma_x:
                    interpolator_kwargs['gamma_xu'] = gamma_x[i]
                if adapt_kwds_cov_model_non_stationarity_x_list:
                    for k in adapt_kwds_cov_model_non_stationarity_x_list_ind:
                        interpolator_kwargs['cov_model_non_stationarity_xu_list'][k][1] = cov_model_non_stationarity_x_list[k][1][i]

                # interpolation
                v_est[i], v_std[i] = interpolator(x[indx], v[indx], np.array(x[i]).reshape(-1, d), cov_model, **interpolator_kwargs)

        else:
            # adapt_kwds = False
            for i in range(n):
                indx = np.delete(ind, i)

                # interpolation
                v_est[i], v_std[i] = interpolator(x[indx], v[indx], np.array(x[i]).reshape(-1, d), cov_model, **interpolator_kwargs)

    # Normalized error
    errn = (v_est - v) / v_std

    # CRPS
    crps = v_std *(1.0/np.sqrt(np.pi) - 2.0*stats.norm.pdf(errn) + errn*(1.0 - 2*stats.norm.cdf(errn)))
    crps_def = np.std(v) * (1.0 - np.sqrt(2.0))/np.sqrt(np.pi)

    # Statistic tests
    pvalue = np.zeros(2)
    # Statistic test 1:
    #   with merrn = mean(errn), the p-value is:
    #   p = P(|Z| >= |merrn|), where Z~N(0, 1/n);
    #   p = P(|Y| >= sqrt(n)*|merrn|), where Y~N(0,1)
    #   p = 2*(1-Phi(sqrt(n)*|merrn|)), with Phi the cdf of N(0,1)
    pvalue[0] = 2.0*(1.0 - stats.norm.cdf(np.sqrt(n)*np.abs(np.mean(errn))))
    # Statistic test 1:
    #   with sserrn = sum(errn**2), the p-value is:
    #   p = P(X >= sserrn), where X~Chi2_n (chi-square with n degrees of freedom)
    #   p = 1-F_X(ssern), with F_X the cdf of Chi2_n
    pvalue[1] = 1.0 - stats.chi2.cdf(np.sum(errn**2), df=n)

    # success of each statistic test wrt significance level
    success = pvalue > significance

    if print_result:
        if logger:
            # CRPS
            logger.info('----- CRPS (negative; the larger, the better) -----')
            logger.info(f'   mean = {np.mean(crps):.4g}')
            logger.info(f'   def. = {crps_def:.4g}')
            # Result of test 1
            logger.info('----- 1) "Normal law test for mean of normalized error" -----')
            logger.info(f'   p-value = {pvalue[0]:.4g}')
            logger.info(f'   success = {success[0]} (wrt significance level {significance})')
            if success[0]:
                logger.info(f'      (-> model has no reason to be rejected)')
            else:
                logger.info(f'      -> model should be REJECTED')

            # Result of test 1
            logger.info('----- 2) "Chi-square test for sum of squares of normalized error" -----')
            logger.info(f'   p-value = {pvalue[1]:.4g}')
            logger.info(f'   success = {success[1]} (wrt significance level {significance})')
            if success[1]:
                logger.info(f'      (-> model has no reason to be rejected)')
            else:
                logger.info(f'      -> model should be REJECTED')

            # Some indicators
            logger.info('----- Statistics of normalized error -----')
            logger.info(f'   mean     = {np.mean(errn):.4g} (should be close to 0)')
            logger.info(f'   std      = {np.std(errn):.4g} (should be close to 1)')
            logger.info(f'   skewness = {stats.skew(errn):.4g} (should be close to 0)')
            logger.info(f'   excess kurtosis = {stats.kurtosis(errn):.4g} (should be close to 0)')

        else:
            # CRPS
            print('----- CRPS (negative; the larger, the better) -----')
            print(f'   mean = {np.mean(crps):.4g}')
            print(f'   def. = {crps_def:.4g}')
            # Result of test 1
            print('----- 1) "Normal law test for mean of normalized error" -----')
            print(f'   p-value = {pvalue[0]:.4g}')
            print(f'   success = {success[0]} (wrt significance level {significance})')
            if success[0]:
                print(f'      (-> model has no reason to be rejected)')
            else:
                print(f'      -> model should be REJECTED')

            # Result of test 1
            print('----- 2) "Chi-square test for sum of squares of normalized error" -----')
            print(f'   p-value = {pvalue[1]:.4g}')
            print(f'   success = {success[1]} (wrt significance level {significance})')
            if success[1]:
                print(f'      (-> model has no reason to be rejected)')
            else:
                print(f'      -> model should be REJECTED')

            # Some indicators
            print('----- Statistics of normalized error -----')
            print(f'   mean     = {np.mean(errn):.4g} (should be close to 0)')
            print(f'   std      = {np.std(errn):.4g} (should be close to 1)')
            print(f'   skewness = {stats.skew(errn):.4g} (should be close to 0)')
            print(f'   excess kurtosis = {stats.kurtosis(errn):.4g} (should be close to 0)')

    if make_plot:
        _, ax = plt.subplots(2,2, figsize=figsize)

        # Cross plot Z(x) vs Z*(x)
        plt.sca(ax[0, 0])
        plt.plot(v, v_est, 'o')
        tmp = [np.min(v), np.max(v)]
        plt.plot(tmp, tmp, ls='dashed')
        plt.xlabel('True value Z(x)')
        plt.ylabel('Estimation Z*(x)')
        plt.grid()
        plt.title('Cross plot Z(x) vs Z*(x)')

        # Histogram of crps
        plt.sca(ax[0, 1])
        plt.hist(crps, density=True, bins=nbins, color='lightblue', edgecolor='gray')
        plt.axvline(x=crps_def, c='orange', ls='dashed', label='crps_def')
        plt.axvline(x=np.mean(crps), c='tab:blue', ls='solid', label='mean')
        plt.xlabel('crps')
        plt.legend()
        plt.title('Histogram (density) of crps')

        # Histogram of normalized error
        plt.sca(ax[1, 0])
        plt.hist(errn, density=True, bins=nbins, color='lightblue', edgecolor='gray')
        plt.xlabel(r'$(Z*(x)-Z(x))/\sigma*(x)$')
        plt.title('Histogram (density) of normalized error')

        # QQ-plot with N(0,1)
        plt.sca(ax[1, 1])
        q = np.linspace(.02, .98, 50)
        plt.plot(stats.norm.ppf(q), np.quantile(errn, q=q))
        t = stats.norm.ppf(q[-1])
        plt.plot([-t, t], [-t, t], ls='dashed')
        plt.xlabel(r'$\mathcal{N}(0,1)$')
        plt.ylabel(r'$(Z*(x)-Z(x))/\sigma*(x)$')
        plt.grid()
        plt.title(r'QQ-plot $\mathcal{N}(0,1)$ vs normalized err.')
        # plt.show()

    # if make_plot:
    #     _, ax = plt.subplots(1,3, figsize=figsize)
    #
    #     # Cross plot Z(x) vs Z*(x)
    #     plt.sca(ax[0])
    #     plt.plot(v, v_est, 'o')
    #     tmp = [np.min(v), np.max(v)]
    #     plt.plot(tmp, tmp, ls='dashed')
    #     plt.xlabel('True value Z(x)')
    #     plt.ylabel('Estimation Z*(x)')
    #     plt.grid()
    #     plt.title('Cross plot Z(x) vs Z*(x)')
    #
    #     # Histogram of normalized error
    #     plt.sca(ax[1])
    #     plt.hist(errn, density=True)
    #     plt.xlabel(r'$(Z*(x)-Z(x))/\sigma*(x)$')
    #     plt.title('Histogram (density) of normalized error')
    #
    #     # QQ-plot with N(0,1)
    #     plt.sca(ax[2])
    #     q = np.linspace(.02, .98, 50)
    #     plt.plot(stats.norm.ppf(q), np.quantile(errn, q=q))
    #     t = stats.norm.ppf(q[-1])
    #     plt.plot([-t, t], [-t, t], ls='dashed')
    #     plt.xlabel(r'$\mathcal{N}(0,1)$')
    #     plt.ylabel(r'$(Z*(x)-Z(x))/\sigma*(x)$')
    #     plt.grid()
    #     plt.title(r'QQ-plot $\mathcal{N}(0,1)$ vs normalized err.')
    #     # plt.show()

    return v_est, v_std, crps, crps_def, pvalue, success
# ----------------------------------------------------------------------------

# ============================================================================
# Sequential Gaussian Simulation based an simple or ordinary kriging
# ============================================================================
# ----------------------------------------------------------------------------
def sgs(x, v, xu, cov_model,
        v_err_std=0.0,
        method='ordinary_kriging',
        mean_x=None,
        mean_xu=None,
        var_x=None,
        var_xu=None,
        alpha_xu=None,
        beta_xu=None,
        gamma_xu=None,
        cov_model_non_stationarity_xu_list=None,
        searchRadius=None,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        nreal=1,
        seed=None,
        pid=None,
        verbose=0,
        logger=None):
    """
    Performs Sequential Gaussian Simulation (SGS) at given location(s).

    This function does SGS at locations `xu`, given data points locations
    `x` with values `v`.

    Parameters
    ----------
    x : 2D array of floats of shape (n, d)
        data points locations, with n the number of data points and d the space
        dimension (1, 2, or 3), each row of `x` is the coordinatates of one data
        point; note: for data in 1D (`d=1`), 1D array of shape `(n,)` is accepted
        for `n` data points

    v : 1D array of floats of shape (n,)
        data points values, with n the number of data points, `v[i]` is the data
        value at location `x[i]`

    xu : 2D array of floats of shape (nu, d)
        points locations where the interpolation has to be done, with nu the
        number of points and d the space dimension (1, 2, or 3, same as for `x`),
        each row of `xu` is the coordinatates of one point;
        note: for data in 1D (`d=1`), 1D array of shape `(nu,)` is accepted 
        for `nu` points

    cov_model : :class:`CovModel1D` or :class:`CovModel2D` or :class:`CovModel3D`
        covariance model in 1D, 2D, or 3D, in same dimension as dimension of
        points (d), i.e.:

        - :class:`CovModel1D` for data in 1D (d=1)
        - :class:`CovModel2D` for data in 2D (d=2)
        - :class:`CovModel3D` for data in 3D (d=3)

        or

        - :class:`CovModel1D` interpreted as an omni-directional covariance model \
        whatever dimension of points (d);

        note: the covariance model must be stationary, however, non stationarity is
        handled: 

        - local rotation by specifying `alpha_xu` (in 2D or 3D), `beta_xu` (in 3D), `gamma_xu` (in 3D)
        - other non-stationarities by specifying `cov_model_non_stationarity_xu_list` (see below)

    v_err_std : 1D array of floats of shape (n,), or float, default: 0.0
        standard deviation of error at data points, with n the number of data points; 
        if `v_err_std` is a float, the same value is used for all data points; 
        this means that at location x[i], the data value is considered as in a Gaussian 
        distribution of mean `v[i]` and standard deviation `v_err_std[i]`

    method : str {'simple_kriging', 'ordinary_kriging'}, default: 'ordinary_kriging'
        type of kriging;
        note: if `method='ordinary_kriging'`, the parameters
        `mean_x`, `mean_xu`, `var_x`, `var_xu` are not used

    mean_x : 1D array-like of floats, or float, optional
        kriging mean value at data points `x`

        - if `mean_x` is a float, the same value is considered for any point
        - if `mean_x=None` (default): the mean of data values, i.e. mean of `v`, \
        is considered for any point

        note: if `method=ordinary_kriging`, parameter `mean_x` is ignored

    mean_xu : 1D array-like of floats, or float, optional
        kriging mean value at points `xu`

        - if `mean_xu` is a float, the same value is considered for any point
        - if `mean_xu=None` (default): the value `mean_x` (assumed to be a single \
        float) is considered for any point

        note: if `method=ordinary_kriging`, parameter `mean_xu` is ignored

    var_x : 1D array-like of floats, or float, optional
        kriging variance value at data points `x`

        - if `var_x` is a float, the same value is considered for any point
        - if `var_x=None` (default): not used  (use of covariance model only)

        note: if `method=ordinary_kriging`, parameter `var_x` is ignored

    var_xu : 1D array-like of floats, or float, optional
        kriging variance value at points `xu`

        - if `var_xu` is a float, the same value is considered for any point
        - if `var_xu=None` (default): not used  (use of covariance model only)

        note: if `method=ordinary_kriging`, parameter `var_xu` is ignored

    alpha_xu : 1D array-like of floats, or float, optional
        azimuth angle in degrees at points `xu`

        - if `alpha_xu` is a float, the same value is considered for any point
        - if `alpha_xu=None` (default): `alpha_xu=0.0` is used for any point

        note: `alpha_xu` is ignored if the covariance model is in 1D

    beta_xu : 1D array-like of floats, or float, optional
        dip angle in degrees at points `xu`

        - if `beta_xu` is a float, the same value is considered for any point
        - if `beta_xu=None` (default): `beta_xu=0.0` is used for any point

        note: `beta_xu` is ignored if the covariance model is in 1D or 2D

    gamma_xu : 1D array-like of floats, or float, optional
        dip angle in degrees at points `xu`

        - if `gamma_xu` is a float, the same value is considered for any point
        - if `gamma_xu=None` (default): `gamma_xu=0.0` is used for any point

        note: `gamma_xu` is ignored if the covariance model is in 1D or 2D

    cov_model_non_stationarity_xu_list : list, optional
        list to set non-stationarities in covariance model; each entry must be
        a tuple (or list) `cm_ns` of length 2 or 3 with:

        - `cm_ns[0]`: str: the name of the method of `cov_model` to be applied
        - `cm_ns[1]`: 1D array-like of floats, or float: \
        used to set the main parameter passed to the method:
            - if array-like: its size must be equal to `nu`, \
            (the array is reshaped if needed), values at points `xu`
            - if a float: same value at all points `xu`
        - `cm_ns[2]`: dict, optional: keyworkds arguments to be passed to the method

        Examples (with the parameter `arg` is set from `val`)
        
        - `('multiply_w', val)` will apply `cov_model.multiply_w(arg)`; 
            this multipies the weight contribution of every elementary contribution of the 
            covariance model
        - `('multiply_w', val, {'elem_ind':0})` will apply `cov_model.multiply_w(arg, elem_ind=0)`;
            this multipies the weight contribution of the elementary contribution of index 0 of the 
            covariance model
        - `('multiply_r', val)` will apply `cov_model.multiply_r(arg)`;
            this multipies the range in all direction of every elementary contribution of the
            covariance model
        - `('multiply_r', val, {'r_ind':0})` will apply `cov_model.multiply_r(arg, r_ind=0)`;
            this multipies the range in the first main direction (index 0) of every elementary 
            contribution of the covariance model
        - `('multiply_r', val, {'r_ind':0, 'elem_ind':0})` will apply `cov_model.multiply_r(arg, r_ind=0, elem_ind=0)`;
            this multipies the range in the first main direction (index 0) of the elementary 
            contribution of index 0 of the covariance model

    searchRadius : float, optional
        if specified, i.e. not `None`: radius of the search neighborhood (ellipsoid
        with same radii along each axis), i.e. the data points at distance to the 
        estimated point greater than `searchRadius` are not taken into account 
        in the kriging system; if `searchRadius` is not `None`, then 
        `searchRadiusRelative` is not used;
        by default (`searchRadius=None`): `searchRadiusRelative` is used to 
        define the search ellipsoid;

    searchRadiusRelative : float, default: 1.2
        used only if `searchRadius` is `None`;
        indicates how the search ellipsoid is limited (should be positive): let
        r_i be the ranges of the covariance model along its main axes, when
        estimating/simulating a cell x, a cell y is taken into account iff it is
        within the ellipsoid centered at x of half axes equal to
        `searchRadiusRelative` * r_i; 
        (note that the distances to the central node are computed in the axes 
        sytem supporting the covariance model and accounting for anisotropy given 
        by the ranges)

    nneighborMax : int, default: 12
        maximal number of neighbors (data points) taken into account in the
        kriging system; the data points the closest to the estimated points are
        taken into account

    nreal : int, default: 1
        number of realization(s)

    seed : int, optional
        seed for initializing random number generator

    pid : int, optional
        process id of the caller (used with multiprocessing)

    verbose : int, default: 0
        verbose mode, higher implies more printing (info)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    vu : 2D array of shape (nreal, nu)
        simulated values at points `xu`
        - vu[i, j] value of the i-th realization at point `xu[j]`
    """
    fname = 'sgs'
    if pid is not None:
        fname = f'{fname} [pid={pid}]'

    # Prevent calculation if covariance model is not stationary
    if not cov_model.is_stationary():
        err_msg = f'{fname}: `cov_model` is not stationary: {fname} cannot be applied (use the other paramters for non-stationary covariance)'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # Get dimension from x (d) and number of data points (n) and size of v
    if x is None:
        n = 0
        d = 0
        if v is not None:
            err_msg = f'{fname}: `x` is None but `v` is not None'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
    
    else:
        x = np.asarray(x)
        # Get dimension (d) from x
        if x.ndim == 1:
            # x is a 1-dimensional array
            x = x.reshape(-1, 1)
            d = 1
        else:
            # x is a 2-dimensional array
            d = x.shape[1]

        n = x.shape[0]

        # Check size of v
        if v is None:
            err_msg = f'{fname}: `x` is not None but `v` is None'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

        v = np.asarray(v).reshape(-1)
        if v.size != n:
            err_msg = f'{fname}: size of `v` is not valid'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    # Set variance of data error (from standard deviation)
    if v_err_std is None:
        v_err_std = 0.0
    v_err_var = np.asarray(v_err_std, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
    if v_err_var.size == 1:
        v_err_var = v_err_var[0] * np.ones(n)
    elif v_err_var.size != n:
        err_msg = f'{fname}: size of `v_err_std` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    v_err_var = v_err_var * v_err_var

    # Get dimension from xu (du) and number of unknown points (nu)
    if xu is None:
        nu = 0
        du = 0
    else:
        xu = np.asarray(xu)
        # Get dimension (du) from xu
        if xu.ndim == 1:
            # xu is a 1-dimensional array
            xu = xu.reshape(-1, 1)
            du = 1
        else:
            # xu is a 2-dimensional array
            du = xu.shape[1]

        nu = xu.shape[0]

    if nu == 0:
        vu = np.empty((nreal, 0), dtype='float')
        return vu

    # Here: nu > 0

    if n > 0:
        # Check dimension of x and xu
        if d != du:
            err_msg = f'{fname}: `x` and `xu` do not have the same dimension'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    else: # n == 0
        d = du # set d to du if no data point
        x = np.empty((0, d), dtype='float') # set x to an empty array with shape (0, d)
        v = np.array([], dtype='float')

    # Check that all data points (locations) are distinct
    for i in range(1, n):
        if np.any(np.isclose(np.sum((x[:i]-x[i])**2, axis=1), 0.0)):
            err_msg = f'{fname}: `x` contains duplicated entries'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    # Identify points in xu that are in x
    ind_x_in_xu = -1 * np.ones(nu, dtype='int')
    if n > 0:
        for j in range(nu):
            ind = np.isclose(np.sum((x-xu[j])**2, axis=1), 0.0)
            if np.any(ind):
                ind_x_in_xu[j] = np.where(ind)[0][0]

    # Check dimension of cov_model and set if used as omni-directional model
    if isinstance(cov_model, CovModel1D):
        omni_dir = True
    else:
        if cov_model.__class__.__name__ != f'CovModel{d}D':
            err_msg = f'{fname}: `cov_model` dimension is incompatible with dimension of points'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

        omni_dir = False

    # Covariance function
    cov_func = cov_model.func() # covariance function
    if omni_dir:
        # covariance model in 1D is used
        cov0 = cov_func(0.)[0] # covariance function at origin (lag=0)
    else:
        cov0 = cov_func(np.zeros(d))[0] # covariance function at origin (lag=0)

    # Method and mean, var
    if method == 'simple_kriging':
        ordinary_kriging = False
        if n == 0:
            mean_x = np.array([], dtype='float')
        else:
            if mean_x is None:
                mean_x = np.mean(v) * np.ones(n)
            else:
                mean_x = np.asarray(mean_x, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                if mean_x.size == 1:
                    mean_x = mean_x * np.ones(n)
                elif mean_x.size != n:
                    err_msg = f'{fname}: size of `mean_x` is not valid'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)
                
        if mean_xu is None:
            if n == 0:
                mean_xu = np.zeros(nu)
            else:
                mean_xu = np.mean(v) * np.ones(nu)
        else:
            mean_xu = np.asarray(mean_xu, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if mean_xu.size == 1:
                mean_xu = mean_xu * np.ones(nu)
            elif mean_xu.size != nu:
                err_msg = f'{fname}: size of `mean_xu` is not valid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

        if n == 0:
            if var_xu is None:
                var_x = None
            else:
                var_x = np.array([], dtype='float')

        if (var_x is None and var_xu is not None) or (var_x is not None and var_xu is None):
            err_msg = f'{fname}: `var_x` and `var_xu` must both be specified'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

        if var_x is not None:
            var_x = np.asarray(var_x, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if var_x.size == 1:
                var_x = var_x * np.ones(n)
            elif var_x.size != n:
                err_msg = f'{fname}: size of `var_x` is not valid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

            varUpdate_x = np.sqrt(var_x/cov0)

        if var_xu is not None:
            var_xu = np.asarray(var_xu, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if var_xu.size == 1:
                var_xu = var_xu * np.ones(nu)
            elif var_xu.size != nu:
                err_msg = f'{fname}: size of `var_xu` is not valid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

            varUpdate_xu = np.sqrt(var_xu/cov0)

    elif method == 'ordinary_kriging':
        if verbose > 0:
            if mean_x is not None:
                if logger:
                    logger.warning(f"{fname}: `mean_x` is ignored with `method='ordinary_kriging'`")
                else:
                    print(f"{fname}: WARNING: `mean_x` is ignored with `method='ordinary_kriging'`")
            if mean_xu is not None:
                if logger:
                    logger.warning(f"{fname}: `mean_xu` is ignored with `method='ordinary_kriging'`")
                else:
                    print(f"{fname}: WARNING: `mean_xu` is ignored with `method='ordinary_kriging'`")
            if var_x is not None:
                if logger:
                    logger.warning(f"{fname}: `var_x` is ignored with `method='ordinary_kriging'`")
                else:
                    print(f"{fname}: WARNING: `var_x` is ignored with `method='ordinary_kriging'`")
            if var_xu is not None:
                if logger:
                    logger.warning(f"{fname}: `var_xu` is ignored with `method='ordinary_kriging'`")
                else:
                    print(f"{fname}: WARNING: `var_xu` is ignored with `method='ordinary_kriging'`")

        ordinary_kriging = True
        mean_x, mean_xu, var_x, var_xu = None, None, None, None

    else:
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # WORK ON A COPY OF COVARIANCE MODEL (IN CASE IT IS ADAPTED)!
    cov_model = copyCovModel(cov_model) 
    cov_model_has_changed = False

    # Rotation given by alpha_xu, beta_xu, gamma_xu
    if alpha_xu is not None:
        if omni_dir:
            err_msg = f'{fname}: `alpha_xu` cannot be used with 1D covariance model'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    if beta_xu is not None:
        if omni_dir or d < 3:
            err_msg = f'{fname}: `beta_xu` cannot be used with 1D or 2D covariance model'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
    
    if gamma_xu is not None:
        if omni_dir or d < 3:
            err_msg = f'{fname}: `gamma_xu` cannot be used with 1D or 2D covariance model'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
        
    if omni_dir:
        rot = False
    else:
        if d == 2:
            # 2D - check only alpha
            if alpha_xu is None:
                rot = False
            else:
                alpha_xu = np.asarray(alpha_xu, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                if alpha_xu.size == 1:
                    if alpha_xu[0] == 0.0:
                        rot = False
                    else:
                        rot_mat = rotationMatrix2D(alpha_xu[0]) # rot_mat : rotation matrix for any xu[i]
                        rot = True
                        rot_mat_unique = True
                elif alpha_xu.size == nu:
                    if np.allclose(alpha_xu, alpha_xu[0]):
                        if np.isclose(alpha_xu[0], 0.0):
                            rot = False
                        else:
                            rot_mat = rotationMatrix2D(alpha_xu[0]) # rot_mat : rotation matrix for any xu[i]
                            rot = True
                            rot_mat_unique = True
                    else:
                        rot_mat = rotationMatrix2D(alpha_xu).transpose(2, 0, 1) # rot_mat[i] : rotation matrix for xu[i]
                        rot = True
                        rot_mat_unique = False
                else:
                    err_msg = f'{fname}: size of `alpha_xu` is not valid'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)

        else: # d == 3
            # 3D
            if alpha_xu is None and beta_xu is None and gamma_xu is None:
                rot = False
            else:
                if alpha_xu is not None:
                    alpha_xu = np.asarray(alpha_xu, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                    if alpha_xu.size == 1:
                        alpha_xu = alpha_xu * np.ones(nu)
                    elif alpha_xu.size != nu:
                        err_msg = f'{fname}: size of `alpha_xu` is not valid'
                        if logger: logger.error(err_msg)
                        raise CovModelError(err_msg)
                else:
                    alpha_xu = np.zeros(nu)

                if beta_xu is not None:
                    beta_xu = np.asarray(beta_xu, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                    if beta_xu.size == 1:
                        beta_xu = beta_xu * np.ones(nu)
                    elif beta_xu.size != nu:
                        err_msg = f'{fname}: size of `beta_xu` is not valid'
                        if logger: logger.error(err_msg)
                        raise CovModelError(err_msg)
                else:
                    beta_xu = np.zeros(nu)

                if gamma_xu is not None:
                    gamma_xu = np.asarray(gamma_xu, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                    if gamma_xu.size == 1:
                        gamma_xu = gamma_xu * np.ones(nu)
                    elif gamma_xu.size != nu:
                        err_msg = f'{fname}: size of `gamma_xu` is not valid'
                        if logger: logger.error(err_msg)
                        raise CovModelError(err_msg)
                else:
                    gamma_xu = np.zeros(nu)

                if np.allclose(np.vstack((alpha_xu, beta_xu, gamma_xu)).T, np.array([alpha_xu[0], beta_xu[0], gamma_xu[0]])):
                    if np.isclose(alpha_xu[0], 0.0) and np.isclose(beta_xu[0], 0.0) and np.isclose(gamma_xu[0], 0.0):
                        rot = False
                    else:
                        rot_mat = rotationMatrix3D(alpha_xu[0], beta_xu[0], gamma_xu[0]) # rot_mat : rotation matrix for any xu[i]
                        rot = True
                        rot_mat_unique = True
                else:
                    rot_mat = rotationMatrix3D(alpha_xu, beta_xu, gamma_xu).transpose(2, 0, 1) # rot_mat[i] : rotation matrix for xu[i]
                    rot = True
                    rot_mat_unique = False

    if rot:
        if d == 2:
            cov_model.set_alpha(0.0)
            cov_model_has_changed = True
        elif d == 3:
            cov_model.set_alpha(0.0)
            cov_model.set_beta(0.0)
            cov_model.set_gamma(0.0)
            cov_model_has_changed = True
        if rot_mat_unique:
            # apply rotation to data points x and points xu
            x = x.dot(rot_mat)
            xu = xu.dot(rot_mat)
            rot = False # no need rotation further

    # here: rot = True means that local rotation are applied

    # Prepare non-stationarities for integration in covariance model
    adapt_cov_model_ind = []
    recompute_cov0 = False
    recompute_dmax_ax = False
    if cov_model_non_stationarity_xu_list is not None:
        if not isinstance(cov_model_non_stationarity_xu_list, list):
            err_msg = f'{fname}: `cov_model_non_stationarity_xu_list` must be a list if not `None`'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
        for i, cm_ns_xu in enumerate(cov_model_non_stationarity_xu_list):
            if not hasattr(cm_ns_xu, '__len__') or len(cm_ns_xu) not in (2, 3):
                err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_xu_list` not valid: should be a tuple or list of length 2 or 3 containing: method name (str), val[, kwds (dict)]'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)
            val = np.asarray(cm_ns_xu[1], dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if val.size != 1:
                if val.size != nu:
                    err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_xu_list`, `(method_name, val[, kwds])`: size of `val` not valid'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)
                if np.allclose(val, val[0]):
                    val = val[:1] # of size 1
                else:
                    adapt_cov_model_ind.append(i)
                    if cm_ns_xu[0] == 'multiply_w':
                        recompute_cov0 = True
                    elif cm_ns_xu[0] == 'multiply_r':
                        recompute_dmax_ax = True
            if val.size == 1:
                method_name = cm_ns_xu[0]
                if len(cm_ns_xu) == 3:
                    kwds = cm_ns_xu[2]
                else:
                    kwds = {}
                try:
                    eval(f'cov_model.{method_name}')(val[0], logger=logger, **kwds)
                except:
                    err_msg = f'{fname}: cannot apply non-stationarity (at points) for covariance model (check parameter `cov_model_non_stationarity_xu_list`)'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)
                cov_model_has_changed = True

    adapt_cov_model = len(adapt_cov_model_ind) > 0

    if cov_model_has_changed:
        # Update - Covariance function and value at 0
        cov_func = cov_model.func() # covariance function
        if omni_dir:
            # covariance model in 1D is used
            cov0 = cov_func(0.)[0] # covariance function at origin (lag=0)
        else:
            cov0 = cov_func(np.zeros(d))[0] # covariance function at origin (lag=0)
    
    # Limited search neighborhood
    if searchRadius is not None:
        if searchRadius <= 0.0:
            err_msg = f'{fname}: search radius (isotropic neighborhood) not valid (negative)'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
        
        if omni_dir:
            dmax_ax = np.array([searchRadius], dtype='float')
        else:
            dmax_ax = searchRadius * np.ones(d)

    else:
        # use searchRadiusRelative
        if searchRadiusRelative <= 0.0:
            err_msg = f'{fname}: search radius relative (factor) not valid (negative)'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
        
        if d == 1 or omni_dir:
            dmax_ax = np.array([cov_model.r()])
        elif d == 2:
            dmax_ax = cov_model.r12()
        elif d == 3:
            dmax_ax = cov_model.r123()

        dmax_ax = searchRadiusRelative * dmax_ax

    dmax_ax_inv2 = 1.0 / (dmax_ax * dmax_ax)

    # Preparation if covariance model will be adapted
    if adapt_cov_model:
        cov_model_base = copyCovModel(cov_model)

    # Maximum number of neighbors
    if nneighborMax is None or nneighborMax < 0:
        err_msg = f'{fname}: `nneighborMax` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    mat = np.ones((nneighborMax+1, nneighborMax+1)) # allocate kriging matrix
    b = np.ones(nneighborMax+1) # allocate second member

    # Allocate memory for output
    vu = np.zeros((nreal, nu))

    # Set all points together (data points and points to be simulated)
    x_all = np.vstack((x, xu))
    v_all = np.hstack((v, np.zeros(nu)))
    v_err_var_all = np.hstack((v_err_var, np.zeros(nu)))
    if mean_x is not None:
        mean_all = np.hstack((mean_x, mean_xu))
    if var_x is not None:
        varUpdate_all = np.hstack((varUpdate_x, varUpdate_xu))

    # Set mu0 : used as mean when mean_x is None and there is no data in the neighborhood during the simulation ...
    if n == 0:
        mu0 = 0.0
    else:
        mu0 = np.mean(v)

    # Array indicating if a point in x_all can be selected for kriging
    ind_sel = np.zeros(n+nu, dtype='bool')

    if seed is None:
        seed = np.random.randint(1, 1000000)
    seed = int(seed)

    if verbose > 0:
        progress_old = 0

    for k in range(nreal):
        # Initialize random number generator
        np.random.seed(seed+k)

        # Initialize ind_sel
        ind_sel[:n] = True  # all data points can be selected
        ind_sel[n:] = False # no simulated points can be selected at the beginning

        # set path
        ind_u = np.random.permutation(nu)

        for j, jind in enumerate(ind_u):
            # Simulation at x0 = xu[jind] = xu[ind_u[j]]
            if verbose > 0:
                progress = int((j+k*nu)/(nreal*nu)*100.0)
                if progress > progress_old:
                    if logger:
                        logger.info(f'{fname}: {progress:3d}% ({k:3d} realizations done of {nreal})')
                    else:
                        print(f'{fname}: {progress:3d}% ({k:3d} realizations done of {nreal})')
                    progress_old = progress
            
            if adapt_cov_model:
                cov_model = copyCovModel(cov_model_base)
                for i in adapt_cov_model_ind:
                    cm_ns_xu = cov_model_non_stationarity_xu_list[i]
                    method_name = cm_ns_xu[0]
                    val = cm_ns_xu[1]
                    if len(cm_ns_xu) == 3:
                        kwds = cm_ns_xu[2]
                    else:
                        kwds = {}
                    try:
                        eval(f'cov_model.{method_name}')(val[jind], logger=logger, **kwds)
                    except:
                        err_msg = f'{fname}: cannot apply non-stationarity (at one point) for covariance model (check parameter `cov_model_non_stationarity_xu_list`)'
                        if logger: logger.error(err_msg)
                        raise CovModelError(err_msg)

                cov_func = cov_model.func() # update covariance function
                if recompute_cov0:
                    if omni_dir:
                        # covariance model in 1D is used
                        cov0 = cov_func(0.)[0] # covariance function at origin (lag=0)
                    else:
                        cov0 = cov_func(np.zeros(d))[0] # covariance function at origin (lag=0)

                if recompute_dmax_ax:
                    if omni_dir:
                        dmax_ax = np.array([cov_model.r()])
                    elif d == 2:
                        dmax_ax = cov_model.r12()
                    elif d == 3:
                        dmax_ax = cov_model.r123()

                    dmax_ax = searchRadiusRelative * dmax_ax
                    dmax_ax_inv2 = 1.0 / (dmax_ax * dmax_ax)

            x0 = xu[jind]
            h = x0 - x_all[ind_sel]
            if rot:
                h = h.dot(rot_mat[jind])
            d2 = np.sum(dmax_ax_inv2 * h**2, axis=1)
            ind = np.where(d2 < 1.0)[0]
            if len(ind) > nneighborMax:
                ind_s = np.argsort(d2[ind])
                ind = ind[ind_s[:nneighborMax]]
            h = h[ind] # lag between x0 and x_all[ind] (used below for right hand side of the kriging system)
            ind = ind_sel.nonzero()[0][ind] # indices in x_all
            nn = len(ind)

            if nn == 0:
                # Mean and std (by kriging) at xu[jind]
                if mean_x is not None:
                    mu = mean_all[n+jind]
                else:
                    mu = mu0
                
                std = np.sqrt(cov0)
                if var_x is not None:
                    std = varUpdate_all[n+jind]*std

            else:
                xneigh = x_all[ind]
                vneigh = v_all[ind]
                v_err_var_neigh = v_err_var_all[ind]

                # Set right hand side of the kriging system (b)
                if omni_dir:
                    # compute norm of lag
                    h = np.sqrt(np.sum(h**2, axis=1))
                b[:nn] = cov_func(h)

                if ordinary_kriging:
                    nmat = nn+1
                else:
                    nmat = nn

                # Set kriging matrix (mat) of order nmat
                for i in range(nn-1):
                    # lag between xneigh[i] and xneigh[j], j=i+1, ..., nn-1
                    h = xneigh[(i+1):] - xneigh[i]
                    if omni_dir:
                        # compute norm of lag
                        h = np.sqrt(np.sum(h**2, axis=1))
                    elif rot:
                        h = h.dot(rot_mat[jind])
                    cov_h = cov_func(h)
                    mat[i, (i+1):nn] = cov_h
                    mat[(i+1):nn, i] = cov_h
                    mat[i, i] = cov0 + v_err_var_neigh[i]

                mat[nn-1,nn-1] = cov0 + v_err_var_neigh[nn-1]

                if ordinary_kriging:
                    mat[:, nn] = 1.0
                    mat[nn, :] = 1.0
                    mat[nn,nn] = 0.0
                    b[nn] = 1.0

                # Solve the kriging system
                w = np.linalg.solve(mat[:nmat,:nmat], b[:nmat])

                # Mean and std (by kriging) at xu[jind]
                if mean_x is not None:
                    # simple kriging
                    std = np.sqrt(max(0, cov0 - np.dot(w, b[:nmat])))
                    if var_x is not None:
                        mu = mean_all[n+jind] + varUpdate_all[n+jind]*(1.0/varUpdate_all[ind]*(vneigh-mean_all[ind])).dot(w)
                        std = varUpdate_all[n+jind]*std
                    else:
                        mu = mean_all[n+jind] + (vneigh-mean_all[ind]).dot(w)
                else:
                    # ordinary kriging
                    std = np.sqrt(max(0, cov0 - np.dot(w, b[:nmat])))
                    mu = vneigh.dot(w[:nn])

            # Draw value in N(mu, std^2)
            v_all[n+jind] = np.random.normal(loc=mu, scale=std)

            if ind_x_in_xu[jind] >= 0:
                # Simulated location is a conditioning data location, then 
                # this conditioning data can no longer be selected for the next
                # simulated locations (the current simulated point will be used)
                ind_sel[ind_x_in_xu[jind]] = False

            ind_sel[n+jind] = True # the current simulated point can now be selected
            
        # Store k-th realization
        vu[k, :] = v_all[n:]

    if verbose > 0:
        if logger:
            logger.info(f'{fname}: {100:3d}% ({nreal:3d} realizations done of {nreal})')
        else:
            print(f'{fname}: {100:3d}% ({nreal:3d} realizations done of {nreal})')

    return vu
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def sgs_mp(
        x, v, xu, cov_model,
        v_err_std=0.0,
        method='ordinary_kriging',
        mean_x=None,
        mean_xu=None,
        var_x=None,
        var_xu=None,
        alpha_xu=None,
        beta_xu=None,
        gamma_xu=None,
        cov_model_non_stationarity_xu_list=None,
        searchRadius=None,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        nreal=1,
        seed=None,
        verbose=0,
        nproc=-1,
        logger=None):
    """
    Computes the same as the function :func:`sgs`, using multiprocessing.

    All the parameters except `nproc` are the same as those of the function
    :func:`sgs`.

    This function launches parallel processes [parallel calls of the
    function :func:`sgs`]; the set of realizations (specified by `nreal`) is
    distributed in a balanced way over the processes.

    The number of processes used (in parallel) is determined by the parameter `nproc` 
    (int, default: -1); a negative number (or zero), -n <= 0, can be specified 
    to use the total number of cpu(s) of the system except n; `nproc` is finally
    at maximum equal to `nreal` but at least 1 by applying:
        
    - if `nproc >= 1`, then `nproc = max(min(nproc, nreal), 1)` is used
    - if `nproc = -n <= 0`, then `nproc = max(min(nmax-n, nreal), 1)` is used, \
    where nmax is the total number of cpu(s) of the system (retrieved by \
    `multiprocessing.cpu_count()`)

    Note: if `nproc=None`, `nproc=-1` is used.

    Note: specifying a `seed` guarantees reproducible results whatever the number
    of processes used.

    See function :func:`sgs` for details.
    """
    fname = 'sgs_mp'

    # Set number of process(es): nproc
    if nproc is None:
        nproc = -1
    
    if nproc <= 0:
        nproc = max(min(multiprocessing.cpu_count() + nproc, nreal), 1)
    else:
        nproc_tmp = nproc
        nproc = max(min(int(nproc), nreal), 1)
        if verbose > 1 and nproc != nproc_tmp:
            if logger:
                logger.info(f'{fname}: number of processes has been changed (now: nproc={nproc})')
            else:
                print(f'{fname}: number of processes has been changed (now: nproc={nproc})')
    
    # Set index for distributing realizations
    q, r = np.divmod(nreal, nproc)
    ids_proc = [i*q + min(i, r) for i in range(nproc+1)]

    if verbose > 0:
        if logger:
            logger.info(f'{fname}: running `sgs` on {nproc} processes...')
        else:
            print(f'{fname}: running `sgs` on {nproc} processes...')

    # Set seed (base)
    if seed is None:
        seed = np.random.randint(1, 1000000)
    seed = int(seed)

    # Set pool of nproc workers
    pool = multiprocessing.Pool(nproc)
    out_pool = []
    for i in range(nproc):
        # Set i-th process
        kwargs = dict(
                    v_err_std=v_err_std,
                    method=method,
                    mean_x=mean_x, 
                    mean_xu=mean_xu, 
                    var_x=var_x, 
                    var_xu=var_xu,
                    alpha_xu=alpha_xu, 
                    beta_xu=beta_xu, 
                    gamma_xu=gamma_xu,
                    cov_model_non_stationarity_xu_list=cov_model_non_stationarity_xu_list,
                    searchRadius=searchRadius,
                    searchRadiusRelative=searchRadiusRelative, 
                    nneighborMax=nneighborMax,
                    nreal=ids_proc[i+1]-ids_proc[i], 
                    seed=seed+ids_proc[i],
                    pid=i,
                    verbose=verbose*(i==0),
                    logger=logger)
        out_pool.append(pool.apply_async(sgs, args=(x, v, xu, cov_model), kwds=kwargs))

    # Properly end working process
    pool.close() # Prevents any more tasks from being submitted to the pool,
    pool.join()  # then, wait for the worker processes to exit.

    # Get result from each process
    out = [w.get() for w in out_pool]
    if np.any([x is None for x in out]):
        err_msg = f'{fname}: an error occured on a process (worker)'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    vu = np.vstack(out)

    return vu
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def sgs_at_inequality_data_points(
        x, v, x_ineq, cov_model,
        v_err_std=0.0,
        v_ineq_min=None, v_ineq_max=None,
        method='ordinary_kriging',
        mean_x=None,
        mean_x_ineq=None,
        var_x=None,
        var_x_ineq=None,
        alpha_x_ineq=None,
        beta_x_ineq=None,
        gamma_x_ineq=None,
        cov_model_non_stationarity_x_ineq_list=None,
        searchRadius=None,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        nGibbsSamplerPath=50,
        nreal=1,
        seed=None,
        pid=None,
        verbose=0,
        logger=None):
    """
    Performs Sequential Gaussian Simulation (SGS) at inequality data point(s).

    This function does SGS at locations `x_ineq`, given data points locations
    `x` with values `v`.

    Parameters
    ----------
    x : 2D array of floats of shape (n, d)
        data points locations, with n the number of data points and d the space
        dimension (1, 2, or 3), each row of `x` is the coordinatates of one data
        point; note: for data in 1D (`d=1`), 1D array of shape `(n,)` is accepted
        for `n` data points

    v : 1D array of floats of shape (n,)
        data points values, with n the number of data points, `v[i]` is the data
        value at location `x[i]`

    x_ineq : 2D array of floats of shape (n_ineq, d)
        points locations where the interpolation has to be done, with n_ineq the
        number of points and d the space dimension (1, 2, or 3, same as for `x`),
        each row of `x_ineq` is the coordinatates of one point;
        note: for data in 1D (`d=1`), 1D array of shape `(n_ineq,)` is accepted 
        for `n_ineq` points

    cov_model : :class:`CovModel1D` or :class:`CovModel2D` or :class:`CovModel3D`
        covariance model in 1D, 2D, or 3D, in same dimension as dimension of
        points (d), i.e.:

        - :class:`CovModel1D` for data in 1D (d=1)
        - :class:`CovModel2D` for data in 2D (d=2)
        - :class:`CovModel3D` for data in 3D (d=3)

        or

        - :class:`CovModel1D` interpreted as an omni-directional covariance model \
        whatever dimension of points (d);

        note: the covariance model must be stationary, however, non stationarity is
        handled: 

        - local rotation by specifying `alpha_x_ineq` (in 2D or 3D), `beta_x_ineq` (in 3D), `gamma_x_ineq` (in 3D)
        - other non-stationarities by specifying `cov_model_non_stationarity_x_ineq_list` (see below)

    v_err_std : 1D array of floats of shape (n,), or float, default: 0.0
        standard deviation of error at data points, with n the number of data points; 
        if `v_err_std` is a float, the same value is used for all data points; 
        this means that at location x[i], the data value is considered as in a Gaussian 
        distribution of mean `v[i]` and standard deviation `v_err_std[i]`

    v_ineq_min : 1D array of floats of shape (n_ineq,), or float, optional
        minimal value (lower bound) for inequality data points, with n_ineq the 
        number of inequality data points (`v_ineq_min[i]` is the value for the 
        location `x_ineq[i]`); if `v_ineq_min` is a float, the same value 
        is used for all inequality data points; if `v_ineq_min=None` (default),
        no minimal value is considered for any inequality data point;
        note: `v_ineq_min[i]` set to `np.nan` or `-np.inf` means that there is 
        no minimal value for point `x_ineq[i]`
        
    v_ineq_max : 1D array of floats of shape (n_ineq,), or float, optional
        maximal value (upper bound) for inequality data points, with n_ineq the 
        number of inequality data points (`v_ineq_max[i]` is the value for the 
        location `x_ineq[i]`); if `v_ineq_max` is a float, the same value 
        is used for all inequality data points; if `v_ineq_max=None` (default),
        no maximal value is considered for any inequality data point;
        note: `v_ineq_max[i]` set to `np.nan` or `np.inf` means that there is 
        no maximal value for point `x_ineq[i]`

    method : str {'simple_kriging', 'ordinary_kriging'}, default: 'ordinary_kriging'
        type of kriging;
        note: if `method='ordinary_kriging'`, the parameters
        `mean_x`, `mean_x_ineq`, `var_x`, `var_x_ineq` are not used

    mean_x : 1D array-like of floats, or float, optional
        kriging mean value at data points `x`

        - if `mean_x` is a float, the same value is considered for any point
        - if `mean_x=None` (default): the mean of data values, i.e. mean of `v`, \
        is considered for any point

        note: if `method=ordinary_kriging`, parameter `mean_x` is ignored

    mean_x_ineq : 1D array-like of floats, or float, optional
        kriging mean value at points `x_ineq`

        - if `mean_x_ineq` is a float, the same value is considered for any point
        - if `mean_x_ineq=None` (default): the value `mean_x` (assumed to be a single \
        float) is considered for any point

        note: if `method=ordinary_kriging`, parameter `mean_x_ineq` is ignored

    var_x : 1D array-like of floats, or float, optional
        kriging variance value at data points `x`

        - if `var_x` is a float, the same value is considered for any point
        - if `var_x=None` (default): not used  (use of covariance model only)

        note: if `method=ordinary_kriging`, parameter `var_x` is ignored

    var_x_ineq : 1D array-like of floats, or float, optional
        kriging variance value at points `x_ineq`

        - if `var_x_ineq` is a float, the same value is considered for any point
        - if `var_x_ineq=None` (default): not used  (use of covariance model only)

        note: if `method=ordinary_kriging`, parameter `var_x_ineq` is ignored

    alpha_x_ineq : 1D array-like of floats, or float, optional
        azimuth angle in degrees at points `x_ineq`

        - if `alpha_x_ineq` is a float, the same value is considered for any point
        - if `alpha_x_ineq=None` (default): `alpha_x_ineq=0.0` is used for any point

        note: `alpha_x_ineq` is ignored if the covariance model is in 1D

    beta_x_ineq : 1D array-like of floats, or float, optional
        dip angle in degrees at points `x_ineq`

        - if `beta_x_ineq` is a float, the same value is considered for any point
        - if `beta_x_ineq=None` (default): `beta_x_ineq=0.0` is used for any point

        note: `beta_x_ineq` is ignored if the covariance model is in 1D or 2D

    gamma_x_ineq : 1D array-like of floats, or float, optional
        dip angle in degrees at points `x_ineq`

        - if `gamma_x_ineq` is a float, the same value is considered for any point
        - if `gamma_x_ineq=None` (default): `gamma_x_ineq=0.0` is used for any point

        note: `gamma_x_ineq` is ignored if the covariance model is in 1D or 2D

    cov_model_non_stationarity_x_ineq_list : list, optional
        list to set non-stationarities in covariance model; each entry must be
        a tuple (or list) `cm_ns` of length 2 or 3 with:

        - `cm_ns[0]`: str: the name of the method of `cov_model` to be applied
        - `cm_ns[1]`: 1D array-like of floats, or float: \
        used to set the main parameter passed to the method:
            - if array-like: its size must be equal to `n_ineq`, \
            (the array is reshaped if needed), values at points `x_ineq`
            - if a float: same value at all points `x_ineq`
        - `cm_ns[2]`: dict, optional: keyworkds arguments to be passed to the method

        Examples (with the parameter `arg` is set from `val`)
        
        - `('multiply_w', val)` will apply `cov_model.multiply_w(arg)`; 
            this multipies the weight contribution of every elementary contribution of the 
            covariance model
        - `('multiply_w', val, {'elem_ind':0})` will apply `cov_model.multiply_w(arg, elem_ind=0)`;
            this multipies the weight contribution of the elementary contribution of index 0 of the 
            covariance model
        - `('multiply_r', val)` will apply `cov_model.multiply_r(arg)`;
            this multipies the range in all direction of every elementary contribution of the
            covariance model
        - `('multiply_r', val, {'r_ind':0})` will apply `cov_model.multiply_r(arg, r_ind=0)`;
            this multipies the range in the first main direction (index 0) of every elementary 
            contribution of the covariance model
        - `('multiply_r', val, {'r_ind':0, 'elem_ind':0})` will apply `cov_model.multiply_r(arg, r_ind=0, elem_ind=0)`;
            this multipies the range in the first main direction (index 0) of the elementary 
            contribution of index 0 of the covariance model

    searchRadius : float, optional
        if specified, i.e. not `None`: radius of the search neighborhood (ellipsoid
        with same radii along each axis), i.e. the data points at distance to the 
        estimated point greater than `searchRadius` are not taken into account 
        in the kriging system; if `searchRadius` is not `None`, then 
        `searchRadiusRelative` is not used;
        by default (`searchRadius=None`): `searchRadiusRelative` is used to 
        define the search ellipsoid;

    searchRadiusRelative : float, default: 1.2
        used only if `searchRadius` is `None`;
        indicates how the search ellipsoid is limited (should be positive): let
        r_i be the ranges of the covariance model along its main axes, when
        estimating/simulating a cell x, a cell y is taken into account iff it is
        within the ellipsoid centered at x of half axes equal to
        `searchRadiusRelative` * r_i; 
        (note that the distances to the central node are computed in the axes 
        sytem supporting the covariance model and accounting for anisotropy given 
        by the ranges)

    nneighborMax : int, default: 12
        maximal number of neighbors (data points) taken into account in the
        kriging system; the data points the closest to the estimated points are
        taken into account

    nGibbsSamplerPath : int, default: 50
        number of Gibbs sampler paths for simulating values at inequality data points
        
    nreal : int, default: 1
        number of realization(s)

    seed : int, optional
        seed for initializing random number generator

    pid : int, optional
        process id of the caller (used with multiprocessing)

    verbose : int, default: 0
        verbose mode, higher implies more printing (info)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    v_ineq : 2D array of shape (nreal, n_ineq)
        simulated values at points `x_ineq`
        - v_ineq[i, j] value of the i-th realization at point `x_ineq[j]`
    """
    fname = 'sgs_at_inequality_data_points'
    if pid is not None:
        fname = f'{fname} [pid={pid}]'

    # Prevent calculation if covariance model is not stationary
    if not cov_model.is_stationary():
        err_msg = f'{fname}: `cov_model` is not stationary: {fname} cannot be applied (use the other paramters for non-stationary covariance)'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # Get dimension from x (d) and number of data points (n) and size of v
    if x is None:
        n = 0
        d = 0
        if v is not None:
            err_msg = f'{fname}: `x` is None but `v` is not None'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
    
    else:
        x = np.asarray(x)
        # Get dimension (d) from x
        if x.ndim == 1:
            # x is a 1-dimensional array
            x = x.reshape(-1, 1)
            d = 1
        else:
            # x is a 2-dimensional array
            d = x.shape[1]

        n = x.shape[0]

        # Check size of v
        if v is None:
            err_msg = f'{fname}: `x` is not None but `v` is None'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

        v = np.asarray(v).reshape(-1)
        if v.size != n:
            err_msg = f'{fname}: size of `v` is not valid'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    # Set variance of data error (from standard deviation)
    if v_err_std is None:
        v_err_std = 0.0
    v_err_var = np.asarray(v_err_std, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
    if v_err_var.size == 1:
        v_err_var = v_err_var[0] * np.ones(n)
    elif v_err_var.size != n:
        err_msg = f'{fname}: size of `v_err_std` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    v_err_var = v_err_var * v_err_var

    # Get dimension from x_ineq (d_ineq) and number of unknown points (n_ineq)
    if x_ineq is None:
        n_ineq = 0
        d_ineq = 0
    else:
        x_ineq = np.asarray(x_ineq)
        # Get dimension (d_ineq) from x_ineq
        if x_ineq.ndim == 1:
            # x_ineq is a 1-dimensional array
            x_ineq = x_ineq.reshape(-1, 1)
            d_ineq = 1
        else:
            # x_ineq is a 2-dimensional array
            d_ineq = x_ineq.shape[1]

        n_ineq = x_ineq.shape[0]

    if n_ineq == 0:
        v_ineq = np.empty((nreal, 0), dtype='float')
        return v_ineq

    # Here: n_ineq > 0

    if n > 0:
        # Check dimension of x and x_ineq
        if d != d_ineq:
            err_msg = f'{fname}: `x` and `x_ineq` do not have the same dimension'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    else: # n == 0
        d = d_ineq # set d to d_ineq if no data point
        x = np.empty((0, d), dtype='float') # set x to an empty array with shape (0, d)
        v = np.array([], dtype='float')

    # Check that all data points (locations) are distinct
    for i in range(1, n):
        if np.any(np.isclose(np.sum((x[:i]-x[i])**2, axis=1), 0.0)):
            err_msg = f'{fname}: `x` contains duplicated entries'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    # Check that all inequality data points (locations) are distinct
    for i in range(1, n_ineq):
        if np.any(np.isclose(np.sum((x_ineq[:i]-x_ineq[i])**2, axis=1), 0.0)):
            err_msg = f'{fname}: `x_ineq` contains duplicated entries'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    # Check that all data points and inequality data points (locations) are distinct
    for i in range(1, n):
        if np.any(np.isclose(np.sum((x_ineq-x[i])**2, axis=1), 0.0)):
            err_msg = f'{fname}: `x` and `x_ineq` contains same entries'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    # Check (and set) v_ineq_min
    if v_ineq_min is None:
        v_ineq_min = np.full((n_ineq, ), -np.inf)
    else:        
        v_ineq_min = np.asarray(v_ineq_min).reshape(-1)
        if v_ineq_min.size == 1:
            v_ineq_min = v_ineq_min * np.ones(n_ineq)
        elif v_ineq_min.size != n_ineq:
            err_msg = f'{fname}: size of `v_ineq_min` is not valid'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

        v_ineq_min[np.isnan(v_ineq_min)] = -np.inf
    
    # Check (and set) v_ineq_max
    if v_ineq_max is None:
        v_ineq_max = np.full((n_ineq, ), np.inf)
    else:        
        v_ineq_max = np.asarray(v_ineq_max).reshape(-1)
        if v_ineq_max.size == 1:
            v_ineq_max = v_ineq_max * np.ones(n_ineq)
        elif v_ineq_max.size != n_ineq:
            err_msg = f'{fname}: size of `v_ineq_max` is not valid'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

        v_ineq_max[np.isnan(v_ineq_max)] = np.inf

    # Check consistency of v_ineq_min and v_ineq_max
    if np.any(v_ineq_max <= v_ineq_min):
        err_msg = f'{fname}: `v_ineq_min` and `v_ineq_max` are not consistent'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # Check dimension of cov_model and set if used as omni-directional model
    if isinstance(cov_model, CovModel1D):
        omni_dir = True
    else:
        if cov_model.__class__.__name__ != f'CovModel{d}D':
            err_msg = f'{fname}: `cov_model` dimension is incompatible with dimension of points'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

        omni_dir = False

    # Covariance function
    cov_func = cov_model.func() # covariance function
    if omni_dir:
        # covariance model in 1D is used
        cov0 = cov_func(0.)[0] # covariance function at origin (lag=0)
    else:
        cov0 = cov_func(np.zeros(d))[0] # covariance function at origin (lag=0)

    # Method and mean, var
    if method == 'simple_kriging':
        ordinary_kriging = False
        if n == 0:
            mean_x = np.array([], dtype='float')
        else:
            if mean_x is None:
                mean_x = np.mean(v) * np.ones(n)
            else:
                mean_x = np.asarray(mean_x, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                if mean_x.size == 1:
                    mean_x = mean_x * np.ones(n)
                elif mean_x.size != n:
                    err_msg = f'{fname}: size of `mean_x` is not valid'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)

        if mean_x_ineq is None:
            if n == 0:
                mean_x_ineq = np.zeros(n_ineq)
            else:
                mean_x_ineq = np.mean(v) * np.ones(n_ineq)
        else:
            mean_x_ineq = np.asarray(mean_x_ineq, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if mean_x_ineq.size == 1:
                mean_x_ineq = mean_x_ineq * np.ones(n_ineq)
            elif mean_x_ineq.size != n_ineq:
                err_msg = f'{fname}: size of `mean_x_ineq` is not valid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

        if n == 0:
            if var_x_ineq is None:
                var_x = None
            else:
                var_x = np.array([], dtype='float')

        if (var_x is None and var_x_ineq is not None) or (var_x is not None and var_x_ineq is None):
            err_msg = f'{fname}: `var_x` and `var_x_ineq` must both be specified'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

        if var_x is not None:
            var_x = np.asarray(var_x, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if var_x.size == 1:
                var_x = var_x * np.ones(n)
            elif var_x.size != n:
                err_msg = f'{fname}: size of `var_x` is not valid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

            varUpdate_x = np.sqrt(var_x/cov0)

        if var_x_ineq is not None:
            var_x_ineq = np.asarray(var_x_ineq, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if var_x_ineq.size == 1:
                var_x_ineq = var_x_ineq * np.ones(n_ineq)
            elif var_x_ineq.size != n_ineq:
                err_msg = f'{fname}: size of `var_x_ineq` is not valid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

            varUpdate_x_ineq = np.sqrt(var_x_ineq/cov0)

    elif method == 'ordinary_kriging':
        if verbose > 0:
            if mean_x is not None:
                if logger:
                    logger.warning(f"{fname}: `mean_x` is ignored with `method='ordinary_kriging'`")
                else:
                    print(f"{fname}: WARNING: `mean_x` is ignored with `method='ordinary_kriging'`")
            if mean_x_ineq is not None:
                if logger:
                    logger.warning(f"{fname}: `mean_x_ineq` is ignored with `method='ordinary_kriging'`")
                else:
                    print(f"{fname}: WARNING: `mean_x_ineq` is ignored with `method='ordinary_kriging'`")
            if var_x is not None:
                if logger:
                    logger.warning(f"{fname}: `var_x` is ignored with `method='ordinary_kriging'`")
                else:
                    print(f"{fname}: WARNING: `var_x` is ignored with `method='ordinary_kriging'`")
            if var_x_ineq is not None:
                if logger:
                    logger.warning(f"{fname}: `var_x_ineq` is ignored with `method='ordinary_kriging'`")
                else:
                    print(f"{fname}: WARNING: `var_x_ineq` is ignored with `method='ordinary_kriging'`")

        ordinary_kriging = True
        mean_x, mean_x_ineq, var_x, var_x_ineq = None, None, None, None

    else:
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # WORK ON A COPY OF COVARIANCE MODEL (IN CASE IT IS ADAPTED)!
    cov_model = copyCovModel(cov_model) 
    cov_model_has_changed = False

    # Rotation given by alpha_x_ineq, beta_x_ineq, gamma_x_ineq
    if alpha_x_ineq is not None:
        if omni_dir:
            err_msg = f'{fname}: `alpha_x_ineq` cannot be used with 1D covariance model'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    if beta_x_ineq is not None:
        if omni_dir or d < 3:
            err_msg = f'{fname}: `beta_x_ineq` cannot be used with 1D or 2D covariance model'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
    
    if gamma_x_ineq is not None:
        if omni_dir or d < 3:
            err_msg = f'{fname}: `gamma_x_ineq` cannot be used with 1D or 2D covariance model'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    if omni_dir:
        rot = False
    else:
        if d == 2:
            # 2D - check only alpha
            if alpha_x_ineq is None:
                rot = False
            else:
                alpha_x_ineq = np.asarray(alpha_x_ineq, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                if alpha_x_ineq.size == 1:
                    if alpha_x_ineq[0] == 0.0:
                        rot = False
                    else:
                        rot_mat = rotationMatrix2D(alpha_x_ineq[0]) # rot_mat : rotation matrix for any x_ineq[i]
                        rot = True
                        rot_mat_unique = True
                elif alpha_x_ineq.size == n_ineq:
                    if np.allclose(alpha_x_ineq, alpha_x_ineq[0]):
                        if np.isclose(alpha_x_ineq[0], 0.0):
                            rot = False
                        else:
                            rot_mat = rotationMatrix2D(alpha_x_ineq[0]) # rot_mat : rotation matrix for any x_ineq[i]
                            rot = True
                            rot_mat_unique = True
                    else:
                        rot_mat = rotationMatrix2D(alpha_x_ineq).transpose(2, 0, 1) # rot_mat[i] : rotation matrix for xineq[i]
                        rot = True
                        rot_mat_unique = False
                else:
                    err_msg = f'{fname}: size of `alpha_x_ineq` is not valid'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)

        else: # d == 3
            # 3D
            if alpha_x_ineq is None and beta_x_ineq is None and gamma_x_ineq is None:
                rot = False
            else:
                if alpha_x_ineq is not None:
                    alpha_x_ineq = np.asarray(alpha_x_ineq, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                    if alpha_x_ineq.size == 1:
                        alpha_x_ineq = alpha_x_ineq * np.ones(n_ineq)
                    elif alpha_x_ineq.size != n_ineq:
                        err_msg = f'{fname}: size of `alpha_x_ineq` is not valid'
                        if logger: logger.error(err_msg)
                        raise CovModelError(err_msg)
                else:
                    alpha_x_ineq = np.zeros(n_ineq)

                if beta_x_ineq is not None:
                    beta_x_ineq = np.asarray(beta_x_ineq, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                    if beta_x_ineq.size == 1:
                        beta_x_ineq = beta_x_ineq * np.ones(n_ineq)
                    elif beta_x_ineq.size != n_ineq:
                        err_msg = f'{fname}: size of `beta_x_ineq` is not valid'
                        if logger: logger.error(err_msg)
                        raise CovModelError(err_msg)
                else:
                    beta_x_ineq = np.zeros(n_ineq)

                if gamma_x_ineq is not None:
                    gamma_x_ineq = np.asarray(gamma_x_ineq, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                    if gamma_x_ineq.size == 1:
                        gamma_x_ineq = gamma_x_ineq * np.ones(n_ineq)
                    elif gamma_x_ineq.size != n_ineq:
                        err_msg = f'{fname}: size of `gamma_x_ineq` is not valid'
                        if logger: logger.error(err_msg)
                        raise CovModelError(err_msg)
                else:
                    gamma_x_ineq = np.zeros(n_ineq)
                
                if np.allclose(np.vstack((alpha_x_ineq, beta_x_ineq, gamma_x_ineq)).T, np.array([alpha_x_ineq[0], beta_x_ineq[0], gamma_x_ineq[0]])):
                    if np.isclose(alpha_x_ineq[0], 0.0) and np.isclose(beta_x_ineq[0], 0.0) and np.isclose(gamma_x_ineq[0], 0.0):
                        rot = False
                    else:
                        rot_mat = rotationMatrix3D(alpha_x_ineq[0], beta_x_ineq[0], gamma_x_ineq[0]) # rot_mat : rotation matrix for any x_ineq[i]
                        rot = True
                        rot_mat_unique = True
                else:
                    rot_mat = rotationMatrix3D(alpha_x_ineq, beta_x_ineq, gamma_x_ineq).transpose(2, 0, 1) # rot_mat[i] : rotation matrix for x_ineq[i]
                    rot = True
                    rot_mat_unique = False

    if rot:
        if d == 2:
            cov_model.set_alpha(0.0)
            cov_model_has_changed = True
        elif d == 3:
            cov_model.set_alpha(0.0)
            cov_model.set_beta(0.0)
            cov_model.set_gamma(0.0)
            cov_model_has_changed = True
        if rot_mat_unique:
            # apply rotation to data points x and points x_ineq
            x = x.dot(rot_mat)
            x_ineq = x_ineq.dot(rot_mat)
            rot = False # no need rotation further

    # here: rot = True means that local rotation are applied

    # Prepare non-stationarities for integration in covariance model
    adapt_cov_model_ind = []
    recompute_cov0 = False
    recompute_dmax_ax = False
    if cov_model_non_stationarity_x_ineq_list is not None:
        if not isinstance(cov_model_non_stationarity_x_ineq_list, list):
            err_msg = f'{fname}: `cov_model_non_stationarity_x_ineq_list` must be a list if not `None`'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
        for i, cm_ns_x_ineq in enumerate(cov_model_non_stationarity_x_ineq_list):
            if not hasattr(cm_ns_x_ineq, '__len__') or len(cm_ns_x_ineq) not in (2, 3):
                err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_x_ineq_list` not valid: should be a tuple or list of length 2 or 3 containing: method name (str), val[, kwds (dict)]'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)
            val = np.asarray(cm_ns_x_ineq[1], dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if val.size != 1:
                if val.size != n_ineq:
                    err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_x_ineq_list`, `(method_name, val[, kwds])`: size of `val` not valid'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)
                if np.allclose(val, val[0]):
                    val = val[:1] # of size 1
                else:
                    adapt_cov_model_ind.append(i)
                    if cm_ns_x_ineq[0] == 'multiply_w':
                        recompute_cov0 = True
                    elif cm_ns_x_ineq[0] == 'multiply_r':
                        recompute_dmax_ax = True
            if val.size == 1:
                method_name = cm_ns_x_ineq[0]
                if len(cm_ns_x_ineq) == 3:
                    kwds = cm_ns_x_ineq[2]
                else:
                    kwds = {}
                try:
                    eval(f'cov_model.{method_name}')(val[0], logger=logger, **kwds)
                except:
                    err_msg = f'{fname}: cannot apply non-stationarity (at points) for covariance model (check parameter `cov_model_non_stationarity_x_ineq_list`)'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)
                cov_model_has_changed = True

    adapt_cov_model = len(adapt_cov_model_ind) > 0

    if cov_model_has_changed:
        # Update - Covariance function and value at 0
        cov_func = cov_model.func() # covariance function
        if omni_dir:
            # covariance model in 1D is used
            cov0 = cov_func(0.)[0] # covariance function at origin (lag=0)
        else:
            cov0 = cov_func(np.zeros(d))[0] # covariance function at origin (lag=0)
    
    # Limited search neighborhood
    if searchRadius is not None:
        if searchRadius <= 0.0:
            err_msg = f'{fname}: search radius (isotropic neighborhood) not valid (negative)'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
        
        if omni_dir:
            dmax_ax = np.array([searchRadius], dtype='float')
        else:
            dmax_ax = searchRadius * np.ones(d)

    else:
        # use searchRadiusRelative
        if searchRadiusRelative <= 0.0:
            err_msg = f'{fname}: search radius relative (factor) not valid (negative)'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
        
        if d == 1 or omni_dir:
            dmax_ax = np.array([cov_model.r()])
        elif d == 2:
            dmax_ax = cov_model.r12()
        elif d == 3:
            dmax_ax = cov_model.r123()

        dmax_ax = searchRadiusRelative * dmax_ax

    dmax_ax_inv2 = 1.0 / (dmax_ax * dmax_ax)

    # Preparation if covariance model will be adapted
    if adapt_cov_model:
        cov_model_base = copyCovModel(cov_model)

    # Maximum number of neighbors
    if nneighborMax is None or nneighborMax < 0:
        err_msg = f'{fname}: `nneighborMax` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    mat = np.ones((nneighborMax+1, nneighborMax+1)) # allocate kriging matrix
    b = np.ones(nneighborMax+1) # allocate second member

     # Min value for std
    std_min = 1.e-10
    eps = 1.e-8

    # Set all points together (data points and inequality data points)
    x_all = np.vstack((x, x_ineq))
    # v_all = np.hstack((v, np.zeros(n_ineq)))
    v_err_var_all = np.hstack((v_err_var, np.zeros(n_ineq)))

    # Precompute solution of kriging for 2nd, 3rd, ... paths (Gibbs sampler)
    # ----------------------------------------------------------------------
    if verbose > 0:
        if logger:
            logger.info(f'{fname}: precomputing solution of kriging systems for all Gibbs sampler paths from the 2nd path...')
        else:
            print(f'{fname}: precomputing solution of kriging systems for all Gibbs sampler paths from the 2nd path...')

    ind_sel = np.ones(n+n_ineq, dtype='bool')

    ind_list = []
    w_list = []
    std_list = []

    for j, x0 in enumerate(x_all[n:]):
        # x0 is the location x_ineq[j]
        ind_sel[n+j] = False

        if adapt_cov_model:
            cov_model = copyCovModel(cov_model_base)
            for i in adapt_cov_model_ind:
                cm_ns_x_ineq = cov_model_non_stationarity_x_ineq_list[i]
                method_name = cm_ns_x_ineq[0]
                val = cm_ns_x_ineq[1]
                if len(cm_ns_x_ineq) == 3:
                    kwds = cm_ns_x_ineq[2]
                else:
                    kwds = {}
                try:
                    eval(f'cov_model.{method_name}')(val[j], logger=logger, **kwds)
                except:
                    err_msg = f'{fname}: cannot apply non-stationarity (at one point) for covariance model (check parameter `cov_model_non_stationarity_x_ineq_list`)'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)

            cov_func = cov_model.func() # update covariance function
            if recompute_cov0:
                if omni_dir:
                    # covariance model in 1D is used
                    cov0 = cov_func(0.)[0] # covariance function at origin (lag=0)
                else:
                    cov0 = cov_func(np.zeros(d))[0] # covariance function at origin (lag=0)

            if recompute_dmax_ax:
                if omni_dir:
                    dmax_ax = np.array([cov_model.r()])
                elif d == 2:
                    dmax_ax = cov_model.r12()
                elif d == 3:
                    dmax_ax = cov_model.r123()

                dmax_ax = searchRadiusRelative * dmax_ax
                dmax_ax_inv2 = 1.0 / (dmax_ax * dmax_ax)
        
        h = x0 - x_all
        if rot:
            h = h.dot(rot_mat[j])
        d2 = np.sum(dmax_ax_inv2 * h**2, axis=1)
        ind = np.where(np.vstack((d2 < 1.0, ind_sel)).all(axis=0))[0]
        if len(ind) > nneighborMax:
            ind_s = np.argsort(d2[ind])
            ind = ind[ind_s[:nneighborMax]]
        
        nn = len(ind)
        if nn == 0:
            w = np.zeros(0)

            std = np.sqrt(cov0)
            if var_x is not None:
                std = varUpdate_x_ineq[j]*std
        else:
            xneigh = x_all[ind]
            v_err_var_neigh = v_err_var_all[ind]

            # Set right hand side of the kriging system (b)
            h = h[ind]
            if omni_dir:
                # compute norm of lag
                h = np.sqrt(np.sum(h**2, axis=1))
            b[:nn] = cov_func(h)

            if ordinary_kriging:
                nmat = nn+1
            else:
                nmat = nn

            # Set kriging matrix (mat) of order nmat
            for i in range(nn-1):
                # lag between xneigh[i] and xneigh[j], j=i+1, ..., nn-1
                h = xneigh[(i+1):] - xneigh[i]
                if omni_dir:
                    # compute norm of lag
                    h = np.sqrt(np.sum(h**2, axis=1))
                elif rot:
                    h = h.dot(rot_mat[j])
                cov_h = cov_func(h)
                mat[i, (i+1):nn] = cov_h
                mat[(i+1):nn, i] = cov_h
                mat[i, i] = cov0 + v_err_var_neigh[i]

            mat[nn-1,nn-1] = cov0 + v_err_var_neigh[nn-1]

            if ordinary_kriging:
                mat[:, nn] = 1.0
                mat[nn, :] = 1.0
                mat[nn,nn] = 0.0
                b[nn] = 1.0

            # Solve the kriging system
            w = np.linalg.solve(mat[:nmat,:nmat], b[:nmat])

            # Std (by kriging) at x_ineq[j]
            if mean_x is not None:
                # simple kriging
                std = np.sqrt(max(0, cov0 - np.dot(w, b[:nmat])))
                if var_x is not None:
                    std = varUpdate_x_ineq[j]*std
            else:
                # ordinary kriging
                std = np.sqrt(max(0, cov0 - np.dot(w, b[:nmat])))

            if np.isnan(std):
                std = np.sqrt(cov0)
                if var_x is not None:
                    std = varUpdate_all[n+ind_ineq[j]]*std
            
            if np.isclose(std, 0):
                std = std_min

        ind_list.append(ind)
        w_list.append(w)
        std_list.append(std)

        ind_sel[n+j] = True

    # Allocate memory for output
    v_ineq = np.zeros((nreal, n_ineq))

    # Set all points together (data points and inequality data points)
    # x_all = np.vstack((x, x_ineq))
    v_all = np.hstack((v, np.zeros(n_ineq)))
    # v_err_var_all = np.hstack((v_err_var, np.zeros(n_ineq)))
    if mean_x is not None:
        mean_all = np.hstack((mean_x, mean_x_ineq))
    if var_x is not None:
        varUpdate_all = np.hstack((varUpdate_x, varUpdate_x_ineq))

    # Set mu0 : used as mean when mean_x is None and there is no data in the neighborhood during the simulation ...
    if n == 0:
        mu0 = 0.0
    else:
        mu0 = np.mean(v)

    # Array indicating if a point in x_all can be selected for kriging
    ind_sel = np.zeros(n+n_ineq, dtype='bool')
    
    if seed is None:
        seed = np.random.randint(1, 1000000)
    seed = int(seed)

    if verbose > 0:
        progress_old = 0

    for k in range(nreal):
        # Initialize random number generator
        np.random.seed(seed+k)

        # Initialize ind_sel
        ind_sel[:n] = True  # all data points can be selected
        ind_sel[n:] = False # no simulated points can be selected at the beginning

        # Firt path (Gibbs sampler)
        # -------------------------
        # set path
        ind_ineq = np.random.permutation(n_ineq)
        
        nGibbs = 0 
        for j, jind in enumerate(ind_ineq):
            # Simulation at x0 = x_ineq[jind] = x_ineq[ind_ineq[j]]
            if verbose > 0:
                progress = int((j+n_ineq*(k*nGibbsSamplerPath+nGibbs))/(nreal*n_ineq*nGibbsSamplerPath)*100.0)
                if progress > progress_old:
                    if logger:
                        logger.info(f'{fname}: {progress:3d}% ({k:3d} realizations done of {nreal})')
                    else:
                        print(f'{fname}: {progress:3d}% ({k:3d} realizations done of {nreal})')
                    progress_old = progress

            if adapt_cov_model:
                cov_model = copyCovModel(cov_model_base)
                for i in adapt_cov_model_ind:
                    cm_ns_x_ineq = cov_model_non_stationarity_x_ineq_list[i]
                    method_name = cm_ns_x_ineq[0]
                    val = cm_ns_x_ineq[1]
                    if len(cm_ns_x_ineq) == 3:
                        kwds = cm_ns_x_ineq[2]
                    else:
                        kwds = {}
                    try:
                        eval(f'cov_model.{method_name}')(val[jind], logger=logger, **kwds)
                    except:
                        err_msg = f'{fname}: cannot apply non-stationarity (at one point) for covariance model (check parameter `cov_model_non_stationarity_x_ineq_list`)'
                        if logger: logger.error(err_msg)
                        raise CovModelError(err_msg)

                cov_func = cov_model.func() # update covariance function
                if recompute_cov0:
                    if omni_dir:
                        # covariance model in 1D is used
                        cov0 = cov_func(0.)[0] # covariance function at origin (lag=0)
                    else:
                        cov0 = cov_func(np.zeros(d))[0] # covariance function at origin (lag=0)

                if recompute_dmax_ax:
                    if omni_dir:
                        dmax_ax = np.array([cov_model.r()])
                    elif d == 2:
                        dmax_ax = cov_model.r12()
                    elif d == 3:
                        dmax_ax = cov_model.r123()

                    dmax_ax = searchRadiusRelative * dmax_ax
                    dmax_ax_inv2 = 1.0 / (dmax_ax * dmax_ax)

            x0 = x_ineq[jind]
            h = x0 - x_all[ind_sel]
            if rot:
                h = h.dot(rot_mat[jind])
            d2 = np.sum(dmax_ax_inv2 * h**2, axis=1)
            ind = np.where(d2 < 1.0)[0]
            if len(ind) > nneighborMax:
                ind_s = np.argsort(d2[ind])
                ind = ind[ind_s[:nneighborMax]]
            h = h[ind] # lag between x0 and x_all[ind] (used below for right hand side of the kriging system)
            ind = ind_sel.nonzero()[0][ind] # indices in x_all
            nn = len(ind)

            if nn == 0:
                # Mean and std (by kriging) at x_ineq[jind]
                if mean_x is not None:
                    mu = mean_all[n+jind]
                else:
                    mu = mu0

                std = np.sqrt(cov0)
                if var_x is not None:
                    std = varUpdate_all[n+jind]*std

            else:
                xneigh = x_all[ind]
                vneigh = v_all[ind]
                v_err_var_neigh = v_err_var_all[ind]

                # Set right hand side of the kriging system (b)
                if omni_dir:
                    # compute norm of lag
                    h = np.sqrt(np.sum(h**2, axis=1))
                b[:nn] = cov_func(h)

                if ordinary_kriging:
                    nmat = nn+1
                else:
                    nmat = nn

                # Set kriging matrix (mat) of order nmat
                for i in range(nn-1):
                    # lag between xneigh[i] and xneigh[j], j=i+1, ..., nn-1
                    h = xneigh[(i+1):] - xneigh[i]
                    if omni_dir:
                        # compute norm of lag
                        h = np.sqrt(np.sum(h**2, axis=1))
                    elif rot:
                        h = h.dot(rot_mat[jind])
                    cov_h = cov_func(h)
                    mat[i, (i+1):nn] = cov_h
                    mat[(i+1):nn, i] = cov_h
                    mat[i, i] = cov0 + v_err_var_neigh[i]

                mat[nn-1,nn-1] = cov0 + v_err_var_neigh[nn-1]

                if ordinary_kriging:
                    mat[:, nn] = 1.0
                    mat[nn, :] = 1.0
                    mat[nn,nn] = 0.0
                    b[nn] = 1.0

                # Solve the kriging system
                w = np.linalg.solve(mat[:nmat,:nmat], b[:nmat])

                # Mean and std (by kriging) at x0
                if mean_x is not None:
                    # simple kriging
                    std = np.sqrt(max(0, cov0 - np.dot(w, b[:nmat])))
                    if var_x is not None:
                        mu = mean_all[n+jind] + varUpdate_all[n+jind]*(1.0/varUpdate_all[ind]*(vneigh-mean_all[ind])).dot(w)
                        std = varUpdate_all[n+jind]*std
                    else:
                        mu = mean_all[n+jind] + (vneigh-mean_all[ind]).dot(w)
                else:
                    # ordinary kriging
                    std = np.sqrt(max(0, cov0 - np.dot(w, b[:nmat])))
                    mu = vneigh.dot(w[:nn])

                if np.isnan(mu) or np.isinf(mu):
                    if mean_x is not None:
                        mu = mean_all[n+jind]
                    else:
                        mu = mu0

                if np.isnan(std) or np.isinf(std):
                    std = np.sqrt(cov0)
                    if var_x is not None:
                        std = varUpdate_all[n+jind]*std
                
                if np.isclose(std, 0):
                    std = std_min

            # Draw value in Z ~ N(mu, std^2) | v_ineq_min[jind] <= Z <= v_ineq_max[jind]
            # tmin = scipy.stats.norm.cdf((v_ineq_min[jind] - mu)/std)
            # tmax = scipy.stats.norm.cdf((v_ineq_max[jind] - mu)/std)
            tmin = min(1.0 - eps, max(eps, scipy.stats.norm.cdf((v_ineq_min[jind] - mu)/std)))
            tmax = min(1.0 - eps, max(eps, scipy.stats.norm.cdf((v_ineq_max[jind] - mu)/std)))
            t = tmin + np.random.random() * (tmax - tmin)
            v_all[n+jind] = mu + std * scipy.stats.norm.ppf(t)

            ind_sel[n+jind] = True

        # Next paths (Gibbs sampler)
        # --------------------------
        for nGibbs in range(1, nGibbsSamplerPath):
            # set path
            ind_ineq = np.random.permutation(n_ineq)

            for j, jind in enumerate(ind_ineq):
                # Simulation at x0 = x_ineq[jind] = x_ineq[ind_ineq[j]]
                if verbose > 0:
                    progress = int((j+n_ineq*(k*nGibbsSamplerPath+nGibbs))/(nreal*n_ineq*nGibbsSamplerPath)*100.0)
                    if progress > progress_old:
                        if logger:
                            logger.info(f'{fname}: {progress:3d}% ({k:3d} realizations done of {nreal})')
                        else:
                            print(f'{fname}: {progress:3d}% ({k:3d} realizations done of {nreal})')
                        progress_old = progress

                ind = ind_list[jind]
                nn = len(ind)

                if nn == 0:
                    # Mean and std (by kriging) at x_ineq[jind]
                    if mean_x is not None:
                        mu = mean_all[n+jind]
                    else:
                        mu = mu0

                    # std = np.sqrt(cov0)
                    # if var_x is not None:
                    #     std = varUpdate_all[n+jind]*std

                else:
                    vneigh = v_all[ind]
                    w = w_list[jind]
                    std = std_list[jind]

                    # Mean (by kriging) at x_ineq[jind]
                    if mean_x is not None:
                        # simple kriging
                        if var_x is not None:
                            mu = mean_all[n+jind] + varUpdate_all[n+jind]*(1.0/varUpdate_all[ind]*(vneigh-mean_all[ind])).dot(w)
                        else:
                            mu = mean_all[n+jind] + (vneigh-mean_all[ind]).dot(w)
                    else:
                        # ordinary kriging
                        mu = vneigh.dot(w[:nn])

                    if np.isnan(mu) or np.isinf(mu):
                        if mean_x is not None:
                            mu = mean_all[n+jind]
                        else:
                            mu = mu0

                # Draw value in Z ~ N(mu, std^2) | v_ineq_min[jind] <= Z <= v_ineq_max[jind]
                # tmin = scipy.stats.norm.cdf((v_ineq_min[jind] - mu)/std)
                # tmax = scipy.stats.norm.cdf((v_ineq_max[jind] - mu)/std)
                tmin = min(1.0 - eps, max(eps, scipy.stats.norm.cdf((v_ineq_min[jind] - mu)/std)))
                tmax = min(1.0 - eps, max(eps, scipy.stats.norm.cdf((v_ineq_max[jind] - mu)/std)))
                t = tmin + np.random.random() * (tmax - tmin)
                v_all[n+jind] = mu + std * scipy.stats.norm.ppf(t)

        # Store k-th realization
        v_ineq[k, :] = v_all[n:]
        # for j in range(n_ineq):
        #     v_ineq[k, j] = v_all[n+j]

    if verbose > 0:
        if logger:
            logger.info(f'{fname}: {100:3d}% ({nreal:3d} realizations done of {nreal})')
        else:
            print(f'{fname}: {100:3d}% ({nreal:3d} realizations done of {nreal})')

    return v_ineq
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def sgs_at_inequality_data_points_mp(
        x, v, x_ineq, cov_model,
        v_err_std=0.0,
        v_ineq_min=None, v_ineq_max=None,
        method='ordinary_kriging',
        mean_x=None,
        mean_x_ineq=None,
        var_x=None,
        var_x_ineq=None,
        alpha_x_ineq=None,
        beta_x_ineq=None,
        gamma_x_ineq=None,
        cov_model_non_stationarity_x_ineq_list=None,
        searchRadius=None,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        nGibbsSamplerPath=50,
        nreal=1,
        seed=None,
        verbose=0,
        nproc=-1,
        logger=None):
    """
    Computes the same as the function :func:`sgs_at_inequality_data_points`, using multiprocessing.

    All the parameters except `nproc` are the same as those of the function
    :func:`sgs_at_inequality_data_points`.

    This function launches parallel processes [parallel calls of the
    function :func:`sgs_at_inequality_data_points`]; the set of realizations 
    (specified by `nreal`) is distributed in a balanced way over the processes.

    The number of processes used (in parallel) is determined by the parameter `nproc` 
    (int, default: -1); a negative number (or zero), -n <= 0, can be specified 
    to use the total number of cpu(s) of the system except n; `nproc` is finally
    at maximum equal to `nreal` but at least 1 by applying:
        
    - if `nproc >= 1`, then `nproc = max(min(nproc, nreal), 1)` is used
    - if `nproc = -n <= 0`, then `nproc = max(min(nmax-n, nreal), 1)` is used, \
    where nmax is the total number of cpu(s) of the system (retrieved by \
    `multiprocessing.cpu_count()`)

    Note: if `nproc=None`, `nproc=-1` is used.

    Note: specifying a `seed` guarantees reproducible results whatever the number
    of processes used.

    See function :func:`sgs_at_inequality_data_points` for details.
    """
    fname = 'sgs_at_inequality_data_points_mp'

    # Set number of process(es): nproc
    if nproc is None:
        nproc = -1
    
    if nproc <= 0:
        nproc = max(min(multiprocessing.cpu_count() + nproc, nreal), 1)
    else:
        nproc_tmp = nproc
        nproc = max(min(int(nproc), nreal), 1)
        if verbose > 1 and nproc != nproc_tmp:
            if logger:
                logger.info(f'{fname}: number of processes has been changed (now: nproc={nproc})')
            else:
                print(f'{fname}: number of processes has been changed (now: nproc={nproc})')
    
    # Set index for distributing realizations
    q, r = np.divmod(nreal, nproc)
    ids_proc = [i*q + min(i, r) for i in range(nproc+1)]

    if verbose > 0:
        if logger:
            logger.info(f'{fname}: running `sgs_at_inequality_data_points` on {nproc} processes...')
        else:
            print(f'{fname}: running `sgs_at_inequality_data_points` on {nproc} processes...')

    # Set seed (base)
    if seed is None:
        seed = np.random.randint(1, 1000000)
    seed = int(seed)

    # Set pool of nproc workers
    pool = multiprocessing.Pool(nproc)
    out_pool = []
    for i in range(nproc):
        # Set i-th process
        kwargs = dict(
                    v_err_std=v_err_std,
                    v_ineq_min=v_ineq_min, 
                    v_ineq_max=v_ineq_max,
                    method=method,
                    mean_x=mean_x, 
                    mean_x_ineq=mean_x_ineq, 
                    var_x=var_x, 
                    var_x_ineq=var_x_ineq,
                    alpha_x_ineq=alpha_x_ineq, 
                    beta_x_ineq=beta_x_ineq, 
                    gamma_x_ineq=gamma_x_ineq,
                    cov_model_non_stationarity_x_ineq_list=cov_model_non_stationarity_x_ineq_list,
                    searchRadius=searchRadius, 
                    searchRadiusRelative=searchRadiusRelative, 
                    nneighborMax=nneighborMax,
                    nGibbsSamplerPath=nGibbsSamplerPath,
                    nreal=ids_proc[i+1]-ids_proc[i], 
                    seed=seed+ids_proc[i],
                    pid=i,
                    verbose=verbose*(i==0),
                    logger=logger)
        out_pool.append(pool.apply_async(sgs_at_inequality_data_points, args=(x, v, x_ineq, cov_model), kwds=kwargs))

    # Properly end working process
    pool.close() # Prevents any more tasks from being submitted to the pool,
    pool.join()  # then, wait for the worker processes to exit.

    # Get result from each process
    out = [w.get() for w in out_pool]
    if np.any([x is None for x in out]):
        err_msg = f'{fname}: an error occured on a process (worker)'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    v_ineq = np.vstack(out)

    return v_ineq
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def sgs_at_inequality_data_points_slow(
        x, v, x_ineq, cov_model,
        v_err_std=0.0,
        v_ineq_min=None, v_ineq_max=None,
        method='ordinary_kriging',
        mean_x=None,
        mean_x_ineq=None,
        var_x=None,
        var_x_ineq=None,
        alpha_x_ineq=None,
        beta_x_ineq=None,
        gamma_x_ineq=None,
        cov_model_non_stationarity_x_ineq_list=None,
        searchRadius=None,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        nGibbsSamplerPath=50,
        nreal=1,
        seed=None,
        pid=None,
        verbose=0,
        logger=None):
    """
    Performs Sequential Gaussian Simulation (SGS) at inequality data point(s).

    This function does SGS at locations `x_ineq`, given data points locations
    `x` with values `v`.

    Parameters
    ----------
    x : 2D array of floats of shape (n, d)
        data points locations, with n the number of data points and d the space
        dimension (1, 2, or 3), each row of `x` is the coordinatates of one data
        point; note: for data in 1D (`d=1`), 1D array of shape `(n,)` is accepted
        for `n` data points

    v : 1D array of floats of shape (n,)
        data points values, with n the number of data points, `v[i]` is the data
        value at location `x[i]`

    x_ineq : 2D array of floats of shape (n_ineq, d)
        points locations where the interpolation has to be done, with n_ineq the
        number of points and d the space dimension (1, 2, or 3, same as for `x`),
        each row of `x_ineq` is the coordinatates of one point;
        note: for data in 1D (`d=1`), 1D array of shape `(n_ineq,)` is accepted 
        for `n_ineq` points

    cov_model : :class:`CovModel1D` or :class:`CovModel2D` or :class:`CovModel3D`
        covariance model in 1D, 2D, or 3D, in same dimension as dimension of
        points (d), i.e.:

        - :class:`CovModel1D` for data in 1D (d=1)
        - :class:`CovModel2D` for data in 2D (d=2)
        - :class:`CovModel3D` for data in 3D (d=3)

        or

        - :class:`CovModel1D` interpreted as an omni-directional covariance model \
        whatever dimension of points (d);

        note: the covariance model must be stationary, however, non stationarity is
        handled: 

        - local rotation by specifying `alpha_x_ineq` (in 2D or 3D), `beta_x_ineq` (in 3D), `gamma_x_ineq` (in 3D)
        - other non-stationarities by specifying `cov_model_non_stationarity_x_ineq_list` (see below)

    v_err_std : 1D array of floats of shape (n,), or float, default: 0.0
        standard deviation of error at data points, with n the number of data points; 
        if `v_err_std` is a float, the same value is used for all data points; 
        this means that at location x[i], the data value is considered as in a Gaussian 
        distribution of mean `v[i]` and standard deviation `v_err_std[i]`

    v_ineq_min : 1D array of floats of shape (n_ineq,), or float, optional
        minimal value (lower bound) for inequality data points, with n_ineq the 
        number of inequality data points (`v_ineq_min[i]` is the value for the 
        location `x_ineq[i]`); if `v_ineq_min` is a float, the same value 
        is used for all inequality data points; if `v_ineq_min=None` (default),
        no minimal value is considered for any inequality data point;
        note: `v_ineq_min[i]` set to `np.nan` or `-np.inf` means that there is 
        no minimal value for point `x_ineq[i]`
        
    v_ineq_max : 1D array of floats of shape (n_ineq,), or float, optional
        maximal value (upper bound) for inequality data points, with n_ineq the 
        number of inequality data points (`v_ineq_max[i]` is the value for the 
        location `x_ineq[i]`); if `v_ineq_max` is a float, the same value 
        is used for all inequality data points; if `v_ineq_max=None` (default),
        no maximal value is considered for any inequality data point;
        note: `v_ineq_max[i]` set to `np.nan` or `np.inf` means that there is 
        no maximal value for point `x_ineq[i]`

    method : str {'simple_kriging', 'ordinary_kriging'}, default: 'ordinary_kriging'
        type of kriging;
        note: if `method='ordinary_kriging'`, the parameters
        `mean_x`, `mean_x_ineq`, `var_x`, `var_x_ineq` are not used

    mean_x : 1D array-like of floats, or float, optional
        kriging mean value at data points `x`

        - if `mean_x` is a float, the same value is considered for any point
        - if `mean_x=None` (default): the mean of data values, i.e. mean of `v`, \
        is considered for any point

        note: if `method=ordinary_kriging`, parameter `mean_x` is ignored

    mean_x_ineq : 1D array-like of floats, or float, optional
        kriging mean value at points `x_ineq`

        - if `mean_x_ineq` is a float, the same value is considered for any point
        - if `mean_x_ineq=None` (default): the value `mean_x` (assumed to be a single \
        float) is considered for any point

        note: if `method=ordinary_kriging`, parameter `mean_x_ineq` is ignored

    var_x : 1D array-like of floats, or float, optional
        kriging variance value at data points `x`

        - if `var_x` is a float, the same value is considered for any point
        - if `var_x=None` (default): not used  (use of covariance model only)

        note: if `method=ordinary_kriging`, parameter `var_x` is ignored

    var_x_ineq : 1D array-like of floats, or float, optional
        kriging variance value at points `x_ineq`

        - if `var_x_ineq` is a float, the same value is considered for any point
        - if `var_x_ineq=None` (default): not used  (use of covariance model only)

        note: if `method=ordinary_kriging`, parameter `var_x_ineq` is ignored

    alpha_x_ineq : 1D array-like of floats, or float, optional
        azimuth angle in degrees at points `x_ineq`

        - if `alpha_x_ineq` is a float, the same value is considered for any point
        - if `alpha_x_ineq=None` (default): `alpha_x_ineq=0.0` is used for any point

        note: `alpha_x_ineq` is ignored if the covariance model is in 1D

    beta_x_ineq : 1D array-like of floats, or float, optional
        dip angle in degrees at points `x_ineq`

        - if `beta_x_ineq` is a float, the same value is considered for any point
        - if `beta_x_ineq=None` (default): `beta_x_ineq=0.0` is used for any point

        note: `beta_x_ineq` is ignored if the covariance model is in 1D or 2D

    gamma_x_ineq : 1D array-like of floats, or float, optional
        dip angle in degrees at points `x_ineq`

        - if `gamma_x_ineq` is a float, the same value is considered for any point
        - if `gamma_x_ineq=None` (default): `gamma_x_ineq=0.0` is used for any point

        note: `gamma_x_ineq` is ignored if the covariance model is in 1D or 2D

    cov_model_non_stationarity_x_ineq_list : list, optional
        list to set non-stationarities in covariance model; each entry must be
        a tuple (or list) `cm_ns` of length 2 or 3 with:

        - `cm_ns[0]`: str: the name of the method of `cov_model` to be applied
        - `cm_ns[1]`: 1D array-like of floats, or float: \
        used to set the main parameter passed to the method:
            - if array-like: its size must be equal to `n_ineq`, \
            (the array is reshaped if needed), values at points `x_ineq`
            - if a float: same value at all points `x_ineq`
        - `cm_ns[2]`: dict, optional: keyworkds arguments to be passed to the method

        Examples (with the parameter `arg` is set from `val`)
        
        - `('multiply_w', val)` will apply `cov_model.multiply_w(arg)`; 
            this multipies the weight contribution of every elementary contribution of the 
            covariance model
        - `('multiply_w', val, {'elem_ind':0})` will apply `cov_model.multiply_w(arg, elem_ind=0)`;
            this multipies the weight contribution of the elementary contribution of index 0 of the 
            covariance model
        - `('multiply_r', val)` will apply `cov_model.multiply_r(arg)`;
            this multipies the range in all direction of every elementary contribution of the
            covariance model
        - `('multiply_r', val, {'r_ind':0})` will apply `cov_model.multiply_r(arg, r_ind=0)`;
            this multipies the range in the first main direction (index 0) of every elementary 
            contribution of the covariance model
        - `('multiply_r', val, {'r_ind':0, 'elem_ind':0})` will apply `cov_model.multiply_r(arg, r_ind=0, elem_ind=0)`;
            this multipies the range in the first main direction (index 0) of the elementary 
            contribution of index 0 of the covariance model

    searchRadius : float, optional
        if specified, i.e. not `None`: radius of the search neighborhood (ellipsoid
        with same radii along each axis), i.e. the data points at distance to the 
        estimated point greater than `searchRadius` are not taken into account 
        in the kriging system; if `searchRadius` is not `None`, then 
        `searchRadiusRelative` is not used;
        by default (`searchRadius=None`): `searchRadiusRelative` is used to 
        define the search ellipsoid;

    searchRadiusRelative : float, default: 1.2
        used only if `searchRadius` is `None`;
        indicates how the search ellipsoid is limited (should be positive): let
        r_i be the ranges of the covariance model along its main axes, when
        estimating/simulating a cell x, a cell y is taken into account iff it is
        within the ellipsoid centered at x of half axes equal to
        `searchRadiusRelative` * r_i; 
        (note that the distances to the central node are computed in the axes 
        sytem supporting the covariance model and accounting for anisotropy given 
        by the ranges)

    nneighborMax : int, default: 12
        maximal number of neighbors (data points) taken into account in the
        kriging system; the data points the closest to the estimated points are
        taken into account

    nGibbsSamplerPath : int, default: 50
        number of Gibbs sampler paths for simulating values at inequality data points

    nreal : int, default: 1
        number of realization(s)

    seed : int, optional
        seed for initializing random number generator

    pid : int, optional
        process id of the caller (used with multiprocessing)

    verbose : int, default: 0
        verbose mode, higher implies more printing (info)

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    v_ineq : 2D array of shape (nreal, n_ineq)
        simulated values at points `x_ineq`
        - v_ineq[i, j] value of the i-th realization at point `x_ineq[j]`
    """
    fname = 'sgs_at_inequality_data_points_slow'
    if pid is not None:
        fname = f'{fname} [pid={pid}]'

    # Prevent calculation if covariance model is not stationary
    if not cov_model.is_stationary():
        err_msg = f'{fname}: `cov_model` is not stationary: {fname} cannot be applied (use the other paramters for non-stationary covariance)'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # Get dimension from x (d) and number of data points (n) and size of v
    if x is None:
        n = 0
        d = 0
        if v is not None:
            err_msg = f'{fname}: `x` is None but `v` is not None'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
    
    else:
        x = np.asarray(x)
        # Get dimension (d) from x
        if x.ndim == 1:
            # x is a 1-dimensional array
            x = x.reshape(-1, 1)
            d = 1
        else:
            # x is a 2-dimensional array
            d = x.shape[1]

        n = x.shape[0]

        # Check size of v
        if v is None:
            err_msg = f'{fname}: `x` is not None but `v` is None'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

        v = np.asarray(v).reshape(-1)
        if v.size != n:
            err_msg = f'{fname}: size of `v` is not valid'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    # Set variance of data error (from standard deviation)
    if v_err_std is None:
        v_err_std = 0.0
    v_err_var = np.asarray(v_err_std, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
    if v_err_var.size == 1:
        v_err_var = v_err_var[0] * np.ones(n)
    elif v_err_var.size != n:
        err_msg = f'{fname}: size of `v_err_std` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    v_err_var = v_err_var * v_err_var

    # Get dimension from x_ineq (d_ineq) and number of unknown points (n_ineq)
    if x_ineq is None:
        n_ineq = 0
        d_ineq = 0
    else:
        x_ineq = np.asarray(x_ineq)
        # Get dimension (d_ineq) from x_ineq
        if x_ineq.ndim == 1:
            # x_ineq is a 1-dimensional array
            x_ineq = x_ineq.reshape(-1, 1)
            d_ineq = 1
        else:
            # x_ineq is a 2-dimensional array
            d_ineq = x_ineq.shape[1]

        n_ineq = x_ineq.shape[0]

    if n_ineq == 0:
        v_ineq = np.empty((nreal, 0), dtype='float')
        return v_ineq

    # Here: n_ineq > 0

    if n > 0:
        # Check dimension of x and x_ineq
        if d != d_ineq:
            err_msg = f'{fname}: `x` and `x_ineq` do not have the same dimension'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    else: # n == 0
        d = d_ineq # set d to d_ineq if no data point
        x = np.empty((0, d), dtype='float') # set x to an empty array with shape (0, d)
        v = np.array([], dtype='float')

    # Check that all data points (locations) are distinct
    for i in range(1, n):
        if np.any(np.isclose(np.sum((x[:i]-x[i])**2, axis=1), 0.0)):
            err_msg = f'{fname}: `x` contains duplicated entries'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    # Check that all inequality data points (locations) are distinct
    for i in range(1, n_ineq):
        if np.any(np.isclose(np.sum((x_ineq[:i]-x_ineq[i])**2, axis=1), 0.0)):
            err_msg = f'{fname}: `x_ineq` contains duplicated entries'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    # Check that all data points and inequality data points (locations) are distinct
    for i in range(1, n):
        if np.any(np.isclose(np.sum((x_ineq-x[i])**2, axis=1), 0.0)):
            err_msg = f'{fname}: `x` and `x_ineq` contains same entries'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    # Check (and set) v_ineq_min
    if v_ineq_min is None:
        v_ineq_min = np.full((n_ineq, ), -np.inf)
    else:        
        v_ineq_min = np.asarray(v_ineq_min).reshape(-1)
        if v_ineq_min.size == 1:
            v_ineq_min = v_ineq_min * np.ones(n_ineq)
        elif v_ineq_min.size != n_ineq:
            err_msg = f'{fname}: size of `v_ineq_min` is not valid'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

        v_ineq_min[np.isnan(v_ineq_min)] = -np.inf
    
    # Check (and set) v_ineq_max
    if v_ineq_max is None:
        v_ineq_max = np.full((n_ineq, ), np.inf)
    else:        
        v_ineq_max = np.asarray(v_ineq_max).reshape(-1)
        if v_ineq_max.size == 1:
            v_ineq_max = v_ineq_max * np.ones(n_ineq)
        elif v_ineq_max.size != n_ineq:
            err_msg = f'{fname}: size of `v_ineq_max` is not valid'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

        v_ineq_max[np.isnan(v_ineq_max)] = np.inf

    # Check consistency of v_ineq_min and v_ineq_max
    if np.any(v_ineq_max <= v_ineq_min):
        err_msg = f'{fname}: `v_ineq_min` and `v_ineq_max` are not consistent'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # Check dimension of cov_model and set if used as omni-directional model
    if isinstance(cov_model, CovModel1D):
        omni_dir = True
    else:
        if cov_model.__class__.__name__ != f'CovModel{d}D':
            err_msg = f'{fname}: `cov_model` dimension is incompatible with dimension of points'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

        omni_dir = False

    # Covariance function
    cov_func = cov_model.func() # covariance function
    if omni_dir:
        # covariance model in 1D is used
        cov0 = cov_func(0.)[0] # covariance function at origin (lag=0)
    else:
        cov0 = cov_func(np.zeros(d))[0] # covariance function at origin (lag=0)

    # Method and mean, var
    if method == 'simple_kriging':
        ordinary_kriging = False
        if n == 0:
            mean_x = np.array([], dtype='float')
        else:
            if mean_x is None:
                mean_x = np.mean(v) * np.ones(n)
            else:
                mean_x = np.asarray(mean_x, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                if mean_x.size == 1:
                    mean_x = mean_x * np.ones(n)
                elif mean_x.size != n:
                    err_msg = f'{fname}: size of `mean_x` is not valid'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)

        if mean_x_ineq is None:
            if n == 0:
                mean_x_ineq = np.zeros(n_ineq)
            else:
                mean_x_ineq = np.mean(v) * np.ones(n_ineq)
        else:
            mean_x_ineq = np.asarray(mean_x_ineq, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if mean_x_ineq.size == 1:
                mean_x_ineq = mean_x_ineq * np.ones(n_ineq)
            elif mean_x_ineq.size != n_ineq:
                err_msg = f'{fname}: size of `mean_x_ineq` is not valid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

        if n == 0:
            if var_x_ineq is None:
                var_x = None
            else:
                var_x = np.array([], dtype='float')

        if (var_x is None and var_x_ineq is not None) or (var_x is not None and var_x_ineq is None):
            err_msg = f'{fname}: `var_x` and `var_x_ineq` must both be specified'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

        if var_x is not None:
            var_x = np.asarray(var_x, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if var_x.size == 1:
                var_x = var_x * np.ones(n)
            elif var_x.size != n:
                err_msg = f'{fname}: size of `var_x` is not valid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

            varUpdate_x = np.sqrt(var_x/cov0)

        if var_x_ineq is not None:
            var_x_ineq = np.asarray(var_x_ineq, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if var_x_ineq.size == 1:
                var_x_ineq = var_x_ineq * np.ones(n_ineq)
            elif var_x_ineq.size != n_ineq:
                err_msg = f'{fname}: size of `var_x_ineq` is not valid'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)

            varUpdate_x_ineq = np.sqrt(var_x_ineq/cov0)

    elif method == 'ordinary_kriging':
        if verbose > 0:
            if mean_x is not None:
                if logger:
                    logger.warning(f"{fname}: `mean_x` is ignored with `method='ordinary_kriging'`")
                else:
                    print(f"{fname}: WARNING: `mean_x` is ignored with `method='ordinary_kriging'`")
            if mean_x_ineq is not None:
                if logger:
                    logger.warning(f"{fname}: `mean_x_ineq` is ignored with `method='ordinary_kriging'`")
                else:
                    print(f"{fname}: WARNING: `mean_x_ineq` is ignored with `method='ordinary_kriging'`")
            if var_x is not None:
                if logger:
                    logger.warning(f"{fname}: `var_x` is ignored with `method='ordinary_kriging'`")
                else:
                    print(f"{fname}: WARNING: `var_x` is ignored with `method='ordinary_kriging'`")
            if var_x_ineq is not None:
                if logger:
                    logger.warning(f"{fname}: `var_x_ineq` is ignored with `method='ordinary_kriging'`")
                else:
                    print(f"{fname}: WARNING: `var_x_ineq` is ignored with `method='ordinary_kriging'`")

        ordinary_kriging = True
        mean_x, mean_x_ineq, var_x, var_x_ineq = None, None, None, None

    else:
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    # WORK ON A COPY OF COVARIANCE MODEL (IN CASE IT IS ADAPTED)!
    cov_model = copyCovModel(cov_model) 
    cov_model_has_changed = False

    # Rotation given by alpha_x_ineq, beta_x_ineq, gamma_x_ineq
    if alpha_x_ineq is not None:
        if omni_dir:
            err_msg = f'{fname}: `alpha_x_ineq` cannot be used with 1D covariance model'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    if beta_x_ineq is not None:
        if omni_dir or d < 3:
            err_msg = f'{fname}: `beta_x_ineq` cannot be used with 1D or 2D covariance model'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
    
    if gamma_x_ineq is not None:
        if omni_dir or d < 3:
            err_msg = f'{fname}: `gamma_x_ineq` cannot be used with 1D or 2D covariance model'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)

    if omni_dir:
        rot = False
    else:
        if d == 2:
            # 2D - check only alpha
            if alpha_x_ineq is None:
                rot = False
            else:
                alpha_x_ineq = np.asarray(alpha_x_ineq, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                if alpha_x_ineq.size == 1:
                    if alpha_x_ineq[0] == 0.0:
                        rot = False
                    else:
                        rot_mat = rotationMatrix2D(alpha_x_ineq[0]) # rot_mat : rotation matrix for any x_ineq[i]
                        rot = True
                        rot_mat_unique = True
                elif alpha_x_ineq.size == n_ineq:
                    if np.allclose(alpha_x_ineq, alpha_x_ineq[0]):
                        if np.isclose(alpha_x_ineq[0], 0.0):
                            rot = False
                        else:
                            rot_mat = rotationMatrix2D(alpha_x_ineq[0]) # rot_mat : rotation matrix for any x_ineq[i]
                            rot = True
                            rot_mat_unique = True
                    else:
                        rot_mat = rotationMatrix2D(alpha_x_ineq).transpose(2, 0, 1) # rot_mat[i] : rotation matrix for xineq[i]
                        rot = True
                        rot_mat_unique = False
                else:
                    err_msg = f'{fname}: size of `alpha_x_ineq` is not valid'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)

        else: # d == 3
            # 3D
            if alpha_x_ineq is None and beta_x_ineq is None and gamma_x_ineq is None:
                rot = False
            else:
                if alpha_x_ineq is not None:
                    alpha_x_ineq = np.asarray(alpha_x_ineq, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                    if alpha_x_ineq.size == 1:
                        alpha_x_ineq = alpha_x_ineq * np.ones(n_ineq)
                    elif alpha_x_ineq.size != n_ineq:
                        err_msg = f'{fname}: size of `alpha_x_ineq` is not valid'
                        if logger: logger.error(err_msg)
                        raise CovModelError(err_msg)
                else:
                    alpha_x_ineq = np.zeros(n_ineq)

                if beta_x_ineq is not None:
                    beta_x_ineq = np.asarray(beta_x_ineq, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                    if beta_x_ineq.size == 1:
                        beta_x_ineq = beta_x_ineq * np.ones(n_ineq)
                    elif beta_x_ineq.size != n_ineq:
                        err_msg = f'{fname}: size of `beta_x_ineq` is not valid'
                        if logger: logger.error(err_msg)
                        raise CovModelError(err_msg)
                else:
                    beta_x_ineq = np.zeros(n_ineq)

                if gamma_x_ineq is not None:
                    gamma_x_ineq = np.asarray(gamma_x_ineq, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
                    if gamma_x_ineq.size == 1:
                        gamma_x_ineq = gamma_x_ineq * np.ones(n_ineq)
                    elif gamma_x_ineq.size != n_ineq:
                        err_msg = f'{fname}: size of `gamma_x_ineq` is not valid'
                        if logger: logger.error(err_msg)
                        raise CovModelError(err_msg)
                else:
                    gamma_x_ineq = np.zeros(n_ineq)
                
                if np.allclose(np.vstack((alpha_x_ineq, beta_x_ineq, gamma_x_ineq)).T, np.array([alpha_x_ineq[0], beta_x_ineq[0], gamma_x_ineq[0]])):
                    if np.isclose(alpha_x_ineq[0], 0.0) and np.isclose(beta_x_ineq[0], 0.0) and np.isclose(gamma_x_ineq[0], 0.0):
                        rot = False
                    else:
                        rot_mat = rotationMatrix3D(alpha_x_ineq[0], beta_x_ineq[0], gamma_x_ineq[0]) # rot_mat : rotation matrix for any x_ineq[i]
                        rot = True
                        rot_mat_unique = True
                else:
                    rot_mat = rotationMatrix3D(alpha_x_ineq, beta_x_ineq, gamma_x_ineq).transpose(2, 0, 1) # rot_mat[i] : rotation matrix for x_ineq[i]
                    rot = True
                    rot_mat_unique = False

    if rot:
        if d == 2:
            cov_model.set_alpha(0.0)
            cov_model_has_changed = True
        elif d == 3:
            cov_model.set_alpha(0.0)
            cov_model.set_beta(0.0)
            cov_model.set_gamma(0.0)
            cov_model_has_changed = True
        if rot_mat_unique:
            # apply rotation to data points x and points x_ineq
            x = x.dot(rot_mat)
            x_ineq = x_ineq.dot(rot_mat)
            rot = False # no need rotation further

    # here: rot = True means that local rotation are applied

    # Prepare non-stationarities for integration in covariance model
    adapt_cov_model_ind = []
    recompute_cov0 = False
    recompute_dmax_ax = False
    if cov_model_non_stationarity_x_ineq_list is not None:
        if not isinstance(cov_model_non_stationarity_x_ineq_list, list):
            err_msg = f'{fname}: `cov_model_non_stationarity_x_ineq_list` must be a list if not `None`'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
        for i, cm_ns_x_ineq in enumerate(cov_model_non_stationarity_x_ineq_list):
            if not hasattr(cm_ns_x_ineq, '__len__') or len(cm_ns_x_ineq) not in (2, 3):
                err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_x_ineq_list` not valid: should be a tuple or list of length 2 or 3 containing: method name (str), val[, kwds (dict)]'
                if logger: logger.error(err_msg)
                raise CovModelError(err_msg)
            val = np.asarray(cm_ns_x_ineq[1], dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if val.size != 1:
                if val.size != n_ineq:
                    err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_x_ineq_list`, `(method_name, val[, kwds])`: size of `val` not valid'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)
                if np.allclose(val, val[0]):
                    val = val[:1] # of size 1
                else:
                    adapt_cov_model_ind.append(i)
                    if cm_ns_x_ineq[0] == 'multiply_w':
                        recompute_cov0 = True
                    elif cm_ns_x_ineq[0] == 'multiply_r':
                        recompute_dmax_ax = True
            if val.size == 1:
                method_name = cm_ns_x_ineq[0]
                if len(cm_ns_x_ineq) == 3:
                    kwds = cm_ns_x_ineq[2]
                else:
                    kwds = {}
                try:
                    eval(f'cov_model.{method_name}')(val[0], logger=logger, **kwds)
                except:
                    err_msg = f'{fname}: cannot apply non-stationarity (at points) for covariance model (check parameter `cov_model_non_stationarity_x_ineq_list`)'
                    if logger: logger.error(err_msg)
                    raise CovModelError(err_msg)
                cov_model_has_changed = True

    adapt_cov_model = len(adapt_cov_model_ind) > 0

    if cov_model_has_changed:
        # Update - Covariance function and value at 0
        cov_func = cov_model.func() # covariance function
        if omni_dir:
            # covariance model in 1D is used
            cov0 = cov_func(0.)[0] # covariance function at origin (lag=0)
        else:
            cov0 = cov_func(np.zeros(d))[0] # covariance function at origin (lag=0)
    
    # Limited search neighborhood
    if searchRadius is not None:
        if searchRadius <= 0.0:
            err_msg = f'{fname}: search radius (isotropic neighborhood) not valid (negative)'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
        
        if omni_dir:
            dmax_ax = np.array([searchRadius], dtype='float')
        else:
            dmax_ax = searchRadius * np.ones(d)

    else:
        # use searchRadiusRelative
        if searchRadiusRelative <= 0.0:
            err_msg = f'{fname}: search radius relative (factor) not valid (negative)'
            if logger: logger.error(err_msg)
            raise CovModelError(err_msg)
        
        if d == 1 or omni_dir:
            dmax_ax = np.array([cov_model.r()])
        elif d == 2:
            dmax_ax = cov_model.r12()
        elif d == 3:
            dmax_ax = cov_model.r123()

        dmax_ax = searchRadiusRelative * dmax_ax

    dmax_ax_inv2 = 1.0 / (dmax_ax * dmax_ax)

    # Preparation if covariance model will be adapted
    if adapt_cov_model:
        cov_model_base = copyCovModel(cov_model)

    # Maximum number of neighbors
    if nneighborMax is None or nneighborMax < 0:
        err_msg = f'{fname}: `nneighborMax` is not valid'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    mat = np.ones((nneighborMax+1, nneighborMax+1)) # allocate kriging matrix
    b = np.ones(nneighborMax+1) # allocate second member

     # Min value for std
    std_min = 1.e-10
    eps = 1.e-8

    # Allocate memory for output
    v_ineq = np.zeros((nreal, n_ineq))

    # Set all points together (data points and inequality data points)
    x_all = np.vstack((x, x_ineq))
    v_all = np.hstack((v, np.zeros(n_ineq)))
    v_err_var_all = np.hstack((v_err_var, np.zeros(n_ineq)))
    if mean_x is not None:
        mean_all = np.hstack((mean_x, mean_x_ineq))
    if var_x is not None:
        varUpdate_all = np.hstack((varUpdate_x, varUpdate_x_ineq))

    # Set mu0 : used as mean when mean_x is None and there is no data in the neighborhood during the simulation ...
    if n == 0:
        mu0 = 0.0
    else:
        mu0 = np.mean(v)

    # Array indicating if a point in x_all can be selected for kriging
    ind_sel = np.zeros(n+n_ineq, dtype='bool')

    if seed is None:
        seed = np.random.randint(1, 1000000)
    seed = int(seed)

    if verbose > 0:
        progress_old = 0

    for k in range(nreal):
        # Initialize random number generator
        np.random.seed(seed+k)

        # Initialize ind_sel
        ind_sel[:n] = True  # all data points can be selected
        ind_sel[n:] = False # no simulated points can be selected at the beginning
     
        for nGibbs in range(nGibbsSamplerPath):
            # set path
            ind_ineq = np.random.permutation(n_ineq)

            for j, jind in enumerate(ind_ineq):
                # Simulation at x0 = x_ineq[jind] = x_ineq[ind_ineq[j]]
                ind_sel[n+jind] = False

                if verbose > 0:
                    progress = int((j+n_ineq*(k*nGibbsSamplerPath+nGibbs))/(nreal*n_ineq*nGibbsSamplerPath)*100.0)
                    if progress > progress_old:
                        if logger:
                            logger.info(f'{fname}: {progress:3d}% ({k:3d} realizations done of {nreal})')
                        else:
                            print(f'{fname}: {progress:3d}% ({k:3d} realizations done of {nreal})')
                        progress_old = progress

                if adapt_cov_model:
                    cov_model = copyCovModel(cov_model_base)
                    for i in adapt_cov_model_ind:
                        cm_ns_x_ineq = cov_model_non_stationarity_x_ineq_list[i]
                        method_name = cm_ns_x_ineq[0]
                        val = cm_ns_x_ineq[1]
                        if len(cm_ns_x_ineq) == 3:
                            kwds = cm_ns_x_ineq[2]
                        else:
                            kwds = {}
                        try:
                            eval(f'cov_model.{method_name}')(val[jind], logger=logger, **kwds)
                        except:
                            err_msg = f'{fname}: cannot apply non-stationarity (at one point) for covariance model (check parameter `cov_model_non_stationarity_x_ineq_list`)'
                            if logger: logger.error(err_msg)
                            raise CovModelError(err_msg)

                    cov_func = cov_model.func() # update covariance function
                    if recompute_cov0:
                        if omni_dir:
                            # covariance model in 1D is used
                            cov0 = cov_func(0.)[0] # covariance function at origin (lag=0)
                        else:
                            cov0 = cov_func(np.zeros(d))[0] # covariance function at origin (lag=0)

                    if recompute_dmax_ax:
                        if omni_dir:
                            dmax_ax = np.array([cov_model.r()])
                        elif d == 2:
                            dmax_ax = cov_model.r12()
                        elif d == 3:
                            dmax_ax = cov_model.r123()

                        dmax_ax = searchRadiusRelative * dmax_ax
                        dmax_ax_inv2 = 1.0 / (dmax_ax * dmax_ax)

                x0 = x_ineq[jind]
                h = x0 - x_all[ind_sel]
                if rot:
                    h = h.dot(rot_mat[jind])
                d2 = np.sum(dmax_ax_inv2 * h**2, axis=1)
                ind = np.where(d2 < 1.0)[0]
                if len(ind) > nneighborMax:
                    ind_s = np.argsort(d2[ind])
                    ind = ind[ind_s[:nneighborMax]]
                h = h[ind] # lag between x0 and x_all[ind] (used below for right hand side of the kriging system)
                ind = ind_sel.nonzero()[0][ind] # indices in x_all
                nn = len(ind)

                if nn == 0:
                    # Mean and std (by kriging) at x_ineq[jind]
                    if mean_x is not None:
                        mu = mean_all[n+jind]
                    else:
                        mu = mu0
                    
                    std = np.sqrt(cov0)
                    if var_x is not None:
                        std = varUpdate_all[n+jind]*std

                else:
                    xneigh = x_all[ind]
                    vneigh = v_all[ind]
                    v_err_var_neigh = v_err_var_all[ind]

                    # Set right hand side of the kriging system (b)
                    if omni_dir:
                        # compute norm of lag
                        h = np.sqrt(np.sum(h**2, axis=1))
                    b[:nn] = cov_func(h)

                    if ordinary_kriging:
                        nmat = nn+1
                    else:
                        nmat = nn

                    # Set kriging matrix (mat) of order nmat
                    for i in range(nn-1):
                        # lag between xneigh[i] and xneigh[j], j=i+1, ..., nn-1
                        h = xneigh[(i+1):] - xneigh[i]
                        if omni_dir:
                            # compute norm of lag
                            h = np.sqrt(np.sum(h**2, axis=1))
                        elif rot:
                            h = h.dot(rot_mat[jind])
                        cov_h = cov_func(h)
                        mat[i, (i+1):nn] = cov_h
                        mat[(i+1):nn, i] = cov_h
                        mat[i, i] = cov0 + v_err_var_neigh[i]

                    mat[nn-1,nn-1] = cov0 + v_err_var_neigh[nn-1]

                    if ordinary_kriging:
                        mat[:, nn] = 1.0
                        mat[nn, :] = 1.0
                        mat[nn,nn] = 0.0
                        b[nn] = 1.0

                    # Solve the kriging system
                    w = np.linalg.solve(mat[:nmat,:nmat], b[:nmat])

                    # Mean and std (by kriging) at x_ineq[jind]
                    if mean_x is not None:
                        # simple kriging
                        std = np.sqrt(max(0, cov0 - np.dot(w, b[:nmat])))
                        if var_x is not None:
                            mu = mean_all[n+jind] + varUpdate_all[n+jind]*(1.0/varUpdate_all[ind]*(vneigh-mean_all[ind])).dot(w)
                            std = varUpdate_all[n+jind]*std
                        else:
                            mu = mean_all[n+jind] + (vneigh-mean_all[ind]).dot(w)
                    else:
                        # ordinary kriging
                        std = np.sqrt(max(0, cov0 - np.dot(w, b[:nmat])))
                        mu = vneigh.dot(w[:nn])

                    if np.isnan(mu) or np.isinf(mu):
                        if mean_x is not None:
                            mu = mean_all[n+jind]
                        else:
                            mu = mu0

                    if np.isnan(std) or np.isinf(std):
                        std = np.sqrt(cov0)
                        if var_x is not None:
                            std = varUpdate_all[n+jind]*std
                    
                    if np.isclose(std, 0):
                        std = std_min

                # Draw value in Z ~ N(mu, std^2) | v_ineq_min[jind] <= Z <= v_ineq_max[jind]
                # tmin = scipy.stats.norm.cdf((v_ineq_min[jind] - mu)/std)
                # tmax = scipy.stats.norm.cdf((v_ineq_max[jind] - mu)/std)
                tmin = min(1.0 - eps, max(eps, scipy.stats.norm.cdf((v_ineq_min[jind] - mu)/std)))
                tmax = min(1.0 - eps, max(eps, scipy.stats.norm.cdf((v_ineq_max[jind] - mu)/std)))
                t = tmin + np.random.random() * (tmax - tmin)
                v_all[n+jind] = mu + std * scipy.stats.norm.ppf(t)

                ind_sel[n+jind] = True

        # Store k-th realization
        v_ineq[k, :] = v_all[n:]

    if verbose > 0:
        if logger:
            logger.info(f'{fname}: {100:3d}% ({nreal:3d} realizations done of {nreal})')
        else:
            print(f'{fname}: {100:3d}% ({nreal:3d} realizations done of {nreal})')

    return v_ineq
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def sgs_at_inequality_data_points_slow_mp(
        x, v, x_ineq, cov_model,
        v_err_std=0.0,
        v_ineq_min=None, v_ineq_max=None,
        method='ordinary_kriging',
        mean_x=None,
        mean_x_ineq=None,
        var_x=None,
        var_x_ineq=None,
        alpha_x_ineq=None,
        beta_x_ineq=None,
        gamma_x_ineq=None,
        cov_model_non_stationarity_x_ineq_list=None,
        searchRadius=None,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        nGibbsSamplerPath=50,
        nreal=1,
        seed=None,
        verbose=0,
        nproc=-1,
        logger=None):
    """
    Computes the same as the function :func:`sgs_at_inequality_data_points_slow`, using multiprocessing.

    All the parameters except `nproc` are the same as those of the function
    :func:`sgs_at_inequality_data_points_slow`.

    This function launches parallel processes [parallel calls of the
    function :func:`sgs_at_inequality_data_points_slow`]; the set of realizations 
    (specified by `nreal`) is distributed in a balanced way over the processes.

    The number of processes used (in parallel) is determined by the parameter `nproc` 
    (int, default: -1); a negative number (or zero), -n <= 0, can be specified 
    to use the total number of cpu(s) of the system except n; `nproc` is finally
    at maximum equal to `nreal` but at least 1 by applying:
        
    - if `nproc >= 1`, then `nproc = max(min(nproc, nreal), 1)` is used
    - if `nproc = -n <= 0`, then `nproc = max(min(nmax-n, nreal), 1)` is used, \
    where nmax is the total number of cpu(s) of the system (retrieved by \
    `multiprocessing.cpu_count()`)

    Note: if `nproc=None`, `nproc=-1` is used.

    Note: specifying a `seed` guarantees reproducible results whatever the number
    of processes used.

    See function :func:`sgs_at_inequality_data_points_slow` for details.
    """
    fname = 'sgs_at_inequality_data_points_slow_mp'

    # Set number of process(es): nproc
    if nproc is None:
        nproc = -1
    
    if nproc <= 0:
        nproc = max(min(multiprocessing.cpu_count() + nproc, nreal), 1)
    else:
        nproc_tmp = nproc
        nproc = max(min(int(nproc), nreal), 1)
        if verbose > 1 and nproc != nproc_tmp:
            if logger:
                logger.info(f'{fname}: number of processes has been changed (now: nproc={nproc})')
            else:
                print(f'{fname}: number of processes has been changed (now: nproc={nproc})')
    
    # Set index for distributing realizations
    q, r = np.divmod(nreal, nproc)
    ids_proc = [i*q + min(i, r) for i in range(nproc+1)]

    if verbose > 0:
        if logger:
            logger.info(f'{fname}: running `sgs_at_inequality_data_points_slow` on {nproc} processes...')
        else:
            print(f'{fname}: running `sgs_at_inequality_data_points_slow` on {nproc} processes...')

    # Set seed (base)
    if seed is None:
        seed = np.random.randint(1, 1000000)
    seed = int(seed)

    # Set pool of nproc workers
    pool = multiprocessing.Pool(nproc)
    out_pool = []
    for i in range(nproc):
        # Set i-th process
        kwargs = dict(
                    v_err_std=v_err_std,
                    v_ineq_min=v_ineq_min, 
                    v_ineq_max=v_ineq_max,
                    method=method,
                    mean_x=mean_x, 
                    mean_x_ineq=mean_x_ineq, 
                    var_x=var_x, 
                    var_x_ineq=var_x_ineq,
                    alpha_x_ineq=alpha_x_ineq, 
                    beta_x_ineq=beta_x_ineq, 
                    gamma_x_ineq=gamma_x_ineq,
                    cov_model_non_stationarity_x_ineq_list=cov_model_non_stationarity_x_ineq_list,
                    searchRadius=searchRadius, 
                    searchRadiusRelative=searchRadiusRelative, 
                    nneighborMax=nneighborMax,
                    nGibbsSamplerPath=nGibbsSamplerPath,
                    nreal=ids_proc[i+1]-ids_proc[i], 
                    seed=seed+ids_proc[i],
                    pid=i,
                    verbose=verbose*(i==0),
                    logger=logger)
        out_pool.append(pool.apply_async(sgs_at_inequality_data_points_slow, args=(x, v, x_ineq, cov_model), kwds=kwargs))

    # Properly end working process
    pool.close() # Prevents any more tasks from being submitted to the pool,
    pool.join()  # then, wait for the worker processes to exit.

    # Get result from each process
    out = [w.get() for w in out_pool]
    if np.any([x is None for x in out]):
        err_msg = f'{fname}: an error occured on a process (worker)'
        if logger: logger.error(err_msg)
        raise CovModelError(err_msg)

    v_ineq = np.vstack(out)

    return v_ineq
# ----------------------------------------------------------------------------

# # ============================================================================
# # Simple and ordinary kriging for indicator variable (categorical variables)
# # ============================================================================
# # To be developed : 
# # - SIS at given points ... 
# # - version of simulateIndicator / estimateIndicator in geosclassicinterface...
# # 
# # ----------------------------------------------------------------------------
# def krige_indicator(
#         category_values,
#         x, v, xu, cov_model,
#         method='ordinary_kriging',
#         probability_x=None,
#         probability_xu=None,
#         alpha_xu=None,
#         beta_xu=None,
#         gamma_xu=None,
#         cov_model_non_stationarity_xu_list=None,
#         use_unique_neighborhood=False,
#         searchRadius=None,
#         searchRadiusRelative=1.2,
#         nneighborMax=12,
#         verbose=0):
#     """
#     Interpolates data by kriging at given location(s).

#     This function performs kriging interpolation at locations `xu` of the values
#     `v` measured at locations `x`.

#     Parameters
#     ----------
#     category_values : 1D array-like
#         sequence of category values; let `ncategory` be the number of categories,
#         then:

#         - if `ncategory=1`: the unique category value given must not be equal to \
#         zero; it is used for a binary case with values "unique category value" \
#         and 0, where 0 indicates the absence of the considered medium; the \
#         conditioning data values should be equal to"unique category value" or 0
#         - if `ncategory>=2`: it is used for a multi-category case with given \
#         category values (distinct); the conditioning data values should be in the \
#         `category_values`

#     x : 2D array of floats of shape (n, d)
#         data points locations, with n the number of data points and d the space
#         dimension (1, 2, or 3), each row of `x` is the coordinatates of one data
#         point; note: for data in 1D (`d=1`), 1D array of shape `(n,)` is accepted
#         for `n` data points

#     v : 1D array of floats of shape (n,)
#         data points values, with n the number of data points, `v[i]` is the data
#         value at location `x[i]`

#     xu : 2D array of floats of shape (nu, d)
#         points locations where the interpolation has to be done, with nu the
#         number of points and d the space dimension (1, 2, or 3, same as for `x`),
#         each row of `xu` is the coordinatates of one point;
#         note: for data in 1D (`d=1`), 1D array of shape `(nu,)` is accepted 
#         for `nu` points

#     cov_model : [sequence of] :class:`CovModel<d>D`
#         sequence of same length as `category_values` of covariance model in 1D,
#         or a unique covariance model in 1D (recycled):
#         covariance model for each category;
#         covariance model in 1D or 2D or 3D;      
#         note: the covariance model must be stationary, however, non stationarity is
#         handled: 

#         - local rotation by specifying `alpha` (in 2D or 3D), `beta` (in 3D), `gamma` (in 3D)
#         - other non-stationarities by specifying `cov_model_non_stationarity_list` (see below)

#     method : str {'simple_kriging', 'ordinary_kriging'}, default: 'ordinary_kriging'
#         type of kriging;

#     probability_x : 2D array-like of floats, or 1D-array of floats, optional
#         probability (kriging mean value) of each category at data points `x`;
#         - `probability[k, i]` : value for the k-th category at `x[i]`

#         note: if 1D-array like, recycled for every category;

#     probability_x : 2D array-like of floats, or 1D-array of floats, optional
#         probability (kriging mean value) of each category at data points `xu`;
#         - `probability[k, i]` : value for the k-th category at `xu[i]`

#         note: if 1D-array like, recycled for every category;

#     alpha_xu : [sequence of] 1D array-like of floats, or float, optional
#         sequence or recycled, as `cov_model`;
#         azimuth angle in degrees at points `xu`

#         - if `alpha_xu` is a float, the same value is considered for any point
#         - if `alpha_xu=None` (default): `alpha_xu=0.0` is used for any point

#         note: `alpha_xu` is ignored if the covariance model is in 1D

#     beta_xu : [sequence of] 1D array-like of floats, or float, optional
#         sequence or recycled, as `cov_model`;
#         dip angle in degrees at points `xu`

#         - if `beta_xu` is a float, the same value is considered for any point
#         - if `beta_xu=None` (default): `beta_xu=0.0` is used for any point

#         note: `beta_xu` is ignored if the covariance model is in 1D or 2D

#     gamma_xu [sequence of] : 1D array-like of floats, or float, optional
#         sequence or recycled, as `cov_model`;
#         dip angle in degrees at points `xu`

#         - if `gamma_xu` is a float, the same value is considered for any point
#         - if `gamma_xu=None` (default): `gamma_xu=0.0` is used for any point

#         note: `gamma_xu` is ignored if the covariance model is in 1D or 2D

#     cov_model_non_stationarity_xu_list [sequence of] : list, optional
#         sequence or recycled, as `cov_model`;
#         list to set non-stationarities in covariance model; each entry must be
#         a tuple (or list) `cm_ns` of length 2 or 3 with:

#         - `cm_ns[0]`: str: the name of the method of `cov_model` to be applied
#         - `cm_ns[1]`: 1D array-like of floats, or float: \
#         used to set the main parameter passed to the method:
#             - if array-like: its size must be equal to `nu`, \
#             (the array is reshaped if needed), values at points `xu`
#             - if a float: same value at all points `xu`
#         - `cm_ns[2]`: dict, optional: keyworkds arguments to be passed to the method

#         Examples (with the parameter `arg` is set from `val`)
        
#         - `('multiply_w', val)` will apply `cov_model.multiply_w(arg)`; 
#             this multipies the weight contribution of every elementary contribution of the 
#             covariance model
#         - `('multiply_w', val, {'elem_ind':0})` will apply `cov_model.multiply_w(arg, elem_ind=0)`;
#             this multipies the weight contribution of the elementary contribution of index 0 of the 
#             covariance model
#         - `('multiply_r', val)` will apply `cov_model.multiply_r(arg)`;
#             this multipies the range in all direction of every elementary contribution of the
#             covariance model
#         - `('multiply_r', val, {'r_ind':0})` will apply `cov_model.multiply_r(arg, r_ind=0)`;
#             this multipies the range in the first main direction (index 0) of every elementary 
#             contribution of the covariance model
#         - `('multiply_r', val, {'r_ind':0, 'elem_ind':0})` will apply `cov_model.multiply_r(arg, r_ind=0, elem_ind=0)`;
#             this multipies the range in the first main direction (index 0) of the elementary 
#             contribution of index 0 of the covariance model

#     use_unique_neighborhood : [sequence of] bool, default: False
#         sequence of same length as `category_values` or recycled:
#         indicates if a unique neighborhood is used:

#         - if True: all data points are taken into account, and the kriging matrix \
#         is computed once; the parameters `searchRadius`, `searchRadiusRelative`, \
#         `nneighborMax` are not used, \
#         and  `alpha_xu`, `beta_xu`, `gamma_xu` must be `None` or constant \
#         and any non-stationarity set in `cov_model_non_stationarity_xu_list` must be constant \
#         (i.e. `cm_ns[1]` must be constant for each entry `cm_ns` in `cov_model_non_stationarity_xu_list`)
#         - if False: only data points within a search neighborhood (ellipsoid) are \
#         taken into account according to `searchRadius`, `searchRadiusRelative`, `nneighborMax`

#     searchRadius : [sequence of] float, optional
#         sequence of same length as `category_values` or recycled:
#         if specified, i.e. not `None`: radius of the search neighborhood (ellipsoid
#         with same radii along each axis), i.e. the data points at distance to the 
#         estimated point greater than `searchRadius` are not taken into account 
#         in the kriging system; if `searchRadius` is not `None`, then 
#         `searchRadiusRelative` is not used;
#         by default (`searchRadius=None`): `searchRadiusRelative` is used to 
#         define the search ellipsoid;

#     searchRadiusRelative : [sequence of] float, default: 1.2
#         sequence of same length as `category_values` or recycled:
#         used only if `searchRadius` is `None`;
#         indicates how the search ellipsoid is limited (should be positive): let
#         r_i be the ranges of the covariance model along its main axes, when
#         estimating/simulating a cell x, a cell y is taken into account iff it is
#         within the ellipsoid centered at x of half axes equal to
#         `searchRadiusRelative` * r_i; 
#         (note that the distances to the central node are computed in the axes 
#         sytem supporting the covariance model and accounting for anisotropy given 
#         by the ranges)

#     nneighborMax : [sequence of] int, default: 12
#         sequence of same length as `category_values` or recycled:
#         maximal number of neighbors (data points) taken into account in the
#         kriging system; the data points the closest to the estimated points are
#         taken into account;
#         note: if `nneighborMax=None` or `nneighborMax<0`, then `nneighborMax` is
#         set to the number of data points

#     verbose : int, default: 0
#         verbose mode, higher implies more printing (info)

#     Returns
#     -------
#     vu_indicator : 2D array of shape (ncategory, nu)
#         kriging estimates at points `xu`, for each category
#     """
#     fname = 'krige_indicator'

#     # Check category values and compute number of category (ncategory
#     try:
#         category_values = np.asarray(category_values, dtype='float').reshape(-1)
#     except:
#         err_msg = f'{fname}: `category_values` invalid'
#         if logger: logger.error(err_msg)
#         raise CovModelError(err_msg)

#     ncategory = len(category_values)
#     if ncategory <= 0:
#         err_msg = f'{fname}: `category_values` is empty'
#         if logger: logger.error(err_msg)
#         raise CovModelError(err_msg)

#     # Get dimension from x (d) and number of data points (n) and size of v
#     if x is None:
#         n = 0
#         d = 0
#         if v is not None:
#             err_msg = f'{fname}: `x` is None but `v` is not None'
#             if logger: logger.error(err_msg)
#             raise CovModelError(err_msg)
    
#     else:
#         x = np.asarray(x)
#         # Get dimension (d) from x
#         if x.ndim == 1:
#             # x is a 1-dimensional array
#             x = x.reshape(-1, 1)
#             d = 1
#         else:
#             # x is a 2-dimensional array
#             d = x.shape[1]

#         n = x.shape[0]

#         # Check size of v
#         if v is None:
#             err_msg = f'{fname}: `x` is not None but `v` is None'
#             if logger: logger.error(err_msg)
#             raise CovModelError(err_msg)

#         v = np.asarray(v).reshape(-1)
#         if v.size != n:
#             err_msg = f'{fname}: size of `v` is not valid'
#             if logger: logger.error(err_msg)
#             raise CovModelError(err_msg)

#         if ncategory > 1:
#             if not np.all([vi in category_values for vi in v]):
#                 err_msg = f'{fname}: `v` contains an invalid category value (not in `category_values`)'
#                 if logger: logger.error(err_msg)
#                 raise CovModelError(err_msg)
#         else:
#             if not np.all([vi in np.hstack((category_values, np.array([0.0]))) for vi in v]):
#                 err_msg = f'{fname}: `v` contains an invalid category value (not in `category_values` nor 0.0)'
#                 if logger: logger.error(err_msg)
#                 raise CovModelError(err_msg)
        
#         v_indicator = np.asarray([np.eye(ncategory)[np.where(np.asarray(category_values) == vi)[0][0]] for vi in v]).T
    
#     # Get dimension from xu (du) and number of unknown points (nu)
#     if xu is None:
#         nu = 0
#         du = 0
#     else:
#         xu = np.asarray(xu)
#         # Get dimension (du) from xu
#         if xu.ndim == 1:
#             # xu is a 1-dimensional array
#             xu = xu.reshape(-1, 1)
#             du = 1
#         else:
#             # xu is a 2-dimensional array
#             du = xu.shape[1]

#         nu = xu.shape[0]

#     if nu == 0:
#         vu, vu_std = np.zeros((0, ncategory), dtype='float'), np.zeros((0, ncategory), dtype='float')
#         return vu, vu_std

#     # Here: nu > 0

#     if n > 0:
#         # Check dimension of x and xu
#         if d != du:
#             err_msg = f'{fname}: `x` and `xu` do not have the same dimension'
#             if logger: logger.error(err_msg)
#             raise CovModelError(err_msg)

#     else: # n == 0
#         d = du # set d to du if no data point

#     # Method and probability
#     if method == 'simple_kriging':
#         if n == 0:
#             probability_x = np.zeros(shape=(0, ncategory), dtype='float')
#         else:
#             if probability_x is None:
#                 probability_x = np.repeat(np.mean(v_indicator, axis=1), n).reshape(ncategory, n)
#             else:
#                 probability_x = np.asarray(probability_x, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
#                 if probability_x.size == ncategory:
#                     probability_x = np.repeat(probability_x, n)
#                 elif probability_x.size == n*ncategory:
#                     probability_x = probability_x.reshape(ncategory, n)
#                 else:
#                     err_msg = f'{fname}: size of `probability_x` is not valid'
#                     if logger: logger.error(err_msg)
#                     raise CovModelError(err_msg)
                
#         if probability_xu is None:
#             if n == 0:
#                 if ncategory > 1:
#                     probability_xu = np.full((ncategory, nu), 1.0/ncategory)
#                 else: # ncategory == 1
#                     probability_xu = np.full((ncategory, nu), 0.5)
#             else:
#                 probability_xu = np.repeat(np.mean(v_indicator, axis=1), nu).reshape(ncategory, nu)
#         else:
#             probability_xu = np.asarray(probability_xu, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
#             if probability_xu.size == ncategory:
#                 probability_xu = np.repeat(probability_xu, nu)
#             elif probability_xu.size == nu*ncategory:
#                 probability_xu = probability_xu.reshape(ncategory, nu)
#             else:
#                 err_msg = f'{fname}: size of `probability_xu` is not valid'
#                 if logger: logger.error(err_msg)
#                 raise CovModelError(err_msg)
            
#     elif method == 'ordinary_kriging':
#         if ncategory > 1:
#             probability_xu = np.full((ncategory, nu), 1.0/ncategory)
#         else: # ncategory == 1
#             probability_xu = np.full((ncategory, nu), 0.5)
#         # probability_x = np.full((ncategory,), None)
#         # probability_xu = np.full((ncategory,), None)

#     else:
#         err_msg = f'{fname}: `method` invalid'
#         if logger: logger.error(err_msg)
#         raise CovModelError(err_msg)

#     # cov_model, alpha_xu, beta_xu, gamm_xu, and cov_model_non_stationarity_xu_list
#     # should be either the same for all categories or given per category
#     cov_model = list(np.asarray(cov_model).reshape(-1))
#     if len(cov_model) == 1:
#         use_same_cov_model_for_all_categories = True
#         cov_model = cov_model[0]
#     elif len(cov_model) != ncategory:
#         err_msg = f'{fname}: `cov_model_for_category` of invalid length'
#         if logger: logger.error(err_msg)
#         raise CovModelError(err_msg)
#     else:
#         use_same_cov_model_for_all_categories = False
#         if alpha_xu is None:
#             alpha_xu = ncategory * [None]
#         if beta_xu is None:
#             beta_xu = ncategory * [None]
#         if gamma_xu is None:
#             gamma_xu = ncategory * [None]
#         if cov_model_non_stationarity_xu_list is None:
#             cov_model_non_stationarity_xu_list = ncategory * [None]

#     if isinstance(use_unique_neighborhood, bool):
#         use_unique_neighborhood = ncategory * [use_unique_neighborhood]
#     elif not isinstance(use_unique_neighborhood, list) or len(use_unique_neighborhood) != ncategory:
#         err_msg = f'{fname}: `use_unique_neighborhood` should be a single entry or a list of length equal to the number of categories'
#         if logger: logger.error(err_msg)
#         raise CovModelError(err_msg)
    
#     if searchRadius is None:
#         searchRadius = ncategory * [None]
#     elif isinstance(searchRadius, float) or isinstance(searchRadius, int):
#         searchRadius = ncategory * [searchRadius]
#     elif not isinstance(searchRadius, list) or len(searchRadius) != ncategory:
#         err_msg = f'{fname}: `searchRadius` should be `None`, a single entry or a list of length equal to the number of categories'
#         if logger: logger.error(err_msg)
#         raise CovModelError(err_msg)
    
#     if searchRadiusRelative is None:
#         searchRadiusRelative = ncategory * [None]
#     elif isinstance(searchRadiusRelative, float) or isinstance(searchRadiusRelative, int):
#         searchRadiusRelative = ncategory * [searchRadiusRelative]
#     elif not isinstance(searchRadiusRelative, list) or len(searchRadiusRelative) != ncategory:
#         err_msg = f'{fname}: `searchRadiusRelative` should be a list of length equal to the number of categories'
#         if logger: logger.error(err_msg)
#         raise CovModelError(err_msg)
    
#     if isinstance(nneighborMax, int):
#         nneighborMax = ncategory * [nneighborMax]          
#     elif not isinstance(nneighborMax, list) or len(nneighborMax) != ncategory:
#         err_msg = f'{fname}: `nneighborMax` should be a single entry or a list of length equal to the number of categories'
#         if logger: logger.error(err_msg)
#         raise CovModelError(err_msg)

#     vu_indicator = np.zeros((ncategory, nu))
#     # vu_std_indicator = np.zeros((ncategory, nu))

#     if use_same_cov_model_for_all_categories:
#         for i in range(ncategory):
#             vu_indicator[i], _ = krige(
#                     x, v_indicator[i], xu, cov_model,
#                     method=method,
#                     mean_x=probability_x[i],
#                     mean_xu=probability_xu[i],
#                     alpha_xu=alpha_xu,
#                     beta_xu=beta_xu,
#                     gamma_xu=gamma_xu,
#                     cov_model_non_stationarity_xu_list=cov_model_non_stationarity_xu_list,
#                     use_unique_neighborhood=use_unique_neighborhood[i],
#                     searchRadius=searchRadius[i],
#                     searchRadiusRelative=searchRadiusRelative[i],
#                     nneighborMax=nneighborMax[i],
#                     verbose=verbose)
#     else:
#         for i in range(ncategory):
#             vu_indicator[i], _ = krige(
#                     x, v_indicator[i], xu, cov_model[i],
#                     method=method,
#                     mean_x=probability_x[i],
#                     mean_xu=probability_xu[i],
#                     alpha_xu=alpha_xu[i],
#                     beta_xu=beta_xu[i],
#                     gamma_xu=gamma_xu[i],
#                     cov_model_non_stationarity_xu_list=cov_model_non_stationarity_xu_list[i],
#                     use_unique_neighborhood=use_unique_neighborhood[i],
#                     searchRadius=searchRadius[i],
#                     searchRadiusRelative=searchRadiusRelative[i],
#                     nneighborMax=nneighborMax[i],
#                     verbose=verbose)
            
#     vu_indicator = np.maximum(0., np.minimum(1., vu_indicator))
#     vu_indicator = vu_indicator/vu_indicator.sum(axis=0)

#     return vu_indicator
# # ----------------------------------------------------------------------------

# ============================================================================

if __name__ == "__main__":
    print("Module 'geone.covModel' example:")

    ########## 1D case ##########
    # Define covariance model
    cov_model = CovModel1D(elem=[
                    ('gaussian', {'w':5., 'r':100}), # elementary contribution
                    ('nugget', {'w':1.})             # elementary contribution
                    ], name='model-1D example')

    # Plot covariance and variogram functions on same plot
    cov_model.plot_model(label='cov', show_ylabel=False)
    cov_model.plot_model(vario=True, label='vario', show_ylabel=False)
    # plt.ylabel('') # remove label for y-axis
    plt.legend()
    plt.title(cov_model.name)
    # Set custom axes (through the origin)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    #ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    #ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))
    plt.show()

    # ########## 2D case ##########
    # Define covariance model
    cov_model = CovModel2D(elem=[
                    ('gaussian', {'w':8.5, 'r':[150, 40]}), # elementary contribution
                    ('nugget', {'w':0.5})                   # elementary contribution
                    ], alpha=-30., name='model-2D example')

    # Plot covariance function (in a new 1x2 figure, without suptitle)
    cov_model.plot_model(show_suptitle=False)
    plt.show()

    # Plot variogram function (in a new 1x2 figure)
    cov_model.plot_model(vario=True)
    plt.show()

    ########## 3D case ##########
    # Define covariance model
    cov_model = CovModel3D(elem=[
                    ('gaussian', {'w':8.5, 'r':[40, 20, 10]}), # elementary contribution
                    ('nugget', {'w':0.5})                      # elementary contribution
                    ], alpha=-30., beta=-40., gamma=20., name='model-3D example')

    # Plot covariance function
    # ... volume (3D)
    cov_model.plot_model3d_volume()
    # ... slice in 3D block
    cov_model.plot_model3d_slice()
    # ... curves along each main axis
    cov_model.plot_model_curves()
    plt.show()

    # Plot variogram function
    # ... volume (3D)
    cov_model.plot_model3d_volume(vario=True)
    # ... slice in 3D block
    cov_model.plot_model3d_slice(vario=True)
    # ... curves along each main axis
    cov_model.plot_model_curves(vario=True)
    plt.show()

    a = input("Press enter to continue...")
