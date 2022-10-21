#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python module:  'covModel.py'
authors:        Julien Straubhaar and Philippe Renard
date:           2018-2020

Module for:
    - definition of (classic) covariance / variogram models in 1D, 2D, and 3D
        (omni-directional or anisotropic)
    - covariance / variogram analysis and fitting
    - ordinary kriging
    - cross-validation (leave-one-out (loo))
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.special
import scipy.optimize
from scipy import stats
import pyvista as pv
import copy

from geone import img
from geone import imgplot as imgplt
from geone import imgplot3d as imgplt3

# ============================================================================
# Definition of 1D elementary covariance models:
#   - nugget, spherical, exponential, gaussian, linear, cubic, sinus_cardinal
#       parameters: w, r
#   - gamma, exponential_generalized, power (non-stationary)
#       parameters: w, r, s (power)
#   - matern
#       parameters: w, r, nu
# ============================================================================
# ----------------------------------------------------------------------------
def cov_nug(h, w=1.0):
    """
    1D-nugget covariance model:

    :param h:   (1-dimensional array or float): lag(s)
    :param w:   (float >0): weight (sill)
    :return:    (1-dimensional array or float) evaluation of the model at h:
                    w * f(h), where
                    f(h) = 1, if h=0
                    f(h) = 0, otherwise
    """
    return w * np.asarray(h==0., dtype=float)

def cov_sph(h, w=1.0, r=1.0):
    """
    1D-shperical covariance model:

    :param h:   (1-dimensional array or float): lag(s)
    :param w:   (float >0): weight (sill)
    :param r:   (float >0): range
    :return:    (1-dimensional array or float) evaluation of the model at h:
                    w * f(|h|/r), where
                    f(t) = 1 - 3/2 * t + 1/2 * t**3, if 0 <= t < 1
                    f(t) = 0,                        if t >= 1
    """
    t = np.minimum(np.abs(h)/r, 1.) # "parallel or element-wise minimum"
    return w * (1 - 0.5 * t * (3. - t**2))

def cov_exp(h, w=1.0, r=1.0):
    """
    1D-exponential covariance model:

    :param h:   (1-dimensional array or float): lag(s)
    :param w:   (float >0): weight (sill)
    :param r:   (float >0): range
    :return:    (1-dimensional array or float) evaluation of the model at h:
                    w * f(|h|/r), where
                    f(t) = exp(-3*t)
    """
    return w * np.exp(-3. * np.abs(h)/r)

def cov_gau(h, w=1.0, r=1.0):
    """
    1D-gaussian covariance model:

    :param h:   (1-dimensional array or float): lag(s)
    :param w:   (float >0): weight (sill)
    :param r:   (float >0): range
    :return:    (1-dimensional array or float) evaluation of the model at h:
                    w * f(h/r), where
                    f(t) = exp(-3*t**2)
    """
    return w * np.exp(-3. * (h/r)**2)

def cov_lin(h, w=1.0, r=1.0):
    """
    1D-linear (with sill) covariance model:

    :param h:   (1-dimensional array or float): lag(s)
    :param w:   (float >0): weight (sill)
    :param r:   (float >0): range
    :return:    (1-dimensional array or float) evaluation of the model at h:
                    w * f(|h|/r), where
                    f(t) = 1 - t, if 0 <= t < 1
                    f(t) = 0,     if t >= 1

    """
    t = np.minimum(np.abs(h)/r, 1.) # "parallel or element-wise minimum"
    return w * (1.0 - t)

def cov_cub(h, w=1.0, r=1.0):
    """
    1D-cubic covariance model:

    :param h:   (1-dimensional array or float): lag(s)
    :param w:   (float >0): weight (sill)
    :param r:   (float >0): range
    :return:    (1-dimensional array or float) evaluation of the model at h:
                    w * f(|h|/r), where
                    f(t) = 1 - 7 * t**2 + 35/4 * t**3 - 7/2 * t**5 + 3/4 * t**7, if 0 <= t < 1
                    f(t) = 0,                                                     if t >= 1
    """
    t = np.minimum(np.abs(h)/r, 1.) # "parallel or element-wise minimum"
    t2 = t**2
    return w * (1 + t2 * (-7. + t * (8.75 + t2 * (-3.5 + 0.75 * t2))))

def cov_sinc(h, w=1.0, r=1.0):
    """
    1D-sinus-cardinal (normalized) covariance model:

    :param h:   (1-dimensional array or float): lag(s)
    :param w:   (float >0): weight (sill)
    :param r:   (float >0): range
    :return:    (1-dimensional array or float) evaluation of the model at h:
                    w * f(h/r), where
                    f(t) = sin(pi*t)/(pi*t)

    """
    # np.sinc(x) = np.sin(np.pi*x)/(np.pi*x)
    return w * np.sinc(h/r)

def cov_gamma(h, w=1.0, r=1.0, s=1.0):
    """
    1D-gamma covariance model:

    :param h:   (1-dimensional array or float): lag(s)
    :param w:   (float >0): weight (sill)
    :param r:   (float >0): range
    :param s:   (float >0): power
    :return:    (1-dimensional array or float) evaluation of the model at h:
                    w * f(|h|/r), where
                    f(t) = 1 / (1 + alpha*t)**s, with alpha = 20**(1/s) - 1

    """
    alpha = 20.0**(1.0/s) - 1.0
    return w / (1.0 + alpha * np.abs(h)/r)**s

def cov_pow(h, w=1.0, r=1.0, s=1.0):
    """
    1D-power covariance model (not really sill and range):

    :param h:   (1-dimensional array or float): lag(s)
    :param w:   (float >0): weight
    :param r:   (float >0): scale
    :param s:   (float btw 0 and 2): power
    :return:    (1-dimensional array or float) evaluation of the model at h:
                    w * f(|h|/r), where
                    f(t) = 1 - t**s

    """
    return w * (1. - (np.abs(h)/r)**s)

def cov_exp_gen(h, w=1.0, r=1.0, s=1.0):
    """
    1D-exponential-generalized covariance model:

    :param h:   (1-dimensional array or float): lag(s)
    :param w:   (float >0): weight (sill)
    :param r:   (float >0): range
    :param s:   (float >0): power
    :return:    (1-dimensional array or float) evaluation of the model at h:
                    w * f(|h|/r), where
                    f(t) = exp(-3*t**s)

    """
    return w * np.exp(-3. * (np.abs(h)/r)**s)

def cov_matern(h, w=1.0, r=1.0, nu=0.5):
    """
    1D-Matern covariance model (the effective range depends on nu):

    :param h:   (1-dimensional array or float): lag(s)
    :param w:   (float >0): weight (sill)
    :param r:   (float >0): scale (the effective range depends on nu)
    :param nu:  (float >0): parameter for Matern covariance
    :return:    (1-dimensional array or float) evaluation of the model at h:
                    w * 1.0/(2.0**(nu-1.0)*Gamma(nu)) * u**nu * K_{nu}(u), where
                    u = np.sqrt(2.0*nu)/r * |h|
                    Gamma is the function gamma
                    K_{nu} is the modified Bessel function of the second kind of parameter nu
                Note that
                    1) cov_matern(h, w, r, nu=0.5) = cov_exp(h, w, 3*r)
                    2) cov_matern(h, w, r, nu) tends to cov_gau(h, w, np.sqrt(6)*r)
                    when nu tends to infinity
    """
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
    Computes the effective range of the 1D-Matern covariance model
    of parameters 'nu' and 'r' (scale).

    :param nu:      (float >0): parameter for Matern covariance
    :param r:       (float >0): parameter r (scale) of the Matern covariance
    :return r:      (float): effective range
    """
    res = scipy.optimize.root_scalar(lambda h: cov_matern(h, w=1.0, r=r, nu=nu) - 0.05, bracket=[1.e-10*r, 4.0*r])
    return res.root
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def cov_matern_get_r_param(nu, r_eff):
    """
    Computes the parameter 'r' (scale) such that the 1D-Matern covariance model
    of parameter 'nu' (given) has an effective range of 'r_eff' (given).

    :param nu:      (float >0): parameter for Matern covariance
    :param r_eff:   (float >0): effective range
    :return r:      (float): parameter r (scale) of the corresponding Matern covariance
    """
    res = scipy.optimize.minimize_scalar(lambda r: (cov_matern_get_effective_range(nu, r) - r_eff)**2, bracket=[1.e-10*r_eff, 4*r_eff])
    return res.x
# ----------------------------------------------------------------------------

# ============================================================================
# Definition of class for covariance models in 1D, 2D, 3D, as combination
# of elementary models and accounting for anisotropy and rotation
# ============================================================================
# ----------------------------------------------------------------------------
class CovModel1D (object):
    """
    Defines a covariance model in 1D:
        elem:   (sequence of 2-tuple) an entry (t, d) of the sequence
                    corresponds to an elementary model with:
                        t: (string) the type, can be
                           'nugget'         (see func geone.covModel.cov_nug)
                           'spherical'      (see func geone.covModel.cov_sph)
                           'exponential'    (see func geone.covModel.cov_exp)
                           'gaussian'       (see func geone.covModel.cov_gau)
                           'linear'         (see func geone.covModel.cov_lin)
                           'cubic'          (see func geone.covModel.cov_cub)
                           'sinus_cardinal' (see func geone.covModel.cov_sinc)
                           'gamma'          (see func geone.covModel.cov_gamma)
                           'power'          (see func geone.covModel.cov_pow)
                           'exponential_generalized'
                                            (see func geone.covModel.cov_exp_gen)
                           'matern'         (see func geone.covModel.cov_matern)
                        d: (dict) dictionary of required parameters to be
                            passed to the elementary model,
                    e.g.
                       (t, d) = ('spherical', {'w':2.0, 'r':1.5})
                       (t, d) = ('power', {'w':2.0, 'r':1.5, 's':1.7})
                       (t, d) = ('matern', {'w':2.0, 'r':1.5, 'nu':1.5})
                    the final model is the sum of the elementary models
        name:   (string) name of the model

    Example: to define a covariance model (1D) that is a combination of
        2 elementary structures:
            - gaussian with a contribution of 10. and a range of 100.,
            - nugget of (contribution) 0.5
    >>> cov_model = CovModel1D(
            elem=[
                ('gaussian', {'w':10., 'r':100.}), # elementary contribution
                ('nugget', {'w':0.5})              # elementary contribution
                ], name='gau+nug')                 # name is not necessary
    """

    def __init__(self,
                 elem=[],
                 name=None):
        for el in elem:
            if el[0] not in (
                    'nugget',
                    'spherical',
                    'exponential',
                    'gaussian',
                    'linear',
                    'cubic',
                    'sinus_cardinal',
                    'gamma',
                    'power',
                    'exponential_generalized',
                    'matern'
                    ):
                print('ERROR: unknown elementary contribution')
                return None
        self.elem = elem
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
        self._is_orientation_stationary = None
        self._is_weight_stationary = None
        self._is_range_stationary = None
        self._is_stationary = None

    # ------------------------------------------------------------------------
    # def __str__(self):
    #     return self.name
    def __repr__(self):
        out = '*** CovModel1D object ***'
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
        out = out + '\n' + '*****'
        return out
    # ------------------------------------------------------------------------

    def is_orientation_stationary(self, recompute=False):
        """Returns a bool (True / False) indicating if the orientation is
        stationary - always True for 1D covariance model.

        :param recompute:   (bool) True to force (re-)computing
        :return:            (bool) self._is_orientation_stationary
        """
        self._is_orientation_stationary = True
        return self._is_orientation_stationary

    def is_weight_stationary(self, recompute=False):
        """Returns a bool (True / False) indicating if the weight is stationary
        - i.e. the weight (sill) of any elementary contribution is defined as a
        unique value - (True), or not (False).

        :param recompute:   (bool) True to force (re-)computing
        :return:            (bool) self._is_weight_stationary
        """
        if self._is_weight_stationary is None or recompute:
            self._is_weight_stationary = not np.any([np.size(el[1]['w']) > 1 for el in self.elem])
        return self._is_weight_stationary

    def is_range_stationary(self, recompute=False):
        """Returns a bool (True / False) indicating if the range in every direction
        is stationary - i.e. the range in of any elementary contribution is defined
        as a unique value - (True), or not (False).

        :param recompute:   (bool) True to force (re-)computing
        :return:            (bool) self._is_range_stationary
        """
        if self._is_range_stationary is None or recompute:
            self._is_range_stationary = True
            for el in self.elem:
                if 'r' in el[1].keys() and np.size(el[1]['r']) > 1:
                    self._is_range_stationary = False
                    break
        return self._is_range_stationary

    def is_stationary(self, recompute=False):
        """Returns a bool (True / False) indicating if all the parameters are
        stationary - i.e. defined as a unique value - (True), or not (False).

        :param recompute:   (bool) True to force (re-)computing
        :return:            (bool) self._is_stationary
        """
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
        """Returns the sill (sum of weight of each elementary contribution).

        :param recompute:   (bool) True to force (re-)computing
        :return:            (float) self._sill
        """
        if self._sill is None or recompute:
            # Prevent calculation if weight is not stationary
            if not self.is_weight_stationary(recompute):
                self._sill = None
                return self._sill
            # print('Computing sill...')
            self._sill = sum([d['w'] for t, d in self.elem if 'w' in d])
        return self._sill

    def r(self, recompute=False):
        """Returns the range (max over elementary contributions).
        For each elementary contribution the "effective" range is retrieved,
        i.e. the distance beyond which the covariance is zero or below 5% of
        the weight (this corresponds to the parameter r for most of covariance
        types).

        :param recompute:   (bool) True to force (re-)computing
        :return:            (float) self._r
        """
        if self._r is None or recompute:
            # Prevent calculation if range is not stationary
            if not self.is_range_stationary(recompute):
                self._r = None
                return self._r
            # print('Computing effective range (max)...')
            r = 0.
            for t, d in self.elem:
                if t in (
                        'spherical',
                        'exponential',
                        'gaussian',
                        'linear',
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
        Returns the covariance model function f where:
            h:      (1-dimensional array or float) 1D-lag(s)
            f(h):   (1-dimensional array) evaluation of the covariance model at h
                        note that the result is casted to a 1-dimensional array
        """
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

                elif t == 'linear':
                    s = s + cov_lin(h, **d)

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
        Returns the varioram model function f(h) where:
            h:      (1-dimensional array or float) 1D-lag(s)
            f(h):   (1-dimensional array) evaluation of the variogram model at h
                        note that the result is casted to a 1-dimensional array
        """
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

                elif t == 'linear':
                    s = s + d['w'] - cov_lin(h, **d)

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

    def plot_model(self, vario=False, hmin=0, hmax=None, npts=500,
        grid=True, show_xlabel=True, show_ylabel=True, **kwargs):
        """
        Plot covariance or variogram function (in current figure axis).

        :param vario:   (bool)
                            - if False: plot covariance function
                            - if True:  plot variogram function
        :param hmin, hmax:  (float) function is plotted for h in interval [hmin, hmax]
                                hmax=None for default: 1.2 * range max
        :param npts:    (int) number of points used in interval [hmin, hmax]
        :param grid:    (bool) indicates if a grid is plotted (True by default)
        :param show_xlabel, show_ylabel:
                        (bool) indicates if (default) label for x axis (resp. y axis)
                            is displayed
        :kwargs:        keyword arguments passed to the funtion plt.plot
        """
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
        if vario:
            g = self.vario_func()(h)
        else:
            g = self.func()(h)

        plt.plot(h, g, **kwargs)

        if show_xlabel:
            plt.xlabel('h')
        if show_ylabel:
            if vario:
                plt.ylabel(r'$\gamma(h)$')
            else:
                plt.ylabel(r'$cov(h)$')

        if grid:
            plt.grid(True)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
class CovModel2D (object):
    """
    Defines a covariance model in 2D:
        elem:   (sequence of 2-tuple) an entry (t, d) of the sequence
                    corresponds to an elementary model with:
                        t: (string) the type, can be
                           'nugget'         (see func geone.covModel.cov_nug)
                           'spherical'      (see func geone.covModel.cov_sph)
                           'exponential'    (see func geone.covModel.cov_exp)
                           'gaussian'       (see func geone.covModel.cov_gau)
                           'linear'         (see func geone.covModel.cov_lin)
                           'cubic'          (see func geone.covModel.cov_cub)
                           'sinus_cardinal' (see func geone.covModel.cov_sinc)
                           'gamma'          (see func geone.covModel.cov_gamma)
                           'power'          (see func geone.covModel.cov_pow)
                           'exponential_generalized'
                                            (see func geone.covModel.cov_exp_gen)
                           'matern'         (see func geone.covModel.cov_matern)
                        d: (dict) dictionary of required parameters to be
                            passed to the elementary model, excepting
                            the parameter 'r' which must be given here
                            as a sequence (array) of ranges along each axis
                    e.g.
                       (t, d) = ('spherical', {'w':2.0, 'r':[1.5, 2.5]})
                       (t, d) = ('power', {'w':2.0, 'r':[1.5, 2.5], 's':1.7})
                       (t, d) = ('matern', {'w':2.0, 'r':[1.5, 2.5], 'nu':1.5})
                    the final model is the sum of the elementary models
        alpha:  (float) azimuth angle in degrees:
                    the system Ox'y', supporting the axes of the model (ranges),
                    is obtained from the system Oxy by applying a rotation of
                    angle -alpha.
                    The 2x2 matrix m for changing the coordinate system from
                    Ox'y' to Oxy is:
                            +                         +
                            |  cos(alpha)   sin(alpha)|
                        m = | -sin(alpha)   cos(alpha)|
                            +                         +
        name:   (string) name of the model

    Example: to define a covariance model (2D) that is a combination of
        2 elementary structures:
            - gaussian with a contribution of 10. and ranges of 150. and 50.,
                along axis x' and axis y' resp. defined by the angle alpha=-30.
                (see above)
            - nugget of (contribution) 0.5
    >>> cov_model = CovModel2D(elem=[
            ('gaussian', {'w':10., 'r':[150, 50]}), # elementary contribution
            ('nugget', {'w':0.5})                   # elementary contribution
            ], alpha=-30., name='')
    """

    def __init__(self,
                 elem=[],
                 alpha=0.,
                 name=None):
        for el in elem:
            if el[0] not in (
                    'nugget',
                    'spherical',
                    'exponential',
                    'gaussian',
                    'linear',
                    'cubic',
                    'sinus_cardinal',
                    'gamma',
                    'power',
                    'exponential_generalized',
                    'matern'
                    ):
                print('ERROR: unknown elementary contribution')
                return None
        self.elem = elem
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

    def is_orientation_stationary(self, recompute=False):
        """Returns a bool (True / False) indicating if the orientation is
        stationary - i.e. the angle alpha is defined as a unique value - (True),
        or not (False).

        :param recompute:   (bool) True to force (re-)computing
        :return:            (bool) self._is_orientation_stationary
        """
        if self._is_orientation_stationary is None or recompute:
            self._is_orientation_stationary = np.size(self.alpha) == 1
        return self._is_orientation_stationary

    def is_weight_stationary(self, recompute=False):
        """Returns a bool (True / False) indicating if the weight is stationary
        - i.e. the weight (sill) of any elementary contribution is defined as a
        unique value - (True), or not (False).

        :param recompute:   (bool) True to force (re-)computing
        :return:            (bool) self._is_weight_stationary
        """
        if self._is_weight_stationary is None or recompute:
            self._is_weight_stationary = not np.any([np.size(el[1]['w']) > 1 for el in self.elem])
        return self._is_weight_stationary

    def is_range_stationary(self, recompute=False):
        """Returns a bool (True / False) indicating if the range in every direction
        is stationary - i.e. the range in any direction and of any elementary
        contribution is defined as a unique value - (True), or not (False).

        :param recompute:   (bool) True to force (re-)computing
        :return:            (bool) self._is_range_stationary
        """
        if self._is_range_stationary is None or recompute:
            self._is_range_stationary = True
            for el in self.elem:
                if 'r' in el[1].keys() and np.any([np.size(ri) > 1 for ri in el[1]['r']]):
                    self._is_range_stationary = False
                    break
        return self._is_range_stationary

    def is_stationary(self, recompute=False):
        """Returns a bool (True / False) indicating if all the parameters are
        stationary - i.e. defined as a unique value - (True), or not (False).

        :param recompute:   (bool) True to force (re-)computing
        :return:            (bool) self._is_stationary
        """
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
        """Returns the sill (sum of weight of each elementary contribution).

        :param recompute:   (bool) True to force (re-)computing
        :return:            (float) self._sill
        """
        if self._sill is None or recompute:
            # Prevent calculation if weight is not stationary
            if not self.is_weight_stationary(recompute):
                self._sill = None
                return self._sill
            # print('Computing sill...')
            self._sill = sum([d['w'] for t, d in self.elem if 'w' in d])
        return self._sill

    def mrot(self, recompute=False):
        """Returns the 2x2 matrix m for changing the coordinate system from Ox'y'
        to Oxy, where Ox' and Oy' are the axes supporting the ranges of the model.

        :param recompute:   (bool) True to force (re-)computing
        :return:            (2d ndarray of shape (2,2)) self._mrot
        """
        if self._mrot is None or recompute:
            # Prevent calculation if orientation is not stationary
            if not self.is_orientation_stationary(recompute):
                self._mrot = None
                return self._mrot
            # print('Computing rotation matrix...')
            a = self.alpha * np.pi/180.
            ca, sa = np.cos(a), np.sin(a)
            self._mrot = np.array([[ca, sa], [-sa, ca]])
        return self._mrot

    def r12(self, recompute=False):
        """Returns the range (max over elementary contributions) along each axis
        in the new coordinate system (corresponding to the axes of the ellipse
        supporting the covariance model).
        For each elementary contribution the "effective" range is retrieved,
        i.e. the distance beyond which the covariance is zero or below 5% of
        the weight (this corresponds to the parameter r for most of covariance
        types).

        :param recompute:   (bool) True to force (re-)computing
        :return:            (1d ndarray of 2 floats) self._r
        """
        if self._r is None or recompute:
            # Prevent calculation if range is not stationary
            if not self.is_range_stationary(recompute):
                self._r = None
                return self._r
            # print('Computing effective range (max)...')
            r = np.array([0., 0.])
            for t, d in self.elem:
                if t in (
                        'spherical',
                        'exponential',
                        'gaussian',
                        'linear',
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
        """Returns the range (max over elementary contributions) along each axis
        in the original coordinate system.
        For each elementary contribution the "effective" range is retrieved,
        i.e. the distance beyond which the covariance is zero or below 5% of
        the weight (this corresponds to the parameter r for most of covariance
        types).

        :param recompute:   (bool) True to force (re-)computing
        :return:            (1d ndarray of 2 floats) effective range along
                                original system axes
        """
        # Prevent calculation if range or orientation is not stationary
        if not self.is_range_stationary(recompute) or not self.is_orientation_stationary(recompute):
            return None
        r12 = self.r12(recompute)
        m = np.abs(self.mrot(recompute))
        return np.maximum(r12[0] * m[:,0], r12[1] * m[:,1]) # element-wise maximum

    def func(self):
        """
        Returns the covariance model function f(h) where:
            h:      (2-dimensional array of dim n x 2, or
                        1-dimensional array of dim 2) 2D-lag(s)
            f(h):   (1-dimensional array of dim n) evaluation of the model at h
        """
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

                elif t == 'linear':
                    s = s + cov_lin(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

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
        Returns the variogram model function f(h) where:
            h:      (2-dimensional array of dim n x 2, or
                        1-dimensional array of dim 2) 2D-lag(s)
            f(h):   (1-dimensional array of dim n) evaluation of the model at h
        """
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

                elif t == 'linear':
                    s = s + d['w'] - cov_lin(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

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
        Plot system Oxy and Ox'y' (in current figure axis).

        :param color0, color1:  colors for main axes x', y'
        """
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

    def plot_model(self, vario=False, plot_map=True, plot_curves=True,
        cmap='terrain', color0='red', color1='green',
        extent=None, ncell=(201, 201),
        h1min=0, h1max=None, h2min=0, h2max=None, n1=500, n2=500,
        grid=True, show_xlabel=True, show_ylabel=True, show_suptitle=True, figsize=None):
        """
        Plot covariance or variogram function
            - map of the function, and / or
            - curves along axis x' and axis y' (where Ox' and Oy' are the axes supporting the ranges of the model)

        :param vario:   (bool)
                            - if False: plot covariance function
                            - if True:  plot variogram function

        :param plot_map, plot_curves:
                        (bool) indicates what is plotted:
                            - plot_map is True  and plot_curves is True :
                                plot map and curves along axis x' and axis y' in a new 1x2 figure
                            - plot_map is True and plot_curves is False:
                                plot map in current figure axis
                            - plot_map is False and plot_curves is True :
                                plot curves along axis x' and axis y' in current figure axis
                            - plot_map is False and plot_curves is False:
                                nothing is done

        :param cmap:            color map
        :param color0, color1:  colors for curves along axis x' and along axis y' resp.

        :param extent:  (hxmin, hxmax, hymin, hymax): 4 floats defining the domain of the map.
                            None for default

        :param ncell:   (nx, ny): 2 ints defining the number of the cells in the map (nx x ny)

        :param h1min, h1max:    function is plotted along x' for h in interval [h1min, h1max] (default h1max if None)
        :param h2min, h2max:    function is plotted along y' for h in interval [h2min, h2max] (default h2max if None)
        :param n1, n2:          number of points in interval [h1min, h1max] and [h2min, h2max] resp.
        :param show_xlabel, show_ylabel:
                        (bool) indicates if label for x axis (resp. y axis)
                            is displayed (True by default), for curves plot
        :param show_suptitle:
                        (bool) indicates if suptitle is displayed (True by default),
                            in case of map and curves are plotted (1x2 figure)
        :param grid:    (bool) indicates if a grid is plotted (True by default) for curves plot
        :param figsize: (tuple of 2 ints) size of the figure, used if a new 1x2 figure is created
                            (i.e. if plot_map and plot_curves are set to True)
        """
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
            if vario:
                gg = self.vario_func()(hh).reshape(ny, nx)
            else:
                gg = self.func()(hh).reshape(ny, nx)

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
            if vario:
                g1 = self.vario_func()(hh1.dot(mrot.T)) # hh1.dot(mrot.T): 2D-lags in system Oxy (what is taken by the function)
            else:
                g1 = self.func()(hh1.dot(mrot.T)) # hh1.dot(mrot.T): 2D-lags in system Oxy (what is taken by the function)

            # Evaluate function along axis y'
            h2 = np.linspace(h2min, h2max, n2)
            hh2 = np.hstack((np.zeros((len(h2),1)), h2.reshape(-1,1))) # (n2,2) array) 2D-lags along y' expressed in system Ox'y'
            if vario:
                g2 = self.vario_func()(hh2.dot(mrot.T)) # hh2.dot(mrot.T): 2D-lags in system Oxy (what is taken by the function)
            else:
                g2 = self.func()(hh2.dot(mrot.T)) # hh2.dot(mrot.T): 2D-lags in system Oxy (what is taken by the function)

        # Plot...
        if plot_map and plot_curves:
            # Figure (new)
            fig, ax = plt.subplots(1,2, figsize=figsize)
            plt.sca(ax[0])

        if plot_map:
            # Plot map and system Ox'y'
            # ... map
            imgplt.drawImage2D(im, cmap=cmap)
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
            if grid:
                plt.grid(True)

        if plot_map and plot_curves and show_suptitle:
            if vario:
                s = ['Model (vario): alpha={}'.format(self.alpha)] + ['{}'.format(el) for el in self.elem]
            else:
                s = ['Model (cov): alpha={}'.format(self.alpha)] + ['{}'.format(el) for el in self.elem]
            plt.suptitle('\n'.join(s))
            # plt.show()

    def plot_model_one_curve(self, main_axis=1, vario=False, hmin=0, hmax=None, npts=500,
        grid=True, show_xlabel=True, show_ylabel=True, **kwargs):
        """
        Plot covariance or variogram curve along one main axis (in current figure axis).

        :param main_axis:   (int) 1 or 2:
                                1: plot curve along x',
                                2: plot curve along y'
        :param vario:   (bool)
                            - if False: plot covariance function
                            - if True:  plot variogram function
        :param hmin, hmax:  (float) function is plotted for h in interval [hmin, hmax]
                                hmax=None for default: 1.2 * range max
        :param npts:    (int) number of points used in interval [hmin, hmax]
        :param grid:    (bool) indicates if a grid is plotted (True by default)
        :param show_xlabel, show_ylabel:
                        (bool) indicates if label for x axis (resp. y axis)
                            is displayed (True by default)
        :kwargs:        keyword arguments passed to the funtion plt.plot
        """
        # Prevent calculation if covariance model is not stationary
        if not self.is_stationary():
            return None
        if main_axis not in (1, 2):
            print('ERROR: main_axis not valid (should be 1 or 2)')
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
            hmax = 1.2*self.r12()[main_axis-1]

        # Rotation matrix
        mrot = self.mrot()

        # Evaluate function along selected axis
        h = np.linspace(hmin, hmax, npts)
        if main_axis == 1:
            hh = np.hstack((h.reshape(-1,1), np.zeros((len(h),1)))) # (npts,2) array of 2D-lags along x' expressed in system Ox'y'
        else:
            hh = np.hstack((np.zeros((len(h),1)), h.reshape(-1,1))) # (npts,2) array of 2D-lags along y' expressed in system Ox'y'
        if vario:
            g = self.vario_func()(hh.dot(mrot.T)) # hh.dot(mrot.T): 2D-lags in system Oxy (what is taken by the function)
        else:
            g = self.func()(hh.dot(mrot.T)) # hh.dot(mrot.T): 2D-lags in system Oxy (what is taken by the function)

        plt.plot(h, g, **kwargs)

        if show_xlabel:
            plt.xlabel('h')
        if show_ylabel:
            if vario:
                plt.ylabel(r'$\gamma(h)$')
            else:
                plt.ylabel(r'$cov(h)$')

        if grid:
            plt.grid(True)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
class CovModel3D (object):
    """
    Defines a covariance model in 3D:
        elem:   (sequence of 2-tuple) an entry (t, d) of the sequence
                    corresponds to an elementary model with:
                        t: (string) the type, can be
                           'nugget'         (see func geone.covModel.cov_nug)
                           'spherical'      (see func geone.covModel.cov_sph)
                           'exponential'    (see func geone.covModel.cov_exp)
                           'gaussian'       (see func geone.covModel.cov_gau)
                           'linear'         (see func geone.covModel.cov_lin)
                           'cubic'          (see func geone.covModel.cov_cub)
                           'sinus_cardinal' (see func geone.covModel.cov_sinc)
                           'gamma'          (see func geone.covModel.cov_gamma)
                           'power'          (see func geone.covModel.cov_pow)
                           'exponential_generalized'
                                            (see func geone.covModel.cov_exp_gen)
                           'matern'         (see func geone.covModel.cov_matern)
                        d: (dict) dictionary of required parameters to be
                            passed to the elementary model, excepting
                            the parameter 'r' which must be given here
                            as a sequence (array) of ranges along each axis
                    e.g.
                       (t, d) = ('spherical', {'w':2.0, 'r':[1.5, 2.5, 3.0]})
                       (t, d) = ('power', {'w':2.0, 'r':[1.5, 2.5, 3.0], 's':1.7})
                       (t, d) = ('matern', {'w':2.0, 'r':[1.5, 2.5, 3.0], 'nu':1.5})
                    the final model is the sum of the elementary models
        alpha, beta, gamma:
                (floats) azimuth, dip and plunge angles in degrees:
                the system Ox'''y''''z''', supporting the axes of the model
                (ranges), is obtained from the system Oxyz as follows:
                Oxyz      -- rotation of angle -alpha around Oz  --> Ox'y'z'
                Ox'y'z'   -- rotation of angle -beta  around Ox' --> Ox''y''z''
                Ox''y''z''-- rotation of angle -gamma around Oy''--> Ox'''y'''z'''
                The 3x3 matrix m for changing the coordinate system from
                Ox'''y'''z''' to Oxy is:
                    +                                                             +
                    |  ca * cc + sa * sb * sc,  sa * cb,  - ca * sc + sa * sb * cc|
                m = |- sa * cc + ca * sb * sc,  ca * cb,    sa * sc + ca * sb * cc|
                    |                 cb * sc,     - sb,                   cb * cc|
                    +                                                             +
                where
                    ca = cos(alpha), cb = cos(beta), cc = cos(gamma),
                    sa = sin(alpha), sb = sin(beta), sc = sin(gamma)
        name:   (string) name of the model

    Example: to define a covariance model (3D) that is a combination of
        2 elementary structures:
            - gaussian with a contributtion of 10. and ranges of 40., 20. and 10.,
                along axis x'' and axis y'', axis z'' resp. defined by the angles
                alpha=-30., beta=-40., and gamma=20. (see above)
            - nugget of (contribution) 0.5
    >>> cov_model = CovModel3D(elem=[
            ('gaussian', {'w':8.5, 'r':[40, 20, 10]}), # elementary contribution
            ('nugget', {'w':0.5})                      # elementary contribution
            ], alpha=-30., beta=-40., gamma=20., name='')
    """

    def __init__(self,
                 elem=[],
                 alpha=0., beta=0., gamma=0.,
                 name=None):
        for el in elem:
            if el[0] not in (
                    'nugget',
                    'spherical',
                    'exponential',
                    'gaussian',
                    'linear',
                    'cubic',
                    'sinus_cardinal',
                    'gamma',
                    'power',
                    'exponential_generalized',
                    'matern'
                    ):
                print('ERROR: unknown elementary contribution')
                return None
        self.elem = elem
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

    def is_orientation_stationary(self, recompute=False):
        """Returns a bool (True / False) indicating if the orientation is
        stationary - i.e. the angles alpha, beta and gamma are defined as a
        unique value - (True), or not (False).

        :param recompute:   (bool) True to force (re-)computing
        :return:            (bool) self._is_orientation_stationary
        """
        if self._is_orientation_stationary is None or recompute:
            self._is_orientation_stationary = np.size(self.alpha) == 1 and np.size(self.beta) == 1 and np.size(self.gamma) == 1
        return self._is_orientation_stationary

    def is_weight_stationary(self, recompute=False):
        """Returns a bool (True / False) indicating if the weight is stationary
        - i.e. the weight (sill) of any elementary contribution is defined as a
        unique value - (True), or not (False).

        :param recompute:   (bool) True to force (re-)computing
        :return:            (bool) self._is_weight_stationary
        """
        if self._is_weight_stationary is None or recompute:
            self._is_weight_stationary = not np.any([np.size(el[1]['w']) > 1 for el in self.elem])
        return self._is_weight_stationary

    def is_range_stationary(self, recompute=False):
        """Returns a bool (True / False) indicating if the range in every direction
        is stationary - i.e. the range in any direction and of any elementary
        contribution is defined as a unique value - (True), or not (False).

        :param recompute:   (bool) True to force (re-)computing
        :return:            (bool) self._is_range_stationary
        """
        if self._is_range_stationary is None or recompute:
            self._is_range_stationary = True
            for el in self.elem:
                if 'r' in el[1].keys() and np.any([np.size(ri) > 1 for ri in el[1]['r']]):
                    self._is_range_stationary = False
                    break
        return self._is_range_stationary

    def is_stationary(self, recompute=False):
        """Returns a bool (True / False) indicating if all the parameters are
        stationary - i.e. defined as a unique value - (True), or not (False).

        :param recompute:   (bool) True to force (re-)computing
        :return:            (bool) self._is_stationary
        """
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
        """Returns the sill (sum of weight of each elementary contribution).

        :param recompute:   (bool) True to force (re-)computing
        :return:            (float) self._sill
        """
        if self._sill is None or recompute:
            # Prevent calculation if weight is not stationary
            if not self.is_weight_stationary(recompute):
                self._sill = None
                return self._sill
            # print('Computing sill...')
            self._sill = sum([d['w'] for t, d in self.elem if 'w' in d])
        return self._sill

    def mrot(self, recompute=False):
        """Returns the 3x3 matrix m for changing the coordinate system from
        Ox'''y'''z''' to Oxyz, where Ox''', Oy''', Oz''' are the axes supporting
        the ranges of the model.

        :param recompute:   (bool) True to force (re-)computing
        :return:            (2d ndarray of shape (3,3)) self._mrot
        """
        if self._mrot is None or recompute:
            # Prevent calculation if orientation is not stationary
            if not self.is_orientation_stationary(recompute):
                self._mrot = None
                return self._mrot
            # print('Computing rotation matrix...')
            a = self.alpha * np.pi/180.
            b = self.beta * np.pi/180.
            c = self.gamma * np.pi/180.
            ca, sa = np.cos(a), np.sin(a)
            cb, sb = np.cos(b), np.sin(b)
            cc, sc = np.cos(c), np.sin(c)
            self._mrot = np.array([
                            [  ca * cc + sa * sb * sc,  sa * cb,  - ca * sc + sa * sb * cc],
                            [- sa * cc + ca * sb * sc,  ca * cb,    sa * sc + ca * sb * cc],
                            [                 cb * sc,     - sb,                  cb * cc ]])
        return self._mrot

    def r123(self, recompute=False):
        """Returns the range (max over elementary contributions) along each axis
        in the new coordinate system (corresponding to the axes of the ellipse
        supporting the covariance model).
        For each elementary contribution the "effective" range is retrieved,
        i.e. the distance beyond which the covariance is zero or below 5% of
        the weight (this corresponds to the parameter r for most of covariance
        types).

        :param recompute:   (bool) True to force (re-)computing
        :return:            (1d ndarray of 3 floats) self._r
        """
        if self._r is None or recompute:
            # Prevent calculation if range is not stationary
            if not self.is_range_stationary(recompute):
                self._r = None
                return self._r
            # print('Computing effective range (max)...')
            r = np.array([0., 0., 0.])
            for t, d in self.elem:
                if t in (
                        'spherical',
                        'exponential',
                        'gaussian',
                        'linear',
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
        """Returns the range (max over elementary contributions) along each axis
        in the original coordinate system.
        For each elementary contribution the "effective" range is retrieved,
        i.e. the distance beyond which the covariance is zero or below 5% of
        the weight (this corresponds to the parameter r for most of covariance
        types).

        :param recompute:   (bool) True to force (re-)computing
        :return:            (1d ndarray of 2 floats) effective range along
                                original system axes
        """
        # Prevent calculation if range or orientation is not stationary
        if not self.is_range_stationary(recompute) or not self.is_orientation_stationary(recompute):
            return None
        r123 = self.r123(recompute)
        m = np.abs(self.mrot(recompute))
        return np.maximum(r123[0] * m[:,0], r123[1] * m[:,1], r123[2] * m[:,2]) # element-wise maximum

    def func(self):
        """
        Returns the covariance model function f(h) where:
            h:      (2-dimensional array of dim n x 3, or
                        1-dimensional array of dim 3) 2D-lag(s)
            f(h):   (1-dimensional array of dim n) evaluation of the model at h
        """
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

                elif t == 'linear':
                    s = s + cov_lin(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

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
        Returns the variogram model function f(h) where:
            h:      (2-dimensional array of dim n x 3, or
                        1-dimensional array of dim 3) 2D-lag(s)
            f(h):   (1-dimensional array of dim n) evaluation of the model at h
        """
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

                elif t == 'linear':
                    s = s + d['w'] - cov_lin(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

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
        Plot system Oxyz and Ox'''y'''z''' (in a new figure).

        :param color0, color1, color2:  colors for main axes x''', y''', z'''

        :param set_3d_subplot:
                        (bool)
                            - True: a new figure is created with one axis "projection='3d'"
                            - False: the plot is done in the current figure axis assumed to be set
                                as "projection='3d'"
                                (this allows to plot in a figure with multiple axes)

        :param figsize: (tuple of 2 ints) size of the figure, not used if set_polar_subplot is False
        """
        # Prevent calculation if orientation is not stationary
        if not self.is_orientation_stationary():
            return None
        mrot = self.mrot()

        if set_3d_subplot:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(projection='3d')
        else:
            ax = plt.gca()

        # Plot system Oxzy and Ox'y'z'
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

    def plot_model3d_volume(self, plotter=None, vario=False,
        color0='red', color1='green', color2='blue',
        extent=None, ncell=(101, 101, 101), **kwargs):
        """
        Plot covariance or variogram function in 3D (using the function drawImage3D_volume
        from geone.imgplot3d (based on pyvista)).

        :param plotter: (pyvista plotter)
                            if given: add element to the plotter, a further call
                                to plotter.show() will be required to show the plot
                            if None (default): a plotter is created and the plot
                                is shown

        :param vario:   (bool)
                            - if False: plot covariance function
                            - if True:  plot variogram function

        :param color0, color1, color2:  colors for main axes x''', y''', z'''

        :param extent:  (hxmin, hxmax, hymin, hymax, hzmin, hzmax): 4 floats defining the domain of the plot.
                            None for default

        :param ncell:   (nx, ny, nz): 3 ints defining the number of the cells in the plot (nx x ny x nz)

        :param kwargs:  keyword arguments passed to the funtion drawImage3D_volume from geone.imgplot3d
                            (cmap, ...)
        """
        # Prevent calculation if covariance model is not stationary
        if not self.is_stationary():
            return None
        # Set extent if needed
        r = max(self.r123())
        hr = 1.1 * r

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
        if vario:
            gg = self.vario_func()(hh).reshape(nz, ny, nx)
        else:
            gg = self.func()(hh).reshape(nz, ny, nx)

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
        imgplt3.drawImage3D_volume(im, plotter=plotter, **kwargs)
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

    def plot_model3d_slice(self, plotter=None, vario=False,
        color0='red', color1='green', color2='blue',
        extent=None, ncell=(101, 101, 101), **kwargs):
        """
        Plot covariance or variogram function in 3D (sclices in 3D volume, using the function
        drawImage3D_slice from geone.imgplot3d (based on pyvista)).

        :param plotter: (pyvista plotter)
                            if given: add element to the plotter, a further call
                                to plotter.show() will be required to show the plot
                            if None (default): a plotter is created and the plot
                                is shown

        :param vario:   (bool)
                            - if False: plot covariance function
                            - if True:  plot variogram function

        :param color0, color1, color2:  colors for main axes x''', y''', z'''

        :param extent:  (hxmin, hxmax, hymin, hymax, hzmin, hzmax): 4 floats defining the domain of the plot.
                            None for default

        :param ncell:   (nx, ny, nz): 3 ints defining the number of the cells in the plot (nx x ny x nz)

        :param kwargs:  keyword arguments passed to the funtion drawImage3D_slice from geone.imgplot3d
                            (cmap, ...)
        """
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
        if vario:
            gg = self.vario_func()(hh).reshape(nz, ny, nx)
        else:
            gg = self.func()(hh).reshape(nz, ny, nx)

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
        imgplt3.drawImage3D_slice(im, plotter=plotter, **kwargs)
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

    def plot_model_curves(self, plotter=None, vario=False,
        color0='red', color1='green', color2='blue',
        h1min=0, h1max=None, h2min=0, h2max=None, h3min=0, h3max=None,
        n1=500, n2=500, n3=500, grid=True, show_xlabel=True, show_ylabel=True):
        """
        Plot covariance or variogram function along the main axes x''', y''', z''' (in current figure axis).

        :param vario:   (bool)
                            - if False: plot covariance function
                            - if True:  plot variogram function

        :param color0, color1, color2:  colors for curves along main axes x''', y''', z'''

        :param h1min, h1max:    function is plotted along x''' for h in interval [h1min, h1max] (default h1max if None)
        :param h2min, h2max:    function is plotted along y''' for h in interval [h2min, h2max] (default h2max if None)
        :param h3min, h3max:    function is plotted along z''' for h in interval [h3min, h3max] (default h1max if None)
        :param n1, n2, n3:      number of points in interval [h1min, h1max], [h2min, h2max] and [h3min, h3max] resp.
        :param show_xlabel, show_ylabel:
                        (bool) indicates if label for x axis (resp. y axis) is displayed (True by default)
        :param grid:    (bool) indicates if a grid is plotted (True by default)
        """
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
        if vario:
            g1 = self.vario_func()(hh1.dot(mrot.T)) # hh1.dot(mrot.T): 3D-lags in system Oxyz (what is taken by the function)
        else:
            g1 = self.func()(hh1.dot(mrot.T)) # hh1.dot(mrot.T): 3D-lags in system Oxz (what is taken by the function)

        # Evaluate function along axis y'''
        h2 = np.linspace(h2min, h2max, n2)
        hh2 = np.hstack((np.zeros((len(h2),1)), h2.reshape(-1,1), np.zeros((len(h2),1)))) # (n1,3) array) 3D-lags along y''' expressed in system Ox''y'''z''''
        if vario:
            g2 = self.vario_func()(hh2.dot(mrot.T)) # hh2.dot(mrot.T): 3D-lags in system Oxyz (what is taken by the function)
        else:
            g2 = self.func()(hh2.dot(mrot.T)) # hh2.dot(mrot.T): 3D-lags in system Oxz (what is taken by the function)

        # Evaluate function along axis z'''
        h3 = np.linspace(h3min, h3max, n3)
        hh3 = np.hstack((np.zeros((len(h3),1)), np.zeros((len(h3),1)), h3.reshape(-1,1))) # (n1,3) array) 3D-lags along z''' expressed in system Ox''y'''z''''
        if vario:
            g3 = self.vario_func()(hh3.dot(mrot.T)) # hh3.dot(mrot.T): 3D-lags in system Oxyz (what is taken by the function)
        else:
            g3 = self.func()(hh3.dot(mrot.T)) # hh3.dot(mrot.T): 3D-lags in system Oxz (what is taken by the function)

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
        if grid:
            plt.grid(True)

    def plot_model_one_curve(self, main_axis=1, vario=False, hmin=0, hmax=None, npts=500,
        grid=True, show_xlabel=True, show_ylabel=True, **kwargs):
        """
        Plot covariance or variogram curve along one main axis (in current figure axis).

        :param main_axis:   (int) 1, 2 or 3:
                                1: plot curve along x''',
                                2: plot curve along y''',
                                3: plot curve along z'''
        :param vario:   (bool)
                            - if False: plot covariance function
                            - if True:  plot variogram function
        :param hmin, hmax:  (float) function is plotted for h in interval [hmin, hmax]
                                hmax=None for default: 1.2 * range max
        :param npts:    (int) number of points used in interval [hmin, hmax]
        :param grid:    (bool) indicates if a grid is plotted (True by default)
        :param show_xlabel, show_ylabel:
                        (bool) indicates if label for x axis (resp. y axis)
                            is displayed (True by default)
        :kwargs:        keyword arguments passed to the funtion plt.plot
        """
        # Prevent calculation if covariance model is not stationary
        if not self.is_stationary():
            return None
        if main_axis not in (1, 2, 3):
            print('ERROR: main_axis not valid (should be 1, 2 or 3)')
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
        if vario:
            g = self.vario_func()(hh.dot(mrot.T)) # hh.dot(mrot.T): 3D-lags in system Oxyz (what is taken by the function)
        else:
            g = self.func()(hh.dot(mrot.T)) # hh.dot(mrot.T): 3D-lags in system Oxyz (what is taken by the function)

        plt.plot(h, g, **kwargs)

        if show_xlabel:
            plt.xlabel('h')
        if show_ylabel:
            if vario:
                plt.ylabel(r'$\gamma(h)$')
            else:
                plt.ylabel(r'$cov(h)$')

        if grid:
            plt.grid(True)
# ----------------------------------------------------------------------------

# ============================================================================
# Definition of function to convert covariance models
# ============================================================================
# ----------------------------------------------------------------------------
def covModel1D_to_covModel2D(cov_model_1d):
    """
    Converts a covariance model in 1D to a omni-directional covariance model in 2D.

    :param cov_model_1d:    (CovModel1D class) covariance model in 1D

    :return cov_model_2d:   (CovModel2D class) covariance model in 2D (omni-directional,
                                defined from cov_model_1d)
    """
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
    Converts a covariance model in 1D to a omni-directional covariance model in 3D.

    :param cov_model_1d:    (CovModel1D class) covariance model in 1D

    :return cov_model_3d:   (CovModel2D class) covariance model in 3D (omni-directional,
                                defined from cov_model_1d)
    """
    cov_model_3d = CovModel3D()
    cov_model_3d.elem = copy.deepcopy(cov_model_1d.elem)
    for el in cov_model_3d.elem:
        for k, val in el[1].items():
            if k == 'r':
                el[1]['r'] = [val, val, val]

    return cov_model_3d
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel2D_to_covModel3D(cov_model_2d, alpha=0., beta=0., gamma=0.):
    """
    Converts a covariance model in 2D to a covariance model in 3D, where
    the angles alpha, beta, gamma define the system supporting the axes of
    the model (ranges) (see CoveModel3D class), and where the ranges along
    the two first axes are set to the range along the first axis from the
    covariance model in 2D, and the range along the third axis is set to
    the range along the second axis from the covariance model in 2D.

    :param cov_model_2d:    (CovModel2D class) covariance model in 2D
                                (attribute cov_model_2d.alpha will be ignored)

    :param alpha, beta, gamma:
                    (floats) angles in degrees defining the system supporting
                    the axes of the covariance model in 3D (ranges)

    :return cov_model_3d:   (CovModel3D class) covariance model in 3D
    """
    cov_model_3d = CovModel3D()
    cov_model_3d.elem = copy.deepcopy(cov_model_2d.elem)
    cov_model_3d.alpha = alpha
    cov_model_3d.beta = beta
    cov_model_3d.gamma = gamma
    for el in cov_model_3d.elem:
        for k, val in el[1].items():
            if k == 'r':
                el[1]['r'] = [val[0], val[0], val[1]]

    return cov_model_3d
# ----------------------------------------------------------------------------

# ============================================================================
# Basic functions for plotting variogram cloud and experimental variogram (1D)
# ============================================================================
# ----------------------------------------------------------------------------
def plot_variogramCloud1D(h, g, npair, grid=True, **kwargs):
    """
    Plot a variogram cloud (1D) (in current figure axis).

    :param h, g:    (1-dimensional array of shape (npair,)) coordinates of the points
                        of the variogram cloud.
    :param npair:   (int) number of points (pairs of data points considered) in the variogram cloud.
    :param grid:    (bool) indicates if a grid is plotted (True by default)
    :kwargs:        keyword arguments passed to the funtion plt.plot
    """
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

    plt.plot(h, g, **kwargs)
    plt.xlabel('h')
    plt.ylabel(r'$1/2(Z(x)-Z(x+h))^2$')
    if grid:
        plt.grid(True)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def plot_variogramExp1D(hexp, gexp, cexp, show_count=True, grid=True, **kwargs):
    """
    Plot an experimental variogram (1D) (in current figure axis).

    :param hexp, gexp:  (1-dimensional array of floats of same length) coordinates of
                            the points of the experimental variogram
    :param cexp:        (1-dimensional array of ints of same length as hexp, gexp)
                            numbers of points from the variogram cloud in each class
    :param show_count:  (bool) indicates if counters (cexp) are shown on plot
    :param grid:        (bool) indicates if a grid is plotted (True by default)
    :kwargs:            keyword arguments passed to the funtion plt.plot
    """
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
    plt.xlabel('h')
    plt.ylabel(r'$1/2(Z(x)-Z(x+h))^2$')
    if grid:
        plt.grid(True)
# ----------------------------------------------------------------------------

# ============================================================================
# Functions for variogram cloud, experimental variogram,
# and covariance model fitting (1D)
# ============================================================================
# ----------------------------------------------------------------------------
def variogramCloud1D(x, v, hmax=np.nan, make_plot=True, grid=True, **kwargs):
    """
    Computes the omni-directional variogram cloud for data set in 1D, 2D or 3D.
        - the pair of the i-th and j-th data points gives the following
            point in the variogram cloud:
                (h(i,j), g(i,j)) = (||x(i)-x(j)||, 0.5 * (v(i)-v(j))^2)
            where x(i) and x(j) are the coordinates of the i-th and j-th data points
            and v(i) and v(j) the values at these points
            (v(i)=Z(x(i)), where Z is the considered variable).

    :param x:       (2-dimensional array of shape (n, d)) coordinates
                        of the data points (n: number of points, d: dimension)
                        Note: for data in 1D, it can be a 1-dimensional array of shape (n,)
    :param v:       (1-dimensional array of shape (n,)) values at data points

    :param hmax:    (float or nan) maximal distance between a pair of data points for
                        being integrated in the variogram cloud.

    :param make_plot:
                    (bool) if True: the plot of the variogram cloud is done (in current figure axis)

    :param grid:    (bool) indicates if a grid is plotted (used if make_plot is True)
    :kwargs:        keyword arguments passed to the function plot_variogramCloud1D (used if make_plot is True)

    :return:    (h, g, npair), where
                    h, g are two 1-dimensional arrays of floats of same length containing
                        the coordinates of the points in the variogram cloud
                    npair is an int, the number of points (pairs of data points considered)
                        in the variogram cloud
    """
    # Get dimension (d) from x
    if np.asarray(x).ndim == 1:
        # x is a 1-dimensional array
        x = np.asarray(x).reshape(-1, 1)
        d = 1
    else:
        # x is a 2-dimensional array
        d = x.shape[1]

    # Number of data points
    n = x.shape[0]

    # Check length of v
    if len(v) != n:
        print("ERROR: length of 'v' is not valid")
        return None, None, None

    if np.isnan(hmax):
        # consider all pairs of points
        npair = int(0.5*(n-1)*n)
        h = np.zeros(npair)
        g = np.zeros(npair)
        j = 0
        for i in range(n-1):
            jj = n-1-i
            h[j:(j+jj)]= np.sqrt(np.sum((x[i,:] - x[(i+1):, :])**2, axis=1))
            g[j:(j+jj)]= 0.5*(v[i] - v[(i+1):])**2
            j = j+jj

    else:
        # consider only pairs of points with a distance less than or equal to hmax
        hmax2 = hmax**2
        h, g = [], []

        npair = 0
        for i in range(n-1):
            htmp = np.sum((x[i,:] - x[(i+1):, :])**2, axis=1)
            ind = np.where(htmp <= hmax2)[0]
            h.append(np.sqrt(htmp[ind]))
            g.append(0.5*(v[i] - v[i+1+ind])**2)
            npair = npair + len(ind)

        if npair > 0:
            h = np.hstack(h)
            g = np.hstack(g)

    if make_plot:
        plot_variogramCloud1D(h, g, npair, grid=grid, **kwargs)
        plt.title('Variogram cloud ({} pts)'.format(npair))

    return (h, g, npair)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def variogramExp1D(x, v, hmax=np.nan, ncla=10, cla_center=None, cla_length=None, variogramCloud=None, make_plot=True, show_count=True, grid=True, **kwargs):
    """
    Computes the exprimental omni-directional variogram for data set in 1D, 2D or 3D.
    The mean point in each class is retrieved from the variogram cloud (returned by
    the function variogramCloud1D).

    :param x:       (2-dimensional array of shape (n, d)) coordinates
                        of the data points (n: number of points, d: dimension)
                        Note: for data in 1D, it can be a 1-dimensional array of shape (n,)
    :param v:       (1-dimensional array of shape (n,)) values at data points

    :param hmax:    (float or nan) maximal distance between a pair of data points for
                        being integrated in the variogram cloud.

    :param ncla:    (int) number of classes:
                        the parameter is used if cla_center is not specified (None),
                        in that situation ncla classes is considered and the class centers are set to
                            cla_center[i] = (i+0.5)*l, i=0,...,ncla-1
                        with l = H / ncla, H being the max of the distance between the two points of
                        the considered pairs (in the variogram cloud).

    :param cla_center:  (sequence of floats) center each class, if specified (not None),
                            then the parameter ncla is not used.

    :param cla_length:  (None, or float or sequence of floats) length of each class
                            - if not specified (None): the length of every class is set to the
                                minimum of difference between two sucessive class centers (np.inf if one class)
                            - if float: the length of every class is set to the specified number
                            - if a sequence, its length should be equal to the number of classes (length of
                                cla_center (or ncla))
                            Finally, the i-th class is determined by its center cla_center[i] and its
                            length cla_length[i], and corresponds to the interval
                                ]cla_center[i]-cla_length[i]/2, cla_center[i]+cla_length[i]/2]
                            along h (lag) axis

    :param variogramCloud:
                    (tuple of length 3) (h, g, npair): variogram cloud (returned by the function variogramCloud1D
                        (npair is not used))
                        If given (not None): this variogram cloud is used (not computed, then x, v, hmax are not used)

    :param make_plot:
                    (bool) if True: the plot of the experimental variogram is done (in current figure axis)

    :param show_count:  (bool) indicates if counters (cexp) are shown on plot (used if make_plot is True)
    :param grid:        (bool) indicates if a grid is plotted (used if make_plot is True)
    :kwargs:            keyword arguments passed to the function plot_variogramExp1D (used if make_plot is True)

    :return:    (hexp, gexp, cexp), where
                    - hexp, gexp are two 1-dimensional arrays of floats of same length containing
                        the coordinates of the points of the experimental variogram, and
                    - cexp is a 1-dimensional array of ints of same length as hexp and gexp, containing
                        the number of points from the variogram cloud in each class
    """
    # Compute variogram cloud if needed (npair won't be used)
    if variogramCloud is None:
        h, g, npair = variogramCloud1D(x, v, hmax=hmax, make_plot=False)
    else:
        h, g, npair = variogramCloud

    if npair == 0:
        print('No point in the variogram cloud (nothing is done).')
        return None, None, None

    # Set classes
    if cla_center is not None:
        cla_center = np.asarray(cla_center, dtype='float').reshape(-1)
        ncla = len(cla_center)
    else:
        length = np.max(h) / ncla
        cla_center = (np.arange(ncla, dtype='float') + 0.5) * length

    if cla_length is not None:
        cla_length = np.asarray(cla_length, dtype='float').reshape(-1)
        if len(cla_length) == 1:
            cla_length = np.repeat(cla_length, ncla)
        elif len(cla_length) != ncla:
            print("ERROR: 'cla_length' not valid")
            return None, None, None
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
        plot_variogramExp1D(hexp, gexp, cexp, show_count=show_count, grid=grid, **kwargs)
        plt.title('Experimental variogram')

    return (hexp, gexp, cexp)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel1D_fit(x, v, cov_model, hmax=np.nan, variogramCloud=None, make_plot=True, **kwargs):
    """
    Fits a covariance model in 1D, used for data in 1D or as omni-directional model
    for data in 2D or 3D.

    The parameter 'cov_model' is a covariance model in 1D (CovModel1D class) with
    the parameters to fit set to nan. For example, with
        cov_model = CovModel1D(elem=[
            ('gaussian', {'w':np.nan, 'r':np.nan}), # elementary contribution
            ('nugget', {'w':np.nan})                # elementary contribution
            ])
    it will fit the weight and range of the gaussian elementary contribution,
    and the nugget (weigth of the nugget contribution).

    :param x:       (2-dimensional array of shape (n, d)) coordinates
                        of the data points (n: number of points, d: dimension)
                        Note: for data in 1D, it can be a 1-dimensional array of shape (n,)
    :param v:       (1-dimensional array of shape (n,)) values at data points

    :param cov_model:   (CovModel1D class) covariance model in 1D with parameters to fit set to nan
                            (see above)

    :param hmax:    (float or nan) maximal distance between a pair of data points for
                        being integrated in the variogram cloud.

    :param variogramCloud:
                    (tuple of length 3) (h, g, npair): variogram cloud (returned by the function variogramCloud1D
                        (npair is not used)).
                        If given (not None): this variogram cloud is used (not computed, then x, v, hmax are not used)

    :param make_plot:
                    (bool) if True: the plot of the optimized variogram is done (in current figure axis)

    :kwargs:        keyword arguments passed to the funtion curve_fit() from scipy.optimize
                        e.g.: p0=<array of initial parameters> (see doc of curve_fit), with
                            an array of floats of length equal to the number of paramters to fit,
                            considered in the order of appearance in the definition of cov_model;
                            bounds=(<array of lower bounds>, <array of upper bounds>)

    :return:        (cov_model_opt, popt) with:
                        - cov_model_opt:    (covModel1D class) optimized covariance model
                        - popt:             (sequence of floats) vector of optimized parameters
                                                returned by curve_fit
    """
    # Check cov_model
    if not isinstance(cov_model, CovModel1D):
        print("ERROR: 'cov_model' is not a covariance model in 1D")
        return None, None
    # if cov_model.__class__.__name__ != 'CovModel1D':
    #     print("ERROR: 'cov_model' is incompatible with dimension (1D)")
    #     return None, None

    # Prevent calculation if covariance model is not stationary
    if not cov_model.is_stationary():
        print("ERROR: 'cov_model' is not stationary: fit can not be applied")
        return None, None

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
        print('No parameter to fit!')
        return (cov_model_opt, np.array([]))

    # Compute variogram cloud if needed (npair won't be used)
    if variogramCloud is None:
        h, g, npair = variogramCloud1D(x, v, hmax=hmax, make_plot=False) # npair won't be used
    else:
        h, g, npair = variogramCloud

    if npair == 0:
        print('No point to fit!')
        return (cov_model_opt, np.nan * np.ones(nparam))

    def func(d, *p):
        """
        Function whose p is the vector of parameters to optimize.
        :param d:   (array) data: x, coordinates of the data points (see above)
        :param p:   vector of parameters (floats) to optimize for the covariance model,
                        variables to fit identified with ielem_to_fit, key_to_fit, computed
                        above
        :return: variogram function of the corresponding covariance model evaluated at data d
        """
        for i, (iel, k) in enumerate(zip(ielem_to_fit, key_to_fit)):
            cov_model_opt.elem[iel][1][k] = p[i]
        return cov_model_opt.vario_func()(d)

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
            print("ERROR: length of 'p0' not compatible")
            return None, None

    # Fit with curve_fit
    popt, pcov = scipy.optimize.curve_fit(func, h, g, **kwargs)

    if make_plot:
        cov_model_opt.plot_model(vario=True, hmax=np.max(h), label='vario opt.')
        s = ['Vario opt.:'] + ['{}'.format(el) for el in cov_model_opt.elem]
        # plt.title(textwrap.TextWrapper(width=50).fill(s))
        plt.title('\n'.join(s))

    return (cov_model_opt, popt)
# ----------------------------------------------------------------------------

# ============================================================================
# Functions for variogram cloud, experimental variogram,
# and covariance model fitting (2D)
# ============================================================================
# ----------------------------------------------------------------------------
def variogramCloud2D(x, v, alpha=0.0, tol_dist=10.0, tol_angle=45.0, hmax=(np.nan, np.nan),
    make_plot=True, color0='red', color1='green', figsize=None):
    """
    Computes two directional variogram clouds for a data set in 2D:
        - one along axis x',
        - one along axis y',
    where the system Ox'y' is obtained from the (usual) system Oxy by applying a rotation of
    angle -alpha (see parameter alpha below).

    :param x:       (2-dimensional array of shape (n, 2)) 2D-coordinates in system Oxy of data points
    :param v:       (1-dimensional array of shape (n,)) values at data points

    :param alpha:   (float) angle in degrees:
                        the system Ox'y', supporting the principal axes along which the variograms
                        are computedof, is obtained from the system Oxy by applying a rotation of
                        angle -alpha.
                        The 2x2 matrix m for changing the coordinate system from
                        Ox'y' to Oxy is:
                                +                         +
                                |  cos(alpha)   sin(alpha)|
                            m = | -sin(alpha)   cos(alpha)|
                                +                         +
    :param tol_dist, tol_angle: (float) tolerances (tol_dist: distance, tol_angle: angle in degrees)
                    used to determines which pair of points are integrated in the variogram clouds.
                    A pair of points (x(i), x(j)) is in the directional variogram cloud along
                    axis x' (resp. y') iff, given the lag vector h = x(i) - x(j),
                        - the distance from the end of vector h issued from origin to that axis
                            is less than or equal to tol_dist and,
                        - the angle between h and that axis is less than or equal to tol_angle

    :param hmax:    (sequence of 2 floats (or nan)): maximal distance between a pair of data points for
                        being integrated in the directional variogram cloud along axis x' and axis y' resp.

    :param make_plot:
                    (bool) if True: the plot of the variogram clouds is done (in a new 2x2 figure)

    :param color0, color1:  colors for variogram cloud along axis x' and along axis y' resp. (used if make_plot is True)

    :param figsize: (tuple of 2 ints) size of the figure (used if make_plot is True)

    :return:    ((h0, g0, npair0), (h1, g1, npair1)), where
                    - (h0, g0, npair0) is the directional variogram cloud along the axis x'
                        (h0, g0 are two 1-dimensional arrays of same length containing
                        the coordinates of the points in the variagram cloud, and
                        npair is an int, the number of points (pairs of data points considered)
                        in the variogram cloud)
                    - (h1, g1, npair1) is the directional variogram cloud along the axis y'
                        (same type of object as for axis x')
    """
    # Number of data points
    n = x.shape[0]

    # Check length of v
    if len(v) != n:
        print("ERROR: length of 'v' is not valid")
        return ((None, None, None), (None, None, None))

    # Rotation matrix
    a = alpha * np.pi/180.
    ca, sa = np.cos(a), np.sin(a)
    mrot = np.array([[ca, sa], [-sa, ca]])

    # Coordinates of data points in the new system Ox'y'
    xnew = x.dot(mrot)

    # Tolerance for distance to origin
    hmax = list(hmax) # for assignment of its components
    for i in (0, 1):
        if np.isnan(hmax[i]):
            hmax[i] = np.inf

    # Tolerance for slope compute from tol_angle
    tol_s = np.tan(tol_angle*np.pi/180)

    eps = 1.e-8 # close to zero

    # Compute variogram clouds
    h0, g0, h1, g1 = [], [], [], []
    for i in range(n-1):
        for j in range(i+1, n):
            h = xnew[i,:] - xnew[j,:]
            habs = np.fabs(h)
            if habs[0] < eps or (habs[0] <= hmax[0] and habs[1] <= tol_dist and habs[1]/habs[0] <= tol_s):
                # Directional variogram along x' contains pair of points (i,j)
                h0.append(habs[0]) # projection along x'
                g0.append(0.5*(v[i]-v[j])**2)
            if habs[1] < eps or (habs[1] <= hmax[1] and habs[0] <= tol_dist and habs[0]/habs[1] <= tol_s):
                # Directional variogram along y' contains pair of points (i,j)
                h1.append(habs[1]) # projection along y'
                g1.append(0.5*(v[i]-v[j])**2)

    h0 = np.asarray(h0)
    g0 = np.asarray(g0)
    npair0 = len(h0)
    h1 = np.asarray(h1)
    g1 = np.asarray(g1)
    npair1 = len(h1)

    if make_plot:
        fig, ax = plt.subplots(2,2, figsize=figsize)

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
        # plt.title("Vario cloud: alpha= {} deg.\ntol_dist ={}deg. / tol_angle ={}deg.".format(alpha, tol_dist, tol_angle))

        plt.sca(ax[0,1])
        # Plot both variogram clouds
        plot_variogramCloud1D(h0, g0, npair0, c=color0, alpha=0.5, label="along x'")
        plot_variogramCloud1D(h1, g1, npair1, c=color1, alpha=0.5, label="along y'")
        plt.legend()
        #plt.title('Total #points = {}'.format(npair0 + npair1))

        plt.sca(ax[1,0])
        # Plot variogram cloud along x'
        plot_variogramCloud1D(h0, g0, npair0, c=color0)
        plt.title("along x' ({} pts)".format(npair0))

        plt.sca(ax[1,1])
        # Plot variogram cloud along y'
        plot_variogramCloud1D(h1, g1, npair1, c=color1)
        plt.title("along y' ({} pts)".format(npair1))

        plt.suptitle("Vario cloud: alpha={}deg.\ntol_dist={} / tol_angle={}deg.".format(alpha, tol_dist, tol_angle))
        # plt.show()

    return ((h0, g0, npair0), (h1, g1, npair1))
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def variogramExp2D(x, v, alpha=0.0, tol_dist=10.0, tol_angle=45.0, hmax=(np.nan, np.nan),
    ncla=(10, 10), cla_center=(None, None), cla_length=(None, None),
    variogramCloud=None, make_plot=True, color0='red', color1='green', figsize=None):
    """
    Computes two directional exprimental variograms for a data set in 2D:
        - one along axis x',
        - one along axis y',
    where the system Ox'y' is obtained from the (usual) system Oxy by applying a rotation of
    angle -alpha (see parameter alpha below).

    The mean point in each class is retrieved from the two directional variogram clouds
    (returned by the function variogramCloud2D).

    :param x:       (2-dimensional array of shape (n, 2)) 2D-coordinates in system Oxy of data points
    :param v:       (1-dimensional array of shape (n,)) values at data points

    :param alpha:   (float) angle in degrees:
                        the system Ox'y', supporting the principal axes along which the variograms
                        are computedof, is obtained from the system Oxy by applying a rotation of
                        angle -alpha.
                        The 2x2 matrix m for changing the coordinate system from
                        Ox'y' to Oxy is:
                                +                         +
                                |  cos(alpha)   sin(alpha)|
                            m = | -sin(alpha)   cos(alpha)|
                                +                         +

    :param tol_dist, tol_angle: (float) tolerances (tol_dist: distance, tol_angle: angle in degrees)
                    used to determines which pair of points are integrated in the variogram clouds.
                    A pair of points (x(i), x(j)) is in the directional variogram cloud along
                    axis x' (resp. y') iff, given the lag vector h = x(i) - x(j),
                        - the distance from the end of vector h issued from origin to that axis
                            is less than or equal to tol_dist and,
                        - the angle between h and that axis is less than or equal to tol_angle

    :param hmax:    (sequence of 2 floats (or nan)): maximal distance between a pair of data points for
                        being integrated in the directional variogram cloud along axis x' and axis y' resp.

    :param ncla:    (sequence of 2 ints) ncla[0], ncla[1]: number of classes
                        for experimental variogram along axis x' (direction 0) and axis y' (direction 1) resp.
                        For direction j:
                        the parameter ncla[j] is used if cla_center[j] is not specified (None),
                        in that situation ncla[j] classes are considered and the class centers are set to
                            cla_center[j][i] = (i+0.5)*l, i=0,...,ncla[j]-1
                        with l = H / ncla[j], H being the max of the distance between the two points of
                        the considered pairs (in the variogram cloud of direction j).

    :param cla_center:  (sequence of 2 sequences of floats) cla_center[0], clac_center[1]: center of each class
                            for experimental variogram along axis x' (direction 0) and axis y' (direction 1) resp.
                            For direction j:
                            if cla_center[j] is specified (not None), then the parameter ncla[j] is not used.

    :param cla_length:  (sequence of length 2 of: None, or float or sequence of floats) cla_length[0], clac_length[1]:
                            length of each class
                            for experimental variogram along axis x' (direction 0) and axis y' (direction 1) resp.
                            For direction j:
                                - if cla_length[j] not specified (None): the length of every class is set to the
                                    minimum of difference between two sucessive class centers (np.inf if one class)
                                - if float: the length of every class is set to the specified number
                                - if a sequence, its length should be equal to the number of classes (length of
                                    cla_center[j] (or ncla[j]))
                                Finally, the i-th class is determined by its center cla_center[j][i] and its
                                length cla_length[j][i], and corresponds to the interval
                                    ]cla_center[j][i]-cla_length[j][i]/2, cla_center[j][i]+cla_length[j][i]/2]
                                along h (lag) axis

    :param variogramCloud:
                    (sequence of 2 tuples of length 3, or None) If given: ((h0, g0, npair0), (h1, g1, npair1)):
                        variogram clouds (returned by the function variogramCloud2D (npair0, npair1 are not used))
                        along axis x' (direction 0) and axis y' (direction 1) resp., then
                        x, v, alpha, tol_dist, tol_angle, hmax are not used
                        (but alpha, tol_dist, tol_angle are used in plot if make_plot is True)

    :param make_plot:
                    (bool) if True: the plot of the experimental variograms is done (in a new 2x2 figure)

    :param color0, color1:  colors for experimental variogram along axis x' and along axis y' resp. (used if make_plot is True)

    :param figsize: (tuple of 2 ints) size of the figure (used if make_plot is True)

    :return:    ((hexp0, gexp0, cexp0), (hexp1, gexp1, cexp1)), where
                    - (hexp0, gexp0, cexp0) is the output for the experimental variogram along axis x':
                        - hexp0, gexp0: are two 1-dimensional arrays of floats of same length containing
                        the coordinates of the points of the experimental variogram along axis x', and
                        - cexp0 is a 1-dimensional array of ints of same length as hexp0 and gexp0, containing
                        the number of points from the variogram cloud in each class
                    - (hexp1, gexp1, cexp1) is the output for the experimental variogram along axis y'
    """
    # Compute variogram clouds if needed
    if variogramCloud is None:
        vc = variogramCloud2D(x, v, alpha=alpha, tol_dist=tol_dist, tol_angle=tol_angle, hmax=hmax, make_plot=False)
    else:
        vc = variogramCloud
    # -> vc[0] = (h0, g0, npair0) and vc[1] = (h1, g1, npair1)

    # Compute variogram experimental in each direction (using function variogramExp1D)
    ve = [None, None]
    for j in (0, 1):
        ve[j] = variogramExp1D(None, None, hmax=np.nan, ncla=ncla[j], cla_center=cla_center[j], cla_length=cla_length[j], variogramCloud=vc[j], make_plot=False)

    (hexp0, gexp0, cexp0), (hexp1, gexp1, cexp1) = ve

    if make_plot:
        # Rotation matrix
        a = alpha * np.pi/180.
        ca, sa = np.cos(a), np.sin(a)
        mrot = np.array([[ca, sa], [-sa, ca]])

        fig, ax = plt.subplots(2,2, figsize=figsize)
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
        plot_variogramExp1D(hexp0, gexp0, cexp0, show_count=False, c=color0, alpha=0.5, label="along x'")
        plot_variogramExp1D(hexp1, gexp1, cexp1, show_count=False, c=color1, alpha=0.5, label="along y'")
        plt.legend()

        plt.sca(ax[1,0])
        # Plot variogram exp along x'
        plot_variogramExp1D(hexp0, gexp0, cexp0, color=color0)
        plt.title("along x'")

        plt.sca(ax[1,1])
        # Plot variogram exp along y'
        plot_variogramExp1D(hexp1, gexp1, cexp1, color=color1)
        plt.title("along y'")

        plt.suptitle("Vario exp.: alpha={}deg.\ntol_dist={} / tol_angle={}deg.".format(alpha, tol_dist, tol_angle))
        # plt.show()

    return ((hexp0, gexp0, cexp0), (hexp1, gexp1, cexp1))
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def variogramExp2D_rose(x, v, r_max=np.nan, r_ncla=10, phi_ncla=12, set_polar_subplot=True, figsize=None, **kwargs):
    """
    Shows shows an experimental variogram for a data set in 2D in the form of a
    rose plot, i.e. the lags vectors between the pairs of data points are divided
    in classes according to length (radius) and angle from the x-axis counter-clockwise
    (warning: opposite sense to the sense given by angle in definition of a covariance model
    in 2D).

    :param x:       (2-dimensional array of shape (n, 2)) 2D-coordinates in system Oxy of data points
    :param v:       (1-dimensional array of shape (n,)) values at data points

    :param r_max:    (float or nan) maximal radius, i.e. maximal length of 2D-lag vector between a pair
                        of data points for being integrated in the variogram rose plot.

    :param r_ncla:      (int) number of classes for radius

    :param phi_ncla:    (int) number of classes for angle for half of the whole disk:
                            on the whole disk, there will be 2*phi_ncla classes

    :param set_polar_subplot:
                        (bool)
                            - True: a new figure is created with one axis "projection='polar'"
                            - False: the plot is done in the current figure axis assumed to be set
                                as "projection='polar'"
                                (this allows to plot in a figure with multiple axes)

    :param figsize: (tuple of 2 ints) size of the figure, not used if set_polar_subplot is False

    :kwargs:            keyword arguments passed to the funtion plt.pcolormesh
                            (cmap, ...)
    """
    # Number of data points
    n = x.shape[0]

    # Check length of v
    if len(v) != n:
        print("ERROR: length of 'v' is not valid")
        return

    # Compute lag vector (h) and gamma value (g) for pair of points with distance less than or equal to hmax
    if np.isnan(r_max):
        # consider all pairs of points
        npair = int(0.5*(n-1)*n)
        h = np.zeros((npair, 2))
        g = np.zeros(npair)
        j = 0
        for i in range(n-1):
            jj = n-1-i
            h[j:(j+jj),:]= x[(i+1):, :] - x[i,:]
            g[j:(j+jj)]= 0.5*(v[i] - v[(i+1):])**2
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
    if np.isnan(r_max):
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
def covModel2D_fit(x, v, cov_model, hmax=np.nan, make_plot=True, figsize=None, **kwargs):
    """
    Fits a covariance model in 2D (for data in 2D).

    The parameter 'cov_model' is a covariance model in 2D (CovModel2D class) with
    the parameters to fit set to nan (a nan replace a float). For example, with
        cov_model = CovModel2D(elem=[
            ('gaussian', {'w':np.nan, 'r':[np.nan, np.nan]}), # elementary contribution
            ('nugget', {'w':np.nan})                          # elementary contribution
            ], alpha=np.nan, name='')
    it will fit the weight and ranges of the gaussian elementary contribution,
    the nugget (weigth of the nugget contribution), and the angle alpha.

    :param x:       (2-dimensional array of shape (n, 2)) coordinates
                        of the data points (n: number of points)
    :param v:       (1-dimensional array of shape (n,)) values at data points

    :param cov_model:   (CovModel2D class) covariance model in 2D with parameters to fit set to nan
                            (see above)

    :param hmax:    (float or nan) maximal distance between a pair of data points for
                        being integrated in the variogram cloud.

    :param make_plot:
                    (bool) if True: the plot of the optimized variogram is done (in a new 1x2 figure)

    :param figsize: (tuple of 2 ints) size of the figure (used if make_plot is True)

    :kwargs:        keyword arguments passed to the funtion curve_fit() from scipy.optimize
                        e.g.: p0=<array of initial parameters> (see doc of curve_fit), with
                            an array of floats of length equal to the number of paramters to fit,
                            considered in the order of appearance in the definition of cov_model;
                            bounds=(<array of lower bounds>, <array of upper bounds>)

    :return:        (cov_model_opt, popt) with:
                        - cov_model_opt:    (covModel2D class) optimized covariance model
                        - popt:             (sequence of floats) vector of optimized parameters
                                                returned by curve_fit
    """
    # Check cov_model
    if not isinstance(cov_model, CovModel2D):
        print("ERROR: 'cov_model' is not a covariance model in 2D")
        return None, None
    # if cov_model.__class__.__name__ != 'CovModel2D':
    #     print("ERROR: 'cov_model' is incompatible with dimension (2D)")
    #     return None, None

    # Prevent calculation if covariance model is not stationary
    if not cov_model.is_stationary():
        print("ERROR: 'cov_model' is not stationary: fit can not be applied")
        return None, None

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
        print('No parameter to fit!')
        return (cov_model_opt, np.array([]))

    # Compute lag vector (h) and gamma value (g) for pair of points with distance less than or equal to hmax
    n = x.shape[0] # number of points
    if np.isnan(hmax):
        # consider all pairs of points
        npair = int(0.5*(n-1)*n)
        h = np.zeros((npair, 2))
        g = np.zeros(npair)
        j = 0
        for i in range(n-1):
            jj = n-1-i
            h[j:(j+jj),:]= x[(i+1):, :] - x[i,:]
            g[j:(j+jj)]= 0.5*(v[i] - v[(i+1):])**2
            j = j+jj

    else:
        # consider only pairs of points with a distance less than or equal to hmax
        hmax2 = hmax**2
        h, g = [], []

        npair = 0
        for i in range(n-1):
            htmp = x[(i+1):, :] - x[i,:] # 2-dimensional array (n-1-i) x dim
            ind = np.where(np.sum(htmp**2, axis=1) <= hmax2)[0]
            h.append(htmp[ind])
            g.append(0.5*(v[i] - v[i+1+ind])**2)
            npair = npair + len(ind)

        if npair > 0:
            h = np.vstack(h)
            g = np.hstack(g)

    if npair == 0:
        print('No point to fit!')
        return (cov_model_opt, np.nan * np.ones(nparam))

    # Defines the function to optimize in a format compatible with curve_fit from scipy.optimize
    def func(d, *p):
        """
        Function whose p is the vector of parameters to optimize.
        :param d:   (array) data: h, lag vector for pair of data points (see above)
        :param p:   vector of parameters (floats) to optimize for the covariance model,
                        variables to fit identified with ielem_to_fit, key_to_fit, ir_to_fit
                        and alpha_to_fitm, computed above
        :return: variogram function of the corresponding covariance model evaluated at data d
        """
        for i, (iel, k, j) in enumerate(zip(ielem_to_fit, key_to_fit, ir_to_fit)):
            if k == 'r':
                cov_model_opt.elem[iel][1]['r'][j] = p[i]
            else:
                cov_model_opt.elem[iel][1][k] = p[i]
        if alpha_to_fit:
            cov_model_opt.alpha = p[-1]
            cov_model_opt._mrot = None # reset attribute _mrot !
        return cov_model_opt.vario_func()(d)

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
            print("ERROR: length of 'p0' not compatible")
            return None, None

    # Fit with curve_fit
    popt, pcov = scipy.optimize.curve_fit(func, h, g, **kwargs)

    if make_plot:
        cov_model_opt.plot_model(vario=True, figsize=figsize)
        # suptitle already in function cov_model_opt.plot_model...
        # s = ['Vario opt.: alpha={}'.format(cov_model_opt.alpha)] + ['{}'.format(el) for el in cov_model_opt.elem]
        # # plt.suptitle(textwrap.TextWrapper(width=50).fill(s))
        # plt.suptitle('\n'.join(s))

    return (cov_model_opt, popt)
# ----------------------------------------------------------------------------

# ============================================================================
# Functions for variogram cloud, experimental variogram,
# and covariance model fitting (3D)
# ============================================================================
# ----------------------------------------------------------------------------
def variogramCloud3D(x, v, alpha=0.0, beta=0.0, gamma=0.0, tol_dist=10.0, tol_angle=45.0, hmax=(np.nan, np.nan, np.nan),
    make_plot=True, color0='red', color1='green', color2='blue', figsize=None):
    """
    Computes three directional variogram clouds for a data set in 3D:
        - one along axis x''',
        - one along axis y''',
        - one along axis z''',
    where the system Ox'''y'''z''' is obtained from the (usual) system Oxyz as follows:
        Oxyz      -- rotation of angle -alpha around Oz  --> Ox'y'z'
        Ox'y'z'   -- rotation of angle -beta  around Ox' --> Ox''y''z''
        Ox''y''z''-- rotation of angle -gamma around Oy''--> Ox'''y'''z'''

    :param x:       (2-dimensional array of shape (n, 3)) 3D-coordinates in system Oxyz of data points
    :param v:       (1-dimensional array of shape (n,)) values at data points

    :param alpha, beta, gamma:
                    (floats) angle in degrees:
                        the system Ox'''y''''z''', supporting the axis of each variogram cloud,
                        is obtained from the system Oxyz as follows:
                        Oxyz      -- rotation of angle -alpha around Oz  --> Ox'y'z'
                        Ox'y'z'   -- rotation of angle -beta  around Ox' --> Ox''y''z''
                        Ox''y''z''-- rotation of angle -gamma around Oy''--> Ox'''y'''z'''
                        The 3x3 matrix m for changing the coordinate system from
                        Ox'''y'''z''' to Oxy is:
                            +                                                             +
                            |  ca * cc + sa * sb * sc,  sa * cb,  - ca * sc + sa * sb * cc|
                        m = |- sa * cc + ca * sb * sc,  ca * cb,    sa * sc + ca * sb * cc|
                            |                 cb * sc,     - sb,                   cb * cc|
                            +                                                             +
                        where
                            ca = cos(alpha), cb = cos(beta), cc = cos(gamma),
                            sa = sin(alpha), sb = sin(beta), sc = sin(gamma)

    :param tol_dist, tol_angle: (float) tolerances (tol_dist: distance, tol_angle: angle in degrees)
                    used to determines which pair of points are integrated in the variogram clouds.
                    A pair of points (x(i), x(j)) is in the directional variogram cloud along
                    axis x''' (resp. y''' and z''') iff, given the lag vector h = x(i) - x(j),
                        - the distance from the end of vector h issued from origin to that axis
                            is less than or equal to tol_dist and,
                        - the angle between h and that axis is less than or equal to tol_angle

    :param hmax:    (sequence of 3 floats (or nan)): maximal distance between a pair of data points for
                        being integrated in the directional variogram cloud along axis x''', axis y'''
                        and axis z''' resp.

    :param make_plot:
                    (bool) if True: the plot of the variogram clouds is done (in a new 2x2 figure)

    :param color0, color1, color2:
                    colors for variogram cloud along axis x''', along axis y''', and along axis z''' resp.
                        (used if make_plot is True)

    :param figsize: (tuple of 2 ints) size of the figure (used if make_plot is True)

    :return:    ((h0, g0, npair0), (h1, g1, npair1), (h2, g2, npair2)), where
                    - (h0, g0, npair0) is the directional variogram cloud along the axis x'''
                        (h0, g0 are two 1-dimensional arrays of same length containing
                        the coordinates of the points in the variagram cloud, and
                        npair is an int, the number of points (pairs of data points considered)
                        in the variogram cloud)
                    - (h1, g1, npair1) is the directional variogram cloud along the axis y'''
                        (same type of object as for axis x''')
                    - (h2, g2, npair2) is the directional variogram cloud along the axis z'''
                        (same type of object as for axis x''')
    """
    # Number of data points
    n = x.shape[0]

    # Check length of v
    if len(v) != n:
        print("ERROR: length of 'v' is not valid")
        return ((None, None, None), (None, None, None), (None, None, None))

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

    # Coordinates of data points in the new system Ox'y'
    xnew = x.dot(mrot)

    # Tolerance for distance to origin
    hmax = list(hmax) # for assignment of its components
    for i in (0, 1, 2):
        if np.isnan(hmax[i]):
            hmax[i] = np.inf

    # Tolerance for slope compute from tol_angle
    tol_s = np.tan(tol_angle*np.pi/180)

    eps = 1.e-8 # close to zero

    # Compute variogram clouds
    h0, g0, h1, g1, h2, g2 = [], [], [], [], [], []
    for i in range(n-1):
        for j in range(i+1, n):
            h = xnew[i,:] - xnew[j,:]
            habs = np.fabs(h)
            # di: distance to axe i (in new system)
            d0 = np.sqrt((h[1]**2 + h[2]**2))
            d1 = np.sqrt((h[0]**2 + h[2]**2))
            d2 = np.sqrt((h[0]**2 + h[1]**2))
            if habs[0] < eps or (habs[0] <= hmax[0] and d0 <= tol_dist and d0/habs[0] <= tol_s):
                # Directional variogram along x''' contains pair of points (i,j)
                h0.append(habs[0]) # projection along x'''
                g0.append(0.5*(v[i]-v[j])**2)
            if habs[1] < eps or (habs[1] <= hmax[1] and d1 <= tol_dist and d1/habs[1] <= tol_s):
                # Directional variogram along y''' contains pair of points (i,j)
                h1.append(habs[1]) # projection along y'''
                g1.append(0.5*(v[i]-v[j])**2)
            if habs[2] < eps or (habs[2] <= hmax[2] and d2 <= tol_dist and d2/habs[2] <= tol_s):
                # Directional variogram along z''' contains pair of points (i,j)
                h2.append(habs[2]) # projection along z'''
                g2.append(0.5*(v[i]-v[j])**2)

    h0 = np.asarray(h0)
    g0 = np.asarray(g0)
    npair0 = len(h0)
    h1 = np.asarray(h1)
    g1 = np.asarray(g1)
    npair1 = len(h1)
    h2 = np.asarray(h2)
    g2 = np.asarray(g2)
    npair2 = len(h2)

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
        plot_variogramCloud1D(h0, g0, npair0, c=color0)
        plt.title("along x''' ({} pts)".format(npair0))

        plt.sca(ax3)
        # Plot variogram cloud along y'''
        plot_variogramCloud1D(h1, g1, npair1, c=color1)
        plt.title("along y''' ({} pts)".format(npair1))

        plt.sca(ax4)
        # Plot variogram cloud along z'''
        plot_variogramCloud1D(h2, g2, npair2, c=color2)
        plt.title("along z''' ({} pts)".format(npair2))

        plt.suptitle("Vario cloud: alpha={}deg. beta={}deg. gamma={}deg.\ntol_dist={} / tol_angle={}deg.".format(alpha, beta, gamma, tol_dist, tol_angle))
        # plt.show()

    return ((h0, g0, npair0), (h1, g1, npair1), (h2, g2, npair2))
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def variogramExp3D(x, v, alpha=0.0, beta=0.0, gamma=0.0, tol_dist=10.0, tol_angle=45.0, hmax=(np.nan, np.nan, np.nan),
    ncla=(10, 10, 10), cla_center=(None, None, None), cla_length=(None, None, None),
    variogramCloud=None, make_plot=True, color0='red', color1='green', color2='blue', figsize=None):
    """
    Computes three directional experimental variograms for a data set in 3D:
        - one along axis x''',
        - one along axis y''',
        - one along axis z''',
    where the system Ox'''y'''z''' is obtained from the (usual) system Oxyz as follows:
        Oxyz      -- rotation of angle -alpha around Oz  --> Ox'y'z'
        Ox'y'z'   -- rotation of angle -beta  around Ox' --> Ox''y''z''
        Ox''y''z''-- rotation of angle -gamma around Oy''--> Ox'''y'''z'''

    The mean point in each class is retrieved from the three directional variogram clouds
    (returned by the function variogramCloud3D).

    :param x:       (2-dimensional array of shape (n, 3)) 3D-coordinates in system Oxyz of data points
    :param v:       (1-dimensional array of shape (n,)) values at data points

    :param alpha, beta, gamma:
                    (floats) angle in degrees:
                        the system Ox'''y''''z''', supporting the axis of each variogram cloud,
                        is obtained from the system Oxyz as follows:
                        Oxyz      -- rotation of angle -alpha around Oz  --> Ox'y'z'
                        Ox'y'z'   -- rotation of angle -beta  around Ox' --> Ox''y''z''
                        Ox''y''z''-- rotation of angle -gamma around Oy''--> Ox'''y'''z'''
                        The 3x3 matrix m for changing the coordinate system from
                        Ox'''y'''z''' to Oxy is:
                            +                                                             +
                            |  ca * cc + sa * sb * sc,  sa * cb,  - ca * sc + sa * sb * cc|
                        m = |- sa * cc + ca * sb * sc,  ca * cb,    sa * sc + ca * sb * cc|
                            |                 cb * sc,     - sb,                   cb * cc|
                            +                                                             +
                        where
                            ca = cos(alpha), cb = cos(beta), cc = cos(gamma),
                            sa = sin(alpha), sb = sin(beta), sc = sin(gamma)

    :param tol_dist, tol_angle: (float) tolerances (tol_dist: distance, tol_angle: angle in degrees)
                    used to determines which pair of points are integrated in the variogram clouds.
                    A pair of points (x(i), x(j)) is in the directional variogram cloud along
                    axis x''' (resp. y''' and z''') iff, given the lag vector h = x(i) - x(j),
                        - the distance from the end of vector h issued from origin to that axis
                            is less than or equal to tol_dist and,
                        - the angle between h and that axis is less than or equal to tol_angle

    :param hmax:    (sequence of 3 floats (or nan)): maximal distance between a pair of data points for
                        being integrated in the directional variogram cloud along axis x''', axis y'''
                        and axis z''' resp.

    :param ncla:    (sequence of 3 ints) ncla[0], ncla[1], ncla[1]: number of classes
                        for experimental variogram along axis x''' (direction 0), axis y''' (direction 1)
                        and axis z''' (direction 2) resp.
                        For direction j:
                        the parameter ncla[j] is used if cla_center[j] is not specified (None),
                        in that situation ncla[j] classes are considered and the class centers are set to
                            cla_center[j][i] = (i+0.5)*l, i=0,...,ncla[j]-1
                        with l = H / ncla[j], H being the max of the distance between the two points of
                        the considered pairs (in the variogram cloud of direction j).

    :param cla_center:  (sequence of 3 sequences of floats) cla_center[0], clac_center[1], clac_center[2]: center
                            of each class for experimental variogram along axis x''' (direction 0), axis y''' (direction 1)
                            and axis z''' (direction 2) resp.
                            For direction j:
                            if cla_center[j] is specified (not None), then the parameter ncla[j] is not used.

    :param cla_length:  (sequence of length 2 of: None, or float or sequence of floats) cla_length[0], clac_length[1]:
                            length of each class
                            for experimental variogram along axis x''' (direction 0), axis y''' (direction 1)
                            and axis z''' (direction 2) resp.
                            For direction j:
                                - if cla_length[j] not specified (None): the length of every class is set to the
                                    minimum of difference between two sucessive class centers (np.inf if one class)
                                - if float: the length of every class is set to the specified number
                                - if a sequence, its length should be equal to the number of classes (length of
                                    cla_center[j] (or ncla[j]))
                                Finally, the i-th class is determined by its center cla_center[j][i] and its
                                length cla_length[j][i], and corresponds to the interval
                                    ]cla_center[j][i]-cla_length[j][i]/2, cla_center[j][i]+cla_length[j][i]/2]
                                along h (lag) axis

    :param variogramCloud:
                    (sequence of 3 tuples of length 3, or None) If given: ((h0, g0, npair0), (h1, g1, npair1), (h2, g2, npair2)):
                        variogram clouds (returned by the function variogramCloud3D (npair0, npair1, npair2 are not used))
                        along axis axis x''' (direction 0), axis y''' (direction 1) and axis z''' (direction 2) resp., then
                        x, v, alpha, beta, gamma, tol_dist, tol_angle, hmax are not used
                        (but alpha, beta, gamma, tol_dist, tol_angle are used in plot if make_plot is True)

    :param make_plot:
                    (bool) if True: the plot of the experimental variograms is done (in a new 2x3 figure)

    :param color0, color1, color2:
                    colors for experimental variogram along axis x''', along axis y''', and along axis z''' resp.
                        (used if make_plot is True)

    :param figsize: (tuple of 2 ints) size of the figure (used if make_plot is True)

    :return:    ((hexp0, gexp0, cexp0), (hexp1, gexp1, cexp1), (hexp2, gexp2, cexp2)), where
                    - (hexp0, gexp0, cexp0) is the output for the experimental variogram along axis x''':
                        - hexp0, gexp0: are two 1-dimensional arrays of floats of same length containing
                        the coordinates of the points of the experimental variogram along axis x''', and
                        - cexp0 is a 1-dimensional array of ints of same length as hexp0 and gexp0, containing
                        the number of points from the variogram cloud in each class
                    - (hexp1, gexp1, cexp1) is the output for the experimental variogram along axis y'''
                    - (hexp2, gexp2, cexp2) is the output for the experimental variogram along axis z'''
    """
    # Compute variogram clouds if needed
    if variogramCloud is None:
        vc = variogramCloud3D(x, v, alpha=alpha, beta=beta, gamma=gamma, tol_dist=tol_dist, tol_angle=tol_angle, hmax=hmax, make_plot=False)
    else:
        vc = variogramCloud
    # -> vc[0] = (h0, g0, npair0) and vc[1] = (h1, g1, npair1) and vc[2] = (h2, g2, npair2)

    # Compute variogram experimental in each direction (using function variogramExp1D)
    ve = [None, None, None]
    for j in (0, 1, 2):
        ve[j] = variogramExp1D(None, None, hmax=np.nan, ncla=ncla[j], cla_center=cla_center[j], cla_length=cla_length[j], variogramCloud=vc[j], make_plot=False)

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
                         [                 cb * sc,     - sb,                  cb * cc ]])

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
        plot_variogramExp1D(hexp0, gexp0, cexp0, show_count=False, c=color0, alpha=0.5, label="along x'''")
        plot_variogramExp1D(hexp1, gexp1, cexp1, show_count=False, c=color1, alpha=0.5, label="along y'''")
        plot_variogramExp1D(hexp2, gexp2, cexp2, show_count=False, c=color2, alpha=0.5, label="along z'''")
        plt.legend()

        plt.sca(ax3)
        # Plot variogram exp along x'''
        plot_variogramExp1D(hexp0, gexp0, cexp0, c=color0)
        plt.title("along x'''")

        plt.sca(ax4)
        # Plot variogram exp along y'''
        plot_variogramExp1D(hexp1, gexp1, cexp1, c=color1)
        plt.title("along y'''")

        plt.sca(ax5)
        # Plot variogram exp along z'''
        plot_variogramExp1D(hexp2, gexp2, cexp2, c=color2)
        plt.title("along z'''")

        plt.suptitle("Vario exp.: alpha={}deg. beta={}deg. gamma={}deg.\ntol_dist={} / tol_angle={}deg.".format(alpha, beta, gamma, tol_dist, tol_angle))
        # plt.show()

    return ((hexp0, gexp0, cexp0), (hexp1, gexp1, cexp1), (hexp2, gexp2, cexp2))
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel3D_fit(x, v, cov_model, hmax=np.nan, make_plot=True, **kwargs):
    """
    Fits a covariance model in 3D (for data in 3D).

    The parameter 'cov_model' is a covariance model in 3D (CovModel3D class) with
    the parameters to fit set to nan (a nan replace a float). For example, with
        cov_model = CovModel3D(elem=[
            ('gaussian', {'w':np.nan, 'r':[np.nan, np.nan, np.nan]}), # elementary contribution
            ('nugget', {'w':np.nan})                                  # elementary contribution
            ], alpha=np.nan, beta=np.nan, gamma=np.nan, name='')
    it will fit the weight and ranges of the gaussian elementary contribution,
    the nugget (weigth of the nugget contribution), and the angles alpha, beta, gamma.

    :param x:       (2-dimensional array of shape (n, 3)) 3D-coordinates
                        of the data points (n: number of points)
    :param v:       (1-dimensional array of shape (n,)) values at data points

    :param cov_model:   (CovModel3D class) covariance model in 3D with parameters to fit set to nan
                            (see above)

    :param hmax:    (float or nan) maximal distance between a pair of data points for
                        being integrated in the variogram cloud.

    :param make_plot:
                    (bool) if True: the plot of the optimized variogram is done (in a new 1x2 figure)

    :kwargs:        keyword arguments passed to the funtion curve_fit() from scipy.optimize
                        e.g.: p0=<array of initial parameters> (see doc of curve_fit), with
                            an array of floats of length equal to the number of paramters to fit,
                            considered in the order of appearance in the definition of cov_model;
                            bounds=(<array of lower bounds>, <array of upper bounds>)

    :return:        (cov_model_opt, popt) with:
                        - cov_model_opt:    (covModel3D class) optimized covariance model
                        - popt:             (sequence of floats) vector of optimized parameters
                                                returned by curve_fit
    """
    # Check cov_model
    if not isinstance(cov_model, CovModel3D):
        print("ERROR: 'cov_model' is not a covariance model in 3D")
        return None, None
    # if cov_model.__class__.__name__ != 'CovModel3D':
    #     print("ERROR: 'cov_model' is incompatible with dimension (3D)")
    #     return None, None

    # Prevent calculation if covariance model is not stationary
    if not cov_model.is_stationary():
        print("ERROR: 'cov_model' is not stationary: fit can not be applied")
        return None, None

    # Work on a (deep) copy of cov_model
    cov_model_opt = copy.deepcopy(cov_model)

    # Get index of element, key of parameters and index of range to fit
    ielem_to_fit=[]
    key_to_fit=[]
    ir_to_fit=[] # if key is equal to 'r' (range), set the index of the range to fit, otherwise set np.nan
    for i, el in enumerate(cov_model_opt.elem):
        for k, val in el[1].items():
            if k == 'r':
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
        print('No parameter to fit!')
        return (cov_model_opt, np.array([]))

    # Compute lag vector (h) and gamma value (g) for pair of points with distance less than or equal to hmax
    n = x.shape[0] # number of points
    if np.isnan(hmax):
        # consider all pairs of points
        npair = int(0.5*(n-1)*n)
        h = np.zeros((npair, 3))
        g = np.zeros(npair)
        j = 0
        for i in range(n-1):
            jj = n-1-i
            h[j:(j+jj),:]= x[(i+1):, :] - x[i,:]
            g[j:(j+jj)]= 0.5*(v[i] - v[(i+1):])**2
            j = j+jj

    else:
        # consider only pairs of points with a distance less than or equal to hmax
        hmax2 = hmax**2
        h, g = [], []

        npair = 0
        for i in range(n-1):
            htmp = x[(i+1):, :] - x[i,:] # 2-dimensional array (n-1-i) x dim
            ind = np.where(np.sum(htmp**2, axis=1) <= hmax2)[0]
            h.append(htmp[ind])
            g.append(0.5*(v[i] - v[i+1+ind])**2)
            npair = npair + len(ind)

        if npair > 0:
            h = np.vstack(h)
            g = np.hstack(g)

    if npair == 0:
        print('No point to fit!')
        return (cov_model_opt, np.nan * np.ones(nparam))

    # Defines the function to optimize in a format compatible with curve_fit from scipy.optimize
    def func(d, *p):
        """
        Function whose p is the vector of parameters to optimize.
        :param d:   (array) data: h, lag vector for pair of data points (see above)
        :param p:   vector of parameters (floats) to optimize for the covariance model,
                        variables to fit identified with ielem_to_fit, key_to_fit, ir_to_fit
                        and alpha_to_fit, beta_to_fit, gamma_to_fit, computed above
        :return: variogram function of the corresponding covariance model evaluated at data d
        """
        for i, (iel, k, j) in enumerate(zip(ielem_to_fit, key_to_fit, ir_to_fit)):
            if k == 'r':
                cov_model_opt.elem[iel][1]['r'][j] = p[i]
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
        return cov_model_opt.vario_func()(d)

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
            print("ERROR: length of 'p0' not compatible")
            return None, None

    # Fit with curve_fit
    popt, pcov = scipy.optimize.curve_fit(func, h, g, **kwargs)

    if make_plot:
        # plt.suptitle(textwrap.TextWrapper(width=50).fill(s))
        s = ['Vario opt.: alpha={}, beta={}, gamma={}'.format(cov_model_opt.alpha, cov_model_opt.beta, cov_model_opt.gamma)] + ['{}'.format(el) for el in cov_model_opt.elem]
        cov_model_opt.plot_model3d_volume(vario=True, text='\n'.join(s), text_kwargs={'font_size':12})
    return (cov_model_opt, popt)
# ----------------------------------------------------------------------------

# ============================================================================
# Simple and ordinary kriging and cross validation by leave-one-out (loo)
# ============================================================================
# ----------------------------------------------------------------------------
def krige(x, v, xu, cov_model, method='simple_kriging', mean=None):
    """
    Performs kriging - interpolates at locations xu the values v measured at locations x.
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

    :param cov_model:
                    covariance model:
                        - in same dimension as dimension of points (d), i.e.:
                            - CovModel1D class if data in 1D (d=1)
                            - CovModel2D class if data in 2D (d=2)
                            - CovModel3D class if data in 3D (d=3)
                        - or CovModel1D whatever dimension of points (d):
                            - used as an omni-directional covariance model

    :param method:  (string) indicates the method used:
                        - 'simple_kriging': interpolation by simple kriging
                        - 'ordinary_kriging': interpolation by ordinary kriging

    :param mean:    (None or float or ndarray) mean of the simulation
                        (for simple kriging only):
                            - None   : mean of hard data values (stationary),
                                       i.e. mean of v
                            - float  : for stationary mean (set manually)
                            - ndarray: of of shape (n + nu,) for non stationary mean,
                                mean at point x and xu
                        For ordinary kriging (method='ordinary_kriging'),
                        this parameter must be set to None

    :return:        (vu, vu_std) with:
                        vu:     (1-dimensional array of shape (nu,)) kriged values (estimates) at points xu
                        vu_std: (1-dimensional array of shape (nu,)) kriged standard deviation at points xu
    """
    # Prevent calculation if covariance model is not stationary
    if not cov_model.is_stationary():
        print("ERROR: 'cov_model' is not stationary: krige can not be applied")
        return None, None

    # Get dimension (d) from x
    if np.asarray(x).ndim == 1:
        # x is a 1-dimensional array
        x = np.asarray(x).reshape(-1, 1)
        d = 1
    else:
        # x is a 2-dimensional array
        d = x.shape[1]

    # Get dimension (du) from xu
    if np.asarray(xu).ndim == 1:
        # xu is a 1-dimensional array
        xu = np.asarray(xu).reshape(-1, 1)
        du = 1
    else:
        # xu is a 2-dimensional array
        du = xu.shape[1]

    # Check dimension of x and xu
    if d != du:
        print("ERROR: 'x' and 'xu' do not have same dimension")
        return None, None

    # Check dimension of cov_model and set if used as omni-directional model
    if cov_model.__class__.__name__ != 'CovModel{}D'.format(d):
        if isinstance(cov_model, CovModel1D):
            omni_dir = True
        else:
            print("ERROR: 'cov_model' is incompatible with dimension of points")
            return None, None
    else:
        omni_dir = False

    # Number of data points
    n = x.shape[0]
    # Number of unknown points
    nu = xu.shape[0]

    # Check size of v
    v = np.asarray(v).reshape(-1)
    if v.size != n:
        print("ERROR: size of 'v' is not valid")
        return None, None

    # Method
    ordinary_kriging = False
    if method == 'simple_kriging':
        if mean is None:
            mean = np.mean(v) * np.ones(n + nu)
        else:
            mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if mean.size == 1:
                mean = mean * np.ones(n + nu)
            elif mean.size != n + nu:
                print("ERROR: size of 'mean' is not valid")
                return None, None
            # if mean.size not in (1, n + nu):
            #     print("ERROR: size of 'mean' is not valid")
            #     return None, None

        nmat = n # order of the kriging matrix
    elif method == 'ordinary_kriging':
        if mean is not None:
            print("ERROR: 'mean' must be None with 'method' set to 'ordinary_kriging'")
            return None, None
        ordinary_kriging = True
        nmat = n + 1 # order of the kriging matrix
    else:
        print("ERROR: 'method' is not valid")
        return None, None

    # Covariance function
    cov_func = cov_model.func() # covariance function
    if omni_dir:
        # covariance model in 1D is used
        cov0 = cov_func(0.) # covariance function at origin (lag=0)
    else:
        cov0 = cov_func(np.zeros(d)) # covariance function at origin (lag=0)

    # Set
    #   - kriging matrix (mat) of order nmat
    #   - right hand side of the kriging system (b),
    #       matrix of dimension nmat x nu
    mat = np.ones((nmat, nmat))
    for i in range(n-1):
        # lag between x[i] and x[j], j=i+1, ..., n-1
        h = x[(i+1):] - x[i]
        if omni_dir:
            # compute norm of lag
            h = np.sqrt(np.sum(h**2, axis=1))
        cov_h = cov_func(h)
        mat[i, (i+1):n] = cov_h
        mat[(i+1):n, i] = cov_h
        mat[i, i] = cov0

    b = np.ones((nmat, nu))
    for i in range(n):
        # lag between x[i] and every xu
        h = xu - x[i]
        if omni_dir:
            # compute norm of lag
            h = np.sqrt(np.sum(h**2, axis=1))
        b[i,:] = cov_func(h)

    if ordinary_kriging:
        mat[-2,-2] = cov0
        mat[-1,-1] = 0.0
    else:
        mat[-1,-1] = cov0

    # Solve the kriging system
    w = np.linalg.solve(mat, b) # w: matrix of dimension nmat x nu

    # Kriged values at unknown points
    if mean is not None:
        vu = mean[n:] + (v-mean[:n]).dot(w[:n,:])
    else:
        vu = v.dot(w[:n,:])

    # Kriged standard deviation at unknown points
    vu_std = np.sqrt(np.maximum(0, cov0 - np.array([np.dot(w[:,i], b[:,i]) for i in range(nu)])))

    return (vu, vu_std)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def cross_valid_loo(x, v, cov_model, confidence=0.05, interpolator=krige, interpolator_kwargs={}, make_plot=True, figsize=None):
    """
    Cross-validation of covariance model by leave-one-out error based on given interpolator.

    Covariance model given should be:
        - in same dimension as dimension of locations x
        - in 1D, it is then used as an omni-directional covariance model
    Interpolator should be a function as the function 'krige' (default),
    with specified keyword arguments.
    Two statisic tests are performed:
        (1) normal law test for mean of normalized error:
            Mean of normalized error times the square root of n-1
            should follow approximately a law N(0,1) (CLT)
        (2) Chi2 test for sum of squares of normalized error:
            Sum of square of normalized error should follow a law
            Chi2 with n-1 degrees of freedom,
    n being the number of data points.
    The statistc test passes with success if the obtained value is within
    the central interval covering the 1-confidence  part of the corresponding
    distribution (by default: confidence is set to 5%), otherwise the test fails.

    :param x:       (2-dimensional array of shape (n, d)) coordinates
                        of the data points (n: number of points, d: dimension)
                        Note: for data in 1D, it can be a 1-dimensional array of shape (n,)
    :param v:       (1-dimensional array of shape (n,)) values at data points

    :param cov_model:   covariance model:
                            - in same dimension as dimension of points (d), i.e.:
                                - CovModel1D class if data in 1D (d=1)
                                - CovModel2D class if data in 2D (d=2)
                                - CovModel3D class if data in 3D (d=3)
                            - or CovModel1D whatever dimension of points (d):
                                - used as an omni-directional covariance model

    :param confidence:  (float) in [0,1] for setting limit in the two statistic tests
                            (see above)

    :param interpolator:
                    (function) function used for interpolation, (default: krige)

    :interpolator_kwargs:
                    (dict) keyword argument passed to interpolator; e.g. with the function
                    krige as interpolator:
                        interpolator_kwargs={'method':'ordinary_kriging'},
                        interpolator_kwargs={'method':'simple_kriging', 'mean':<value>}

    :param make_plot:
                    (bool) if True: a plot is done (in a new 1x2 figure)

    :param figsize: (tuple of 2 ints) size of the figure (used if make_plot is True)

    :return:    (v_est, v_std, test_normal, test_chi2), tuple of length 4:
                    v_est: (1-dimensional array of shape (n,)) estimated values at data points
                    v_std: (1-dimensional array of shape (n,)) standard deviation values at data points
                    test_normal:    (bool) result of test (1) (normal law), True if success, False otherwise
                    test_chi2:      (bool) result of test (1) (chi2), True if success, False otherwise
    """
    # Prevent calculation if covariance model is not stationary
    if not cov_model.is_stationary():
        print("ERROR: 'cov_model' is not stationary: cross validation can not be applied")
        return None, None

    # Get dimension (d) from x
    if np.asarray(x).ndim == 1:
        # x is a 1-dimensional array
        x = np.asarray(x).reshape(-1, 1)
        d = 1
    else:
        # x is a 2-dimensional array
        d = x.shape[1]

    # Check dimension of cov_model and set if used as omni-directional model
    if cov_model.__class__.__name__ != 'CovModel{}D'.format(d):
        if isinstance(cov_model, CovModel1D):
            omni_dir = True
        else:
            print("ERROR: 'cov_model' is incompatible with dimension of points")
            return None, None
    else:
        omni_dir = False

    # Number of data points
    n = x.shape[0]

    # Check size of v
    v = np.asarray(v).reshape(-1)
    if v.size != n:
        print("ERROR: size of 'v' is not valid")
        return None, None

    # Leave-one-out (loo) cross validation
    v_est, v_std = np.zeros(n), np.zeros(n)
    ind = np.arange(n)
    for i in range(n):
        indx = np.delete(ind, i)
        v_est[i], v_std[i] = interpolator(x[indx], v[indx], np.array(x[i]).reshape(-1, d), cov_model, **interpolator_kwargs)

    # Normalized error
    err = (v_est - v) / v_std
    # Each err[i] should follows a law N(0,1), the set of err[i] has n-1 degrees of freedom (?), and:
    #   (1) sqrt(n-1)*mean(err) follows approximately a law N(0,1) (CLT)
    #   (2) sum(err^2) follows a law Chi2 with n-1 degrees of freedom
    me = np.mean(err)
    s2 = np.sum(err**2)

    t = np.sqrt(n-1)*me
    tlim = stats.norm.ppf(1.-0.5*confidence)
    if np.abs(t) > tlim:
        print("Model does not pass test for mean of normalized error!")
        print("   Mean of normalized error times square root of number of data points = {}, not within interval +/-{}".format(t, tlim))
        test_normal = False
    else:
        test_normal = True

    s2lim = stats.chi2.ppf(1.-confidence, df=n-1)
    if s2 > s2lim:
        print("Model does not pass test for sum of square of normalized error (chi2)!")
        print("   Sum of squares of normalized error = {}, above limit: {}".format(s2, s2lim))
        test_chi2 = False
    else:
        test_chi2 = True

    if make_plot:
        fig, ax = plt.subplots(1,2, figsize=figsize)

        plt.sca(ax[0])
        plt.plot(v, v_est, 'o')
        tmp = [np.min(v), np.max(v)]
        plt.plot(tmp, tmp, ls='dashed')
        plt.xlabel('True value Z(x)')
        plt.ylabel('Estimation Z*(x)')
        # plt.plot(v_est, v, 'o')
        # tmp = [np.min(v_est), np.max(v_est)]
        # plt.plot(tmp, tmp, ls='dashed')
        # plt.xlabel('Estimation Z*(x)')
        # plt.ylabel('True value Z(x)')
        plt.title('Cross plot Z(x) vs Z*(x)')

        plt.sca(ax[1])
        plt.hist(err, density=True)
        plt.xlabel(r'Normalized error $(Z*(x)-Z(x))/\sigma*(x)$')

        # plt.show()

    return (v_est, v_std, test_normal, test_chi2)
# ----------------------------------------------------------------------------

# ============================================================================
# Sequential Gaussian Simulation based an simple or ordinary kriging
# ============================================================================
# ----------------------------------------------------------------------------
def sgs(x, v, xu, cov_model, method='simple_kriging', mean=None, nreal=1):
    """
    Performs Sequential Gaussian Simulation (SGS) - simulates at locations xu the values
    of a variable from the measured values v at locations x. A full neighborhood is used!
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

    :param cov_model:
                    covariance model:
                        - in same dimension as dimension of points (d), i.e.:
                            - CovModel1D class if data in 1D (d=1)
                            - CovModel2D class if data in 2D (d=2)
                            - CovModel3D class if data in 3D (d=3)
                        - or CovModel1D whatever dimension of points (d):
                            - used as an omni-directional covariance model

    :param method:  (string) indicates the method used:
                        - 'simple_kriging': interpolation by simple kriging
                        - 'ordinary_kriging': interpolation by ordinary kriging

    :param mean:    (None or float or ndarray) mean of the simulation
                        (for simple kriging only):
                            - None   : mean of hard data values (stationary),
                                       i.e. mean of v
                            - float  : for stationary mean (set manually)
                            - ndarray: of of shape (n + nu,) for non stationary mean,
                                mean at point x and xu
                        For ordinary kriging (method='ordinary_kriging'),
                        this parameter must be set to None

    :return:        vu: (2-dimensional array of shape (nreal, nu)):
                        vu[i] are the simulated values at points xu for the i-th realization
    """
    # Prevent calculation if covariance model is not stationary
    if not cov_model.is_stationary():
        print("ERROR: 'cov_model' is not stationary: sgs can not be applied")
        return None

    # Get dimension (d) from x
    if np.asarray(x).ndim == 1:
        # x is a 1-dimensional array
        x = np.asarray(x).reshape(-1, 1)
        d = 1
    else:
        # x is a 2-dimensional array
        d = x.shape[1]

    # Get dimension (du) from xu
    if np.asarray(xu).ndim == 1:
        # xu is a 1-dimensional array
        xu = np.asarray(xu).reshape(-1, 1)
        du = 1
    else:
        # xu is a 2-dimensional array
        du = xu.shape[1]

    # Check dimension of x and xu
    if d != du:
        print("ERROR: 'x' and 'xu' do not have same dimension")
        return None

    # Check dimension of cov_model and set if used as omni-directional model
    if cov_model.__class__.__name__ != 'CovModel{}D'.format(d):
        if isinstance(cov_model, CovModel1D):
            omni_dir = True
        else:
            print("ERROR: 'cov_model' is incompatible with dimension of points")
            return None
    else:
        omni_dir = False

    # Number of data points
    n = x.shape[0]
    # Number of unknown points
    nu = xu.shape[0]

    # Check size of v
    v = np.asarray(v).reshape(-1)
    if v.size != n:
        print("ERROR: size of 'v' is not valid")
        return None

    # Method
    ordinary_kriging = False
    if method == 'simple_kriging':
        if mean is None:
            if n == 0:
                # no data point
                mean = np.zeros(nu)
            else:
                mean = np.mean(v) * np.ones(n + nu)
        else:
            mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if mean.size == 1:
                mean = mean * np.ones(n + nu)
            elif mean.size != n + nu:
                print("ERROR: size of 'mean' is not valid")
                return None
            # if mean.size not in (1, n + nu):
            #     print("ERROR: size of 'mean' is not valid")
            #     return None
    elif method == 'ordinary_kriging':
        if mean is not None:
            print("ERROR: 'mean' must be None with 'method' set to 'ordinary_kriging'")
            return None
        ordinary_kriging = True
        # nmat = n + 1 # order of the kriging matrix
    else:
        print("ERROR: 'method' is not valid")
        return None

    # Allocate memory for output
    vu = np.zeros((nreal, nu))
    if vu.size == 0:
        return vu

    # Covariance function
    cov_func = cov_model.func() # covariance function
    if omni_dir:
        # covariance model in 1D is used
        cov0 = cov_func(0.) # covariance function at origin (lag=0)
    else:
        cov0 = cov_func(np.zeros(d)) # covariance function at origin (lag=0)

    # Set (simple) kriging matrix (mat) of order nmat = n + nu:
    #     mat = mat_x_x,  mat_x_xu,
    #           mat_xu_x, mat_xu_xu,
    # where
    #     mat_x_x:    covariance matrix for location x and x, of size n x n
    #                 (symmetric)
    #     mat_x_xu:   covariance matrix for location x and xu, of size n x nu
    #     mat_xu_x:   covariance matrix for location xu and x, of size nu x n
    #                 (transpose of mat_x_xu)
    #     mat_xu_xu:  covariance matrix for location xu and xu, of size nu x nu
    #                 (symmetric)
    nmat = n + nu
    mat = np.ones((nmat, nmat))
    # mat_x_x
    for i in range(n-1):
        # lag between x[i] and x[j], j=i+1, ..., n-1
        h = x[(i+1):] - x[i]
        if omni_dir:
            # compute norm of lag
            h = np.sqrt(np.sum(h**2, axis=1))
        cov_h = cov_func(h)
        mat[i, (i+1):n] = cov_h
        mat[(i+1):n, i] = cov_h
        mat[i, i] = cov0
    mat[n-1, n-1] = cov0

    # mat_x_xu, mat_xu_x
    for i in range(n):
        # lag between x[i] and xu[j], j=0, ..., n-1
        h = xu - x[i]
        if omni_dir:
            # compute norm of lag
            h = np.sqrt(np.sum(h**2, axis=1))
        cov_h = cov_func(h)
        mat[i, n:] = cov_h
        mat[n:, i] = cov_h

    # mat_xu_xu
    for i in range(nu-1):
        # lag between xu[i] and xu[j], j=i+1, ..., nu-1
        h = xu[(i+1):] - xu[i]
        if omni_dir:
            # compute norm of lag
            h = np.sqrt(np.sum(h**2, axis=1))
        cov_h = cov_func(h)
        mat[n+i, (n+i+1):(n+nu)] = cov_h
        mat[(n+i+1):(n+nu), n+i] = cov_h
        mat[n+i, n+i] = cov0
    mat[-1,-1] = cov0

    for i in range(nreal):
        # set index path visiting xu
        indu = np.random.permutation(nu)

        ind = np.hstack((np.arange(n), n + indu))
        for j, k in enumerate(indu):
            # Simulate value at xu[k] (= xu[indu[j]])
            nj = n + j
            # Solve the kriging system
            if ordinary_kriging:
                try:
                    w = np.linalg.solve(
                            np.vstack((np.hstack((mat[ind[:nj], :][:, ind[:nj]], np.ones((nj, 1)))), np.hstack((np.ones(nj), np.array([0.]))))), # kriging matrix
                            np.hstack((mat[ind[:nj], ind[nj]], np.array([1.]))) # second member
                        )
                except:
                    print("ERROR: unable to solve kriging system...")
                    return None
                # Mean (kriged) value at xu[k]
                mu = np.hstack((v, vu[i, indu[:j]])).dot(w[:nj])
                # Standard deviation (of kriging) at xu[k]
                std = np.sqrt(np.maximum(0, cov0 - np.dot(w, np.hstack((mat[ind[:nj], ind[nj]], np.array([1.]))))))
            else:
                try:
                    w = np.linalg.solve(
                            mat[ind[:nj], :][:, ind[:nj]], # kriging matrix
                            mat[ind[:nj], ind[nj]], # second member
                        )
                except:
                    print("ERROR: unable to solve kriging system...")
                    return None
                # Mean (kriged) value at xu[k]
                mu = mean[ind[nj]] + (np.hstack((v, vu[i, indu[:j]])) - mean[ind[:nj]]).dot(w[:nj])
                # Standard deviation (of kriging) at xu[k]
                std = np.sqrt(np.maximum(0, cov0 - np.dot(w, mat[ind[:nj], ind[nj]])))
            # Draw value in N(mu, std^2)
            vu[i, k] = np.random.normal(loc=mu, scale=std)

    return vu
# ----------------------------------------------------------------------------

# --- OLD ---
# # ----------------------------------------------------------------------------
# def simple_kriging(x, v, xu, cov_model, mean=0):
#     """
#     Simple kriging - interpolates at locations xu the values v measured at locations x.
#     Covariance model given should be:
#         - in same dimension as dimension of locations x, xu
#         - in 1D, it is then used as an omni-directional covariance model
#     (see below).
#
#     :param x:       (2-dimensional array of shape (n, d)) coordinates
#                         of the data points (n: number of points, d: dimension)
#                         Note: for data in 1D, it can be a 1-dimensional array of shape (n,)
#     :param v:       (1-dimensional array of shape (n,)) values at data points
#
#     :param xu:      (2-dimensional array of shape (nu, d)) coordinates
#                         of the points where the interpolation has to be done
#                         (nu: number of points, d: dimension same as for x),
#                         called unknown points
#                         Note: for data in 1D, it can be a 1-dimensional array of shape (nu,)
#
#     :param cov_model:   covariance model:
#                             - in same dimension as dimension of points (d), i.e.:
#                                 - CovModel1D class if data in 1D (d=1)
#                                 - CovModel2D class if data in 2D (d=2)
#                                 - CovModel3D class if data in 3D (d=3)
#                             - or CovModel1D whatever dimension of points (d):
#                                 - used as an omni-directional covariance model
#     :param mean:    (float) mean for simple kriging: the value (v-mean) are interpolated by
#                         simple kriging, and then the mean is added to get the final interpolated
#                         values
#
#     :return:        (vu, vu_std) with:
#                         vu:     (1-dimensional array of shape (nu,)) kriged values (estimates) at points xu
#                         vu_std: (1-dimensional array of shape (nu,)) kriged standard deviation at points xu
#     """
#     # Get dimension (d) from x
#     if np.asarray(x).ndim == 1:
#         # x is a 1-dimensional array
#         x = np.asarray(x).reshape(-1, 1)
#         d = 1
#     else:
#         # x is a 2-dimensional array
#         d = x.shape[1]
#
#     # Get dimension (du) from xu
#     if np.asarray(xu).ndim == 1:
#         # xu is a 1-dimensional array
#         xu = np.asarray(xu).reshape(-1, 1)
#         du = 1
#     else:
#         # xu is a 2-dimensional array
#         du = xu.shape[1]
#
#     # Check dimension of x and xu
#     if d != du:
#         print("ERROR: 'x' and 'xu' do not have same dimension")
#         return None, None
#
#     # Check dimension of cov_model and set if used as omni-directional model
#     if cov_model.__class__.__name__ != 'CovModel{}D'.format(d):
#         if cov_model.__class__.__name__ == 'CovModel1D':
#             omni_dir = True
#         else:
#             print("ERROR: 'cov_model' is incompatible with dimension of points")
#             return None, None
#     else:
#         omni_dir = False
#
#     # Number of data points
#     n = x.shape[0]
#     # Number of unknown points
#     nu = xu.shape[0]
#
#     # Check length of v
#     if len(v) != n:
#         print("ERROR: length of 'v' is not valid")
#         return None, None
#
#     # Covariance function
#     cov_func = cov_model.func() # covariance function
#     if omni_dir:
#         # covariance model in 1D is used
#         cov0 = cov_func(0.) # covariance function at origin (lag=0)
#     else:
#         cov0 = cov_func(np.zeros(d)) # covariance function at origin (lag=0)
#
#     # Fill matrix of simple kriging system (matOK)
#     nSK = n # order of the matrix
#     matSK = np.ones((nSK, nSK))
#     for i in range(n-1):
#         # lag between x[i] and x[j], j=i+1, ..., n-1
#         h = x[(i+1):] - x[i]
#         if omni_dir:
#             # compute norm of lag
#             h = np.sqrt(np.sum(h**2, axis=1))
#         cov_h = cov_func(h)
#         matSK[i, (i+1):] = cov_h
#         matSK[(i+1):, i] = cov_h
#         matSK[i,i] = cov0
#     matSK[-1,-1] = cov0
#
#     # Right hand side of the simple kriging system (b):
#     #   b is a matrix of dimension nSK x nu
#     b = np.ones((nSK, nu))
#     for i in range(n):
#         # lag between x[i] and every xu
#         h = xu - x[i]
#         if omni_dir:
#             # compute norm of lag
#             h = np.sqrt(np.sum(h**2, axis=1))
#         b[i,:] = cov_func(h)
#
#     # Solve the kriging system
#     w = np.linalg.solve(matSK,b) # w: matrix of dimension nOK x nu
#
#     # Kriged values at unknown points
#     vu = mean + (v-mean).dot(w)
#
#     # Kriged standard deviation at unknown points
#     vu_std = np.sqrt(np.maximum(0, cov0 - np.array([np.dot(w[:,i], b[:,i]) for i in range(nu)])))
#
#     return (vu, vu_std)
# # ----------------------------------------------------------------------------
#
# # ----------------------------------------------------------------------------
# def ordinary_kriging(x, v, xu, cov_model):
#     """
#     Ordinary kriging - interpolates at locations xu the values v measured at locations x.
#     Covariance model given should be:
#         - in same dimension as dimension of locations x, xu
#         - in 1D, it is then used as an omni-directional covariance model
#     (see below).
#
#     :param x:       (2-dimensional array of shape (n, d)) coordinates
#                         of the data points (n: number of points, d: dimension)
#                         Note: for data in 1D, it can be a 1-dimensional array of shape (n,)
#     :param v:       (1-dimensional array of shape (n,)) values at data points
#
#     :param xu:      (2-dimensional array of shape (nu, d)) coordinates
#                         of the points where the interpolation has to be done
#                         (nu: number of points, d: dimension same as for x),
#                         called unknown points
#                         Note: for data in 1D, it can be a 1-dimensional array of shape (nu,)
#
#     :param cov_model:   covariance model:
#                             - in same dimension as dimension of points (d), i.e.:
#                                 - CovModel1D class if data in 1D (d=1)
#                                 - CovModel2D class if data in 2D (d=2)
#                                 - CovModel3D class if data in 3D (d=3)
#                             - or CovModel1D whatever dimension of points (d):
#                                 - used as an omni-directional covariance model
#
#     :return:        (vu, vu_std) with:
#                         vu:     (1-dimensional array of shape (nu,)) kriged values (estimates) at points xu
#                         vu_std: (1-dimensional array of shape (nu,)) kriged standard deviation at points xu
#     """
#     # Get dimension (d) from x
#     if np.asarray(x).ndim == 1:
#         # x is a 1-dimensional array
#         x = np.asarray(x).reshape(-1, 1)
#         d = 1
#     else:
#         # x is a 2-dimensional array
#         d = x.shape[1]
#
#     # Get dimension (du) from xu
#     if np.asarray(xu).ndim == 1:
#         # xu is a 1-dimensional array
#         xu = np.asarray(xu).reshape(-1, 1)
#         du = 1
#     else:
#         # xu is a 2-dimensional array
#         du = xu.shape[1]
#
#     # Check dimension of x and xu
#     if d != du:
#         print("ERROR: 'x' and 'xu' do not have same dimension")
#         return None, None
#
#     # Check dimension of cov_model and set if used as omni-directional model
#     if cov_model.__class__.__name__ != 'CovModel{}D'.format(d):
#         if cov_model.__class__.__name__ == 'CovModel1D':
#             omni_dir = True
#         else:
#             print("ERROR: 'cov_model' is incompatible with dimension of points")
#             return None, None
#     else:
#         omni_dir = False
#
#     # Number of data points
#     n = x.shape[0]
#     # Number of unknown points
#     nu = xu.shape[0]
#
#     # Check length of v
#     if len(v) != n:
#         print("ERROR: length of 'v' is not valid")
#         return None, None
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
# # ----------------------------------------------------------------------------
# def cross_valid_loo(x, v, cov_model, confidence=0.05, interpolator=ordinary_kriging, interpolator_kwargs={}, make_plot=True, figsize=None):
#     """
#     Cross-validation of covariance model by leave-one-out error based on given interpolator.
#
#     Covariance model given should be:
#         - in same dimension as dimension of locations x
#         - in 1D, it is then used as an omni-directional covariance model
#     Interpolator should be:
#         - the function 'ordinary_kriging' (default)
#         - the function 'simple_kriging'; in this case, the mean can be passed in
#           the dictionary: interpolator_kwargs={'mean':<value>}
#     Two statisic tests are performed:
#         (1) normal law test for mean of normalized error:
#             Mean of normalized error times the square root of n-1
#             should follow approximately a law N(0,1) (CLT)
#         (2) Chi2 test for sum of squares of normalized error:
#             Sum of square of normalized error should follow a law
#             Chi2 with n-1 degrees of freedom,
#     n being the number of data points.
#     The statistc test passes with success if the obtained value is within
#     the central interval covering the 1-confidence  part of the corresponding
#     distribution (by default: confidence is set to 5%), otherwise the test fails.
#
#     :param x:       (2-dimensional array of shape (n, d)) coordinates
#                         of the data points (n: number of points, d: dimension)
#                         Note: for data in 1D, it can be a 1-dimensional array of shape (n,)
#     :param v:       (1-dimensional array of shape (n,)) values at data points
#
#     :param cov_model:   covariance model:
#                             - in same dimension as dimension of points (d), i.e.:
#                                 - CovModel1D class if data in 1D (d=1)
#                                 - CovModel2D class if data in 2D (d=2)
#                                 - CovModel3D class if data in 3D (d=3)
#                             - or CovModel1D whatever dimension of points (d):
#                                 - used as an omni-directional covariance model
#
#     :param confidence:  (float) in [0,1] for setting limit in the two statistic tests
#                             (see above)
#
#     :param interpolator:
#                     (function) function used for interpolation, ordinary_kriging (default),
#                     or simple_kriging
#
#     :interpolator_kwargs:
#                     (dict) keyword argument passed to interpolator; e.g. if
#                     interpolator=simple_kriging, then interpolator_kwargs={'mean':<value>},
#                     allows to specify the mean used
#
#     :param make_plot:
#                     (bool) if True: a plot is done (in a new 1x2 figure)
#
#     :param figsize: (tuple of 2 ints) size of the figure (used if make_plot is True)
#
#     :return:    (v_est, v_std, test_normal, test_chi2), tuple of length 4:
#                     v_est: (1-dimensional array of shape (n,)) estimated values at data points
#                     v_std: (1-dimensional array of shape (n,)) standard deviation values at data points
#                     test_normal:    (bool) result of test (1) (normal law), True if success, False otherwise
#                     test_chi2:      (bool) result of test (1) (chi2), True if success, False otherwise
#     """
#     # Get dimension (d) from x
#     if np.asarray(x).ndim == 1:
#         # x is a 1-dimensional array
#         x = np.asarray(x).reshape(-1, 1)
#         d = 1
#     else:
#         # x is a 2-dimensional array
#         d = x.shape[1]
#
#     # Check dimension of cov_model and set if used as omni-directional model
#     if cov_model.__class__.__name__ != 'CovModel{}D'.format(d):
#         if cov_model.__class__.__name__ == 'CovModel1D':
#             omni_dir = True
#         else:
#             print("ERROR: 'cov_model' is incompatible with dimension of points")
#             return None, None
#     else:
#         omni_dir = False
#
#     # Number of data points
#     n = x.shape[0]
#
#     # Check length of v
#     if len(v) != n:
#         print("ERROR: length of 'v' is not valid")
#         return None, None
#
#     # Leave-one-out (loo) cross validation
#     v_est, v_std = np.zeros(n), np.zeros(n)
#     ind = np.arange(n)
#     for i in range(n):
#         indx = np.delete(ind, i)
#         v_est[i], v_std[i] = interpolator(x[indx], v[indx], np.array(x[i]).reshape(-1, d), cov_model, **interpolator_kwargs)
#
#     # Normalized error
#     err = (v_est - v) / v_std
#     # Each err[i] should follows a law N(0,1), the set of err[i] has n-1 degrees of freedom (?), and:
#     #   (1) sqrt(n-1)*mean(err) follows approximately a law N(0,1) (CLT)
#     #   (2) sum(err^2) follows a law Chi2 with n-1 degrees of freedom
#     me = np.mean(err)
#     s2 = np.sum(err**2)
#
#     t = np.sqrt(n-1)*me
#     tlim = stats.norm.ppf(1.-0.5*confidence)
#     if np.abs(t) > tlim:
#         print("Model does not pass test for mean of normalized error!")
#         print("   Mean of normalized error times square root of number of data points = {}, not within interval +/-{}".format(t, tlim))
#         test_normal = False
#     else:
#         test_normal = True
#
#     s2lim = stats.chi2.ppf(1.-confidence, df=n-1)
#     if s2 > s2lim:
#         print("Model does not pass test for sum of square of normalized error (chi2)!")
#         print("   Sum of squares of normalized error = {}, above limit: {}".format(s2, s2lim))
#         test_chi2 = False
#     else:
#         test_chi2 = True
#
#     if make_plot:
#         fig, ax = plt.subplots(1,2, figsize=figsize)
#
#         plt.sca(ax[0])
#         plt.plot(v, v_est, 'o')
#         tmp = [np.min(v), np.max(v)]
#         plt.plot(tmp, tmp, ls='dashed')
#         plt.xlabel('True value Z(x)')
#         plt.ylabel('Estimation Z*(x)')
#         # plt.plot(v_est, v, 'o')
#         # tmp = [np.min(v_est), np.max(v_est)]
#         # plt.plot(tmp, tmp, ls='dashed')
#         # plt.xlabel('Estimation Z*(x)')
#         # plt.ylabel('True value Z(x)')
#         plt.title('Cross plot Z(x) vs Z*(x)')
#
#         plt.sca(ax[1])
#         plt.hist(err, density=True)
#         plt.xlabel(r'Normalized error $(Z*(x)-Z(x))/\sigma*(x)$')
#
#         # plt.show()
#
#     return (v_est, v_std, test_normal, test_chi2)
# # ----------------------------------------------------------------------------
# # ----------------------------------------------------------------------------
# def cross_valid_loo_ok(x, v, cov_model, confidence=0.05, make_plot=True, figsize=None):
#     """
#     Cross-validation of covariance model by leave-one-out error based on ordinary kriging.
#     Covariance model given should be:
#         - in same dimension as dimension of locations x
#         - in 1D, it is then used as an omni-directional covariance model
#     Two statisic tests are performed:
#         (1) normal law test for mean of normalized error:
#             Mean of normalized error times the square root of n-1
#             should follow approximately a law N(0,1) (CLT)
#         (2) Chi2 test for sum of squares of normalized error:
#             Sum of square of normalized error should follow a law
#             Chi2 with n-1 degrees of freedom,
#     n being the number of data points.
#     The statistc test passes with success if the obtained value is within
#     the central interval covering the 1-confidence  part of the corresponding
#     distribution (by default: confidence is set to 5%), otherwise the test fails.
#
#     :param x:       (2-dimensional array of shape (n, d)) coordinates
#                         of the data points (n: number of points, d: dimension)
#                         Note: for data in 1D, it can be a 1-dimensional array of shape (n,)
#     :param v:       (1-dimensional array of shape (n,)) values at data points
#
#     :param cov_model:   covariance model:
#                             - in same dimension as dimension of points (d), i.e.:
#                                 - CovModel1D class if data in 1D (d=1)
#                                 - CovModel2D class if data in 2D (d=2)
#                                 - CovModel3D class if data in 3D (d=3)
#                             - or CovModel1D whatever dimension of points (d):
#                                 - used as an omni-directional covariance model
#
#     :param confidence:  (float) in [0,1] for setting limit in the two statistic tests
#                             (see above)
#
#     :param make_plot:
#                     (bool) if True: a plot is done (in a new 1x2 figure)
#
#     :param figsize: (tuple of 2 ints) size of the figure (used if make_plot is True)
#
#     :return:    (valid_code1, valid_code2), a tuple of 2 bools:
#                     valid_code1: True if test (1) passed with success, False otherwise
#                     valid_code2: True if test (2) passed with success, False otherwise
#     """
#     # Get dimension (d) from x
#     if np.asarray(x).ndim == 1:
#         # x is a 1-dimensional array
#         x = np.asarray(x).reshape(-1, 1)
#         d = 1
#     else:
#         # x is a 2-dimensional array
#         d = x.shape[1]
#
#     # Check dimension of cov_model and set if used as omni-directional model
#     if cov_model.__class__.__name__ != 'CovModel{}D'.format(d):
#         if cov_model.__class__.__name__ == 'CovModel1D':
#             omni_dir = True
#         else:
#             print("ERROR: 'cov_model' is incompatible with dimension of points")
#             return None, None
#     else:
#         omni_dir = False
#
#     # Number of data points
#     n = x.shape[0]
#
#     # Check length of v
#     if len(v) != n:
#         print("ERROR: length of 'v' is not valid")
#         return None, None
#
#     # Leave-one-out (loo) cross validation
#     v_est, v_std = np.zeros(n), np.zeros(n)
#     ind = np.arange(n)
#     for i in range(n):
#         indx = np.delete(ind, i)
#         v_est[i], v_std[i] = ordinary_kriging(x[indx], v[indx], np.array(x[i]).reshape(-1, d), cov_model)
#
#     # Normalized error
#     err = (v_est - v) / v_std
#     # Each err[i] should follows a law N(0,1), the set of err[i] has n-1 degrees of freedom (?), and:
#     #   (1) sqrt(n-1)*mean(err) follows approximately a law N(0,1) (CLT)
#     #   (2) sum(err^2) follows a law Chi2 with n-1 degrees of freedom
#     me = np.mean(err)
#     s2 = np.sum(err**2)
#
#     t = np.sqrt(n-1)*me
#     tlim = stats.norm.ppf(1.-0.5*confidence)
#     if np.abs(t) > tlim:
#         print("Model does not pass test for mean of normalized error!")
#         print("   Mean of normalized error times square root of number of data points = {}, not within interval +/-{}".format(t, tlim))
#         valid_code1 = False
#     else:
#         valid_code1 = True
#
#     s2lim = stats.chi2.ppf(1.-confidence, df=n-1)
#     if s2 > s2lim:
#         print("Model does not pass test for sum of square of normalized error (chi2)!")
#         print("   Sum of squares of normalized error = {}, above limit: {}".format(s2, s2lim))
#         valid_code2 = False
#     else:
#         valid_code2 = True
#
#     if make_plot:
#         fig, ax = plt.subplots(1,2, figsize=figsize)
#
#         plt.sca(ax[0])
#         plt.plot(v_est, v, 'o')
#         tmp = [np.min(v_est), np.max(v_est)]
#         plt.plot(tmp, tmp, ls='dashed')
#         plt.xlabel('Estimation Z*(x)')
#         plt.ylabel('True value Z(x)')
#         plt.title('Cross plot Z(x) vs Z*(x)')
#
#         plt.sca(ax[1])
#         plt.hist(err, density=True)
#         plt.xlabel(r'Normalized error $(Z*(x)-Z(x))/\sigma*(x)$')
#
#         # plt.show()
#
#     return (valid_code1, valid_code2)
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
