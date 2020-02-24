#!/usr/bin/python3
#-*- coding: utf-8 -*-

"""
Python module:  'covModel.py'
author:         Julien Straubhaar
date:           jan-2018

Definition of classes for covariance / variogram models in 1D, 2D and 3D.
"""

import numpy as np

# ============================================================================
# Definition of 1D elementary covariance models:
#   - nugget, spherical, exponential, gaussian, cubic,
#   - power (non-stationary)
# ----------------------------------------------------------------------------

def cov_nug(h, w=1.0):
    """
    1D-nugget covariance model:

    :param h:   (1-dimensional array or float): lag(s)
    :param w:   (float >0): weight (sill)
    :return:    (1-dimensional array or float) evaluation of the model at h
    """
    return (w * np.asarray(h==0., dtype=float))

def cov_sph(h, w=1.0, r=1.0):
    """
    1D-shperical covariance model:

    :param h:   (1-dimensional array or float): lag(s)
    :param w:   (float >0): weight (sill)
    :param r:   (float >0): range
    :return:    (1-dimensional array or float) evaluation of the model at h
    """
    t = np.minimum(np.abs(h)/r, 1.) # "parallel or element-wise minimum"
    return (w * (1 - 0.5 * t * (3. - t**2))) # w * (1 - 3/2 * t + 1/2 * t^3)

def cov_exp(h, w=1.0, r=1.0):
    """
    1D-gaussian covariance model (with sill=1 and range=1):

    :param h:   (1-dimensional array or float): lag(s)
    :param w:   (float >0): weight (sill)
    :param r:   (float >0): range
    :return:    (1-dimensional array or float) evaluation of the model at h
    """
    return (w * np.exp(-3. * np.abs(h)/r)) # w * exp(-3*|h|/r)

def cov_gau(h, w=1.0, r=1.0):
    """
    1D-gaussian covariance model (with sill=1 and range=1):

    :param h:   (1-dimensional array or float): lag(s)
    :param w:   (float >0): weight (sill)
    :param r:   (float >0): range
    :return:    (1-dimensional array or float) evaluation of the model at h
    """
    return (w * np.exp(-3. * (h/r)**2)) # w * exp(-3*(h/r)^2)

def cov_cub(h, w=1.0, r=1.0):
    """
    1D-cubic covariance model (with sill=1 and range=1):

    :param h:   (1-dimensional array or float): lag(s)
    :param w:   (float >0): weight (sill)
    :param r:   (float >0): range
    :return:    (1-dimensional array or float) evaluation of the model at h
    """
    t = np.minimum(np.abs(h)/r, 1.) # "parallel or element-wise minimum"
    t2 = t**2
    return (w * (1 + t2 * (-7. + t * (8.75 + t2 * (-3.5 + 0.75 * t2))))) # w * (1 - 7 * t^2 + 35/4 * t^3 - 7/2 * t^5 + 3/4 * t^7)

def cov_pow(h, w=1.0, r=1.0, s=1.0):
    """
    1D-power covariance model (with sill=1 and range=1):

    :param h:   (1-dimensional array or float): lag(s)
    :param w:   (float >0): weight (sill)
    :param r:   (float >0): range
    :param s:   (float btw 0 and 2): power
    :return:    (1-dimensional array or float) evaluation of the model at h
    """
    return (w * (1. - (h/r)**s))
# ----------------------------------------------------------------------------


# ============================================================================
# Definition of class for covariance models in 1D, 2D, 3D, as combination
# of elementary models and accounting for anisotropy and rotation
# ----------------------------------------------------------------------------

class CovModel1D (object):
    """
    Defines a covariance model in 1D:
        elem:   (sequence of 2-tuple) an entry (t, d) of the sequence
                    corresponds to an elementary model with:
                        t: (string) the type, could be
                           'nugget', 'spherical', 'exponential', 'gaussian',
                           'cubic', 'power'
                        d: (dict) dictionary of required parameters to be
                            passed to the elementary model,
                    e.g.
                       (t, d) = ('power', {w:2.0, r:1.5, s:1.7})
                    the final model is the sum of the elementary models
        name:   (string) name of the model
    """

    def __init__(self,
                 elem=[],
                 name=""):
        self.elem = elem
        self.name = name

    def sill(self):
        """Returns the sill."""
        return sum([d['w'] for t, d in self.elem if 'w' in d])

    def r(self):
        """Returns the range (max)."""
        r = 0.
        for t, d in self.elem:
            if 'r' in d:
                r = max(r, d['r'])

        return (r)

    def func(self):
        """
        Returns the covariance model function f(h) where:
            h:      (1-dimensional array or float) 1D-lag(s)
            f(h):   (1-dimensional array) evaluation of the model at h
                        note that the result is casted to a 1-dimensional array
        """
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

                elif t == 'cubic':
                    s = s + cov_cub(h, **d)

                elif t == 'power':
                    s = s + cov_pow(h, **d)

            return s

        return f

    def vario_func(self):
        """
        Returns the varioram model function f(h) where:
            h:      (1-dimensional array or float) 1D-lag(s)
            f(h):   (1-dimensional array) evaluation of the model at h
                        note that the result is casted to a 1-dimensional array
        """
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

                elif t == 'cubic':
                    s = s + d['w'] - cov_cub(h, **d)

                elif t == 'power':
                    s = s + d['w'] - cov_pow(h, **d)

            return s

        return f
# ----------------------------------------------------------------------------

class CovModel2D (object):
    """
    Defines a covariance model in 2D:
        elem:   (sequence of 2-tuple) an entry (t, d) of the sequence
                    corresponds to an elementary model with:
                        t: (string) the type, could be
                           'nugget','spherical','exponential', 'gaussian',
                           'cubic', 'power'
                        d: (dict) dictionary of required parameters to be
                            passed to the elementary model, excepting
                            the parameter 'r' which must be given here
                            as a sequence of range along each axis
                    e.g.
                       (t, d) = ('power', {w:2.0, r:[1.5, 2.5], s:1.7})
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
    """

    def __init__(self,
                 elem=[],
                 alpha=0.,
                 name=""):
        self.elem = elem
        self.alpha = alpha
        self.name = name

    def sill(self):
        """Returns the sill."""
        return sum([d['w'] for t, d in self.elem if 'w' in d])

    def mrot(self):
        """Returns the 2x2 matrix m for changing the coordinate system from Ox'y'
        to Oxy, where Ox' and Oy' are the axes supporting the ranges of the model."""
        a = self.alpha * np.pi/180.
        ca, sa = np.cos(a), np.sin(a)
        return (np.array([[ca, sa], [-sa, ca]]))

    def r12(self):
        """Returns the range (max) along each axis in the new coordinate system
        (corresponding the axes of the ellipse supporting the covariance model).
        """
        r = [0., 0.]
        for t, d in self.elem:
            if 'r' in d:
                r = np.maximum(r, d['r']) # element-wise maximum

        return r

    def rxy(self):
        """Returns the range (max) along each axis in the original coordinate
        system.
        """
        r12 = self.r12()
        m = np.abs(self.mrot())

        return np.maximum(r12[0] * m[:,0], r12[1] * m[:,1]) # element-wise maximum

    def func(self):
        """
        Returns the covariance model function f(h) where:
            h:      (2-dimensional array of dim n x 2, or
                        1-dimensional array of dim 2) 2D-lag(s)
            f(h):   (1-dimensional array of dim n) evaluation of the model at h
        """
        def f(h):
            h = np.array(h).reshape(-1,2)  # cast to 2-dimensional array with 2 columns if needed
            if self.alpha != 0:
                hnew = np.dot(h,self.mrot()).reshape(-1,2)
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

                elif t == 'cubic':
                    s = s + cov_cub(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'power':
                    s = s + cov_pow(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

            return s

        return f

    def vario_func(self):
        """
        Returns the variogram model function f(h) where:
            h:      (2-dimensional array of dim n x 2, or
                        1-dimensional array of dim 2) 2D-lag(s)
            f(h):   (1-dimensional array of dim n) evaluation of the model at h
        """
        def f(h):
            h = np.array(h).reshape(-1,2)  # cast to 2-dimensional array with 2 columns if needed
            if self.alpha != 0:
                hnew = np.dot(h,self.mrot()).reshape(-1,2)
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

                elif t == 'cubic':
                    s = s + d['w'] - cov_cub(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'power':
                    s = s + d['w'] - cov_pow(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

            return s

        return f
# ----------------------------------------------------------------------------

class CovModel3D (object):
    """
    Defines a covariance model in 3D:
        elem:   (sequence of 2-tuple) an entry (t, d) of the sequence
                    corresponds to an elementary model with:
                        t: (string) the type, could be
                           'nugget','spherical','exponential', 'gaussian',
                           'cubic', 'power'
                        d: (dict) dictionary of required parameters to be
                            passed to the elementary model, excepting
                            the parameter 'r' which must be given here
                            as a sequence of range along each axis
                    e.g.
                       (t, d) = ('power', {w:2.0, r:[1.5, 2.5], s:1.7})
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
    """

    def __init__(self,
                 elem=[],
                 alpha=0., beta=0., gamma=0.,
                 name=""):
        self.elem = elem
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.name = name

    def sill(self):
        """Returns the sill."""
        return sum([d['w'] for t, d in self.elem if 'w' in d])

    def mrot(self):
        """Returns the 3x3 matrix m for changing the coordinate system from
        Ox'''y'''z''' to Oxyz, where Ox''', Oy''', Oz''' are the axes supporting
        the ranges of the model."""
        a = self.alpha * np.pi/180.
        b = self.beta * np.pi/180.
        c = self.gamma * np.pi/180.
        ca, sa = np.cos(a), np.sin(a)
        cb, sb = np.cos(b), np.sin(b)
        cc, sc = np.cos(c), np.sin(c)

        return (np.array([[  ca * cc + sa * sb * sc,  sa * cb,  - ca * sc + sa * sb * cc],
                          [- sa * cc + ca * sb * sc,  ca * cb,    sa * sc + ca * sb * cc],
                          [                 cb * sc,     - sb,                  cb * cc ]]))

    def r123(self):
        """Returns the range (max) along each axis in the new coordinate system
        (corresponding the axes of the ellipse supporting the covariance model).
        """
        r = [0., 0., 0.]
        for t, d in self.elem:
            if 'r' in d:
                r = np.maximum(r, d['r']) # element-wise maximum

        return r

    def rxyz(self):
        """Returns the range (max) along each axis in the original coordinate
        system.
        """
        r123 = self.r123()
        m = np.abs(self.mrot())

        return np.maximum(r123[0] * m[:,0], r123[1] * m[:,1], r123[2] * m[:,2]) # element-wise maximum

    def func(self):
        """
        Returns the covariance model function f(h) where:
            h:      (2-dimensional array of dim n x 3, or
                        1-dimensional array of dim 3) 2D-lag(s)
            f(h):   (1-dimensional array of dim n) evaluation of the model at h
        """
        def f(h):
            h = np.array(h).reshape(-1,3)  # cast to 2-dimensional array with 3 columns if needed
            if self.alpha != 0 or self.beta != 0 or self.gamma != 0:
                hnew = np.dot(h,self.mrot()).reshape(-1,3)
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

                elif t == 'cubic':
                    s = s + cov_cub(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'power':
                    s = s + cov_pow(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

            return s

        return f

    def vario_func(self):
        """
        Returns the variogram model function f(h) where:
            h:      (2-dimensional array of dim n x 3, or
                        1-dimensional array of dim 3) 2D-lag(s)
            f(h):   (1-dimensional array of dim n) evaluation of the model at h
        """
        def f(h):
            h = np.array(h).reshape(-1,3)  # cast to 2-dimensional array with 3 columns if needed
            if self.alpha != 0 or self.beta != 0 or self.gamma != 0:
                hnew = np.dot(h,self.mrot()).reshape(-1,3)
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

                elif t == 'cubic':
                    s = s + d['w'] - cov_cub(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

                elif t == 'power':
                    s = s + d['w'] - cov_pow(np.sqrt(np.sum((hnew/d['r'])**2, axis=1)), **dnew)

            return s

        return f
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Module 'geone.covModel' example:")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    ########## 1D case ##########
    # Define covariance model
    cov_model = CovModel1D(elem=[
                    ('gaussian', {'w':5., 'r':100}), # elementary contribution
                    ('nugget', {'w':1.})             # elementary contribution
                    ], name='model-1D example')

    # Get covariance and variogram function
    cov_fun = cov_model.func()
    vario_fun = cov_model.vario_func()

    # Get range and sill
    w = cov_model.sill()
    r = cov_model.r()

    h = np.linspace(0, 1.5*r, 200)
    ch = cov_fun(h)
    vh = vario_fun(h)

    fig, ax = plt.subplots(figsize=(16,10))

    plt.plot(h, ch, label='cov')
    plt.plot(h, vh, label='vario')
    plt.axhline(w, c='gray', ls='dashed')
    plt.axvline(r, c='gray', ls='dashed')
    plt.legend()
    plt.title(cov_model.name)

    # Set axes through the origin
    # ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    #ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    #ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))

    fig.show()

    # ########## 2D case ##########
    # Define covariance model
    cov_model = CovModel2D(elem=[
                    ('gaussian', {'w':8.5, 'r':[150, 40]}), # elementary contribution
                    ('nugget', {'w':0.5})                   # elementary contribution
                    ], alpha=-30, name='model-2D example')

    # Get covariance and variogram function
    cov_fun = cov_model.func()
    vario_fun = cov_model.vario_func()

    # Get ranges and sill
    w = cov_model.sill()
    r = max(cov_model.r12())

    hx = np.linspace(-1.2*r, 1.2*r, 100)
    hy = np.linspace(-1.2*r, 1.2*r, 100)

    hhx, hhy = np.meshgrid(hx, hy)

    hh = np.hstack((hhx.reshape(-1,1), hhy.reshape(-1,1))) # 2D-lags: (n,2) array

    ch = cov_fun(hh).reshape(len(hy), len(hx))
    vh = vario_fun(hh).reshape(len(hy), len(hx))

    xmin, xmax = min(hx), max(hx)
    ymin, ymax = min(hy), max(hy)

    # fig: cov and vario using imshow
    fig, ax = plt.subplots(1,2,figsize=(16,10))
    cbarShrink = 0.6

    plt.subplot(1,2,1)
    im_plot = plt.imshow(ch, cmap='viridis',
                         origin='lower', extent=[xmin,xmax,ymin,ymax],
                         interpolation='none')
    plt.colorbar(shrink=cbarShrink)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('{}: cov'.format(cov_model.name))

    plt.subplot(1,2,2)
    im_plot = plt.imshow(vh, cmap='viridis',
                         origin='lower', extent=[xmin,xmax,ymin,ymax],
                         interpolation='none')
    plt.colorbar(shrink=cbarShrink)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('{}: vario'.format(cov_model.name))

    fig.show()

    # fig: cov using plot_surface
    fig = plt.figure()
    ax = Axes3D(fig) # or: ax = fig.gca(projection='3d')
    surf = ax.plot_surface(hhx, hhy, ch, cmap='viridis', rstride=1, cstride=1)
    fig.colorbar(surf, shrink=.7)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('{}: cov'.format(cov_model.name))

    fig.show()

    # fig: vario using plot_surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(hhx, hhy, vh, cmap='viridis', rstride=1, cstride=1)
    fig.colorbar(surf, shrink=.7)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('{}: vario'.format(cov_model.name))
    fig.show()

    # # fig: cov and vario using plot_surface
    # fig = plt.figure(figsize=(16,10))
    # ax = fig.add_subplot(1, 2, 1, projection='3d')
    # surf = ax.plot_surface(hhx, hhy, ch, cmap='viridis', rstride=1, cstride=1)
    # fig.colorbar(surf, shrink=.7)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_title('{}: cov'.format(cov_model.name))
    #
    # ax = fig.add_subplot(1, 2, 2, projection='3d')
    # surf = ax.plot_surface(hhx, hhy, vh, cmap='viridis', rstride=1, cstride=1)
    # fig.colorbar(surf, shrink=.7)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_title('{}: vario'.format(cov_model.name))
    # fig.show()

    ########## 3D case ##########
    # Define covariance model
    cov_model = CovModel3D(elem=[
                    ('gaussian', {'w':8.5, 'r':[40, 20, 10]}), # elementary contribution
                    ('nugget', {'w':0.5})                      # elementary contribution
                    ], alpha=-30, beta=-45, gamma=20, name='model-3D example')

    # Get covariance and variogram function
    cov_fun = cov_model.func()
    vario_fun = cov_model.vario_func()

    # Get ranges and sill
    w = cov_model.sill()
    r = max(cov_model.r123())

    nx, ny, nz = 101, 101, 101
    hx = np.linspace(-1.1*r, 1.1*r, 101)
    hy = np.linspace(-1.1*r, 1.1*r, 101)
    hz = np.linspace(-1.1*r, 1.1*r, 101)

    hhy, hhz, hhx = np.meshgrid(hy, hz, hx) # as this!!!
    # hhz, hhy, hhx = np.meshgrid(hz, hy, hx, indexing='ij') # alternative

    hh = np.hstack((hhx.reshape(-1,1), hhy.reshape(-1,1), hhz.reshape(-1,1))) # 3D-lags: (n,3) array

    ch = cov_fun(hh).reshape(len(hz), len(hy), len(hx))
    vh = vario_fun(hh).reshape(len(hz), len(hy), len(hx))

    xmin, xmax = min(hx), max(hx)
    ymin, ymax = min(hy), max(hy)
    zmin, zmax = min(hz), max(hz)

    ix0, iy0, iz0 = int((nx-1)/2), int((ny-1)/2), int((nz-1)/2)

    fig, ax = plt.subplots(2,3,figsize=(24,15))
    cbarShrink = 0.8
    # cov ...
    # ... xy slice
    plt.subplot(2,3,1)
    im_plot = plt.imshow(ch[iz0,:,:],
                         origin='lower', extent=[xmin,xmax,ymin,ymax],
                         interpolation='none')
    plt.colorbar(shrink=cbarShrink)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('{}: cov xy-slice z={}'.format(cov_model.name, hz[iz0]))

    # ... xz slice
    plt.subplot(2,3,2)
    im_plot = plt.imshow(ch[:,iy0,:],
                         origin='lower', extent=[xmin,xmax,zmin,zmax],
                         interpolation='none')
    plt.colorbar(shrink=cbarShrink)
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('{}: cov xz-slice y={}'.format(cov_model.name, hy[iy0]))

    # ... yz slice
    plt.subplot(2,3,3)
    im_plot = plt.imshow(ch[:,:,ix0],
                         origin='lower', extent=[ymin,ymax,zmin,zmax],
                         interpolation='none')
    plt.colorbar(shrink=cbarShrink)
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('{}: cov yz-slice x={}'.format(cov_model.name, hx[ix0]))

    # vario ...
    # ... xy slice
    plt.subplot(2,3,4)
    im_plot = plt.imshow(vh[iz0,:,:],
                         origin='lower', extent=[xmin,xmax,ymin,ymax],
                         interpolation='none')
    plt.colorbar(shrink=cbarShrink)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('{}: vario xy-slice z={}'.format(cov_model.name, hz[iz0]))

    # ... xz slice
    plt.subplot(2,3,5)
    im_plot = plt.imshow(vh[:,iy0,:],
                         origin='lower', extent=[xmin,xmax,zmin,zmax],
                         interpolation='none')
    plt.colorbar(shrink=cbarShrink)
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('{}: vario xz-slice y={}'.format(cov_model.name, hy[iy0]))

    # ... yz slice
    plt.subplot(2,3,6)
    im_plot = plt.imshow(vh[:,:,ix0],
                         origin='lower', extent=[ymin,ymax,zmin,zmax],
                         interpolation='none')
    plt.colorbar(shrink=cbarShrink)
    plt.xlabel('y')
    plt.ylabel('z')
    plt.title('{}: vario yz-slice x={}'.format(cov_model.name, hx[ix0]))

    #plt.show()
    fig.show()

    a = input("Press enter to continue...")
