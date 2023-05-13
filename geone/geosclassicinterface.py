#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python module:  'geosclassicinterface.py'
author:         Julien Straubhaar
date:           jun-2021

Module interfacing classical geostatistics for python (estimation and simulation
based on simple and ordinary kriging).
"""

import numpy as np
import sys, os
import multiprocessing

from geone import img
from geone.geosclassic_core import geosclassic
from geone import covModel as gcm
from geone.img import Img, PointSet

version = [geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER]

# ----------------------------------------------------------------------------
def img_py2C(im_py):
    """
    Converts an image from python to C.

    :param im_py:   (Img class) image (python class)
    :return im_c:   (MPDS_IMAGE *) image converted (C struct)
    """

    im_c = geosclassic.malloc_MPDS_IMAGE()
    geosclassic.MPDSInitImage(im_c)

    err = geosclassic.MPDSMallocImage(im_c, im_py.nxyz(), im_py.nv)
    if err:
        print ('ERROR: can not convert image from python to C')
        return

    im_c.grid.nx = im_py.nx
    im_c.grid.ny = im_py.ny
    im_c.grid.nz = im_py.nz

    im_c.grid.sx = im_py.sx
    im_c.grid.sy = im_py.sy
    im_c.grid.sz = im_py.sz

    im_c.grid.ox = im_py.ox
    im_c.grid.oy = im_py.oy
    im_c.grid.oz = im_py.oz

    im_c.grid.nxy = im_py.nxy()
    im_c.grid.nxyz = im_py.nxyz()

    im_c.nvar = im_py.nv

    im_c.nxyzv = im_py.nxyz() * im_py.nv

    for i in range(im_py.nv):
        geosclassic.mpds_set_varname(im_c.varName, i, im_py.varname[i])
        # geosclassic.charp_array_setitem(im_c.varName, i, im_py.varname[i]) # does not work!

    v = im_py.val.reshape(-1)
    np.putmask(v, np.isnan(v), geosclassic.MPDS_MISSING_VALUE)
    geosclassic.mpds_set_real_vector_from_array(im_c.var, 0, v)
    np.putmask(v, v == geosclassic.MPDS_MISSING_VALUE, np.nan) # replace missing_value by np.nan (restore) (v is not a copy...)

    return im_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def img_C2py(im_c):
    """
    Converts an image from C to python.

    :param im_c:    (MPDS_IMAGE *) image (C struct)
    :return im_py:  (Img class) image converted (python class)
    """

    nxyz = im_c.grid.nx * im_c.grid.ny * im_c.grid.nz
    nxyzv = nxyz * im_c.nvar

    varname = [geosclassic.mpds_get_varname(im_c.varName, i) for i in range(im_c.nvar)]
    # varname = [geosclassic.charp_array_getitem(im_c.varName, i) for i in range(im_c.nvar)] # also works

    v = np.zeros(nxyzv)
    geosclassic.mpds_get_array_from_real_vector(im_c.var, 0, v)

    im_py = Img(nx=im_c.grid.nx, ny=im_c.grid.ny, nz=im_c.grid.nz,
                sx=im_c.grid.sx, sy=im_c.grid.sy, sz=im_c.grid.sz,
                ox=im_c.grid.ox, oy=im_c.grid.oy, oz=im_c.grid.oz,
                nv=im_c.nvar, val=v, varname=varname)

    np.putmask(im_py.val, im_py.val == geosclassic.MPDS_MISSING_VALUE, np.nan)

    return im_py
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def ps_py2C(ps_py):
    """
    Converts a point set from python to C.

    :param ps_py:   (PointSet class) point set (python class)
    :return ps_c:   (MPDS_POINTSET *) point set converted (C struct)
    """

    if ps_py.nv < 4:
        print ('ERROR: point set (python) have less than 4 variables')
        return

    nvar = ps_py.nv - 3

    ps_c = geosclassic.malloc_MPDS_POINTSET()
    geosclassic.MPDSInitPointSet(ps_c)

    err = geosclassic.MPDSMallocPointSet(ps_c, ps_py.npt, nvar)
    if err:
        print ('ERROR: can not convert point set from python to C')
        return

    ps_c.npoint = ps_py.npt
    ps_c.nvar = nvar

    for i in range(nvar):
        geosclassic.mpds_set_varname(ps_c.varName, i, ps_py.varname[i+3])

    geosclassic.mpds_set_real_vector_from_array(ps_c.x, 0, ps_py.val[0].reshape(-1))
    geosclassic.mpds_set_real_vector_from_array(ps_c.y, 0, ps_py.val[1].reshape(-1))
    geosclassic.mpds_set_real_vector_from_array(ps_c.z, 0, ps_py.val[2].reshape(-1))

    v = ps_py.val[3:].reshape(-1)
    np.putmask(v, np.isnan(v), geosclassic.MPDS_MISSING_VALUE)
    geosclassic.mpds_set_real_vector_from_array(ps_c.var, 0, v)
    np.putmask(v, v == geosclassic.MPDS_MISSING_VALUE, np.nan)  # replace missing_value by np.nan (restore) (v is not a copy...)

    return ps_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def ps_C2py(ps_c):
    """
    Converts a point set from C to python.

    :param ps_c:    (MPDS_POINTSET *) point set (C struct)
    :return ps_py:  (PointSet class) point set converted (python class)
    """

    varname = ['X', 'Y', 'Z'] + [geosclassic.mpds_get_varname(ps_c.varName, i) for i in range(ps_c.nvar)]

    v = np.zeros(ps_c.npoint*ps_c.nvar)
    geosclassic.mpds_get_array_from_real_vector(ps_c.var, 0, v)

    # coord = np.zeros(ps_c.npoint)
    # geosclassic.mpds_get_array_from_real_vector(ps_c.z, 0, coord)
    # v = np.hstack(coord,v)
    # geosclassic.mpds_get_array_from_real_vector(ps_c.y, 0, coord)
    # v = np.hstack(coord,v)
    # geosclassic.mpds_get_array_from_real_vector(ps_c.x, 0, coord)
    # v = np.hstack(coord,v)

    cx = np.zeros(ps_c.npoint)
    cy = np.zeros(ps_c.npoint)
    cz = np.zeros(ps_c.npoint)
    geosclassic.mpds_get_array_from_real_vector(ps_c.x, 0, cx)
    geosclassic.mpds_get_array_from_real_vector(ps_c.y, 0, cy)
    geosclassic.mpds_get_array_from_real_vector(ps_c.z, 0, cz)
    v = np.hstack((cx, cy, cz, v))

    ps_py = PointSet(npt=ps_c.npoint,
                     nv=ps_c.nvar+3, val=v, varname=varname)

    np.putmask(ps_py.val, ps_py.val == geosclassic.MPDS_MISSING_VALUE, np.nan)

    return ps_py
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel1Delem_py2C(covModelElem_py, nx, ny, nz, sx, sy, sz, ox, oy, oz):
    """
    Converts an elementary covariance model 1D from python to C.
    Simulation grid geometry is specified in case of non-stationary covariance
    model.

    :param covModelElem_py:
        (2-tuple) elementary covariance model 1D in python:
            (t, d) corresponds to an elementary model with:
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
                d: (dict) dictionary of required parameters to be passed to the
                    elementary model (value can be a "single value" or an array
                    that matches the dimension of the simulation grid (for
                    non-stationary covariance model)
            e.g.
                (t, d) = ('power', {w:2.0, r:1.5, s:1.7})
    :param nx, ny, nz:  (ints) number of simulation grid (SG) cells in each
                            direction
    :param sx, sy, sz:  (floats) cell size in each direction
    :param ox, oy, oz:  (floats) origin of the SG (bottom-lower-left corner)
    :return (covModelElem_c, flag):
        covModelElem_c: (MPDS_COVMODELELEM *) covariance model elem. converted
            (C struct)
        flag: (bool) indicating if the conversion has been done correctly (True)
            or not (False)
    """

    covModelElem_c = geosclassic.malloc_MPDS_COVMODELELEM()
    geosclassic.MPDSGeosClassicInitCovModelElem(covModelElem_c)

    w_flag = True   # weight to be set if True
    r_flag = True   # ranges to be set if True
    s_flag = False  # s (additional parameter) to be set if True

    # type
    if covModelElem_py[0] == 'nugget':
        covModelElem_c.covModelType = geosclassic.COV_NUGGET
        r_flag = False
    elif covModelElem_py[0] == 'spherical':
        covModelElem_c.covModelType = geosclassic.COV_SPHERICAL
    elif covModelElem_py[0] == 'exponential':
        covModelElem_c.covModelType = geosclassic.COV_EXPONENTIAL
    elif covModelElem_py[0] == 'gaussian':
        covModelElem_c.covModelType = geosclassic.COV_GAUSSIAN
    elif covModelElem_py[0] == 'linear':
        covModelElem_c.covModelType = geosclassic.COV_LINEAR
    elif covModelElem_py[0] == 'cubic':
        covModelElem_c.covModelType = geosclassic.COV_CUBIC
    elif covModelElem_py[0] == 'sinus_cardinal':
        covModelElem_c.covModelType = geosclassic.COV_SINUS_CARDINAL
    elif covModelElem_py[0] == 'gamma':
        covModelElem_c.covModelType = geosclassic.COV_GAMMA
        s_flag = True
        s_name = 's'
    elif covModelElem_py[0] == 'power':
        covModelElem_c.covModelType = geosclassic.COV_POWER
        s_flag = True
        s_name = 's'
    elif covModelElem_py[0] == 'exponential_generalized':
        covModelElem_c.covModelType = geosclassic.COV_EXPONENTIAL_GENERALIZED
        s_flag = True
        s_name = 's'
    elif covModelElem_py[0] == 'matern':
        covModelElem_c.covModelType = geosclassic.COV_MATERN
        s_flag = True
        s_name = 'nu'

    # weight
    if w_flag:
        param = covModelElem_py[1]['w']
        if np.size(param) == 1:
            covModelElem_c.weightImageFlag = geosclassic.FALSE
            covModelElem_c.weightValue = float(param)
        else:
            covModelElem_c.weightImageFlag = geosclassic.TRUE
            try:
                param = np.asarray(param).reshape(nz, ny, nx)
            except:
                print("ERROR: can not convert covModel1D from python to C ('w' not compatible with simulation grid)")
                return covModelElem_c, False
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param)
            covModelElem_c.weightImage = img_py2C(im)

    # ranges
    if r_flag:
        # ... range rx
        param = covModelElem_py[1]['r']
        if np.size(param) == 1:
            covModelElem_c.rxImageFlag = geosclassic.FALSE
            covModelElem_c.rxValue = float(param)
        else:
            covModelElem_c.rxImageFlag = geosclassic.TRUE
            try:
                param = np.asarray(param).reshape(nz, ny, nx)
            except:
                print("ERROR: can not convert covModel1D from python to C ('r(x)' not compatible with simulation grid)")
                return covModelElem_c, False
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param)
            covModelElem_c.rxImage = img_py2C(im)
        # ... range ry
        covModelElem_c.ryImageFlag = geosclassic.FALSE
        covModelElem_c.ryValue = 0.0
        # ... range rz
        covModelElem_c.rzImageFlag = geosclassic.FALSE
        covModelElem_c.rzValue = 0.0

    # s (additional parameter)
    if s_flag:
        param = covModelElem_py[1][s_name]
        if np.size(param) == 1:
            covModelElem_c.sImageFlag = geosclassic.FALSE
            covModelElem_c.sValue = float(param)
        else:
            covModelElem_c.sImageFlag = geosclassic.TRUE
            try:
                param = np.asarray(param).reshape(nz, ny, nx)
            except:
                print("ERROR: can not convert covModel1D from python to C ('s' not compatible with simulation grid)")
                return covModelElem_c, False
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param)
            covModelElem_c.sImage = img_py2C(im)

    return covModelElem_c, True
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel2Delem_py2C(covModelElem_py, nx, ny, nz, sx, sy, sz, ox, oy, oz):
    """
    Converts an elementary covariance model 2D from python to C.
    Simulation grid geometry is specified in case of non-stationary covariance
    model.

    :param covModelElem_py:
        (2-tuple) elementary covariance model 2D in python:
            (t, d) corresponds to an elementary model with:
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
                d: (dict) dictionary of required parameters to be passed to the
                    elementary model (value can be a "single value" or an array
                    that matches the dimension of the simulation grid (for
                    non-stationary covariance model)
            e.g.
                (t, d) = ('gaussian', {'w':10., 'r':[150, 50]})
    :param nx, ny, nz:  (ints) number of simulation grid (SG) cells in each
                            direction
    :param sx, sy, sz:  (floats) cell size in each direction
    :param ox, oy, oz:  (floats) origin of the SG (bottom-lower-left corner)
    :return (covModelElem_c, flag):
        covModelElem_c: (MPDS_COVMODELELEM *) covariance model elem. converted
            (C struct)
        flag: (bool) indicating if the conversion has been done correctly (True)
            or not (False)
    """

    covModelElem_c = geosclassic.malloc_MPDS_COVMODELELEM()
    geosclassic.MPDSGeosClassicInitCovModelElem(covModelElem_c)

    w_flag = True   # weight to be set if True
    r_flag = True   # ranges to be set if True
    s_flag = False  # s (additional parameter) to be set if True

    # type
    if covModelElem_py[0] == 'nugget':
        covModelElem_c.covModelType = geosclassic.COV_NUGGET
        r_flag = False
    elif covModelElem_py[0] == 'spherical':
        covModelElem_c.covModelType = geosclassic.COV_SPHERICAL
    elif covModelElem_py[0] == 'exponential':
        covModelElem_c.covModelType = geosclassic.COV_EXPONENTIAL
    elif covModelElem_py[0] == 'gaussian':
        covModelElem_c.covModelType = geosclassic.COV_GAUSSIAN
    elif covModelElem_py[0] == 'linear':
        covModelElem_c.covModelType = geosclassic.COV_LINEAR
    elif covModelElem_py[0] == 'cubic':
        covModelElem_c.covModelType = geosclassic.COV_CUBIC
    elif covModelElem_py[0] == 'sinus_cardinal':
        covModelElem_c.covModelType = geosclassic.COV_SINUS_CARDINAL
    elif covModelElem_py[0] == 'gamma':
        covModelElem_c.covModelType = geosclassic.COV_GAMMA
        s_flag = True
        s_name = 's'
    elif covModelElem_py[0] == 'power':
        covModelElem_c.covModelType = geosclassic.COV_POWER
        s_flag = True
        s_name = 's'
    elif covModelElem_py[0] == 'exponential_generalized':
        covModelElem_c.covModelType = geosclassic.COV_EXPONENTIAL_GENERALIZED
        s_flag = True
        s_name = 's'
    elif covModelElem_py[0] == 'matern':
        covModelElem_c.covModelType = geosclassic.COV_MATERN
        s_flag = True
        s_name = 'nu'

    # weight
    if w_flag:
        param = covModelElem_py[1]['w']
        if np.size(param) == 1:
            covModelElem_c.weightImageFlag = geosclassic.FALSE
            covModelElem_c.weightValue = float(param)
        else:
            covModelElem_c.weightImageFlag = geosclassic.TRUE
            try:
                param = np.asarray(param).reshape(nz, ny, nx)
            except:
                print("ERROR: can not convert covModel2D from python to C ('w' not compatible with simulation grid)")
                return covModelElem_c, False
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param)
            covModelElem_c.weightImage = img_py2C(im)

    # ranges
    if r_flag:
        # ... range rx
        param = covModelElem_py[1]['r'][0]
        if np.size(param) == 1:
            covModelElem_c.rxImageFlag = geosclassic.FALSE
            covModelElem_c.rxValue = float(param)
        else:
            covModelElem_c.rxImageFlag = geosclassic.TRUE
            try:
                param = np.asarray(param).reshape(nz, ny, nx)
            except:
                print("ERROR: can not convert covModel2D from python to C ('r(x)' not compatible with simulation grid)")
                return covModelElem_c, False
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param)
            covModelElem_c.rxImage = img_py2C(im)
        # ... range ry
        param = covModelElem_py[1]['r'][1]
        if np.size(param) == 1:
            covModelElem_c.ryImageFlag = geosclassic.FALSE
            covModelElem_c.ryValue = float(param)
        else:
            covModelElem_c.ryImageFlag = geosclassic.TRUE
            try:
                param = np.asarray(param).reshape(nz, ny, nx)
            except:
                print("ERROR: can not convert covModel2D from python to C ('r(y)' not compatible with simulation grid)")
                return covModelElem_c, False
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param)
            covModelElem_c.ryImage = img_py2C(im)
        # ... range rz
        covModelElem_c.rzImageFlag = geosclassic.FALSE
        covModelElem_c.rzValue = 0.0

    # s (additional parameter)
    if s_flag:
        param = covModelElem_py[1][s_name]
        if np.size(param) == 1:
            covModelElem_c.sImageFlag = geosclassic.FALSE
            covModelElem_c.sValue = float(param)
        else:
            covModelElem_c.sImageFlag = geosclassic.TRUE
            try:
                param = np.asarray(param).reshape(nz, ny, nx)
            except:
                print("ERROR: can not convert covModel2D from python to C ('s' not compatible with simulation grid)")
                return covModelElem_c, False
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param)
            covModelElem_c.sImage = img_py2C(im)

    return covModelElem_c, True
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel3Delem_py2C(covModelElem_py, nx, ny, nz, sx, sy, sz, ox, oy, oz):
    """
    Converts an elementary covariance model 3D from python to C.
    Simulation grid geometry is specified in case of non-stationary covariance model.

    :param covModelElem_py:
        (2-tuple) elementary covariance model 3D in python:
            (t, d) corresponds to an elementary model with:
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
                d: (dict) dictionary of required parameters to be passed to the
                    elementary model (value can be a "single value" or an array
                    that matches the dimension of the simulation grid (for
                    non-stationary covariance model)
            e.g.
                (t, d) = ('power', {w:2.0, r:[1.5, 2.5, 3.0], s:1.7})
    :param nx, ny, nz:  (ints) number of simulation grid (SG) cells in each
                            direction
    :param sx, sy, sz:  (floats) cell size in each direction
    :param ox, oy, oz:  (floats) origin of the SG (bottom-lower-left corner)
    :return (covModelElem_c, flag):
        covModelElem_c: (MPDS_COVMODELELEM *) covariance model elem. converted
            (C struct)
        flag: (bool) indicating if the conversion has been done correctly (True)
            or not (False)
    """

    covModelElem_c = geosclassic.malloc_MPDS_COVMODELELEM()
    geosclassic.MPDSGeosClassicInitCovModelElem(covModelElem_c)

    w_flag = True   # weight to be set if True
    r_flag = True   # ranges to be set if True
    s_flag = False  # s (additional parameter) to be set if True

    # type
    if covModelElem_py[0] == 'nugget':
        covModelElem_c.covModelType = geosclassic.COV_NUGGET
        r_flag = False
    elif covModelElem_py[0] == 'spherical':
        covModelElem_c.covModelType = geosclassic.COV_SPHERICAL
    elif covModelElem_py[0] == 'exponential':
        covModelElem_c.covModelType = geosclassic.COV_EXPONENTIAL
    elif covModelElem_py[0] == 'gaussian':
        covModelElem_c.covModelType = geosclassic.COV_GAUSSIAN
    elif covModelElem_py[0] == 'linear':
        covModelElem_c.covModelType = geosclassic.COV_LINEAR
    elif covModelElem_py[0] == 'cubic':
        covModelElem_c.covModelType = geosclassic.COV_CUBIC
    elif covModelElem_py[0] == 'sinus_cardinal':
        covModelElem_c.covModelType = geosclassic.COV_SINUS_CARDINAL
    elif covModelElem_py[0] == 'gamma':
        covModelElem_c.covModelType = geosclassic.COV_GAMMA
        s_flag = True
        s_name = 's'
    elif covModelElem_py[0] == 'power':
        covModelElem_c.covModelType = geosclassic.COV_POWER
        s_flag = True
        s_name = 's'
    elif covModelElem_py[0] == 'exponential_generalized':
        covModelElem_c.covModelType = geosclassic.COV_EXPONENTIAL_GENERALIZED
        s_flag = True
        s_name = 's'
    elif covModelElem_py[0] == 'matern':
        covModelElem_c.covModelType = geosclassic.COV_MATERN
        s_flag = True
        s_name = 'nu'

    # weight
    if w_flag:
        param = covModelElem_py[1]['w']
        if np.size(param) == 1:
            covModelElem_c.weightImageFlag = geosclassic.FALSE
            covModelElem_c.weightValue = float(param)
        else:
            covModelElem_c.weightImageFlag = geosclassic.TRUE
            try:
                param = np.asarray(param).reshape(nz, ny, nx)
            except:
                print("ERROR: can not convert covModel3D from python to C ('w' not compatible with simulation grid)")
                return covModelElem_c, False
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param)
            covModelElem_c.weightImage = img_py2C(im)

    # ranges
    if r_flag:
        # ... range rx
        param = covModelElem_py[1]['r'][0]
        if np.size(param) == 1:
            covModelElem_c.rxImageFlag = geosclassic.FALSE
            covModelElem_c.rxValue = float(param)
        else:
            covModelElem_c.rxImageFlag = geosclassic.TRUE
            try:
                param = np.asarray(param).reshape(nz, ny, nx)
            except:
                print("ERROR: can not convert covModel3D from python to C ('r(x)' not compatible with simulation grid)")
                return covModelElem_c, False
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param)
            covModelElem_c.rxImage = img_py2C(im)
        # ... range ry
        param = covModelElem_py[1]['r'][1]
        if np.size(param) == 1:
            covModelElem_c.ryImageFlag = geosclassic.FALSE
            covModelElem_c.ryValue = float(param)
        else:
            covModelElem_c.ryImageFlag = geosclassic.TRUE
            try:
                param = np.asarray(param).reshape(nz, ny, nx)
            except:
                print("ERROR: can not convert covModel3D from python to C ('r(y)' not compatible with simulation grid)")
                return covModelElem_c, False
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param)
            covModelElem_c.ryImage = img_py2C(im)
        # ... range rz
        param = covModelElem_py[1]['r'][2]
        if np.size(param) == 1:
            covModelElem_c.rzImageFlag = geosclassic.FALSE
            covModelElem_c.rzValue = float(param)
        else:
            covModelElem_c.rzImageFlag = geosclassic.TRUE
            try:
                param = np.asarray(param).reshape(nz, ny, nx)
            except:
                print("ERROR: can not convert covModel3D from python to C ('r(z)' not compatible with simulation grid)")
                return covModelElem_c, False
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param)
            covModelElem_c.rzImage = img_py2C(im)

    # s (additional parameter)
    if s_flag:
        param = covModelElem_py[1][s_name]
        if np.size(param) == 1:
            covModelElem_c.sImageFlag = geosclassic.FALSE
            covModelElem_c.sValue = float(param)
        else:
            covModelElem_c.sImageFlag = geosclassic.TRUE
            try:
                param = np.asarray(param).reshape(nz, ny, nx)
            except:
                print("ERROR: can not convert covModel3D from python to C ('s' not compatible with simulation grid)")
                return covModelElem_c, False
            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param)
            covModelElem_c.sImage = img_py2C(im)

    return covModelElem_c, True
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel1D_py2C(covModel_py, nx, ny, nz, sx, sy, sz, ox, oy, oz):
    """
    Converts a covariance model 1D from python to C.
    Simulation grid geometry is specified in case of non-stationary covariance
    model.

    :param covModel_py: (CovModel1D class) covariance model 1D (python class)
    :param nx, ny, nz : (ints) number of simulation grid (SG) cells in each
                            direction
    :param sx, sy, sz : (floats) cell size in each direction
    :param ox, oy, oz : (floats) origin of the SG (bottom-lower-left corner)
    :return (covModel_c, flag):
        covModel_c: (MPDS_COVMODEL *) covariance model converted (C struct)
        flag: (bool) indicating if the conversion has been done correctly (True)
            or not (False)
    """

    covModel_c = geosclassic.malloc_MPDS_COVMODEL()
    geosclassic.MPDSGeosClassicInitCovModel(covModel_c)

    n = len(covModel_py.elem)
    covModel_c.nelem = n
    covModel_c.covModelElem = geosclassic.new_MPDS_COVMODELELEM_array(n)
    for i, covModelElem in enumerate(covModel_py.elem):
        covModelElem_c, flag = covModel1Delem_py2C(covModelElem, nx, ny, nz, sx, sy, sz, ox, oy, oz)
        if flag:
            geosclassic.MPDS_COVMODELELEM_array_setitem(covModel_c.covModelElem, i, covModelElem_c)
        # geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
        if not flag:
            return covModel_c, False

    # covModel_c.angle1, covModel_c.angle2, covModel_c.angle3: keep to 0.0
    covModel_c.angle1 = 0.0
    covModel_c.angle2 = 0.0
    covModel_c.angle3 = 0.0

    return covModel_c, True
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel2D_py2C(covModel_py, nx, ny, nz, sx, sy, sz, ox, oy, oz):
    """
    Converts a covariance model 2D from python to C.
    Simulation grid geometry is specified in case of non-stationary covariance
    model.

    :param covModel_py: (CovModel2D class) covariance model 2D (python class)
    :param nx, ny, nz : (ints) number of simulation grid (SG) cells in each
                            direction
    :param sx, sy, sz : (floats) cell size in each direction
    :param ox, oy, oz : (floats) origin of the SG (bottom-lower-left corner)
    :return (covModel_c, flag):
        covModel_c: (MPDS_COVMODEL *) covariance model converted (C struct)
        flag: (bool) indicating if the conversion has been done correctly (True)
            or not (False)
    """

    covModel_c = geosclassic.malloc_MPDS_COVMODEL()
    geosclassic.MPDSGeosClassicInitCovModel(covModel_c)

    n = len(covModel_py.elem)
    covModel_c.nelem = n
    covModel_c.covModelElem = geosclassic.new_MPDS_COVMODELELEM_array(n)
    for i, covModelElem in enumerate(covModel_py.elem):
        covModelElem_c, flag = covModel2Delem_py2C(covModelElem, nx, ny, nz, sx, sy, sz, ox, oy, oz)
        if flag:
            geosclassic.MPDS_COVMODELELEM_array_setitem(covModel_c.covModelElem, i, covModelElem_c)
        # geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
        if not flag:
            return covModel_c, False

    # covModel_c.angle2, covModel_c.angle3: keep to 0.0
    # angle1
    param = covModel_py.alpha
    if np.size(param) == 1:
        covModel_c.angle1ImageFlag = geosclassic.FALSE
        covModel_c.angle1Value = float(param)
    else:
        covModel_c.angle1ImageFlag = geosclassic.TRUE
        try:
            param = np.asarray(param).reshape(nz, ny, nx)
        except:
            print("ERROR: can not convert covModel2D from python to C ('alpha' not compatible with simulation grid)")
            return covModel_c, False
        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=1, val=param)
        covModel_c.angle1Image = img_py2C(im)
    # angle2
    covModel_c.angle2 = 0.0
    # angle3
    covModel_c.angle3 = 0.0

    return covModel_c, True
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel3D_py2C(covModel_py, nx, ny, nz, sx, sy, sz, ox, oy, oz):
    """
    Converts a covariance model 3D from python to C.
    Simulation grid geometry is specified in case of non-stationary covariance
    model.

    :param covModel_py: (CovModel3D class) covariance model 3D (python class)
    :param nx, ny, nz : (ints) number of simulation grid (SG) cells in each
                            direction
    :param sx, sy, sz : (floats) cell size in each direction
    :param ox, oy, oz : (floats) origin of the SG (bottom-lower-left corner)
    :return (covModel_c, flag):
        covModel_c: (MPDS_COVMODEL *) covariance model converted (C struct)
        flag: (bool) indicating if the conversion has been done correctly (True)
            or not (False)
    """

    covModel_c = geosclassic.malloc_MPDS_COVMODEL()
    geosclassic.MPDSGeosClassicInitCovModel(covModel_c)

    n = len(covModel_py.elem)
    covModel_c.nelem = n
    covModel_c.covModelElem = geosclassic.new_MPDS_COVMODELELEM_array(n)
    for i, covModelElem in enumerate(covModel_py.elem):
        covModelElem_c, flag = covModel3Delem_py2C(covModelElem, nx, ny, nz, sx, sy, sz, ox, oy, oz)
        if flag:
            geosclassic.MPDS_COVMODELELEM_array_setitem(covModel_c.covModelElem, i, covModelElem_c)
        # geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
        if not flag:
            return covModel_c, False

    # angle1
    param = covModel_py.alpha
    if np.size(param) == 1:
        covModel_c.angle1ImageFlag = geosclassic.FALSE
        covModel_c.angle1Value = float(param)
    else:
        covModel_c.angle1ImageFlag = geosclassic.TRUE
        try:
            param = np.asarray(param).reshape(nz, ny, nx)
        except:
            print("ERROR: can not convert covModel3D from python to C ('alpha' not compatible with simulation grid)")
            return covModel_c, False
        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=1, val=param)
        covModel_c.angle1Image = img_py2C(im)
    # angle2
    param = covModel_py.beta
    if np.size(param) == 1:
        covModel_c.angle2ImageFlag = geosclassic.FALSE
        covModel_c.angle2Value = float(param)
    else:
        covModel_c.angle2ImageFlag = geosclassic.TRUE
        try:
            param = np.asarray(param).reshape(nz, ny, nx)
        except:
            print("ERROR: can not convert covModel3D from python to C ('beta' not compatible with simulation grid)")
            return covModel_c, False
        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=1, val=param)
        covModel_c.angle2Image = img_py2C(im)
    # angle3
    param = covModel_py.gamma
    if np.size(param) == 1:
        covModel_c.angle3ImageFlag = geosclassic.FALSE
        covModel_c.angle3Value = float(param)
    else:
        covModel_c.angle3ImageFlag = geosclassic.TRUE
        try:
            param = np.asarray(param).reshape(nz, ny, nx)
        except:
            print("ERROR: can not convert covModel3D from python to C ('gamma' not compatible with simulation grid)")
            return covModel_c, False
        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=1, val=param)
        covModel_c.angle3Image = img_py2C(im)

    return covModel_c, True
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor):
    """
    Get geosclassic output for python from C.

    :param mpds_geosClassicOutput:
                            (MPDS_GEOSCLASSICOUTPUT *) output - (C struct)
    :param mpds_progressMonitor:
                            (MPDS_PROGRESSMONITOR *) progress monitor - (C struct)

    :return geosclassic_output:
        (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}

        image:  (Img class) output image, with image.nv variables (simulation
                    or estimates and standard deviation)
                    (image is None if mpds_geosClassicOutput->outputImage is
                    NULL)
        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)
        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    # Initialization
    image = None
    nwarning, warnings = None, None

    image = img_C2py(mpds_geosClassicOutput.outputImage)

    nwarning = mpds_progressMonitor.nwarning
    warnings = []
    if mpds_progressMonitor.nwarningNumber:
        tmp = np.zeros(mpds_progressMonitor.nwarningNumber, dtype='intc') # 'intc' for C-compatibility
        geosclassic.mpds_get_array_from_int_vector(mpds_progressMonitor.warningNumberList, 0, tmp)
        warningNumberList = np.asarray(tmp, dtype='int') # 'int' or equivalently 'int64'
        for iwarn in warningNumberList:
            warning_message = geosclassic.mpds_get_warning_message(int(iwarn)) # int() required!
            warning_message = warning_message.replace('\n', '')
            warnings.append(warning_message)

    return {'image':image, 'nwarning':nwarning, 'warnings':warnings}
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def fill_mpds_geosClassicInput(
        space_dim,
        cov_model,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        varname,
        outputReportFile,
        computationMode,
        dataImage,
        dataPointSet,
        mask,
        mean,
        var,
        searchRadiusRelative,
        nneighborMax,
        searchNeighborhoodSortMode,
        nGibbsSamplerPathMin,
        nGibbsSamplerPathMax,
        seed,
        nreal):
    """
    Fills a mpds_geosClassicInput C structure from given parameters.

    :return (mpds_geosClassicInput, flag):
        mpds_geosClassicInput: C structure for "GeosClassicSim" program (C)
        flag: (bool) indicating if the filling has been done correctly (True)
            or not (False)
    """

    nxy = nx * ny
    nxyz = nxy * nz

    # Allocate mpds_geosClassicInput
    mpds_geosClassicInput = geosclassic.malloc_MPDS_GEOSCLASSICINPUT()

    # Init mpds_geosClassicInput
    geosclassic.MPDSGeosClassicInitGeosClassicInput(mpds_geosClassicInput)

    # mpds_geosClassicInput.consoleAppFlag
    mpds_geosClassicInput.consoleAppFlag = geosclassic.FALSE

    # mpds_geosClassicInput.simGrid
    mpds_geosClassicInput.simGrid = geosclassic.malloc_MPDS_GRID()

    mpds_geosClassicInput.simGrid.nx = int(nx)
    mpds_geosClassicInput.simGrid.ny = int(ny)
    mpds_geosClassicInput.simGrid.nz = int(nz)

    mpds_geosClassicInput.simGrid.sx = float(sx)
    mpds_geosClassicInput.simGrid.sy = float(sy)
    mpds_geosClassicInput.simGrid.sz = float(sz)

    mpds_geosClassicInput.simGrid.ox = float(ox)
    mpds_geosClassicInput.simGrid.oy = float(oy)
    mpds_geosClassicInput.simGrid.oz = float(oz)

    mpds_geosClassicInput.simGrid.nxy = nxy
    mpds_geosClassicInput.simGrid.nxyz = nxyz

    # mpds_geosClassicInput.varname
    geosclassic.mpds_allocate_and_set_geosClassicInput_varname(mpds_geosClassicInput, varname)

    # mpds_geosClassicInput.outputMode
    mpds_geosClassicInput.outputMode = geosclassic.GEOS_CLASSIC_OUTPUT_NO_FILE

    # mpds_geosClassicInput.outputReportFlag and mpds_geosClassicInput.outputReportFileName
    if outputReportFile is not None:
        mpds_geosClassicInput.outputReportFlag = geosclassic.TRUE
        geosclassic.mpds_allocate_and_set_geosClassicInput_outputReportFileName(mpds_geosClassicInput, outputReportFile)
    else:
        mpds_geosClassicInput.outputReportFlag = geosclassic.FALSE

    # mpds_geosClassicInput.computationMode
    mpds_geosClassicInput.computationMode = int(computationMode)

    # mpds_geosClassicInput.covModel
    if space_dim==1:
        cov_model_c, flag = covModel1D_py2C(cov_model, nx, ny, nz, sx, sy, sz, ox, oy, oz)
    elif space_dim==2:
        cov_model_c, flag = covModel2D_py2C(cov_model, nx, ny, nz, sx, sy, sz, ox, oy, oz)
    elif space_dim==3:
        cov_model_c, flag = covModel3D_py2C(cov_model, nx, ny, nz, sx, sy, sz, ox, oy, oz)

    if flag:
        mpds_geosClassicInput.covModel = cov_model_c
    else:
        return mpds_geosClassicInput, False

    # mpds_geosClassicInput.searchRadiusRelative
    mpds_geosClassicInput.searchRadiusRelative = float(searchRadiusRelative)

    # mpds_geosClassicInput.nneighborMax
    mpds_geosClassicInput.nneighborMax = int(nneighborMax)

    # mpds_geosClassicInput.searchNeighborhoodSortMode
    mpds_geosClassicInput.searchNeighborhoodSortMode = int(searchNeighborhoodSortMode)

    # mpds_geosClassicInput.ndataImage and mpds_geosClassicInput.dataImage
    if dataImage is None:
        mpds_geosClassicInput.ndataImage = 0
    else:
        dataImage = np.asarray(dataImage).reshape(-1)
        n = len(dataImage)
        mpds_geosClassicInput.ndataImage = n
        mpds_geosClassicInput.dataImage = geosclassic.new_MPDS_IMAGE_array(n)
        for i, dataIm in enumerate(dataImage):
            im_c = img_py2C(dataIm)
            geosclassic.MPDS_IMAGE_array_setitem(mpds_geosClassicInput.dataImage, i, im_c)
            geosclassic.free_MPDS_IMAGE(im_c)
            # geosclassic.MPDS_IMAGE_array_setitem(mpds_geosClassicInput.dataImage, i, img_py2C(dataIm))

    # mpds_geosClassicInput.ndataPointSet and mpds_geosClassicInput.dataPointSet
    if dataPointSet is None:
        mpds_geosClassicInput.ndataPointSet = 0
    else:
        dataPointSet = np.asarray(dataPointSet).reshape(-1)
        n = len(dataPointSet)
        mpds_geosClassicInput.ndataPointSet = n
        mpds_geosClassicInput.dataPointSet = geosclassic.new_MPDS_POINTSET_array(n)
        for i, dataPS in enumerate(dataPointSet):
            ps_c = ps_py2C(dataPS)
            geosclassic.MPDS_POINTSET_array_setitem(mpds_geosClassicInput.dataPointSet, i, ps_c)
            # geosclassic.free_MPDS_POINTSET(ps_c)
            #
            # geosclassic.MPDS_POINTSET_array_setitem(mpds_geosClassicInput.dataPointSet, i, ps_py2C(dataPS))

    # mpds_geosClassicInput.maskImageFlag and mpds_geosClassicInput.maskImage
    if mask is None:
        mpds_geosClassicInput.maskImageFlag = geosclassic.FALSE
    else:
        mpds_geosClassicInput.maskImageFlag = geosclassic.TRUE
        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=1, val=mask)
        mpds_geosClassicInput.maskImage = img_py2C(im)

    # mpds_geosClassicInput.meanUsage, mpds_geosClassicInput.meanValue, mpds_geosClassicInput.meanImage
    if mean is None:
        mpds_geosClassicInput.meanUsage = 0
    elif mean.size == 1:
        mpds_geosClassicInput.meanUsage = 1
        mpds_geosClassicInput.meanValue = float(mean[0])
    elif mean.size == nxyz:
        mpds_geosClassicInput.meanUsage = 2
        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=1, val=mean)
        mpds_geosClassicInput.meanImage = img_py2C(im)
    else:
        # print("ERROR: can not integrate 'mean' (not compatible with simulation grid)")
        return mpds_geosClassicInput, False

    # mpds_geosClassicInput.varianceUsage, mpds_geosClassicInput.varianceValue, mpds_geosClassicInput.varianceImage
    if var is None:
        mpds_geosClassicInput.varianceUsage = 0
    elif var.size == 1:
        mpds_geosClassicInput.varianceUsage = 1
        mpds_geosClassicInput.varianceValue = var[0]
    elif var.size == nxyz:
        mpds_geosClassicInput.varianceUsage = 2
        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=1, val=var)
        mpds_geosClassicInput.varianceImage = img_py2C(im)
    else:
        # print("ERROR: can not integrate 'var' (not compatible with simulation grid)")
        return mpds_geosClassicInput, False

    # mpds_geosClassicInput.nGibbsSamplerPathMin
    mpds_geosClassicInput.nGibbsSamplerPathMin = int(nGibbsSamplerPathMin)

    # mpds_geosClassicInput.nGibbsSamplerPathMax
    mpds_geosClassicInput.nGibbsSamplerPathMax = int(nGibbsSamplerPathMax)

    # mpds_geosClassicInput.seed
    if seed is None:
        seed = np.random.randint(1,1000000)
    mpds_geosClassicInput.seed = int(seed)

    # mpds_geosClassicInput.seedIncrement
    mpds_geosClassicInput.seedIncrement = 1

    # mpds_geosClassicInput.nrealization
    mpds_geosClassicInput.nrealization = int(nreal)

    return mpds_geosClassicInput, True
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def simulate1D(
        cov_model,
        dimension, spacing=1.0, origin=0.0,
        method='simple_kriging',
        nreal=1,
        mean=None, var=None,
        x=None, v=None,
        xIneqMin=None, vIneqMin=None,
        xIneqMax=None, vIneqMax=None,
        mask=None,
        searchRadiusRelative=1.0,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        nGibbsSamplerPathMin=50,
        nGibbsSamplerPathMax=200,
        seed=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2):
    """
    Generates 1D simulations (Sequential Gaussian Simulation, SGS) based on
    simple or ordinary kriging.

    :param cov_model:   covariance model:
                            (CovModel1D class) covariance model in 1D, see
                                definition of the class in module geone.covModel

    :param dimension:   (int) nx, number of cells
    :param spacing:     (float) sx, spacing between two adjacent cells
    :param origin:      (float) ox, origin of the 1D simulation
                            - used for localizing the conditioning points
    :param method:      (string) indicates the method used:
                            - 'simple_kriging':
                                simulation based on simple kriging
                            - 'ordinary_kriging':
                                simulation based on ordinary kriging
    :param nreal:       (int) number of realizations

    :param mean:        (None or callable (function) or float or ndarray) mean of
                            the simulation:
                            - None   : mean of hard data values (stationary),
                                       (0 if no hard data)
                            - callable (function):
                                       function of one argument (xi) that returns
                                       the mean at xi (in the grid)
                            - float  : for stationary mean (set manually)
                            - ndarray: for non stationary mean, must contain
                                       as many entries as number of grid cells
                                       (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param var:         (None or callable (function) or float or ndarray) variance
                            of the simulation (for simple kriging only):
                            - None   : variance not modified
                                       (only covariance model is used)
                            - callable (function):
                                       function of one argument (xi) that returns
                                       the variance at xi (in the grid)
                            - float  : for stationary variance (set manually)
                            - ndarray: for non stationary variance, must contain
                                       as many entries as number of grid cells
                                       (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            this parameter must be None (only covariance model
                            is used)

    :param x:           (1-dimensional array or float or None) coordinate of
                            conditioning points for hard data
    :param v:           (1-dimensional array or float or None) value
                            at conditioning points for hard data
                            (same type as x)

    :param xIneqMin:    (1-dimensional array or float or None) coordinate of
                            conditioning points for inequality data minimal bound
    :param vIneqMin:    (1-dimensional array or float or None) value
                            at conditioning points for inequality data minimal
                            bound (same type as xIneqMin)

    :param xIneqMax:    (1-dimensional array or float or None) coordinate of
                            conditioning points for inequality data maximal bound
    :param vIneqMax:    (1-dimensional array or float or None) value
                            at conditioning points for inequality data maximal
                            bound (same type as xIneqMax)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not
                            simulated cell (nunber of entries should be equal to
                            the number of grid cells)

    :param searchRadiusRelative:
                        (float) indicating how restricting the search ellipsoid
                            (should be positive): let r_i be the ranges of the
                            covariance model along its main axes, if x is a node
                            to be simulated, a node y is taken into account iff
                            it is within the ellipsoid centered at x of half-axes
                            searchRadiusRelative * r_i
                            Note:
                                - if a range is a variable parameter, its maximal
                                value over the simulation grid is considered

    :param nneighborMax:
                        (int) maximum number of nodes retrieved from the search
                            ellipsoid, set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (int) indicating how to sort the search neighboorhood
                            nodes (neighbors), they are sorted in increasing
                            order according to:
                            - searchNeighborhoodSortMode = 0:
                                distance in the usual axes system
                            - searchNeighborhoodSortMode = 1:
                                distance in the axes sytem supporting the
                                covariance model and accounting for anisotropy
                                given by the ranges
                            - searchNeighborhoodSortMode = 2:
                                minus the evaluation of the covariance model
                            Notes:
                            - if the covariance model has any variable parameter
                                (non-stationary), then
                                searchNeighborhoodSortMode = 2 is not allowed
                            - if the covariance model has any range or angle set
                                as a variable parameter, then
                                searchNeighborhoodSortMode must be set to 0
                            - greatest possible value as default

    :param nGibbsSamplerPathMin, nGibbsSamplerPathMax:
                        (int) minimal and maximal number of Gibbs sampler paths
                            to deal with inequality data; the conditioning
                            locations with inequality data are first simulated
                            (with truncated gaussian distribution) sequentially;
                            then, these locations are re-simulated following a
                            new path as many times as needed; the total number
                            of paths will be between nGibbsSamplerPathMin and
                            nGibbsSamplerPathMax

    :param seed:        (int or None) initial seed, if None an initial seed
                            between 1 and 999999 is generated with
                            numpy.random.randint

    :param outputReportFile:
                        (string or None) name of the report file,
                            if None: no report file

    :param nthreads:
                        (int) number of thread(s) to use for "GeosClassicSim"
                            program (C), (nthreads = -n <= 0: for maximal number
                            of threads except n, but at least 1)
    :param verbose:
                (int) verbose mode, integer >=0, higher implies more display
                    - 0: no display
                    - 1: only errors
                    - 2: errors and warnings (+ some info)
                    - 3 (or >2): all information

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img class) output image, with image.nv=nreal variables (each
                    variable is one realization)
                    (image is None if mpds_geosClassicOutput->outputImage is
                    NULL)
        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)
        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    # --- Set grid geometry and varname
    # Set grid geometry
    nx, ny, nz = dimension, 1, 1
    sx, sy, sz = spacing, 1.0, 1.0
    ox, oy, oz = origin, 0.0, 0.0

    nxy = nx * ny
    nxyz = nxy * nz

    # spatial dimension
    space_dim = 1

    # Set varname
    varname = 'V0'

    # --- Check and prepare parameters
    # cov_model
    if not isinstance(cov_model, gcm.CovModel1D):
        if verbose > 0:
            print("ERROR (SIMULATE1D): 'cov_model' (first argument) is not valid")
        return None

    for el in cov_model.elem:
        # weight
        w = el[1]['w']
        if np.size(w) != 1 and np.size(w) != nxyz:
            if verbose > 0:
                print("ERROR (SIMULATE1D): 'cov_model': weight ('w') not compatible with simulation grid")
            return None
        # ranges
        if 'r' in el[1].keys():
            r  = el[1]['r']
            if np.size(r) != 1 and np.size(r) != nxyz:
                if verbose > 0:
                    print("ERROR (SIMULATE1D): 'cov_model': range ('r') not compatible with simulation grid")
                return None
        # additional parameter (s)
        if 's' in el[1].keys():
            s  = el[1]['s']
            if np.size(s) != 1 and np.size(s) != nxyz:
                if verbose > 0:
                    print("ERROR (SIMULATE1D): 'cov_model': parameter ('s') not compatible with simulation grid")
                return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     if verbose > 0:
    #         print("ERROR (SIMULATE1D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 3
    elif method == 'ordinary_kriging':
        computationMode = 2
    else:
        if verbose > 0:
            print("ERROR (SIMULATE1D): 'method' is not valid")
        return None

    # data points: x, v, xIneqMin, vIneqMin, xIneqMax, vIneqMax
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            if verbose > 0:
                print("(ERROR (SIMULATE1D): length of 'v' is not valid")
            return None
        xc = x
        yc = np.ones_like(xc) * oy + 0.5 * sy
        zc = np.ones_like(xc) * oz + 0.5 * sz
        dataPointSet.append(
            PointSet(npt=v.shape[0], nv=4, val=np.array((xc, yc, zc, v)), varname=['X', 'Y', 'Z', varname])
            )

    # data point set from xIneqMin, vIneqMin
    if xIneqMin is not None:
        xIneqMin = np.asarray(xIneqMin, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        vIneqMin = np.asarray(vIneqMin, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(vIneqMin) != xIneqMin.shape[0]:
            if verbose > 0:
                print("(ERROR (SIMULATE1D): length of 'vIneqMin' is not valid")
            return None
        xc = xIneqMin
        yc = np.ones_like(xc) * oy + 0.5 * sy
        zc = np.ones_like(xc) * oz + 0.5 * sz
        dataPointSet.append(
            PointSet(npt=vIneqMin.shape[0], nv=4, val=np.array((xc, yc, zc, vIneqMin)), varname=['X', 'Y', 'Z', '{}_min'.format(varname)])
            )

    # data point set from xIneqMax, vIneqMax
    if xIneqMax is not None:
        xIneqMax = np.asarray(xIneqMax, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        vIneqMax = np.asarray(vIneqMax, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(vIneqMax) != xIneqMax.shape[0]:
            if verbose > 0:
                print("(ERROR (SIMULATE1D): length of 'vIneqMax' is not valid")
            return None
        xc = xIneqMax
        yc = np.ones_like(xc) * oy + 0.5 * sy
        zc = np.ones_like(xc) * oz + 0.5 * sz
        dataPointSet.append(
            PointSet(npt=vIneqMax.shape[0], nv=4, val=np.array((xc, yc, zc, vIneqMax)), varname=['X', 'Y', 'Z', '{}_max'.format(varname)])
            )

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            if verbose > 0:
                print("ERROR (SIMULATE1D): 'mask' is not valid")
            return None

    # Check parameters - searchRadiusRelative
    if searchRadiusRelative < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
        if verbose > 0:
            print("ERROR (SIMULATE1D): 'searchRadiusRelative' too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
        return None

    # Check parameters - nneighborMax
    if nneighborMax != -1 and nneighborMax <= 0:
        if verbose > 0:
            print("ERROR (SIMULATE1D): 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
        return None

    # Check parameters - searchNeighborhoodSortMode
    if searchNeighborhoodSortMode is None:
        # set greatest possible value
        if cov_model.is_stationary():
            searchNeighborhoodSortMode = 2
        elif cov_model.is_orientation_stationary() and cov_model.is_range_stationary():
            searchNeighborhoodSortMode = 1
        else:
            searchNeighborhoodSortMode = 0
    else:
        if searchNeighborhoodSortMode == 2:
            if not cov_model.is_stationary():
                if verbose > 0:
                    print("ERROR (SIMULATE1D): 'searchNeighborhoodSortMode=2' not allowed with non-stationary covariance model")
                return None
        elif searchNeighborhoodSortMode == 1:
            if not cov_model.is_orientation_stationary() or not cov_model.is_range_stationary():
                if verbose > 0:
                    print("ERROR (SIMULATE1D): 'searchNeighborhoodSortMode=1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                return None

    # Check parameters - mean
    if mean is not None:
        # if method == 'ordinary_kriging':
        #     if verbose > 0:
        #         print("ERROR (SIMULATE1D): specifying 'mean' not allowed with ordinary kriging")
        #     return None
        if callable(mean):
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            mean = mean(xi) # replace function 'mean' by its evaluation on the grid
        else:
            mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if mean.size not in (1, nxyz):
            if verbose > 0:
                print("ERROR (SIMULATE1D): size of 'mean' is not valid")
            return None

    # Check parameters - var
    if var is not None:
        if method == 'ordinary_kriging':
            if verbose > 0:
                print("ERROR (SIMULATE1D): specifying 'var' not allowed with ordinary kriging")
            return None
        if callable(var):
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            var = var(xi) # replace function 'var' by its evaluation on the grid
        else:
            var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if var.size not in (1, nxyz):
            if verbose > 0:
                print("ERROR (SIMULATE1D): size of 'var' is not valid")
            return None

    # Check parameters - nreal
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose >= 2:
            print('SIMULATE1D: nreal <= 0: nothing to do!')
        return None

    # --- Fill mpds_geosClassicInput structure (C)
    mpds_geosClassicInput, flag = fill_mpds_geosClassicInput(
        space_dim,
        cov_model,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        varname,
        outputReportFile,
        computationMode,
        None,
        dataPointSet,
        mask,
        mean,
        var,
        searchRadiusRelative,
        nneighborMax,
        searchNeighborhoodSortMode,
        nGibbsSamplerPathMin,
        nGibbsSamplerPathMax,
        seed,
        nreal)

    if not flag:
        if verbose > 0:
            print("ERROR (SIMULATE1D): can not fill input structure!")
        return None

    # --- Prepare mpds_geosClassicIOutput structure (C)
    # Allocate mpds_geosClassicOutput
    mpds_geosClassicOutput = geosclassic.malloc_MPDS_GEOSCLASSICOUTPUT()

    # Init mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicInitGeosClassicOutput(mpds_geosClassicOutput)

    # --- Set progress monitor
    mpds_progressMonitor = geosclassic.malloc_MPDS_PROGRESSMONITOR()
    geosclassic.MPDSInitProgressMonitor(mpds_progressMonitor)

    # Set function to update progress monitor:
    # according to geosclassic.MPDS_SHOW_PROGRESS_MONITOR set to 4 for compilation of py module
    # the function
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor1_ptr: warning only
    if verbose < 3:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

    # Free memory on C side: mpds_geosClassicInput
    geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
    #geosclassic.MPDSFree(mpds_geosClassicInput)
    geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)

    if err:
        if verbose > 0:
            err_message = geosclassic.mpds_get_error_message(-err)
            err_message = err_message.replace('\n', '')
            print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 2 and geosclassic_output and geosclassic_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(geosclassic_output['nwarning']))
        for i, warning_message in enumerate(geosclassic_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def simulate1D_mp(
        cov_model,
        dimension, spacing=1.0, origin=0.0,
        method='simple_kriging',
        nreal=1,
        mean=None, var=None,
        x=None, v=None,
        xIneqMin=None, vIneqMin=None,
        xIneqMax=None, vIneqMax=None,
        mask=None,
        searchRadiusRelative=1.0,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        nGibbsSamplerPathMin=50,
        nGibbsSamplerPathMax=200,
        seed=None,
        outputReportFile=None,
        treat_image_one_by_one=False,
        nproc=None, nthreads_per_proc=None,
        verbose=2):
    """
    Generates 1D simulations (Sequential Gaussian Simulation, SGS) based on
    simple or ordinary kriging.

    Launches multiple processes (based on multiprocessing package):
        - nproc parallel processes using each one nthreads_per_proc threads will
            be launched [parallel calls of the function simulate1D],
        - the set of realizations (specified by nreal) is
            distributed in a balanced way over the processes,
        - in terms of resources, this implies the use of
            nproc * nthreads_per_proc cpu(s).

    :param cov_model:   covariance model:
                            (CovModel1D class) covariance model in 1D, see
                                definition of the class in module geone.covModel

    :param dimension:   (int) nx, number of cells
    :param spacing:     (float) sx, spacing between two adjacent cells
    :param origin:      (float) ox, origin of the 1D simulation
                            - used for localizing the conditioning points
    :param method:      (string) indicates the method used:
                            - 'simple_kriging':
                                simulation based on simple kriging
                            - 'ordinary_kriging':
                                simulation based on ordinary kriging
    :param nreal:       (int) number of realizations

    :param mean:        (None or callable (function) or float or ndarray) mean of
                            the simulation:
                            - None   : mean of hard data values (stationary),
                                       (0 if no hard data)
                            - callable (function):
                                       function of one argument (xi) that returns
                                       the mean at xi (in the grid)
                            - float  : for stationary mean (set manually)
                            - ndarray: for non stationary mean, must contain
                                       as many entries as number of grid cells
                                       (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param var:         (None or callable (function) or float or ndarray) variance
                            of the simulation (for simple kriging only):
                            - None   : variance not modified
                                       (only covariance model is used)
                            - callable (function):
                                       function of one argument (xi) that returns
                                       the variance at xi (in the grid)
                            - float  : for stationary variance (set manually)
                            - ndarray: for non stationary variance, must contain
                                       as many entries as number of grid cells
                                       (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            this parameter must be None (only covariance model
                            is used)

    :param x:           (1-dimensional array or float or None) coordinate of
                            conditioning points for hard data
    :param v:           (1-dimensional array or float or None) value
                            at conditioning points for hard data
                            (same type as x)

    :param xIneqMin:    (1-dimensional array or float or None) coordinate of
                            conditioning points for inequality data minimal bound
    :param vIneqMin:    (1-dimensional array or float or None) value
                            at conditioning points for inequality data minimal
                            bound (same type as xIneqMin)

    :param xIneqMax:    (1-dimensional array or float or None) coordinate of
                            conditioning points for inequality data maximal bound
    :param vIneqMax:    (1-dimensional array or float or None) value
                            at conditioning points for inequality data maximal
                            bound (same type as xIneqMax)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not
                            simulated cell (nunber of entries should be equal to
                            the number of grid cells)

    :param searchRadiusRelative:
                        (float) indicating how restricting the search ellipsoid
                            (should be positive): let r_i be the ranges of the
                            covariance model along its main axes, if x is a node
                            to be simulated, a node y is taken into account iff
                            it is within the ellipsoid centered at x of half-axes
                            searchRadiusRelative * r_i
                            Note:
                                - if a range is a variable parameter, its maximal
                                value over the simulation grid is considered

    :param nneighborMax:
                        (int) maximum number of nodes retrieved from the search
                            ellipsoid, set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (int) indicating how to sort the search neighboorhood
                            nodes (neighbors), they are sorted in increasing
                            order according to:
                            - searchNeighborhoodSortMode = 0:
                                distance in the usual axes system
                            - searchNeighborhoodSortMode = 1:
                                distance in the axes sytem supporting the
                                covariance model and accounting for anisotropy
                                given by the ranges
                            - searchNeighborhoodSortMode = 2:
                                minus the evaluation of the covariance model
                            Notes:
                            - if the covariance model has any variable parameter
                                (non-stationary), then
                                searchNeighborhoodSortMode = 2 is not allowed
                            - if the covariance model has any range or angle set
                                as a variable parameter, then
                                searchNeighborhoodSortMode must be set to 0
                            - greatest possible value as default

    :param nGibbsSamplerPathMin, nGibbsSamplerPathMax:
                        (int) minimal and maximal number of Gibbs sampler paths
                            to deal with inequality data; the conditioning
                            locations with inequality data are first simulated
                            (with truncated gaussian distribution) sequentially;
                            then, these locations are re-simulated following a
                            new path as many times as needed; the total number
                            of paths will be between nGibbsSamplerPathMin and
                            nGibbsSamplerPathMax

    :param seed:        (int or None) initial seed, if None an initial seed
                            between 1 and 999999 is generated with
                            numpy.random.randint

    :param outputReportFile:
                        (string or None) name of the report file,
                            if None: no report file [if given, a suffix
                            ".<process_index>" is added for the report file of
                            each process]

    :param treat_image_one_by_one:
                        (bool) keyword argument passed to the function
                            geone.img.gatherImages
                            - if False (default) images (result of each process)
                            are gathered at once, and then removed (faster)
                            - if True, images (result of each process) are
                            gathered one by one, i.e. successively gathered and
                            removed (slower, may save memory)

    :param nproc:
                (int) number of processes (can be modified in the function)
                    nproc = None: nproc is set to
                        min(nmax-1, nreal) (but at least 1),
                    where nmax is the total number of cpu(s) of the system
                    (retrieved by multiprocessing.cpu_count())

    :param nthreads_per_proc:
                (int) number of thread(s) per process (should be > 0 or None):
                    nthreads_per_proc = None: nthreads_per_proc is automatically
                    computed as the maximal integer (but at least 1) such that
                            nproc * nthreads_per_proc <= nmax-1
                    where nmax is the total number of cpu(s) of the system
                    (retrieved by multiprocessing.cpu_count())

    :param verbose:
                (int) indicates what information is displayed:
                    - 0: no display
                    - 1: only errors (and note(s))
                    - 2: version and warning(s) encountered

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img class) output image, with image.nv=nreal variables (each
                    variable is one realization)
                    (image is None if mpds_geosClassicOutput->outputImage is
                    NULL)
        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)
        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    # Set number of processes: nproc
    if nproc is None:
        nproc = max(min(multiprocessing.cpu_count()-1, nreal), 1)
    else:
        nproc_tmp = nproc
        nproc = max(min(int(nproc), nreal), 1)
        if verbose > 0 and nproc != nproc_tmp:
            print('NOTE: number of processes has been changed (now: nproc={})'.format(nproc))

    # Set number of threads per process: nth
    if nthreads_per_proc is None:
        nth = max(int(np.floor((multiprocessing.cpu_count()-1) / nproc)), 1)
    else:
        nth = max(int(nthreads_per_proc), 1)
        if verbose > 0 and nth != nthreads_per_proc:
            print('NOTE: number of threads per process has been changed (now: nthreads_per_proc={})'.format(nth))

    if verbose > 0 and nproc * nth > multiprocessing.cpu_count():
        print('NOTE: total number of cpu(s) used will exceed number of cpu(s) of the system...')

    # Set the distribution of the realizations over the processes
    # Condider the Euclidean division of nreal by nproc:
    #     nreal = q * nproc + r, with 0 <= r < nproc
    # Then, (q+1) realizations will be done on process 0, 1, ..., r-1, and q realization on process r, ..., nproc-1
    # Define the list real_index_proc of length (nproc+1) such that
    #   real_index_proc[i], ..., real_index_proc[i+1] - 1 : are the realization indices run on process i
    q, r = np.divmod(nreal, nproc)
    real_index_proc = [i*q + min(i, r) for i in range(nproc+1)]

    if verbose >= 2:
        print('Geos-Classic running on {} process(es)... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(nproc, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching geos-classic...

    # mpds_geosClassicInput.seed
    if seed is None:
        seed = np.random.randint(1,1000000)
    seed = int(seed)

    outputReportFile_p = None

    # Set pool of nproc workers
    pool = multiprocessing.Pool(nproc)
    out_pool = []
    for i in range(nproc):
        # Adapt input for i-th process
        nreal_p = real_index_proc[i+1] - real_index_proc[i]
        seed_p = seed + real_index_proc[i]
        if outputReportFile is not None:
            outputReportFile_p = outputReportFile + f'.{i}'
        if i==0:
            verbose_p = min(verbose, 1) # allow to print error for process i
        else:
            verbose_p = 0
        # Launch geos-classic (i-th process)
        out_pool.append(
            pool.apply_async(simulate1D,
                args=(cov_model,
                dimension, spacing, origin,
                method,
                nreal_p,                     # nreal (adjusted)
                mean, var,
                x, v,
                xIneqMin, vIneqMin,
                xIneqMax, vIneqMax,
                mask,
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                nGibbsSamplerPathMin,
                nGibbsSamplerPathMax,
                seed_p,                      # seed (adjusted)
                outputReportFile_p,          # outputReportFile (adjusted)
                nth,                         # nthreads
                verbose_p)                   # verbose (adjusted)
                )
            )

    # Properly end working process
    pool.close() # Prevents any more tasks from being submitted to the pool,
    pool.join()  # then, wait for the worker processes to exit.

    # Get result from each process
    geosclassic_output_proc = [p.get() for p in out_pool]

    if np.any([out is None for out in geosclassic_output_proc]):
        return None

    image = None
    nwarning, warnings = None, None

    # Gather results from every process
    # image
    image = np.hstack([out['image'] for out in geosclassic_output_proc])
    # ... remove None entries
    image = image[[x is not None for x in image]]
    # .. set to None if every entry is None
    if np.all([x is None for x in image]):
        image = None

    # nwarning
    nwarning = np.sum([out['nwarning'] for out in geosclassic_output_proc])
    # warnings
    warnings = list(np.unique(np.hstack([out['warnings'] for out in geosclassic_output_proc])))

    # Gather images and adjust variable names
    if image is not None:
        all_image = img.gatherImages(image, keep_varname=True, rem_var_from_source=True, treat_image_one_by_one=treat_image_one_by_one)
        ndigit = geosclassic.MPDS_GEOS_CLASSIC_NB_DIGIT_FOR_REALIZATION_NUMBER
        for j in range(all_image.nv):
            all_image.varname[j] = all_image.varname[j][:-ndigit] + f'{j:0{ndigit}d}'

    geosclassic_output = {'image':all_image, 'nwarning':nwarning, 'warnings':warnings}

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete (all process(es))')

    # Show (print) encountered warnings
    if verbose >= 2 and geosclassic_output and geosclassic_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(geosclassic_output['nwarning']))
        for i, warning_message in enumerate(geosclassic_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def simulate2D(
        cov_model,
        dimension, spacing=(1.0, 1.0), origin=(0.0, 0.0),
        method='simple_kriging',
        nreal=1,
        mean=None, var=None,
        x=None, v=None,
        xIneqMin=None, vIneqMin=None,
        xIneqMax=None, vIneqMax=None,
        mask=None,
        searchRadiusRelative=1.0,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        nGibbsSamplerPathMin=50,
        nGibbsSamplerPathMax=200,
        seed=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2):
    """
    Generates 2D simulations (Sequential Gaussian Simulation, SGS) based on
    simple or ordinary kriging.

    :param cov_model:   covariance model:
                            (CovModel2D class) covariance model in 2D, see
                                definition of the class in module geone.covModel
                        or
                            (CovModel1D class) covariance model in 1D, see
                                definition of the class in module geone.covModel,
                                it is then transformed to an isotropic (omni-
                                directional) covariance model in 2D

    :param dimension:   (sequence of 2 ints) (nx, ny), number of cells
                            in x-, y-axis direction
    :param spacing:     (sequence of 2 floats) (sx, sy), spacing between
                            two adjacent cells in x-, y-axis direction
    :param origin:      (sequence of 2 floats) (ox, oy), origin of the 2D
                            simulation - used for localizing the conditioning
                            points
    :param method:      (string) indicates the method used:
                            - 'simple_kriging':
                                simulation based on simple kriging
                            - 'ordinary_kriging':
                                simulation based on ordinary kriging
    :param nreal:       (int) number of realizations

    :param mean:        (None or callable (function) or float or ndarray) mean
                            of the simulation:
                            - None   : mean of hard data values (stationary),
                                       (0 if no hard data)
                            - callable (function):
                                       function of two arguments (xi, yi) that
                                       returns the mean at (xi, yi) (in the grid)
                            - float  : for stationary mean (set manually)
                            - ndarray: for non stationary mean, must contain
                                       as many entries as number of grid cells
                                       (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param var:         (None or callable (function) or float or ndarray)
                            variance of the simulation (for simple kriging only):
                            - None   : variance not modified
                                       (only covariance function/model is used)
                            - callable (function):
                                       function of two arguments (xi, yi) that
                                       returns the variance at (xi, yi) (in the
                                       grid)
                            - float  : for stationary variance (set manually)
                            - ndarray: for non stationary variance, must contain
                                       as many entries as number of grid cells
                                       (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            this parameter must be None (only covariance model
                            is used)

    :param x:           (2-dimensional array of dim n x 2, or
                            1-dimensional array of dim 2 or None) coordinate of
                            conditioning points for hard data
    :param v:           (1-dimensional array or float or None) value
                            at conditioning points for hard data
                            (same type as x)

    :param xIneqMin:    (2-dimensional array of dim n x 2, or
                            1-dimensional array of dim 2 or None) coordinate of
                            conditioning points for inequality data minimal bound
    :param vIneqMin:    (1-dimensional array or float or None) value
                            at conditioning points for inequality data minimal
                            bound (same type as xIneqMin)

    :param xIneqMax:    (2-dimensional array of dim n x 2, or
                            1-dimensional array of dim 2 or None) coordinate of
                            conditioning points for inequality data maximal bound
    :param vIneqMax:    (1-dimensional array or float or None) value
                            at conditioning points for inequality data maximal
                            bound (same type as xIneqMax)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not
                            simulated cell (nunber of entries should be equal to
                            the number of grid cells)

    :param searchRadiusRelative:
                        (float) indicating how restricting the search ellipsoid
                            (should be positive): let r_i be the ranges of the
                            covariance model along its main axes, if x is a node
                            to be simulated, a node y is taken into account iff
                            it is within the ellipsoid centered at x of half-axes
                            searchRadiusRelative * r_i
                            Notes:
                                - if a range is a variable parameter, its maximal
                                value over the simulation grid is considered
                                - if orientation of the covariance model is
                                non-stationary, a "circular search neighborhood"
                                is considered with the radius set to the maximum
                                of all ranges

    :param nneighborMax:
                        (int) maximum number of nodes retrieved from the search
                            ellipsoid, set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (int) indicating how to sort the search neighboorhood
                            nodes (neighbors), they are sorted in increasing
                            order according to:
                            - searchNeighborhoodSortMode = 0:
                                distance in the usual axes system
                            - searchNeighborhoodSortMode = 1:
                                distance in the axes sytem supporting the
                                covariance model and accounting for anisotropy
                                given by the ranges
                            - searchNeighborhoodSortMode = 2:
                                minus the evaluation of the covariance model
                            Notes:
                            - if the covariance model has any variable parameter
                                (non-stationary), then
                                searchNeighborhoodSortMode = 2 is not allowed
                            - if the covariance model has any range or angle set
                                as a variable parameter, then
                                searchNeighborhoodSortMode must be set to 0
                            - greatest possible value as default

    :param nGibbsSamplerPathMin, nGibbsSamplerPathMax:
                        (int) minimal and maximal number of Gibbs sampler paths
                            to deal with inequality data; the conditioning
                            locations with inequality data are first simulated
                            (with truncated gaussian distribution) sequentially;
                            then, these locations are re-simulated following a
                            new path as many times as needed; the total number
                            of paths will be between nGibbsSamplerPathMin and
                            nGibbsSamplerPathMax

    :param seed:        (int or None) initial seed, if None an initial seed
                            between 1 and 999999 is generated with
                            numpy.random.randint

    :param outputReportFile:
                        (string or None) name of the report file,
                            if None: no report file

    :param nthreads:
                        (int) number of thread(s) to use for "GeosClassicSim"
                            program (C), (nthreads = -n <= 0: for maximal number
                            of threads except n, but at least 1)
    :param verbose:
                (int) verbose mode, integer >=0, higher implies more display
                    - 0: no display
                    - 1: only errors
                    - 2: errors and warnings (+ some info)
                    - 3 (or >2): all information

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img class) output image, with image.nv=nreal variables (each
                    variable is one realization)
                    (image is None if mpds_geosClassicOutput->outputImage is
                    NULL)
        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)
        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    # --- Set grid geometry and varname
    # Set grid geometry
    nx, ny, nz = *dimension, 1
    sx, sy, sz = *spacing, 1.0
    ox, oy, oz = *origin, 0.0

    nxy = nx * ny
    nxyz = nxy * nz

    # spatial dimension
    space_dim = 2

    # Set varname
    varname = 'V0'

    # --- Check and prepare parameters
    # cov_model
    if isinstance(cov_model, gcm.CovModel1D):
        cov_model = gcm.covModel1D_to_covModel2D(cov_model) # convert model 1D in 2D
            # -> will not be modified cov_model at exit

    if not isinstance(cov_model, gcm.CovModel2D):
        if verbose > 0:
            print("ERROR (SIMULATE2D): 'cov_model' (first argument) is not valid")
        return None

    for el in cov_model.elem:
        # weight
        w = el[1]['w']
        if np.size(w) != 1 and np.size(w) != nxyz:
            if verbose > 0:
                print("ERROR (SIMULATE2D): 'cov_model': weight ('w') not compatible with simulation grid")
            return None
        # ranges
        if 'r' in el[1].keys():
            for r in el[1]['r']:
                if np.size(r) != 1 and np.size(r) != nxyz:
                    if verbose > 0:
                        print("ERROR (SIMULATE2D): 'cov_model': range ('r') not compatible with simulation grid")
                    return None
        # additional parameter (s)
        if 's' in el[1].keys():
            s  = el[1]['s']
            if np.size(s) != 1 and np.size(s) != nxyz:
                if verbose > 0:
                    print("ERROR (SIMULATE2D): 'cov_model': parameter ('s') not compatible with simulation grid")
                return None

    # alpha
    angle = cov_model.alpha
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        if verbose > 0:
            print("ERROR (SIMULATE2D): 'cov_model': angle (alpha) not compatible with simulation grid")
        return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     if verbose > 0:
    #         print("ERROR (SIMULATE2D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 3
    elif method == 'ordinary_kriging':
        computationMode = 2
    else:
        if verbose > 0:
            print("ERROR (SIMULATE2D): 'method' is not valid")
        return None

    # data points: x, v, xIneqMin, vIneqMin, xIneqMax, vIneqMax
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 2) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            if verbose > 0:
                print("(ERROR (SIMULATE2D): length of 'v' is not valid")
            return None
        xc = x[:,0]
        yc = x[:,1]
        zc = np.ones_like(xc) * oz + 0.5 * sz
        dataPointSet.append(
            PointSet(npt=v.shape[0], nv=4, val=np.array((xc, yc, zc, v)), varname=['X', 'Y', 'Z', varname])
            )

    # data point set from xIneqMin, vIneqMin
    if xIneqMin is not None:
        xIneqMin = np.asarray(xIneqMin, dtype='float').reshape(-1, 2) # cast in 2-dimensional array if needed
        vIneqMin = np.asarray(vIneqMin, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(vIneqMin) != xIneqMin.shape[0]:
            if verbose > 0:
                print("(ERROR (SIMULATE2D): length of 'vIneqMin' is not valid")
            return None
        xc = xIneqMin[:,0]
        yc = xIneqMin[:,1]
        zc = np.ones_like(xc) * oz + 0.5 * sz
        dataPointSet.append(
            PointSet(npt=vIneqMin.shape[0], nv=4, val=np.array((xc, yc, zc, vIneqMin)), varname=['X', 'Y', 'Z', '{}_min'.format(varname)])
            )

    # data point set from xIneqMax, vIneqMax
    if xIneqMax is not None:
        xIneqMax = np.asarray(xIneqMax, dtype='float').reshape(-1, 2) # cast in 2-dimensional array if needed
        vIneqMax = np.asarray(vIneqMax, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(vIneqMax) != xIneqMax.shape[0]:
            if verbose > 0:
                print("(ERROR (SIMULATE2D): length of 'vIneqMax' is not valid")
            return None
        xc = xIneqMax[:,0]
        yc = xIneqMax[:,1]
        zc = np.ones_like(xc) * oz + 0.5 * sz
        dataPointSet.append(
            PointSet(npt=vIneqMax.shape[0], nv=4, val=np.array((xc, yc, zc, vIneqMax)), varname=['X', 'Y', 'Z', '{}_max'.format(varname)])
            )

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            if verbose > 0:
                print("ERROR (SIMULATE2D): 'mask' is not valid")
            return None

    # Check parameters - searchRadiusRelative
    if searchRadiusRelative < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
        if verbose > 0:
            print("ERROR (SIMULATE2D): 'searchRadiusRelative' too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
        return None

    # Check parameters - nneighborMax
    if nneighborMax != -1 and nneighborMax <= 0:
        if verbose > 0:
            print("ERROR (SIMULATE2D): 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
        return None

    # Check parameters - searchNeighborhoodSortMode
    if searchNeighborhoodSortMode is None:
        # set greatest possible value
        if cov_model.is_stationary():
            searchNeighborhoodSortMode = 2
        elif cov_model.is_orientation_stationary() and cov_model.is_range_stationary():
            searchNeighborhoodSortMode = 1
        else:
            searchNeighborhoodSortMode = 0
    else:
        if searchNeighborhoodSortMode == 2:
            if not cov_model.is_stationary():
                if verbose > 0:
                    print("ERROR (SIMULATE2D): 'searchNeighborhoodSortMode=2' not allowed with non-stationary covariance model")
                return None
        elif searchNeighborhoodSortMode == 1:
            if not cov_model.is_orientation_stationary() or not cov_model.is_range_stationary():
                if verbose > 0:
                    print("ERROR (SIMULATE2D): 'searchNeighborhoodSortMode=1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                return None

    # Check parameters - mean
    if mean is not None:
        # if method == 'ordinary_kriging':
        #     if verbose > 0:
        #         print("ERROR (SIMULATE2D): specifying 'mean' not allowed with ordinary kriging")
        #     return None
        if callable(mean):
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            xxi, yyi = np.meshgrid(xi, yi)
            mean = mean(xxi, yyi) # replace function 'mean' by its evaluation on the grid
        else:
            mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if mean.size not in (1, nxyz):
            if verbose > 0:
                print("ERROR (SIMULATE2D): size of 'mean' is not valid")
            return None

    # Check parameters - var
    if var is not None:
        if method == 'ordinary_kriging':
            if verbose > 0:
                print("ERROR (SIMULATE2D): specifying 'var' not allowed with ordinary kriging")
            return None
        if callable(var):
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            xxi, yyi = np.meshgrid(xi, yi)
            var = var(xxi, yyi) # replace function 'var' by its evaluation on the grid
        else:
            var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if var.size not in (1, nxyz):
            if verbose > 0:
                print("ERROR (SIMULATE2D): size of 'var' is not valid")
            return None

    # Check parameters - nreal
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose >= 2:
            print('SIMULATE2D: nreal <= 0: nothing to do!')
        return None

    # --- Fill mpds_geosClassicInput structure (C)
    mpds_geosClassicInput, flag = fill_mpds_geosClassicInput(
        space_dim,
        cov_model,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        varname,
        outputReportFile,
        computationMode,
        None,
        dataPointSet,
        mask,
        mean,
        var,
        searchRadiusRelative,
        nneighborMax,
        searchNeighborhoodSortMode,
        nGibbsSamplerPathMin,
        nGibbsSamplerPathMax,
        seed,
        nreal)

    if not flag:
        if verbose > 0:
            print("ERROR (SIMULATE2D): can not fill input structure!")
        return None

    # --- Prepare mpds_geosClassicIOutput structure (C)
    # Allocate mpds_geosClassicOutput
    mpds_geosClassicOutput = geosclassic.malloc_MPDS_GEOSCLASSICOUTPUT()

    # Init mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicInitGeosClassicOutput(mpds_geosClassicOutput)

    # --- Set progress monitor
    mpds_progressMonitor = geosclassic.malloc_MPDS_PROGRESSMONITOR()
    geosclassic.MPDSInitProgressMonitor(mpds_progressMonitor)

    # Set function to update progress monitor:
    # according to geosclassic.MPDS_SHOW_PROGRESS_MONITOR set to 4 for compilation of py module
    # the function
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor1_ptr: warning only
    if verbose < 3:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

    # Free memory on C side: mpds_geosClassicInput
    geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
    #geosclassic.MPDSFree(mpds_geosClassicInput)
    geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)

    if err:
        if verbose > 0:
            err_message = geosclassic.mpds_get_error_message(-err)
            err_message = err_message.replace('\n', '')
            print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 2 and geosclassic_output and geosclassic_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(geosclassic_output['nwarning']))
        for i, warning_message in enumerate(geosclassic_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def simulate2D_mp(
        cov_model,
        dimension, spacing=(1.0, 1.0), origin=(0.0, 0.0),
        method='simple_kriging',
        nreal=1,
        mean=None, var=None,
        x=None, v=None,
        xIneqMin=None, vIneqMin=None,
        xIneqMax=None, vIneqMax=None,
        mask=None,
        searchRadiusRelative=1.0,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        nGibbsSamplerPathMin=50,
        nGibbsSamplerPathMax=200,
        seed=None,
        outputReportFile=None,
        treat_image_one_by_one=False,
        nproc=None, nthreads_per_proc=None,
        verbose=2):
    """
    Generates 2D simulations (Sequential Gaussian Simulation, SGS) based on
    simple or ordinary kriging.

    Launches multiple processes (based on multiprocessing package):
        - nproc parallel processes using each one nthreads_per_proc threads will
            be launched [parallel calls of the function simulate2D],
        - the set of realizations (specified by nreal) is
            distributed in a balanced way over the processes,
        - in terms of resources, this implies the use of
            nproc * nthreads_per_proc cpu(s).

    :param cov_model:   covariance model:
                            (CovModel2D class) covariance model in 2D, see
                                definition of the class in module geone.covModel
                        or
                            (CovModel1D class) covariance model in 1D, see
                                definition of the class in module geone.covModel,
                                it is then transformed to an isotropic (omni-
                                directional) covariance model in 2D

    :param dimension:   (sequence of 2 ints) (nx, ny), number of cells
                            in x-, y-axis direction
    :param spacing:     (sequence of 2 floats) (sx, sy), spacing between
                            two adjacent cells in x-, y-axis direction
    :param origin:      (sequence of 2 floats) (ox, oy), origin of the 2D
                            simulation - used for localizing the conditioning
                            points
    :param method:      (string) indicates the method used:
                            - 'simple_kriging':
                                simulation based on simple kriging
                            - 'ordinary_kriging':
                                simulation based on ordinary kriging
    :param nreal:       (int) number of realizations

    :param mean:        (None or callable (function) or float or ndarray) mean
                            of the simulation:
                            - None   : mean of hard data values (stationary),
                                       (0 if no hard data)
                            - callable (function):
                                       function of two arguments (xi, yi) that
                                       returns the mean at (xi, yi) (in the grid)
                            - float  : for stationary mean (set manually)
                            - ndarray: for non stationary mean, must contain
                                       as many entries as number of grid cells
                                       (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param var:         (None or callable (function) or float or ndarray)
                            variance of the simulation (for simple kriging only):
                            - None   : variance not modified
                                       (only covariance function/model is used)
                            - callable (function):
                                       function of two arguments (xi, yi) that
                                       returns the variance at (xi, yi) (in the
                                       grid)
                            - float  : for stationary variance (set manually)
                            - ndarray: for non stationary variance, must contain
                                       as many entries as number of grid cells
                                       (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            this parameter must be None (only covariance model
                            is used)

    :param x:           (2-dimensional array of dim n x 2, or
                            1-dimensional array of dim 2 or None) coordinate of
                            conditioning points for hard data
    :param v:           (1-dimensional array or float or None) value
                            at conditioning points for hard data
                            (same type as x)

    :param xIneqMin:    (2-dimensional array of dim n x 2, or
                            1-dimensional array of dim 2 or None) coordinate of
                            conditioning points for inequality data minimal bound
    :param vIneqMin:    (1-dimensional array or float or None) value
                            at conditioning points for inequality data minimal
                            bound (same type as xIneqMin)

    :param xIneqMax:    (2-dimensional array of dim n x 2, or
                            1-dimensional array of dim 2 or None) coordinate of
                            conditioning points for inequality data maximal bound
    :param vIneqMax:    (1-dimensional array or float or None) value
                            at conditioning points for inequality data maximal
                            bound (same type as xIneqMax)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not
                            simulated cell (nunber of entries should be equal to
                            the number of grid cells)

    :param searchRadiusRelative:
                        (float) indicating how restricting the search ellipsoid
                            (should be positive): let r_i be the ranges of the
                            covariance model along its main axes, if x is a node
                            to be simulated, a node y is taken into account iff
                            it is within the ellipsoid centered at x of half-axes
                            searchRadiusRelative * r_i
                            Notes:
                                - if a range is a variable parameter, its maximal
                                value over the simulation grid is considered
                                - if orientation of the covariance model is
                                non-stationary, a "circular search neighborhood"
                                is considered with the radius set to the maximum
                                of all ranges

    :param nneighborMax:
                        (int) maximum number of nodes retrieved from the search
                            ellipsoid, set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (int) indicating how to sort the search neighboorhood
                            nodes (neighbors), they are sorted in increasing
                            order according to:
                            - searchNeighborhoodSortMode = 0:
                                distance in the usual axes system
                            - searchNeighborhoodSortMode = 1:
                                distance in the axes sytem supporting the
                                covariance model and accounting for anisotropy
                                given by the ranges
                            - searchNeighborhoodSortMode = 2:
                                minus the evaluation of the covariance model
                            Notes:
                            - if the covariance model has any variable parameter
                                (non-stationary), then
                                searchNeighborhoodSortMode = 2 is not allowed
                            - if the covariance model has any range or angle set
                                as a variable parameter, then
                                searchNeighborhoodSortMode must be set to 0
                            - greatest possible value as default

    :param nGibbsSamplerPathMin, nGibbsSamplerPathMax:
                        (int) minimal and maximal number of Gibbs sampler paths
                            to deal with inequality data; the conditioning
                            locations with inequality data are first simulated
                            (with truncated gaussian distribution) sequentially;
                            then, these locations are re-simulated following a
                            new path as many times as needed; the total number
                            of paths will be between nGibbsSamplerPathMin and
                            nGibbsSamplerPathMax

    :param seed:        (int or None) initial seed, if None an initial seed
                            between 1 and 999999 is generated with
                            numpy.random.randint

    :param outputReportFile:
                        (string or None) name of the report file,
                            if None: no report file [if given, a suffix
                            ".<process_index>" is added for the report file of
                            each process]

    :param treat_image_one_by_one:
                        (bool) keyword argument passed to the function
                            geone.img.gatherImages
                            - if False (default) images (result of each process)
                            are gathered at once, and then removed (faster)
                            - if True, images (result of each process) are
                            gathered one by one, i.e. successively gathered and
                            removed (slower, may save memory)

    :param nproc:
                (int) number of processes (can be modified in the function)
                    nproc = None: nproc is set to
                        min(nmax-1, nreal) (but at least 1),
                    where nmax is the total number of cpu(s) of the system
                    (retrieved by multiprocessing.cpu_count())

    :param nthreads_per_proc:
                (int) number of thread(s) per process (should be > 0 or None):
                    nthreads_per_proc = None: nthreads_per_proc is automatically
                    computed as the maximal integer (but at least 1) such that
                            nproc * nthreads_per_proc <= nmax-1
                    where nmax is the total number of cpu(s) of the system
                    (retrieved by multiprocessing.cpu_count())

    :param verbose:
                (int) indicates what information is displayed:
                    - 0: no display
                    - 1: only errors (and note(s))
                    - 2: version and warning(s) encountered

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img class) output image, with image.nv=nreal variables (each
                    variable is one realization)
                    (image is None if mpds_geosClassicOutput->outputImage is
                    NULL)
        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)
        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    # Set number of processes: nproc
    if nproc is None:
        nproc = max(min(multiprocessing.cpu_count()-1, nreal), 1)
    else:
        nproc_tmp = nproc
        nproc = max(min(int(nproc), nreal), 1)
        if verbose > 0 and nproc != nproc_tmp:
            print('NOTE: number of processes has been changed (now: nproc={})'.format(nproc))

    # Set number of threads per process: nth
    if nthreads_per_proc is None:
        nth = max(int(np.floor((multiprocessing.cpu_count()-1) / nproc)), 1)
    else:
        nth = max(int(nthreads_per_proc), 1)
        if verbose > 0 and nth != nthreads_per_proc:
            print('NOTE: number of threads per process has been changed (now: nthreads_per_proc={})'.format(nth))

    if verbose > 0 and nproc * nth > multiprocessing.cpu_count():
        print('NOTE: total number of cpu(s) used will exceed number of cpu(s) of the system...')

    # Set the distribution of the realizations over the processes
    # Condider the Euclidean division of nreal by nproc:
    #     nreal = q * nproc + r, with 0 <= r < nproc
    # Then, (q+1) realizations will be done on process 0, 1, ..., r-1, and q realization on process r, ..., nproc-1
    # Define the list real_index_proc of length (nproc+1) such that
    #   real_index_proc[i], ..., real_index_proc[i+1] - 1 : are the realization indices run on process i
    q, r = np.divmod(nreal, nproc)
    real_index_proc = [i*q + min(i, r) for i in range(nproc+1)]

    if verbose >= 2:
        print('Geos-Classic running on {} process(es)... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(nproc, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching geos-classic...

    # mpds_geosClassicInput.seed
    if seed is None:
        seed = np.random.randint(1,1000000)
    seed = int(seed)

    outputReportFile_p = None

    # Set pool of nproc workers
    pool = multiprocessing.Pool(nproc)
    out_pool = []
    for i in range(nproc):
        # Adapt input for i-th process
        nreal_p = real_index_proc[i+1] - real_index_proc[i]
        seed_p = seed + real_index_proc[i]
        if outputReportFile is not None:
            outputReportFile_p = outputReportFile + f'.{i}'
        if i==0:
            verbose_p = min(verbose, 1) # allow to print error for process i
        else:
            verbose_p = 0
        # Launch geos-classic (i-th process)
        out_pool.append(
            pool.apply_async(simulate2D,
                args=(cov_model,
                dimension, spacing, origin,
                method,
                nreal_p,                     # nreal (adjusted)
                mean, var,
                x, v,
                xIneqMin, vIneqMin,
                xIneqMax, vIneqMax,
                mask,
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                nGibbsSamplerPathMin,
                nGibbsSamplerPathMax,
                seed_p,                      # seed (adjusted)
                outputReportFile_p,          # outputReportFile (adjusted)
                nth,                         # nthreads
                verbose_p)                   # verbose (adjusted)
                )
            )

    # Properly end working process
    pool.close() # Prevents any more tasks from being submitted to the pool,
    pool.join()  # then, wait for the worker processes to exit.

    # Get result from each process
    geosclassic_output_proc = [p.get() for p in out_pool]

    if np.any([out is None for out in geosclassic_output_proc]):
        return None

    image = None
    nwarning, warnings = None, None

    # Gather results from every process
    # image
    image = np.hstack([out['image'] for out in geosclassic_output_proc])
    # ... remove None entries
    image = image[[x is not None for x in image]]
    # .. set to None if every entry is None
    if np.all([x is None for x in image]):
        image = None

    # nwarning
    nwarning = np.sum([out['nwarning'] for out in geosclassic_output_proc])
    # warnings
    warnings = list(np.unique(np.hstack([out['warnings'] for out in geosclassic_output_proc])))

    # Gather images and adjust variable names
    if image is not None:
        all_image = img.gatherImages(image, keep_varname=True, rem_var_from_source=True, treat_image_one_by_one=treat_image_one_by_one)
        ndigit = geosclassic.MPDS_GEOS_CLASSIC_NB_DIGIT_FOR_REALIZATION_NUMBER
        for j in range(all_image.nv):
            all_image.varname[j] = all_image.varname[j][:-ndigit] + f'{j:0{ndigit}d}'

    geosclassic_output = {'image':all_image, 'nwarning':nwarning, 'warnings':warnings}

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete (all process(es))')

    # Show (print) encountered warnings
    if verbose >= 2 and geosclassic_output and geosclassic_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(geosclassic_output['nwarning']))
        for i, warning_message in enumerate(geosclassic_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def simulate3D(
        cov_model,
        dimension, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
        method='simple_kriging',
        nreal=1,
        mean=None, var=None,
        x=None, v=None,
        xIneqMin=None, vIneqMin=None,
        xIneqMax=None, vIneqMax=None,
        mask=None,
        searchRadiusRelative=1.0,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        nGibbsSamplerPathMin=50,
        nGibbsSamplerPathMax=200,
        seed=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2):
    """
    Generates 3D simulations (Sequential Gaussian Simulation, SGS) based on
    simple or ordinary kriging.

    :param cov_model:   covariance model:
                            (CovModel3D class) covariance model in 3D, see
                                definition of the class in module geone.covModel
                        or
                            (CovModel1D class) covariance model in 1D, see
                                definition of the class in module geone.covModel,
                                it is then transformed to an isotropic (omni-
                                directional) covariance model in 3D

    :param dimension:   (sequence of 3 ints) (nx, ny, nz), number of cells
                            in x-, y-, z-axis direction
    :param spacing:     (sequence of 3 floats) (sx, sy, sz), spacing between
                            two adjacent cells in x-, y-, z-axis direction
    :param origin:      (sequence of 3 floats) (ox, oy, oy), origin of the 3D
                            simulation - used for localizing the conditioning
                            points
    :param method:      (string) indicates the method used:
                            - 'simple_kriging':
                                simulation based on simple kriging
                            - 'ordinary_kriging':
                                simulation based on ordinary kriging
    :param nreal:       (int) number of realizations

    :param mean:        (None or callable (function) or float or ndarray) mean
                            of the simulation:
                            - None   : mean of hard data values (stationary),
                                       (0 if no hard data)
                            - callable (function):
                                       function of three arguments (xi, yi, zi)
                                       that returns the mean at (xi, yi, zi) (in
                                       the grid)
                            - float  : for stationary mean (set manually)
                            - ndarray: for non stationary mean, must contain
                                       as many entries as number of grid cells
                                       (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param var:         (None or callable (function) or float or ndarray)
                            variance of the simulation (for simple kriging only):
                            - None   : variance not modified
                                       (only covariance function/model is used)
                            - callable (function):
                                       function of three arguments (xi, yi, zi)
                                       that returns the variance at (xi, yi, zi)
                                       (in the grid)
                            - float  : for stationary variance (set manually)
                            - ndarray: for non stationary variance, must contain
                                       as many entries as number of grid cells
                                       (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            this parameter must be None (only covariance model
                            is used)

    :param x:           (2-dimensional array of dim n x 3, or
                            1-dimensional array of dim 3 or None) coordinate of
                            conditioning points for hard data
    :param v:           (1-dimensional array or float or None) value
                            at conditioning points for hard data
                            (same type as x)

    :param xIneqMin:    (2-dimensional array of dim n x 3, or
                            1-dimensional array of dim 3 or None) coordinate of
                            conditioning points for inequality data minimal bound
    :param vIneqMin:    (1-dimensional array or float or None) value
                            at conditioning points for inequality data minimal
                            bound (same type as xIneqMin)

    :param xIneqMax:    (2-dimensional array of dim n x 3, or
                            1-dimensional array of dim 3 or None) coordinate of
                            conditioning points for inequality data maximal bound
    :param vIneqMax:    (1-dimensional array or float or None) value
                            at conditioning points for inequality data maximal
                            bound (same type as xIneqMax)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not
                            simulated cell (nunber of entries should be equal to
                            the number of grid cells)

    :param searchRadiusRelative:
                        (float) indicating how restricting the search ellipsoid
                            (should be positive): let r_i be the ranges of the
                            covariance model along its main axes, if x is a node
                            to be simulated, a node y is taken into account iff
                            it is within the ellipsoid centered at x of half-axes
                            searchRadiusRelative * r_i
                            Notes:
                                - if a range is a variable parameter, its maximal
                                value over the simulation grid is considered
                                - if orientation of the covariance model is
                                non-stationary, a "circular search neighborhood"
                                is considered with the radius set to the maximum
                                of all ranges

    :param nneighborMax:
                        (int) maximum number of nodes retrieved from the search
                            ellipsoid, set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (int) indicating how to sort the search neighboorhood
                            nodes (neighbors), they are sorted in increasing
                            order according to:
                            - searchNeighborhoodSortMode = 0:
                                distance in the usual axes system
                            - searchNeighborhoodSortMode = 1:
                                distance in the axes sytem supporting the
                                covariance model and accounting for anisotropy
                                given by the ranges
                            - searchNeighborhoodSortMode = 2:
                                minus the evaluation of the covariance model
                            Notes:
                            - if the covariance model has any variable parameter
                                (non-stationary), then
                                searchNeighborhoodSortMode = 2 is not allowed
                            - if the covariance model has any range or angle set
                                as a variable parameter, then
                                searchNeighborhoodSortMode must be set to 0
                            - greatest possible value as default

    :param nGibbsSamplerPathMin, nGibbsSamplerPathMax:
                        (int) minimal and maximal number of Gibbs sampler paths
                            to deal with inequality data; the conditioning
                            locations with inequality data are first simulated
                            (with truncated gaussian distribution) sequentially;
                            then, these locations are re-simulated following a
                            new path as many times as needed; the total number
                            of paths will be between nGibbsSamplerPathMin and
                            nGibbsSamplerPathMax

    :param seed:        (int or None) initial seed, if None an initial seed
                            between 1 and 999999 is generated with
                            numpy.random.randint

    :param outputReportFile:
                        (string or None) name of the report file,
                            if None: no report file

    :param nthreads:
                        (int) number of thread(s) to use for "GeosClassicSim"
                            program (C), (nthreads = -n <= 0: for maximal number
                            of threads except n, but at least 1)
    :param verbose:
                (int) verbose mode, integer >=0, higher implies more display
                    - 0: no display
                    - 1: only errors
                    - 2: errors and warnings (+ some info)
                    - 3 (or >2): all information

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img class) output image, with image.nv=nreal variables (each
                    variable is one realization)
                    (image is None if mpds_geosClassicOutput->outputImage is
                    NULL)
        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)
        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    # --- Set grid geometry and varname
    # Set grid geometry
    nx, ny, nz = dimension
    sx, sy, sz = spacing
    ox, oy, oz = origin

    nxy = nx * ny
    nxyz = nxy * nz

    # spatial dimension
    space_dim = 3

    # Set varname
    varname = 'V0'

    # --- Check and prepare parameters
    # cov_model
    if isinstance(cov_model, gcm.CovModel1D):
        cov_model = gcm.covModel1D_to_covModel3D(cov_model) # convert model 1D in 3D
            # -> will not be modified cov_model at exit

    if not isinstance(cov_model, gcm.CovModel3D):
        if verbose > 0:
            print("ERROR (SIMULATE3D): 'cov_model' (first argument) is not valid")
        return None

    for el in cov_model.elem:
        # weight
        w = el[1]['w']
        if np.size(w) != 1 and np.size(w) != nxyz:
            if verbose > 0:
                print("ERROR (SIMULATE3D): 'cov_model': weight ('w') not compatible with simulation grid")
            return None
        # ranges
        if 'r' in el[1].keys():
            for r in el[1]['r']:
                if np.size(r) != 1 and np.size(r) != nxyz:
                    if verbose > 0:
                        print("ERROR (SIMULATE3D): 'cov_model': range ('r') not compatible with simulation grid")
                    return None
        # additional parameter (s)
        if 's' in el[1].keys():
            s  = el[1]['s']
            if np.size(s) != 1 and np.size(s) != nxyz:
                if verbose > 0:
                    print("ERROR (SIMULATE3D): 'cov_model': parameter ('s') not compatible with simulation grid")
                return None

    # alpha
    angle = cov_model.alpha
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        if verbose > 0:
            print("ERROR (SIMULATE3D): 'cov_model': angle (alpha) not compatible with simulation grid")
        return None

    # beta
    angle = cov_model.beta
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        if verbose > 0:
            print("ERROR (SIMULATE3D): 'cov_model': angle (beta) not compatible with simulation grid")
        return None

    # gamma
    angle = cov_model.gamma
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        if verbose > 0:
            print("ERROR (SIMULATE3D): 'cov_model': angle (gamma) not compatible with simulation grid")
        return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     if verbose > 0:
    #         print("ERROR (SIMULATE3D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 3
    elif method == 'ordinary_kriging':
        computationMode = 2
    else:
        if verbose > 0:
            print("ERROR (SIMULATE3D): 'method' is not valid")
        return None

    # data points: x, v, xIneqMin, vIneqMin, xIneqMax, vIneqMax
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 3) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            if verbose > 0:
                print("(ERROR (SIMULATE3D): length of 'v' is not valid")
            return None
        xc = x[:,0]
        yc = x[:,1]
        zc = x[:,2]
        dataPointSet.append(
            PointSet(npt=v.shape[0], nv=4, val=np.array((xc, yc, zc, v)), varname=['X', 'Y', 'Z', varname])
            )

    # data point set from xIneqMin, vIneqMin
    if xIneqMin is not None:
        xIneqMin = np.asarray(xIneqMin, dtype='float').reshape(-1, 3) # cast in 2-dimensional array if needed
        vIneqMin = np.asarray(vIneqMin, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(vIneqMin) != xIneqMin.shape[0]:
            if verbose > 0:
                print("(ERROR (SIMULATE3D): length of 'vIneqMin' is not valid")
            return None
        xc = xIneqMin[:,0]
        yc = xIneqMin[:,1]
        zc = xIneqMin[:,2]
        dataPointSet.append(
            PointSet(npt=vIneqMin.shape[0], nv=4, val=np.array((xc, yc, zc, vIneqMin)), varname=['X', 'Y', 'Z', '{}_min'.format(varname)])
            )

    # data point set from xIneqMax, vIneqMax
    if xIneqMax is not None:
        xIneqMax = np.asarray(xIneqMax, dtype='float').reshape(-1, 3) # cast in 2-dimensional array if needed
        vIneqMax = np.asarray(vIneqMax, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(vIneqMax) != xIneqMax.shape[0]:
            if verbose > 0:
                print("(ERROR (SIMULATE3D): length of 'vIneqMax' is not valid")
            return None
        xc = xIneqMax[:,0]
        yc = xIneqMax[:,1]
        zc = xIneqMax[:,2]
        dataPointSet.append(
            PointSet(npt=vIneqMax.shape[0], nv=4, val=np.array((xc, yc, zc, vIneqMax)), varname=['X', 'Y', 'Z', '{}_max'.format(varname)])
            )

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            if verbose > 0:
                print("ERROR (SIMULATE3D): 'mask' is not valid")
            return None

    # Check parameters - searchRadiusRelative
    if searchRadiusRelative < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
        if verbose > 0:
            print("ERROR (SIMULATE3D): 'searchRadiusRelative' too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
        return None

    # Check parameters - nneighborMax
    if nneighborMax != -1 and nneighborMax <= 0:
        if verbose > 0:
            print("ERROR (SIMULATE3D): 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
        return None

    # Check parameters - searchNeighborhoodSortMode
    if searchNeighborhoodSortMode is None:
        # set greatest possible value
        if cov_model.is_stationary():
            searchNeighborhoodSortMode = 2
        elif cov_model.is_orientation_stationary() and cov_model.is_range_stationary():
            searchNeighborhoodSortMode = 1
        else:
            searchNeighborhoodSortMode = 0
    else:
        if searchNeighborhoodSortMode == 2:
            if not cov_model.is_stationary():
                if verbose > 0:
                    print("ERROR (SIMULATE3D): 'searchNeighborhoodSortMode=2' not allowed with non-stationary covariance model")
                return None
        elif searchNeighborhoodSortMode == 1:
            if not cov_model.is_orientation_stationary() or not cov_model.is_range_stationary():
                if verbose > 0:
                    print("ERROR (SIMULATE3D): 'searchNeighborhoodSortMode=1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                return None

    # Check parameters - mean
    if mean is not None:
        # if method == 'ordinary_kriging':
        #     if verbose > 0:
        #         print("ERROR (SIMULATE3D): specifying 'mean' not allowed with ordinary kriging")
        #     return None
        if callable(mean):
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            zi = oz + sz*(0.5+np.arange(nz)) # z-coordinate of cell center
            zzi, yyi, xxi = np.meshgrid(zi, yi, xi, indexing='ij')
            mean = mean(xxi, yyi, zzi) # replace function 'mean' by its evaluation on the grid
        else:
            mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if mean.size not in (1, nxyz):
            if verbose > 0:
                print("ERROR (SIMULATE3D): size of 'mean' is not valid")
            return None

    # Check parameters - var
    if var is not None:
        if method == 'ordinary_kriging':
            if verbose > 0:
                print("ERROR (SIMULATE3D): specifying 'var' not allowed with ordinary kriging")
            return None
        if callable(var):
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            zi = oz + sz*(0.5+np.arange(nz)) # z-coordinate of cell center
            zzi, yyi, xxi = np.meshgrid(zi, yi, xi, indexing='ij')
            var = var(xxi, yyi, zzi) # replace function 'var' by its evaluation on the grid
        else:
            var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if var.size not in (1, nxyz):
            if verbose > 0:
                print("ERROR (SIMULATE3D): size of 'var' is not valid")
            return None

    # Check parameters - nreal
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose >= 2:
            print('SIMULATE3D: nreal <= 0: nothing to do!')
        return None

    # --- Fill mpds_geosClassicInput structure (C)
    mpds_geosClassicInput, flag = fill_mpds_geosClassicInput(
        space_dim,
        cov_model,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        varname,
        outputReportFile,
        computationMode,
        None,
        dataPointSet,
        mask,
        mean,
        var,
        searchRadiusRelative,
        nneighborMax,
        searchNeighborhoodSortMode,
        nGibbsSamplerPathMin,
        nGibbsSamplerPathMax,
        seed,
        nreal)

    if not flag:
        if verbose > 0:
            print("ERROR (SIMULATE3D): can not fill input structure!")
        return None

    # --- Prepare mpds_geosClassicIOutput structure (C)
    # Allocate mpds_geosClassicOutput
    mpds_geosClassicOutput = geosclassic.malloc_MPDS_GEOSCLASSICOUTPUT()

    # Init mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicInitGeosClassicOutput(mpds_geosClassicOutput)

    # --- Set progress monitor
    mpds_progressMonitor = geosclassic.malloc_MPDS_PROGRESSMONITOR()
    geosclassic.MPDSInitProgressMonitor(mpds_progressMonitor)

    # Set function to update progress monitor:
    # according to geosclassic.MPDS_SHOW_PROGRESS_MONITOR set to 4 for compilation of py module
    # the function
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor1_ptr: warning only
    if verbose < 3:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

    # Free memory on C side: mpds_geosClassicInput
    geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
    #geosclassic.MPDSFree(mpds_geosClassicInput)
    geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)

    if err:
        if verbose > 0:
            err_message = geosclassic.mpds_get_error_message(-err)
            err_message = err_message.replace('\n', '')
            print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 2 and geosclassic_output and geosclassic_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(geosclassic_output['nwarning']))
        for i, warning_message in enumerate(geosclassic_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def simulate3D_mp(
        cov_model,
        dimension, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
        method='simple_kriging',
        nreal=1,
        mean=None, var=None,
        x=None, v=None,
        xIneqMin=None, vIneqMin=None,
        xIneqMax=None, vIneqMax=None,
        mask=None,
        searchRadiusRelative=1.0,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        nGibbsSamplerPathMin=50,
        nGibbsSamplerPathMax=200,
        seed=None,
        outputReportFile=None,
        treat_image_one_by_one=False,
        nproc=None, nthreads_per_proc=None,
        verbose=2):
    """
    Generates 3D simulations (Sequential Gaussian Simulation, SGS) based on
    simple or ordinary kriging.

    Launches multiple processes (based on multiprocessing package):
        - nproc parallel processes using each one nthreads_per_proc threads will
            be launched [parallel calls of the function simulate3D],
        - the set of realizations (specified by nreal) is
            distributed in a balanced way over the processes,
        - in terms of resources, this implies the use of
            nproc * nthreads_per_proc cpu(s).

    :param cov_model:   covariance model:
                            (CovModel3D class) covariance model in 3D, see
                                definition of the class in module geone.covModel
                        or
                            (CovModel1D class) covariance model in 1D, see
                                definition of the class in module geone.covModel,
                                it is then transformed to an isotropic (omni-
                                directional) covariance model in 3D

    :param dimension:   (sequence of 3 ints) (nx, ny, nz), number of cells
                            in x-, y-, z-axis direction
    :param spacing:     (sequence of 3 floats) (sx, sy, sz), spacing between
                            two adjacent cells in x-, y-, z-axis direction
    :param origin:      (sequence of 3 floats) (ox, oy, oy), origin of the 3D
                            simulation - used for localizing the conditioning
                            points
    :param method:      (string) indicates the method used:
                            - 'simple_kriging':
                                simulation based on simple kriging
                            - 'ordinary_kriging':
                                simulation based on ordinary kriging
    :param nreal:       (int) number of realizations

    :param mean:        (None or callable (function) or float or ndarray) mean
                            of the simulation:
                            - None   : mean of hard data values (stationary),
                                       (0 if no hard data)
                            - callable (function):
                                       function of three arguments (xi, yi, zi)
                                       that returns the mean at (xi, yi, zi) (in
                                       the grid)
                            - float  : for stationary mean (set manually)
                            - ndarray: for non stationary mean, must contain
                                       as many entries as number of grid cells
                                       (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param var:         (None or callable (function) or float or ndarray)
                            variance of the simulation (for simple kriging only):
                            - None   : variance not modified
                                       (only covariance function/model is used)
                            - callable (function):
                                       function of three arguments (xi, yi, zi)
                                       that returns the variance at (xi, yi, zi)
                                       (in the grid)
                            - float  : for stationary variance (set manually)
                            - ndarray: for non stationary variance, must contain
                                       as many entries as number of grid cells
                                       (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            this parameter must be None (only covariance model
                            is used)

    :param x:           (2-dimensional array of dim n x 3, or
                            1-dimensional array of dim 3 or None) coordinate of
                            conditioning points for hard data
    :param v:           (1-dimensional array or float or None) value
                            at conditioning points for hard data
                            (same type as x)

    :param xIneqMin:    (2-dimensional array of dim n x 3, or
                            1-dimensional array of dim 3 or None) coordinate of
                            conditioning points for inequality data minimal bound
    :param vIneqMin:    (1-dimensional array or float or None) value
                            at conditioning points for inequality data minimal
                            bound (same type as xIneqMin)

    :param xIneqMax:    (2-dimensional array of dim n x 3, or
                            1-dimensional array of dim 3 or None) coordinate of
                            conditioning points for inequality data maximal bound
    :param vIneqMax:    (1-dimensional array or float or None) value
                            at conditioning points for inequality data maximal
                            bound (same type as xIneqMax)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not
                            simulated cell (nunber of entries should be equal to
                            the number of grid cells)

    :param searchRadiusRelative:
                        (float) indicating how restricting the search ellipsoid
                            (should be positive): let r_i be the ranges of the
                            covariance model along its main axes, if x is a node
                            to be simulated, a node y is taken into account iff
                            it is within the ellipsoid centered at x of half-axes
                            searchRadiusRelative * r_i
                            Notes:
                                - if a range is a variable parameter, its maximal
                                value over the simulation grid is considered
                                - if orientation of the covariance model is
                                non-stationary, a "circular search neighborhood"
                                is considered with the radius set to the maximum
                                of all ranges

    :param nneighborMax:
                        (int) maximum number of nodes retrieved from the search
                            ellipsoid, set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (int) indicating how to sort the search neighboorhood
                            nodes (neighbors), they are sorted in increasing
                            order according to:
                            - searchNeighborhoodSortMode = 0:
                                distance in the usual axes system
                            - searchNeighborhoodSortMode = 1:
                                distance in the axes sytem supporting the
                                covariance model and accounting for anisotropy
                                given by the ranges
                            - searchNeighborhoodSortMode = 2:
                                minus the evaluation of the covariance model
                            Notes:
                            - if the covariance model has any variable parameter
                                (non-stationary), then
                                searchNeighborhoodSortMode = 2 is not allowed
                            - if the covariance model has any range or angle set
                                as a variable parameter, then
                                searchNeighborhoodSortMode must be set to 0
                            - greatest possible value as default

    :param nGibbsSamplerPathMin, nGibbsSamplerPathMax:
                        (int) minimal and maximal number of Gibbs sampler paths
                            to deal with inequality data; the conditioning
                            locations with inequality data are first simulated
                            (with truncated gaussian distribution) sequentially;
                            then, these locations are re-simulated following a
                            new path as many times as needed; the total number
                            of paths will be between nGibbsSamplerPathMin and
                            nGibbsSamplerPathMax

    :param seed:        (int or None) initial seed, if None an initial seed
                            between 1 and 999999 is generated with
                            numpy.random.randint

    :param outputReportFile:
                        (string or None) name of the report file,
                            if None: no report file [if given, a suffix
                            ".<process_index>" is added for the report file of
                            each process]

    :param treat_image_one_by_one:
                        (bool) keyword argument passed to the function
                            geone.img.gatherImages
                            - if False (default) images (result of each process)
                            are gathered at once, and then removed (faster)
                            - if True, images (result of each process) are
                            gathered one by one, i.e. successively gathered and
                            removed (slower, may save memory)

    :param nproc:
                (int) number of processes (can be modified in the function)
                    nproc = None: nproc is set to
                        min(nmax-1, nreal) (but at least 1),
                    where nmax is the total number of cpu(s) of the system
                    (retrieved by multiprocessing.cpu_count())

    :param nthreads_per_proc:
                (int) number of thread(s) per process (should be > 0 or None):
                    nthreads_per_proc = None: nthreads_per_proc is automatically
                    computed as the maximal integer (but at least 1) such that
                            nproc * nthreads_per_proc <= nmax-1
                    where nmax is the total number of cpu(s) of the system
                    (retrieved by multiprocessing.cpu_count())

    :param verbose:
                (int) indicates what information is displayed:
                    - 0: no display
                    - 1: only errors (and note(s))
                    - 2: version and warning(s) encountered

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img class) output image, with image.nv=nreal variables (each
                    variable is one realization)
                    (image is None if mpds_geosClassicOutput->outputImage is
                    NULL)
        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)
        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    # Set number of processes: nproc
    if nproc is None:
        nproc = max(min(multiprocessing.cpu_count()-1, nreal), 1)
    else:
        nproc_tmp = nproc
        nproc = max(min(int(nproc), nreal), 1)
        if verbose > 0 and nproc != nproc_tmp:
            print('NOTE: number of processes has been changed (now: nproc={})'.format(nproc))

    # Set number of threads per process: nth
    if nthreads_per_proc is None:
        nth = max(int(np.floor((multiprocessing.cpu_count()-1) / nproc)), 1)
    else:
        nth = max(int(nthreads_per_proc), 1)
        if verbose > 0 and nth != nthreads_per_proc:
            print('NOTE: number of threads per process has been changed (now: nthreads_per_proc={})'.format(nth))

    if verbose > 0 and nproc * nth > multiprocessing.cpu_count():
        print('NOTE: total number of cpu(s) used will exceed number of cpu(s) of the system...')

    # Set the distribution of the realizations over the processes
    # Condider the Euclidean division of nreal by nproc:
    #     nreal = q * nproc + r, with 0 <= r < nproc
    # Then, (q+1) realizations will be done on process 0, 1, ..., r-1, and q realization on process r, ..., nproc-1
    # Define the list real_index_proc of length (nproc+1) such that
    #   real_index_proc[i], ..., real_index_proc[i+1] - 1 : are the realization indices run on process i
    q, r = np.divmod(nreal, nproc)
    real_index_proc = [i*q + min(i, r) for i in range(nproc+1)]

    if verbose >= 2:
        print('Geos-Classic running on {} process(es)... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(nproc, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching geos-classic...

    # mpds_geosClassicInput.seed
    if seed is None:
        seed = np.random.randint(1,1000000)
    seed = int(seed)

    outputReportFile_p = None

    # Set pool of nproc workers
    pool = multiprocessing.Pool(nproc)
    out_pool = []
    for i in range(nproc):
        # Adapt input for i-th process
        nreal_p = real_index_proc[i+1] - real_index_proc[i]
        seed_p = seed + real_index_proc[i]
        if outputReportFile is not None:
            outputReportFile_p = outputReportFile + f'.{i}'
        if i==0:
            verbose_p = min(verbose, 1) # allow to print error for process i
        else:
            verbose_p = 0
        # Launch geos-classic (i-th process)
        out_pool.append(
            pool.apply_async(simulate3D,
                args=(cov_model,
                dimension, spacing, origin,
                method,
                nreal_p,                     # nreal (adjusted)
                mean, var,
                x, v,
                xIneqMin, vIneqMin,
                xIneqMax, vIneqMax,
                mask,
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                nGibbsSamplerPathMin,
                nGibbsSamplerPathMax,
                seed_p,                      # seed (adjusted)
                outputReportFile_p,          # outputReportFile (adjusted)
                nth,                         # nthreads
                verbose_p)                   # verbose (adjusted)
                )
            )

    # Properly end working process
    pool.close() # Prevents any more tasks from being submitted to the pool,
    pool.join()  # then, wait for the worker processes to exit.

    # Get result from each process
    geosclassic_output_proc = [p.get() for p in out_pool]

    if np.any([out is None for out in geosclassic_output_proc]):
        return None

    image = None
    nwarning, warnings = None, None

    # Gather results from every process
    # image
    image = np.hstack([out['image'] for out in geosclassic_output_proc])
    # ... remove None entries
    image = image[[x is not None for x in image]]
    # .. set to None if every entry is None
    if np.all([x is None for x in image]):
        image = None

    # nwarning
    nwarning = np.sum([out['nwarning'] for out in geosclassic_output_proc])
    # warnings
    warnings = list(np.unique(np.hstack([out['warnings'] for out in geosclassic_output_proc])))

    # Gather images and adjust variable names
    if image is not None:
        all_image = img.gatherImages(image, keep_varname=True, rem_var_from_source=True, treat_image_one_by_one=treat_image_one_by_one)
        ndigit = geosclassic.MPDS_GEOS_CLASSIC_NB_DIGIT_FOR_REALIZATION_NUMBER
        for j in range(all_image.nv):
            all_image.varname[j] = all_image.varname[j][:-ndigit] + f'{j:0{ndigit}d}'

    geosclassic_output = {'image':all_image, 'nwarning':nwarning, 'warnings':warnings}

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete (all process(es))')

    # Show (print) encountered warnings
    if verbose >= 2 and geosclassic_output and geosclassic_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(geosclassic_output['nwarning']))
        for i, warning_message in enumerate(geosclassic_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def estimate1D(
        cov_model,
        dimension, spacing=1.0, origin=0.0,
        method='simple_kriging',
        mean=None, var=None,
        x=None, v=None,
        mask=None,
        use_unique_neighborhood=False,
        searchRadiusRelative=1.0,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2):
    """
    Computes estimate and standard deviation for 1D grid of simple or ordinary
    kriging.

    :param cov_model:   covariance model:
                            (CovModel1D class) covariance model in 1D, see
                                definition of the class in module geone.covModel

    :param dimension:   (int) nx, number of cells
    :param spacing:     (float) sx, spacing between two adjacent cells
    :param origin:      (float) ox, origin of the 1D simulation
                            - used for localizing the conditioning points
    :param method:      (string) indicates the method used:
                            - 'simple_kriging':
                                simulation based on simple kriging
                            - 'ordinary_kriging':
                                simulation based on ordinary kriging

    :param mean:        (None or callable (function) or float or ndarray) mean of
                            the simulation:
                            - None   : mean of hard data values (stationary),
                                       (0 if no hard data)
                            - callable (function):
                                       function of one argument (xi) that returns
                                       the mean at xi (in the grid)
                            - float  : for stationary mean (set manually)
                            - ndarray: for non stationary mean, must contain
                                       as many entries as number of grid cells
                                       (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param var:         (None or callable (function) or float or ndarray) variance
                            of the simulation (for simple kriging only):
                            - None   : variance not modified
                                       (only covariance model is used)
                            - callable (function):
                                       function of one argument (xi) that returns
                                       the variance at xi (in the grid)
                            - float  : for stationary variance (set manually)
                            - ndarray: for non stationary variance, must contain
                                       as many entries as number of grid cells
                                       (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            this parameter must be None (only covariance model
                            is used)

    :param x:           (1-dimensional array or float or None) coordinate of
                            conditioning points for hard data
    :param v:           (1-dimensional array or float or None) value
                            at conditioning points for hard data
                            (same type as x)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not
                            simulated cell (nunber of entries should be equal to
                            the number of grid cells)

    :param use_unique_neighborhood:
                        (bool) indicating if a unique neighborhood is used
                            - True: all data points are taken into account for
                                computing estimates and standard deviation;
                                in this case: parameters
                                    searchRadiusRelative,
                                    nneighborMax,
                                    searchNeighborhoodSortMode,
                                are unused
                            - False: only data points within a search ellipsoid
                                are taken into account for computing estimates
                                and standard deviation (see parameters
                                searchRadiusRelative, nneighborMax,
                                searchNeighborhoodSortMode)

    :param searchRadiusRelative:
                        (float) indicating how restricting the search ellipsoid
                            (should be positive): let r_i be the ranges of the
                            covariance model along its main axes, if x is a node
                            to be simulated, a node y is taken into account iff
                            it is within the ellipsoid centered at x of half-axes
                            searchRadiusRelative * r_i
                            Note:
                                - if a range is a variable parameter, its maximal
                                value over the simulation grid is considered

    :param nneighborMax:
                        (int) maximum number of nodes retrieved from the search
                            ellipsoid, set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (int) indicating how to sort the search neighboorhood
                            nodes (neighbors), they are sorted in increasing
                            order according to:
                            - searchNeighborhoodSortMode = 0:
                                distance in the usual axes system
                            - searchNeighborhoodSortMode = 1:
                                distance in the axes sytem supporting the
                                covariance model and accounting for anisotropy
                                given by the ranges
                            - searchNeighborhoodSortMode = 2:
                                minus the evaluation of the covariance model
                            Notes:
                            - if the covariance model has any variable parameter
                                (non-stationary), then
                                searchNeighborhoodSortMode = 2 is not allowed
                            - if the covariance model has any range or angle set
                                as a variable parameter, then
                                searchNeighborhoodSortMode must be set to 0
                            - greatest possible value as default

    :param outputReportFile:
                        (string or None) name of the report file,
                            if None: no report file

    :param nthreads:
                        (int) number of thread(s) to use for "GeosClassicSim"
                            program (C), (nthreads = -n <= 0: for maximal number
                            of threads except n, but at least 1)
    :param verbose:
                (int) verbose mode, integer >=0, higher implies more display
                    - 0: no display
                    - 1: only errors
                    - 2: errors and warnings (+ some info)
                    - 3 (or >2): all information

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img class) output image, with image.nv=2 variables (estimate and
                    standard deviation)
                    (image is None if mpds_geosClassicOutput->outputImage is
                    NULL)
        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)
        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    # --- Set grid geometry and varname
    # Set grid geometry
    nx, ny, nz = dimension, 1, 1
    sx, sy, sz = spacing, 1.0, 1.0
    ox, oy, oz = origin, 0.0, 0.0

    nxy = nx * ny
    nxyz = nxy * nz

    # spatial dimension
    space_dim = 1

    # Set varname
    varname = 'V0'

    # --- Check and prepare parameters
    # cov_model
    if not isinstance(cov_model, gcm.CovModel1D):
        if verbose > 0:
            print("ERROR (ESTIMATE1D): 'cov_model' (first argument) is not valid")
        return None

    for el in cov_model.elem:
        # weight
        w = el[1]['w']
        if np.size(w) != 1 and np.size(w) != nxyz:
            if verbose > 0:
                print("ERROR (ESTIMATE1D): 'cov_model': weight ('w') not compatible with simulation grid")
            return None
        # ranges
        if 'r' in el[1].keys():
            r  = el[1]['r']
            if np.size(r) != 1 and np.size(r) != nxyz:
                if verbose > 0:
                    print("ERROR (ESTIMATE1D): 'cov_model': range ('r') not compatible with simulation grid")
                return None
        # additional parameter (s)
        if 's' in el[1].keys():
            s  = el[1]['s']
            if np.size(s) != 1 and np.size(s) != nxyz:
                if verbose > 0:
                    print("ERROR (ESTIMATE1D): 'cov_model': parameter ('s') not compatible with simulation grid")
                return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     if verbose > 0:
    #         print("ERROR (ESTIMATE1D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 1
    elif method == 'ordinary_kriging':
        computationMode = 0
    else:
        if verbose > 0:
            print("ERROR (ESTIMATE1D): 'method' is not valid")
        return None

    # data points: x, v
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            if verbose > 0:
                print("(ERROR (ESTIMATE1D): length of 'v' is not valid")
            return None
        xc = x
        yc = np.ones_like(xc) * oy + 0.5 * sy
        zc = np.ones_like(xc) * oz + 0.5 * sz
        dataPointSet.append(
            PointSet(npt=v.shape[0], nv=4, val=np.array((xc, yc, zc, v)), varname=['X', 'Y', 'Z', varname])
            )

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            if verbose > 0:
                print("ERROR (ESTIMATE1D): 'mask' is not valid")
            return None

    # If unique neighborhood is used, set searchRadiusRelative to -1
    #    (and initialize nneighborMax, searchNeighborhoodSortMode (unused))
    if use_unique_neighborhood:
        searchRadiusRelative = -1.0
        nneighborMax = 1
        searchNeighborhoodSortMode = 0

    else:
       # Check parameters - searchRadiusRelative
       if searchRadiusRelative < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
           if verbose > 0:
               print("ERROR (ESTIMATE1D): 'searchRadiusRelative' too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
           return None

       # Check parameters - nneighborMax
       if nneighborMax != -1 and nneighborMax <= 0:
           if verbose > 0:
               print("ERROR (ESTIMATE1D): 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
           return None

       # Check parameters - searchNeighborhoodSortMode
       if searchNeighborhoodSortMode is None:
           # set greatest possible value
           if cov_model.is_stationary():
               searchNeighborhoodSortMode = 2
           elif cov_model.is_orientation_stationary() and cov_model.is_range_stationary():
               searchNeighborhoodSortMode = 1
           else:
               searchNeighborhoodSortMode = 0
       else:
           if searchNeighborhoodSortMode == 2:
               if not cov_model.is_stationary():
                   if verbose > 0:
                       print("ERROR (ESTIMATE1D): 'searchNeighborhoodSortMode=2' not allowed with non-stationary covariance model")
                   return None
           elif searchNeighborhoodSortMode == 1:
               if not cov_model.is_orientation_stationary() or not cov_model.is_range_stationary():
                   if verbose > 0:
                       print("ERROR (ESTIMATE1D): 'searchNeighborhoodSortMode=1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                   return None

    # Check parameters - mean
    if mean is not None:
        # if method == 'ordinary_kriging':
        #     if verbose > 0:
        #         print("ERROR (ESTIMATE1D): specifying 'mean' not allowed with ordinary kriging")
        #     return None
        if callable(mean):
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            mean = mean(xi) # replace function 'mean' by its evaluation on the grid
        else:
            mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if mean.size not in (1, nxyz):
            if verbose > 0:
                print("ERROR (ESTIMATE1D): size of 'mean' is not valid")
            return None

    # Check parameters - var
    if var is not None:
        if method == 'ordinary_kriging':
            if verbose > 0:
                print("ERROR (ESTIMATE1D): specifying 'var' not allowed with ordinary kriging")
            return None
        if callable(var):
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            var = var(xi) # replace function 'var' by its evaluation on the grid
        else:
            var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if var.size not in (1, nxyz):
            if verbose > 0:
                print("ERROR (ESTIMATE1D): size of 'var' is not valid")
            return None

    # --- Fill mpds_geosClassicInput structure (C)
    mpds_geosClassicInput, flag = fill_mpds_geosClassicInput(
        space_dim,
        cov_model,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        varname,
        outputReportFile,
        computationMode,
        None,
        dataPointSet,
        mask,
        mean,
        var,
        searchRadiusRelative,
        nneighborMax,
        searchNeighborhoodSortMode,
        0,
        0,
        0,
        0)

    if not flag:
        if verbose > 0:
            print("ERROR (ESTIMATE1D): can not fill input structure!")
        return None

    # --- Prepare mpds_geosClassicIOutput structure (C)
    # Allocate mpds_geosClassicOutput
    mpds_geosClassicOutput = geosclassic.malloc_MPDS_GEOSCLASSICOUTPUT()

    # Init mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicInitGeosClassicOutput(mpds_geosClassicOutput)

    # --- Set progress monitor
    mpds_progressMonitor = geosclassic.malloc_MPDS_PROGRESSMONITOR()
    geosclassic.MPDSInitProgressMonitor(mpds_progressMonitor)

    # Set function to update progress monitor:
    # according to geosclassic.MPDS_SHOW_PROGRESS_MONITOR set to 4 for compilation of py module
    # the function
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor1_ptr: warning only
    if verbose < 3:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

    # Free memory on C side: mpds_geosClassicInput
    geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
    #geosclassic.MPDSFree(mpds_geosClassicInput)
    geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)

    if err:
        if verbose > 0:
            err_message = geosclassic.mpds_get_error_message(-err)
            err_message = err_message.replace('\n', '')
            print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 2 and geosclassic_output and geosclassic_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(geosclassic_output['nwarning']))
        for i, warning_message in enumerate(geosclassic_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def estimate2D(
        cov_model,
        dimension, spacing=(1.0, 1.0), origin=(0.0, 0.0),
        method='simple_kriging',
        mean=None, var=None,
        x=None, v=None,
        mask=None,
        use_unique_neighborhood=False,
        searchRadiusRelative=1.0,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2):
    """
    Computes estimate and standard deviation for 2D grid of simple or ordinary
    kriging.

    :param cov_model:   covariance model:
                            (CovModel2D class) covariance model in 2D, see
                                definition of the class in module geone.covModel
                        or
                            (CovModel1D class) covariance model in 1D, see
                                definition of the class in module geone.covModel,
                                it is then transformed to an isotropic (omni-
                                directional) covariance model in 2D

    :param dimension:   (sequence of 2 ints) (nx, ny), number of cells
                            in x-, y-axis direction
    :param spacing:     (sequence of 2 floats) (sx, sy), spacing between
                            two adjacent cells in x-, y-axis direction
    :param origin:      (sequence of 2 floats) (ox, oy), origin of the 2D
                            simulation - used for localizing the conditioning
                            points
    :param method:      (string) indicates the method used:
                            - 'simple_kriging':
                                simulation based on simple kriging
                            - 'ordinary_kriging':
                                simulation based on ordinary kriging

    :param mean:        (None or callable (function) or float or ndarray) mean
                            of the simulation:
                            - None   : mean of hard data values (stationary),
                                       (0 if no hard data)
                            - callable (function):
                                       function of two arguments (xi, yi) that
                                       returns the mean at (xi, yi) (in the grid)
                            - float  : for stationary mean (set manually)
                            - ndarray: for non stationary mean, must contain
                                       as many entries as number of grid cells
                                       (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param var:         (None or callable (function) or float or ndarray)
                            variance of the simulation (for simple kriging only):
                            - None   : variance not modified
                                       (only covariance function/model is used)
                            - callable (function):
                                       function of two arguments (xi, yi) that
                                       returns the variance at (xi, yi) (in the
                                       grid)
                            - float  : for stationary variance (set manually)
                            - ndarray: for non stationary variance, must contain
                                       as many entries as number of grid cells
                                       (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            this parameter must be None (only covariance model
                            is used)

    :param x:           (2-dimensional array of dim n x 2, or
                            1-dimensional array of dim 2 or None) coordinate of
                            conditioning points for hard data
    :param v:           (1-dimensional array or float or None) value
                            at conditioning points for hard data
                            (same type as x)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not
                            simulated cell (nunber of entries should be equal to
                            the number of grid cells)

    :param use_unique_neighborhood:
                        (bool) indicating if a unique neighborhood is used
                            - True: all data points are taken into account for
                                computing estimates and standard deviation;
                                in this case: parameters
                                    searchRadiusRelative,
                                    nneighborMax,
                                    searchNeighborhoodSortMode,
                                are unused
                            - False: only data points within a search ellipsoid
                                are taken into account for computing estimates
                                and standard deviation (see parameters
                                searchRadiusRelative, nneighborMax,
                                searchNeighborhoodSortMode)

    :param searchRadiusRelative:
                        (float) indicating how restricting the search ellipsoid
                            (should be positive): let r_i be the ranges of the
                            covariance model along its main axes, if x is a node
                            to be simulated, a node y is taken into account iff
                            it is within the ellipsoid centered at x of half-axes
                            searchRadiusRelative * r_i
                            Notes:
                                - if a range is a variable parameter, its maximal
                                value over the simulation grid is considered
                                - if orientation of the covariance model is
                                non-stationary, a "circular search neighborhood"
                                is considered with the radius set to the maximum
                                of all ranges

    :param nneighborMax:
                        (int) maximum number of nodes retrieved from the search
                            ellipsoid, set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (int) indicating how to sort the search neighboorhood
                            nodes (neighbors), they are sorted in increasing
                            order according to:
                            - searchNeighborhoodSortMode = 0:
                                distance in the usual axes system
                            - searchNeighborhoodSortMode = 1:
                                distance in the axes sytem supporting the
                                covariance model and accounting for anisotropy
                                given by the ranges
                            - searchNeighborhoodSortMode = 2:
                                minus the evaluation of the covariance model
                            Notes:
                            - if the covariance model has any variable parameter
                                (non-stationary), then
                                searchNeighborhoodSortMode = 2 is not allowed
                            - if the covariance model has any range or angle set
                                as a variable parameter, then
                                searchNeighborhoodSortMode must be set to 0
                            - greatest possible value as default

    :param outputReportFile:
                        (string or None) name of the report file,
                            if None: no report file

    :param nthreads:
                        (int) number of thread(s) to use for "GeosClassicSim"
                            program (C), (nthreads = -n <= 0: for maximal number
                            of threads except n, but at least 1)
    :param verbose:
                (int) verbose mode, integer >=0, higher implies more display
                    - 0: no display
                    - 1: only errors
                    - 2: errors and warnings (+ some info)
                    - 3 (or >2): all information

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img class) output image, with image.nv=2 variables (estimate and
                    standard deviation)
                    (image is None if mpds_geosClassicOutput->outputImage is
                    NULL)
        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)
        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    # --- Set grid geometry and varname
    # Set grid geometry
    nx, ny, nz = *dimension, 1
    sx, sy, sz = *spacing, 1.0
    ox, oy, oz = *origin, 0.0

    nxy = nx * ny
    nxyz = nxy * nz

    # spatial dimension
    space_dim = 2

    # Set varname
    varname = 'V0'

    # --- Check and prepare parameters
    # cov_model
    if isinstance(cov_model, gcm.CovModel1D):
        cov_model = gcm.covModel1D_to_covModel2D(cov_model) # convert model 1D in 2D
            # -> will not be modified cov_model at exit

    if not isinstance(cov_model, gcm.CovModel2D):
        if verbose > 0:
            print("ERROR (ESTIMATE2D): 'cov_model' (first argument) is not valid")
        return None

    for el in cov_model.elem:
        # weight
        w = el[1]['w']
        if np.size(w) != 1 and np.size(w) != nxyz:
            if verbose > 0:
                print("ERROR (ESTIMATE2D): 'cov_model': weight ('w') not compatible with simulation grid")
            return None
        # ranges
        if 'r' in el[1].keys():
            for r in el[1]['r']:
                if np.size(r) != 1 and np.size(r) != nxyz:
                    if verbose > 0:
                        print("ERROR (ESTIMATE2D): 'cov_model': range ('r') not compatible with simulation grid")
                    return None
        # additional parameter (s)
        if 's' in el[1].keys():
            s  = el[1]['s']
            if np.size(s) != 1 and np.size(s) != nxyz:
                if verbose > 0:
                    print("ERROR (ESTIMATE2D): 'cov_model': parameter ('s') not compatible with simulation grid")
                return None

    # alpha
    angle = cov_model.alpha
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        if verbose > 0:
            print("ERROR (ESTIMATE2D): 'cov_model': angle (alpha) not compatible with simulation grid")
        return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     if verbose > 0:
    #         print("ERROR (ESTIMATE2D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 1
    elif method == 'ordinary_kriging':
        computationMode = 0
    else:
        if verbose > 0:
            print("ERROR (ESTIMATE2D): 'method' is not valid")
        return None

    # data points: x, v
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 2) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            if verbose > 0:
                print("(ERROR (ESTIMATE2D): length of 'v' is not valid")
            return None
        xc = x[:,0]
        yc = x[:,1]
        zc = np.ones_like(xc) * oz + 0.5 * sz
        dataPointSet.append(
            PointSet(npt=v.shape[0], nv=4, val=np.array((xc, yc, zc, v)), varname=['X', 'Y', 'Z', varname])
            )

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            if verbose > 0:
                print("ERROR (ESTIMATE2D): 'mask' is not valid")
            return None

    # If unique neighborhood is used, set searchRadiusRelative to -1
    #    (and initialize nneighborMax, searchNeighborhoodSortMode (unused))
    if use_unique_neighborhood:
        searchRadiusRelative = -1.0
        nneighborMax = 1
        searchNeighborhoodSortMode = 0

    else:
       # Check parameters - searchRadiusRelative
       if searchRadiusRelative < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
           if verbose > 0:
               print("ERROR (ESTIMATE2D): 'searchRadiusRelative' too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
           return None

       # Check parameters - nneighborMax
       if nneighborMax != -1 and nneighborMax <= 0:
           if verbose > 0:
               print("ERROR (ESTIMATE2D): 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
           return None

       # Check parameters - searchNeighborhoodSortMode
       if searchNeighborhoodSortMode is None:
           # set greatest possible value
           if cov_model.is_stationary():
               searchNeighborhoodSortMode = 2
           elif cov_model.is_orientation_stationary() and cov_model.is_range_stationary():
               searchNeighborhoodSortMode = 1
           else:
               searchNeighborhoodSortMode = 0
       else:
           if searchNeighborhoodSortMode == 2:
               if not cov_model.is_stationary():
                   if verbose > 0:
                       print("ERROR (ESTIMATE2D): 'searchNeighborhoodSortMode=2' not allowed with non-stationary covariance model")
                   return None
           elif searchNeighborhoodSortMode == 1:
               if not cov_model.is_orientation_stationary() or not cov_model.is_range_stationary():
                   if verbose > 0:
                       print("ERROR (ESTIMATE2D): 'searchNeighborhoodSortMode=1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                   return None

    # Check parameters - mean
    if mean is not None:
        # if method == 'ordinary_kriging':
        #     if verbose > 0:
        #         print("ERROR (ESTIMATE2D): specifying 'mean' not allowed with ordinary kriging")
        #     return None
        if callable(mean):
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            xxi, yyi = np.meshgrid(xi, yi)
            mean = mean(xxi, yyi) # replace function 'mean' by its evaluation on the grid
        else:
            mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if mean.size not in (1, nxyz):
            if verbose > 0:
                print("ERROR (ESTIMATE2D): size of 'mean' is not valid")
            return None

    # Check parameters - var
    if var is not None:
        if method == 'ordinary_kriging':
            if verbose > 0:
                print("ERROR (ESTIMATE2D): specifying 'var' not allowed with ordinary kriging")
            return None
        if callable(var):
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            xxi, yyi = np.meshgrid(xi, yi)
            var = var(xxi, yyi) # replace function 'var' by its evaluation on the grid
        else:
            var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if var.size not in (1, nxyz):
            if verbose > 0:
                print("ERROR (ESTIMATE2D): size of 'var' is not valid")
            return None

    # --- Fill mpds_geosClassicInput structure (C)
    mpds_geosClassicInput, flag = fill_mpds_geosClassicInput(
        space_dim,
        cov_model,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        varname,
        outputReportFile,
        computationMode,
        None,
        dataPointSet,
        mask,
        mean,
        var,
        searchRadiusRelative,
        nneighborMax,
        searchNeighborhoodSortMode,
        0,
        0,
        0,
        0)

    if not flag:
        if verbose > 0:
            print("ERROR (ESTIMATE2D): can not fill input structure!")
        return None

    # --- Prepare mpds_geosClassicIOutput structure (C)
    # Allocate mpds_geosClassicOutput
    mpds_geosClassicOutput = geosclassic.malloc_MPDS_GEOSCLASSICOUTPUT()

    # Init mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicInitGeosClassicOutput(mpds_geosClassicOutput)

    # --- Set progress monitor
    mpds_progressMonitor = geosclassic.malloc_MPDS_PROGRESSMONITOR()
    geosclassic.MPDSInitProgressMonitor(mpds_progressMonitor)

    # Set function to update progress monitor:
    # according to geosclassic.MPDS_SHOW_PROGRESS_MONITOR set to 4 for compilation of py module
    # the function
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor1_ptr: warning only
    if verbose < 3:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

    # Free memory on C side: mpds_geosClassicInput
    geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
    #geosclassic.MPDSFree(mpds_geosClassicInput)
    geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)

    if err:
        if verbose > 0:
            err_message = geosclassic.mpds_get_error_message(-err)
            err_message = err_message.replace('\n', '')
            print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 2 and geosclassic_output and geosclassic_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(geosclassic_output['nwarning']))
        for i, warning_message in enumerate(geosclassic_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def estimate3D(
        cov_model,
        dimension, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
        method='simple_kriging',
        mean=None, var=None,
        x=None, v=None,
        mask=None,
        use_unique_neighborhood=False,
        searchRadiusRelative=1.0,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2):
    """
    Computes estimate and standard deviation for 3D grid of simple or ordinary
    kriging.

    :param cov_model:   covariance model:
                            (CovModel3D class) covariance model in 3D, see
                                definition of the class in module geone.covModel
                        or
                            (CovModel1D class) covariance model in 1D, see
                                definition of the class in module geone.covModel,
                                it is then transformed to an isotropic (omni-
                                directional) covariance model in 3D

    :param dimension:   (sequence of 3 ints) (nx, ny, nz), number of cells
                            in x-, y-, z-axis direction
    :param spacing:     (sequence of 3 floats) (sx, sy, sz), spacing between
                            two adjacent cells in x-, y-, z-axis direction
    :param origin:      (sequence of 3 floats) (ox, oy, oy), origin of the 3D
                            simulation - used for localizing the conditioning
                            points
    :param method:      (string) indicates the method used:
                            - 'simple_kriging':
                                simulation based on simple kriging
                            - 'ordinary_kriging':
                                simulation based on ordinary kriging

    :param mean:        (None or callable (function) or float or ndarray) mean
                            of the simulation:
                            - None   : mean of hard data values (stationary),
                                       (0 if no hard data)
                            - callable (function):
                                       function of three arguments (xi, yi, zi)
                                       that returns the mean at (xi, yi, zi) (in
                                       the grid)
                            - float  : for stationary mean (set manually)
                            - ndarray: for non stationary mean, must contain
                                       as many entries as number of grid cells
                                       (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param var:         (None or callable (function) or float or ndarray)
                            variance of the simulation (for simple kriging only):
                            - None   : variance not modified
                                       (only covariance function/model is used)
                            - callable (function):
                                       function of three arguments (xi, yi, zi)
                                       that returns the variance at (xi, yi, zi)
                                       (in the grid)
                            - float  : for stationary variance (set manually)
                            - ndarray: for non stationary variance, must contain
                                       as many entries as number of grid cells
                                       (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            this parameter must be None (only covariance model
                            is used)

    :param x:           (2-dimensional array of dim n x 3, or
                            1-dimensional array of dim 3 or None) coordinate of
                            conditioning points for hard data
    :param v:           (1-dimensional array or float or None) value
                            at conditioning points for hard data
                            (same type as x)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not
                            simulated cell (nunber of entries should be equal to
                            the number of grid cells)

    :param use_unique_neighborhood:
                        (bool) indicating if a unique neighborhood is used
                            - True: all data points are taken into account for
                                computing estimates and standard deviation;
                                in this case: parameters
                                    searchRadiusRelative,
                                    nneighborMax,
                                    searchNeighborhoodSortMode,
                                are unused
                            - False: only data points within a search ellipsoid
                                are taken into account for computing estimates
                                and standard deviation (see parameters
                                searchRadiusRelative, nneighborMax,
                                searchNeighborhoodSortMode)

    :param searchRadiusRelative:
                        (float) indicating how restricting the search ellipsoid
                            (should be positive): let r_i be the ranges of the
                            covariance model along its main axes, if x is a node
                            to be simulated, a node y is taken into account iff
                            it is within the ellipsoid centered at x of half-axes
                            searchRadiusRelative * r_i
                            Notes:
                                - if a range is a variable parameter, its maximal
                                value over the simulation grid is considered
                                - if orientation of the covariance model is
                                non-stationary, a "circular search neighborhood"
                                is considered with the radius set to the maximum
                                of all ranges

    :param nneighborMax:
                        (int) maximum number of nodes retrieved from the search
                            ellipsoid, set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (int) indicating how to sort the search neighboorhood
                            nodes (neighbors), they are sorted in increasing
                            order according to:
                            - searchNeighborhoodSortMode = 0:
                                distance in the usual axes system
                            - searchNeighborhoodSortMode = 1:
                                distance in the axes sytem supporting the
                                covariance model and accounting for anisotropy
                                given by the ranges
                            - searchNeighborhoodSortMode = 2:
                                minus the evaluation of the covariance model
                            Notes:
                            - if the covariance model has any variable parameter
                                (non-stationary), then
                                searchNeighborhoodSortMode = 2 is not allowed
                            - if the covariance model has any range or angle set
                                as a variable parameter, then
                                searchNeighborhoodSortMode must be set to 0
                            - greatest possible value as default

    :param outputReportFile:
                        (string or None) name of the report file,
                            if None: no report file

    :param nthreads:
                        (int) number of thread(s) to use for "GeosClassicSim"
                            program (C), (nthreads = -n <= 0: for maximal number
                            of threads except n, but at least 1)
    :param verbose:
                (int) verbose mode, integer >=0, higher implies more display
                    - 0: no display
                    - 1: only errors
                    - 2: errors and warnings (+ some info)
                    - 3 (or >2): all information

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img class) output image, with image.nv=2 variables (estimate and
                    standard deviation)
                    (image is None if mpds_geosClassicOutput->outputImage is
                    NULL)
        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)
        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    # --- Set grid geometry and varname
    # Set grid geometry
    nx, ny, nz = dimension
    sx, sy, sz = spacing
    ox, oy, oz = origin

    nxy = nx * ny
    nxyz = nxy * nz

    # spatial dimension
    space_dim = 3

    # Set varname
    varname = 'V0'

    # --- Check and prepare parameters
    # cov_model
    if isinstance(cov_model, gcm.CovModel1D):
        cov_model = gcm.covModel1D_to_covModel3D(cov_model) # convert model 1D in 3D
            # -> will not be modified cov_model at exit

    if not isinstance(cov_model, gcm.CovModel3D):
        if verbose > 0:
            print("ERROR (ESTIMATE3D): 'cov_model' (first argument) is not valid")
        return None

    for el in cov_model.elem:
        # weight
        w = el[1]['w']
        if np.size(w) != 1 and np.size(w) != nxyz:
            if verbose > 0:
                print("ERROR (ESTIMATE3D): 'cov_model': weight ('w') not compatible with simulation grid")
            return None
        # ranges
        if 'r' in el[1].keys():
            for r in el[1]['r']:
                if np.size(r) != 1 and np.size(r) != nxyz:
                    if verbose > 0:
                        print("ERROR (ESTIMATE3D): 'cov_model': range ('r') not compatible with simulation grid")
                    return None
        # additional parameter (s)
        if 's' in el[1].keys():
            s  = el[1]['s']
            if np.size(s) != 1 and np.size(s) != nxyz:
                if verbose > 0:
                    print("ERROR (ESTIMATE3D): 'cov_model': parameter ('s') not compatible with simulation grid")
                return None

    # alpha
    angle = cov_model.alpha
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        if verbose > 0:
            print("ERROR (ESTIMATE3D): 'cov_model': angle (alpha) not compatible with simulation grid")
        return None

    # beta
    angle = cov_model.beta
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        if verbose > 0:
            print("ERROR (ESTIMATE3D): 'cov_model': angle (beta) not compatible with simulation grid")
        return None

    # gamma
    angle = cov_model.gamma
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        if verbose > 0:
            print("ERROR (ESTIMATE3D): 'cov_model': angle (gamma) not compatible with simulation grid")
        return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     if verbose > 0:
    #         print("ERROR (ESTIMATE3D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 1
    elif method == 'ordinary_kriging':
        computationMode = 0
    else:
        if verbose > 0:
            print("ERROR (ESTIMATE3D): 'method' is not valid")
        return None

    # data points: x, v
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 3) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            if verbose > 0:
                print("(ERROR (ESTIMATE3D): length of 'v' is not valid")
            return None
        xc = x[:,0]
        yc = x[:,1]
        zc = x[:,2]
        dataPointSet.append(
            PointSet(npt=v.shape[0], nv=4, val=np.array((xc, yc, zc, v)), varname=['X', 'Y', 'Z', varname])
            )

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            if verbose > 0:
                print("ERROR (ESTIMATE3D): 'mask' is not valid")
            return None

    # If unique neighborhood is used, set searchRadiusRelative to -1
    #    (and initialize nneighborMax, searchNeighborhoodSortMode (unused))
    if use_unique_neighborhood:
        searchRadiusRelative = -1.0
        nneighborMax = 1
        searchNeighborhoodSortMode = 0

    else:
       # Check parameters - searchRadiusRelative
       if searchRadiusRelative < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
           if verbose > 0:
               print("ERROR (ESTIMATE3D): 'searchRadiusRelative' too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
           return None

       # Check parameters - nneighborMax
       if nneighborMax != -1 and nneighborMax <= 0:
           if verbose > 0:
               print("ERROR (ESTIMATE3D): 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
           return None

       # Check parameters - searchNeighborhoodSortMode
       if searchNeighborhoodSortMode is None:
           # set greatest possible value
           if cov_model.is_stationary():
               searchNeighborhoodSortMode = 2
           elif cov_model.is_orientation_stationary() and cov_model.is_range_stationary():
               searchNeighborhoodSortMode = 1
           else:
               searchNeighborhoodSortMode = 0
       else:
           if searchNeighborhoodSortMode == 2:
               if not cov_model.is_stationary():
                   if verbose > 0:
                       print("ERROR (ESTIMATE3D): 'searchNeighborhoodSortMode=2' not allowed with non-stationary covariance model")
                   return None
           elif searchNeighborhoodSortMode == 1:
               if not cov_model.is_orientation_stationary() or not cov_model.is_range_stationary():
                   if verbose > 0:
                       print("ERROR (ESTIMATE3D): 'searchNeighborhoodSortMode=1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                   return None

    # Check parameters - mean
    if mean is not None:
        # if method == 'ordinary_kriging':
        #     if verbose > 0:
        #         print("ERROR (ESTIMATE3D): specifying 'mean' not allowed with ordinary kriging")
        #     return None
        if callable(mean):
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            zi = oz + sz*(0.5+np.arange(nz)) # z-coordinate of cell center
            zzi, yyi, xxi = np.meshgrid(zi, yi, xi, indexing='ij')
            mean = mean(xxi, yyi, zzi) # replace function 'mean' by its evaluation on the grid
        else:
            mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if mean.size not in (1, nxyz):
            if verbose > 0:
                print("ERROR (ESTIMATE3D): size of 'mean' is not valid")
            return None

    # Check parameters - var
    if var is not None:
        if method == 'ordinary_kriging':
            if verbose > 0:
                print("ERROR (ESTIMATE3D): specifying 'var' not allowed with ordinary kriging")
            return None
        if callable(var):
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            zi = oz + sz*(0.5+np.arange(nz)) # z-coordinate of cell center
            zzi, yyi, xxi = np.meshgrid(zi, yi, xi, indexing='ij')
            var = var(xxi, yyi, zzi) # replace function 'var' by its evaluation on the grid
        else:
            var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if var.size not in (1, nxyz):
            if verbose > 0:
                print("ERROR (ESTIMATE3D): size of 'var' is not valid")
            return None

    # --- Fill mpds_geosClassicInput structure (C)
    mpds_geosClassicInput, flag = fill_mpds_geosClassicInput(
        space_dim,
        cov_model,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        varname,
        outputReportFile,
        computationMode,
        None,
        dataPointSet,
        mask,
        mean,
        var,
        searchRadiusRelative,
        nneighborMax,
        searchNeighborhoodSortMode,
        0,
        0,
        0,
        0)

    if not flag:
        if verbose > 0:
            print("ERROR (ESTIMATE3D): can not fill input structure!")
        return None

    # --- Prepare mpds_geosClassicIOutput structure (C)
    # Allocate mpds_geosClassicOutput
    mpds_geosClassicOutput = geosclassic.malloc_MPDS_GEOSCLASSICOUTPUT()

    # Init mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicInitGeosClassicOutput(mpds_geosClassicOutput)

    # --- Set progress monitor
    mpds_progressMonitor = geosclassic.malloc_MPDS_PROGRESSMONITOR()
    geosclassic.MPDSInitProgressMonitor(mpds_progressMonitor)

    # Set function to update progress monitor:
    # according to geosclassic.MPDS_SHOW_PROGRESS_MONITOR set to 4 for compilation of py module
    # the function
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor1_ptr: warning only
    if verbose < 3:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

    # Free memory on C side: mpds_geosClassicInput
    geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
    #geosclassic.MPDSFree(mpds_geosClassicInput)
    geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)

    if err:
        if verbose > 0:
            err_message = geosclassic.mpds_get_error_message(-err)
            err_message = err_message.replace('\n', '')
            print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 2 and geosclassic_output and geosclassic_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(geosclassic_output['nwarning']))
        for i, warning_message in enumerate(geosclassic_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def fill_mpds_geosClassicIndicatorInput(
        space_dim,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        varname,
        ncategory,
        categoryValue,
        outputReportFile,
        computationMode,
        cov_model_for_category,
        dataImage,
        dataPointSet,
        mask,
        probability,
        searchRadiusRelative,
        nneighborMax,
        searchNeighborhoodSortMode,
        seed,
        nreal):
    """
    Fills a mpds_geosClassicIndicatorInput C structure from given parameters.

    :return (mpds_geosClassicIndicatorInput, flag):
        mpds_geosClassicIndicatorInput: C structure for "GeosClassicIndicatorSim"
            program (C)
        flag: (bool) indicating if the filling has been done correctly (True)
            or not (False)
    """

    nxy = nx * ny
    nxyz = nxy * nz

    # Allocate mpds_geosClassicIndicatorInput
    mpds_geosClassicIndicatorInput = geosclassic.malloc_MPDS_GEOSCLASSICINDICATORINPUT()

    # Init mpds_geosClassicIndicatorInput
    geosclassic.MPDSGeosClassicInitGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)

    # mpds_geosClassicIndicatorInput.consoleAppFlag
    mpds_geosClassicIndicatorInput.consoleAppFlag = geosclassic.FALSE

    # mpds_geosClassicIndicatorInput.simGrid
    mpds_geosClassicIndicatorInput.simGrid = geosclassic.malloc_MPDS_GRID()

    mpds_geosClassicIndicatorInput.simGrid.nx = int(nx)
    mpds_geosClassicIndicatorInput.simGrid.ny = int(ny)
    mpds_geosClassicIndicatorInput.simGrid.nz = int(nz)

    mpds_geosClassicIndicatorInput.simGrid.sx = float(sx)
    mpds_geosClassicIndicatorInput.simGrid.sy = float(sy)
    mpds_geosClassicIndicatorInput.simGrid.sz = float(sz)

    mpds_geosClassicIndicatorInput.simGrid.ox = float(ox)
    mpds_geosClassicIndicatorInput.simGrid.oy = float(oy)
    mpds_geosClassicIndicatorInput.simGrid.oz = float(oz)

    mpds_geosClassicIndicatorInput.simGrid.nxy = nxy
    mpds_geosClassicIndicatorInput.simGrid.nxyz = nxyz

    # mpds_geosClassicIndicatorInput.varname
    geosclassic.mpds_allocate_and_set_geosClassicIndicatorInput_varname(mpds_geosClassicIndicatorInput, varname)

    # mpds_geosClassicIndicatorInput.ncategory
    mpds_geosClassicIndicatorInput.ncategory = ncategory

    # mpds_geosClassicIndicatorInput.categoryValue
    mpds_geosClassicIndicatorInput.categoryValue = geosclassic.new_real_array(ncategory)
    geosclassic.mpds_set_real_vector_from_array(mpds_geosClassicIndicatorInput.categoryValue, 0, np.asarray(categoryValue).reshape(-1))

    # mpds_geosClassicIndicatorInput.outputMode
    mpds_geosClassicIndicatorInput.outputMode = geosclassic.GEOS_CLASSIC_OUTPUT_NO_FILE

    # mpds_geosClassicIndicatorInput.outputReportFlag and mpds_geosClassicIndicatorInput.outputReportFileName
    if outputReportFile is not None:
        mpds_geosClassicIndicatorInput.outputReportFlag = geosclassic.TRUE
        geosclassic.mpds_allocate_and_set_geosClassicIndicatorInput_outputReportFileName(mpds_geosClassicIndicatorInput, outputReportFile)
    else:
        mpds_geosClassicIndicatorInput.outputReportFlag = geosclassic.FALSE

    # mpds_geosClassicIndicatorInput.computationMode
    mpds_geosClassicIndicatorInput.computationMode = int(computationMode)

    # mpds_geosClassicIndicatorInput.covModel
    mpds_geosClassicIndicatorInput.covModel = geosclassic.new_MPDS_COVMODEL_array(int(ncategory))
    for i, cov_model in enumerate(cov_model_for_category):
        cov_model_c = geosclassic.malloc_MPDS_COVMODEL()
        geosclassic.MPDSGeosClassicInitCovModel(cov_model_c)
        if space_dim==1:
            cov_model_c, flag = covModel1D_py2C(cov_model, nx, ny, nz, sx, sy, sz, ox, oy, oz)
        elif space_dim==2:
            cov_model_c, flag = covModel2D_py2C(cov_model, nx, ny, nz, sx, sy, sz, ox, oy, oz)
        elif space_dim==3:
            cov_model_c, flag = covModel3D_py2C(cov_model, nx, ny, nz, sx, sy, sz, ox, oy, oz)
        if flag:
            geosclassic.MPDS_COVMODEL_array_setitem(mpds_geosClassicIndicatorInput.covModel, i, cov_model_c)
            # geosclassic.free_MPDS_COVMODEL(cov_model_c)
        else:
            geosclassic.free_MPDS_COVMODEL(cov_model_c)
            return mpds_geosClassicIndicatorInput, False

    # mpds_geosClassicIndicatorInput.searchRadiusRelative
    mpds_geosClassicIndicatorInput.searchRadiusRelative = geosclassic.new_real_array(int(ncategory))
    geosclassic.mpds_set_real_vector_from_array(
        mpds_geosClassicIndicatorInput.searchRadiusRelative, 0,
        np.asarray(searchRadiusRelative).reshape(int(ncategory)))

    # mpds_geosClassicIndicatorInput.nneighborMax
    mpds_geosClassicIndicatorInput.nneighborMax = geosclassic.new_int_array(int(ncategory))
    geosclassic.mpds_set_int_vector_from_array(
        mpds_geosClassicIndicatorInput.nneighborMax, 0,
        np.asarray(nneighborMax).reshape(int(ncategory)))

    # mpds_geosClassicIndicatorInput.searchNeighborhoodSortMode
    mpds_geosClassicIndicatorInput.searchNeighborhoodSortMode = geosclassic.new_int_array(int(ncategory))
    geosclassic.mpds_set_int_vector_from_array(
        mpds_geosClassicIndicatorInput.searchNeighborhoodSortMode, 0,
        np.asarray(searchNeighborhoodSortMode).reshape(int(ncategory)))

    # mpds_geosClassicIndicatorInput.ndataImage and mpds_geosClassicIndicatorInput.dataImage
    if dataImage is None:
        mpds_geosClassicIndicatorInput.ndataImage = 0
    else:
        dataImage = np.asarray(dataImage).reshape(-1)
        n = len(dataImage)
        mpds_geosClassicIndicatorInput.ndataImage = n
        mpds_geosClassicIndicatorInput.dataImage = geosclassic.new_MPDS_IMAGE_array(n)
        for i, dataIm in enumerate(dataImage):
            im_c = img_py2C(dataIm)
            geosclassic.MPDS_IMAGE_array_setitem(mpds_geosClassicIndicatorInput.dataImage, i, im_c)
            # geosclassic.free_MPDS_IMAGE(im_c)
            #
            # geosclassic.MPDS_IMAGE_array_setitem(mpds_geosClassicIndicatorInput.dataImage, i, img_py2C(dataIm))

    # mpds_geosClassicIndicatorInput.ndataPointSet and mpds_geosClassicIndicatorInput.dataPointSet
    if dataPointSet is None:
        mpds_geosClassicIndicatorInput.ndataPointSet = 0
    else:
        dataPointSet = np.asarray(dataPointSet).reshape(-1)
        n = len(dataPointSet)
        mpds_geosClassicIndicatorInput.ndataPointSet = n
        mpds_geosClassicIndicatorInput.dataPointSet = geosclassic.new_MPDS_POINTSET_array(n)
        for i, dataPS in enumerate(dataPointSet):
            ps_c = ps_py2C(dataPS)
            geosclassic.MPDS_POINTSET_array_setitem(mpds_geosClassicIndicatorInput.dataPointSet, i, ps_c)
            # geosclassic.free_MPDS_POINTSET(ps_c)
            #
            # geosclassic.MPDS_POINTSET_array_setitem(mpds_geosClassicIndicatorInput.dataPointSet, i, ps_py2C(dataPS))

    # mpds_geosClassicIndicatorInput.maskImageFlag and mpds_geosClassicIndicatorInput.maskImage
    if mask is None:
        mpds_geosClassicIndicatorInput.maskImageFlag = geosclassic.FALSE
    else:
        mpds_geosClassicIndicatorInput.maskImageFlag = geosclassic.TRUE
        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=1, val=mask)
        mpds_geosClassicIndicatorInput.maskImage = img_py2C(im)

    # mpds_geosClassicIndicatorInput.probabilityUsage, mpds_geosClassicIndicatorInput.probabilityValue, mpds_geosClassicIndicatorInput.probabilityImage
    if probability is None:
        mpds_geosClassicIndicatorInput.probabilityUsage = 0
    elif probability.size == ncategory:
        mpds_geosClassicIndicatorInput.probabilityUsage = 1
        # mpds_geosClassicIndicatorInput.probabilityValue
        mpds_geosClassicIndicatorInput.probabilityValue = geosclassic.new_real_array(int(ncategory))
        geosclassic.mpds_set_real_vector_from_array(
            mpds_geosClassicIndicatorInput.probabilityValue, 0,
            np.asarray(probability).reshape(int(ncategory)))
    elif probability.size == ncategory*nxyz:
        mpds_geosClassicIndicatorInput.probabilityUsage = 2
        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=ncategory, val=probability)
        mpds_geosClassicIndicatorInput.probabilityImage = img_py2C(im)
    else:
        print("ERROR: can not integrate 'probability' (not compatible with simulation grid)")
        return mpds_geosClassicIndicatorInput, False

    # mpds_geosClassicIndicatorInput.seed
    if seed is None:
        seed = np.random.randint(1,1000000)
    mpds_geosClassicIndicatorInput.seed = int(seed)

    # mpds_geosClassicIndicatorInput.seedIncrement
    mpds_geosClassicIndicatorInput.seedIncrement = 1

    # mpds_geosClassicIndicatorInput.nrealization
    mpds_geosClassicIndicatorInput.nrealization = int(nreal)

    return mpds_geosClassicIndicatorInput, True
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def simulateIndicator1D(
        category_values,
        cov_model_for_category,
        dimension, spacing=1.0, origin=0.0,
        method='simple_kriging',
        nreal=1,
        probability=None,
        x=None, v=None,
        mask=None,
        searchRadiusRelative=1.,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        seed=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2):
    """
    Generates 1D simulations (Sequential Indicator Simulation, SIS) based on
    simple or ordinary kriging.

    :param category_values:
                        (sequence of floats or ints) list of the category values;
                            let ncategory be the length of the list, then:
                            - if ncategory == 1:
                                - the unique category value given must not be
                                equal to 0
                                - it is used for a binary case with values
                                ("unique category value", 0), where 0 indicates
                                the absence of the considered medium
                                - conditioning data values should be
                                "unique category value" or 0
                            - if ncategory >= 2:
                                - it is used for a multi-category case with given
                                values (distinct)
                                - conditioning data values should be in the list
                                of given values

    :param cov_model_for_category:
                        (sequence of covariance model of length ncategory (see
                            category_values), or one covariance model, recycled)
                            a covariance model per category,
                            with entry for covariance model:
                                (CovModel1D class) covariance model in 1D, see
                                    definition of the class in
                                    module geone.covModel

    :param dimension:   (int) nx, number of cells
    :param spacing:     (float) sx, spacing between two adjacent cells
    :param origin:      (float) ox, origin of the 1D simulation
                            - used for localizing the conditioning points
    :param method:      (string) indicates the method used:
                            - 'simple_kriging':
                                simulation based on simple kriging
                            - 'ordinary_kriging':
                                simulation based on ordinary kriging
    :param nreal:       (int) number of realizations

    :param probability: (None or sequence of floats of length ncategory or
                            ndarray of floats) probability for each category:
                                - None :
                                    proportion of each category in the hard data
                                    values (stationary), (uniform distribution if
                                    no hard data)
                                - sequence of floats of length ncategory:
                                    for stationary probabilities (set manually),
                                    if ncategory > 1, the sum of the probabilities
                                    must be equal to one
                                - ndarray: for non stationary probabilities,
                                    must contain ncategory * ngrid_cells entries
                                    where ngrid_cells is the number of grid cells
                                    (reshaped if needed); the first ngrid_cells
                                    entries are the probabities for the first
                                    category in the simulation grid, the next
                                    ngrid_cells entries those of the second
                                    category, and so on
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param x:           (1-dimensional array or float or None) coordinate of
                            conditioning points for hard data
    :param v:           (1-dimensional array or float or None) value
                            at conditioning points for hard data
                            (same type as x)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not
                            simulated cell (nunber of entries should be equal to
                            the number of grid cells)

    :param searchRadiusRelative:
                        (sequence of ncategory floats (or float, recycled))
                            indicating how restricting the search ellipsoid
                            (should be positive) for each category: let r_i be
                            the ranges of the covariance model along its main
                            axes, if x is a node to be simulated, a node y is
                            taken into account iff it is within the ellipsoid
                            centered at x of half-axes searchRadiusRelative * r_i
                            Note:
                                - if a range is a variable parameter, its maximal
                                value over the simulation grid is considered

    :param nneighborMax:
                        (sequence of ncategory ints (or int, recycled)) maximum
                            number of nodes retrieved from the search ellipsoid,
                            for each category, set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (sequence of ncategory ints (or int, recycled))
                            indicating how to sort the search neighboorhood nodes
                            (neighbors) for each category, they are sorted in
                            increasing order according to:
                            - searchNeighborhoodSortMode = 0:
                                distance in the usual axes system
                            - searchNeighborhoodSortMode = 1:
                                distance in the axes sytem supporting the
                                covariance model and accounting for anisotropy
                                given by the ranges
                            - searchNeighborhoodSortMode = 2:
                                minus the evaluation of the covariance model
                            Notes:
                            - if the covariance model has any variable parameter
                                (non-stationary), then
                                searchNeighborhoodSortMode = 2 is not allowed
                            - if the covariance model has any range or angle set
                                as a variable parameter, then
                                searchNeighborhoodSortMode must be set to 0
                            - greatest possible value as default

    :param seed:        (int or None) initial seed, if None an initial seed
                            between 1 and 999999 is generated with
                            numpy.random.randint

    :param outputReportFile:
                        (string or None) name of the report file,
                            if None: no report file

    :param nthreads:
                        (int) number of thread(s) to use for "GeosClassicSim"
                            program (C), (nthreads = -n <= 0: for maximal number
                            of threads except n, but at least 1)
    :param verbose:
                (int) verbose mode, integer >=0, higher implies more display
                    - 0: no display
                    - 1: only errors
                    - 2: errors and warnings (+ some info)
                    - 3 (or >2): all information

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img class) output image, with image.nv=nreal variables (each
                    variable is one realization)
                    (image is None if mpds_geosClassicOutput->outputImage is
                    NULL)
        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)
        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    # --- Set grid geometry and varname
    # Set grid geometry
    nx, ny, nz = dimension, 1, 1
    sx, sy, sz = spacing, 1.0, 1.0
    ox, oy, oz = origin, 0.0, 0.0

    nxy = nx * ny
    nxyz = nxy * nz

    # spatial dimension
    space_dim = 1

    # Set varname
    varname = 'V0'

    # --- Check and prepare parameters
    # category_values and ncategory (computed)
    try:
        category_values = np.asarray(category_values, dtype='float').reshape(-1)
    except:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR1D): 'category_values' is not valid")
        return None

    ncategory = len(category_values)
    if ncategory <= 0:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR1D): 'category_values' is empty")
        return None

    # cov_model_for_category
    cm_for_cat = cov_model_for_category # no need to work on a copy in 1D

    cm_for_cat = np.asarray(cm_for_cat).reshape(-1)
    if len(cm_for_cat) == 1:
        cm_for_cat = np.repeat(cm_for_cat, ncategory)
    elif len(cm_for_cat) != ncategory:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR1D): 'cov_model_for_category' of invalid length")
        return None
    if not np.all([isinstance(c, gcm.CovModel1D) for c in cm_for_cat]):
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR1D): 'cov_model_for_category' should contains CovModel1D objects")
        return None

    for cov_model in cm_for_cat:
        for el in cov_model.elem:
            # weight
            w = el[1]['w']
            if np.size(w) != 1 and np.size(w) != nxyz:
                if verbose > 0:
                    print("ERROR (SIMULATE_INDICATOR1D): covariance model: weight ('w') not compatible with simulation grid")
                return None
            # ranges
            if 'r' in el[1].keys():
                r  = el[1]['r']
                if np.size(r) != 1 and np.size(r) != nxyz:
                    if verbose > 0:
                        print("ERROR (SIMULATE_INDICATOR1D): covariance model: range ('r') not compatible with simulation grid")
                    return None
            # additional parameter (s)
            if 's' in el[1].keys():
                s  = el[1]['s']
                if np.size(s) != 1 and np.size(s) != nxyz:
                    if verbose > 0:
                        print("ERROR (SIMULATE_INDICATOR1D): covariance model: parameter ('s') not compatible with simulation grid")
                    return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     if verbose > 0:
    #         print("ERROR (SIMULATE_INDICATOR1D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 3
    elif method == 'ordinary_kriging':
        computationMode = 2
    else:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR1D): 'method' is not valid")
        return None

    # data points: x, v
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            if verbose > 0:
                print("(ERROR (SIMULATE_INDICATOR1D): length of 'v' is not valid")
            return None
        xc = x
        yc = np.ones_like(xc) * oy + 0.5 * sy
        zc = np.ones_like(xc) * oz + 0.5 * sz
        dataPointSet.append(
            PointSet(npt=v.shape[0], nv=4, val=np.array((xc, yc, zc, v)), varname=['X', 'Y', 'Z', varname])
            )

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            if verbose > 0:
                print("ERROR (SIMULATE_INDICATOR1D): 'mask' is not valid")
            return None

    # Check parameters - searchRadiusRelative
    searchRadiusRelative = np.asarray(searchRadiusRelative, dtype='float').reshape(-1)
    if len(searchRadiusRelative) == 1:
        searchRadiusRelative = np.repeat(searchRadiusRelative, ncategory)
    elif len(searchRadiusRelative) != ncategory:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR1D): 'searchRadiusRelative' of invalid length")
        return None

    for srr in searchRadiusRelative:
        if srr < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
            if verbose > 0:
                print("ERROR (SIMULATE_INDICATOR1D): a 'searchRadiusRelative' is too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
            return None

    # Check parameters - nneighborMax
    nneighborMax = np.asarray(nneighborMax, dtype='intc').reshape(-1)
    if len(nneighborMax) == 1:
        nneighborMax = np.repeat(nneighborMax, ncategory)
    elif len(nneighborMax) != ncategory:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR1D): 'nneighborMax' of invalid length")
        return None

    for nn in nneighborMax:
        if nn != -1 and nn <= 0:
            if verbose > 0:
                print("ERROR (SIMULATE_INDICATOR1D): any 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
            return None

    # Check parameters - searchNeighborhoodSortMode
    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode).reshape(-1)
    if len(searchNeighborhoodSortMode) == 1:
        searchNeighborhoodSortMode = np.repeat(searchNeighborhoodSortMode, ncategory)
    elif len(searchNeighborhoodSortMode) != ncategory:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR1D): 'searchNeighborhoodSortMode' of invalid length")
        return None

    for i in range(ncategory):
        if searchNeighborhoodSortMode[i] is None:
            # set greatest possible value
            if cm_for_cat[i].is_stationary():
                searchNeighborhoodSortMode[i] = 2
            elif cm_for_cat[i].is_orientation_stationary() and cm_for_cat[i].is_range_stationary():
                searchNeighborhoodSortMode[i] = 1
            else:
                searchNeighborhoodSortMode[i] = 0
        else:
            if searchNeighborhoodSortMode[i] == 2:
                if not cm_for_cat[i].is_stationary():
                    if verbose > 0:
                        print("ERROR (SIMULATE_INDICATOR1D): 'searchNeighborhoodSortMode set to 2' not allowed with non-stationary covariance model")
                    return None
            elif searchNeighborhoodSortMode[i] == 1:
                if not cm_for_cat[i].is_orientation_stationary() or not cm_for_cat[i].is_range_stationary():
                    if verbose > 0:
                        print("ERROR (SIMULATE_INDICATOR1D): 'searchNeighborhoodSortMode set to 1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                    return None

    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode, dtype='intc')

    # Check parameters - probability
    if probability is not None:
        # if method == 'ordinary_kriging':
        #     if verbose > 0:
        #         print("ERROR (SIMULATE_INDICATOR1D): specifying 'probability' not allowed with ordinary kriging")
        #     return None
        probability = np.asarray(probability, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if probability.size not in (ncategory, ncategory*nxyz):
            if verbose > 0:
                print("ERROR (SIMULATE_INDICATOR1D): size of 'probability' is not valid")
            return None

    # Check parameters - nreal
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose >= 1:
            if verbose > 0:
                print('SIMULATE_INDICATOR1D: nreal <= 0: nothing to do!')
        return None

    # --- Fill mpds_geosClassicInput structure (C)
    mpds_geosClassicIndicatorInput, flag = fill_mpds_geosClassicIndicatorInput(
        space_dim,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        varname,
        ncategory,
        category_values,
        outputReportFile,
        computationMode,
        cm_for_cat,
        None,
        dataPointSet,
        mask,
        probability,
        searchRadiusRelative,
        nneighborMax,
        searchNeighborhoodSortMode,
        seed,
        nreal)

    if not flag:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR1D): can not fill input structure!")
        return None

    # --- Prepare mpds_geosClassicIOutput structure (C)
    # Allocate mpds_geosClassicOutput
    mpds_geosClassicOutput = geosclassic.malloc_MPDS_GEOSCLASSICOUTPUT()

    # Init mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicInitGeosClassicOutput(mpds_geosClassicOutput)

    # --- Set progress monitor
    mpds_progressMonitor = geosclassic.malloc_MPDS_PROGRESSMONITOR()
    geosclassic.MPDSInitProgressMonitor(mpds_progressMonitor)

    # Set function to update progress monitor:
    # according to geosclassic.MPDS_SHOW_PROGRESS_MONITOR set to 4 for compilation of py module
    # the function
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor1_ptr: warning only
    if verbose < 3:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

    # Free memory on C side: mpds_geosClassicIndicatorInput
    geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
    #geosclassic.MPDSFree(mpds_geosClassicIndicatorInput)
    geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)

    if err:
        if verbose > 0:
            err_message = geosclassic.mpds_get_error_message(-err)
            err_message = err_message.replace('\n', '')
            print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 2 and geosclassic_output and geosclassic_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(geosclassic_output['nwarning']))
        for i, warning_message in enumerate(geosclassic_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def simulateIndicator1D_mp(
        category_values,
        cov_model_for_category,
        dimension, spacing=1.0, origin=0.0,
        method='simple_kriging',
        nreal=1,
        probability=None,
        x=None, v=None,
        mask=None,
        searchRadiusRelative=1.,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        seed=None,
        outputReportFile=None,
        treat_image_one_by_one=False,
        nproc=None, nthreads_per_proc=None,
        verbose=2):
    """
    Generates 1D simulations (Sequential Indicator Simulation, SIS) based on
    simple or ordinary kriging.

    Launches multiple processes (based on multiprocessing package):
        - nproc parallel processes using each one nthreads_per_proc threads will
            be launched [parallel calls of the function simulateIndicator1D],
        - the set of realizations (specified by nreal) is
            distributed in a balanced way over the processes,
        - in terms of resources, this implies the use of
            nproc * nthreads_per_proc cpu(s).

    :param category_values:
                        (sequence of floats or ints) list of the category values;
                            let ncategory be the length of the list, then:
                            - if ncategory == 1:
                                - the unique category value given must not be
                                equal to 0
                                - it is used for a binary case with values
                                ("unique category value", 0), where 0 indicates
                                the absence of the considered medium
                                - conditioning data values should be
                                "unique category value" or 0
                            - if ncategory >= 2:
                                - it is used for a multi-category case with given
                                values (distinct)
                                - conditioning data values should be in the list
                                of given values

    :param cov_model_for_category:
                        (sequence of covariance model of length ncategory (see
                            category_values), or one covariance model, recycled)
                            a covariance model per category,
                            with entry for covariance model:
                                (CovModel1D class) covariance model in 1D, see
                                    definition of the class in
                                    module geone.covModel

    :param dimension:   (int) nx, number of cells
    :param spacing:     (float) sx, spacing between two adjacent cells
    :param origin:      (float) ox, origin of the 1D simulation
                            - used for localizing the conditioning points
    :param method:      (string) indicates the method used:
                            - 'simple_kriging':
                                simulation based on simple kriging
                            - 'ordinary_kriging':
                                simulation based on ordinary kriging
    :param nreal:       (int) number of realizations

    :param probability: (None or sequence of floats of length ncategory or
                            ndarray of floats) probability for each category:
                                - None :
                                    proportion of each category in the hard data
                                    values (stationary), (uniform distribution if
                                    no hard data)
                                - sequence of floats of length ncategory:
                                    for stationary probabilities (set manually),
                                    if ncategory > 1, the sum of the probabilities
                                    must be equal to one
                                - ndarray: for non stationary probabilities,
                                    must contain ncategory * ngrid_cells entries
                                    where ngrid_cells is the number of grid cells
                                    (reshaped if needed); the first ngrid_cells
                                    entries are the probabities for the first
                                    category in the simulation grid, the next
                                    ngrid_cells entries those of the second
                                    category, and so on
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param x:           (1-dimensional array or float or None) coordinate of
                            conditioning points for hard data
    :param v:           (1-dimensional array or float or None) value
                            at conditioning points for hard data
                            (same type as x)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not
                            simulated cell (nunber of entries should be equal to
                            the number of grid cells)

    :param searchRadiusRelative:
                        (sequence of ncategory floats (or float, recycled))
                            indicating how restricting the search ellipsoid
                            (should be positive) for each category: let r_i be
                            the ranges of the covariance model along its main
                            axes, if x is a node to be simulated, a node y is
                            taken into account iff it is within the ellipsoid
                            centered at x of half-axes searchRadiusRelative * r_i
                            Note:
                                - if a range is a variable parameter, its maximal
                                value over the simulation grid is considered

    :param nneighborMax:
                        (sequence of ncategory ints (or int, recycled)) maximum
                            number of nodes retrieved from the search ellipsoid,
                            for each category, set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (sequence of ncategory ints (or int, recycled))
                            indicating how to sort the search neighboorhood nodes
                            (neighbors) for each category, they are sorted in
                            increasing order according to:
                            - searchNeighborhoodSortMode = 0:
                                distance in the usual axes system
                            - searchNeighborhoodSortMode = 1:
                                distance in the axes sytem supporting the
                                covariance model and accounting for anisotropy
                                given by the ranges
                            - searchNeighborhoodSortMode = 2:
                                minus the evaluation of the covariance model
                            Notes:
                            - if the covariance model has any variable parameter
                                (non-stationary), then
                                searchNeighborhoodSortMode = 2 is not allowed
                            - if the covariance model has any range or angle set
                                as a variable parameter, then
                                searchNeighborhoodSortMode must be set to 0
                            - greatest possible value as default

    :param seed:        (int or None) initial seed, if None an initial seed
                            between 1 and 999999 is generated with
                            numpy.random.randint

    :param outputReportFile:
                        (string or None) name of the report file,
                            if None: no report file [if given, a suffix
                            ".<process_index>" is added for the report file of
                            each process]

    :param treat_image_one_by_one:
                        (bool) keyword argument passed to the function
                            geone.img.gatherImages
                            - if False (default) images (result of each process)
                            are gathered at once, and then removed (faster)
                            - if True, images (result of each process) are
                            gathered one by one, i.e. successively gathered and
                            removed (slower, may save memory)

    :param nproc:
                (int) number of processes (can be modified in the function)
                    nproc = None: nproc is set to
                        min(nmax-1, nreal) (but at least 1),
                    where nmax is the total number of cpu(s) of the system
                    (retrieved by multiprocessing.cpu_count())

    :param nthreads_per_proc:
                (int) number of thread(s) per process (should be > 0 or None):
                    nthreads_per_proc = None: nthreads_per_proc is automatically
                    computed as the maximal integer (but at least 1) such that
                            nproc * nthreads_per_proc <= nmax-1
                    where nmax is the total number of cpu(s) of the system
                    (retrieved by multiprocessing.cpu_count())

    :param verbose:
                (int) indicates what information is displayed:
                    - 0: no display
                    - 1: only errors (and note(s))
                    - 2: version and warning(s) encountered

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img class) output image, with image.nv=nreal variables (each
                    variable is one realization)
                    (image is None if mpds_geosClassicOutput->outputImage is
                    NULL)
        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)
        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    # Set number of processes: nproc
    if nproc is None:
        nproc = max(min(multiprocessing.cpu_count()-1, nreal), 1)
    else:
        nproc_tmp = nproc
        nproc = max(min(int(nproc), nreal), 1)
        if verbose > 0 and nproc != nproc_tmp:
            print('NOTE: number of processes has been changed (now: nproc={})'.format(nproc))

    # Set number of threads per process: nth
    if nthreads_per_proc is None:
        nth = max(int(np.floor((multiprocessing.cpu_count()-1) / nproc)), 1)
    else:
        nth = max(int(nthreads_per_proc), 1)
        if verbose > 0 and nth != nthreads_per_proc:
            print('NOTE: number of threads per process has been changed (now: nthreads_per_proc={})'.format(nth))

    if verbose > 0 and nproc * nth > multiprocessing.cpu_count():
        print('NOTE: total number of cpu(s) used will exceed number of cpu(s) of the system...')

    # Set the distribution of the realizations over the processes
    # Condider the Euclidean division of nreal by nproc:
    #     nreal = q * nproc + r, with 0 <= r < nproc
    # Then, (q+1) realizations will be done on process 0, 1, ..., r-1, and q realization on process r, ..., nproc-1
    # Define the list real_index_proc of length (nproc+1) such that
    #   real_index_proc[i], ..., real_index_proc[i+1] - 1 : are the realization indices run on process i
    q, r = np.divmod(nreal, nproc)
    real_index_proc = [i*q + min(i, r) for i in range(nproc+1)]

    if verbose >= 2:
        print('Geos-Classic running on {} process(es)... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(nproc, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching geos-classic...

    # mpds_geosClassicInput.seed
    if seed is None:
        seed = np.random.randint(1,1000000)
    seed = int(seed)

    outputReportFile_p = None

    # Set pool of nproc workers
    pool = multiprocessing.Pool(nproc)
    out_pool = []
    for i in range(nproc):
        # Adapt input for i-th process
        nreal_p = real_index_proc[i+1] - real_index_proc[i]
        seed_p = seed + real_index_proc[i]
        if outputReportFile is not None:
            outputReportFile_p = outputReportFile + f'.{i}'
        if i==0:
            verbose_p = min(verbose, 1) # allow to print error for process i
        else:
            verbose_p = 0
        # Launch geos-classic (i-th process)
        out_pool.append(
            pool.apply_async(simulateIndicator1D,
                args=(category_values,
                cov_model_for_category,
                dimension, spacing, origin,
                method,
                nreal_p,                     # nreal (adjusted)
                probability,
                x, v,
                mask,
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                seed_p,                      # seed (adjusted)
                outputReportFile_p,          # outputReportFile (adjusted)
                nth,                         # nthreads
                verbose_p)                   # verbose (adjusted)
                )
            )

    # Properly end working process
    pool.close() # Prevents any more tasks from being submitted to the pool,
    pool.join()  # then, wait for the worker processes to exit.

    # Get result from each process
    geosclassic_output_proc = [p.get() for p in out_pool]

    if np.any([out is None for out in geosclassic_output_proc]):
        return None

    image = None
    nwarning, warnings = None, None

    # Gather results from every process
    # image
    image = np.hstack([out['image'] for out in geosclassic_output_proc])
    # ... remove None entries
    image = image[[x is not None for x in image]]
    # .. set to None if every entry is None
    if np.all([x is None for x in image]):
        image = None

    # nwarning
    nwarning = np.sum([out['nwarning'] for out in geosclassic_output_proc])
    # warnings
    warnings = list(np.unique(np.hstack([out['warnings'] for out in geosclassic_output_proc])))

    # Gather images and adjust variable names
    if image is not None:
        all_image = img.gatherImages(image, keep_varname=True, rem_var_from_source=True, treat_image_one_by_one=treat_image_one_by_one)
        ndigit = geosclassic.MPDS_GEOS_CLASSIC_NB_DIGIT_FOR_REALIZATION_NUMBER
        for j in range(all_image.nv):
            all_image.varname[j] = all_image.varname[j][:-ndigit] + f'{j:0{ndigit}d}'

    geosclassic_output = {'image':all_image, 'nwarning':nwarning, 'warnings':warnings}

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete (all process(es))')

    # Show (print) encountered warnings
    if verbose >= 2 and geosclassic_output and geosclassic_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(geosclassic_output['nwarning']))
        for i, warning_message in enumerate(geosclassic_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def simulateIndicator2D(
        category_values,
        cov_model_for_category,
        dimension, spacing=(1.0, 1.0), origin=(0.0, 0.0),
        method='simple_kriging',
        nreal=1,
        probability=None,
        x=None, v=None,
        mask=None,
        searchRadiusRelative=1.,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        seed=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2):
    """
    Generates 2D simulations (Sequential Indicator Simulation, SIS) based on
    simple or ordinary kriging.

    :param category_values:
                        (sequence of floats or ints) list of the category values;
                            let ncategory be the length of the list, then:
                            - if ncategory == 1:
                                - the unique category value given must not be
                                equal to 0
                                - it is used for a binary case with values
                                ("unique category value", 0), where 0 indicates
                                the absence of the considered medium
                                - conditioning data values should be
                                "unique category value" or 0
                            - if ncategory >= 2:
                                - it is used for a multi-category case with given
                                values (distinct)
                                - conditioning data values should be in the list
                                of given values

    :param cov_model_for_category:
                        (sequence of covariance model of length ncategory (see
                            category_values), or one covariance model, recycled)
                            a covariance model per category,
                            with entry for covariance model:
                                (CovModel2D class) covariance model in 2D, see
                                    definition of the class in module
                                    geone.covModel
                            or
                                (CovModel1D class) covariance model in 1D, see
                                    definition of the class in module
                                    geone.covModel, it is then transformed to an
                                    isotropic (omni-directional) covariance model
                                    in 2D

    :param dimension:   (sequence of 2 ints) (nx, ny), number of cells
                            in x-, y-axis direction
    :param spacing:     (sequence of 2 floats) (sx, sy), spacing between
                            two adjacent cells in x-, y-axis direction
    :param origin:      (sequence of 2 floats) (ox, oy), origin of the 2D
                            simulation - used for localizing the conditioning
                            points
    :param method:      (string) indicates the method used:
                            - 'simple_kriging':
                                simulation based on simple kriging
                            - 'ordinary_kriging':
                                simulation based on ordinary kriging
    :param nreal:       (int) number of realizations

    :param probability: (None or sequence of floats of length ncategory or
                            ndarray of floats) probability for each category:
                                - None :
                                    proportion of each category in the hard data
                                    values (stationary), (uniform distribution if
                                    no hard data)
                                - sequence of floats of length ncategory:
                                    for stationary probabilities (set manually),
                                    if ncategory > 1, the sum of the probabilities
                                    must be equal to one
                                - ndarray: for non stationary probabilities,
                                    must contain ncategory * ngrid_cells entries
                                    where ngrid_cells is the number of grid cells
                                    (reshaped if needed); the first ngrid_cells
                                    entries are the probabities for the first
                                    category in the simulation grid, the next
                                    ngrid_cells entries those of the second
                                    category, and so on
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param x:           (2-dimensional array of dim n x 2, or
                            1-dimensional array of dim 2 or None) coordinate of
                            conditioning points for hard data
    :param v:           (1-dimensional array or float or None) value
                            at conditioning points for hard data
                            (same type as x)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not
                            simulated cell (nunber of entries should be equal to
                            the number of grid cells)

    :param searchRadiusRelative:
                        (sequence of ncategory floats (or float, recycled))
                            indicating how restricting the search ellipsoid
                            (should be positive) for each category: let r_i be
                            the ranges of the covariance model along its main
                            axes, if x is a node to be simulated, a node y is
                            taken into account iff it is within the ellipsoid
                            centered at x of half-axes searchRadiusRelative * r_i
                            Note:
                                - if a range is a variable parameter, its maximal
                                value over the simulation grid is considered
                                - if orientation of the covariance model is
                                non-stationary, a "circular search neighborhood"
                                is considered with the radius set to the maximum
                                of all ranges

    :param nneighborMax:
                        (sequence of ncategory ints (or int, recycled)) maximum
                            number of nodes retrieved from the search ellipsoid,
                            for each category, set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (sequence of ncategory ints (or int, recycled))
                            indicating how to sort the search neighboorhood nodes
                            (neighbors) for each category, they are sorted in
                            increasing order according to:
                            - searchNeighborhoodSortMode = 0:
                                distance in the usual axes system
                            - searchNeighborhoodSortMode = 1:
                                distance in the axes sytem supporting the
                                covariance model and accounting for anisotropy
                                given by the ranges
                            - searchNeighborhoodSortMode = 2:
                                minus the evaluation of the covariance model
                            Notes:
                            - if the covariance model has any variable parameter
                                (non-stationary), then
                                searchNeighborhoodSortMode = 2 is not allowed
                            - if the covariance model has any range or angle set
                                as a variable parameter, then
                                searchNeighborhoodSortMode must be set to 0
                            - greatest possible value as default

    :param seed:        (int or None) initial seed, if None an initial seed
                            between 1 and 999999 is generated with
                            numpy.random.randint

    :param outputReportFile:
                        (string or None) name of the report file,
                            if None: no report file

    :param nthreads:
                        (int) number of thread(s) to use for "GeosClassicSim"
                            program (C), (nthreads = -n <= 0: for maximal number
                            of threads except n, but at least 1)
    :param verbose:
                (int) verbose mode, integer >=0, higher implies more display
                    - 0: no display
                    - 1: only errors
                    - 2: errors and warnings (+ some info)
                    - 3 (or >2): all information

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img class) output image, with image.nv=nreal variables (each
                    variable is one realization)
                    (image is None if mpds_geosClassicOutput->outputImage is
                    NULL)
        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)
        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    # --- Set grid geometry and varname
    # Set grid geometry
    nx, ny, nz = *dimension, 1
    sx, sy, sz = *spacing, 1.0
    ox, oy, oz = *origin, 0.0

    nxy = nx * ny
    nxyz = nxy * nz

    # spatial dimension
    space_dim = 2

    # Set varname
    varname = 'V0'

    # --- Check and prepare parameters
    # category_values and ncategory (computed)
    try:
        category_values = np.asarray(category_values, dtype='float').reshape(-1)
    except:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR2D): 'category_values' is not valid")
        return None

    ncategory = len(category_values)
    if ncategory <= 0:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR2D): 'category_values' is empty")
        return None

    # cov_model_for_category
    cov_model_for_category = np.asarray(cov_model_for_category).reshape(-1)
    if not np.all([isinstance(c, gcm.CovModel2D) for c in cov_model_for_category]):
        # cov model will be converted:
        #    as applying modification in an array is persistent at exit,
        #    work on a copy to ensure no modification of the initial entry
        cm_for_cat = np.deepcopy(cov_model_for_category)
    else:
        cm_for_cat = cov_model_for_category

    cm_for_cat = np.asarray(cm_for_cat).reshape(-1)
    for i in range(len(cm_for_cat)):
        if isinstance(cm_for_cat[i], gcm.CovModel1D):
            cm_for_cat[i] = gcm.covModel1D_to_covModel2D(cm_for_cat[i]) # convert model 1D in 2D
    if len(cm_for_cat) == 1:
        cm_for_cat = np.repeat(cm_for_cat, ncategory)
    elif len(cm_for_cat) != ncategory:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR2D): 'cov_model_for_category' of invalid length")
        return None
    if not np.all([isinstance(c, gcm.CovModel2D) for c in cm_for_cat]):
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR2D): 'cov_model_for_category' should contains CovModel2D objects")
        return None

    for cov_model in cm_for_cat:
        for el in cov_model.elem:
            # weight
            w = el[1]['w']
            if np.size(w) != 1 and np.size(w) != nxyz:
                if verbose > 0:
                    print("ERROR (SIMULATE_INDICATOR2D): covariance model: weight ('w') not compatible with simulation grid")
                return None
            # ranges
            if 'r' in el[1].keys():
                for r in el[1]['r']:
                    if np.size(r) != 1 and np.size(r) != nxyz:
                        if verbose > 0:
                            print("ERROR (SIMULATE_INDICATOR2D): covariance model: range ('r') not compatible with simulation grid")
                        return None
            # additional parameter (s)
            if 's' in el[1].keys():
                s  = el[1]['s']
                if np.size(s) != 1 and np.size(s) != nxyz:
                    if verbose > 0:
                        print("ERROR (SIMULATE_INDICATOR2D): covariance model: parameter ('s') not compatible with simulation grid")
                    return None

        # alpha
        angle = cov_model.alpha
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            if verbose > 0:
                print("ERROR (SIMULATE_INDICATOR2D): covariance model: angle (alpha) not compatible with simulation grid")
            return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     if verbose > 0:
    #         print("ERROR (SIMULATE_INDICATOR2D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 3
    elif method == 'ordinary_kriging':
        computationMode = 2
    else:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR2D): 'method' is not valid")
        return None

    # data points: x, v
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 2) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            if verbose > 0:
                print("(ERROR (SIMULATE_INDICATOR2D): length of 'v' is not valid")
            return None
        xc = x[:,0]
        yc = x[:,1]
        zc = np.ones_like(xc) * oz + 0.5 * sz
        dataPointSet.append(
            PointSet(npt=v.shape[0], nv=4, val=np.array((xc, yc, zc, v)), varname=['X', 'Y', 'Z', varname])
            )

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            if verbose > 0:
                print("ERROR (SIMULATE_INDICATOR2D): 'mask' is not valid")
            return None

    # Check parameters - searchRadiusRelative
    searchRadiusRelative = np.asarray(searchRadiusRelative, dtype='float').reshape(-1)
    if len(searchRadiusRelative) == 1:
        searchRadiusRelative = np.repeat(searchRadiusRelative, ncategory)
    elif len(searchRadiusRelative) != ncategory:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR2D): 'searchRadiusRelative' of invalid length")
        return None

    for srr in searchRadiusRelative:
        if srr < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
            if verbose > 0:
                print("ERROR (SIMULATE_INDICATOR2D): a 'searchRadiusRelative' is too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
            return None

    # Check parameters - nneighborMax
    nneighborMax = np.asarray(nneighborMax, dtype='intc').reshape(-1)
    if len(nneighborMax) == 1:
        nneighborMax = np.repeat(nneighborMax, ncategory)
    elif len(nneighborMax) != ncategory:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR2D): 'nneighborMax' of invalid length")
        return None

    for nn in nneighborMax:
        if nn != -1 and nn <= 0:
            if verbose > 0:
                print("ERROR (SIMULATE_INDICATOR2D): any 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
            return None

    # Check parameters - searchNeighborhoodSortMode
    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode).reshape(-1)
    if len(searchNeighborhoodSortMode) == 1:
        searchNeighborhoodSortMode = np.repeat(searchNeighborhoodSortMode, ncategory)
    elif len(searchNeighborhoodSortMode) != ncategory:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR2D): 'searchNeighborhoodSortMode' of invalid length")
        return None

    for i in range(ncategory):
        if searchNeighborhoodSortMode[i] is None:
            # set greatest possible value
            if cm_for_cat[i].is_stationary():
                searchNeighborhoodSortMode[i] = 2
            elif cm_for_cat[i].is_orientation_stationary() and cm_for_cat[i].is_range_stationary():
                searchNeighborhoodSortMode[i] = 1
            else:
                searchNeighborhoodSortMode[i] = 0
        else:
            if searchNeighborhoodSortMode[i] == 2:
                if not cm_for_cat[i].is_stationary():
                    if verbose > 0:
                        print("ERROR (SIMULATE_INDICATOR2D): 'searchNeighborhoodSortMode set to 2' not allowed with non-stationary covariance model")
                    return None
            elif searchNeighborhoodSortMode[i] == 1:
                if not cm_for_cat[i].is_orientation_stationary() or not cm_for_cat[i].is_range_stationary():
                    if verbose > 0:
                        print("ERROR (SIMULATE_INDICATOR2D): 'searchNeighborhoodSortMode set to 1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                    return None

    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode, dtype='intc')

    # Check parameters - probability
    if probability is not None:
        # if method == 'ordinary_kriging':
        #     if verbose > 0:
        #         print("ERROR (SIMULATE_INDICATOR2D): specifying 'probability' not allowed with ordinary kriging")
        #     return None
        probability = np.asarray(probability, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if probability.size not in (ncategory, ncategory*nxyz):
            if verbose > 0:
                print("ERROR (SIMULATE_INDICATOR2D): size of 'probability' is not valid")
            return None

    # Check parameters - nreal
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose >= 2:
            print('SIMULATE_INDICATOR2D: nreal <= 0: nothing to do!')
        return None

    # --- Fill mpds_geosClassicInput structure (C)
    mpds_geosClassicIndicatorInput, flag = fill_mpds_geosClassicIndicatorInput(
        space_dim,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        varname,
        ncategory,
        category_values,
        outputReportFile,
        computationMode,
        cm_for_cat,
        None,
        dataPointSet,
        mask,
        probability,
        searchRadiusRelative,
        nneighborMax,
        searchNeighborhoodSortMode,
        seed,
        nreal)

    if not flag:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR2D): can not fill input structure!")
        return None

    # --- Prepare mpds_geosClassicIOutput structure (C)
    # Allocate mpds_geosClassicOutput
    mpds_geosClassicOutput = geosclassic.malloc_MPDS_GEOSCLASSICOUTPUT()

    # Init mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicInitGeosClassicOutput(mpds_geosClassicOutput)

    # --- Set progress monitor
    mpds_progressMonitor = geosclassic.malloc_MPDS_PROGRESSMONITOR()
    geosclassic.MPDSInitProgressMonitor(mpds_progressMonitor)

    # Set function to update progress monitor:
    # according to geosclassic.MPDS_SHOW_PROGRESS_MONITOR set to 4 for compilation of py module
    # the function
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor1_ptr: warning only
    if verbose < 3:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

    # Free memory on C side: mpds_geosClassicIndicatorInput
    geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
    #geosclassic.MPDSFree(mpds_geosClassicIndicatorInput)
    geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)

    if err:
        if verbose > 0:
            err_message = geosclassic.mpds_get_error_message(-err)
            err_message = err_message.replace('\n', '')
            print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 2 and geosclassic_output and geosclassic_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(geosclassic_output['nwarning']))
        for i, warning_message in enumerate(geosclassic_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def simulateIndicator2D_mp(
        category_values,
        cov_model_for_category,
        dimension, spacing=(1.0, 1.0), origin=(0.0, 0.0),
        method='simple_kriging',
        nreal=1,
        probability=None,
        x=None, v=None,
        mask=None,
        searchRadiusRelative=1.,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        seed=None,
        outputReportFile=None,
        treat_image_one_by_one=False,
        nproc=None, nthreads_per_proc=None,
        verbose=2):
    """
    Generates 2D simulations (Sequential Indicator Simulation, SIS) based on
    simple or ordinary kriging.

    Launches multiple processes (based on multiprocessing package):
        - nproc parallel processes using each one nthreads_per_proc threads will
            be launched [parallel calls of the function simulateIndicator2D],
        - the set of realizations (specified by nreal) is
            distributed in a balanced way over the processes,
        - in terms of resources, this implies the use of
            nproc * nthreads_per_proc cpu(s).

    :param category_values:
                        (sequence of floats or ints) list of the category values;
                            let ncategory be the length of the list, then:
                            - if ncategory == 1:
                                - the unique category value given must not be
                                equal to 0
                                - it is used for a binary case with values
                                ("unique category value", 0), where 0 indicates
                                the absence of the considered medium
                                - conditioning data values should be
                                "unique category value" or 0
                            - if ncategory >= 2:
                                - it is used for a multi-category case with given
                                values (distinct)
                                - conditioning data values should be in the list
                                of given values

    :param cov_model_for_category:
                        (sequence of covariance model of length ncategory (see
                            category_values), or one covariance model, recycled)
                            a covariance model per category,
                            with entry for covariance model:
                                (CovModel2D class) covariance model in 2D, see
                                    definition of the class in module
                                    geone.covModel
                            or
                                (CovModel1D class) covariance model in 1D, see
                                    definition of the class in module
                                    geone.covModel, it is then transformed to an
                                    isotropic (omni-directional) covariance model
                                    in 2D

    :param dimension:   (sequence of 2 ints) (nx, ny), number of cells
                            in x-, y-axis direction
    :param spacing:     (sequence of 2 floats) (sx, sy), spacing between
                            two adjacent cells in x-, y-axis direction
    :param origin:      (sequence of 2 floats) (ox, oy), origin of the 2D
                            simulation - used for localizing the conditioning
                            points
    :param method:      (string) indicates the method used:
                            - 'simple_kriging':
                                simulation based on simple kriging
                            - 'ordinary_kriging':
                                simulation based on ordinary kriging
    :param nreal:       (int) number of realizations

    :param probability: (None or sequence of floats of length ncategory or
                            ndarray of floats) probability for each category:
                                - None :
                                    proportion of each category in the hard data
                                    values (stationary), (uniform distribution if
                                    no hard data)
                                - sequence of floats of length ncategory:
                                    for stationary probabilities (set manually),
                                    if ncategory > 1, the sum of the probabilities
                                    must be equal to one
                                - ndarray: for non stationary probabilities,
                                    must contain ncategory * ngrid_cells entries
                                    where ngrid_cells is the number of grid cells
                                    (reshaped if needed); the first ngrid_cells
                                    entries are the probabities for the first
                                    category in the simulation grid, the next
                                    ngrid_cells entries those of the second
                                    category, and so on
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param x:           (2-dimensional array of dim n x 2, or
                            1-dimensional array of dim 2 or None) coordinate of
                            conditioning points for hard data
    :param v:           (1-dimensional array or float or None) value
                            at conditioning points for hard data
                            (same type as x)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not
                            simulated cell (nunber of entries should be equal to
                            the number of grid cells)

    :param searchRadiusRelative:
                        (sequence of ncategory floats (or float, recycled))
                            indicating how restricting the search ellipsoid
                            (should be positive) for each category: let r_i be
                            the ranges of the covariance model along its main
                            axes, if x is a node to be simulated, a node y is
                            taken into account iff it is within the ellipsoid
                            centered at x of half-axes searchRadiusRelative * r_i
                            Note:
                                - if a range is a variable parameter, its maximal
                                value over the simulation grid is considered
                                - if orientation of the covariance model is
                                non-stationary, a "circular search neighborhood"
                                is considered with the radius set to the maximum
                                of all ranges

    :param nneighborMax:
                        (sequence of ncategory ints (or int, recycled)) maximum
                            number of nodes retrieved from the search ellipsoid,
                            for each category, set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (sequence of ncategory ints (or int, recycled))
                            indicating how to sort the search neighboorhood nodes
                            (neighbors) for each category, they are sorted in
                            increasing order according to:
                            - searchNeighborhoodSortMode = 0:
                                distance in the usual axes system
                            - searchNeighborhoodSortMode = 1:
                                distance in the axes sytem supporting the
                                covariance model and accounting for anisotropy
                                given by the ranges
                            - searchNeighborhoodSortMode = 2:
                                minus the evaluation of the covariance model
                            Notes:
                            - if the covariance model has any variable parameter
                                (non-stationary), then
                                searchNeighborhoodSortMode = 2 is not allowed
                            - if the covariance model has any range or angle set
                                as a variable parameter, then
                                searchNeighborhoodSortMode must be set to 0
                            - greatest possible value as default

    :param seed:        (int or None) initial seed, if None an initial seed
                            between 1 and 999999 is generated with
                            numpy.random.randint

    :param outputReportFile:
                        (string or None) name of the report file,
                            if None: no report file [if given, a suffix
                            ".<process_index>" is added for the report file of
                            each process]

    :param treat_image_one_by_one:
                        (bool) keyword argument passed to the function
                            geone.img.gatherImages
                            - if False (default) images (result of each process)
                            are gathered at once, and then removed (faster)
                            - if True, images (result of each process) are
                            gathered one by one, i.e. successively gathered and
                            removed (slower, may save memory)

    :param nproc:
                (int) number of processes (can be modified in the function)
                    nproc = None: nproc is set to
                        min(nmax-1, nreal) (but at least 1),
                    where nmax is the total number of cpu(s) of the system
                    (retrieved by multiprocessing.cpu_count())

    :param nthreads_per_proc:
                (int) number of thread(s) per process (should be > 0 or None):
                    nthreads_per_proc = None: nthreads_per_proc is automatically
                    computed as the maximal integer (but at least 1) such that
                            nproc * nthreads_per_proc <= nmax-1
                    where nmax is the total number of cpu(s) of the system
                    (retrieved by multiprocessing.cpu_count())

    :param verbose:
                (int) indicates what information is displayed:
                    - 0: no display
                    - 1: only errors (and note(s))
                    - 2: version and warning(s) encountered

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img class) output image, with image.nv=nreal variables (each
                    variable is one realization)
                    (image is None if mpds_geosClassicOutput->outputImage is
                    NULL)
        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)
        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    # Set number of processes: nproc
    if nproc is None:
        nproc = max(min(multiprocessing.cpu_count()-1, nreal), 1)
    else:
        nproc_tmp = nproc
        nproc = max(min(int(nproc), nreal), 1)
        if verbose > 0 and nproc != nproc_tmp:
            print('NOTE: number of processes has been changed (now: nproc={})'.format(nproc))

    # Set number of threads per process: nth
    if nthreads_per_proc is None:
        nth = max(int(np.floor((multiprocessing.cpu_count()-1) / nproc)), 1)
    else:
        nth = max(int(nthreads_per_proc), 1)
        if verbose > 0 and nth != nthreads_per_proc:
            print('NOTE: number of threads per process has been changed (now: nthreads_per_proc={})'.format(nth))

    if verbose > 0 and nproc * nth > multiprocessing.cpu_count():
        print('NOTE: total number of cpu(s) used will exceed number of cpu(s) of the system...')

    # Set the distribution of the realizations over the processes
    # Condider the Euclidean division of nreal by nproc:
    #     nreal = q * nproc + r, with 0 <= r < nproc
    # Then, (q+1) realizations will be done on process 0, 1, ..., r-1, and q realization on process r, ..., nproc-1
    # Define the list real_index_proc of length (nproc+1) such that
    #   real_index_proc[i], ..., real_index_proc[i+1] - 1 : are the realization indices run on process i
    q, r = np.divmod(nreal, nproc)
    real_index_proc = [i*q + min(i, r) for i in range(nproc+1)]

    if verbose >= 2:
        print('Geos-Classic running on {} process(es)... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(nproc, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching geos-classic...

    # mpds_geosClassicInput.seed
    if seed is None:
        seed = np.random.randint(1,1000000)
    seed = int(seed)

    outputReportFile_p = None

    # Set pool of nproc workers
    pool = multiprocessing.Pool(nproc)
    out_pool = []
    for i in range(nproc):
        # Adapt input for i-th process
        nreal_p = real_index_proc[i+1] - real_index_proc[i]
        seed_p = seed + real_index_proc[i]
        if outputReportFile is not None:
            outputReportFile_p = outputReportFile + f'.{i}'
        if i==0:
            verbose_p = min(verbose, 1) # allow to print error for process i
        else:
            verbose_p = 0
        # Launch geos-classic (i-th process)
        out_pool.append(
            pool.apply_async(simulateIndicator2D,
                args=(category_values,
                cov_model_for_category,
                dimension, spacing, origin,
                method,
                nreal_p,                     # nreal (adjusted)
                probability,
                x, v,
                mask,
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                seed_p,                      # seed (adjusted)
                outputReportFile_p,          # outputReportFile (adjusted)
                nth,                         # nthreads
                verbose_p)                   # verbose (adjusted)
                )
            )

    # Properly end working process
    pool.close() # Prevents any more tasks from being submitted to the pool,
    pool.join()  # then, wait for the worker processes to exit.

    # Get result from each process
    geosclassic_output_proc = [p.get() for p in out_pool]

    if np.any([out is None for out in geosclassic_output_proc]):
        return None

    image = None
    nwarning, warnings = None, None

    # Gather results from every process
    # image
    image = np.hstack([out['image'] for out in geosclassic_output_proc])
    # ... remove None entries
    image = image[[x is not None for x in image]]
    # .. set to None if every entry is None
    if np.all([x is None for x in image]):
        image = None

    # nwarning
    nwarning = np.sum([out['nwarning'] for out in geosclassic_output_proc])
    # warnings
    warnings = list(np.unique(np.hstack([out['warnings'] for out in geosclassic_output_proc])))

    # Gather images and adjust variable names
    if image is not None:
        all_image = img.gatherImages(image, keep_varname=True, rem_var_from_source=True, treat_image_one_by_one=treat_image_one_by_one)
        ndigit = geosclassic.MPDS_GEOS_CLASSIC_NB_DIGIT_FOR_REALIZATION_NUMBER
        for j in range(all_image.nv):
            all_image.varname[j] = all_image.varname[j][:-ndigit] + f'{j:0{ndigit}d}'

    geosclassic_output = {'image':all_image, 'nwarning':nwarning, 'warnings':warnings}

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete (all process(es))')

    # Show (print) encountered warnings
    if verbose >= 2 and geosclassic_output and geosclassic_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(geosclassic_output['nwarning']))
        for i, warning_message in enumerate(geosclassic_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def simulateIndicator3D(
        category_values,
        cov_model_for_category,
        dimension, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
        method='simple_kriging',
        nreal=1,
        probability=None,
        x=None, v=None,
        mask=None,
        searchRadiusRelative=1.,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        seed=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2):
    """
    Generates 3D simulations (Sequential Indicator Simulation, SIS) based on
    simple or ordinary kriging.

    :param category_values:
                        (sequence of floats or ints) list of the category values;
                            let ncategory be the length of the list, then:
                            - if ncategory == 1:
                                - the unique category value given must not be
                                equal to 0
                                - it is used for a binary case with values
                                ("unique category value", 0), where 0 indicates
                                the absence of the considered medium
                                - conditioning data values should be
                                "unique category value" or 0
                            - if ncategory >= 2:
                                - it is used for a multi-category case with given
                                values (distinct)
                                - conditioning data values should be in the list
                                of given values

    :param cov_model_for_category:
                        (sequence of covariance model of length ncategory (see
                            category_values), or one covariance model, recycled)
                            a covariance model per category,
                            with entry for covariance model:
                                (CovModel3D class) covariance model in 3D, see
                                    definition of the class in module
                                    geone.covModel
                            or
                                (CovModel1D class) covariance model in 1D, see
                                    definition of the class in module
                                    geone.covModel, it is then transformed to an
                                    isotropic (omni-directional) covariance model
                                    in 3D

    :param dimension:   (sequence of 3 ints) (nx, ny, nz), number of cells
                            in x-, y-, z-axis direction
    :param spacing:     (sequence of 3 floats) (sx, sy, sz), spacing between
                            two adjacent cells in x-, y-, z-axis direction
    :param origin:      (sequence of 3 floats) (ox, oy, oy), origin of the 3D
                            simulation - used for localizing the conditioning
                            points
    :param method:      (string) indicates the method used:
                            - 'simple_kriging':
                                simulation based on simple kriging
                            - 'ordinary_kriging':
                                simulation based on ordinary kriging
    :param nreal:       (int) number of realizations

    :param probability: (None or sequence of floats of length ncategory or
                            ndarray of floats) probability for each category:
                                - None :
                                    proportion of each category in the hard data
                                    values (stationary), (uniform distribution if
                                    no hard data)
                                - sequence of floats of length ncategory:
                                    for stationary probabilities (set manually),
                                    if ncategory > 1, the sum of the probabilities
                                    must be equal to one
                                - ndarray: for non stationary probabilities,
                                    must contain ncategory * ngrid_cells entries
                                    where ngrid_cells is the number of grid cells
                                    (reshaped if needed); the first ngrid_cells
                                    entries are the probabities for the first
                                    category in the simulation grid, the next
                                    ngrid_cells entries those of the second
                                    category, and so on
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param x:           (2-dimensional array of dim n x 3, or
                            1-dimensional array of dim 3 or None) coordinate of
                            conditioning points for hard data
    :param v:           (1-dimensional array or float or None) value
                            at conditioning points for hard data
                            (same type as x)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not
                            simulated cell (nunber of entries should be equal to
                            the number of grid cells)

    :param searchRadiusRelative:
                        (sequence of ncategory floats (or float, recycled))
                            indicating how restricting the search ellipsoid
                            (should be positive) for each category: let r_i be
                            the ranges of the covariance model along its main
                            axes, if x is a node to be simulated, a node y is
                            taken into account iff it is within the ellipsoid
                            centered at x of half-axes searchRadiusRelative * r_i
                            Note:
                                - if a range is a variable parameter, its maximal
                                value over the simulation grid is considered
                                - if orientation of the covariance model is
                                non-stationary, a "circular search neighborhood"
                                is considered with the radius set to the maximum
                                of all ranges

    :param nneighborMax:
                        (sequence of ncategory ints (or int, recycled)) maximum
                            number of nodes retrieved from the search ellipsoid,
                            for each category, set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (sequence of ncategory ints (or int, recycled))
                            indicating how to sort the search neighboorhood nodes
                            (neighbors) for each category, they are sorted in
                            increasing order according to:
                            - searchNeighborhoodSortMode = 0:
                                distance in the usual axes system
                            - searchNeighborhoodSortMode = 1:
                                distance in the axes sytem supporting the
                                covariance model and accounting for anisotropy
                                given by the ranges
                            - searchNeighborhoodSortMode = 2:
                                minus the evaluation of the covariance model
                            Notes:
                            - if the covariance model has any variable parameter
                                (non-stationary), then
                                searchNeighborhoodSortMode = 2 is not allowed
                            - if the covariance model has any range or angle set
                                as a variable parameter, then
                                searchNeighborhoodSortMode must be set to 0
                            - greatest possible value as default

    :param seed:        (int or None) initial seed, if None an initial seed
                            between 1 and 999999 is generated with
                            numpy.random.randint

    :param outputReportFile:
                        (string or None) name of the report file,
                            if None: no report file

    :param nthreads:
                        (int) number of thread(s) to use for "GeosClassicSim"
                            program (C), (nthreads = -n <= 0: for maximal number
                            of threads except n, but at least 1)
    :param verbose:
                (int) verbose mode, integer >=0, higher implies more display
                    - 0: no display
                    - 1: only errors
                    - 2: errors and warnings (+ some info)
                    - 3 (or >2): all information

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img class) output image, with image.nv=nreal variables (each
                    variable is one realization)
                    (image is None if mpds_geosClassicOutput->outputImage is
                    NULL)
        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)
        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    # --- Set grid geometry and varname
    # Set grid geometry
    nx, ny, nz = dimension
    sx, sy, sz = spacing
    ox, oy, oz = origin

    nxy = nx * ny
    nxyz = nxy * nz

    # spatial dimension
    space_dim = 3

    # Set varname
    varname = 'V0'

    # --- Check and prepare parameters
    # category_values and ncategory (computed)
    try:
        category_values = np.asarray(category_values, dtype='float').reshape(-1)
    except:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR3D): 'category_values' is not valid")
        return None

    ncategory = len(category_values)
    if ncategory <= 0:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR3D): 'category_values' is empty")
        return None

    # cov_model_for_category
    cov_model_for_category = np.asarray(cov_model_for_category).reshape(-1)
    if not np.all([isinstance(c, gcm.CovModel3D) for c in cov_model_for_category]):
        # cov model will be converted:
        #    as applying modification in an array is persistent at exit,
        #    work on a copy to ensure no modification of the initial entry
        cm_for_cat = np.deepcopy(cov_model_for_category)
    else:
        cm_for_cat = cov_model_for_category

    cm_for_cat = np.asarray(cm_for_cat).reshape(-1)
    for i in range(len(cm_for_cat)):
        if isinstance(cm_for_cat[i], gcm.CovModel1D):
            cm_for_cat[i] = gcm.covModel1D_to_covModel3D(cm_for_cat[i]) # convert model 1D in 3D
    if len(cm_for_cat) == 1:
        cm_for_cat = np.repeat(cm_for_cat, ncategory)
    elif len(cm_for_cat) != ncategory:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR3D): 'cov_model_for_category' of invalid length")
        return None
    if not np.all([isinstance(c, gcm.CovModel3D) for c in cm_for_cat]):
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR3D): 'cov_model_for_category' should contains CovModel3D objects")
        return None

    for cov_model in cm_for_cat:
        for el in cov_model.elem:
            # weight
            w = el[1]['w']
            if np.size(w) != 1 and np.size(w) != nxyz:
                if verbose > 0:
                    print("ERROR (SIMULATE_INDICATOR3D): covariance model: weight ('w') not compatible with simulation grid")
                return None
            # ranges
            if 'r' in el[1].keys():
                for r in el[1]['r']:
                    if np.size(r) != 1 and np.size(r) != nxyz:
                        if verbose > 0:
                            print("ERROR (SIMULATE_INDICATOR3D): covariance model: range ('r') not compatible with simulation grid")
                        return None
            # additional parameter (s)
            if 's' in el[1].keys():
                s  = el[1]['s']
                if np.size(s) != 1 and np.size(s) != nxyz:
                    if verbose > 0:
                        print("ERROR (SIMULATE_INDICATOR3D): covariance model: parameter ('s') not compatible with simulation grid")
                    return None

        # alpha
        angle = cov_model.alpha
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            if verbose > 0:
                print("ERROR (SIMULATE_INDICATOR3D): covariance model: angle (alpha) not compatible with simulation grid")
            return None

        # beta
        angle = cov_model.beta
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            if verbose > 0:
                print("ERROR (SIMULATE_INDICATOR3D): covariance model: angle (beta) not compatible with simulation grid")
            return None

        # gamma
        angle = cov_model.gamma
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            if verbose > 0:
                print("ERROR (SIMULATE_INDICATOR3D): covariance model: angle (gamma) not compatible with simulation grid")
            return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     if verbose > 0:
    #         print("ERROR (SIMULATE_INDICATOR3D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 3
    elif method == 'ordinary_kriging':
        computationMode = 2
    else:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR3D): 'method' is not valid")
        return None

    # data points: x, v
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 3) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            if verbose > 0:
                print("(ERROR (SIMULATE_INDICATOR3D): length of 'v' is not valid")
            return None
        xc = x[:,0]
        yc = x[:,1]
        zc = x[:,2]
        dataPointSet.append(
            PointSet(npt=v.shape[0], nv=4, val=np.array((xc, yc, zc, v)), varname=['X', 'Y', 'Z', varname])
            )

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            if verbose > 0:
                print("ERROR (SIMULATE_INDICATOR3D): 'mask' is not valid")
            return None

    # Check parameters - searchRadiusRelative
    searchRadiusRelative = np.asarray(searchRadiusRelative, dtype='float').reshape(-1)
    if len(searchRadiusRelative) == 1:
        searchRadiusRelative = np.repeat(searchRadiusRelative, ncategory)
    elif len(searchRadiusRelative) != ncategory:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR3D): 'searchRadiusRelative' of invalid length")
        return None

    for srr in searchRadiusRelative:
        if srr < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
            if verbose > 0:
                print("ERROR (SIMULATE_INDICATOR3D): a 'searchRadiusRelative' is too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
            return None

    # Check parameters - nneighborMax
    nneighborMax = np.asarray(nneighborMax, dtype='intc').reshape(-1)
    if len(nneighborMax) == 1:
        nneighborMax = np.repeat(nneighborMax, ncategory)
    elif len(nneighborMax) != ncategory:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR3D): 'nneighborMax' of invalid length")
        return None

    for nn in nneighborMax:
        if nn != -1 and nn <= 0:
            if verbose > 0:
                print("ERROR (SIMULATE_INDICATOR3D): any 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
            return None

    # Check parameters - searchNeighborhoodSortMode
    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode).reshape(-1)
    if len(searchNeighborhoodSortMode) == 1:
        searchNeighborhoodSortMode = np.repeat(searchNeighborhoodSortMode, ncategory)
    elif len(searchNeighborhoodSortMode) != ncategory:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR3D): 'searchNeighborhoodSortMode' of invalid length")
        return None

    for i in range(ncategory):
        if searchNeighborhoodSortMode[i] is None:
            # set greatest possible value
            if cm_for_cat[i].is_stationary():
                searchNeighborhoodSortMode[i] = 2
            elif cm_for_cat[i].is_orientation_stationary() and cm_for_cat[i].is_range_stationary():
                searchNeighborhoodSortMode[i] = 1
            else:
                searchNeighborhoodSortMode[i] = 0
        else:
            if searchNeighborhoodSortMode[i] == 2:
                if not cm_for_cat[i].is_stationary():
                    if verbose > 0:
                        print("ERROR (SIMULATE_INDICATOR3D): 'searchNeighborhoodSortMode set to 2' not allowed with non-stationary covariance model")
                    return None
            elif searchNeighborhoodSortMode[i] == 1:
                if not cm_for_cat[i].is_orientation_stationary() or not cm_for_cat[i].is_range_stationary():
                    if verbose > 0:
                        print("ERROR (SIMULATE_INDICATOR3D): 'searchNeighborhoodSortMode set to 1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                    return None

    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode, dtype='intc')

    # Check parameters - probability
    if probability is not None:
        # if method == 'ordinary_kriging':
        #     if verbose > 0:
        #         print("ERROR (SIMULATE_INDICATOR3D): specifying 'probability' not allowed with ordinary kriging")
        #     return None
        probability = np.asarray(probability, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if probability.size not in (ncategory, ncategory*nxyz):
            if verbose > 0:
                print("ERROR (SIMULATE_INDICATOR3D): size of 'probability' is not valid")
            return None

    # Check parameters - nreal
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose >= 2:
            print('SIMULATE_INDICATOR3D: nreal <= 0: nothing to do!')
        return None

    # --- Fill mpds_geosClassicInput structure (C)
    mpds_geosClassicIndicatorInput, flag = fill_mpds_geosClassicIndicatorInput(
        space_dim,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        varname,
        ncategory,
        category_values,
        outputReportFile,
        computationMode,
        cm_for_cat,
        None,
        dataPointSet,
        mask,
        probability,
        searchRadiusRelative,
        nneighborMax,
        searchNeighborhoodSortMode,
        seed,
        nreal)

    if not flag:
        if verbose > 0:
            print("ERROR (SIMULATE_INDICATOR3D): can not fill input structure!")
        return None

    # --- Prepare mpds_geosClassicIOutput structure (C)
    # Allocate mpds_geosClassicOutput
    mpds_geosClassicOutput = geosclassic.malloc_MPDS_GEOSCLASSICOUTPUT()

    # Init mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicInitGeosClassicOutput(mpds_geosClassicOutput)

    # --- Set progress monitor
    mpds_progressMonitor = geosclassic.malloc_MPDS_PROGRESSMONITOR()
    geosclassic.MPDSInitProgressMonitor(mpds_progressMonitor)

    # Set function to update progress monitor:
    # according to geosclassic.MPDS_SHOW_PROGRESS_MONITOR set to 4 for compilation of py module
    # the function
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor1_ptr: warning only
    if verbose < 3:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

    # Free memory on C side: mpds_geosClassicIndicatorInput
    geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
    #geosclassic.MPDSFree(mpds_geosClassicIndicatorInput)
    geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)

    if err:
        if verbose > 0:
            err_message = geosclassic.mpds_get_error_message(-err)
            err_message = err_message.replace('\n', '')
            print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 2 and geosclassic_output and geosclassic_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(geosclassic_output['nwarning']))
        for i, warning_message in enumerate(geosclassic_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def simulateIndicator3D_mp(
        category_values,
        cov_model_for_category,
        dimension, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
        method='simple_kriging',
        nreal=1,
        probability=None,
        x=None, v=None,
        mask=None,
        searchRadiusRelative=1.,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        seed=None,
        outputReportFile=None,
        treat_image_one_by_one=False,
        nproc=None, nthreads_per_proc=None,
        verbose=2):
    """
    Generates 3D simulations (Sequential Indicator Simulation, SIS) based on
    simple or ordinary kriging.

    Launches multiple processes (based on multiprocessing package):
        - nproc parallel processes using each one nthreads_per_proc threads will
            be launched [parallel calls of the function simulateIndicator3D],
        - the set of realizations (specified by nreal) is
            distributed in a balanced way over the processes,
        - in terms of resources, this implies the use of
            nproc * nthreads_per_proc cpu(s).

    :param category_values:
                        (sequence of floats or ints) list of the category values;
                            let ncategory be the length of the list, then:
                            - if ncategory == 1:
                                - the unique category value given must not be
                                equal to 0
                                - it is used for a binary case with values
                                ("unique category value", 0), where 0 indicates
                                the absence of the considered medium
                                - conditioning data values should be
                                "unique category value" or 0
                            - if ncategory >= 2:
                                - it is used for a multi-category case with given
                                values (distinct)
                                - conditioning data values should be in the list
                                of given values

    :param cov_model_for_category:
                        (sequence of covariance model of length ncategory (see
                            category_values), or one covariance model, recycled)
                            a covariance model per category,
                            with entry for covariance model:
                                (CovModel3D class) covariance model in 3D, see
                                    definition of the class in module
                                    geone.covModel
                            or
                                (CovModel1D class) covariance model in 1D, see
                                    definition of the class in module
                                    geone.covModel, it is then transformed to an
                                    isotropic (omni-directional) covariance model
                                    in 3D

    :param dimension:   (sequence of 3 ints) (nx, ny, nz), number of cells
                            in x-, y-, z-axis direction
    :param spacing:     (sequence of 3 floats) (sx, sy, sz), spacing between
                            two adjacent cells in x-, y-, z-axis direction
    :param origin:      (sequence of 3 floats) (ox, oy, oy), origin of the 3D
                            simulation - used for localizing the conditioning
                            points
    :param method:      (string) indicates the method used:
                            - 'simple_kriging':
                                simulation based on simple kriging
                            - 'ordinary_kriging':
                                simulation based on ordinary kriging
    :param nreal:       (int) number of realizations

    :param probability: (None or sequence of floats of length ncategory or
                            ndarray of floats) probability for each category:
                                - None :
                                    proportion of each category in the hard data
                                    values (stationary), (uniform distribution if
                                    no hard data)
                                - sequence of floats of length ncategory:
                                    for stationary probabilities (set manually),
                                    if ncategory > 1, the sum of the probabilities
                                    must be equal to one
                                - ndarray: for non stationary probabilities,
                                    must contain ncategory * ngrid_cells entries
                                    where ngrid_cells is the number of grid cells
                                    (reshaped if needed); the first ngrid_cells
                                    entries are the probabities for the first
                                    category in the simulation grid, the next
                                    ngrid_cells entries those of the second
                                    category, and so on
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param x:           (2-dimensional array of dim n x 3, or
                            1-dimensional array of dim 3 or None) coordinate of
                            conditioning points for hard data
    :param v:           (1-dimensional array or float or None) value
                            at conditioning points for hard data
                            (same type as x)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not
                            simulated cell (nunber of entries should be equal to
                            the number of grid cells)

    :param searchRadiusRelative:
                        (sequence of ncategory floats (or float, recycled))
                            indicating how restricting the search ellipsoid
                            (should be positive) for each category: let r_i be
                            the ranges of the covariance model along its main
                            axes, if x is a node to be simulated, a node y is
                            taken into account iff it is within the ellipsoid
                            centered at x of half-axes searchRadiusRelative * r_i
                            Note:
                                - if a range is a variable parameter, its maximal
                                value over the simulation grid is considered
                                - if orientation of the covariance model is
                                non-stationary, a "circular search neighborhood"
                                is considered with the radius set to the maximum
                                of all ranges

    :param nneighborMax:
                        (sequence of ncategory ints (or int, recycled)) maximum
                            number of nodes retrieved from the search ellipsoid,
                            for each category, set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (sequence of ncategory ints (or int, recycled))
                            indicating how to sort the search neighboorhood nodes
                            (neighbors) for each category, they are sorted in
                            increasing order according to:
                            - searchNeighborhoodSortMode = 0:
                                distance in the usual axes system
                            - searchNeighborhoodSortMode = 1:
                                distance in the axes sytem supporting the
                                covariance model and accounting for anisotropy
                                given by the ranges
                            - searchNeighborhoodSortMode = 2:
                                minus the evaluation of the covariance model
                            Notes:
                            - if the covariance model has any variable parameter
                                (non-stationary), then
                                searchNeighborhoodSortMode = 2 is not allowed
                            - if the covariance model has any range or angle set
                                as a variable parameter, then
                                searchNeighborhoodSortMode must be set to 0
                            - greatest possible value as default

    :param seed:        (int or None) initial seed, if None an initial seed
                            between 1 and 999999 is generated with
                            numpy.random.randint

    :param outputReportFile:
                        (string or None) name of the report file,
                            if None: no report file [if given, a suffix
                            ".<process_index>" is added for the report file of
                            each process]

    :param treat_image_one_by_one:
                        (bool) keyword argument passed to the function
                            geone.img.gatherImages
                            - if False (default) images (result of each process)
                            are gathered at once, and then removed (faster)
                            - if True, images (result of each process) are
                            gathered one by one, i.e. successively gathered and
                            removed (slower, may save memory)

    :param nproc:
                (int) number of processes (can be modified in the function)
                    nproc = None: nproc is set to
                        min(nmax-1, nreal) (but at least 1),
                    where nmax is the total number of cpu(s) of the system
                    (retrieved by multiprocessing.cpu_count())

    :param nthreads_per_proc:
                (int) number of thread(s) per process (should be > 0 or None):
                    nthreads_per_proc = None: nthreads_per_proc is automatically
                    computed as the maximal integer (but at least 1) such that
                            nproc * nthreads_per_proc <= nmax-1
                    where nmax is the total number of cpu(s) of the system
                    (retrieved by multiprocessing.cpu_count())

    :param verbose:
                (int) indicates what information is displayed:
                    - 0: no display
                    - 1: only errors (and note(s))
                    - 2: version and warning(s) encountered

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img class) output image, with image.nv=nreal variables (each
                    variable is one realization)
                    (image is None if mpds_geosClassicOutput->outputImage is
                    NULL)
        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)
        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    # Set number of processes: nproc
    if nproc is None:
        nproc = max(min(multiprocessing.cpu_count()-1, nreal), 1)
    else:
        nproc_tmp = nproc
        nproc = max(min(int(nproc), nreal), 1)
        if verbose > 0 and nproc != nproc_tmp:
            print('NOTE: number of processes has been changed (now: nproc={})'.format(nproc))

    # Set number of threads per process: nth
    if nthreads_per_proc is None:
        nth = max(int(np.floor((multiprocessing.cpu_count()-1) / nproc)), 1)
    else:
        nth = max(int(nthreads_per_proc), 1)
        if verbose > 0 and nth != nthreads_per_proc:
            print('NOTE: number of threads per process has been changed (now: nthreads_per_proc={})'.format(nth))

    if verbose > 0 and nproc * nth > multiprocessing.cpu_count():
        print('NOTE: total number of cpu(s) used will exceed number of cpu(s) of the system...')

    # Set the distribution of the realizations over the processes
    # Condider the Euclidean division of nreal by nproc:
    #     nreal = q * nproc + r, with 0 <= r < nproc
    # Then, (q+1) realizations will be done on process 0, 1, ..., r-1, and q realization on process r, ..., nproc-1
    # Define the list real_index_proc of length (nproc+1) such that
    #   real_index_proc[i], ..., real_index_proc[i+1] - 1 : are the realization indices run on process i
    q, r = np.divmod(nreal, nproc)
    real_index_proc = [i*q + min(i, r) for i in range(nproc+1)]

    if verbose >= 2:
        print('Geos-Classic running on {} process(es)... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(nproc, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching geos-classic...

    # mpds_geosClassicInput.seed
    if seed is None:
        seed = np.random.randint(1,1000000)
    seed = int(seed)

    outputReportFile_p = None

    # Set pool of nproc workers
    pool = multiprocessing.Pool(nproc)
    out_pool = []
    for i in range(nproc):
        # Adapt input for i-th process
        nreal_p = real_index_proc[i+1] - real_index_proc[i]
        seed_p = seed + real_index_proc[i]
        if outputReportFile is not None:
            outputReportFile_p = outputReportFile + f'.{i}'
        if i==0:
            verbose_p = min(verbose, 1) # allow to print error for process i
        else:
            verbose_p = 0
        # Launch geos-classic (i-th process)
        out_pool.append(
            pool.apply_async(simulateIndicator3D,
                args=(category_values,
                cov_model_for_category,
                dimension, spacing, origin,
                method,
                nreal_p,                     # nreal (adjusted)
                probability,
                x, v,
                mask,
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                seed_p,                      # seed (adjusted)
                outputReportFile_p,          # outputReportFile (adjusted)
                nth,                         # nthreads
                verbose_p)                   # verbose (adjusted)
                )
            )

    # Properly end working process
    pool.close() # Prevents any more tasks from being submitted to the pool,
    pool.join()  # then, wait for the worker processes to exit.

    # Get result from each process
    geosclassic_output_proc = [p.get() for p in out_pool]

    if np.any([out is None for out in geosclassic_output_proc]):
        return None

    image = None
    nwarning, warnings = None, None

    # Gather results from every process
    # image
    image = np.hstack([out['image'] for out in geosclassic_output_proc])
    # ... remove None entries
    image = image[[x is not None for x in image]]
    # .. set to None if every entry is None
    if np.all([x is None for x in image]):
        image = None

    # nwarning
    nwarning = np.sum([out['nwarning'] for out in geosclassic_output_proc])
    # warnings
    warnings = list(np.unique(np.hstack([out['warnings'] for out in geosclassic_output_proc])))

    # Gather images and adjust variable names
    if image is not None:
        all_image = img.gatherImages(image, keep_varname=True, rem_var_from_source=True, treat_image_one_by_one=treat_image_one_by_one)
        ndigit = geosclassic.MPDS_GEOS_CLASSIC_NB_DIGIT_FOR_REALIZATION_NUMBER
        for j in range(all_image.nv):
            all_image.varname[j] = all_image.varname[j][:-ndigit] + f'{j:0{ndigit}d}'

    geosclassic_output = {'image':all_image, 'nwarning':nwarning, 'warnings':warnings}

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete (all process(es))')

    # Show (print) encountered warnings
    if verbose >= 2 and geosclassic_output and geosclassic_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(geosclassic_output['nwarning']))
        for i, warning_message in enumerate(geosclassic_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def estimateIndicator1D(
        category_values,
        cov_model_for_category,
        dimension, spacing=1.0, origin=0.0,
        method='simple_kriging',
        probability=None,
        x=None, v=None,
        mask=None,
        use_unique_neighborhood=False,
        searchRadiusRelative=1.,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        seed=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2):
    """
    Computes estimate probabilities of categories (indicators) for 1D grid
    based on simple or ordinary kriging.

    :param category_values:
                        (sequence of floats or ints) list of the category values;
                            let ncategory be the length of the list, then:
                            - if ncategory == 1:
                                - the unique category value given must not be
                                equal to 0
                                - it is used for a binary case with values
                                ("unique category value", 0), where 0 indicates
                                the absence of the considered medium
                                - conditioning data values should be
                                "unique category value" or 0
                            - if ncategory >= 2:
                                - it is used for a multi-category case with given
                                values (distinct)
                                - conditioning data values should be in the list
                                of given values

    :param cov_model_for_category:
                        (sequence of covariance model of length ncategory (see
                            category_values), or one covariance model, recycled)
                            a covariance model per category,
                            with entry for covariance model:
                                (CovModel1D class) covariance model in 1D, see
                                    definition of the class in
                                    module geone.covModel

    :param dimension:   (int) nx, number of cells
    :param spacing:     (float) sx, spacing between two adjacent cells
    :param origin:      (float) ox, origin of the 1D simulation
                            - used for localizing the conditioning points
    :param method:      (string) indicates the method used:
                            - 'simple_kriging':
                                simulation based on simple kriging
                            - 'ordinary_kriging':
                                simulation based on ordinary kriging

    :param probability: (None or sequence of floats of length ncategory or
                            ndarray of floats) probability for each category:
                                - None :
                                    proportion of each category in the hard data
                                    values (stationary), (uniform distribution if
                                    no hard data)
                                - sequence of floats of length ncategory:
                                    for stationary probabilities (set manually),
                                    if ncategory > 1, the sum of the probabilities
                                    must be equal to one
                                - ndarray: for non stationary probabilities,
                                    must contain ncategory * ngrid_cells entries
                                    where ngrid_cells is the number of grid cells
                                    (reshaped if needed); the first ngrid_cells
                                    entries are the probabities for the first
                                    category in the simulation grid, the next
                                    ngrid_cells entries those of the second
                                    category, and so on
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param x:           (1-dimensional array or float or None) coordinate of
                            conditioning points for hard data
    :param v:           (1-dimensional array or float or None) value
                            at conditioning points for hard data
                            (same type as x)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not
                            simulated cell (nunber of entries should be equal to
                            the number of grid cells)

    :param use_unique_neighborhood:
                        (sequence of ncategory bools (or bool, recycled))
                            indicating if a unique neighborhood is used, for each
                            category:
                            - True: all data points are taken into account for
                                computing estimate probabilities;
                                in this case: parameters
                                    searchRadiusRelative,
                                    nneighborMax,
                                    searchNeighborhoodSortMode,
                                are unused
                            - False: only data points within a search ellipsoid
                                are taken into account for computing estimate
                                probabilities (see parameters
                                searchRadiusRelative, nneighborMax,
                                searchNeighborhoodSortMode)

    :param searchRadiusRelative:
                        (sequence of ncategory floats (or float, recycled))
                            indicating how restricting the search ellipsoid
                            (should be positive) for each category: let r_i be
                            the ranges of the covariance model along its main
                            axes, if x is a node to be simulated, a node y is
                            taken into account iff it is within the ellipsoid
                            centered at x of half-axes searchRadiusRelative * r_i
                            Note:
                                - if a range is a variable parameter, its maximal
                                value over the simulation grid is considered

    :param nneighborMax:
                        (sequence of ncategory ints (or int, recycled)) maximum
                            number of nodes retrieved from the search ellipsoid,
                            for each category, set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (sequence of ncategory ints (or int, recycled))
                            indicating how to sort the search neighboorhood nodes
                            (neighbors) for each category, they are sorted in
                            increasing order according to:
                            - searchNeighborhoodSortMode = 0:
                                distance in the usual axes system
                            - searchNeighborhoodSortMode = 1:
                                distance in the axes sytem supporting the
                                covariance model and accounting for anisotropy
                                given by the ranges
                            - searchNeighborhoodSortMode = 2:
                                minus the evaluation of the covariance model
                            Notes:
                            - if the covariance model has any variable parameter
                                (non-stationary), then
                                searchNeighborhoodSortMode = 2 is not allowed
                            - if the covariance model has any range or angle set
                                as a variable parameter, then
                                searchNeighborhoodSortMode must be set to 0
                            - greatest possible value as default

    :param outputReportFile:
                        (string or None) name of the report file,
                            if None: no report file

    :param nthreads:
                        (int) number of thread(s) to use for "GeosClassicSim"
                            program (C), (nthreads = -n <= 0: for maximal number
                            of threads except n, but at least 1)
    :param verbose:
                (int) verbose mode, integer >=0, higher implies more display
                    - 0: no display
                    - 1: only errors
                    - 2: errors and warnings (+ some info)
                    - 3 (or >2): all information

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img class) output image, with image.nv=ncategory variables
                    (estimate probabilities (for each category))
                    (image is None if mpds_geosClassicOutput->outputImage is
                    NULL)
        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)
        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    # --- Set grid geometry and varname
    # Set grid geometry
    nx, ny, nz = dimension, 1, 1
    sx, sy, sz = spacing, 1.0, 1.0
    ox, oy, oz = origin, 0.0, 0.0

    nxy = nx * ny
    nxyz = nxy * nz

    # spatial dimension
    space_dim = 1

    # Set varname
    varname = 'V0'

    # --- Check and prepare parameters
    # category_values and ncategory (computed)
    try:
        category_values = np.asarray(category_values, dtype='float').reshape(-1)
    except:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR1D): 'category_values' is not valid")
        return None

    ncategory = len(category_values)
    if ncategory <= 0:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR1D): 'category_values' is empty")
        return None

    # cov_model_for_category
    cm_for_cat = cov_model_for_category # no need to work on a copy in 1D

    cm_for_cat = np.asarray(cm_for_cat).reshape(-1)
    if len(cm_for_cat) == 1:
        cm_for_cat = np.repeat(cm_for_cat, ncategory)
    elif len(cm_for_cat) != ncategory:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR1D): 'cov_model_for_category' of invalid length")
        return None
    if not np.all([isinstance(c, gcm.CovModel1D) for c in cm_for_cat]):
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR1D): 'cov_model_for_category' should contains CovModel1D objects")
        return None

    for cov_model in cm_for_cat:
        for el in cov_model.elem:
            # weight
            w = el[1]['w']
            if np.size(w) != 1 and np.size(w) != nxyz:
                if verbose > 0:
                    print("ERROR (ESTIMATE_INDICATOR1D): covariance model: weight ('w') not compatible with simulation grid")
                return None
            # ranges
            if 'r' in el[1].keys():
                r  = el[1]['r']
                if np.size(r) != 1 and np.size(r) != nxyz:
                    if verbose > 0:
                        print("ERROR (ESTIMATE_INDICATOR1D): covariance model: range ('r') not compatible with simulation grid")
                    return None
            # additional parameter (s)
            if 's' in el[1].keys():
                s  = el[1]['s']
                if np.size(s) != 1 and np.size(s) != nxyz:
                    if verbose > 0:
                        print("ERROR (ESTIMATE_INDICATOR1D): covariance model: parameter ('s') not compatible with simulation grid")
                    return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     if verbose > 0:
    #         print("ERROR (ESTIMATE_INDICATOR1D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 1
    elif method == 'ordinary_kriging':
        computationMode = 0
    else:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR1D): 'method' is not valid")
        return None

    # data points: x, v
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            if verbose > 0:
                print("(ERROR (ESTIMATE_INDICATOR1D): length of 'v' is not valid")
            return None
        xc = x
        yc = np.ones_like(xc) * oy + 0.5 * sy
        zc = np.ones_like(xc) * oz + 0.5 * sz
        dataPointSet.append(
            PointSet(npt=v.shape[0], nv=4, val=np.array((xc, yc, zc, v)), varname=['X', 'Y', 'Z', varname])
            )

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            if verbose > 0:
                print("ERROR (ESTIMATE_INDICATOR1D): 'mask' is not valid")
            return None

    # Check parameters - use_unique_neighborhood (length)
    use_unique_neighborhood = np.asarray(use_unique_neighborhood, dtype='bool').reshape(-1)
    if len(use_unique_neighborhood) == 1:
        use_unique_neighborhood = np.repeat(use_unique_neighborhood, ncategory)
    elif len(use_unique_neighborhood) != ncategory:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR1D): 'use_unique_neighborhood' of invalid length")
        return None

    # Check parameters - searchRadiusRelative (length)
    searchRadiusRelative = np.asarray(searchRadiusRelative, dtype='float').reshape(-1)
    if len(searchRadiusRelative) == 1:
        searchRadiusRelative = np.repeat(searchRadiusRelative, ncategory)
    elif len(searchRadiusRelative) != ncategory:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR1D): 'searchRadiusRelative' of invalid length")
        return None

    # Check parameters - nneighborMax (length)
    nneighborMax = np.asarray(nneighborMax, dtype='intc').reshape(-1)
    if len(nneighborMax) == 1:
        nneighborMax = np.repeat(nneighborMax, ncategory)
    elif len(nneighborMax) != ncategory:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR1D): 'nneighborMax' of invalid length")
        return None

    # Check parameters - searchNeighborhoodSortMode (length)
    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode).reshape(-1)
    if len(searchNeighborhoodSortMode) == 1:
        searchNeighborhoodSortMode = np.repeat(searchNeighborhoodSortMode, ncategory)
    elif len(searchNeighborhoodSortMode) != ncategory:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR1D): 'searchNeighborhoodSortMode' of invalid length")
        return None

    # If unique neighborhood is used, set searchRadiusRelative to -1
    #    (and initialize nneighborMax, searchNeighborhoodSortMode (unused))
    # else: check the parameters
    for i in range(ncategory):
        if use_unique_neighborhood[i]:
            searchRadiusRelative[i] = -1.0
            nneighborMax[i] = 1
            searchNeighborhoodSortMode[i] = 0

        else:
            if searchRadiusRelative[i] < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
                if verbose > 0:
                    print("ERROR (ESTIMATE_INDICATOR1D): a 'searchRadiusRelative' is too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
                return None

            if nneighborMax[i] != -1 and nneighborMax[i] <= 0:
                if verbose > 0:
                    print("ERROR (ESTIMATE_INDICATOR1D): any 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
                return None

            if searchNeighborhoodSortMode[i] is None:
                # set greatest possible value
                if cm_for_cat[i].is_stationary():
                    searchNeighborhoodSortMode[i] = 2
                elif cm_for_cat[i].is_orientation_stationary() and cm_for_cat[i].is_range_stationary():
                    searchNeighborhoodSortMode[i] = 1
                else:
                    searchNeighborhoodSortMode[i] = 0
            else:
                if searchNeighborhoodSortMode[i] == 2:
                    if not cm_for_cat[i].is_stationary():
                        if verbose > 0:
                            print("ERROR (ESTIMATE_INDICATOR1D): 'searchNeighborhoodSortMode set to 2' not allowed with non-stationary covariance model")
                        return None
                elif searchNeighborhoodSortMode[i] == 1:
                    if not cm_for_cat[i].is_orientation_stationary() or not cm_for_cat[i].is_range_stationary():
                        if verbose > 0:
                            print("ERROR (ESTIMATE_INDICATOR1D): 'searchNeighborhoodSortMode set to 1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                        return None

    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode, dtype='intc')

    # Check parameters - probability
    if probability is not None:
        # if method == 'ordinary_kriging':
        #     if verbose > 0:
        #         print("ERROR (ESTIMATE_INDICATOR1D): specifying 'probability' not allowed with ordinary kriging")
        #     return None
        probability = np.asarray(probability, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if probability.size not in (ncategory, ncategory*nxyz):
            if verbose > 0:
                print("ERROR (ESTIMATE_INDICATOR1D): size of 'probability' is not valid")
            return None

    # --- Fill mpds_geosClassicInput structure (C)
    mpds_geosClassicIndicatorInput, flag = fill_mpds_geosClassicIndicatorInput(
        space_dim,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        varname,
        ncategory,
        category_values,
        outputReportFile,
        computationMode,
        cm_for_cat,
        None,
        dataPointSet,
        mask,
        probability,
        searchRadiusRelative,
        nneighborMax,
        searchNeighborhoodSortMode,
        0,
        0)

    if not flag:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR1D): can not fill input structure!")
        return None

    # --- Prepare mpds_geosClassicIOutput structure (C)
    # Allocate mpds_geosClassicOutput
    mpds_geosClassicOutput = geosclassic.malloc_MPDS_GEOSCLASSICOUTPUT()

    # Init mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicInitGeosClassicOutput(mpds_geosClassicOutput)

    # --- Set progress monitor
    mpds_progressMonitor = geosclassic.malloc_MPDS_PROGRESSMONITOR()
    geosclassic.MPDSInitProgressMonitor(mpds_progressMonitor)

    # Set function to update progress monitor:
    # according to geosclassic.MPDS_SHOW_PROGRESS_MONITOR set to 4 for compilation of py module
    # the function
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor1_ptr: warning only
    if verbose < 3:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

    # Free memory on C side: mpds_geosClassicIndicatorInput
    geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
    #geosclassic.MPDSFree(mpds_geosClassicIndicatorInput)
    geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)

    if err:
        if verbose > 0:
            err_message = geosclassic.mpds_get_error_message(-err)
            err_message = err_message.replace('\n', '')
            print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 2 and geosclassic_output and geosclassic_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(geosclassic_output['nwarning']))
        for i, warning_message in enumerate(geosclassic_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def estimateIndicator2D(
        category_values,
        cov_model_for_category,
        dimension, spacing=(1.0, 1.0), origin=(0.0, 0.0),
        method='simple_kriging',
        probability=None,
        x=None, v=None,
        mask=None,
        use_unique_neighborhood=False,
        searchRadiusRelative=1.,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        seed=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2):
    """
    Computes estimate probabilities of categories (indicators) for 2D grid
    based on simple or ordinary kriging.

    :param category_values:
                        (sequence of floats or ints) list of the category values;
                            let ncategory be the length of the list, then:
                            - if ncategory == 1:
                                - the unique category value given must not be
                                equal to 0
                                - it is used for a binary case with values
                                ("unique category value", 0), where 0 indicates
                                the absence of the considered medium
                                - conditioning data values should be
                                "unique category value" or 0
                            - if ncategory >= 2:
                                - it is used for a multi-category case with given
                                values (distinct)
                                - conditioning data values should be in the list
                                of given values

    :param cov_model_for_category:
                        (sequence of covariance model of length ncategory (see
                            category_values), or one covariance model, recycled)
                            a covariance model per category,
                            with entry for covariance model:
                                (CovModel2D class) covariance model in 2D, see
                                    definition of the class in module
                                    geone.covModel
                            or
                                (CovModel1D class) covariance model in 1D, see
                                    definition of the class in module
                                    geone.covModel, it is then transformed to an
                                    isotropic (omni-directional) covariance model
                                    in 2D

    :param dimension:   (sequence of 2 ints) (nx, ny), number of cells
                            in x-, y-axis direction
    :param spacing:     (sequence of 2 floats) (sx, sy), spacing between
                            two adjacent cells in x-, y-axis direction
    :param origin:      (sequence of 2 floats) (ox, oy), origin of the 2D
                            simulation - used for localizing the conditioning
                            points
    :param method:      (string) indicates the method used:
                            - 'simple_kriging':
                                simulation based on simple kriging
                            - 'ordinary_kriging':
                                simulation based on ordinary kriging

    :param probability: (None or sequence of floats of length ncategory or
                            ndarray of floats) probability for each category:
                                - None :
                                    proportion of each category in the hard data
                                    values (stationary), (uniform distribution if
                                    no hard data)
                                - sequence of floats of length ncategory:
                                    for stationary probabilities (set manually),
                                    if ncategory > 1, the sum of the probabilities
                                    must be equal to one
                                - ndarray: for non stationary probabilities,
                                    must contain ncategory * ngrid_cells entries
                                    where ngrid_cells is the number of grid cells
                                    (reshaped if needed); the first ngrid_cells
                                    entries are the probabities for the first
                                    category in the simulation grid, the next
                                    ngrid_cells entries those of the second
                                    category, and so on
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param x:           (2-dimensional array of dim n x 2, or
                            1-dimensional array of dim 2 or None) coordinate of
                            conditioning points for hard data
    :param v:           (1-dimensional array or float or None) value
                            at conditioning points for hard data
                            (same type as x)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not
                            simulated cell (nunber of entries should be equal to
                            the number of grid cells)

    :param use_unique_neighborhood:
                        (sequence of ncategory bools (or bool, recycled))
                            indicating if a unique neighborhood is used, for each
                            category:
                            - True: all data points are taken into account for
                                computing estimate probabilities;
                                in this case: parameters
                                    searchRadiusRelative,
                                    nneighborMax,
                                    searchNeighborhoodSortMode,
                                are unused
                            - False: only data points within a search ellipsoid
                                are taken into account for computing estimate
                                probabilities (see parameters
                                searchRadiusRelative, nneighborMax,
                                searchNeighborhoodSortMode)

    :param searchRadiusRelative:
                        (sequence of ncategory floats (or float, recycled))
                            indicating how restricting the search ellipsoid
                            (should be positive) for each category: let r_i be
                            the ranges of the covariance model along its main
                            axes, if x is a node to be simulated, a node y is
                            taken into account iff it is within the ellipsoid
                            centered at x of half-axes searchRadiusRelative * r_i
                            Note:
                                - if a range is a variable parameter, its maximal
                                value over the simulation grid is considered
                                - if orientation of the covariance model is
                                non-stationary, a "circular search neighborhood"
                                is considered with the radius set to the maximum
                                of all ranges

    :param nneighborMax:
                        (sequence of ncategory ints (or int, recycled)) maximum
                            number of nodes retrieved from the search ellipsoid,
                            for each category, set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (sequence of ncategory ints (or int, recycled))
                            indicating how to sort the search neighboorhood nodes
                            (neighbors) for each category, they are sorted in
                            increasing order according to:
                            - searchNeighborhoodSortMode = 0:
                                distance in the usual axes system
                            - searchNeighborhoodSortMode = 1:
                                distance in the axes sytem supporting the
                                covariance model and accounting for anisotropy
                                given by the ranges
                            - searchNeighborhoodSortMode = 2:
                                minus the evaluation of the covariance model
                            Notes:
                            - if the covariance model has any variable parameter
                                (non-stationary), then
                                searchNeighborhoodSortMode = 2 is not allowed
                            - if the covariance model has any range or angle set
                                as a variable parameter, then
                                searchNeighborhoodSortMode must be set to 0
                            - greatest possible value as default

    :param outputReportFile:
                        (string or None) name of the report file,
                            if None: no report file

    :param nthreads:
                        (int) number of thread(s) to use for "GeosClassicSim"
                            program (C), (nthreads = -n <= 0: for maximal number
                            of threads except n, but at least 1)
    :param verbose:
                (int) verbose mode, integer >=0, higher implies more display
                    - 0: no display
                    - 1: only errors
                    - 2: errors and warnings (+ some info)
                    - 3 (or >2): all information

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img class) output image, with image.nv=ncategory variables
                    (estimate probabilities (for each category))
                    (image is None if mpds_geosClassicOutput->outputImage is
                    NULL)
        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)
        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    # --- Set grid geometry and varname
    # Set grid geometry
    nx, ny, nz = *dimension, 1
    sx, sy, sz = *spacing, 1.0
    ox, oy, oz = *origin, 0.0

    nxy = nx * ny
    nxyz = nxy * nz

    # spatial dimension
    space_dim = 2

    # Set varname
    varname = 'V0'

    # --- Check and prepare parameters
    # category_values and ncategory (computed)
    try:
        category_values = np.asarray(category_values, dtype='float').reshape(-1)
    except:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR2D): 'category_values' is not valid")
        return None

    ncategory = len(category_values)
    if ncategory <= 0:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR2D): 'category_values' is empty")
        return None

    # cov_model_for_category
    cov_model_for_category = np.asarray(cov_model_for_category).reshape(-1)
    if not np.all([isinstance(c, gcm.CovModel2D) for c in cov_model_for_category]):
        # cov model will be converted:
        #    as applying modification in an array is persistent at exit,
        #    work on a copy to ensure no modification of the initial entry
        cm_for_cat = np.deepcopy(cov_model_for_category)
    else:
        cm_for_cat = cov_model_for_category

    cm_for_cat = np.asarray(cm_for_cat).reshape(-1)
    if len(cm_for_cat) == 1:
        cm_for_cat = np.repeat(cm_for_cat, ncategory)
    elif len(cm_for_cat) != ncategory:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR2D): 'cov_model_for_category' of invalid length")
        return None
    if not np.all([isinstance(c, gcm.CovModel2D) for c in cm_for_cat]):
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR2D): 'cov_model_for_category' should contains CovModel2D objects")
        return None

    for cov_model in cm_for_cat:
        for el in cov_model.elem:
            # weight
            w = el[1]['w']
            if np.size(w) != 1 and np.size(w) != nxyz:
                if verbose > 0:
                    print("ERROR (ESTIMATE_INDICATOR2D): covariance model: weight ('w') not compatible with simulation grid")
                return None
            # ranges
            if 'r' in el[1].keys():
                for r in el[1]['r']:
                    if np.size(r) != 1 and np.size(r) != nxyz:
                        if verbose > 0:
                            print("ERROR (ESTIMATE_INDICATOR2D): covariance model: range ('r') not compatible with simulation grid")
                        return None
            # additional parameter (s)
            if 's' in el[1].keys():
                s  = el[1]['s']
                if np.size(s) != 1 and np.size(s) != nxyz:
                    if verbose > 0:
                        print("ERROR (ESTIMATE_INDICATOR2D): covariance model: parameter ('s') not compatible with simulation grid")
                    return None

        # alpha
        angle = cov_model.alpha
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            if verbose > 0:
                print("ERROR (ESTIMATE_INDICATOR2D): covariance model: angle (alpha) not compatible with simulation grid")
            return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     if verbose > 0:
    #         print("ERROR (ESTIMATE_INDICATOR2D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 1
    elif method == 'ordinary_kriging':
        computationMode = 0
    else:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR2D): 'method' is not valid")
        return None

    # data points: x, v
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 2) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            if verbose > 0:
                print("(ERROR (ESTIMATE_INDICATOR2D): length of 'v' is not valid")
            return None
        xc = x[:,0]
        yc = x[:,1]
        zc = np.ones_like(xc) * oz + 0.5 * sz
        dataPointSet.append(
            PointSet(npt=v.shape[0], nv=4, val=np.array((xc, yc, zc, v)), varname=['X', 'Y', 'Z', varname])
            )

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            if verbose > 0:
                print("ERROR (ESTIMATE_INDICATOR2D): 'mask' is not valid")
            return None

    # Check parameters - use_unique_neighborhood (length)
    use_unique_neighborhood = np.asarray(use_unique_neighborhood, dtype='bool').reshape(-1)
    if len(use_unique_neighborhood) == 1:
        use_unique_neighborhood = np.repeat(use_unique_neighborhood, ncategory)
    elif len(use_unique_neighborhood) != ncategory:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR2D): 'use_unique_neighborhood' of invalid length")
        return None

    # Check parameters - searchRadiusRelative (length)
    searchRadiusRelative = np.asarray(searchRadiusRelative, dtype='float').reshape(-1)
    if len(searchRadiusRelative) == 1:
        searchRadiusRelative = np.repeat(searchRadiusRelative, ncategory)
    elif len(searchRadiusRelative) != ncategory:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR2D): 'searchRadiusRelative' of invalid length")
        return None

    # Check parameters - nneighborMax (length)
    nneighborMax = np.asarray(nneighborMax, dtype='intc').reshape(-1)
    if len(nneighborMax) == 1:
        nneighborMax = np.repeat(nneighborMax, ncategory)
    elif len(nneighborMax) != ncategory:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR2D): 'nneighborMax' of invalid length")
        return None

    # Check parameters - searchNeighborhoodSortMode (length)
    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode).reshape(-1)
    if len(searchNeighborhoodSortMode) == 1:
        searchNeighborhoodSortMode = np.repeat(searchNeighborhoodSortMode, ncategory)
    elif len(searchNeighborhoodSortMode) != ncategory:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR2D): 'searchNeighborhoodSortMode' of invalid length")
        return None

    # If unique neighborhood is used, set searchRadiusRelative to -1
    #    (and initialize nneighborMax, searchNeighborhoodSortMode (unused))
    # else: check the parameters
    for i in range(ncategory):
        if use_unique_neighborhood[i]:
            searchRadiusRelative[i] = -1.0
            nneighborMax[i] = 1
            searchNeighborhoodSortMode[i] = 0

        else:
            if searchRadiusRelative[i] < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
                if verbose > 0:
                    print("ERROR (ESTIMATE_INDICATOR2D): a 'searchRadiusRelative' is too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
                return None

            if nneighborMax[i] != -1 and nneighborMax[i] <= 0:
                if verbose > 0:
                    print("ERROR (ESTIMATE_INDICATOR2D): any 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
                return None

            if searchNeighborhoodSortMode[i] is None:
                # set greatest possible value
                if cm_for_cat[i].is_stationary():
                    searchNeighborhoodSortMode[i] = 2
                elif cm_for_cat[i].is_orientation_stationary() and cm_for_cat[i].is_range_stationary():
                    searchNeighborhoodSortMode[i] = 1
                else:
                    searchNeighborhoodSortMode[i] = 0
            else:
                if searchNeighborhoodSortMode[i] == 2:
                    if not cm_for_cat[i].is_stationary():
                        if verbose > 0:
                            print("ERROR (ESTIMATE_INDICATOR2D): 'searchNeighborhoodSortMode set to 2' not allowed with non-stationary covariance model")
                        return None
                elif searchNeighborhoodSortMode[i] == 1:
                    if not cm_for_cat[i].is_orientation_stationary() or not cm_for_cat[i].is_range_stationary():
                        if verbose > 0:
                            print("ERROR (ESTIMATE_INDICATOR2D): 'searchNeighborhoodSortMode set to 1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                        return None

    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode, dtype='intc')

    # Check parameters - probability
    if probability is not None:
        # if method == 'ordinary_kriging':
        #     if verbose > 0:
        #         print("ERROR (ESTIMATE_INDICATOR2D): specifying 'probability' not allowed with ordinary kriging")
        #     return None
        probability = np.asarray(probability, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if probability.size not in (ncategory, ncategory*nxyz):
            if verbose > 0:
                print("ERROR (ESTIMATE_INDICATOR2D): size of 'probability' is not valid")
            return None

    # --- Fill mpds_geosClassicInput structure (C)
    mpds_geosClassicIndicatorInput, flag = fill_mpds_geosClassicIndicatorInput(
        space_dim,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        varname,
        ncategory,
        category_values,
        outputReportFile,
        computationMode,
        cm_for_cat,
        None,
        dataPointSet,
        mask,
        probability,
        searchRadiusRelative,
        nneighborMax,
        searchNeighborhoodSortMode,
        0,
        0)

    if not flag:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR2D): can not fill input structure!")
        return None

    # --- Prepare mpds_geosClassicIOutput structure (C)
    # Allocate mpds_geosClassicOutput
    mpds_geosClassicOutput = geosclassic.malloc_MPDS_GEOSCLASSICOUTPUT()

    # Init mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicInitGeosClassicOutput(mpds_geosClassicOutput)

    # --- Set progress monitor
    mpds_progressMonitor = geosclassic.malloc_MPDS_PROGRESSMONITOR()
    geosclassic.MPDSInitProgressMonitor(mpds_progressMonitor)

    # Set function to update progress monitor:
    # according to geosclassic.MPDS_SHOW_PROGRESS_MONITOR set to 4 for compilation of py module
    # the function
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor1_ptr: warning only
    if verbose < 3:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

    # Free memory on C side: mpds_geosClassicIndicatorInput
    geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
    #geosclassic.MPDSFree(mpds_geosClassicIndicatorInput)
    geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)

    if err:
        if verbose > 0:
            err_message = geosclassic.mpds_get_error_message(-err)
            err_message = err_message.replace('\n', '')
            print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 2 and geosclassic_output and geosclassic_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(geosclassic_output['nwarning']))
        for i, warning_message in enumerate(geosclassic_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def estimateIndicator3D(
        category_values,
        cov_model_for_category,
        dimension, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
        method='simple_kriging',
        probability=None,
        x=None, v=None,
        mask=None,
        use_unique_neighborhood=False,
        searchRadiusRelative=1.,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        seed=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2):
    """
    Computes estimate probabilities of categories (indicators) for 3D grid
    based on simple or ordinary kriging.

    :param category_values:
                        (sequence of floats or ints) list of the category values;
                            let ncategory be the length of the list, then:
                            - if ncategory == 1:
                                - the unique category value given must not be
                                equal to 0
                                - it is used for a binary case with values
                                ("unique category value", 0), where 0 indicates
                                the absence of the considered medium
                                - conditioning data values should be
                                "unique category value" or 0
                            - if ncategory >= 2:
                                - it is used for a multi-category case with given
                                values (distinct)
                                - conditioning data values should be in the list
                                of given values

    :param cov_model_for_category:
                        (sequence of covariance model of length ncategory (see
                            category_values), or one covariance model, recycled)
                            a covariance model per category,
                            with entry for covariance model:
                                (CovModel3D class) covariance model in 3D, see
                                    definition of the class in module
                                    geone.covModel
                            or
                                (CovModel1D class) covariance model in 1D, see
                                    definition of the class in module
                                    geone.covModel, it is then transformed to an
                                    isotropic (omni-directional) covariance model
                                    in 3D

    :param dimension:   (sequence of 3 ints) (nx, ny, nz), number of cells
                            in x-, y-, z-axis direction
    :param spacing:     (sequence of 3 floats) (sx, sy, sz), spacing between
                            two adjacent cells in x-, y-, z-axis direction
    :param origin:      (sequence of 3 floats) (ox, oy, oy), origin of the 3D
                            simulation - used for localizing the conditioning
                            points
    :param method:      (string) indicates the method used:
                            - 'simple_kriging':
                                simulation based on simple kriging
                            - 'ordinary_kriging':
                                simulation based on ordinary kriging

    :param probability: (None or sequence of floats of length ncategory or
                            ndarray of floats) probability for each category:
                                - None :
                                    proportion of each category in the hard data
                                    values (stationary), (uniform distribution if
                                    no hard data)
                                - sequence of floats of length ncategory:
                                    for stationary probabilities (set manually),
                                    if ncategory > 1, the sum of the probabilities
                                    must be equal to one
                                - ndarray: for non stationary probabilities,
                                    must contain ncategory * ngrid_cells entries
                                    where ngrid_cells is the number of grid cells
                                    (reshaped if needed); the first ngrid_cells
                                    entries are the probabities for the first
                                    category in the simulation grid, the next
                                    ngrid_cells entries those of the second
                                    category, and so on
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param x:           (2-dimensional array of dim n x 3, or
                            1-dimensional array of dim 3 or None) coordinate of
                            conditioning points for hard data
    :param v:           (1-dimensional array or float or None) value
                            at conditioning points for hard data
                            (same type as x)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not
                            simulated cell (nunber of entries should be equal to
                            the number of grid cells)

    :param use_unique_neighborhood:
                        (sequence of ncategory bools (or bool, recycled))
                            indicating if a unique neighborhood is used, for each
                            category:
                            - True: all data points are taken into account for
                                computing estimate probabilities;
                                in this case: parameters
                                    searchRadiusRelative,
                                    nneighborMax,
                                    searchNeighborhoodSortMode,
                                are unused
                            - False: only data points within a search ellipsoid
                                are taken into account for computing estimate
                                probabilities (see parameters
                                searchRadiusRelative, nneighborMax,
                                searchNeighborhoodSortMode)

    :param searchRadiusRelative:
                        (sequence of ncategory floats (or float, recycled))
                            indicating how restricting the search ellipsoid
                            (should be positive) for each category: let r_i be
                            the ranges of the covariance model along its main
                            axes, if x is a node to be simulated, a node y is
                            taken into account iff it is within the ellipsoid
                            centered at x of half-axes searchRadiusRelative * r_i
                            Note:
                                - if a range is a variable parameter, its maximal
                                value over the simulation grid is considered
                                - if orientation of the covariance model is
                                non-stationary, a "circular search neighborhood"
                                is considered with the radius set to the maximum
                                of all ranges

    :param nneighborMax:
                        (sequence of ncategory ints (or int, recycled)) maximum
                            number of nodes retrieved from the search ellipsoid,
                            for each category, set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (sequence of ncategory ints (or int, recycled))
                            indicating how to sort the search neighboorhood nodes
                            (neighbors) for each category, they are sorted in
                            increasing order according to:
                            - searchNeighborhoodSortMode = 0:
                                distance in the usual axes system
                            - searchNeighborhoodSortMode = 1:
                                distance in the axes sytem supporting the
                                covariance model and accounting for anisotropy
                                given by the ranges
                            - searchNeighborhoodSortMode = 2:
                                minus the evaluation of the covariance model
                            Notes:
                            - if the covariance model has any variable parameter
                                (non-stationary), then
                                searchNeighborhoodSortMode = 2 is not allowed
                            - if the covariance model has any range or angle set
                                as a variable parameter, then
                                searchNeighborhoodSortMode must be set to 0
                            - greatest possible value as default

    :param outputReportFile:
                        (string or None) name of the report file,
                            if None: no report file

    :param nthreads:
                        (int) number of thread(s) to use for "GeosClassicSim"
                            program (C), (nthreads = -n <= 0: for maximal number
                            of threads except n, but at least 1)
    :param verbose:
                (int) verbose mode, integer >=0, higher implies more display
                    - 0: no display
                    - 1: only errors
                    - 2: errors and warnings (+ some info)
                    - 3 (or >2): all information

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img class) output image, with image.nv=ncategory variables
                    (estimate probabilities (for each category))
                    (image is None if mpds_geosClassicOutput->outputImage is
                    NULL)
        nwarning:
                (int) total number of warning(s) encountered
                    (same warnings can be counted several times)
        warnings:
                (list of strings) list of distinct warnings encountered
                    (can be empty)
    """

    # --- Set grid geometry and varname
    # Set grid geometry
    nx, ny, nz = dimension
    sx, sy, sz = spacing
    ox, oy, oz = origin

    nxy = nx * ny
    nxyz = nxy * nz

    # spatial dimension
    space_dim = 3

    # Set varname
    varname = 'V0'

    # --- Check and prepare parameters
    # category_values and ncategory (computed)
    try:
        category_values = np.asarray(category_values, dtype='float').reshape(-1)
    except:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR3D): 'category_values' is not valid")
        return None

    ncategory = len(category_values)
    if ncategory <= 0:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR3D): 'category_values' is empty")
        return None

    # cov_model_for_category
    cov_model_for_category = np.asarray(cov_model_for_category).reshape(-1)
    if not np.all([isinstance(c, gcm.CovModel3D) for c in cov_model_for_category]):
        # cov model will be converted:
        #    as applying modification in an array is persistent at exit,
        #    work on a copy to ensure no modification of the initial entry
        cm_for_cat = np.deepcopy(cov_model_for_category)
    else:
        cm_for_cat = cov_model_for_category

    cm_for_cat = np.asarray(cm_for_cat).reshape(-1)
    if len(cm_for_cat) == 1:
        cm_for_cat = np.repeat(cm_for_cat, ncategory)
    elif len(cm_for_cat) != ncategory:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR3D): 'cov_model_for_category' of invalid length")
        return None
    if not np.all([isinstance(c, gcm.CovModel3D) for c in cm_for_cat]):
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR3D): 'cov_model_for_category' should contains CovModel3D objects")
        return None

    for cov_model in cm_for_cat:
        for el in cov_model.elem:
            # weight
            w = el[1]['w']
            if np.size(w) != 1 and np.size(w) != nxyz:
                if verbose > 0:
                    print("ERROR (ESTIMATE_INDICATOR3D): covariance model: weight ('w') not compatible with simulation grid")
                return None
            # ranges
            if 'r' in el[1].keys():
                for r in el[1]['r']:
                    if np.size(r) != 1 and np.size(r) != nxyz:
                        if verbose > 0:
                            print("ERROR (ESTIMATE_INDICATOR3D): covariance model: range ('r') not compatible with simulation grid")
                        return None
            # additional parameter (s)
            if 's' in el[1].keys():
                s  = el[1]['s']
                if np.size(s) != 1 and np.size(s) != nxyz:
                    if verbose > 0:
                        print("ERROR (ESTIMATE_INDICATOR3D): covariance model: parameter ('s') not compatible with simulation grid")
                    return None

        # alpha
        angle = cov_model.alpha
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            if verbose > 0:
                print("ERROR (ESTIMATE_INDICATOR3D): covariance model: angle (alpha) not compatible with simulation grid")
            return None

        # beta
        angle = cov_model.beta
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            if verbose > 0:
                print("ERROR (ESTIMATE_INDICATOR3D): covariance model: angle (beta) not compatible with simulation grid")
            return None

        # gamma
        angle = cov_model.gamma
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            if verbose > 0:
                print("ERROR (ESTIMATE_INDICATOR3D): covariance model: angle (gamma) not compatible with simulation grid")
            return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     if verbose > 0:
    #         print("ERROR (ESTIMATE_INDICATOR3D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 1
    elif method == 'ordinary_kriging':
        computationMode = 0
    else:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR3D): 'method' is not valid")
        return None

    # data points: x, v
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 3) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            if verbose > 0:
                print("(ERROR (SIMUL_3D): length of 'v' is not valid")
            return None
        xc = x[:,0]
        yc = x[:,1]
        zc = x[:,2]
        dataPointSet.append(
            PointSet(npt=v.shape[0], nv=4, val=np.array((xc, yc, zc, v)), varname=['X', 'Y', 'Z', varname])
            )

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            if verbose > 0:
                print("ERROR (ESTIMATE_INDICATOR3D): 'mask' is not valid")
            return None

    # Check parameters - use_unique_neighborhood (length)
    use_unique_neighborhood = np.asarray(use_unique_neighborhood, dtype='bool').reshape(-1)
    if len(use_unique_neighborhood) == 1:
        use_unique_neighborhood = np.repeat(use_unique_neighborhood, ncategory)
    elif len(use_unique_neighborhood) != ncategory:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR3D): 'use_unique_neighborhood' of invalid length")
        return None

    # Check parameters - searchRadiusRelative (length)
    searchRadiusRelative = np.asarray(searchRadiusRelative, dtype='float').reshape(-1)
    if len(searchRadiusRelative) == 1:
        searchRadiusRelative = np.repeat(searchRadiusRelative, ncategory)
    elif len(searchRadiusRelative) != ncategory:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR3D): 'searchRadiusRelative' of invalid length")
        return None

    # Check parameters - nneighborMax (length)
    nneighborMax = np.asarray(nneighborMax, dtype='intc').reshape(-1)
    if len(nneighborMax) == 1:
        nneighborMax = np.repeat(nneighborMax, ncategory)
    elif len(nneighborMax) != ncategory:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR3D): 'nneighborMax' of invalid length")
        return None

    # Check parameters - searchNeighborhoodSortMode (length)
    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode).reshape(-1)
    if len(searchNeighborhoodSortMode) == 1:
        searchNeighborhoodSortMode = np.repeat(searchNeighborhoodSortMode, ncategory)
    elif len(searchNeighborhoodSortMode) != ncategory:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR3D): 'searchNeighborhoodSortMode' of invalid length")
        return None

    # If unique neighborhood is used, set searchRadiusRelative to -1
    #    (and initialize nneighborMax, searchNeighborhoodSortMode (unused))
    # else: check the parameters
    for i in range(ncategory):
        if use_unique_neighborhood[i]:
            searchRadiusRelative[i] = -1.0
            nneighborMax[i] = 1
            searchNeighborhoodSortMode[i] = 0

        else:
            if searchRadiusRelative[i] < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
                if verbose > 0:
                    print("ERROR (ESTIMATE_INDICATOR3D): a 'searchRadiusRelative' is too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
                return None

            if nneighborMax[i] != -1 and nneighborMax[i] <= 0:
                if verbose > 0:
                    print("ERROR (ESTIMATE_INDICATOR3D): any 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
                return None

            if searchNeighborhoodSortMode[i] is None:
                # set greatest possible value
                if cm_for_cat[i].is_stationary():
                    searchNeighborhoodSortMode[i] = 2
                elif cm_for_cat[i].is_orientation_stationary() and cm_for_cat[i].is_range_stationary():
                    searchNeighborhoodSortMode[i] = 1
                else:
                    searchNeighborhoodSortMode[i] = 0
            else:
                if searchNeighborhoodSortMode[i] == 2:
                    if not cm_for_cat[i].is_stationary():
                        if verbose > 0:
                            print("ERROR (ESTIMATE_INDICATOR3D): 'searchNeighborhoodSortMode set to 2' not allowed with non-stationary covariance model")
                        return None
                elif searchNeighborhoodSortMode[i] == 1:
                    if not cm_for_cat[i].is_orientation_stationary() or not cm_for_cat[i].is_range_stationary():
                        if verbose > 0:
                            print("ERROR (ESTIMATE_INDICATOR3D): 'searchNeighborhoodSortMode set to 1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                        return None

    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode, dtype='intc')

    # Check parameters - probability
    if probability is not None:
        # if method == 'ordinary_kriging':
        #     if verbose > 0:
        #         print("ERROR (ESTIMATE_INDICATOR3D): specifying 'probability' not allowed with ordinary kriging")
        #     return None
        probability = np.asarray(probability, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if probability.size not in (ncategory, ncategory*nxyz):
            if verbose > 0:
                print("ERROR (ESTIMATE_INDICATOR3D): size of 'probability' is not valid")
            return None

    # --- Fill mpds_geosClassicInput structure (C)
    mpds_geosClassicIndicatorInput, flag = fill_mpds_geosClassicIndicatorInput(
        space_dim,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        varname,
        ncategory,
        category_values,
        outputReportFile,
        computationMode,
        cm_for_cat,
        None,
        dataPointSet,
        mask,
        probability,
        searchRadiusRelative,
        nneighborMax,
        searchNeighborhoodSortMode,
        0,
        0)

    if not flag:
        if verbose > 0:
            print("ERROR (ESTIMATE_INDICATOR3D): can not fill input structure!")
        return None

    # --- Prepare mpds_geosClassicIOutput structure (C)
    # Allocate mpds_geosClassicOutput
    mpds_geosClassicOutput = geosclassic.malloc_MPDS_GEOSCLASSICOUTPUT()

    # Init mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicInitGeosClassicOutput(mpds_geosClassicOutput)

    # --- Set progress monitor
    mpds_progressMonitor = geosclassic.malloc_MPDS_PROGRESSMONITOR()
    geosclassic.MPDSInitProgressMonitor(mpds_progressMonitor)

    # Set function to update progress monitor:
    # according to geosclassic.MPDS_SHOW_PROGRESS_MONITOR set to 4 for compilation of py module
    # the function
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor1_ptr: warning only
    if verbose < 3:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

    # Free memory on C side: mpds_geosClassicIndicatorInput
    geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
    #geosclassic.MPDSFree(mpds_geosClassicIndicatorInput)
    geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)

    if err:
        if verbose > 0:
            err_message = geosclassic.mpds_get_error_message(-err)
            err_message = err_message.replace('\n', '')
            print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 2 and geosclassic_output and geosclassic_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(geosclassic_output['nwarning']))
        for i, warning_message in enumerate(geosclassic_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imgDistanceImage(
        input_image,
        distance_type='L2',
        distance_negative=False,
        nthreads=-1):
    """
    Computes the image of the distances to the set of non zero values in the
    input image. The distances are computed for each variable v over the image
    grid: distance to the set S = {v!=0}. Distance is equal to zero for all cells
    in S if the keyword argument distance_negative is False (default). If
    distance_negative is True, the distance to the border of S is computed for
    the cells in the interior of S (i.e. in S but not on the border), and the
    opposite (negative) is retrieved for that cells. The output image has the
    same number of variable(s) and the same size (grid geometry) as the input
    image.

    Algorithm from "A GENERAL ALGORITHM FOR COMPUTING DISTANCE TRANSFORMS IN
    LINEAR TIME" by A. MEIJSTER, J.B.T.M. ROERDINK, and W.H. HESSELINK

    :param input_image:     (Img class) input image
    :param distance_type:   (string) type of distance, available types:
                                'L1', 'L2' (default)

    :param distance_negative:
                            (bool) indicates what is computed for cell in the
                                set S = {v!=0} (for a variable v):
                                - False (default): distance set to zero for all
                                    cells in S
                                - True: zero for cells on the border of S, and
                                    negative distance to the border of S for
                                    cells in the interior of S (i.e. in S but
                                    not in the border)

    :param nthreads:        (int) number of thread(s) to use for program (C),
                                (nthreads = -n <= 0: for maximal number of
                                threads except n, but at least 1)

    :return output_image:   (Img class) output image containing the computed
                                distances.
    """

    # --- Check
    if distance_type not in ('L1', 'L2'):
        print("ERROR: unknown 'distance_type'")
        return None

    # Set input image "in C"
    input_image_c = img_py2C(input_image)

    # Allocate output image "in C"
    output_image_c = geosclassic.malloc_MPDS_IMAGE()
    geosclassic.MPDSInitImage(output_image_c)

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    # --- Compute distances (launch C code)
    if distance_type == 'L1':
        if distance_negative:
            err = geosclassic.MPDSOMPImageDistanceL1Sign(input_image_c, output_image_c, nth)
        else:
            err = geosclassic.MPDSOMPImageDistanceL1(input_image_c, output_image_c, nth)
    elif distance_type == 'L2':
        if distance_negative:
            err = geosclassic.MPDSOMPImageDistanceEuclideanSign(input_image_c, output_image_c, nth)
        else:
            err = geosclassic.MPDSOMPImageDistanceEuclidean(input_image_c, output_image_c, nth)
    else:
        print("ERROR: 'distance_type' not valid")
        return None

    # --- Retrieve output image "in python"
    if err:
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        print(err_message)
        output_image = None
    else:
        output_image = img_C2py(output_image_c)

    # Free memory on C side: input_image_c
    geosclassic.MPDSFreeImage(input_image_c)
    #geosclassic.MPDSFree(input_image_c)
    geosclassic.free_MPDS_IMAGE(input_image_c)

    # Free memory on C side: output_image_c
    geosclassic.MPDSFreeImage(output_image_c)
    #geosclassic.MPDSFree(output_image_c)
    geosclassic.free_MPDS_IMAGE(output_image_c)

    return output_image
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imgGeobodyImage(
        input_image,
        var_index=0,
        bound_inf=0.0,
        bound_sup=None,
        bound_inf_excluded=True,
        bound_sup_excluded=True,
        complementary_set=False,
        connect_type='connect_face'):
    """
    Computes the geobody image (map) for one variable of the input image.
    For the considered variable v, an indicator I is defined as
        I(x) = 1 if v(x) is between bound_inf and bound_sup
        I(x) = 0 otherwise
    Then lower (resp. upper) bound bound_inf (resp. bound_sup) is exluded from
    the set I=1 if bound_inf_excluded (resp. bound_sup_excluded) is True or
    included if bound_inf_excluded (resp. bound_sup_excluded) is False.
    Hence:
        - with bound_inf_excluded, bound_sup_excluded = (True, True):
            I(x) = 1 iff bound_inf < v(x) < bound_sup
            default: I(x) = 1 iff 0 < v(x)
        - with bound_inf_excluded, bound_sup_excluded = (True, False):
            I(x) = 1 iff bound_inf < v(x) <= bound_sup
        - with bound_inf_excluded, bound_sup_excluded = (False, True):
            I(x) = 1 iff bound_inf <= v(x) < bound_sup
        - with bound_inf_excluded, bound_sup_excluded = (False, False):
            I(x) = 1 iff bound_inf <= v(x) <= bound_sup

    If complementary_set is True, the variable IC(x) = 1 - I(x) is used
    instead of variable I, i.e. the set I=0 and I=1 are swapped.

    The geobody image (map) is computed for the indicator variable I, which
    consists in labelling the connected components from 1 to n, i.e.
        C(x) = 0     if I(x) = 0
        C(x) = k > 0 if I(x) = 1 and x is in the k-th connected component

    Two cells x and y in the grid are said connected, x <-> y, if there exists
    a path between x and y going composed of adjacent cells all in the set I=1.
    Following this definition, we have
        x <-> y iff C(x) = C(y) > 0

    The definition of adjacent cells depends on the keyword argument
    connect_type:
        - connect_type = connect_face (default):
            two grid cells are adjacent if they have a common face
        - connect_type = connect_face_edge:
            two grid cells are adjacent if they have a common face
            or a common edge
        - connect_type = connect_face_edge_corner:
            two grid cells are adjacent if they have a common face
            or a common edge or a common corner

    Algorithm used is described in: Hoshen and Kopelman (1976) Physical Review B,
    14(8):3438.

    :param input_image:     (Img class) input image
    :param var_index:       (int) index of the considered variable in input image
                                (default: 0)
    :param bound_inf:       (float) lower bound of the interval defining the
                                indicator variable (default: 0.0)
    :param bound_sup:       (float) upper bound of the interval defining the
                                indicator variable (default: None, bound_sup is
                                set to "infinity")
    :param bound_inf_excluded:
                            (bool) lower bound is excluded from the interval
                                defining the indicator variable (True, default)
                                or included (False)
    :param bound_sup_excluded:
                            (bool) upper bound is excluded from the interval
                                defining the indicator variable (True, default)
                                or included (False)
    :param complementary_set:
                            (bool) the complementary indicator variable
                                (IC = 1-I) is used if True, indicator variable I
                                is used if False (default)
    :param connect_type:    (string) indicates which definition of adjacent
                                cells is used (see above), available mode:
                                    'connect_face' (default),
                                    'connect_face_edge',
                                    'connect_face_edge_corner'

    :return output_image:   (Img class) output image containing the geobody
                                labels.
    """

    # --- Check
    if connect_type not in ('connect_face', 'connect_face_edge', 'connect_face_edge_corner'):
        print("ERROR: unknown 'connect_type'")
        return None

    if var_index < 0 or var_index >= input_image.nv:
        print("ERROR: 'var_index' not valid")
        return None

    if bound_sup is None:
        bound_sup = 1. + np.nanmax(input_image.val[var_index])

    # Allocate variable in C
    rangeValueMin_c = geosclassic.new_real_array(1)
    geosclassic.mpds_set_real_vector_from_array(rangeValueMin_c, 0, np.array([bound_inf], dtype='float'))

    rangeValueMax_c = geosclassic.new_real_array(1)
    geosclassic.mpds_set_real_vector_from_array(rangeValueMax_c, 0, np.array([bound_sup], dtype='float'))

    ngeobody_c = geosclassic.new_int_array(1)

    # Set input image "in C"
    input_image_c = img_py2C(input_image)

    # Allocate output image "in C"
    output_image_c = geosclassic.malloc_MPDS_IMAGE()
    geosclassic.MPDSInitImage(output_image_c)

    # --- Compute geobody image (launch C code)
    if connect_type == 'connect_face':
        g = geosclassic.MPDSImageGeobody6
    elif connect_type == 'connect_face_edge':
        g = geosclassic.MPDSImageGeobody18
    elif connect_type == 'connect_face_edge_corner':
        g = geosclassic.MPDSImageGeobody26
    else:
        print("ERROR: 'connect_type' not valid")
        return None

    err = g(input_image_c, output_image_c, var_index,
            complementary_set,
            1, rangeValueMin_c, rangeValueMax_c, bound_inf_excluded, bound_sup_excluded,
            ngeobody_c)

    # --- Retrieve output image "in python"
    if err:
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        print(err_message)
        output_image = None
    else:
        output_image = img_C2py(output_image_c)
        # # Retrieve the number of geobody (not used, this is simple the max of the output image (max label))
        # ngeobody = np.zeros(1, dtype='intc') # 'intc' for C-compatibility
        # geosclassic.mpds_get_array_from_int_vector(ngeobody_c, 0, ngeobody)
        # ngeobody = ngeobody[0]

    # Free memory on C side: input_image_c
    geosclassic.MPDSFreeImage(input_image_c)
    #geosclassic.MPDSFree(input_image_c)
    geosclassic.free_MPDS_IMAGE(input_image_c)

    # Free memory on C side: output_image_c
    geosclassic.MPDSFreeImage(output_image_c)
    #geosclassic.MPDSFree(output_image_c)
    geosclassic.free_MPDS_IMAGE(output_image_c)

    # Free memory on C side: rangeValueMin_c, rangeValueMax_c, ngeobody_c
    geosclassic.delete_real_array(rangeValueMin_c)
    geosclassic.delete_real_array(rangeValueMax_c)
    geosclassic.delete_int_array(ngeobody_c)
    # geosclassic.MPDSFree(rangeValueMin_c)
    # geosclassic.MPDSFree(rangeValueMax_c)
    # geosclassic.MPDSFree(ngeobody_c)

    return output_image
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imgTwoPointStatisticsImage(
        input_image,
        var_index=0,
        hx_min=None,
        hx_max=None,
        hx_step=1,
        hy_min=None,
        hy_max=None,
        hy_step=1,
        hz_min=None,
        hz_max=None,
        hz_step=1,
        stat_type='covariance',
        show_progress=False,
        nthreads=-1):
    """
    Computes two-point statistics image (map) for one variable of the input image,
    i.e. g(h) (see below) for given lag vector h.
    The type of two-point statistics is indicated by the keyword argument
    stat_type. Available two-point statistics (h is a lag vector, v(x) is the
    value of the variable at cell x):
        correlogram            : g(h) = cor(v(x), v(x+h)) (linear correlation)
        connectivity_func0     : g(h) = P(v(x)=v(x+h) > 0)
        connectivity_func1     : g(h) = P(v(x)=v(x+h) > 0 | v(x) > 0)
        connectivity_func2     : g(h) = P(v(x)=v(x+h) > 0 | v(x) > 0, v(x+h) > 0)
        covariance (default)   : g(h) = cov(v(x), v(x+h))
        covariance_not_centered: g(h) = E[v(x)*v(x+h)]
        transiogram            : g(h) = P(v(x+h) > 0 | v(x) > 0)
        variogram              : g(h) = 0.5 * E[(v(x)-v(x+h))**2]

    Notes:
    - a transiogram can be applied on a binary variable
    - a connectivity function (connectivity_func[012]) should be applied on
        a variable consisting of geobody (connected component) labels,
        i.e. input_image should be the output image returned by the function
        imgGeobodyImage;
        in that case, denoting I(x) is the indicator variable defined as
        I(x) = 1 iff v(x)>0, the variable v is the geobody label for the
        indicator variable I an we have the relations
            P(v(x) = v(x+h) > 0)
            = P(v(x)=v(x+h) > 0 | v(x) > 0, v(x+h) > 0) * P(v(x) > 0, v(x+h) > 0)
            = P(v(x)=v(x+h) > 0 | v(x) > 0, v(x+h) > 0) * P(I(x)*I(x+h))
            = P(v(x)=v(x+h) > 0 | v(x) > 0, v(x+h) > 0) * E(I(x)*I(x+h))
        i.e.
            P(x <-> x+h) = P(x <-> x+h | x, x+h in {I=1}) * E(I(x)*I(x+h))
        "connectivity_func0(v) = connectivity_func2(v)*covariance_not_centered(I)"
        (see definition of "is connected to" (<->) in the function
        imgGeobodyImage).
        See reference:
            Renard P, Allard D (2013), Connectivity metrics for subsurface flow
            and transport. Adv Water Resour 51:168–196.
            https://doi.org/10.1016/j.advwatres.2011.12.001

    The output image has one variable and its grid is defined according the
    considered lags h given through the arguments:
        hx_min, hx_max, hx_step,
        hy_min, hy_max, hy_step,
        hz_min, hz_max, hz_step.

    The minimal (resp. maximal) lag in x direction, expressed in number of cells
    (in the input image), is given by hx_min (resp. hx_max); every hx_step cells
    from hx_min up to at most hx_max are considered as lag in x direction.
    Hence, the output image grid will have 1 + (hx_max-hx_min)//hx_step cells
    in x direction. This is similar for y and z direction.

    For example, hx_min=-10, hx_max=10, hx_step=2 implies that lags in x
    direction of -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10 cells (in input image)
    will be considered.

    :param input_image:     (Img class) input image
    :param var_index:       (int) index of the considered variable in input
                                image (default: 0)
    :param hx_min, hy_min, hz_min:
                            (int) minimal lags in x, y, z directions, expressed
                                in number of cells, default (None):
                                    hx_min = - (input_image.nx // 2)
                                    hy_min = - (input_image.ny // 2)
                                    hz_min = - (input_image.nz // 2)

    :param hx_max, hy_max, hz_max:
                            (int) maximal lags in x, y, z directions, expressed
                                in number of cells, default (None):
                                    hx_max = input_image.nx // 2
                                    hy_max = input_image.ny // 2
                                    hz_max = input_image.nz // 2
                                of cells, default (None): input_image.nx // 2

    :param hx_step, hy_step, hz_step:
                            (int) steps for considered lags in x, y, z
                                directions, expressed in number of cells,
                                default: 1

    :param stat_type:       (string) type of two-point statistics that is
                                computed, available type (see above):
                                    'correlogram',
                                    'connectivity_func0',
                                    'connectivity_func1',
                                    'connectivity_func2',
                                    'covariance',
                                    'covariance_not_centered',
                                    'transiogram',
                                    'variogram'
                                For type 'connectivity_func[012]', the input
                                image is assumed to be a geobody image (see above)

    :param show_progress:   (bool) indicates if progress is displayed (True) or
                                not (False), default: False

    :param nthreads:        (int) number of thread(s) to use for program (C),
                                (nthreads = -n <= 0: for maximal number of
                                threads except n, but at least 1)

    :return output_image:   (Img class) output image containing the computed
                                distances.
    """

    # --- Check
    if stat_type not in ('correlogram',
                         'connectivity_func0',
                         'connectivity_func1',
                         'connectivity_func2',
                         'covariance',
                         'covariance_not_centered',
                         'transiogram',
                         'variogram'):
        print("ERROR: unknown 'stat_type'")
        return None

    if var_index < 0 or var_index >= input_image.nv:
        print("ERROR: 'var_index' not valid")
        return None

    # --- Prepare parameters
    if hx_min is None:
        hx_min = -(input_image.nx // 2)
    else:
        hx_min = int(hx_min) # ensure int type

    if hx_max is None:
        hx_max = input_image.nx // 2
    else:
        hx_max = int(hx_max) # ensure int type

    hx_step = int(hx_step) # ensure int type

    if hy_min is None:
        hy_min = -(input_image.ny // 2)
    else:
        hy_min = int(hy_min) # ensure int type

    if hy_max is None:
        hy_max = input_image.ny // 2
    else:
        hy_max = int(hy_max) # ensure int type

    hy_step = int(hy_step) # ensure int type

    if hz_min is None:
        hz_min = -(input_image.nz // 2)
    else:
        hz_min = int(hz_min) # ensure int type

    if hz_max is None:
        hz_max = input_image.nz // 2
    else:
        hz_max = int(hz_max) # ensure int type

    hz_step = int(hz_step) # ensure int type

    # Set input image "in C"
    input_image_c = img_py2C(input_image)

    # Allocate output image "in C"
    output_image_c = geosclassic.malloc_MPDS_IMAGE()
    geosclassic.MPDSInitImage(output_image_c)

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    # --- Compute two-point statistics (launch C code)
    if stat_type == 'correlogram':
        g = geosclassic.MPDSOMPImageCorrelogram
    elif stat_type == 'covariance':
        g = geosclassic.MPDSOMPImageCovariance
    elif stat_type == 'connectivity_func0':
        g = geosclassic.MPDSOMPImageConnectivityFunction0
    elif stat_type == 'connectivity_func1':
        g = geosclassic.MPDSOMPImageConnectivityFunction1
    elif stat_type == 'connectivity_func2':
        g = geosclassic.MPDSOMPImageConnectivityFunction2
    elif stat_type == 'covariance_not_centered':
        g = geosclassic.MPDSOMPImageCovarianceNotCentred
    elif stat_type == 'transiogram':
        g = geosclassic.MPDSOMPImageTransiogram
    elif stat_type == 'variogram':
        g = geosclassic.MPDSOMPImageVariogram
    else:
        print("ERROR: 'stat_type' not valid")
        return None

    err = g(input_image_c, output_image_c, var_index,
            hx_min, hx_max, hx_step,
            hy_min, hy_max, hy_step,
            hz_min, hz_max, hz_step,
            show_progress, nth)

    # --- Retrieve output image "in python"
    if err:
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        print(err_message)
        output_image = None
    else:
        output_image = img_C2py(output_image_c)

    # Free memory on C side: input_image_c
    geosclassic.MPDSFreeImage(input_image_c)
    #geosclassic.MPDSFree(input_image_c)
    geosclassic.free_MPDS_IMAGE(input_image_c)

    # Free memory on C side: output_image_c
    geosclassic.MPDSFreeImage(output_image_c)
    #geosclassic.MPDSFree(output_image_c)
    geosclassic.free_MPDS_IMAGE(output_image_c)

    return output_image
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imgConnectivityGammaValue(
        input_image,
        var_index=0,
        geobody_image_in_input=False,
        complementary_set=False,
        connect_type='connect_face'):
    """
    Computes the Gamma value for one variable v of the input image:
        Gamma = 1/m^2 * sum_{i=1,...,N} n(i)^2,
    where
        N is the number of connected components (geobodies)
            of the set {v>0}
        n(i) is the size (number of cells) in the i-th connected component
        m is the size (number of cells) of the set {v>0},
        note: Gamma is set to 1.0 if N = 0
    i.e. the indicator variable I(x) = 1 iff v(x) > 0, is considered.
    The Gamma value is a global indicator of the connectivity for the binary
    image of variable I
    See reference:
        Renard P, Allard D (2013), Connectivity metrics for subsurface flow
        and transport. Adv Water Resour 51:168–196.
        https://doi.org/10.1016/j.advwatres.2011.12.001

    The definition of adjacent cells, required to compute the connected
    components, depends on the keyword argument connect_type:
        - connect_type = connect_face (default):
            two grid cells are adjacent if they have a common face
        - connect_type = connect_face_edge:
            two grid cells are adjacent if they have a common face
            or a common edge
        - connect_type = connect_face_edge_corner:
            two grid cells are adjacent if they have a common face
            or a common edge or a common corner

    :param input_image:     (Img class) input image
    :param var_index:       (int) index of the considered variable in input
                                image (default: 0)
    :param geobody_image_in_input:
                            (bool)
                                - True: the input image is already the geobody
                                    image, (variable 'var_index' is the geobody
                                    label) in this case the keyword arguments
                                    'complementary_set' and 'connect_type' are
                                    ignored, the geobody image is not computed
                                - False: the geobody image for the indicator
                                    variable {v>0} (v variable of index
                                    'var_index') is computed (default)
    :param complementary_set:
                            (bool) the complementary indicator variable
                                (IC = 1-I) is used if True, indicator variable I
                                is used if False (default)
    :param connect_type:    (string) indicates which definition of adjacent
                                cells is used (see above), available mode:
                                    'connect_face' (default),
                                    'connect_face_edge',
                                    'connect_face_edge_corner'

    :return:                (float) Gamma value (see above)
    """

    # --- Check and prepare
    if var_index < 0 or var_index >= input_image.nv:
        print("ERROR: 'var_index' not valid")
        return None

    if not geobody_image_in_input and connect_type not in ('connect_face', 'connect_face_edge', 'connect_face_edge_corner'):
        print("ERROR: unknown 'connect_type'")
        return None

    # Compute geobody image
    if not geobody_image_in_input:
        im_geobody = imgGeobodyImage(input_image,
                                     var_index,
                                     bound_inf=0.0,
                                     bound_sup=None,
                                     bound_inf_excluded=True,
                                     bound_sup_excluded=True,
                                     complementary_set=complementary_set,
                                     connect_type=connect_type)
        iv = 0
    else:
        im_geobody = input_image
        iv = var_index

    # Compute Gamma value
    if im_geobody is not None:
        ngeo = int(im_geobody.val[iv].max())
        if ngeo == 0:
            gamma = 1.0
        else:
            gamma = np.sum(np.array([float(np.sum(im_geobody.val[iv] == i))**2 for i in np.arange(1, ngeo+1)])) / float(np.sum(im_geobody.val[iv] != 0))**2
    else:
        return None

    return gamma
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imgConnectivityGammaCurves(
        input_image,
        threshold_min=None,
        threshold_max=None,
        nthreshold=50,
        connect_type='connect_face',
        show_progress=False,
        nthreads=-1):
    """
    Computes Gamma curves for an input image containing one variable v
    (continuous).
    For a threshold t:
        - we consider the indicator variable I(t) defined as
            I(t)(x) = 1 iif v(x) <= t
        - we compute
            gamma(t) = 1/m^2 * sum_{i=1,...,N} n(i)^2,
          where
            N is the number of connected components (geobodies)
                of the set {I(t)=1}
            n(i) is the size (number of cells) in the i-th connected component
            m is the size (number of cells) of the set {I(t)=1}
            note: gamma(t) is set to 1.0 if N = 0
        - we compute also gammaC(t), the gamma value for the complementary set
            {IC(t)=1} where IC(t)(x) = 1 - I(t)(x)
    This is repeated for different threshold values t, which gives the curves
    gamma(t) and gammaC(t).
    The Gamma value gamma(t) (resp. gammaC(t)) is a global indicator of the
    connectivity for the binary variable I(t) (resp. IC(t)).
    See reference:
        Renard P, Allard D (2013), Connectivity metrics for subsurface flow
        and transport. Adv Water Resour 51:168–196.
        https://doi.org/10.1016/j.advwatres.2011.12.001

    The definition of adjacent cells, required to compute the connected
    components, depends on the keyword argument connect_type:
        - connect_type = connect_face (default):
            two grid cells are adjacent if they have a common face
        - connect_type = connect_face_edge:
            two grid cells are adjacent if they have a common face
            or a common edge
        - connect_type = connect_face_edge_corner:
            two grid cells are adjacent if they have a common face
            or a common edge or a common corner

    :param input_image:     (Img class) input image, should have only one
                                variable
    :param threshold_min:   (float) minimal value of the threshold,
                                default (None): min of the input variable values
                                                minus 1.e-10
    :param threshold_max:   (float) maximal value of the threshold,
                                default (None): max of the input variable values
                                                plus 1.e-10
    :param nthreshold:      (int) number of thresholds considered (default: 50),
                                the threshold values will be:
                                numpy.linspace(threshold_min,
                                               threshold_max,
                                               nthreshold)
    :param connect_type:    (string) indicates which definition of adjacent
                                cells is used (see above), available mode:
                                    'connect_face' (default),
                                    'connect_face_edge',
                                    'connect_face_edge_corner'

    :param show_progress:   (bool) indicates if progress is displayed (True) or
                                not (False), default: False

    :param nthreads:        (int) number of thread(s) to use for program (C),
                                (nthreads = -n <= 0: for maximal number of
                                threads except n, but at least 1)

    :return out_array:      (numpy 2d-array of floats) array of shape
                                (nthreshold, 3), with the threshold values in
                                the column of index 0, and the corresponding
                                gamma and gammaC values in the column of index 1
                                and column of index 2, i.e.:
                                    out_array[:,0]: numpy.linspace(threshold_min,
                                                                   threshold_max,
                                                                   nthreshold)
                                    out_array[i,1]: gamma(out_array[i,0])
                                    out_array[i,2]: gammaC(out_array[i,0])
    """

    # --- Check and prepare
    if input_image.nv != 1:
        print("ERROR: input image must have one variable only")
        return None

    if threshold_min is None:
        threshold_min = np.nanmin(input_image.val) - 1.e-10

    if threshold_max is None:
        threshold_max = np.nanmax(input_image.val) + 1.e-10

    if threshold_min > threshold_max:
        print("ERROR: 'threshold_min' is greater than 'threshold_max'")
        return None

    if nthreshold < 0:
        print("ERROR: 'nthreshold' is negative")
        return None
    elif nthreshold == 1:
        threshold_step = 1.0
    else:
        threshold_step = (threshold_max - threshold_min) / (nthreshold - 1)

    if threshold_step < geosclassic.MPDS_EPSILON:
        print("ERROR: threshold step too small")
        return None

    if connect_type not in ('connect_face', 'connect_face_edge', 'connect_face_edge_corner'):
        print("ERROR: unknown 'connect_type'")
        return None

    # Set input image "in C"
    input_image_c = img_py2C(input_image)

    # Allocate output variable in C
    threshold_c = geosclassic.new_real_array(nthreshold)
    gamma_c = geosclassic.new_real_array(nthreshold)
    gammaC_c = geosclassic.new_real_array(nthreshold)

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    # --- Compute Gamma curves (launch C code)
    if connect_type == 'connect_face':
        g = geosclassic.MPDSOMPImageConnectivity6GlobalIndicatorCurve
    elif connect_type == 'connect_face_edge':
        g = geosclassic.MPDSOMPImageConnectivity18GlobalIndicatorCurve
    elif connect_type == 'connect_face_edge_corner':
        g = geosclassic.MPDSOMPImageConnectivity26GlobalIndicatorCurve
    else:
        print("ERROR: 'connect_type' not valid")
        return None

    err = g(input_image_c, nthreshold, threshold_min, threshold_step,
            threshold_c, gamma_c, gammaC_c,
            show_progress, nth)

    # --- Retrieve output "in python"
    if err:
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        print(err_message)
        out_array = None
    else:
        threshold = np.zeros(nthreshold)
        geosclassic.mpds_get_array_from_real_vector(threshold_c, 0, threshold)

        gamma = np.zeros(nthreshold)
        geosclassic.mpds_get_array_from_real_vector(gamma_c, 0, gamma)

        gammaC = np.zeros(nthreshold)
        geosclassic.mpds_get_array_from_real_vector(gammaC_c, 0, gammaC)

        out_array = np.array((threshold, gamma, gammaC)).reshape(3, -1).T

    # Free memory on C side: input_image_c
    geosclassic.MPDSFreeImage(input_image_c)
    #geosclassic.MPDSFree(input_image_c)
    geosclassic.free_MPDS_IMAGE(input_image_c)

    # Free memory on C side: threshold_c, gamma_c, gammaC_c
    geosclassic.delete_real_array(threshold_c)
    geosclassic.delete_real_array(gamma_c)
    geosclassic.delete_real_array(gammaC_c)
    # geosclassic.MPDSFree(threshold_c)
    # geosclassic.MPDSFree(gamma_c)
    # geosclassic.MPDSFree(gammaC_c)

    return out_array
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imgConnectivityEulerNumber(
        input_image,
        var_index=0,
        geobody_image_in_input=False,
        complementary_set=False,
        nthreads=-1):
    """
    Computes the Euler number defined related to one variable v of the input
    image, defined for the 3D image grid as
        E = number of connected components (geobodies)
            + number of "holes"
            - number of "handles"
    for the set {v>0}, i.e. the indicator variable I(x) = 1 iff v(x)>0, is
    considered.
    The Euler number E can be computed by the formula:
        E = sum_{i=1,...,N} (e0(i) - e1(i) + e2(i) - e3(i)),
    where
        - N the number of connected component (geobodies) in the set {I=1}
        - for a geobody i:
            e0(i) : the number of vertices (dim 0) in the i-th geobody
            e1(i) : the number of edges (dim 1) in the i-th geobody
            e2(i) : the number of faces (dim 2) in the i-th geobody
            e3(i) : the number of volumes (dim 3) in the i-th geobody
        where vertices, edges, faces, and volumes of each grid cell
        (3D parallelepiped element) are considered.
    See reference:
        Renard P, Allard D (2013), Connectivity metrics for subsurface flow
        and transport. Adv Water Resour 51:168–196.
        https://doi.org/10.1016/j.advwatres.2011.12.001

    Note that the connected components are computed considering two cells as
    adjacent as soon as they have a common face (connect_type='connect_face'
    for the computation of the geobody image (see function imgGeobodyImage).

    :param input_image:     (Img class) input image
    :param var_index:       (int) index of the considered variable in input
                                image (default: 0)
    :param geobody_image_in_input:
                            (bool)
                                - True: the input image is already the geobody
                                    image, (variable 'var_index' is the geobody
                                    label) in this case the keyword arguments
                                    'complementary_set' and 'connect_type' are
                                    ignored, the geobody image is not computed
                                - False: the geobody image for the indicator
                                    variable {v>0} (v variable of index
                                    'var_index') is computed (default)
    :param complementary_set:
                            (bool) the complementary indicator variable
                                (IC = 1-I) is used if True, indicator variable I
                                is used if False (default)

    :param nthreads:        (int) number of thread(s) to use for program (C),
                                (nthreads = -n <= 0: for maximal number of
                                threads except n, but at least 1)

    :return:                (float) Euler number (see above)
    """

    # --- Check and prepare
    if var_index < 0 or var_index >= input_image.nv:
        print("ERROR: 'var_index' not valid")
        return None

    # Compute geobody image
    if not geobody_image_in_input:
        im_geobody = imgGeobodyImage(input_image,
                                     var_index,
                                     bound_inf=0.0,
                                     bound_sup=None,
                                     bound_inf_excluded=True,
                                     bound_sup_excluded=True,
                                     complementary_set=complementary_set,
                                     connect_type='connect_face')
        iv = 0
    else:
        im_geobody = input_image
        iv = var_index

    # Compute Euler Number
    if im_geobody is not None:
        # Set geobody image "in C"
        im_geobody_c = img_py2C(im_geobody)

        # Allocate euler number "in C"
        euler_number_c = geosclassic.new_int_array(1)

        # --- Set number of threads
        if nthreads <= 0:
            nth = max(os.cpu_count() + nthreads, 1)
        else:
            nth = nthreads

        # Compute Euler number (launch C code)
        err = geosclassic.MPDSOMPImageConnectivityEulerNumber(im_geobody_c, var_index, euler_number_c, nth)

        # --- Retrieve output "in python"
        if err:
            err_message = geosclassic.mpds_get_error_message(-err)
            err_message = err_message.replace('\n', '')
            print(err_message)
            euler_number = None
        else:
            euler_number = np.zeros(1, dtype='intc') # 'intc' for C-compatibility
            geosclassic.mpds_get_array_from_int_vector(euler_number_c, 0, euler_number)
            euler_number = euler_number[0]

        # Free memory on C side: im_geobody_c
        geosclassic.MPDSFreeImage(im_geobody_c)
        #geosclassic.MPDSFree(im_geobody_c)
        geosclassic.free_MPDS_IMAGE(im_geobody_c)

        # Free memory on C side: euler_number_c
        geosclassic.delete_int_array(euler_number_c)
        # geosclassic.MPDSFree(euler_number_c)

    else:
        return None

    return euler_number
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imgConnectivityEulerNumberCurves(
        input_image,
        threshold_min=None,
        threshold_max=None,
        nthreshold=50,
        show_progress=False,
        nthreads=-1):
    """
    Computes the curves of Euler number for an input image containing one
    variable v (continuous).
    For a threshold t:
        - we consider the indicator variable I(t) defined as
            I(t)(x) = 1 iif v(x) <= t
        - we compute the Euler number
            E(t) = number of connected components (geobodies)
                  + number of "holes"
                  - number of "handles",
                  for the set {I(t)=1}
        - we compute also EC(t), the Euler number for the complementary set
            {IC(t)=1} where IC(t)(x) = 1 - I(t)(x)
    This is repeated for different threshold values t, which gives the curves
    of Euler numbers E(t) and EC(t).
    See function imgConnectivityEulerNumber for detail about Euler number.
    See reference:
        Renard P, Allard D (2013), Connectivity metrics for subsurface flow
        and transport. Adv Water Resour 51:168–196.
        https://doi.org/10.1016/j.advwatres.2011.12.001

    Note that the connected components are computed considering two cells as
    adjacent as soon as they have a common face (connect_type='connect_face'
    for the computation of the geobody image (see function imgGeobodyImage)).

    :param input_image:     (Img class) input image, should have only one
                                variable
    :param threshold_min:   (float) minimal value of the threshold,
                                default (None): min of the input variable values
                                                minus 1.e-10
    :param threshold_max:   (float) maximal value of the threshold,
                                default (None): max of the input variable values
                                                plus 1.e-10
    :param nthreshold:      (int) number of thresholds considered (default: 50),
                                the threshold values will be:
                                numpy.linspace(threshold_min,
                                               threshold_max,
                                               nthreshold)

    :param show_progress:   (bool) indicates if progress is displayed (True) or
                                not (False), default: False

    :param nthreads:        (int) number of thread(s) to use for program (C),
                                (nthreads = -n <= 0: for maximal number of
                                threads except n, but at least 1)

    :return out_array:      (numpy 2d-array of floats) array of shape
                                (nthreshold, 3), with the threshold values in
                                the column of index 0, and the corresponding
                                Euler numbers E and EC in the column of index 1
                                and column of index 2, i.e.:
                                    out_array[:,0]: numpy.linspace(threshold_min,
                                                                   threshold_max,
                                                                   nthreshold)
                                    out_array[i,1]: E(out_array[i,0])
                                    out_array[i,2]: EC(out_array[i,0])
    """

    # --- Check and prepare
    if input_image.nv != 1:
        print("ERROR: input image must have one variable only")
        return None

    if threshold_min is None:
        threshold_min = np.nanmin(input_image.val) - 1.e-10

    if threshold_max is None:
        threshold_max = np.nanmax(input_image.val) + 1.e-10

    if threshold_min > threshold_max:
        print("ERROR: 'threshold_min' is greater than 'threshold_max'")
        return None

    if nthreshold < 0:
        print("ERROR: 'nthreshold' is negative")
        return None
    elif nthreshold == 1:
        threshold_step = 1.0
    else:
        threshold_step = (threshold_max - threshold_min) / (nthreshold - 1)

    if threshold_step < geosclassic.MPDS_EPSILON:
        print("ERROR: threshold step too small")
        return None

    # Set input image "in C"
    input_image_c = img_py2C(input_image)

    # Allocate output variable in C
    threshold_c = geosclassic.new_real_array(nthreshold)
    euler_number_c = geosclassic.new_int_array(nthreshold)
    euler_numberC_c = geosclassic.new_int_array(nthreshold)

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    # --- Compute Euler number curves (launch C code)
    err = geosclassic.MPDSOMPImageConnectivity6EulerNumberCurve(
            input_image_c, nthreshold, threshold_min, threshold_step,
            threshold_c, euler_number_c, euler_numberC_c,
            show_progress, nth)

    # --- Retrieve output "in python"
    if err:
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        print(err_message)
        out_array = None
    else:
        threshold = np.zeros(nthreshold)
        geosclassic.mpds_get_array_from_real_vector(threshold_c, 0, threshold)

        euler_number = np.zeros(nthreshold, dtype='intc') # 'intc' for C-compatibility
        geosclassic.mpds_get_array_from_int_vector(euler_number_c, 0, euler_number)

        euler_numberC = np.zeros(nthreshold, dtype='intc') # 'intc' for C-compatibility
        geosclassic.mpds_get_array_from_int_vector(euler_numberC_c, 0, euler_numberC)

        out_array = np.array((threshold, euler_number, euler_numberC)).reshape(3, -1).T

    # Free memory on C side: input_image_c
    geosclassic.MPDSFreeImage(input_image_c)
    #geosclassic.MPDSFree(input_image_c)
    geosclassic.free_MPDS_IMAGE(input_image_c)

    # Free memory on C side: threshold_c, gamma_c, gammaC_c
    geosclassic.delete_real_array(threshold_c)
    geosclassic.delete_int_array(euler_number_c)
    geosclassic.delete_int_array(euler_numberC_c)
    # geosclassic.MPDSFree(threshold_c)
    # geosclassic.MPDSFree(euler_number_c)
    # geosclassic.MPDSFree(euler_numberC_c)

    return out_array
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Module 'geone.geosclassicinterface'.")
