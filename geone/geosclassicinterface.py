#!/usr/bin/python3
#-*- coding: utf-8 -*-

"""
Python module:  'geosclassicinterface.py'
author:         Julien Straubhaar
date:           jun-2021

Module interfacing classical geostatistics for python (estimation and simulation
based on simple and ordinary kriging).
"""

import numpy as np
import sys, os
# import multiprocessing

from geone import img
from geone.geosclassic_core import geosclassic
from geone import covModel as gcm
from geone.img import Img, PointSet

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

    coord = np.zeros(ps_c.npoint)
    geosclassic.mpds_get_array_from_real_vector(ps_c.z, 0, coord)
    v = np.hstack(coord,v)
    geosclassic.mpds_get_array_from_real_vector(ps_c.y, 0, coord)
    v = np.hstack(coord,v)
    geosclassic.mpds_get_array_from_real_vector(ps_c.x, 0, coord)
    v = np.hstack(coord,v)

    ps_py = PointSet(npt=ps_c.npoint,
                     nv=ps_c.nvar+3, val=v, varname=varname)

    np.putmask(ps_py.val, ps_py.val == geosclassic.MPDS_MISSING_VALUE, np.nan)

    return ps_py
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel1Delem_py2C(covModelElem_py, nx, ny, nz, sx, sy, sz, ox, oy, oz):
    """
    Converts an elementary covariance model 1D from python to C.
    Simulation grid geometry is specified in case of non-stationary covariance model.

    :param covModelElem_py: (2-tuple) elementary covariance model 1D in python:
                                (t, d) corresponds to an elementary model with:
                                    t: (string) the type, could be
                                        'nugget', 'spherical', 'exponential',
                                        'gaussian', 'cubic', 'power'
                                    d: (dict) dictionary of required parameters
                                        to be passed to the elementary model
                                        (value can be a "singe value" or an
                                        array that matches the dimension of the
                                        simulation grid (for non-stationary
                                        covariance model)
                                e.g.
                                    (t, d) = ('power', {w:2.0, r:1.5, s:1.7})
    :param nx, ny, nz    :  (ints) number of simulation grid (SG) cells in each direction
    :param sx, sy, sz    :  (floats) cell size in each direction
    :param ox, oy, oz    :  (floats) origin of the SG (bottom-lower-left corner)
    :return (covModelElem_c, flag):
                            covModelElem_c: (MPDS_COVMODELELEM *) covariance model elem. converted (C struct)
                            flag: (bool) indicating if the conversion has been done correctly (True) or not (False)
    """

    covModelElem_c = geosclassic.malloc_MPDS_COVMODELELEM()
    geosclassic.MPDSGeosClassicInitCovModelElem(covModelElem_c)

    if covModelElem_py[0] == 'nugget':
        # type
        covModelElem_c.covModelType = geosclassic.COV_NUGGET
        # weight
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
    elif covModelElem_py[0] == 'spherical':
        # type
        covModelElem_c.covModelType = geosclassic.COV_SPHERICAL
        # weight
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
    elif covModelElem_py[0] == 'exponential':
        # type
        covModelElem_c.covModelType = geosclassic.COV_EXPONENTIAL
        # weight
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
    elif covModelElem_py[0] == 'gaussian':
        # type
        covModelElem_c.covModelType = geosclassic.COV_GAUSSIAN
        # weight
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
    elif covModelElem_py[0] == 'cubic':
        # type
        covModelElem_c.covModelType = geosclassic.COV_CUBIC
        # weight
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
    elif covModelElem_py[0] == 'power':
        # type
        covModelElem_c.covModelType = geosclassic.COV_POWER
        # weight
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
        # other parameters
        # ... range s
        param = covModelElem_py[1]['s']
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
    Simulation grid geometry is specified in case of non-stationary covariance model.

    :param covModelElem_py: (2-tuple) elementary covariance model 2D in python:
                                (t, d) corresponds to an elementary model with:
                                    t: (string) the type, could be
                                        'nugget', 'spherical', 'exponential',
                                        'gaussian', 'cubic', 'power'
                                    d: (dict) dictionary of required parameters
                                        to be passed to the elementary model
                                        (value can be a "singe value" or an
                                        array that matches the dimension of the
                                        simulation grid (for non-stationary
                                        covariance model)
                                e.g.
                                    (t, d) = ('gaussian', {'w':10., 'r':[150, 50]})
    :param nx, ny, nz    :  (ints) number of simulation grid (SG) cells in each direction
    :param sx, sy, sz    :  (floats) cell size in each direction
    :param ox, oy, oz    :  (floats) origin of the SG (bottom-lower-left corner)
    :return (covModelElem_c, flag):
                            covModelElem_c: (MPDS_COVMODELELEM *) covariance model elem. converted (C struct)
                            flag: (bool) indicating if the conversion has been done correctly (True) or not (False)
    """

    covModelElem_c = geosclassic.malloc_MPDS_COVMODELELEM()
    geosclassic.MPDSGeosClassicInitCovModelElem(covModelElem_c)

    if covModelElem_py[0] == 'nugget':
        # type
        covModelElem_c.covModelType = geosclassic.COV_NUGGET
        # weight
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
    elif covModelElem_py[0] == 'spherical':
        # type
        covModelElem_c.covModelType = geosclassic.COV_SPHERICAL
        # weight
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
    elif covModelElem_py[0] == 'exponential':
        # type
        covModelElem_c.covModelType = geosclassic.COV_EXPONENTIAL
        # weight
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
    elif covModelElem_py[0] == 'gaussian':
        # type
        covModelElem_c.covModelType = geosclassic.COV_GAUSSIAN
        # weight
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
    elif covModelElem_py[0] == 'cubic':
        # type
        covModelElem_c.covModelType = geosclassic.COV_CUBIC
        # weight
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
    elif covModelElem_py[0] == 'power':
        # type
        covModelElem_c.covModelType = geosclassic.COV_POWER
        # weight
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
        # other parameters
        # ... range s
        param = covModelElem_py[1]['s']
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

    :param covModelElem_py: (2-tuple) elementary covariance model 3D in python:
                                (t, d) corresponds to an elementary model with:
                                    t: (string) the type, could be
                                        'nugget', 'spherical', 'exponential',
                                        'gaussian', 'cubic', 'power'
                                    d: (dict) dictionary of required parameters
                                        to be passed to the elementary model
                                        (value can be a "singe value" or an
                                        array that matches the dimension of the
                                        simulation grid (for non-stationary
                                        covariance model)
                                e.g.
                                    (t, d) = ('power', {w:2.0, r:[1.5, 2.5, 3.0], s:1.7})
    :param nx, ny, nz    :  (ints) number of simulation grid (SG) cells in each direction
    :param sx, sy, sz    :  (floats) cell size in each direction
    :param ox, oy, oz    :  (floats) origin of the SG (bottom-lower-left corner)
    :return (covModelElem_c, flag):
                            covModelElem_c: (MPDS_COVMODELELEM *) covariance model elem. converted (C struct)
                            flag: (bool) indicating if the conversion has been done correctly (True) or not (False)
    """

    covModelElem_c = geosclassic.malloc_MPDS_COVMODELELEM()
    geosclassic.MPDSGeosClassicInitCovModelElem(covModelElem_c)

    if covModelElem_py[0] == 'nugget':
        # type
        covModelElem_c.covModelType = geosclassic.COV_NUGGET
        # weight
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
    elif covModelElem_py[0] == 'spherical':
        # type
        covModelElem_c.covModelType = geosclassic.COV_SPHERICAL
        # weight
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
    elif covModelElem_py[0] == 'exponential':
        # type
        covModelElem_c.covModelType = geosclassic.COV_EXPONENTIAL
        # weight
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
    elif covModelElem_py[0] == 'gaussian':
        # type
        covModelElem_c.covModelType = geosclassic.COV_GAUSSIAN
        # weight
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
    elif covModelElem_py[0] == 'cubic':
        # type
        covModelElem_c.covModelType = geosclassic.COV_CUBIC
        # weight
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
    elif covModelElem_py[0] == 'power':
        # type
        covModelElem_c.covModelType = geosclassic.COV_POWER
        # weight
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
        # other parameters
        # ... range s
        param = covModelElem_py[1]['s']
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
    Simulation grid geometry is specified in case of non-stationary covariance model.

    :param covModel_py: (CovModel1D class) covariance model 1D (python class)
    :param nx, ny, nz : (ints) number of simulation grid (SG) cells in each direction
    :param sx, sy, sz : (floats) cell size in each direction
    :param ox, oy, oz : (floats) origin of the SG (bottom-lower-left corner)
    :return (covModel_c, flag):
                            covModel_c: (MPDS_COVMODEL *) covariance model converted (C struct)
                            flag: (bool) indicating if the conversion has been done correctly (True) or not (False)
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
        else:
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
    Simulation grid geometry is specified in case of non-stationary covariance model.

    :param covModel_py: (CovModel2D class) covariance model 2D (python class)
    :param nx, ny, nz : (ints) number of simulation grid (SG) cells in each direction
    :param sx, sy, sz : (floats) cell size in each direction
    :param ox, oy, oz : (floats) origin of the SG (bottom-lower-left corner)
    :return (covModel_c, flag):
                            covModel_c: (MPDS_COVMODEL *) covariance model converted (C struct)
                            flag: (bool) indicating if the conversion has been done correctly (True) or not (False)
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
        else:
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
    Simulation grid geometry is specified in case of non-stationary covariance model.

    :param covModel_py: (CovModel3D class) covariance model 3D (python class)
    :param nx, ny, nz : (ints) number of simulation grid (SG) cells in each direction
    :param sx, sy, sz : (floats) cell size in each direction
    :param ox, oy, oz : (floats) origin of the SG (bottom-lower-left corner)
    :return (covModel_c, flag):
                            covModel_c: (MPDS_COVMODEL *) covariance model converted (C struct)
                            flag: (bool) indicating if the conversion has been done correctly (True) or not (False)
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
        else:
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

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img (class)) output image, with image.nv variables (simulation or
                    estimates and standard deviation)
                    (image is None if mpds_geosClassicOutput->outputImage is NULL)
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
        tmp = np.zeros(mpds_progressMonitor.nwarningNumber, dtype='int32') # 'int32' for C-compatibility
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
        nGibbsSamplerPath,
        seed,
        nreal):
    """
    Fills a mpds_geosClassicInput C structure from given parameters.

    :return (mpds_geosClassicInput, flag):
                    mpds_geosClassicInput: C structure for "GeosClassicSim" program (C)
                    flag: (bool) indicating if the filling has been done correctly (True) or not (False)
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
    geosclassic.mpds_set_geosClassicInput_varname(mpds_geosClassicInput, varname)

    # mpds_geosClassicInput.outputMode
    mpds_geosClassicInput.outputMode = geosclassic.GEOS_CLASSIC_OUTPUT_NO_FILE

    # mpds_geosClassicInput.outputReportFlag and mpds_geosClassicInput.outputReportFileName
    if outputReportFile is not None:
        mpds_geosClassicInput.outputReportFlag = geosclassic.TRUE
        geosclassic.mpds_set_geosClassicInput_outputReportFileName(mpds_geosClassicInput, outputReportFile)
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
            geosclassic.MPDS_IMAGE_array_setitem(mpds_geosClassicInput.dataImage, i, img_py2C(dataIm))

    # mpds_geosClassicInput.ndataPointSet and mpds_geosClassicInput.dataPointSet
    if dataPointSet is None:
        mpds_geosClassicInput.ndataPointSet = 0
    else:
        dataPointSet = np.asarray(dataPointSet).reshape(-1)
        n = len(dataPointSet)
        mpds_geosClassicInput.ndataPointSet = n
        mpds_geosClassicInput.dataPointSet = geosclassic.new_MPDS_POINTSET_array(n)
        for i, dataPS in enumerate(dataPointSet):
            geosclassic.MPDS_POINTSET_array_setitem(mpds_geosClassicInput.dataPointSet, i, ps_py2C(dataPS))

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
        print("ERROR: can not integrate 'mean' (not compatible with simulation grid)")
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
        print("ERROR: can not integrate 'var' (not compatible with simulation grid)")
        return mpds_geosClassicInput, False

    # mpds_geosClassicInput.nGibbsSamplerPath
    mpds_geosClassicInput.nGibbsSamplerPath = int(nGibbsSamplerPath)

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
        nGibbsSamplerPath=50,
        seed=None,
        outputReportFile=None,
        nthreads=-1, verbose=2):
    """
    Generates 1D simulations (Sequential Gaussian Simulation, SGS) based on
    simple or ordinary kriging.

    :param cov_model:   (CovModel1D class) covariance model in 1D, see
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

    :param mean:        (None or float or ndarray) mean of the simulation:
                            - None   : mean of hard data values (stationary),
                                       (0 if no hard data)
                            - float  : for stationary mean (set manually)
                            - ndarray: for non stationary mean, must contain
                                as many entries as number of grid cells
                                (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param var:         (None or float or ndarray) variance of the simulation
                            (for simple kriging only):
                                - None   : variance not modified
                                           (only covariance model is used)
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
                            at conditioning points for inequality data minimal bound
                            (same type as xIneqMin)

    :param xIneqMax:    (1-dimensional array or float or None) coordinate of
                            conditioning points for inequality data maximal bound
    :param vIneqMax:    (1-dimensional array or float or None) value
                            at conditioning points for inequality data maximal bound
                            (same type as xIneqMax)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not simulated
                            cell (nunber of entries should be equal to the number of
                            grid cells)

    :param searchRadiusRelative:
                        (float) indicating how restricting the search ellipsoid (should be positive):
                            let r_i be the ranges of the covariance model along its main axes,
                            if x is a node to be simulated, a node y is taken into account iff it is
                            within the ellipsoid centered at x of half-axes searchRadiusRelative * r_i
                            Note:
                                - if a range is a variable parameter, its maximal value over the simulation grid
                                  is considered

    :param nneighborMax:(int) maximum number of nodes retrieved from the search ellipsoid,
                            set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (int) indicating how to sort the search neighboorhood nodes
                            (neighbors), they are sorted in increasing order according to:
                                - searchNeighborhoodSortMode = 0:
                                  distance in the usual axes system
                                - searchNeighborhoodSortMode = 1:
                                  distance in the axes sytem supporting the covariance model
                                  and accounting for anisotropy given by the ranges
                                - searchNeighborhoodSortMode = 2:
                                  minus the evaluation of the covariance model
                            Note:
                                - if the covariance model has any variable parameter (non-stationary),
                                  then searchNeighborhoodSortMode = 2 is not allowed
                                - if the covariance model has any range or angle set as a variable parameter,
                                  then searchNeighborhoodSortMode must be set to 0
                                - greatest possible value as default

    :param nGibbsSamplerPath:
                        (int) number of Gibbs sampler paths to deal with inequality data
                            the conditioning locations with inequality data are first simulated
                            (with truncated gaussian distribution) sequentially; then, these
                            locations are re-simulated following a new path as many times as
                            desired; this parameter (nGibbsSamplerPath) is the total number
                            of path(s)

    :param seed:        (int or None) initial seed, if None an initial seed between
                            1 and 999999 is generated with numpy.random.randint

    :param outputReportFile:
                    (string or None) name of the report file, if None: no report file

    :param nthreads:
                (int) number of thread(s) to use for "GeosClassicSim" program (C),
                    (nthreads = -n <= 0: for maximal number of threads except n,
                    but at least 1)
    :param verbose:
                (int) indicates what is displayed during the GeosClassicSim run:
                    - 0: nothing
                    - 1: warning only
                    - 2 (or >1): warning and progress

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img (class)) output image, with image.nv=nreal variables (each
                    variable is one realization)
                    (image is None if mpds_geosClassicOutput->outputImage is NULL)
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
        print("ERROR (SIMUL_1D): 'cov_model' (first argument) is not valid")
        return None

    for el in cov_model.elem:
        # weight
        w = el[1]['w']
        if np.size(w) != 1 and np.size(w) != nxyz:
            print("ERROR (SIMUL_1D): 'cov_model': weight ('w') not compatible with simulation grid")
            return None
        # ranges
        if 'r' in el[1].keys():
            r  = el[1]['r']
            if np.size(r) != 1 and np.size(r) != nxyz:
                print("ERROR (SIMUL_1D): 'cov_model': range ('r') not compatible with simulation grid")
                return None
        # additional parameter (s)
        if 's' in el[1].keys():
            s  = el[1]['s']
            if np.size(s) != 1 and np.size(s) != nxyz:
                print("ERROR (SIMUL_1D): 'cov_model': parameter ('s') not compatible with simulation grid")
                return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     print("ERROR (SIMUL_1D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 3
    elif method == 'ordinary_kriging':
        computationMode = 2
    else:
        print("ERROR (SIMUL_1D): 'method' is not valid")
        return None

    # data points: x, v, xIneqMin, vIneqMin, xIneqMax, vIneqMax
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            print("(ERROR (SIMUL_1D): length of 'v' is not valid")
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
            print("(ERROR (SIMUL_1D): length of 'vIneqMin' is not valid")
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
            print("(ERROR (SIMUL_1D): length of 'vIneqMax' is not valid")
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
            print("ERROR (SIMUL_1D): 'mask' is not valid")
            return None

    # Check parameters - searchRadiusRelative
    if searchRadiusRelative < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
        print("ERROR (SIMUL_1D): 'searchRadiusRelative' too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
        return None

    # Check parameters - nneighborMax
    if nneighborMax != -1 and nneighborMax <= 0:
        print("ERROR (SIMUL_1D): 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
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
                print("ERROR (SIMUL_1D): 'searchNeighborhoodSortMode=2' not allowed with non-stationary covariance model")
                return None
        elif searchNeighborhoodSortMode == 1:
            if not cov_model.is_orientation_stationary() or not cov_model.is_range_stationary():
                print("ERROR (SIMUL_1D): 'searchNeighborhoodSortMode=1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                return None

    # Check parameters - mean
    if mean is not None:
        # if method == 'ordinary_kriging':
        #     print("ERROR (SIMUL_1D): specifying 'mean' not allowed with ordinary kriging")
        #     return None
        mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if mean.size not in (1, nxyz):
            print("ERROR (SIMUL_1D): size of 'mean' is not valid")
            return None

    # Check parameters - var
    if var is not None:
        if method == 'ordinary_kriging':
            print("ERROR (SIMUL_1D): specifying 'var' not allowed with ordinary kriging")
            return None
        var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if var.size not in (1, nxyz):
            print("ERROR (SIMUL_1D): size of 'var' is not valid")
            return None

    # Check parameters - nreal
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose >= 2:
            print('SIMUL_1D: nreal <= 0: nothing to do!')
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
        nGibbsSamplerPath,
        seed,
        nreal)

    if not flag:
        print("ERROR (SIMUL_1D): can not fill input structure!")
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
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr: warning only
    if verbose == 0:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    elif verbose == 1:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim"
    # err = geosclassic.MPDSGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth )

    # Free memory on C side: mpds_geosClassicInput
    geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
    #geosclassic.MPDSFree(mpds_geosClassicInput)
    geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)

    if err:
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree (mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 1 and geosclassic_output and geosclassic_output['nwarning']:
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
        nGibbsSamplerPath=50,
        seed=None,
        outputReportFile=None,
        nthreads=-1, verbose=2):
    """
    Generates 2D simulations (Sequential Gaussian Simulation, SGS) based on
    simple or ordinary kriging.

    :param cov_model:   (CovModel2D class) covariance model in 2D, see
                            definition of the class in module geone.covModel

    :param dimension:   (sequence of 2 ints) (nx, ny), number of cells
                            in x-, y-axis direction
    :param spacing:     (sequence of 2 floats) (sx, sy), spacing between
                            two adjacent cells in x-, y-axis direction
    :param origin:      (sequence of 2 floats) (ox, oy), origin of the 2D simulation
                            - used for localizing the conditioning points
    :param method:      (string) indicates the method used:
                            - 'simple_kriging':
                                simulation based on simple kriging
                            - 'ordinary_kriging':
                                simulation based on ordinary kriging
    :param nreal:       (int) number of realizations

    :param mean:        (None or float or ndarray) mean of the simulation:
                            - None   : mean of hard data values (stationary),
                                       (0 if no hard data)
                            - float  : for stationary mean (set manually)
                            - ndarray: for non stationary mean, must contain
                                as many entries as number of grid cells
                                (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param var:         (None or float or ndarray) variance of the simulation
                            (for simple kriging only):
                                - None   : variance not modified
                                           (only covariance model is used)
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
                            at conditioning points for inequality data minimal bound
                            (same type as xIneqMin)

    :param xIneqMax:    (2-dimensional array of dim n x 2, or
                            1-dimensional array of dim 2 or None) coordinate of
                            conditioning points for inequality data maximal bound
    :param vIneqMax:    (1-dimensional array or float or None) value
                            at conditioning points for inequality data maximal bound
                            (same type as xIneqMax)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not simulated
                            cell (nunber of entries should be equal to the number of
                            grid cells)

    :param searchRadiusRelative:
                        (float) indicating how restricting the search ellipsoid (should be positive):
                            let r_i be the ranges of the covariance model along its main axes,
                            if x is a node to be simulated, a node y is taken into account iff it is
                            within the ellipsoid centered at x of half-axes searchRadiusRelative * r_i
                            Note:
                                - if a range is a variable parameter, its maximal value over the simulation grid
                                  is considered
                                - if orientation of the covariance model is non-stationary, a "circular search neighborhood"
                                  is considered with the radius set to the maximum of all ranges

    :param nneighborMax:(int) maximum number of nodes retrieved from the search ellipsoid,
                            set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (int) indicating how to sort the search neighboorhood nodes
                            (neighbors), they are sorted in increasing order according to:
                                - searchNeighborhoodSortMode = 0:
                                  distance in the usual axes system
                                - searchNeighborhoodSortMode = 1:
                                  distance in the axes sytem supporting the covariance model
                                  and accounting for anisotropy given by the ranges
                                - searchNeighborhoodSortMode = 2:
                                  minus the evaluation of the covariance model
                            Note:
                                - if the covariance model has any variable parameter (non-stationary),
                                  then searchNeighborhoodSortMode = 2 is not allowed
                                - if the covariance model has any range or angle set as a variable parameter,
                                  then searchNeighborhoodSortMode must be set to 0
                                - greatest possible value as default

    :param nGibbsSamplerPath:
                        (int) number of Gibbs sampler paths to deal with inequality data
                            the conditioning locations with inequality data are first simulated
                            (with truncated gaussian distribution) sequentially; then, these
                            locations are re-simulated following a new path as many times as
                            desired; this parameter (nGibbsSamplerPath) is the total number
                            of path(s)

    :param seed:        (int or None) initial seed, if None an initial seed between
                            1 and 999999 is generated with numpy.random.randint

    :param outputReportFile:
                    (string or None) name of the report file, if None: no report file

    :param nthreads:
                (int) number of thread(s) to use for "GeosClassicSim" program (C),
                    (nthreads = -n <= 0: for maximal number of threads except n,
                    but at least 1)
    :param verbose:
                (int) indicates what is displayed during the GeosClassicSim run:
                    - 0: nothing
                    - 1: warning only
                    - 2 (or >1): warning and progress

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img (class)) output image, with image.nv=nreal variables (each
                    variable is one realization)
                    (image is None if mpds_geosClassicOutput->outputImage is NULL)
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
    if not isinstance(cov_model, gcm.CovModel2D):
        print("ERROR (SIMUL_2D): 'cov_model' (first argument) is not valid")
        return None

    for el in cov_model.elem:
        # weight
        w = el[1]['w']
        if np.size(w) != 1 and np.size(w) != nxyz:
            print("ERROR (SIMUL_2D): 'cov_model': weight ('w') not compatible with simulation grid")
            return None
        # ranges
        if 'r' in el[1].keys():
            for r in el[1]['r']:
                if np.size(r) != 1 and np.size(r) != nxyz:
                    print("ERROR (SIMUL_2D): 'cov_model': range ('r') not compatible with simulation grid")
                    return None
        # additional parameter (s)
        if 's' in el[1].keys():
            s  = el[1]['s']
            if np.size(s) != 1 and np.size(s) != nxyz:
                print("ERROR (SIMUL_2D): 'cov_model': parameter ('s') not compatible with simulation grid")
                return None

    # alpha
    angle = cov_model.alpha
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        print("ERROR (SIMUL_2D): 'cov_model': angle (alpha) not compatible with simulation grid")
        return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     print("ERROR (SIMUL_2D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 3
    elif method == 'ordinary_kriging':
        computationMode = 2
    else:
        print("ERROR (SIMUL_2D): 'method' is not valid")
        return None

    # data points: x, v, xIneqMin, vIneqMin, xIneqMax, vIneqMax
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 2) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            print("(ERROR (SIMUL_2D): length of 'v' is not valid")
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
            print("(ERROR (SIMUL_2D): length of 'vIneqMin' is not valid")
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
            print("(ERROR (SIMUL_2D): length of 'vIneqMax' is not valid")
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
            print("ERROR (SIMUL_2D): 'mask' is not valid")
            return None

    # Check parameters - searchRadiusRelative
    if searchRadiusRelative < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
        print("ERROR (SIMUL_2D): 'searchRadiusRelative' too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
        return None

    # Check parameters - nneighborMax
    if nneighborMax != -1 and nneighborMax <= 0:
        print("ERROR (SIMUL_2D): 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
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
                print("ERROR (SIMUL_2D): 'searchNeighborhoodSortMode=2' not allowed with non-stationary covariance model")
                return None
        elif searchNeighborhoodSortMode == 1:
            if not cov_model.is_orientation_stationary() or not cov_model.is_range_stationary():
                print("ERROR (SIMUL_2D): 'searchNeighborhoodSortMode=1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                return None

    # Check parameters - mean
    if mean is not None:
        # if method == 'ordinary_kriging':
        #     print("ERROR (SIMUL_2D): specifying 'mean' not allowed with ordinary kriging")
        #     return None
        mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if mean.size not in (1, nxyz):
            print("ERROR (SIMUL_2D): size of 'mean' is not valid")
            return None

    # Check parameters - var
    if var is not None:
        if method == 'ordinary_kriging':
            print("ERROR (SIMUL_2D): specifying 'var' not allowed with ordinary kriging")
            return None
        var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if var.size not in (1, nxyz):
            print("ERROR (SIMUL_2D): size of 'var' is not valid")
            return None

    # Check parameters - nreal
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose >= 2:
            print('SIMUL_2D: nreal <= 0: nothing to do!')
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
        nGibbsSamplerPath,
        seed,
        nreal)

    if not flag:
        print("ERROR (SIMUL_2D): can not fill input structure!")
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
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr: warning only
    if verbose == 0:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    elif verbose == 1:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim"
    # err = geosclassic.MPDSGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth )

    # Free memory on C side: mpds_geosClassicInput
    geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
    #geosclassic.MPDSFree(mpds_geosClassicInput)
    geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)

    if err:
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree (mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 1 and geosclassic_output and geosclassic_output['nwarning']:
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
        nGibbsSamplerPath=50,
        seed=None,
        outputReportFile=None,
        nthreads=-1, verbose=2):
    """
    Generates 3D simulations (Sequential Gaussian Simulation, SGS) based on
    simple or ordinary kriging.

    :param cov_model:   (CovModel3D class) covariance model in 3D, see
                            definition of the class in module geone.covModel

    :param dimension:   (sequence of 3 ints) (nx, ny, nz), number of cells
                            in x-, y-, z-axis direction
    :param spacing:     (sequence of 3 floats) (sx, sy, sz), spacing between
                            two adjacent cells in x-, y-, z-axis direction
    :param origin:      (sequence of 3 floats) (ox, oy, oy), origin of the 3D simulation
                            - used for localizing the conditioning points
    :param method:      (string) indicates the method used:
                            - 'simple_kriging':
                                simulation based on simple kriging
                            - 'ordinary_kriging':
                                simulation based on ordinary kriging
    :param nreal:       (int) number of realizations

    :param mean:        (None or float or ndarray) mean of the simulation:
                            - None   : mean of hard data values (stationary),
                                       (0 if no hard data)
                            - float  : for stationary mean (set manually)
                            - ndarray: for non stationary mean, must contain
                                as many entries as number of grid cells
                                (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param var:         (None or float or ndarray) variance of the simulation
                            (for simple kriging only):
                                - None   : variance not modified
                                           (only covariance model is used)
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
                            at conditioning points for inequality data minimal bound
                            (same type as xIneqMin)

    :param xIneqMax:    (2-dimensional array of dim n x 3, or
                            1-dimensional array of dim 3 or None) coordinate of
                            conditioning points for inequality data maximal bound
    :param vIneqMax:    (1-dimensional array or float or None) value
                            at conditioning points for inequality data maximal bound
                            (same type as xIneqMax)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not simulated
                            cell (nunber of entries should be equal to the number of
                            grid cells)

    :param searchRadiusRelative:
                        (float) indicating how restricting the search ellipsoid (should be positive):
                            let r_i be the ranges of the covariance model along its main axes,
                            if x is a node to be simulated, a node y is taken into account iff it is
                            within the ellipsoid centered at x of half-axes searchRadiusRelative * r_i
                            Note:
                                - if a range is a variable parameter, its maximal value over the simulation grid
                                  is considered
                                - if orientation of the covariance model is non-stationary, a "circular search neighborhood"
                                  is considered with the radius set to the maximum of all ranges

    :param nneighborMax:(int) maximum number of nodes retrieved from the search ellipsoid,
                            set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (int) indicating how to sort the search neighboorhood nodes
                            (neighbors), they are sorted in increasing order according to:
                                - searchNeighborhoodSortMode = 0:
                                  distance in the usual axes system
                                - searchNeighborhoodSortMode = 1:
                                  distance in the axes sytem supporting the covariance model
                                  and accounting for anisotropy given by the ranges
                                - searchNeighborhoodSortMode = 2:
                                  minus the evaluation of the covariance model
                            Note:
                                - if the covariance model has any variable parameter (non-stationary),
                                  then searchNeighborhoodSortMode = 2 is not allowed
                                - if the covariance model has any range or angle set as a variable parameter,
                                  then searchNeighborhoodSortMode must be set to 0
                                - greatest possible value as default

    :param nGibbsSamplerPath:
                        (int) number of Gibbs sampler paths to deal with inequality data
                            the conditioning locations with inequality data are first simulated
                            (with truncated gaussian distribution) sequentially; then, these
                            locations are re-simulated following a new path as many times as
                            desired; this parameter (nGibbsSamplerPath) is the total number
                            of path(s)

    :param seed:        (int or None) initial seed, if None an initial seed between
                            1 and 999999 is generated with numpy.random.randint

    :param outputReportFile:
                    (string or None) name of the report file, if None: no report file

    :param nthreads:
                (int) number of thread(s) to use for "GeosClassicSim" program (C),
                    (nthreads = -n <= 0: for maximal number of threads except n,
                    but at least 1)
    :param verbose:
                (int) indicates what is displayed during the GeosClassicSim run:
                    - 0: nothing
                    - 1: warning only
                    - 2 (or >1): warning and progress

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img (class)) output image, with image.nv=nreal variables (each
                    variable is one realization)
                    (image is None if mpds_geosClassicOutput->outputImage is NULL)
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
    if not isinstance(cov_model, gcm.CovModel3D):
        print("ERROR (SIMUL_3D): 'cov_model' (first argument) is not valid")
        return None

    for el in cov_model.elem:
        # weight
        w = el[1]['w']
        if np.size(w) != 1 and np.size(w) != nxyz:
            print("ERROR (SIMUL_3D): 'cov_model': weight ('w') not compatible with simulation grid")
            return None
        # ranges
        if 'r' in el[1].keys():
            for r in el[1]['r']:
                if np.size(r) != 1 and np.size(r) != nxyz:
                    print("ERROR (SIMUL_3D): 'cov_model': range ('r') not compatible with simulation grid")
                    return None
        # additional parameter (s)
        if 's' in el[1].keys():
            s  = el[1]['s']
            if np.size(s) != 1 and np.size(s) != nxyz:
                print("ERROR (SIMUL_3D): 'cov_model': parameter ('s') not compatible with simulation grid")
                return None

    # alpha
    angle = cov_model.alpha
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        print("ERROR (SIMUL_3D): 'cov_model': angle (alpha) not compatible with simulation grid")
        return None

    # beta
    angle = cov_model.beta
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        print("ERROR (SIMUL_3D): 'cov_model': angle (beta) not compatible with simulation grid")
        return None

    # gamma
    angle = cov_model.gamma
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        print("ERROR (SIMUL_3D): 'cov_model': angle (gamma) not compatible with simulation grid")
        return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     print("ERROR (SIMUL_3D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 3
    elif method == 'ordinary_kriging':
        computationMode = 2
    else:
        print("ERROR (SIMUL_3D): 'method' is not valid")
        return None

    # data points: x, v, xIneqMin, vIneqMin, xIneqMax, vIneqMax
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 3) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            print("(ERROR (SIMUL_3D): length of 'v' is not valid")
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
            print("(ERROR (SIMUL_3D): length of 'vIneqMin' is not valid")
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
            print("(ERROR (SIMUL_3D): length of 'vIneqMax' is not valid")
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
            print("ERROR (SIMUL_3D): 'mask' is not valid")
            return None

    # Check parameters - searchRadiusRelative
    if searchRadiusRelative < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
        print("ERROR (SIMUL_3D): 'searchRadiusRelative' too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
        return None

    # Check parameters - nneighborMax
    if nneighborMax != -1 and nneighborMax <= 0:
        print("ERROR (SIMUL_3D): 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
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
                print("ERROR (SIMUL_3D): 'searchNeighborhoodSortMode=2' not allowed with non-stationary covariance model")
                return None
        elif searchNeighborhoodSortMode == 1:
            if not cov_model.is_orientation_stationary() or not cov_model.is_range_stationary():
                print("ERROR (SIMUL_3D): 'searchNeighborhoodSortMode=1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                return None

    # Check parameters - mean
    if mean is not None:
        # if method == 'ordinary_kriging':
        #     print("ERROR (SIMUL_3D): specifying 'mean' not allowed with ordinary kriging")
        #     return None
        mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if mean.size not in (1, nxyz):
            print("ERROR (SIMUL_3D): size of 'mean' is not valid")
            return None

    # Check parameters - var
    if var is not None:
        if method == 'ordinary_kriging':
            print("ERROR (SIMUL_3D): specifying 'var' not allowed with ordinary kriging")
            return None
        var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if var.size not in (1, nxyz):
            print("ERROR (SIMUL_3D): size of 'var' is not valid")
            return None

    # Check parameters - nreal
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose >= 2:
            print('SIMUL_3D: nreal <= 0: nothing to do!')
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
        nGibbsSamplerPath,
        seed,
        nreal)

    if not flag:
        print("ERROR (SIMUL_3D): can not fill input structure!")
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
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr: warning only
    if verbose == 0:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    elif verbose == 1:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim"
    # err = geosclassic.MPDSGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth )

    # Free memory on C side: mpds_geosClassicInput
    geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
    #geosclassic.MPDSFree(mpds_geosClassicInput)
    geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)

    if err:
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree (mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 1 and geosclassic_output and geosclassic_output['nwarning']:
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
        nthreads=-1, verbose=2):
    """
    Computes estimate and standard deviation for 1D grid of simple or ordinary kriging.

    :param cov_model:   (CovModel1D class) covariance model in 1D, see
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

    :param mean:        (None or float or ndarray) mean of the simulation:
                            - None   : mean of hard data values (stationary),
                                       (0 if no hard data)
                            - float  : for stationary mean (set manually)
                            - ndarray: for non stationary mean, must contain
                                as many entries as number of grid cells
                                (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param var:         (None or float or ndarray) variance of the simulation
                            (for simple kriging only):
                                - None   : variance not modified
                                           (only covariance model is used)
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
                            over the SG: 1 for simulated cell / 0 for not simulated
                            cell (nunber of entries should be equal to the number of
                            grid cells)

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
                        (float) indicating how restricting the search ellipsoid (should be positive):
                            let r_i be the ranges of the covariance model along its main axes,
                            if x is a node to be simulated, a node y is taken into account iff it is
                            within the ellipsoid centered at x of half-axes searchRadiusRelative * r_i
                            (unused if use_unique_neighborhood is True)
                            Note:
                                - if a range is a variable parameter, its maximal value over the simulation grid
                                  is considered

    :param nneighborMax:(int) maximum number of nodes retrieved from the search ellipsoid,
                            set -1 for unlimited
                            (unused if use_unique_neighborhood is True)

    :param searchNeighborhoodSortMode:
                        (int) indicating how to sort the search neighboorhood nodes
                            (neighbors), they are sorted in increasing order according to:
                                - searchNeighborhoodSortMode = 0:
                                  distance in the usual axes system
                                - searchNeighborhoodSortMode = 1:
                                  distance in the axes sytem supporting the covariance model
                                  and accounting for anisotropy given by the ranges
                                - searchNeighborhoodSortMode = 2:
                                  minus the evaluation of the covariance model
                            (unused if use_unique_neighborhood is True)
                            Note:
                                - if the covariance model has any variable parameter (non-stationary),
                                  then searchNeighborhoodSortMode = 2 is not allowed
                                - if the covariance model has any range or angle set as a variable parameter,
                                  then searchNeighborhoodSortMode must be set to 0
                                - greatest possible value as default

    :param outputReportFile:
                    (string or None) name of the report file, if None: no report file

    :param nthreads:
                (int) number of thread(s) to use for "GeosClassicSim" program (C),
                    (nthreads = -n <= 0: for maximal number of threads except n,
                    but at least 1)
    :param verbose:
                (int) indicates what is displayed during the GeosClassicSim run:
                    - 0: nothing
                    - 1: warning only
                    - 2 (or >1): warning and progress

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img (class)) output image, with image.nv=2 variables (estimate and
                    standard deviation)
                    (image is None if mpds_geosClassicOutput->outputImage is NULL)
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
        print("ERROR (ESTIM_1D): 'cov_model' (first argument) is not valid")
        return None

    for el in cov_model.elem:
        # weight
        w = el[1]['w']
        if np.size(w) != 1 and np.size(w) != nxyz:
            print("ERROR (ESTIM_1D): 'cov_model': weight ('w') not compatible with simulation grid")
            return None
        # ranges
        if 'r' in el[1].keys():
            r  = el[1]['r']
            if np.size(r) != 1 and np.size(r) != nxyz:
                print("ERROR (ESTIM_1D): 'cov_model': range ('r') not compatible with simulation grid")
                return None
        # additional parameter (s)
        if 's' in el[1].keys():
            s  = el[1]['s']
            if np.size(s) != 1 and np.size(s) != nxyz:
                print("ERROR (ESTIM_1D): 'cov_model': parameter ('s') not compatible with simulation grid")
                return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     print("ERROR (ESTIM_1D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 1
    elif method == 'ordinary_kriging':
        computationMode = 0
    else:
        print("ERROR (ESTIM_1D): 'method' is not valid")
        return None

    # data points: x, v
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            print("(ERROR (ESTIM_1D): length of 'v' is not valid")
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
            print("ERROR (ESTIM_1D): 'mask' is not valid")
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
           print("ERROR (ESTIM_1D): 'searchRadiusRelative' too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
           return None

       # Check parameters - nneighborMax
       if nneighborMax != -1 and nneighborMax <= 0:
           print("ERROR (ESTIM_1D): 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
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
                   print("ERROR (ESTIM_1D): 'searchNeighborhoodSortMode=2' not allowed with non-stationary covariance model")
                   return None
           elif searchNeighborhoodSortMode == 1:
               if not cov_model.is_orientation_stationary() or not cov_model.is_range_stationary():
                   print("ERROR (ESTIM_1D): 'searchNeighborhoodSortMode=1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                   return None

    # Check parameters - mean
    if mean is not None:
        # if method == 'ordinary_kriging':
        #     print("ERROR (ESTIM_1D): specifying 'mean' not allowed with ordinary kriging")
        #     return None
        mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if mean.size not in (1, nxyz):
            print("ERROR (ESTIM_1D): size of 'mean' is not valid")
            return None

    # Check parameters - var
    if var is not None:
        if method == 'ordinary_kriging':
            print("ERROR (ESTIM_1D): specifying 'var' not allowed with ordinary kriging")
            return None
        var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if var.size not in (1, nxyz):
            print("ERROR (ESTIM_1D): size of 'var' is not valid")
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
        0)

    if not flag:
        print("ERROR (ESTIM_1D): can not fill input structure!")
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
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr: warning only
    if verbose == 0:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    elif verbose == 1:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim"
    # err = geosclassic.MPDSGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth )

    # Free memory on C side: mpds_geosClassicInput
    geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
    #geosclassic.MPDSFree(mpds_geosClassicInput)
    geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)

    if err:
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree (mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 1 and geosclassic_output and geosclassic_output['nwarning']:
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
        nthreads=-1, verbose=2):
    """
    Computes estimate and standard deviation for 2D grid of simple or ordinary kriging.

    :param cov_model:   (CovModel2D class) covariance model in 2D, see
                            definition of the class in module geone.covModel

    :param dimension:   (sequence of 2 ints) (nx, ny), number of cells
                            in x-, y-axis direction
    :param spacing:     (sequence of 2 floats) (sx, sy), spacing between
                            two adjacent cells in x-, y-axis direction
    :param origin:      (sequence of 2 floats) (ox, oy), origin of the 2D simulation
                            - used for localizing the conditioning points
    :param method:      (string) indicates the method used:
                            - 'simple_kriging':
                                simulation based on simple kriging
                            - 'ordinary_kriging':
                                simulation based on ordinary kriging

    :param mean:        (None or float or ndarray) mean of the simulation:
                            - None   : mean of hard data values (stationary),
                                       (0 if no hard data)
                            - float  : for stationary mean (set manually)
                            - ndarray: for non stationary mean, must contain
                                as many entries as number of grid cells
                                (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param var:         (None or float or ndarray) variance of the simulation
                            (for simple kriging only):
                                - None   : variance not modified
                                           (only covariance model is used)
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
                            over the SG: 1 for simulated cell / 0 for not simulated
                            cell (nunber of entries should be equal to the number of
                            grid cells)

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
                        (float) indicating how restricting the search ellipsoid (should be positive):
                            let r_i be the ranges of the covariance model along its main axes,
                            if x is a node to be simulated, a node y is taken into account iff it is
                            within the ellipsoid centered at x of half-axes searchRadiusRelative * r_i
                            (unused if use_unique_neighborhood is True)
                            Note:
                                - if a range is a variable parameter, its maximal value over the simulation grid
                                  is considered
                                - if orientation of the covariance model is non-stationary, a "circular search neighborhood"
                                  is considered with the radius set to the maximum of all ranges

    :param nneighborMax:(int) maximum number of nodes retrieved from the search ellipsoid,
                            set -1 for unlimited
                            (unused if use_unique_neighborhood is True)

    :param searchNeighborhoodSortMode:
                        (int) indicating how to sort the search neighboorhood nodes
                            (neighbors), they are sorted in increasing order according to:
                                - searchNeighborhoodSortMode = 0:
                                  distance in the usual axes system
                                - searchNeighborhoodSortMode = 1:
                                  distance in the axes sytem supporting the covariance model
                                  and accounting for anisotropy given by the ranges
                                - searchNeighborhoodSortMode = 2:
                                  minus the evaluation of the covariance model
                            (unused if use_unique_neighborhood is True)
                            Note:
                                - if the covariance model has any variable parameter (non-stationary),
                                  then searchNeighborhoodSortMode = 2 is not allowed
                                - if the covariance model has any range or angle set as a variable parameter,
                                  then searchNeighborhoodSortMode must be set to 0
                                - greatest possible value as default

    :param outputReportFile:
                    (string or None) name of the report file, if None: no report file

    :param nthreads:
                (int) number of thread(s) to use for "GeosClassicSim" program (C),
                    (nthreads = -n <= 0: for maximal number of threads except n,
                    but at least 1)
    :param verbose:
                (int) indicates what is displayed during the GeosClassicSim run:
                    - 0: nothing
                    - 1: warning only
                    - 2 (or >1): warning and progress

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img (class)) output image, with image.nv=2 variables (estimate and
                    standard deviation)
                    (image is None if mpds_geosClassicOutput->outputImage is NULL)
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
    if not isinstance(cov_model, gcm.CovModel2D):
        print("ERROR (ESTIM_2D): 'cov_model' (first argument) is not valid")
        return None

    for el in cov_model.elem:
        # weight
        w = el[1]['w']
        if np.size(w) != 1 and np.size(w) != nxyz:
            print("ERROR (ESTIM_2D): 'cov_model': weight ('w') not compatible with simulation grid")
            return None
        # ranges
        if 'r' in el[1].keys():
            for r in el[1]['r']:
                if np.size(r) != 1 and np.size(r) != nxyz:
                    print("ERROR (ESTIM_2D): 'cov_model': range ('r') not compatible with simulation grid")
                    return None
        # additional parameter (s)
        if 's' in el[1].keys():
            s  = el[1]['s']
            if np.size(s) != 1 and np.size(s) != nxyz:
                print("ERROR (ESTIM_2D): 'cov_model': parameter ('s') not compatible with simulation grid")
                return None

    # alpha
    angle = cov_model.alpha
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        print("ERROR (ESTIM_2D): 'cov_model': angle (alpha) not compatible with simulation grid")
        return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     print("ERROR (ESTIM_2D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 1
    elif method == 'ordinary_kriging':
        computationMode = 0
    else:
        print("ERROR (ESTIM_2D): 'method' is not valid")
        return None

    # data points: x, v
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 2) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            print("(ERROR (ESTIM_2D): length of 'v' is not valid")
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
            print("ERROR (ESTIM_2D): 'mask' is not valid")
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
           print("ERROR (ESTIM_2D): 'searchRadiusRelative' too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
           return None

       # Check parameters - nneighborMax
       if nneighborMax != -1 and nneighborMax <= 0:
           print("ERROR (ESTIM_2D): 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
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
                   print("ERROR (ESTIM_2D): 'searchNeighborhoodSortMode=2' not allowed with non-stationary covariance model")
                   return None
           elif searchNeighborhoodSortMode == 1:
               if not cov_model.is_orientation_stationary() or not cov_model.is_range_stationary():
                   print("ERROR (ESTIM_2D): 'searchNeighborhoodSortMode=1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                   return None

    # Check parameters - mean
    if mean is not None:
        # if method == 'ordinary_kriging':
        #     print("ERROR (ESTIM_2D): specifying 'mean' not allowed with ordinary kriging")
        #     return None
        mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if mean.size not in (1, nxyz):
            print("ERROR (ESTIM_2D): size of 'mean' is not valid")
            return None

    # Check parameters - var
    if var is not None:
        if method == 'ordinary_kriging':
            print("ERROR (ESTIM_2D): specifying 'var' not allowed with ordinary kriging")
            return None
        var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if var.size not in (1, nxyz):
            print("ERROR (ESTIM_2D): size of 'var' is not valid")
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
        0)

    if not flag:
        print("ERROR (ESTIM_2D): can not fill input structure!")
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
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr: warning only
    if verbose == 0:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    elif verbose == 1:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim"
    # err = geosclassic.MPDSGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth )

    # Free memory on C side: mpds_geosClassicInput
    geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
    #geosclassic.MPDSFree(mpds_geosClassicInput)
    geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)

    if err:
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree (mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 1 and geosclassic_output and geosclassic_output['nwarning']:
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
        nthreads=-1, verbose=2):
    """
    Computes estimate and standard deviation for 3D grid of simple or ordinary kriging.

    :param cov_model:   (CovModel3D class) covariance model in 3D, see
                            definition of the class in module geone.covModel

    :param dimension:   (sequence of 3 ints) (nx, ny, nz), number of cells
                            in x-, y-, z-axis direction
    :param spacing:     (sequence of 3 floats) (sx, sy, sz), spacing between
                            two adjacent cells in x-, y-, z-axis direction
    :param origin:      (sequence of 3 floats) (ox, oy, oy), origin of the 3D simulation
                            - used for localizing the conditioning points
    :param method:      (string) indicates the method used:
                            - 'simple_kriging':
                                simulation based on simple kriging
                            - 'ordinary_kriging':
                                simulation based on ordinary kriging

    :param mean:        (None or float or ndarray) mean of the simulation:
                            - None   : mean of hard data values (stationary),
                                       (0 if no hard data)
                            - float  : for stationary mean (set manually)
                            - ndarray: for non stationary mean, must contain
                                as many entries as number of grid cells
                                (reshaped if needed)
                            For ordinary kriging (method='ordinary_kriging'),
                            it is used for case with no neighbor

    :param var:         (None or float or ndarray) variance of the simulation
                            (for simple kriging only):
                                - None   : variance not modified
                                           (only covariance model is used)
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
                            over the SG: 1 for simulated cell / 0 for not simulated
                            cell (nunber of entries should be equal to the number of
                            grid cells)

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
                        (float) indicating how restricting the search ellipsoid (should be positive):
                            let r_i be the ranges of the covariance model along its main axes,
                            if x is a node to be simulated, a node y is taken into account iff it is
                            within the ellipsoid centered at x of half-axes searchRadiusRelative * r_i
                            (unused if use_unique_neighborhood is True)
                            Note:
                                - if a range is a variable parameter, its maximal value over the simulation grid
                                  is considered
                                - if orientation of the covariance model is non-stationary, a "circular search neighborhood"
                                  is considered with the radius set to the maximum of all ranges

    :param nneighborMax:(int) maximum number of nodes retrieved from the search ellipsoid,
                            set -1 for unlimited
                            (unused if use_unique_neighborhood is True)

    :param searchNeighborhoodSortMode:
                        (int) indicating how to sort the search neighboorhood nodes
                            (neighbors), they are sorted in increasing order according to:
                                - searchNeighborhoodSortMode = 0:
                                  distance in the usual axes system
                                - searchNeighborhoodSortMode = 1:
                                  distance in the axes sytem supporting the covariance model
                                  and accounting for anisotropy given by the ranges
                                - searchNeighborhoodSortMode = 2:
                                  minus the evaluation of the covariance model
                            (unused if use_unique_neighborhood is True)
                            Note:
                                - if the covariance model has any variable parameter (non-stationary),
                                  then searchNeighborhoodSortMode = 2 is not allowed
                                - if the covariance model has any range or angle set as a variable parameter,
                                  then searchNeighborhoodSortMode must be set to 0
                                - greatest possible value as default

    :param outputReportFile:
                    (string or None) name of the report file, if None: no report file

    :param nthreads:
                (int) number of thread(s) to use for "GeosClassicSim" program (C),
                    (nthreads = -n <= 0: for maximal number of threads except n,
                    but at least 1)
    :param verbose:
                (int) indicates what is displayed during the GeosClassicSim run:
                    - 0: nothing
                    - 1: warning only
                    - 2 (or >1): warning and progress

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img (class)) output image, with image.nv=2 variables (estimate and
                    standard deviation)
                    (image is None if mpds_geosClassicOutput->outputImage is NULL)
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
    if not isinstance(cov_model, gcm.CovModel3D):
        print("ERROR (ESTIM_3D): 'cov_model' (first argument) is not valid")
        return None

    for el in cov_model.elem:
        # weight
        w = el[1]['w']
        if np.size(w) != 1 and np.size(w) != nxyz:
            print("ERROR (ESTIM_3D): 'cov_model': weight ('w') not compatible with simulation grid")
            return None
        # ranges
        if 'r' in el[1].keys():
            for r in el[1]['r']:
                if np.size(r) != 1 and np.size(r) != nxyz:
                    print("ERROR (ESTIM_3D): 'cov_model': range ('r') not compatible with simulation grid")
                    return None
        # additional parameter (s)
        if 's' in el[1].keys():
            s  = el[1]['s']
            if np.size(s) != 1 and np.size(s) != nxyz:
                print("ERROR (ESTIM_3D): 'cov_model': parameter ('s') not compatible with simulation grid")
                return None

    # alpha
    angle = cov_model.alpha
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        print("ERROR (ESTIM_3D): 'cov_model': angle (alpha) not compatible with simulation grid")
        return None

    # beta
    angle = cov_model.beta
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        print("ERROR (ESTIM_3D): 'cov_model': angle (beta) not compatible with simulation grid")
        return None

    # gamma
    angle = cov_model.gamma
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        print("ERROR (ESTIM_3D): 'cov_model': angle (gamma) not compatible with simulation grid")
        return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     print("ERROR (ESTIM_3D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 1
    elif method == 'ordinary_kriging':
        computationMode = 0
    else:
        print("ERROR (ESTIM_3D): 'method' is not valid")
        return None

    # data points: x, v
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 3) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            print("(ERROR (ESTIM_3D): length of 'v' is not valid")
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
            print("ERROR (ESTIM_3D): 'mask' is not valid")
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
           print("ERROR (ESTIM_3D): 'searchRadiusRelative' too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
           return None

       # Check parameters - nneighborMax
       if nneighborMax != -1 and nneighborMax <= 0:
           print("ERROR (ESTIM_3D): 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
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
                   print("ERROR (ESTIM_3D): 'searchNeighborhoodSortMode=2' not allowed with non-stationary covariance model")
                   return None
           elif searchNeighborhoodSortMode == 1:
               if not cov_model.is_orientation_stationary() or not cov_model.is_range_stationary():
                   print("ERROR (ESTIM_3D): 'searchNeighborhoodSortMode=1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                   return None

    # Check parameters - mean
    if mean is not None:
        # if method == 'ordinary_kriging':
        #     print("ERROR (ESTIM_3D): specifying 'mean' not allowed with ordinary kriging")
        #     return None
        mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if mean.size not in (1, nxyz):
            print("ERROR (ESTIM_3D): size of 'mean' is not valid")
            return None

    # Check parameters - var
    if var is not None:
        if method == 'ordinary_kriging':
            print("ERROR (ESTIM_3D): specifying 'var' not allowed with ordinary kriging")
            return None
        var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if var.size not in (1, nxyz):
            print("ERROR (ESTIM_3D): size of 'var' is not valid")
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
        0)

    if not flag:
        print("ERROR (ESTIM_3D): can not fill input structure!")
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
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr: warning only
    if verbose == 0:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    elif verbose == 1:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim"
    # err = geosclassic.MPDSGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth )

    # Free memory on C side: mpds_geosClassicInput
    geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
    #geosclassic.MPDSFree(mpds_geosClassicInput)
    geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)

    if err:
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree (mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 1 and geosclassic_output and geosclassic_output['nwarning']:
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
                    mpds_geosClassicIndicatorInput: C structure for "GeosClassicIndicatorSim" program (C)
                    flag: (bool) indicating if the filling has been done correctly (True) or not (False)
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
    geosclassic.mpds_set_geosClassicIndicatorInput_varname(mpds_geosClassicIndicatorInput, varname)

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
        geosclassic.mpds_set_geosClassicIndicatorInput_outputReportFileName(mpds_geosClassicIndicatorInput, outputReportFile)
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
        else:
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
            geosclassic.MPDS_IMAGE_array_setitem(mpds_geosClassicIndicatorInput.dataImage, i, img_py2C(dataIm))

    # mpds_geosClassicIndicatorInput.ndataPointSet and mpds_geosClassicIndicatorInput.dataPointSet
    if dataPointSet is None:
        mpds_geosClassicIndicatorInput.ndataPointSet = 0
    else:
        dataPointSet = np.asarray(dataPointSet).reshape(-1)
        n = len(dataPointSet)
        mpds_geosClassicIndicatorInput.ndataPointSet = n
        mpds_geosClassicIndicatorInput.dataPointSet = geosclassic.new_MPDS_POINTSET_array(n)
        for i, dataPS in enumerate(dataPointSet):
            geosclassic.MPDS_POINTSET_array_setitem(mpds_geosClassicIndicatorInput.dataPointSet, i, ps_py2C(dataPS))

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
        nthreads=-1, verbose=2):
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
                        (sequence of CovModel1D class of length ncategory (see
                            see category_values), or one CovModel1D, recycled)
                            covariance model in 1D per category, see definition of
                            the class in module geone.covModel

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
                            over the SG: 1 for simulated cell / 0 for not simulated
                            cell (nunber of entries should be equal to the number of
                            grid cells)

    :param searchRadiusRelative:
                        (sequence of ncategory floats (or float, recycled))
                            indicating how restricting the search ellipsoid (should be positive)
                            for each category:
                            let r_i be the ranges of the covariance model along its main axes,
                            if x is a node to be simulated, a node y is taken into account iff it is
                            within the ellipsoid centered at x of half-axes searchRadiusRelative * r_i
                            Note:
                                - if a range is a variable parameter, its maximal value over the simulation grid
                                  is considered

    :param nneighborMax:(sequence of ncategory ints (or int, recycled)) maximum number of
                            nodes retrieved from the search ellipsoid, for each category,
                            set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (sequence of ncategory ints (or int, recycled)) indicating
                            how to sort the search neighboorhood nodes (neighbors)
                            for each category, they are sorted in increasing order
                            according to:
                                - searchNeighborhoodSortMode = 0:
                                  distance in the usual axes system
                                - searchNeighborhoodSortMode = 1:
                                  distance in the axes sytem supporting the covariance model
                                  and accounting for anisotropy given by the ranges
                                - searchNeighborhoodSortMode = 2:
                                  minus the evaluation of the covariance model
                            Note:
                                - if the covariance model has any variable parameter (non-stationary),
                                  then searchNeighborhoodSortMode = 2 is not allowed
                                - if the covariance model has any range or angle set as a variable parameter,
                                  then searchNeighborhoodSortMode must be set to 0
                                - greatest possible value as default

    :param seed:        (int or None) initial seed, if None an initial seed between
                            1 and 999999 is generated with numpy.random.randint

    :param outputReportFile:
                    (string or None) name of the report file, if None: no report file

    :param nthreads:
                (int) number of thread(s) to use for "GeosClassicSim" program (C),
                    (nthreads = -n <= 0: for maximal number of threads except n,
                    but at least 1)
    :param verbose:
                (int) indicates what is displayed during the GeosClassicSim run:
                    - 0: nothing
                    - 1: warning only
                    - 2 (or >1): warning and progress

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img (class)) output image, with image.nv=nreal variables (each
                    variable is one realization)
                    (image is None if mpds_geosClassicOutput->outputImage is NULL)
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
        print("ERROR (SIMUL_INDIC_1D): 'category_values' is not valid")
        return None

    ncategory = len(category_values)
    if ncategory <= 0:
        print("ERROR (SIMUL_INDIC_1D): 'category_values' is empty")
        return None

    # cov_model_for_category
    cov_model_for_category = np.asarray(cov_model_for_category).reshape(-1)
    if len(cov_model_for_category) == 1:
        cov_model_for_category = np.repeat(cov_model_for_category, ncategory)
    elif len(cov_model_for_category) != ncategory:
        print("ERROR (SIMUL_INDIC_1D): 'cov_model_for_category' of invalid length")
        return None
    if not np.all([isinstance(c, gcm.CovModel1D) for c in cov_model_for_category]):
        print("ERROR (SIMUL_INDIC_1D): 'cov_model_for_category' should contains CovModel1D objects")
        return None

    for cov_model in cov_model_for_category:
        for el in cov_model.elem:
            # weight
            w = el[1]['w']
            if np.size(w) != 1 and np.size(w) != nxyz:
                print("ERROR (SIMUL_INDIC_1D): covariance model: weight ('w') not compatible with simulation grid")
                return None
            # ranges
            if 'r' in el[1].keys():
                r  = el[1]['r']
                if np.size(r) != 1 and np.size(r) != nxyz:
                    print("ERROR (SIMUL_INDIC_1D): covariance model: range ('r') not compatible with simulation grid")
                    return None
            # additional parameter (s)
            if 's' in el[1].keys():
                s  = el[1]['s']
                if np.size(s) != 1 and np.size(s) != nxyz:
                    print("ERROR (SIMUL_INDIC_1D): covariance model: parameter ('s') not compatible with simulation grid")
                    return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     print("ERROR (SIMUL_INDIC_1D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 3
    elif method == 'ordinary_kriging':
        computationMode = 2
    else:
        print("ERROR (SIMUL_INDIC_1D): 'method' is not valid")
        return None

    # data points: x, v
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            print("(ERROR (SIMUL_1D): length of 'v' is not valid")
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
            print("ERROR (SIMUL_INDIC_1D): 'mask' is not valid")
            return None

    # Check parameters - searchRadiusRelative
    searchRadiusRelative = np.asarray(searchRadiusRelative, dtype='float').reshape(-1)
    if len(searchRadiusRelative) == 1:
        searchRadiusRelative = np.repeat(searchRadiusRelative, ncategory)
    elif len(searchRadiusRelative) != ncategory:
        print("ERROR (SIMUL_INDIC_1D): 'searchRadiusRelative' of invalid length")
        return None

    for srr in searchRadiusRelative:
        if srr < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
            print("ERROR (SIMUL_INDIC_1D): a 'searchRadiusRelative' is too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
            return None

    # Check parameters - nneighborMax
    nneighborMax = np.asarray(nneighborMax, dtype='intc').reshape(-1)
    if len(nneighborMax) == 1:
        nneighborMax = np.repeat(nneighborMax, ncategory)
    elif len(nneighborMax) != ncategory:
        print("ERROR (SIMUL_INDIC_1D): 'nneighborMax' of invalid length")
        return None

    for nn in nneighborMax:
        if nn != -1 and nn <= 0:
            print("ERROR (SIMUL_INDIC_1D): any 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
            return None

    # Check parameters - searchNeighborhoodSortMode
    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode).reshape(-1)
    if len(searchNeighborhoodSortMode) == 1:
        searchNeighborhoodSortMode = np.repeat(searchNeighborhoodSortMode, ncategory)
    elif len(searchNeighborhoodSortMode) != ncategory:
        print("ERROR (SIMUL_INDIC_1D): 'searchNeighborhoodSortMode' of invalid length")
        return None

    for i in range(ncategory):
        if searchNeighborhoodSortMode[i] is None:
            # set greatest possible value
            if cov_model_for_category[i].is_stationary():
                searchNeighborhoodSortMode[i] = 2
            elif cov_model_for_category[i].is_orientation_stationary() and cov_model_for_category[i].is_range_stationary():
                searchNeighborhoodSortMode[i] = 1
            else:
                searchNeighborhoodSortMode[i] = 0
        else:
            if searchNeighborhoodSortMode[i] == 2:
                if not cov_model_for_category[i].is_stationary():
                    print("ERROR (SIMUL_INDIC_1D): 'searchNeighborhoodSortMode set to 2' not allowed with non-stationary covariance model")
                    return None
            elif searchNeighborhoodSortMode[i] == 1:
                if not cov_model_for_category[i].is_orientation_stationary() or not cov_model_for_category[i].is_range_stationary():
                    print("ERROR (SIMUL_INDIC_1D): 'searchNeighborhoodSortMode set to 1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                    return None

    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode, dtype='intc')

    # Check parameters - probability
    if probability is not None:
        # if method == 'ordinary_kriging':
        #     print("ERROR (SIMUL_INDIC_1D): specifying 'probability' not allowed with ordinary kriging")
        #     return None
        probability = np.asarray(probability, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if probability.size not in (ncategory, ncategory*nxyz):
            print("ERROR (SIMUL_INDIC_1D): size of 'probability' is not valid")
            return None

    # Check parameters - nreal
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose >= 2:
            print('SIMUL_INDIC_1D: nreal <= 0: nothing to do!')
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
        cov_model_for_category,
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
        print("ERROR (SIMUL_INDIC_1D): can not fill input structure!")
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
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr: warning only
    if verbose == 0:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    elif verbose == 1:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim"
    # err = geosclassic.MPDSGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth )

    # Free memory on C side: mpds_geosClassicIndicatorInput
    geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
    #geosclassic.MPDSFree(mpds_geosClassicIndicatorInput)
    geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)

    if err:
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree (mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 1 and geosclassic_output and geosclassic_output['nwarning']:
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
        nthreads=-1, verbose=2):
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
                        (sequence of CovModel2D class of length ncategory (see
                            see category_values), or one CovModel2D, recycled)
                            covariance model in 2D per category, see definition of
                            the class in module geone.covModel

    :param dimension:   (sequence of 2 ints) (nx, ny), number of cells
                            in x-, y-axis direction
    :param spacing:     (sequence of 2 floats) (sx, sy), spacing between
                            two adjacent cells in x-, y-axis direction
    :param origin:      (sequence of 2 floats) (ox, oy), origin of the 2D simulation
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

    :param x:           (2-dimensional array of dim n x 2, or
                            1-dimensional array of dim 2 or None) coordinate of
                            conditioning points for hard data
    :param v:           (1-dimensional array or float or None) value
                            at conditioning points for hard data
                            (same type as x)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not simulated
                            cell (nunber of entries should be equal to the number of
                            grid cells)

    :param searchRadiusRelative:
                        (sequence of ncategory floats (or float, recycled))
                            indicating how restricting the search ellipsoid (should be positive)
                            for each category:
                            let r_i be the ranges of the covariance model along its main axes,
                            if x is a node to be simulated, a node y is taken into account iff it is
                            within the ellipsoid centered at x of half-axes searchRadiusRelative * r_i
                            Note:
                                - if a range is a variable parameter, its maximal value over the simulation grid
                                  is considered
                                - if orientation of the covariance model is non-stationary, a "circular search neighborhood"
                                  is considered with the radius set to the maximum of all ranges

    :param nneighborMax:(sequence of ncategory ints (or int, recycled)) maximum number of
                            nodes retrieved from the search ellipsoid, for each category,
                            set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (sequence of ncategory ints (or int, recycled)) indicating
                            how to sort the search neighboorhood nodes (neighbors)
                            for each category, they are sorted in increasing order
                            according to:
                                - searchNeighborhoodSortMode = 0:
                                  distance in the usual axes system
                                - searchNeighborhoodSortMode = 1:
                                  distance in the axes sytem supporting the covariance model
                                  and accounting for anisotropy given by the ranges
                                - searchNeighborhoodSortMode = 2:
                                  minus the evaluation of the covariance model
                            Note:
                                - if the covariance model has any variable parameter (non-stationary),
                                  then searchNeighborhoodSortMode = 2 is not allowed
                                - if the covariance model has any range or angle set as a variable parameter,
                                  then searchNeighborhoodSortMode must be set to 0
                                - greatest possible value as default

    :param seed:        (int or None) initial seed, if None an initial seed between
                            1 and 999999 is generated with numpy.random.randint

    :param outputReportFile:
                    (string or None) name of the report file, if None: no report file

    :param nthreads:
                (int) number of thread(s) to use for "GeosClassicSim" program (C),
                    (nthreads = -n <= 0: for maximal number of threads except n,
                    but at least 1)
    :param verbose:
                (int) indicates what is displayed during the GeosClassicSim run:
                    - 0: nothing
                    - 1: warning only
                    - 2 (or >1): warning and progress

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img (class)) output image, with image.nv=nreal variables (each
                    variable is one realization)
                    (image is None if mpds_geosClassicOutput->outputImage is NULL)
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
        print("ERROR (SIMUL_INDIC_2D): 'category_values' is not valid")
        return None

    ncategory = len(category_values)
    if ncategory <= 0:
        print("ERROR (SIMUL_INDIC_2D): 'category_values' is empty")
        return None

    # cov_model_for_category
    cov_model_for_category = np.asarray(cov_model_for_category).reshape(-1)
    if len(cov_model_for_category) == 1:
        cov_model_for_category = np.repeat(cov_model_for_category, ncategory)
    elif len(cov_model_for_category) != ncategory:
        print("ERROR (SIMUL_INDIC_2D): 'cov_model_for_category' of invalid length")
        return None
    if not np.all([isinstance(c, gcm.CovModel2D) for c in cov_model_for_category]):
        print("ERROR (SIMUL_INDIC_2D): 'cov_model_for_category' should contains CovModel2D objects")
        return None

    for cov_model in cov_model_for_category:
        for el in cov_model.elem:
            # weight
            w = el[1]['w']
            if np.size(w) != 1 and np.size(w) != nxyz:
                print("ERROR (SIMUL_INDIC_2D): covariance model: weight ('w') not compatible with simulation grid")
                return None
            # ranges
            if 'r' in el[1].keys():
                for r in el[1]['r']:
                    if np.size(r) != 1 and np.size(r) != nxyz:
                        print("ERROR (SIMUL_INDIC_2D): covariance model: range ('r') not compatible with simulation grid")
                        return None
            # additional parameter (s)
            if 's' in el[1].keys():
                s  = el[1]['s']
                if np.size(s) != 1 and np.size(s) != nxyz:
                    print("ERROR (SIMUL_INDIC_2D): covariance model: parameter ('s') not compatible with simulation grid")
                    return None

        # alpha
        angle = cov_model.alpha
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            print("ERROR (SIMUL_INDIC_2D): covariance model: angle (alpha) not compatible with simulation grid")
            return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     print("ERROR (SIMUL_INDIC_2D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 3
    elif method == 'ordinary_kriging':
        computationMode = 2
    else:
        print("ERROR (SIMUL_INDIC_2D): 'method' is not valid")
        return None

    # data points: x, v
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 2) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            print("(ERROR (SIMUL_INDIC_2D): length of 'v' is not valid")
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
            print("ERROR (SIMUL_INDIC_2D): 'mask' is not valid")
            return None

    # Check parameters - searchRadiusRelative
    searchRadiusRelative = np.asarray(searchRadiusRelative, dtype='float').reshape(-1)
    if len(searchRadiusRelative) == 1:
        searchRadiusRelative = np.repeat(searchRadiusRelative, ncategory)
    elif len(searchRadiusRelative) != ncategory:
        print("ERROR (SIMUL_INDIC_2D): 'searchRadiusRelative' of invalid length")
        return None

    for srr in searchRadiusRelative:
        if srr < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
            print("ERROR (SIMUL_INDIC_2D): a 'searchRadiusRelative' is too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
            return None

    # Check parameters - nneighborMax
    nneighborMax = np.asarray(nneighborMax, dtype='intc').reshape(-1)
    if len(nneighborMax) == 1:
        nneighborMax = np.repeat(nneighborMax, ncategory)
    elif len(nneighborMax) != ncategory:
        print("ERROR (SIMUL_INDIC_2D): 'nneighborMax' of invalid length")
        return None

    for nn in nneighborMax:
        if nn != -1 and nn <= 0:
            print("ERROR (SIMUL_INDIC_2D): any 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
            return None

    # Check parameters - searchNeighborhoodSortMode
    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode).reshape(-1)
    if len(searchNeighborhoodSortMode) == 1:
        searchNeighborhoodSortMode = np.repeat(searchNeighborhoodSortMode, ncategory)
    elif len(searchNeighborhoodSortMode) != ncategory:
        print("ERROR (SIMUL_INDIC_2D): 'searchNeighborhoodSortMode' of invalid length")
        return None

    for i in range(ncategory):
        if searchNeighborhoodSortMode[i] is None:
            # set greatest possible value
            if cov_model_for_category[i].is_stationary():
                searchNeighborhoodSortMode[i] = 2
            elif cov_model_for_category[i].is_orientation_stationary() and cov_model_for_category[i].is_range_stationary():
                searchNeighborhoodSortMode[i] = 1
            else:
                searchNeighborhoodSortMode[i] = 0
        else:
            if searchNeighborhoodSortMode[i] == 2:
                if not cov_model_for_category[i].is_stationary():
                    print("ERROR (SIMUL_INDIC_2D): 'searchNeighborhoodSortMode set to 2' not allowed with non-stationary covariance model")
                    return None
            elif searchNeighborhoodSortMode[i] == 1:
                if not cov_model_for_category[i].is_orientation_stationary() or not cov_model_for_category[i].is_range_stationary():
                    print("ERROR (SIMUL_INDIC_2D): 'searchNeighborhoodSortMode set to 1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                    return None

    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode, dtype='intc')

    # Check parameters - probability
    if probability is not None:
        # if method == 'ordinary_kriging':
        #     print("ERROR (SIMUL_INDIC_2D): specifying 'probability' not allowed with ordinary kriging")
        #     return None
        probability = np.asarray(probability, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if probability.size not in (ncategory, ncategory*nxyz):
            print("ERROR (SIMUL_INDIC_2D): size of 'probability' is not valid")
            return None

    # Check parameters - nreal
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose >= 2:
            print('SIMUL_INDIC_2D: nreal <= 0: nothing to do!')
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
        cov_model_for_category,
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
        print("ERROR (SIMUL_INDIC_2D): can not fill input structure!")
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
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr: warning only
    if verbose == 0:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    elif verbose == 1:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim"
    # err = geosclassic.MPDSGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth )

    # Free memory on C side: mpds_geosClassicIndicatorInput
    geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
    #geosclassic.MPDSFree(mpds_geosClassicIndicatorInput)
    geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)

    if err:
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree (mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 1 and geosclassic_output and geosclassic_output['nwarning']:
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
        nthreads=-1, verbose=2):
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
                        (sequence of CovModel3D class of length ncategory (see
                            see category_values), or one CovModel3D, recycled)
                            covariance model in 3D per category, see definition of
                            the class in module geone.covModel

    :param dimension:   (sequence of 3 ints) (nx, ny, nz), number of cells
                            in x-, y-, z-axis direction
    :param spacing:     (sequence of 3 floats) (sx, sy, sz), spacing between
                            two adjacent cells in x-, y-, z-axis direction
    :param origin:      (sequence of 3 floats) (ox, oy, oy), origin of the 3D simulation
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

    :param x:           (2-dimensional array of dim n x 3, or
                            1-dimensional array of dim 3 or None) coordinate of
                            conditioning points for hard data
    :param v:           (1-dimensional array or float or None) value
                            at conditioning points for hard data
                            (same type as x)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not simulated
                            cell (nunber of entries should be equal to the number of
                            grid cells)

    :param searchRadiusRelative:
                        (sequence of ncategory floats (or float, recycled))
                            indicating how restricting the search ellipsoid (should be positive)
                            for each category:
                            let r_i be the ranges of the covariance model along its main axes,
                            if x is a node to be simulated, a node y is taken into account iff it is
                            within the ellipsoid centered at x of half-axes searchRadiusRelative * r_i
                            Note:
                                - if a range is a variable parameter, its maximal value over the simulation grid
                                  is considered
                                - if orientation of the covariance model is non-stationary, a "circular search neighborhood"
                                  is considered with the radius set to the maximum of all ranges

    :param nneighborMax:(sequence of ncategory ints (or int, recycled)) maximum number of
                            nodes retrieved from the search ellipsoid, for each category,
                            set -1 for unlimited

    :param searchNeighborhoodSortMode:
                        (sequence of ncategory ints (or int, recycled)) indicating
                            how to sort the search neighboorhood nodes (neighbors)
                            for each category, they are sorted in increasing order
                            according to:
                                - searchNeighborhoodSortMode = 0:
                                  distance in the usual axes system
                                - searchNeighborhoodSortMode = 1:
                                  distance in the axes sytem supporting the covariance model
                                  and accounting for anisotropy given by the ranges
                                - searchNeighborhoodSortMode = 2:
                                  minus the evaluation of the covariance model
                            Note:
                                - if the covariance model has any variable parameter (non-stationary),
                                  then searchNeighborhoodSortMode = 2 is not allowed
                                - if the covariance model has any range or angle set as a variable parameter,
                                  then searchNeighborhoodSortMode must be set to 0
                                - greatest possible value as default

    :param seed:        (int or None) initial seed, if None an initial seed between
                            1 and 999999 is generated with numpy.random.randint

    :param outputReportFile:
                    (string or None) name of the report file, if None: no report file

    :param nthreads:
                (int) number of thread(s) to use for "GeosClassicSim" program (C),
                    (nthreads = -n <= 0: for maximal number of threads except n,
                    but at least 1)
    :param verbose:
                (int) indicates what is displayed during the GeosClassicSim run:
                    - 0: nothing
                    - 1: warning only
                    - 2 (or >1): warning and progress

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img (class)) output image, with image.nv=nreal variables (each
                    variable is one realization)
                    (image is None if mpds_geosClassicOutput->outputImage is NULL)
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
    space_dim = 2

    # Set varname
    varname = 'V0'

    # --- Check and prepare parameters
    # category_values and ncategory (computed)
    try:
        category_values = np.asarray(category_values, dtype='float').reshape(-1)
    except:
        print("ERROR (SIMUL_INDIC_3D): 'category_values' is not valid")
        return None

    ncategory = len(category_values)
    if ncategory <= 0:
        print("ERROR (SIMUL_INDIC_3D): 'category_values' is empty")
        return None

    # cov_model_for_category
    cov_model_for_category = np.asarray(cov_model_for_category).reshape(-1)
    if len(cov_model_for_category) == 1:
        cov_model_for_category = np.repeat(cov_model_for_category, ncategory)
    elif len(cov_model_for_category) != ncategory:
        print("ERROR (SIMUL_INDIC_3D): 'cov_model_for_category' of invalid length")
        return None
    if not np.all([isinstance(c, gcm.CovModel3D) for c in cov_model_for_category]):
        print("ERROR (SIMUL_INDIC_3D): 'cov_model_for_category' should contains CovModel3D objects")
        return None

    for cov_model in cov_model_for_category:
        for el in cov_model.elem:
            # weight
            w = el[1]['w']
            if np.size(w) != 1 and np.size(w) != nxyz:
                print("ERROR (SIMUL_INDIC_3D): covariance model: weight ('w') not compatible with simulation grid")
                return None
            # ranges
            if 'r' in el[1].keys():
                for r in el[1]['r']:
                    if np.size(r) != 1 and np.size(r) != nxyz:
                        print("ERROR (SIMUL_INDIC_3D): covariance model: range ('r') not compatible with simulation grid")
                        return None
            # additional parameter (s)
            if 's' in el[1].keys():
                s  = el[1]['s']
                if np.size(s) != 1 and np.size(s) != nxyz:
                    print("ERROR (SIMUL_INDIC_3D): covariance model: parameter ('s') not compatible with simulation grid")
                    return None

        # alpha
        angle = cov_model.alpha
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            print("ERROR (SIMUL_INDIC_3D): covariance model: angle (alpha) not compatible with simulation grid")
            return None

        # beta
        angle = cov_model.beta
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            print("ERROR (SIMUL_INDIC_3D): covariance model: angle (beta) not compatible with simulation grid")
            return None

        # gamma
        angle = cov_model.gamma
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            print("ERROR (SIMUL_INDIC_3D): covariance model: angle (gamma) not compatible with simulation grid")
            return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     print("ERROR (SIMUL_INDIC_3D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 3
    elif method == 'ordinary_kriging':
        computationMode = 2
    else:
        print("ERROR (SIMUL_INDIC_3D): 'method' is not valid")
        return None

    # data points: x, v
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 3) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
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
            print("ERROR (SIMUL_INDIC_3D): 'mask' is not valid")
            return None

    # Check parameters - searchRadiusRelative
    searchRadiusRelative = np.asarray(searchRadiusRelative, dtype='float').reshape(-1)
    if len(searchRadiusRelative) == 1:
        searchRadiusRelative = np.repeat(searchRadiusRelative, ncategory)
    elif len(searchRadiusRelative) != ncategory:
        print("ERROR (SIMUL_INDIC_3D): 'searchRadiusRelative' of invalid length")
        return None

    for srr in searchRadiusRelative:
        if srr < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
            print("ERROR (SIMUL_INDIC_3D): a 'searchRadiusRelative' is too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
            return None

    # Check parameters - nneighborMax
    nneighborMax = np.asarray(nneighborMax, dtype='intc').reshape(-1)
    if len(nneighborMax) == 1:
        nneighborMax = np.repeat(nneighborMax, ncategory)
    elif len(nneighborMax) != ncategory:
        print("ERROR (SIMUL_INDIC_3D): 'nneighborMax' of invalid length")
        return None

    for nn in nneighborMax:
        if nn != -1 and nn <= 0:
            print("ERROR (SIMUL_INDIC_3D): any 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
            return None

    # Check parameters - searchNeighborhoodSortMode
    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode).reshape(-1)
    if len(searchNeighborhoodSortMode) == 1:
        searchNeighborhoodSortMode = np.repeat(searchNeighborhoodSortMode, ncategory)
    elif len(searchNeighborhoodSortMode) != ncategory:
        print("ERROR (SIMUL_INDIC_3D): 'searchNeighborhoodSortMode' of invalid length")
        return None

    for i in range(ncategory):
        if searchNeighborhoodSortMode[i] is None:
            # set greatest possible value
            if cov_model_for_category[i].is_stationary():
                searchNeighborhoodSortMode[i] = 2
            elif cov_model_for_category[i].is_orientation_stationary() and cov_model_for_category[i].is_range_stationary():
                searchNeighborhoodSortMode[i] = 1
            else:
                searchNeighborhoodSortMode[i] = 0
        else:
            if searchNeighborhoodSortMode[i] == 2:
                if not cov_model_for_category[i].is_stationary():
                    print("ERROR (SIMUL_INDIC_3D): 'searchNeighborhoodSortMode set to 2' not allowed with non-stationary covariance model")
                    return None
            elif searchNeighborhoodSortMode[i] == 1:
                if not cov_model_for_category[i].is_orientation_stationary() or not cov_model_for_category[i].is_range_stationary():
                    print("ERROR (SIMUL_INDIC_3D): 'searchNeighborhoodSortMode set to 1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                    return None

    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode, dtype='intc')

    # Check parameters - probability
    if probability is not None:
        # if method == 'ordinary_kriging':
        #     print("ERROR (SIMUL_INDIC_3D): specifying 'probability' not allowed with ordinary kriging")
        #     return None
        probability = np.asarray(probability, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if probability.size not in (ncategory, ncategory*nxyz):
            print("ERROR (SIMUL_INDIC_3D): size of 'probability' is not valid")
            return None

    # Check parameters - nreal
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose >= 2:
            print('SIMUL_INDIC_3D: nreal <= 0: nothing to do!')
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
        cov_model_for_category,
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
        print("ERROR (SIMUL_INDIC_3D): can not fill input structure!")
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
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr: warning only
    if verbose == 0:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    elif verbose == 1:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim"
    # err = geosclassic.MPDSGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth )

    # Free memory on C side: mpds_geosClassicIndicatorInput
    geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
    #geosclassic.MPDSFree(mpds_geosClassicIndicatorInput)
    geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)

    if err:
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree (mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 1 and geosclassic_output and geosclassic_output['nwarning']:
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
        nthreads=-1, verbose=2):
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
                        (sequence of CovModel1D class of length ncategory (see
                            see category_values), or one CovModel1D, recycled)
                            covariance model in 1D per category, see definition of
                            the class in module geone.covModel

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
                            over the SG: 1 for simulated cell / 0 for not simulated
                            cell (nunber of entries should be equal to the number of
                            grid cells)

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
                            indicating how restricting the search ellipsoid (should be positive)
                            for each category:
                            let r_i be the ranges of the covariance model along its main axes,
                            if x is a node to be simulated, a node y is taken into account iff it is
                            within the ellipsoid centered at x of half-axes searchRadiusRelative * r_i
                            (unused if use_unique_neighborhood is True)
                            Note:
                                - if a range is a variable parameter, its maximal value over the simulation grid
                                  is considered

    :param nneighborMax:(sequence of ncategory ints (or int, recycled)) maximum number of
                            nodes retrieved from the search ellipsoid, for each category,
                            set -1 for unlimited
                            (unused if use_unique_neighborhood is True)

    :param searchNeighborhoodSortMode:
                        (sequence of ncategory ints (or int, recycled)) indicating
                            how to sort the search neighboorhood nodes (neighbors)
                            for each category, they are sorted in increasing order
                            according to:
                                - searchNeighborhoodSortMode = 0:
                                  distance in the usual axes system
                                - searchNeighborhoodSortMode = 1:
                                  distance in the axes sytem supporting the covariance model
                                  and accounting for anisotropy given by the ranges
                                - searchNeighborhoodSortMode = 2:
                                  minus the evaluation of the covariance model
                            (unused if use_unique_neighborhood is True)
                            Note:
                                - if the covariance model has any variable parameter (non-stationary),
                                  then searchNeighborhoodSortMode = 2 is not allowed
                                - if the covariance model has any range or angle set as a variable parameter,
                                  then searchNeighborhoodSortMode must be set to 0
                                - greatest possible value as default

    :param outputReportFile:
                    (string or None) name of the report file, if None: no report file

    :param nthreads:
                (int) number of thread(s) to use for "GeosClassicSim" program (C),
                    (nthreads = -n <= 0: for maximal number of threads except n,
                    but at least 1)
    :param verbose:
                (int) indicates what is displayed during the GeosClassicSim run:
                    - 0: nothing
                    - 1: warning only
                    - 2 (or >1): warning and progress

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img (class)) output image, with image.nv=ncategory variables
                    (estimate probabilities (for each category))
                    (image is None if mpds_geosClassicOutput->outputImage is NULL)
        image:  (Img (class)) output image, with image.nv=nreal variables (each
                    variable is one realization)
                    (image is None if mpds_geosClassicOutput->outputImage is NULL)
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
        print("ERROR (ESTIM_INDIC_1D): 'category_values' is not valid")
        return None

    ncategory = len(category_values)
    if ncategory <= 0:
        print("ERROR (ESTIM_INDIC_1D): 'category_values' is empty")
        return None

    # cov_model_for_category
    cov_model_for_category = np.asarray(cov_model_for_category).reshape(-1)
    if len(cov_model_for_category) == 1:
        cov_model_for_category = np.repeat(cov_model_for_category, ncategory)
    elif len(cov_model_for_category) != ncategory:
        print("ERROR (ESTIM_INDIC_1D): 'cov_model_for_category' of invalid length")
        return None
    if not np.all([isinstance(c, gcm.CovModel1D) for c in cov_model_for_category]):
        print("ERROR (ESTIM_INDIC_1D): 'cov_model_for_category' should contains CovModel1D objects")
        return None

    for cov_model in cov_model_for_category:
        for el in cov_model.elem:
            # weight
            w = el[1]['w']
            if np.size(w) != 1 and np.size(w) != nxyz:
                print("ERROR (ESTIM_INDIC_1D): covariance model: weight ('w') not compatible with simulation grid")
                return None
            # ranges
            if 'r' in el[1].keys():
                r  = el[1]['r']
                if np.size(r) != 1 and np.size(r) != nxyz:
                    print("ERROR (ESTIM_INDIC_1D): covariance model: range ('r') not compatible with simulation grid")
                    return None
            # additional parameter (s)
            if 's' in el[1].keys():
                s  = el[1]['s']
                if np.size(s) != 1 and np.size(s) != nxyz:
                    print("ERROR (ESTIM_INDIC_1D): covariance model: parameter ('s') not compatible with simulation grid")
                    return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     print("ERROR (ESTIM_INDIC_1D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 1
    elif method == 'ordinary_kriging':
        computationMode = 0
    else:
        print("ERROR (ESTIM_INDIC_1D): 'method' is not valid")
        return None

    # data points: x, v
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            print("(ERROR (SIMUL_1D): length of 'v' is not valid")
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
            print("ERROR (ESTIM_INDIC_1D): 'mask' is not valid")
            return None

    # Check parameters - use_unique_neighborhood (length)
    use_unique_neighborhood = np.asarray(use_unique_neighborhood, dtype='bool').reshape(-1)
    if len(use_unique_neighborhood) == 1:
        use_unique_neighborhood = np.repeat(use_unique_neighborhood, ncategory)
    elif len(use_unique_neighborhood) != ncategory:
        print("ERROR (ESTIM_INDIC_1D): 'use_unique_neighborhood' of invalid length")
        return None

    # Check parameters - searchRadiusRelative (length)
    searchRadiusRelative = np.asarray(searchRadiusRelative, dtype='float').reshape(-1)
    if len(searchRadiusRelative) == 1:
        searchRadiusRelative = np.repeat(searchRadiusRelative, ncategory)
    elif len(searchRadiusRelative) != ncategory:
        print("ERROR (ESTIM_INDIC_1D): 'searchRadiusRelative' of invalid length")
        return None

    # Check parameters - nneighborMax (length)
    nneighborMax = np.asarray(nneighborMax, dtype='intc').reshape(-1)
    if len(nneighborMax) == 1:
        nneighborMax = np.repeat(nneighborMax, ncategory)
    elif len(nneighborMax) != ncategory:
        print("ERROR (ESTIM_INDIC_1D): 'nneighborMax' of invalid length")
        return None

    # Check parameters - searchNeighborhoodSortMode (length)
    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode).reshape(-1)
    if len(searchNeighborhoodSortMode) == 1:
        searchNeighborhoodSortMode = np.repeat(searchNeighborhoodSortMode, ncategory)
    elif len(searchNeighborhoodSortMode) != ncategory:
        print("ERROR (ESTIM_INDIC_1D): 'searchNeighborhoodSortMode' of invalid length")
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
                print("ERROR (ESTIM_INDIC_1D): a 'searchRadiusRelative' is too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
                return None

            if nneighborMax[i] != -1 and nneighborMax[i] <= 0:
                print("ERROR (ESTIM_INDIC_1D): any 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
                return None

            if searchNeighborhoodSortMode[i] is None:
                # set greatest possible value
                if cov_model_for_category[i].is_stationary():
                    searchNeighborhoodSortMode[i] = 2
                elif cov_model_for_category[i].is_orientation_stationary() and cov_model_for_category[i].is_range_stationary():
                    searchNeighborhoodSortMode[i] = 1
                else:
                    searchNeighborhoodSortMode[i] = 0
            else:
                if searchNeighborhoodSortMode[i] == 2:
                    if not cov_model_for_category[i].is_stationary():
                        print("ERROR (ESTIM_INDIC_1D): 'searchNeighborhoodSortMode set to 2' not allowed with non-stationary covariance model")
                        return None
                elif searchNeighborhoodSortMode[i] == 1:
                    if not cov_model_for_category[i].is_orientation_stationary() or not cov_model_for_category[i].is_range_stationary():
                        print("ERROR (ESTIM_INDIC_1D): 'searchNeighborhoodSortMode set to 1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                        return None

    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode, dtype='intc')

    # Check parameters - probability
    if probability is not None:
        # if method == 'ordinary_kriging':
        #     print("ERROR (ESTIM_INDIC_1D): specifying 'probability' not allowed with ordinary kriging")
        #     return None
        probability = np.asarray(probability, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if probability.size not in (ncategory, ncategory*nxyz):
            print("ERROR (ESTIM_INDIC_1D): size of 'probability' is not valid")
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
        cov_model_for_category,
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
        print("ERROR (ESTIM_INDIC_1D): can not fill input structure!")
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
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr: warning only
    if verbose == 0:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    elif verbose == 1:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim"
    # err = geosclassic.MPDSGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth )

    # Free memory on C side: mpds_geosClassicIndicatorInput
    geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
    #geosclassic.MPDSFree(mpds_geosClassicIndicatorInput)
    geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)

    if err:
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree (mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 1 and geosclassic_output and geosclassic_output['nwarning']:
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
        nthreads=-1, verbose=2):
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
                        (sequence of CovModel2D class of length ncategory (see
                            see category_values), or one CovModel2D, recycled)
                            covariance model in 2D per category, see definition of
                            the class in module geone.covModel

    :param dimension:   (sequence of 2 ints) (nx, ny), number of cells
                            in x-, y-axis direction
    :param spacing:     (sequence of 2 floats) (sx, sy), spacing between
                            two adjacent cells in x-, y-axis direction
    :param origin:      (sequence of 2 floats) (ox, oy), origin of the 2D simulation
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

    :param x:           (2-dimensional array of dim n x 2, or
                            1-dimensional array of dim 2 or None) coordinate of
                            conditioning points for hard data
    :param v:           (1-dimensional array or float or None) value
                            at conditioning points for hard data
                            (same type as x)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not simulated
                            cell (nunber of entries should be equal to the number of
                            grid cells)

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
                            indicating how restricting the search ellipsoid (should be positive)
                            for each category:
                            let r_i be the ranges of the covariance model along its main axes,
                            if x is a node to be simulated, a node y is taken into account iff it is
                            within the ellipsoid centered at x of half-axes searchRadiusRelative * r_i
                            (unused if use_unique_neighborhood is True)
                            Note:
                                - if a range is a variable parameter, its maximal value over the simulation grid
                                  is considered
                                - if orientation of the covariance model is non-stationary, a "circular search neighborhood"
                                  is considered with the radius set to the maximum of all ranges

    :param nneighborMax:(sequence of ncategory ints (or int, recycled)) maximum number of
                            nodes retrieved from the search ellipsoid, for each category,
                            set -1 for unlimited
                            (unused if use_unique_neighborhood is True)

    :param searchNeighborhoodSortMode:
                        (sequence of ncategory ints (or int, recycled)) indicating
                            how to sort the search neighboorhood nodes (neighbors)
                            for each category, they are sorted in increasing order
                            according to:
                                - searchNeighborhoodSortMode = 0:
                                  distance in the usual axes system
                                - searchNeighborhoodSortMode = 1:
                                  distance in the axes sytem supporting the covariance model
                                  and accounting for anisotropy given by the ranges
                                - searchNeighborhoodSortMode = 2:
                                  minus the evaluation of the covariance model
                            (unused if use_unique_neighborhood is True)
                            Note:
                                - if the covariance model has any variable parameter (non-stationary),
                                  then searchNeighborhoodSortMode = 2 is not allowed
                                - if the covariance model has any range or angle set as a variable parameter,
                                  then searchNeighborhoodSortMode must be set to 0
                                - greatest possible value as default

    :param outputReportFile:
                    (string or None) name of the report file, if None: no report file

    :param nthreads:
                (int) number of thread(s) to use for "GeosClassicSim" program (C),
                    (nthreads = -n <= 0: for maximal number of threads except n,
                    but at least 1)
    :param verbose:
                (int) indicates what is displayed during the GeosClassicSim run:
                    - 0: nothing
                    - 1: warning only
                    - 2 (or >1): warning and progress

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img (class)) output image, with image.nv=ncategory variables
                    (estimate probabilities (for each category))
                    (image is None if mpds_geosClassicOutput->outputImage is NULL)
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
        print("ERROR (ESTIM_INDIC_2D): 'category_values' is not valid")
        return None

    ncategory = len(category_values)
    if ncategory <= 0:
        print("ERROR (ESTIM_INDIC_2D): 'category_values' is empty")
        return None

    # cov_model_for_category
    cov_model_for_category = np.asarray(cov_model_for_category).reshape(-1)
    if len(cov_model_for_category) == 1:
        cov_model_for_category = np.repeat(cov_model_for_category, ncategory)
    elif len(cov_model_for_category) != ncategory:
        print("ERROR (ESTIM_INDIC_2D): 'cov_model_for_category' of invalid length")
        return None
    if not np.all([isinstance(c, gcm.CovModel2D) for c in cov_model_for_category]):
        print("ERROR (ESTIM_INDIC_2D): 'cov_model_for_category' should contains CovModel2D objects")
        return None

    for cov_model in cov_model_for_category:
        for el in cov_model.elem:
            # weight
            w = el[1]['w']
            if np.size(w) != 1 and np.size(w) != nxyz:
                print("ERROR (ESTIM_INDIC_2D): covariance model: weight ('w') not compatible with simulation grid")
                return None
            # ranges
            if 'r' in el[1].keys():
                for r in el[1]['r']:
                    if np.size(r) != 1 and np.size(r) != nxyz:
                        print("ERROR (ESTIM_INDIC_2D): covariance model: range ('r') not compatible with simulation grid")
                        return None
            # additional parameter (s)
            if 's' in el[1].keys():
                s  = el[1]['s']
                if np.size(s) != 1 and np.size(s) != nxyz:
                    print("ERROR (ESTIM_INDIC_2D): covariance model: parameter ('s') not compatible with simulation grid")
                    return None

        # alpha
        angle = cov_model.alpha
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            print("ERROR (ESTIM_INDIC_2D): covariance model: angle (alpha) not compatible with simulation grid")
            return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     print("ERROR (ESTIM_INDIC_2D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 1
    elif method == 'ordinary_kriging':
        computationMode = 0
    else:
        print("ERROR (ESTIM_INDIC_2D): 'method' is not valid")
        return None

    # data points: x, v
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 2) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            print("(ERROR (ESTIM_INDIC_2D): length of 'v' is not valid")
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
            print("ERROR (ESTIM_INDIC_2D): 'mask' is not valid")
            return None

    # Check parameters - use_unique_neighborhood (length)
    use_unique_neighborhood = np.asarray(use_unique_neighborhood, dtype='bool').reshape(-1)
    if len(use_unique_neighborhood) == 1:
        use_unique_neighborhood = np.repeat(use_unique_neighborhood, ncategory)
    elif len(use_unique_neighborhood) != ncategory:
        print("ERROR (ESTIM_INDIC_2D): 'use_unique_neighborhood' of invalid length")
        return None

    # Check parameters - searchRadiusRelative (length)
    searchRadiusRelative = np.asarray(searchRadiusRelative, dtype='float').reshape(-1)
    if len(searchRadiusRelative) == 1:
        searchRadiusRelative = np.repeat(searchRadiusRelative, ncategory)
    elif len(searchRadiusRelative) != ncategory:
        print("ERROR (ESTIM_INDIC_2D): 'searchRadiusRelative' of invalid length")
        return None

    # Check parameters - nneighborMax (length)
    nneighborMax = np.asarray(nneighborMax, dtype='intc').reshape(-1)
    if len(nneighborMax) == 1:
        nneighborMax = np.repeat(nneighborMax, ncategory)
    elif len(nneighborMax) != ncategory:
        print("ERROR (ESTIM_INDIC_2D): 'nneighborMax' of invalid length")
        return None

    # Check parameters - searchNeighborhoodSortMode (length)
    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode).reshape(-1)
    if len(searchNeighborhoodSortMode) == 1:
        searchNeighborhoodSortMode = np.repeat(searchNeighborhoodSortMode, ncategory)
    elif len(searchNeighborhoodSortMode) != ncategory:
        print("ERROR (ESTIM_INDIC_2D): 'searchNeighborhoodSortMode' of invalid length")
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
                print("ERROR (ESTIM_INDIC_2D): a 'searchRadiusRelative' is too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
                return None

            if nneighborMax[i] != -1 and nneighborMax[i] <= 0:
                print("ERROR (ESTIM_INDIC_2D): any 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
                return None

            if searchNeighborhoodSortMode[i] is None:
                # set greatest possible value
                if cov_model_for_category[i].is_stationary():
                    searchNeighborhoodSortMode[i] = 2
                elif cov_model_for_category[i].is_orientation_stationary() and cov_model_for_category[i].is_range_stationary():
                    searchNeighborhoodSortMode[i] = 1
                else:
                    searchNeighborhoodSortMode[i] = 0
            else:
                if searchNeighborhoodSortMode[i] == 2:
                    if not cov_model_for_category[i].is_stationary():
                        print("ERROR (ESTIM_INDIC_2D): 'searchNeighborhoodSortMode set to 2' not allowed with non-stationary covariance model")
                        return None
                elif searchNeighborhoodSortMode[i] == 1:
                    if not cov_model_for_category[i].is_orientation_stationary() or not cov_model_for_category[i].is_range_stationary():
                        print("ERROR (ESTIM_INDIC_2D): 'searchNeighborhoodSortMode set to 1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                        return None

    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode, dtype='intc')

    # Check parameters - probability
    if probability is not None:
        # if method == 'ordinary_kriging':
        #     print("ERROR (ESTIM_INDIC_2D): specifying 'probability' not allowed with ordinary kriging")
        #     return None
        probability = np.asarray(probability, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if probability.size not in (ncategory, ncategory*nxyz):
            print("ERROR (ESTIM_INDIC_2D): size of 'probability' is not valid")
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
        cov_model_for_category,
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
        print("ERROR (ESTIM_INDIC_2D): can not fill input structure!")
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
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr: warning only
    if verbose == 0:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    elif verbose == 1:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim"
    # err = geosclassic.MPDSGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth )

    # Free memory on C side: mpds_geosClassicIndicatorInput
    geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
    #geosclassic.MPDSFree(mpds_geosClassicIndicatorInput)
    geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)

    if err:
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree (mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 1 and geosclassic_output and geosclassic_output['nwarning']:
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
        nthreads=-1, verbose=2):
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
                        (sequence of CovModel3D class of length ncategory (see
                            see category_values), or one CovModel3D, recycled)
                            covariance model in 3D per category, see definition of
                            the class in module geone.covModel

    :param dimension:   (sequence of 3 ints) (nx, ny, nz), number of cells
                            in x-, y-, z-axis direction
    :param spacing:     (sequence of 3 floats) (sx, sy, sz), spacing between
                            two adjacent cells in x-, y-, z-axis direction
    :param origin:      (sequence of 3 floats) (ox, oy, oy), origin of the 3D simulation
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

    :param x:           (2-dimensional array of dim n x 3, or
                            1-dimensional array of dim 3 or None) coordinate of
                            conditioning points for hard data
    :param v:           (1-dimensional array or float or None) value
                            at conditioning points for hard data
                            (same type as x)

    :param mask:        (nd-array of ints, or None) if given, mask values
                            over the SG: 1 for simulated cell / 0 for not simulated
                            cell (nunber of entries should be equal to the number of
                            grid cells)

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
                            indicating how restricting the search ellipsoid (should be positive)
                            for each category:
                            let r_i be the ranges of the covariance model along its main axes,
                            if x is a node to be simulated, a node y is taken into account iff it is
                            within the ellipsoid centered at x of half-axes searchRadiusRelative * r_i
                            (unused if use_unique_neighborhood is True)
                            Note:
                                - if a range is a variable parameter, its maximal value over the simulation grid
                                  is considered
                                - if orientation of the covariance model is non-stationary, a "circular search neighborhood"
                                  is considered with the radius set to the maximum of all ranges

    :param nneighborMax:(sequence of ncategory ints (or int, recycled)) maximum number of
                            nodes retrieved from the search ellipsoid, for each category,
                            set -1 for unlimited
                            (unused if use_unique_neighborhood is True)

    :param searchNeighborhoodSortMode:
                        (sequence of ncategory ints (or int, recycled)) indicating
                            how to sort the search neighboorhood nodes (neighbors)
                            for each category, they are sorted in increasing order
                            according to:
                                - searchNeighborhoodSortMode = 0:
                                  distance in the usual axes system
                                - searchNeighborhoodSortMode = 1:
                                  distance in the axes sytem supporting the covariance model
                                  and accounting for anisotropy given by the ranges
                                - searchNeighborhoodSortMode = 2:
                                  minus the evaluation of the covariance model
                            (unused if use_unique_neighborhood is True)
                            Note:
                                - if the covariance model has any variable parameter (non-stationary),
                                  then searchNeighborhoodSortMode = 2 is not allowed
                                - if the covariance model has any range or angle set as a variable parameter,
                                  then searchNeighborhoodSortMode must be set to 0
                                - greatest possible value as default

    :param outputReportFile:
                    (string or None) name of the report file, if None: no report file

    :param nthreads:
                (int) number of thread(s) to use for "GeosClassicSim" program (C),
                    (nthreads = -n <= 0: for maximal number of threads except n,
                    but at least 1)
    :param verbose:
                (int) indicates what is displayed during the GeosClassicSim run:
                    - 0: nothing
                    - 1: warning only
                    - 2 (or >1): warning and progress

    :return geosclassic_output: (dict)
            {'image':image,
             'nwarning':nwarning,
             'warnings':warnings}
        image:  (Img (class)) output image, with image.nv=ncategory variables
                    (estimate probabilities (for each category))
                    (image is None if mpds_geosClassicOutput->outputImage is NULL)
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
    space_dim = 2

    # Set varname
    varname = 'V0'

    # --- Check and prepare parameters
    # category_values and ncategory (computed)
    try:
        category_values = np.asarray(category_values, dtype='float').reshape(-1)
    except:
        print("ERROR (ESTIM_INDIC_3D): 'category_values' is not valid")
        return None

    ncategory = len(category_values)
    if ncategory <= 0:
        print("ERROR (ESTIM_INDIC_3D): 'category_values' is empty")
        return None

    # cov_model_for_category
    cov_model_for_category = np.asarray(cov_model_for_category).reshape(-1)
    if len(cov_model_for_category) == 1:
        cov_model_for_category = np.repeat(cov_model_for_category, ncategory)
    elif len(cov_model_for_category) != ncategory:
        print("ERROR (ESTIM_INDIC_3D): 'cov_model_for_category' of invalid length")
        return None
    if not np.all([isinstance(c, gcm.CovModel3D) for c in cov_model_for_category]):
        print("ERROR (ESTIM_INDIC_3D): 'cov_model_for_category' should contains CovModel3D objects")
        return None

    for cov_model in cov_model_for_category:
        for el in cov_model.elem:
            # weight
            w = el[1]['w']
            if np.size(w) != 1 and np.size(w) != nxyz:
                print("ERROR (ESTIM_INDIC_3D): covariance model: weight ('w') not compatible with simulation grid")
                return None
            # ranges
            if 'r' in el[1].keys():
                for r in el[1]['r']:
                    if np.size(r) != 1 and np.size(r) != nxyz:
                        print("ERROR (ESTIM_INDIC_3D): covariance model: range ('r') not compatible with simulation grid")
                        return None
            # additional parameter (s)
            if 's' in el[1].keys():
                s  = el[1]['s']
                if np.size(s) != 1 and np.size(s) != nxyz:
                    print("ERROR (ESTIM_INDIC_3D): covariance model: parameter ('s') not compatible with simulation grid")
                    return None

        # alpha
        angle = cov_model.alpha
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            print("ERROR (ESTIM_INDIC_3D): covariance model: angle (alpha) not compatible with simulation grid")
            return None

        # beta
        angle = cov_model.beta
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            print("ERROR (ESTIM_INDIC_3D): covariance model: angle (beta) not compatible with simulation grid")
            return None

        # gamma
        angle = cov_model.gamma
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            print("ERROR (ESTIM_INDIC_3D): covariance model: angle (gamma) not compatible with simulation grid")
            return None

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    # if method not in ('simple_kriging', 'ordinary_kriging'):
    #     print("ERROR (ESTIM_INDIC_3D): 'method' is not valid")
    #     return None
    if method == 'simple_kriging':
        computationMode = 1
    elif method == 'ordinary_kriging':
        computationMode = 0
    else:
        print("ERROR (ESTIM_INDIC_3D): 'method' is not valid")
        return None

    # data points: x, v
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 3) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
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
            print("ERROR (ESTIM_INDIC_3D): 'mask' is not valid")
            return None

    # Check parameters - use_unique_neighborhood (length)
    use_unique_neighborhood = np.asarray(use_unique_neighborhood, dtype='bool').reshape(-1)
    if len(use_unique_neighborhood) == 1:
        use_unique_neighborhood = np.repeat(use_unique_neighborhood, ncategory)
    elif len(use_unique_neighborhood) != ncategory:
        print("ERROR (ESTIM_INDIC_3D): 'use_unique_neighborhood' of invalid length")
        return None

    # Check parameters - searchRadiusRelative (length)
    searchRadiusRelative = np.asarray(searchRadiusRelative, dtype='float').reshape(-1)
    if len(searchRadiusRelative) == 1:
        searchRadiusRelative = np.repeat(searchRadiusRelative, ncategory)
    elif len(searchRadiusRelative) != ncategory:
        print("ERROR (ESTIM_INDIC_3D): 'searchRadiusRelative' of invalid length")
        return None

    # Check parameters - nneighborMax (length)
    nneighborMax = np.asarray(nneighborMax, dtype='intc').reshape(-1)
    if len(nneighborMax) == 1:
        nneighborMax = np.repeat(nneighborMax, ncategory)
    elif len(nneighborMax) != ncategory:
        print("ERROR (ESTIM_INDIC_3D): 'nneighborMax' of invalid length")
        return None

    # Check parameters - searchNeighborhoodSortMode (length)
    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode).reshape(-1)
    if len(searchNeighborhoodSortMode) == 1:
        searchNeighborhoodSortMode = np.repeat(searchNeighborhoodSortMode, ncategory)
    elif len(searchNeighborhoodSortMode) != ncategory:
        print("ERROR (ESTIM_INDIC_3D): 'searchNeighborhoodSortMode' of invalid length")
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
                print("ERROR (ESTIM_INDIC_3D): a 'searchRadiusRelative' is too small (should be at least {})".format(geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN))
                return None

            if nneighborMax[i] != -1 and nneighborMax[i] <= 0:
                print("ERROR (ESTIM_INDIC_3D): any 'nneighborMax' should be greater than 0 or equal to -1 (unlimited)")
                return None

            if searchNeighborhoodSortMode[i] is None:
                # set greatest possible value
                if cov_model_for_category[i].is_stationary():
                    searchNeighborhoodSortMode[i] = 2
                elif cov_model_for_category[i].is_orientation_stationary() and cov_model_for_category[i].is_range_stationary():
                    searchNeighborhoodSortMode[i] = 1
                else:
                    searchNeighborhoodSortMode[i] = 0
            else:
                if searchNeighborhoodSortMode[i] == 2:
                    if not cov_model_for_category[i].is_stationary():
                        print("ERROR (ESTIM_INDIC_3D): 'searchNeighborhoodSortMode set to 2' not allowed with non-stationary covariance model")
                        return None
                elif searchNeighborhoodSortMode[i] == 1:
                    if not cov_model_for_category[i].is_orientation_stationary() or not cov_model_for_category[i].is_range_stationary():
                        print("ERROR (ESTIM_INDIC_3D): 'searchNeighborhoodSortMode set to 1' not allowed with non-stationary range or non-stationary orientation in covariance model")
                        return None

    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode, dtype='intc')

    # Check parameters - probability
    if probability is not None:
        # if method == 'ordinary_kriging':
        #     print("ERROR (ESTIM_INDIC_3D): specifying 'probability' not allowed with ordinary kriging")
        #     return None
        probability = np.asarray(probability, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if probability.size not in (ncategory, ncategory*nxyz):
            print("ERROR (ESTIM_INDIC_3D): size of 'probability' is not valid")
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
        cov_model_for_category,
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
        print("ERROR (ESTIM_INDIC_3D): can not fill input structure!")
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
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr: warning only
    if verbose == 0:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    elif verbose == 1:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorWarningOnlyStdout_ptr
    else:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitorAllOnlyPercentStdout_ptr

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose >= 2:
        print('Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        sys.stdout.flush()
        sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim"
    # err = geosclassic.MPDSGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth )

    # Free memory on C side: mpds_geosClassicIndicatorInput
    geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
    #geosclassic.MPDSFree(mpds_geosClassicIndicatorInput)
    geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)

    if err:
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        print(err_message)
        geosclassic_output = None
    else:
        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    #geosclassic.MPDSFree (mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    #geosclassic.MPDSFree(mpds_progressMonitor)
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose >= 2 and geosclassic_output:
        print('Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose >= 1 and geosclassic_output and geosclassic_output['nwarning']:
        print('\nWarnings encountered ({} times in all):'.format(geosclassic_output['nwarning']))
        for i, warning_message in enumerate(geosclassic_output['warnings']):
            print('#{:3d}: {}'.format(i+1, warning_message))

    return geosclassic_output
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Module 'geone.geosclassicinterface'.")
