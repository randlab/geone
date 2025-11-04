#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# Python module:  'geosclassicinterface.py'
# author:         Julien Straubhaar
# date:           jun-2021
# -------------------------------------------------------------------------

"""
Module for interfacing classical geostatistics programs (in C) for python
(estimation and simulation based on simple and ordinary kriging).
"""

import numpy as np
import scipy
import sys, os
import copy
import multiprocessing

from geone import img
from geone.geosclassic_core import geosclassic
from geone import covModel as gcm
from geone.img import Img, PointSet

version = [geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER]

# ============================================================================
class GeosclassicinterfaceError(Exception):
    """
    Custom exception related to `geosclassicinterface` module.
    """
    pass
# ============================================================================

# ----------------------------------------------------------------------------
def img_py2C(im_py, logger=None):
    """
    Converts an image from python to C.

    Parameters
    ----------
    im_py : :class:`geone.img.Img`
        image in python
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    im_c : \\(MPDS_IMAGE \\*\\)
        image in C
    """
    fname = 'img_py2C'

    im_c = geosclassic.malloc_MPDS_IMAGE()
    geosclassic.MPDSInitImage(im_c)

    err = geosclassic.MPDSMallocImage(im_c, im_py.nxyz(), im_py.nv)
    if err:
        # Free memory on C side
        geosclassic.MPDSFreeImage(im_c)
        geosclassic.free_MPDS_IMAGE(im_c)
        # Raise error
        err_msg = f'{fname}: cannot convert image from python to C'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

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
def img_C2py(im_c, logger=None):
    """
    Converts an image from C to python.

    Parameters
    ----------
    im_c : \\(MPDS_IMAGE \\*\\)
        image in C

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    im_py : :class:`geone.img.Img`
        image in python
    """
    # fname = 'img_C2py'

    nxyz = im_c.grid.nx * im_c.grid.ny * im_c.grid.nz
    nxyzv = nxyz * im_c.nvar

    varname = [geosclassic.mpds_get_varname(im_c.varName, i) for i in range(im_c.nvar)]
    # varname = [geosclassic.charp_array_getitem(im_c.varName, i) for i in range(im_c.nvar)] # also works

    v = np.zeros(nxyzv)
    geosclassic.mpds_get_array_from_real_vector(im_c.var, 0, v)

    im_py = Img(nx=im_c.grid.nx, ny=im_c.grid.ny, nz=im_c.grid.nz,
                sx=im_c.grid.sx, sy=im_c.grid.sy, sz=im_c.grid.sz,
                ox=im_c.grid.ox, oy=im_c.grid.oy, oz=im_c.grid.oz,
                nv=im_c.nvar, val=v, varname=varname,
                logger=logger)

    np.putmask(im_py.val, im_py.val == geosclassic.MPDS_MISSING_VALUE, np.nan)

    return im_py
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def ps_py2C(ps_py, logger=None):
    """
    Converts an image from python to C.

    Parameters
    ----------
    ps_py : :class:`geone.img.PointSet`
        point set in python
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    ps_c : \\(MPDS_POINTSET \\*\\)
        point set in C
    """
    fname = 'ps_py2C'

    if ps_py.nv < 4:
        err_msg = f'{fname}: point set (python) have less than 4 variables'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    nvar = ps_py.nv - 3

    ps_c = geosclassic.malloc_MPDS_POINTSET()
    geosclassic.MPDSInitPointSet(ps_c)

    err = geosclassic.MPDSMallocPointSet(ps_c, ps_py.npt, nvar)
    if err:
        # Free memory on C side
        geosclassic.MPDSFreePointSet(ps_c)
        geosclassic.free_MPDS_POINTSET(ps_c)
        # Raise error
        err_msg = f'{fname}: cannot convert point set from python to C'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

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
    Converts an image from C to python.

    Parameters
    ----------
    ps_c : \\(MPDS_POINTSET \\*\\)
        point set in C

    Returns
    -------
    ps_py : :class:`geone.img.PointSet`
        point set in python
    """
    # fname = 'ps_C2py'

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
def covModel1Delem_py2C(
        covModelElem_py,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        logger=None):
    """
    Converts an elementary covariance model 1D from python to C.

    Simulation grid geometry is specified in case of non-stationary covariance
    model.

    Parameters
    ----------
    covModelElem_py : 2-tuple
        elementary covariance model 1D in python, `covModelElem_py` = (t, d),
        with:

        * t : str
            type of elementary covariance model, can be

            - 'nugget'         (see function :func:`geone.covModel.cov_nug`)
            - 'spherical'      (see function :func:`geone.covModel.cov_sph`)
            - 'exponential'    (see function :func:`geone.covModel.cov_exp`)
            - 'gaussian'       (see function :func:`geone.covModel.cov_gau`)
            - 'triangular'     (see function :func:`geone.covModel.cov_tri`)
            - 'cubic'          (see function :func:`geone.covModel.cov_cub`)
            - 'sinus_cardinal' (see function :func:`geone.covModel.cov_sinc`)
            - 'gamma'          (see function :func:`geone.covModel.cov_gamma`)
            - 'power'          (see function :func:`geone.covModel.cov_pow`)
            - 'exponential_generalized' (see function :func:`geone.covModel.cov_exp_gen`)
            - 'matern'         (see function :func:`geone.covModel.cov_matern`)

        * d : dict
            dictionary of required parameters to be passed to the elementary
            model `t` (value can be a "single value" or an array that matches
            the dimension of the simulation grid (for non-stationary
            covariance model)

        e.g.

        - (t, d) = ('spherical', {'w':2.0, 'r':1.5})
        - (t, d) = ('power', {'w':2.0, 'r':1.5, 's':1.7})
        - (t, d) = ('matern', {'w':2.0, 'r':1.5, 'nu':1.5})

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
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    covModelElem_c : \\(MPDS_COVMODELELEM \\*\\)
        elementary covariance model in C
    """
    fname = 'covModel1Delem_py2C'

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
    elif covModelElem_py[0] == 'triangular':
        covModelElem_c.covModelType = geosclassic.COV_TRIANGULAR
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
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (1D) from python to C ('w' not compatible with simulation grid)"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param,
                     logger=logger)
            try:
                covModelElem_c.weightImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (1D) from python to C ('w')"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

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
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (1D) from python to C ('r(x)' not compatible with simulation grid)"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param,
                     logger=logger)
            try:
                covModelElem_c.rxImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (1D) from python to C ('r(x)')"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

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
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (1D) from python to C ('s' not compatible with simulation grid)"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param,
                     logger=logger)
            try:
                covModelElem_c.sImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (1D) from python to C ('s')"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

    return covModelElem_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel2Delem_py2C(
        covModelElem_py,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        logger=None):
    """
    Converts an elementary covariance model 2D from python to C.

    Simulation grid geometry is specified in case of non-stationary covariance
    model.

    Parameters
    ----------
    covModelElem_py : 2-tuple
        elementary covariance model 2D in python, `covModelElem_py` = (t, d),
        with:

        * t : str
            type of elementary covariance model, can be

            - 'nugget'         (see function :func:`geone.covModel.cov_nug`)
            - 'spherical'      (see function :func:`geone.covModel.cov_sph`)
            - 'exponential'    (see function :func:`geone.covModel.cov_exp`)
            - 'gaussian'       (see function :func:`geone.covModel.cov_gau`)
            - 'triangular'     (see function :func:`geone.covModel.cov_tri`)
            - 'cubic'          (see function :func:`geone.covModel.cov_cub`)
            - 'sinus_cardinal' (see function :func:`geone.covModel.cov_sinc`)
            - 'gamma'          (see function :func:`geone.covModel.cov_gamma`)
            - 'power'          (see function :func:`geone.covModel.cov_pow`)
            - 'exponential_generalized' (see function :func:`geone.covModel.cov_exp_gen`)
            - 'matern'         (see function :func:`geone.covModel.cov_matern`)

        * d : dict
            dictionary of required parameters to be passed to the elementary
            model `t` (value can be a "single value" or an array that matches
            the dimension of the simulation grid (for non-stationary
            covariance model)

        e.g.

        - (t, d) = ('spherical', {'w':2.0, 'r':[1.5, 2.5]})
        - (t, d) = ('power', {'w':2.0, 'r':[1.5, 2.5], 's':1.7})
        - (t, d) = ('matern', {'w':2.0, 'r':[1.5, 2.5], 'nu':1.5})

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
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    covModelElem_c : \\(MPDS_COVMODELELEM \\*\\)
        elementary covariance model in C
    """
    fname = 'covModel2Delem_py2C'

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
    elif covModelElem_py[0] == 'triangular':
        covModelElem_c.covModelType = geosclassic.COV_TRIANGULAR
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
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (2D) from python to C ('w' not compatible with simulation grid)"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param,
                     logger=logger)
            try:
                covModelElem_c.weightImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (2D) from python to C ('w')"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

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
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (2D) from python to C ('r(x)' not compatible with simulation grid)"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param,
                     logger=logger)
            try:
                covModelElem_c.rxImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (2D) from python to C ('r(x)')"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

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
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (2D) from python to C ('r(y)' not compatible with simulation grid)"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param,
                     logger=logger)
            try:
                covModelElem_c.ryImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (2D) from python to C ('r(y)')"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

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
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (2D) from python to C ('s' not compatible with simulation grid)"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param,
                     logger=logger)
            try:
                covModelElem_c.sImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (2D) from python to C ('s')"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

    return covModelElem_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel3Delem_py2C(
        covModelElem_py,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        logger=None):
    """
    Converts an elementary covariance model 3D from python to C.

    Simulation grid geometry is specified in case of non-stationary covariance
    model.

    Parameters
    ----------
    covModelElem_py : 2-tuple
        elementary covariance model 3D in python, `covModelElem_py` = (t, d),
        with:

        * t : str
            type of elementary covariance model, can be

            - 'nugget'         (see function :func:`geone.covModel.cov_nug`)
            - 'spherical'      (see function :func:`geone.covModel.cov_sph`)
            - 'exponential'    (see function :func:`geone.covModel.cov_exp`)
            - 'gaussian'       (see function :func:`geone.covModel.cov_gau`)
            - 'triangular'     (see function :func:`geone.covModel.cov_tri`)
            - 'cubic'          (see function :func:`geone.covModel.cov_cub`)
            - 'sinus_cardinal' (see function :func:`geone.covModel.cov_sinc`)
            - 'gamma'          (see function :func:`geone.covModel.cov_gamma`)
            - 'power'          (see function :func:`geone.covModel.cov_pow`)
            - 'exponential_generalized' (see function :func:`geone.covModel.cov_exp_gen`)
            - 'matern'         (see function :func:`geone.covModel.cov_matern`)

        * d : dict
            dictionary of required parameters to be passed to the elementary
            model `t` (value can be a "single value" or an array that matches
            the dimension of the simulation grid (for non-stationary
            covariance model)

        e.g.

        - (t, d) = ('spherical', {'w':2.0, 'r':[1.5, 2.5, 3.0]})
        - (t, d) = ('power', {'w':2.0, 'r':[1.5, 2.5, 3.0], 's':1.7})
        - (t, d) = ('matern', {'w':2.0, 'r':[1.5, 2.5, 3.0], 'nu':1.5})

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
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    covModelElem_c : \\(MPDS_COVMODELELEM \\*\\)
        elementary covariance model in C
    """
    fname = 'covModel3Delem_py2C'

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
    elif covModelElem_py[0] == 'triangular':
        covModelElem_c.covModelType = geosclassic.COV_TRIANGULAR
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
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (3D) from python to C ('w' not compatible with simulation grid)"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param,
                     logger=logger)
            try:
                covModelElem_c.weightImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (3D) from python to C ('w')"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

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
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (3D) from python to C ('r(x)' not compatible with simulation grid)"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param,
                     logger=logger)
            try:
                covModelElem_c.rxImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (3D) from python to C ('r(x)')"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

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
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (3D) from python to C ('r(y)' not compatible with simulation grid)"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param,
                     logger=logger)
            try:
                covModelElem_c.ryImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (3D) from python to C ('r(y)')"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

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
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (3D) from python to C ('r(z)' not compatible with simulation grid)"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param,
                     logger=logger)
            try:
                covModelElem_c.rzImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (3D) from python to C ('r(z)')"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

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
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (3D) from python to C ('s' not compatible with simulation grid)"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            im = Img(nx=nx, ny=ny, nz=nz,
                     sx=sx, sy=sy, sz=sz,
                     ox=ox, oy=oy, oz=oz,
                     nv=1, val=param,
                     logger=logger)
            try:
                covModelElem_c.sImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeCovModelElem(covModelElem_c)
                geosclassic.free_MPDS_COVMODELELEM(covModelElem_c)
                # Raise error
                err_msg = f"{fname}: cannot convert covModelElem (3D) from python to C ('s')"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

    return covModelElem_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel1D_py2C(
        covModel_py,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        logger=None):
    """
    Converts a covariance model 1D from python to C.

    Simulation grid geometry is specified in case of non-stationary covariance
    model.

    Parameters
    ----------
    covModel_py : :class:`geone.covModel.CovModel1D`
        covariance model 1D in python

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
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    covModel_c : \\(MPDS_COVMODEL \\*\\)
        covariance model in C
    """
    fname = 'covModel1D_py2C'

    covModel_c = geosclassic.malloc_MPDS_COVMODEL()
    geosclassic.MPDSGeosClassicInitCovModel(covModel_c)

    n = len(covModel_py.elem)
    covModel_c.nelem = n
    covModel_c.covModelElem = geosclassic.new_MPDS_COVMODELELEM_array(n)
    for i, covModelElem in enumerate(covModel_py.elem):
        try:
            covModelElem_c = covModel1Delem_py2C(covModelElem, nx, ny, nz, sx, sy, sz, ox, oy, oz)
        except Exception as exc:
            # Free memory on C side
            geosclassic.MPDSGeosClassicFreeCovModel(covModel_c)
            geosclassic.free_MPDS_COVMODEL(covModel_c)
            # Raise error
            err_msg = f'{fname}: cannot convert covModel1D from python to C'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

        geosclassic.MPDS_COVMODELELEM_array_setitem(covModel_c.covModelElem, i, covModelElem_c)

    # covModel_c.angle1, covModel_c.angle2, covModel_c.angle3: keep to 0.0
    covModel_c.angle1 = 0.0
    covModel_c.angle2 = 0.0
    covModel_c.angle3 = 0.0

    return covModel_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel2D_py2C(
        covModel_py,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        logger=None):
    """
    Converts a covariance model 2D from python to C.

    Simulation grid geometry is specified in case of non-stationary covariance
    model.

    Parameters
    ----------
    covModel_py : :class:`geone.covModel.CovModel2D`
        covariance model 2D in python

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
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    covModel_c : \\(MPDS_COVMODEL \\*\\)
        covariance model in C
    """
    fname = 'covModel2D_py2C'

    covModel_c = geosclassic.malloc_MPDS_COVMODEL()
    geosclassic.MPDSGeosClassicInitCovModel(covModel_c)

    n = len(covModel_py.elem)
    covModel_c.nelem = n
    covModel_c.covModelElem = geosclassic.new_MPDS_COVMODELELEM_array(n)
    for i, covModelElem in enumerate(covModel_py.elem):
        try:
            covModelElem_c = covModel2Delem_py2C(covModelElem, nx, ny, nz, sx, sy, sz, ox, oy, oz)
        except Exception as exc:
            # Free memory on C side
            geosclassic.MPDSGeosClassicFreeCovModel(covModel_c)
            geosclassic.free_MPDS_COVMODEL(covModel_c)
            # Raise error
            err_msg = f'{fname}: cannot convert covModel2D from python to C'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

        geosclassic.MPDS_COVMODELELEM_array_setitem(covModel_c.covModelElem, i, covModelElem_c)

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
            # Free memory on C side
            geosclassic.MPDSGeosClassicFreeCovModel(covModel_c)
            geosclassic.free_MPDS_COVMODEL(covModel_c)
            # Raise error
            err_msg = f"{fname}: cannot convert covModel2D from python to C ('alpha' not compatible with simulation grid)"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=1, val=param,
                 logger=logger)
        try:
            covModel_c.angle1Image = img_py2C(im, logger=logger)
        except Exception as exc:
            # Free memory on C side
            geosclassic.MPDSGeosClassicFreeCovModel(covModel_c)
            geosclassic.free_MPDS_COVMODEL(covModel_c)
            # Raise error
            err_msg = f"{fname}: cannot convert covModel2D from python to C ('alpha')"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

    # angle2
    covModel_c.angle2 = 0.0

    # angle3
    covModel_c.angle3 = 0.0

    return covModel_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel3D_py2C(
        covModel_py,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        logger=None):
    """
    Converts a covariance model 3D from python to C.

    Simulation grid geometry is specified in case of non-stationary covariance
    model.

    Parameters
    ----------
    covModel_py : :class:`geone.covModel.CovModel3D`
        covariance model 3D in python

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
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    covModel_c : \\(MPDS_COVMODEL \\*\\)
        covariance model in C
    """
    fname = 'covModel3D_py2C'

    covModel_c = geosclassic.malloc_MPDS_COVMODEL()
    geosclassic.MPDSGeosClassicInitCovModel(covModel_c)

    n = len(covModel_py.elem)
    covModel_c.nelem = n
    covModel_c.covModelElem = geosclassic.new_MPDS_COVMODELELEM_array(n)
    for i, covModelElem in enumerate(covModel_py.elem):
        try:
            covModelElem_c = covModel3Delem_py2C(covModelElem, nx, ny, nz, sx, sy, sz, ox, oy, oz)
        except Exception as exc:
            # Free memory on C side
            geosclassic.MPDSGeosClassicFreeCovModel(covModel_c)
            geosclassic.free_MPDS_COVMODEL(covModel_c)
            # Raise error
            err_msg = f'{fname}: cannot convert covModel3D from python to C'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

        geosclassic.MPDS_COVMODELELEM_array_setitem(covModel_c.covModelElem, i, covModelElem_c)

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
            # Free memory on C side
            geosclassic.MPDSGeosClassicFreeCovModel(covModel_c)
            geosclassic.free_MPDS_COVMODEL(covModel_c)
            # Raise error
            err_msg = f"{fname}: cannot convert covModel3D from python to C ('alpha' not compatible with simulation grid)"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=1, val=param,
                 logger=logger)
        try:
            covModel_c.angle1Image = img_py2C(im, logger=logger)
        except Exception as exc:
            # Free memory on C side
            geosclassic.MPDSGeosClassicFreeCovModel(covModel_c)
            geosclassic.free_MPDS_COVMODEL(covModel_c)
            # Raise error
            err_msg = f"{fname}: cannot convert covModel3D from python to C ('alpha')"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

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
            # Free memory on C side
            geosclassic.MPDSGeosClassicFreeCovModel(covModel_c)
            geosclassic.free_MPDS_COVMODEL(covModel_c)
            # Raise error
            err_msg = f"{fname}: cannot convert covModel3D from python to C ('beta' not compatible with simulation grid)"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=1, val=param,
                 logger=logger)
        try:
            covModel_c.angle2Image = img_py2C(im, logger=logger)
        except Exception as exc:
            # Free memory on C side
            geosclassic.MPDSGeosClassicFreeCovModel(covModel_c)
            geosclassic.free_MPDS_COVMODEL(covModel_c)
            # Raise error
            err_msg = f"{fname}: cannot convert covModel3D from python to C ('beta')"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

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
            # Free memory on C side
            geosclassic.MPDSGeosClassicFreeCovModel(covModel_c)
            geosclassic.free_MPDS_COVMODEL(covModel_c)
            # Raise error
            err_msg = f"{fname}: cannot convert covModel3D from python to C ('gamma' not compatible with simulation grid)"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=1, val=param,
                 logger=logger)
        try:
            covModel_c.angle3Image = img_py2C(im, logger=logger)
        except Exception as exc:
            # Free memory on C side
            geosclassic.MPDSGeosClassicFreeCovModel(covModel_c)
            geosclassic.free_MPDS_COVMODEL(covModel_c)
            # Raise error
            err_msg = f"{fname}: cannot convert covModel3D from python to C ('gamma')"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

    return covModel_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor, logger=None):
    """
    Converts geosclassic output from C to python.

    Parameters
    ----------
    mpds_geosClassicOutput : \\(MPDS_GEOSCLASSICOUTPUT \\*\\)
        geosclassic output in C

    mpds_progressMonitor : \\(MPDS_PROGRESSMONITOR \\*\\)
        progress monitor in C

    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    geosclassic_output : dict
        geosclassic output in python, dictionary

        {'image':image, 'nwarning':nwarning, 'warnings':warnings}

        with:

        - image : :class:`geone.img.Img`
            output image, with `image.nv` variables (output variables:
            simulations or estimates and standard deviations);
            note: `image=None` if `mpds_geosClassicOutput->outputImage=NULL`
        - nwarning : int
            total number of warning(s) encountered (same warnings can be counted
            several times)
        - warnings : list of strs
            list of distinct warnings encountered (can be empty)
    """
    # fname = 'geosclassic_output_C2py'

    # Initialization
    image = None
    nwarning, warnings = None, None

    image = img_C2py(mpds_geosClassicOutput.outputImage, logger=logger)

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
        searchRadius,
        searchRadiusRelative,
        nneighborMax,
        searchNeighborhoodSortMode,
        nGibbsSamplerPathMin,
        nGibbsSamplerPathMax,
        seed,
        nreal,
        logger=None):
    """
    Fills a mpds_geosClassicInput C structure from given parameters.

    This function should not be called directly, it is used in other functions
    of this module.

    Parameters
    ----------
    space_dim : int
        space dimension (1, 2, or 3)

    cov_model : :class:`geone.covModel.CovModel<d>D`
        covariance model

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

    varname : str
        variable name

    outputReportFile : str, optional
        name of the report file (if desired in output); by default (`None`): no
        report file

    computationMode : int
        computation mode:

        - `computationMode=0`: estimation, ordinary kriging
        - `computationMode=1`: estimation, simple kriging
        - `computationMode=2`: simulation, ordinary kriging
        - `computationMode=3`: simulation, simple kriging

    dataImage : sequence of :class:`geone.img.Img`, or `None`
        list of data image(s)

    dataPointSet : sequence of :class:`geone.img.PointSet`, or `None`
        list of data point set(s)

    mask : array-like, or `None`
        mask value in grid cells

    mean : float, or array-like, or `None`
        mean value in grid cells

    var : float, or array-like, or `None`
        variance value in grid cells

    searchRadius : float
        searchRadius parameter

    searchRadiusRelative : float
        searchRadiusRelative parameter

    nneighborMax : int
        nneighborMax parameter

    searchNeighborhoodSortMode : int
        searchNeighborhoodSortMode parameter

    nGibbsSamplerPathMin : int
        nGibbsSamplerPathMin parameter

    nGibbsSamplerPathMax : int
        nGibbsSamplerPathMax parameter

    seed : int
        seed parameter

    nreal : int
        nreal parameter
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    mpds_geosClassicInput : \\(MPDS_GEOSCLASSICINPUT \\*\\)
        geosclassic input in C, intended for "GeosClassicSim" C program
    """
    fname = 'fill_mpds_geosClassicInput'

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
    try:
        if space_dim==1:
            cov_model_c = covModel1D_py2C(cov_model, nx, ny, nz, sx, sy, sz, ox, oy, oz, logger=logger)
        elif space_dim==2:
            if isinstance(cov_model, gcm.CovModel1D):
                cov_model = gcm.covModel1D_to_covModel2D(cov_model) # convert model 1D in 2D
                # -> will not be modified cov_model at exit
            cov_model_c = covModel2D_py2C(cov_model, nx, ny, nz, sx, sy, sz, ox, oy, oz, logger=logger)
        elif space_dim==3:
            if isinstance(cov_model, gcm.CovModel1D):
                cov_model = gcm.covModel1D_to_covModel3D(cov_model) # convert model 1D in 3D
                # -> will not be modified cov_model at exit
            cov_model_c = covModel3D_py2C(cov_model, nx, ny, nz, sx, sy, sz, ox, oy, oz, logger=logger)

    except Exception as exc:
        # Free memory on C side
        geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
        geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)
        # Raise error
        err_msg = f'{fname}: cannot fill mpds_geosClassicInput C structure (covModel)'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg) from exc

    mpds_geosClassicInput.covModel = cov_model_c

    # mpds_geosClassicInput.searchRadius
    mpds_geosClassicInput.searchRadius = float(searchRadius)

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
            try:
                im_c = img_py2C(dataIm, logger=logger)
            except Exception as exc:
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
                geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)
                # Raise error
                err_msg = f'{fname}: cannot fill mpds_geosClassicInput C structure (dataImage)'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

            geosclassic.MPDS_IMAGE_array_setitem(mpds_geosClassicInput.dataImage, i, im_c)
            geosclassic.free_MPDS_IMAGE(im_c)

    # mpds_geosClassicInput.ndataPointSet and mpds_geosClassicInput.dataPointSet
    if dataPointSet is None:
        mpds_geosClassicInput.ndataPointSet = 0
    else:
        dataPointSet = np.asarray(dataPointSet).reshape(-1)
        n = len(dataPointSet)
        mpds_geosClassicInput.ndataPointSet = n
        mpds_geosClassicInput.dataPointSet = geosclassic.new_MPDS_POINTSET_array(n)
        for i, dataPS in enumerate(dataPointSet):
            try:
                ps_c = ps_py2C(dataPS, logger=logger)
            except Exception as exc:
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
                geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)
                # Raise error
                err_msg = f'{fname}: cannot fill mpds_geosClassicInput C structure (dataPointSet)'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

            geosclassic.MPDS_POINTSET_array_setitem(mpds_geosClassicInput.dataPointSet, i, ps_c)

    # mpds_geosClassicInput.maskImageFlag and mpds_geosClassicInput.maskImage
    if mask is None:
        mpds_geosClassicInput.maskImageFlag = geosclassic.FALSE
    else:
        mpds_geosClassicInput.maskImageFlag = geosclassic.TRUE
        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=1, val=mask,
                 logger=logger)
        try:
            mpds_geosClassicInput.maskImage = img_py2C(im, logger=logger)
        except Exception as exc:
            # Free memory on C side
            geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
            geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)
            # Raise error
            err_msg = f'{fname}: cannot fill mpds_geosClassicInput C structure (mask)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

    # mpds_geosClassicInput.meanUsage, mpds_geosClassicInput.meanValue, mpds_geosClassicInput.meanImage
    if mean is None:
        mpds_geosClassicInput.meanUsage = 0
    else:
        mean = np.atleast_1d(mean)
        if mean.size == 1:
            mpds_geosClassicInput.meanUsage = 1
            mpds_geosClassicInput.meanValue = float(mean[0])

        elif mean.size == nxyz:
            mpds_geosClassicInput.meanUsage = 2
            im = Img(nx=nx, ny=ny, nz=nz,
                    sx=sx, sy=sy, sz=sz,
                    ox=ox, oy=oy, oz=oz,
                    nv=1, val=mean,
                    logger=logger)
            try:
                mpds_geosClassicInput.meanImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
                geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)
                # Raise error
                err_msg = f'{fname}: cannot fill mpds_geosClassicInput C structure (meanImage)'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

        else:
            # Free memory on C side
            geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
            geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)
            # Raise error
            err_msg = f'{fname}: cannot fill mpds_geosClassicInput C structure (`mean` not compatible with simulation grid)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # mpds_geosClassicInput.varianceUsage, mpds_geosClassicInput.varianceValue, mpds_geosClassicInput.varianceImage
    if var is None:
        mpds_geosClassicInput.varianceUsage = 0
    else:
        var = np.atleast_1d(var)
        if var.size == 1:
            mpds_geosClassicInput.varianceUsage = 1
            mpds_geosClassicInput.varianceValue = var[0]
        
        elif var.size == nxyz:
            mpds_geosClassicInput.varianceUsage = 2
            im = Img(nx=nx, ny=ny, nz=nz,
                    sx=sx, sy=sy, sz=sz,
                    ox=ox, oy=oy, oz=oz,
                    nv=1, val=var,
                    logger=logger)
            try:
                mpds_geosClassicInput.varianceImage = img_py2C(im, logger=logger)
            except Exception as exc:
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
                geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)
                # Raise error
                err_msg = f'{fname}: cannot fill mpds_geosClassicInput C structure (varianceImage)'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

        else:
            # Free memory on C side
            geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
            geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)
            # Raise error
            err_msg = f'{fname}: cannot fill mpds_geosClassicInput C structure (`var` not compatible with simulation grid)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # mpds_geosClassicInput.nGibbsSamplerPathMin
    mpds_geosClassicInput.nGibbsSamplerPathMin = int(nGibbsSamplerPathMin)

    # mpds_geosClassicInput.nGibbsSamplerPathMax
    mpds_geosClassicInput.nGibbsSamplerPathMax = int(nGibbsSamplerPathMax)

    # mpds_geosClassicInput.seed
    if seed is None:
        seed = np.random.randint(1, 1000000)
    mpds_geosClassicInput.seed = int(seed)

    # mpds_geosClassicInput.seedIncrement
    mpds_geosClassicInput.seedIncrement = 1

    # mpds_geosClassicInput.nrealization
    mpds_geosClassicInput.nrealization = int(nreal)

    return mpds_geosClassicInput
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def run_MPDSOMPGeosClassicSim(
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
        searchRadius,
        searchRadiusRelative,
        nneighborMax,
        searchNeighborhoodSortMode,
        nGibbsSamplerPathMin,
        nGibbsSamplerPathMax,
        seed,
        nreal,
        nthreads,
        progress_monitor=0,
        pid=None,
        verbose=0,
        logger=None):
    """
    Launches external C function `MPDSOMPGeosClassicSim` (Geos-Classic library).

    This function should not be called directly, it is used in other functions
    of this module.

    All the parameters (except `nthreads`, `progress_monitor`, `pid`, `verbose` and `logger`) 
    are the parameters for the input structure `mpds_geosClassicInput` (C); 
    `nthreads` is the number of threads used by the function `MPDSOMPGeosClassicSim`; 
    `progress_monitor` is an integer specifying the function used to update the progress 
    monitor (0 : no output, 1: warning only, 4: more detailed); 
    `pid` is the process id of the caller (int, used with multiprocessing);
    `verbose` is the verbosity (int).

    The function returns a dictionary `geosclassic_output` with:
    - `geosclassic_output['image']` : (class:`geone.img.Img`) output image
    - `geosclassic_output['nwarning']` : (int) total number of warnings encountered
    - `geosclassic_output['warnings']` : (list of strs) list of distinct warnings encountered
    """
    fname = 'run_MPDSOMPGeosClassicSim'
    if pid is not None:
        fname = f'{fname} [pid={pid}]'

    if verbose > 2:
        if logger:
            logger.info(f'{fname}: Filling C structure for Geos-Classic program')
        else:
            print(f'{fname}: Filling C structure for Geos-Classic program')

    # --- Fill mpds_geosClassicInput structure (C)
    try:
        mpds_geosClassicInput = fill_mpds_geosClassicInput(
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
                searchRadius,
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                nGibbsSamplerPathMin,
                nGibbsSamplerPathMax,
                seed,
                nreal,
                logger=logger)

    except Exception as exc:
        err_msg = f'{fname}: cannot fill mpds_geosClassicInput C structure'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg) from exc

    # --- Prepare mpds_geosClassicIOutput structure (C)
    # Allocate mpds_geosClassicOutput
    mpds_geosClassicOutput = geosclassic.malloc_MPDS_GEOSCLASSICOUTPUT()

    # Init mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicInitGeosClassicOutput(mpds_geosClassicOutput)

    # --- Set progress monitor
    mpds_progressMonitor = geosclassic.malloc_MPDS_PROGRESSMONITOR()
    geosclassic.MPDSInitProgressMonitor(mpds_progressMonitor)

    # Set function to update progress monitor
    # note: according to geosclassic.MPDS_SHOW_PROGRESS_MONITOR set to 4 for compilation of 
    #   py module, the function
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor1_ptr: warning only
    if progress_monitor == 0:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    elif progress_monitor == 1:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor1_ptr
    # elif progress_monitor == 2:
    #     mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor2_ptr
    # elif progress_monitor == 3:
    #     mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor3_ptr
    elif progress_monitor == 4:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr
    else:
        err_msg = f'{fname}: `progress_monitor` not valid (should be 0, 1, or 4)'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    if verbose > 1:
        if logger:
            logger.info(  
                f'{fname}: Geos-Classic running... [' + \
                f'VERSION {geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER:s} / ' + \
                f'BUILD NUMBER {geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER:s} / ' + \
                f'OpenMP {nthreads:d} thread(s)]')
        else:
            print(  
                f'{fname}: Geos-Classic running... [' + \
                f'VERSION {geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER:s} / ' + \
                f'BUILD NUMBER {geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER:s} / ' + \
                f'OpenMP {nthreads:d} thread(s)]')
            sys.stdout.flush()
            sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nthreads)

    # Free memory on C side: mpds_geosClassicInput
    geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
    geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)

    if err:
        # Free memory on C side: mpds_geosClassicOutput
        geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
        geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)
        # Free memory on C side: mpds_progressMonitor
        geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)
        # Raise error
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        err_msg = f'{fname}: {err_message}'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor, logger=logger)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose > 1 and geosclassic_output:
        if logger:
            logger.info(f'{fname}: Geos-Classic run complete')
        else:
            print(f'{fname}: Geos-Classic run complete')

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def _run_krige_and_MPDSOMPGeosClassicSim(
        ir0, ir1,                   # range of realizations (indices)
        v_ineq_sim,                 # sim values (at ineq data locations)
        x_new, v, x_cc, cov_model,  # krige parameters ...
        v_new_err_std,
        method,
        mean_x_new,
        mean_x_cc,
        var_x_new,
        var_x_cc,
        alpha_x_cc,
        beta_x_cc,
        gamma_x_cc,
        cov_model_non_stationarity_x_cc_list,
        use_unique_neighborhood,
        searchRadius,
        searchRadiusRelative,
        nneighborMax,
        space_dim,            # MPDSOMPGeosClassicSim parameters ... 
        cov_model_grid,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        varname,
        outputReportFile,
        computationMode,
        dataImage,
        dataPointSet,
        mask,
        mean_grid,
        var_grid,
        searchRadius_geosClassic,
        searchRadiusRelative_geosClassic,
        nneighborMax_geosClassic,
        searchNeighborhoodSortMode,
        nGibbsSamplerPathMin,
        nGibbsSamplerPathMax,
        seed,
        nthreads,            # for MPDSOMPGeosClassicSim
        progress_monitor=0,
        pid=None,
        verbose=0,
        logger=None):
    """
    Launches `geone.covModel.krige`, then external C function `MPDSOMPGeosClassicSim` (Geos-Classic library).

    This function should not be called directly, it is used in other functions
    of this module - ad-hoc function for internal use!

    All the parameters (except `ir0`, `ir1`, `v_ineq_sim`, `nthreads`, 
    `progress_monitor`, `pid`, `verbose` and `logger`) are the parameters for the function 
    `geone.covModel.krige` or for the input structure `mpds_geosClassicInput` (C); 
    `ir0` and `ir1` define the range of realization (indices) to be done;
    `v_ineq_sim` is the array of all simulations at ineq data locations;
    `nthreads` is the number of threads used by the function `MPDSOMPGeosClassicSim`; 
    `progress_monitor` is an integer specifying the function used to update the progress 
    monitor (0 : no output, 1: warning only, 4: more detailed); 
    `pid` is the process id of the caller (int, used with multiprocessing);
    `verbose` is the verbosity (int).

    The function returns a dictionary `geosclassic_output` with:
    - `geosclassic_output['image']` : (class:`geone.img.Img`) output image
    - `geosclassic_output['nwarning']` : (int) total number of warnings encountered
    - `geosclassic_output['warnings']` : (list of strs) list of distinct warnings encountered
    """
    fname = '_run_krige_and_MPDSOMPGeosClassicSim'
    if pid is not None:
        fname = f'{fname} [pid={pid}]'

    outputReportFile_ir = None # default

    image = img.Img(nx=nx, ny=ny, nz=nz, sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz, nv=ir1-ir0, val=np.nan, logger=logger)

    nwarning = 0
    warnings = []

    for ir in range(ir0, ir1):
        if verbose > 0:
            if logger:
                logger.info(f"{fname}: realization {ir:5d} of {ir1-1}...")
            else:
                print(f"{fname}: realization {ir:5d} of {ir1-1}...")

        # 2b. New dataset for realization `ir`
        v_new = np.hstack((v, v_ineq_sim[ir]))
        
        # 3b. Kriging new dataset for realization `ir` at the center of grid cells containing at least one data point
        # Do kriging at cell centers x_cc
        if verbose > 1:
            if logger:
                logger.info(f"{fname}: real {ir:5d}: do kriging at the center of grid cells containing at least one data point...")
            else:
                print(f"{fname}: real {ir:5d}: do kriging at the center of grid cells containing at least one data point...")

        v_cc, v_cc_err_std = gcm.krige(
                                x_new, v_new, x_cc, cov_model,
                                v_err_std=v_new_err_std,
                                method=method,
                                mean_x=mean_x_new,
                                mean_xu=mean_x_cc,
                                var_x=var_x_new,
                                var_xu=var_x_cc,
                                alpha_xu=alpha_x_cc,
                                beta_xu=beta_x_cc,
                                gamma_xu=gamma_x_cc,
                                cov_model_non_stationarity_xu_list=cov_model_non_stationarity_x_cc_list,
                                use_unique_neighborhood=use_unique_neighborhood,
                                searchRadius=searchRadius,
                                searchRadiusRelative=searchRadiusRelative,
                                nneighborMax=nneighborMax,
                                pid=pid,
                                verbose=(verbose>2),
                                logger=logger)

        # Update data point set: set values and error std in point set
        dataPointSet.val[3] = v_cc
        dataPointSet.val[4] = v_cc_err_std

        # 4b. Do 1 SGS realization (realization `ir`) on the grid (at cell centers) using data points at cell centers
        if verbose > 1:
            if logger:
                logger.info(f"{fname}: real {ir:5d}: do sgs on the grid (at cell centers) using data points at cell centers...")
            else:
                print(f"{fname}: real {ir:5d}: do sgs on the grid (at cell centers) using data points at cell centers...")

        if outputReportFile is not None:
            outputReportFile_ir = outputReportFile 
            if pid is not None:
                outputReportFile_ir = outputReportFile_ir + f'.{pid}'

        if verbose >= 3:
            progress_monitor = 4
        else:
            progress_monitor = 0
    
        args = (
                space_dim,
                cov_model_grid,
                nx, ny, nz,
                sx, sy, sz,
                ox, oy, oz,
                varname,
                outputReportFile_ir,
                computationMode,
                dataImage,
                dataPointSet,
                mask,
                mean_grid,
                var_grid,
                searchRadius_geosClassic,
                searchRadiusRelative_geosClassic,
                nneighborMax_geosClassic,
                searchNeighborhoodSortMode,
                nGibbsSamplerPathMin,
                nGibbsSamplerPathMax,
                seed + ir, # seed for realization index ir
                1,         # one real
                nthreads,
            )
        kwds = {
                'progress_monitor':progress_monitor, 
                'pid':pid,
                'verbose':(verbose>3), # 0 # max(0, verbose-1)
                'logger':logger
            }
        geosclassic_output = run_MPDSOMPGeosClassicSim(*args, **kwds)
        
        image.val[ir-ir0] = geosclassic_output['image'].val[0]
        nwarning = nwarning + int(geosclassic_output['nwarning'])
        warnings.extend(geosclassic_output['warnings'])

    warnings = list(np.unique(warnings))

    geosclassic_output = {'image':image, 'nwarning':nwarning, 'warnings':warnings}

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def simulate(
        cov_model,
        dimension, 
        spacing=None, 
        origin=None,
        method='ordinary_kriging',
        x=None, 
        v=None, 
        v_err_std=0.0,
        x_ineq=None, 
        v_ineq_min=None, 
        v_ineq_max=None,
        mean=None, 
        var=None,
        alpha=None,
        beta=None,
        gamma=None,
        cov_model_non_stationarity_list=None,
        preprocess_data_and_ineq_in_grid=True,
        preprocess_ineq_less_constrained=True,
        mode_transform_ineq_to_data=False,
        transform_ineq_to_data_with_err_nsim=100,
        transform_ineq_to_data_with_err_p=0.95,
        use_unique_neighborhood=False,
        mask=None,
        add_data_point_to_mask=True,
        searchRadius=None,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        searchNeighborhoodSortMode=1,
        nGibbsSamplerPath=50,
        seed=None,
        nreal=1,
        outputReportFile=None,
        nproc=-1,
        nthreads_per_proc=-1,
        nproc_sgs_at_ineq=None,
        treat_image_one_by_one=False,
        verbose=2,
        logger=None):
    """
    Generates simulations (Sequential Gaussian Simulation, SGS) in a grid.

    A simulation takes place in (center of) grid cells, based on simple or
    ordinary kriging. The space dimension (1, 2, or 3) is detected (from the 
    parameter `dimension`).

    The simulations can be conditioned by
    
    - data: location: `x`, value: `v`, and errors standard deviation: `v_err_std`
    - inequality data: location: `x_ineq`, lower bounds: `v_ineq_min`, \
    and upper bounds `v_ineq_max` (no error)

    Note that if `v_err_std > 0.0`, the data are considered as data with error, the
    error being a zero-mean Gaussian with standard deviation `v_err_std`.

    Before starting the simulations, the data points and inequality data points can
    be first pre-processed (optional, if `preprocess_data_and_ineq_in_grid=True`) in
    order to have at most one data point or inequality data point per grid cell, by 
    proceeding as follows:
    
    - if one grid cell contains both data point(s) and inequality data point(s), then \
    the inequality data point(s) are removed
    - if one grid cell contains several data points (no inequality data), they \
    are aggregated in that cell into one data point by taking the mean position, the \
    mean value, and the maximal error standard deviation of data points in that cell
    - if one grid cell contains several inequality data points (no equality data) they \
    are aggregated in that cell into one data point by taking the mean position, and \
    if `preprocess_ineq_less_constrained=True`: the minimal lower bound (from `v_ineq_min`), \
    and the maximal upper bound (from `v_ineq_max`), or \
    if `preprocess_ineq_less_constrained=False`: the maximal lower bound (from `v_ineq_min`), \
    and the minimal upper bound (from `v_ineq_max`) \
    i.e. the less (resp. more) contraining bounds are kept if \
    `preprocess_ineq_less_constrained=True` (resp. `preprocess_ineq_less_constrained=False`); \
    note that the minimum or maximum is taken over non-infinite bounds (if any)

    Two simulation modes are available:

    Mode A: `mode_transform_ineq_to_data=True`
    ------------------------------------------

    1a. If there is inequality data, they are first transformed to new data with 
    error: for that, `transform_ineq_to_data_with_err_nsim` simulations at 
    inequality data points are generated, then at each location, the ensemble of values
    is transformed to a new data value with an error standard deviation using the 
    function :func:`geone.covModel.values_to_mean_and_err_std`

    2a. A new dataset is formed by grouping the original data (`x`, `v`, with the 
    error std `v_err_std`) and the new data with their own error std from step 1a.

    3a. The dataset from step 2a is then kriged at the center of grid cells containing 
    at least one data point, this results in a dataset at cell centers (values and error 
    std are respectively the kriging estimates and the kriging std)

    4a. The dataset at cell centers from step 3a is used to generate 
    `nreal` SGS realizations on the grid (at cell centers), 
    using an external C function (Geos-Classic library)

    Mode B: `mode_transform_ineq_to_data=False`
    -------------------------------------------

    1b. If there is inequality data, `nreal` simulations at inequality data points are 
    done, each simulation being new data (without error)

    2b. For each simulation of step 1b: a new dataset is formed by grouping the original 
    data (`x`, `v`, with the error std  `v_err_std`) and the new data of that simulation

    3b. Each dataset from step 2b is kriged at the center of grid cells containing at least 
    one data point, this results in a dataset at cell centers (values and error std are 
    respectively the kriging estimates and the kriging std)

    4b. Each dataset at cell centers from step 3b (`nreal` at total) is used to generate 
    1 SGS realization on the grid (at cell centers), using an external C function 
    (Geos-Classic library)
    
    Multiprocessing
    ---------------
    Multiprocessing (parallel processes) may be enabled in the following steps:
    
    - step 1a/1b above, the simulations are done on `nproc_sgs_at_ineq` processes \
    (python) using each one thread; if `nproc_sgs_at_ineq > 1` multiprocessing is used
    - step 4a/4b above, the simulations are done using an external C function (Geos-Classic \
    library) on `nproc` processes using each `nthreads_per_proc`; \
    if `nproc > 1` multiprocessing is used
    
    Parameters
    ----------
    cov_model : :class:`geone.covModel.CovModel<d>D`
        covariance model in 1D or 2D or 3D;      
        note: the covariance model must be stationary, however, non stationarity is
        handled: 

        - local rotation by specifying `alpha` (in 2D or 3D), `beta` (in 3D), `gamma` (in 3D)
        - other non-stationarities by specifying `cov_model_non_stationarity_list` (see below)

    dimension : [sequence of] int(s)
        number of cells along each axis, for simulation in:

        - 1D: `dimension=nx`
        - 2D: `dimension=(nx, ny)`
        - 3D: `dimension=(nx, ny, nz)`

        note : this parameter determines the space dimension (1, 2, or 3)

    spacing : [sequence of] float(s), optional
        cell size along each axis, for simulation in:

        - 1D: `spacing=sx`
        - 2D: `spacing=(sx, sy)`
        - 3D: `spacing=(sx, sy, sz)`

        by default (`None`): 1.0 along each axis

    origin : [sequence of] float(s), optional
        origin of the grid ("corner of the first cell"), for simulation in:

        - 1D: `origin=ox`
        - 2D: `origin=(ox, oy)`
        - 3D: `origin=(ox, oy, oz)`

        by default (`None`): 0.0 along each axis

    method : str {'simple_kriging', 'ordinary_kriging'}, default: 'ordinary_kriging'
        type of kriging

    x : array-like of floats, optional
        data points locations (float coordinates), array of shape `(n, d)`, 
        where `n` is the number of data points and `d` the spce dimension;
        note: `x` is (cast in array and) reshaped according to space dimension
        (i.e. other shape in input can be handled)

    v : 1D array-like of floats of shape (n,), optional
        data values at `x` (`v[i]` is the data value at `x[i]`); 
        note: if one data point, a float is accepted

    v_err_std : 1D array of floats of shape (n,), or float, default: 0.0
        standard deviation of error at data points, array of same length as `v`;
        if `v_err_std` is a float, the same value is used for all data points; 
        this means that at location x[i], the data value is considered as in a Gaussian 
        distribution of mean `v[i]` and standard deviation `v_err_std[i]`

    x_ineq : array-like of floats, optional
        inequality data points locations (float coordinates), array of shape `(n, d)`,
        where `n` is the number of data points and `d` the spce dimension;
        note: `x_ineq` is (cast in array and) reshaped according to space dimension
        (i.e. other shape in input can be handled)

    v_ineq_min : 1D array of floats of shape (n_ineq,), or float, optional
        minimal value (lower bound) at `x_ineq` (`v_ineq_min[i]` is the lower bound
        at `x_ineq[i]`); 
        notes: 
        
        - if `v_ineq_min=None` (default), no minimal value is considered for any \
        inequality data point (even if `n_ineq > 0`);
        - `v_ineq_min[i]` set to `np.nan` or `-np.inf` means that there is \
        no minimal value for point `x_ineq[i]`
        - if one inequality data point, a float is accepted
        
    v_ineq_max : 1D array of floats of shape (n_ineq,), or float, optional
        maximal value (upper bound) at `x_ineq` (`v_ineq_max[i]` is the upper bound
        at `x_ineq[i]`); 
        notes: 
        
        - if `v_ineq_max=None` (default), no maximal value is considered for any \
        inequality data point (even if `n_ineq > 0`);
        - `v_ineq_max[i]` set to `np.nan` or `np.inf` means that there is \
        no maximal value for point `x_ineq[i]`
        - if one inequality data point, a float is accepted

    mean : function (callable), or array-like of floats, or float, optional
        kriging mean value:

        - if a function: function of `d` argument(s) where `d` is the space dimension \
        that returns a value for each location given by their coordinate(s) in argument
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), values at grid cells
        - if a float: same value at every grid cell
        - by default (`None`): the mean of data value (`v`) (0.0 if no data) is \
        considered at every grid cell

        note: if `method=ordinary_kriging`, parameter `mean` is ignored

    var : function (callable), or array-like of floats, or float, optional
        kriging variance value:

        - if a function: function of `d` argument(s) where `d` is the space dimension \
        that returns a value for each location given by their coordinate(s) in argument
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), values at grid cells
        - if a float: same value at every grid cell
        - by default (`None`): not used (use of covariance model only)

        note: if `method=ordinary_kriging`, parameter `var` is ignored

    alpha : function (callable), or array-like of floats, or float, optional
        azimuth angle in degrees (see :class:`geone.covModel.CovModel<d>D`):

        - if a function: function of `d` argument(s) where `d` is the space dimension \
        that returns a value for each location given by their coordinate(s) in argument
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), values at grid cells
        - if a float: same value at every grid cell
        - by default (`None`): not used (use of covariance model only)

        note: not used in 1D

    beta : function (callable), or array-like of floats, or float, optional
        dip angle in degrees (see :class:`geone.covModel.CovModel<d>D`):

        - if a function: function of `d` argument(s) where `d` is the space dimension \
        that returns a value for each location given by their coordinate(s) in argument
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), values at grid cells
        - if a float: same value at every grid cell
        - by default (`None`): not used (use of covariance model only)

        note: not used in 1D or 2D

    gamma : function (callable), or array-like of floats, or float, optional
        plunge angle in degrees (see :class:`geone.covModel.CovModel<d>D`):

        - if a function: function of `d` argument(s) where `d` is the space dimension \
        that returns a value for each location given by their coordinate(s) in argument
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), values at grid cells
        - if a float: same value at every grid cell
        - by default (`None`): not used (use of covariance model only)

        note: not used in 1D or 2D

    cov_model_non_stationarity_list : list, optional
        list to set non-stationarities in covariance model; each entry must be
        a tuple (or list) `cm_ns` of length 2 or 3 with:

        - `cm_ns[0]`: str: the name of the method of `cov_model` to be applied
        - `cm_ns[1]`: function (callable), or array-like of floats, or float: \
        used to set the main parameter passed to the method:
            - if a function: function of `d` argument(s) where `d` is the space dimension \
            that returns a value for each location given by their coordinate(s) in argument
            - if array-like: its size must be equal to the number of grid cells \
            (the array is reshaped if needed), values at grid cells
            - if a float: same value at every grid cell
        - `cm_ns[2]`: dict, optional: keyworkds arguments to be passed to the method

        Examples (where the parameter `arg` is set from `val`):
        
        - `('multiply_w', val)` will apply `cov_model.multiply_w(arg)`; \
        this multipies the weight contribution of every elementary contribution of the \
        covariance model
        - `('multiply_w', val, {'elem_ind':0})` will apply `cov_model.multiply_w(arg, elem_ind=0)`; \
        this multipies the weight contribution of the elementary contribution of index 0 of the \
        covariance model
        - `('multiply_r', val)` will apply `cov_model.multiply_r(arg)`; \
        this multipies the range in all direction of every elementary contribution of the \
        covariance model
        - `('multiply_r', val, {'r_ind':0})` will apply `cov_model.multiply_r(arg, r_ind=0)`; \
        this multipies the range in the first main direction (index 0) of every elementary \
        contribution of the covariance model
        - `('multiply_r', val, {'r_ind':0, 'elem_ind':0})` will apply `cov_model.multiply_r(arg, r_ind=0, elem_ind=0)`; \
        this multipies the range in the first main direction (index 0) of the elementary \
        contribution of index 0 of the covariance model
        
    preprocess_data_and_ineq_in_grid : bool, default: True
        if `True`: data points and inequality data points are first pre-processed
        in order to have at most one data point or inequality data point per grid cell 
        (see above)

    preprocess_ineq_less_constrained : bool, default: True
        the less (resp. more) contrained bounds are kept for aggregated inequality 
        data points if `preprocess_ineq_less_constrained` is `True` (resp. `False`);
        used only if preprocess_data_and_ineq_in_grid=True`

    mode_transform_ineq_to_data : bool, default: False
        if `True`: mode A, described in points 1a. to 4a. above is used;
        if `False`: mode B, described in points 1b. to 4b. above is used

    transform_ineq_to_data_with_err_nsim : int, default: 100
        number of simulations at inequality data points to transform them to
        new data with error (see above);
        used only if `mode_transform_ineq_to_data=True`
    
    transform_ineq_to_data_with_err_p : float, default: 0.95
        probability used to transform the ensemble of values at inequality
        data points to new data with error (see above);
        used only if `mode_transform_ineq_to_data=True`
    
    use_unique_neighborhood : bool, default: False
        indicates if a unique neighborhood is used (all data points are considered)
        to krige data at cell centers (step 3a/3b above)
    
    mask : array-like, optional
        mask value at grid cells (value 1 for simulated cells, value 0 for not
        simulated cells); the size of the array must be equal to the number of
        grid cells (the array is reshaped if needed)

    add_data_point_to_mask : bool, default: True
        - if `True`: any grid cell that contains a data point is added to (the \
        simulated part of) the mask (if present), i.e. mask value at those cells \
        are set to 1; at the end of the computation the "new mask cells" are \
        removed (by setting a missing value (`numpy.nan`) for the variable out of \
        the original mask)
        - if `False`: original mask is kept as given in input, and data point \
        falling out of (the simulated part of) the mask (if present) are ignored

    searchRadius : float, optional
        if specified, i.e. not `None`: radius of the search neighborhood (ellipsoid
        with same radii along each axis), i.e. the data points at distance to the 
        estimated point greater than `searchRadius` are not taken into account 
        in the kriging system; if `searchRadius` is not `None`, then 
        `searchRadiusRelative` is not used;
        by default (`searchRadius=None`): `searchRadiusRelative` is used to 
        define the search ellipsoid;
        
        note : parameter considered for 
        
        - simulation at inequality data points (step 1a/1b)
        - kriging to cell centers (step 3a/3b) unless `use_unique_neighborhood=True`
        - simulation on the grid (step 4a/4b)

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
        note (Geos-Classic library): if a range r_i is non-stationary over the grid, 
        its maximal value over the grid is considered;

        note : parameter considered for 
        
        - simulation at inequality data points (step 1a/1b)
        - kriging to cell centers (step 3a/3b) unless `use_unique_neighborhood=True`
        - simulation on the grid (step 4a/4b)
        
    nneighborMax : int, default: 12
        maximal number of neighbors (data points) taken into account in the
        kriging system, must be greater than or equal to 1; the data points the 
        closest to the simulated point are taken into account (see the parameters 
        `searchRadius`, `searchRadiusRelative`, `searchNeighborhoodSortMode`);

        note : parameter considered for 
        
        - simulation at inequality data points (step 1a/1b)
        - kriging to cell centers (step 3a/3b) unless `use_unique_neighborhood=True`
        - simulation on the grid (step 4a/4b)

    searchNeighborhoodSortMode : int, default: 1
        indicates how to sort the search neighboorhood cells (neighbors), for 
        external C function (Geos-Classic library); they are sorted in increasing 
        order according to:

        - `searchNeighborhoodSortMode=0`: distance in the usual axes system
        - `searchNeighborhoodSortMode=1`: distance in the axes sytem supporting \
        the covariance model and accounting for anisotropy given by the ranges \
        (if `searchRadius=None`)
        - `searchNeighborhoodSortMode=2`: minus the evaluation of the covariance \
        model

        Notes:

        - if the covariance model is non-stationary (see note for `cov_model`), then \
        `searchNeighborhoodSortMode=2` is not allowed
        - if the covariance model has any non-stationary range or non-stationary \
        angle and `searchNeighborhoodSortMode=1`: "maximal ranges" (adapted to \
        direction from the central cell) are used to compute distance for sorting \
        the neighbors

        note: parameter considered for simulation on the grid (step 4a/4b)

    nGibbsSamplerPath : int, default: 50
        number of Gibbs sampler paths for simulating values at inequality data points

    seed : int, optional
        seed for initializing random number generator

    nreal : int, default: 1
        number of realization(s)

    outputReportFile : str, optional
        name of the report file (if desired in output); by default (`None`): no
        report file

    nproc : int, default: -1
        number of process(es) for external C function (Geos-Classic library): 
        a negative number, -n <= 0, can be specified to use the total number of cpu(s) 
        of the system except n; `nproc` is finally at maximum equal to `nreal` but at 
        least 1 by applying:
        
        - if `nproc >= 1`, then `nproc = max(min(nproc, nreal), 1)` is used
        - if `nproc = -n <= 0`, then `nproc = max(min(nmax-n, nreal), 1)` is used, \
        where nmax is the total number of cpu(s) of the system (retrieved by \
        `multiprocessing.cpu_count()`)

        note: if `nproc=None`, `nproc=-1` is used

    nthreads_per_proc : int, default: -1
        number of thread(s) per process for external C function (Geos-Classic library);
        if `nthreads_per_proc = -n <= 0`: `nthreads_per_proc` is automatically 
        computed as the maximal integer (but at least 1) such that 
        `nproc*nthreads_per_proc <= nmax-n`, where nmax is the total number of cpu(s)
        of the system (retrieved by `multiprocessing.cpu_count()`); 
        
        note: if `nthreads_per_proc=None`, `nthreads_per_proc=-1` is used

    nproc_sgs_at_ineq : int, optional
        number of process(es) for sgs at inequality data points:

        - if `nproc_sgs_at_ineq = 1`: multiprocessing is not enabled and the function \
        `gcm.sgs_at_inequality_data_points` is used
        - if `nproc_sgs_at_ineq > 1`: multiprocessing is enabled and the function \
        `gcm.sgs_at_inequality_data_points_mp` with parameter `nproc=nproc_sgs_at_ineq` \
        is used

        by default (`None`): `nproc_sgs_at_ineq` is set to `nproc*nthreads_per_proc`

    treat_image_one_by_one : bool, default: False
        used only if multiprocessing is enabled  with `nproc > 1`;
        keyword argument passed to the function :func:`geone.img.gatherImages`:

        - if `True`: images (result of each process) are gathered one by one, \
        i.e. the variables of each image are inserted in an output image one by \
        one and removed from the source (slower, may save memory)
        - if `False`: images (result of each process) are gathered at once, \
        i.e. the variables of all images are inserted in an output image at once, \
        and then removed (faster)

    verbose : int, default: 2
        verbose mode, higher implies more printing (info):

        - 0: no display
        - 1: warnings
        - 2: warnings + basic info
        - 3 (or >2): all information

        note that if an error occurred, it is raised
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    geosclassic_output : dict
        geosclassic output in python, dictionary

        {'image':image, 'nwarning':nwarning, 'warnings':warnings}

        with:

        - image : :class:`geone.img.Img`
            output image, with `image.nv=nreal` variables (simulations);
            note: `image=None` if `mpds_geosClassicOutput->outputImage=NULL`

        - nwarning : int
            total number of warning(s) encountered (same warnings can be counted
            several times)

        - warnings : list of strs
            list of distinct warnings encountered (can be empty)
    """
    fname = 'simulate'
    
    # Prevent calculation if covariance model is not stationary
    if not cov_model.is_stationary():
        err_msg = f'{fname}: `cov_model` is not stationary: {fname} cannot be applied (use the other paramters for non-stationary covariance)'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Number of realization : nreal
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose > 0:
            if logger:
                logger.warning(f'{fname}: `nreal` <= 0: `None` is returned')
            else:
                print(f'{fname}: WARNING: `nreal` <= 0: `None` is returned')
        return None

    # Set space dimension: d
    if hasattr(dimension, '__len__'):
        d = len(dimension)
    else:
        # assume dimension is an int, nx
        d = 1

    if d not in (1, 2, 3):
        err_msg = f'{fname}: space dimension not valid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check (or set) argument 'spacing'
    if spacing is None:
        spacing = tuple(np.ones(d))
    else:
        if hasattr(spacing, '__len__') and len(spacing) != d:
            err_msg = f'{fname}: `spacing` of incompatible length'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Check (or set) argument 'origin'
    if origin is None:
        origin = tuple(np.zeros(d))
    else:
        if hasattr(origin, '__len__') and len(origin) != d:
            err_msg = f'{fname}: `origin` of incompatible length'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Set grid geometry
    if d == 1:
        nx, ny, nz = dimension, 1, 1
        sx, sy, sz = spacing, 1.0, 1.0
        ox, oy, oz = origin, 0.0, 0.0
    elif d == 2:
        nx, ny, nz = *dimension, 1
        sx, sy, sz = *spacing, 1.0 
        ox, oy, oz = *origin, 0.0 
    else: # d == 3
        nx, ny, nz = dimension
        sx, sy, sz = spacing
        ox, oy, oz = origin

    # nxy = nx * ny
    # nxyz = nxy * nz

    # Check dimension of cov_model and set if used as omni-directional model
    if isinstance(cov_model, gcm.CovModel1D):
        omni_dir = True
    else:
        if cov_model.__class__.__name__ != f'CovModel{d}D':
            err_msg = f'{fname}: `cov_model` dimension is incompatible with `dimension`'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        omni_dir = False

    # Check parameters for non-stationarity
    # ... rotation given by `alpha`, `beta`, `gamma`
    if alpha is not None:
        if omni_dir:
            err_msg = f'{fname}: `alpha` cannot be used with 1D covariance model'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)
        if not callable(alpha) and not isinstance(alpha, np.ndarray) and not isinstance(alpha, float) and not isinstance(alpha, int):
            err_msg = f'{fname}: `alpha` not valid: should be callable or array or float or int'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if beta is not None:
        if omni_dir or d < 3:
            err_msg = f'{fname}: `beta` cannot be used with 1D or 2D covariance model'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)
        if not callable(beta) and not isinstance(beta, np.ndarray) and not isinstance(beta, float) and not isinstance(beta, int):
            err_msg = f'{fname}: `beta` not valid: should be callable or array or float or int'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)
    
    if gamma is not None:
        if omni_dir or d < 3:
            err_msg = f'{fname}: `gamma` cannot be used with 1D or 2D covariance model'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)
        if not callable(gamma) and not isinstance(gamma, np.ndarray) and not isinstance(gamma, float) and not isinstance(gamma, int):
            err_msg = f'{fname}: `gamma` not valid: should be callable or array or float or int'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # ... other non-stationarities given by cov_model_non_stationarity_list
    if cov_model_non_stationarity_list is not None:
        if not isinstance(cov_model_non_stationarity_list, list):
            err_msg = f'{fname}: `cov_model_non_stationarity_list` must be a list if not `None`'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)
        for i, cm_ns in enumerate(cov_model_non_stationarity_list):
            if not hasattr(cm_ns, '__len__') or len(cm_ns) not in (2, 3):
                err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list` not valid: should be a tuple or list of length 2 or 3 containing: method name (str), value(s) or callable (for arg)[, kwds (dict)]'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
            if not isinstance(cm_ns[0], str):
                err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list`, `(method_name, val[, kwds])`: `method_name` not valid: should be a string'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
            if not callable(cm_ns[1]) and not isinstance(cm_ns[1], np.ndarray) and not isinstance(cm_ns[1], float) and not isinstance(cm_ns[1], int):
                err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list`, `(method_name, val[, kwds])`: `val` not valid: should be callable or array or float or int'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
            if len(cm_ns) == 3 and not isinstance(cm_ns[2], dict):
                err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list`, `(method_name, val[, kwds])`: `kwds` not valid: should be a dictionary'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # Set varname
    varname = 'V0'

    # Preparation of data points
    if x is None:
        x = np.empty((0, d), dtype='float')
        n = 0
        if v is not None:
            err_msg = f'{fname}: `x` is None but `v` is not None'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        v = np.array([], dtype='float')

    else:
        x = np.asarray(x, dtype='float').reshape(-1, d) # cast in 2-dimensional array if needed
        n = x.shape[0]

        # Check if all data points are in the grid
        if verbose > 0:
            if d == 1:
                if np.any(x[:, 0] < ox) or np.any(x[:, 0] > ox + nx*sx):
                    if logger:
                        logger.warning(f'{fname}: data point(s) out of the grid')
                    else:
                        print(f'{fname}: WARNING: data point(s) out of the grid')
            elif d == 2:
                if    np.any(x[:, 0] < ox) or np.any(x[:, 0] > ox + nx*sx) \
                   or np.any(x[:, 1] < oy) or np.any(x[:, 1] > oy + ny*sy):
                    if logger:
                        logger.warning(f'{fname}: data point(s) out of the grid')
                    else:
                        print(f'{fname}: WARNING: data point(s) out of the grid')
            else: # d == 3
                if    np.any(x[:, 0] < ox) or np.any(x[:, 0] > ox + nx*sx) \
                   or np.any(x[:, 1] < oy) or np.any(x[:, 1] > oy + ny*sy) \
                   or np.any(x[:, 2] < oz) or np.any(x[:, 2] > oz + nz*sz):
                    if logger:
                        logger.warning(f'{fname}: data point(s) out of the grid')
                    else:
                        print(f'{fname}: WARNING: data point(s) out of the grid')

        # Check v
        if v is None:
            err_msg = f'{fname}: `x` is not None but `v` is None'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if v.size != n:
            err_msg = f'{fname}: size of `v` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # ... set standard deviation of data error
    if v_err_std is None:
        v_err_std = 0.0
    v_err_std = np.asarray(v_err_std, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
    if v_err_std.size == 1:
        v_err_std = v_err_std[0] * np.ones(n)
    elif v_err_std.size != n:
        err_msg = f'{fname}: size of `v_err_std` is not valid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Preparation of inequality data points
    if x_ineq is None:
        x_ineq = np.empty((0, d), dtype='float')
        n_ineq = 0
    
    else:
        x_ineq = np.asarray(x_ineq, dtype='float').reshape(-1, d) # cast in 2-dimensional array if needed
        n_ineq = x_ineq.shape[0]

        # Check if all inequality data points are in the grid
        if verbose > 0:
            if d == 1:
                if np.any(x_ineq[:, 0] < ox) or np.any(x_ineq[:, 0] > ox + nx*sx):
                    if logger:
                        logger.warning(f'{fname}: inequality data point(s) out of the grid')
                    else:
                        print(f'{fname}: WARNING: inequality data point(s) out of the grid')
            elif d == 2:
                if    np.any(x_ineq[:, 0] < ox) or np.any(x_ineq[:, 0] > ox + nx*sx) \
                   or np.any(x_ineq[:, 1] < oy) or np.any(x_ineq[:, 1] > oy + ny*sy):
                    if logger:
                        logger.warning(f'{fname}: inequality data point(s) out of the grid')
                    else:
                        print(f'{fname}: WARNING: inequality data point(s) out of the grid')
            else: # d == 3
                if    np.any(x_ineq[:, 0] < ox) or np.any(x_ineq[:, 0] > ox + nx*sx) \
                   or np.any(x_ineq[:, 1] < oy) or np.any(x_ineq[:, 1] > oy + ny*sy) \
                   or np.any(x_ineq[:, 2] < oz) or np.any(x_ineq[:, 2] > oz + nz*sz):
                    if logger:
                        logger.warning(f'{fname}: inequality data point(s) out of the grid')
                    else:
                        print(f'{fname}: WARNING: inequality data point(s) out of the grid')

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
                raise GeosclassicinterfaceError(err_msg)

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
                raise GeosclassicinterfaceError(err_msg)

            v_ineq_max[np.isnan(v_ineq_max)] = np.inf

        # Remove inequality data points with v_ineq_min = -np.inf and v_ineq_max = np.inf
        ind = np.vstack((np.isnan(v_ineq_min), np.isnan(v_ineq_max))).all(axis=0)
        if ind.any():
            x_ineq = x_ineq[~ind]
            v_ineq_min = v_ineq_min[~ind]
            v_ineq_max = v_ineq_max[~ind]
            n_ineq_new = len(x_ineq)
            if verbose > 0:
                if logger:
                    logger.warning(f'{fname}: {n_ineq-n_ineq_new} inequality data points removed (infinite lower and upper bounds))')
                else:
                    print(f'{fname}: WARNING: {n_ineq-n_ineq_new} inequality data points removed (infinite lower and upper bounds))')
            n_ineq = n_ineq_new

        # Check consistency of v_ineq_min and v_ineq_max
        if np.any(v_ineq_max <= v_ineq_min):
            err_msg = f'{fname}: `v_ineq_min` and `v_ineq_max` are not consistent'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Pre-process data and inequality data (if required)
    if preprocess_data_and_ineq_in_grid:
        if n > 0:
            # Get single grid index for data points locations (`x`)
            ix, iy, iz = img.pointToGridIndex(*np.hstack((x, np.zeros((n, 3-d)))).T, sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz)
            ig_x = img.gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz)
            ig_x_unique, ig_x_inv = np.unique(ig_x, return_inverse=True)
        else:
            ig_x = np.array([], dtype='int')
            ig_x_unique = np.array([], dtype='int')

        if n_ineq > 0:
            # Get single grid index for inequality data points locations (`x_ineq`)
            ix, iy, iz = img.pointToGridIndex(*np.hstack((x_ineq, np.zeros((n_ineq, 3-d)))).T, sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz)
            ig_x_ineq = img.gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz)
            ig_x_ineq_unique, ig_x_ineq_inv = np.unique(ig_x_ineq, return_inverse=True)
        else:
            ig_x_ineq = np.array([], dtype='int')
            ig_x_ineq_unique = np.array([], dtype='int')
        
        # Remove inequality data points falling in same cells as data points
        ig_inter, _, ig2 = np.intersect1d(ig_x_unique, ig_x_ineq_unique, assume_unique=True, return_indices=True)
        if len(ig_inter):
            ind = np.vstack([ig_x_ineq == ig_x_ineq_unique[i] for i in ig2]).any(axis=0)
            x_ineq = x_ineq[~ind]
            v_ineq_min = v_ineq_min[~ind]
            v_ineq_max = v_ineq_max[~ind]
            ig_x_ineq = ig_x_ineq[~ind]
            ig_x_ineq_unique, ig_x_ineq_inv = np.unique(ig_x_ineq, return_inverse=True)
            n_ineq_new = len(x_ineq)
            if verbose > 2:
                if logger:
                    logger.info(f'{fname}: Preprocessing data: {n_ineq-n_ineq_new} inequality data points removed (in same grid cell as data points)')
                else:
                    print(f'{fname}: Preprocessing data: {n_ineq-n_ineq_new} inequality data points removed (in same grid cell as data points)')
            n_ineq = n_ineq_new

        # Aggregate data points that fall into the same grid cell
        if len(ig_x) != len(ig_x_unique):
            x         = np.array([x[ig_x_inv==j].mean(axis=0)   for j in range(len(ig_x_unique))])
            v         = np.array([v[ig_x_inv==j].mean()         for j in range(len(ig_x_unique))])
            v_err_std = np.array([v_err_std[ig_x_inv==j].mean() for j in range(len(ig_x_unique))])
            n_new = len(x)
            if verbose > 2:
                if logger:
                    logger.info(f'{fname}: Preprocessing data: data points aggregated (in same grid cells): {n} points (initial) -> {n_new} points (final)')
                else:
                    print(f'{fname}: Preprocessing data: data points aggregated (in same grid cells): {n} points (initial) -> {n_new} points (final)')
            n = n_new

        # Aggregate inequality data points that fall into the same grid cell
        if len(ig_x_ineq) != len(ig_x_ineq_unique):
            n_ineq_new = len(ig_x_ineq_unique)
            x_ineq_new = np.zeros((n_ineq_new, d))
            v_ineq_min_new = np.zeros(n_ineq_new)
            v_ineq_max_new = np.zeros(n_ineq_new)
            for j in range(n_ineq_new):
                ind = ig_x_ineq_inv == j
                x_ineq_new[j] = x_ineq[ind].mean(axis=0)

                vm = v_ineq_min[ind]
                ind2 = np.isinf(vm)
                if ind2.all():
                    v_ineq_min_new[j] = -np.inf
                else:
                    if preprocess_ineq_less_constrained:
                        v_ineq_min_new[j] = vm[~ind2].min()
                    else:
                        v_ineq_min_new[j] = vm[~ind2].max()

                vm = v_ineq_max[ind]
                ind2 = np.isinf(vm)
                if ind2.all():
                    v_ineq_max_new[j] = np.inf
                else:
                    if preprocess_ineq_less_constrained:
                        v_ineq_max_new[j] = vm[~ind2].max()
                    else:
                        v_ineq_max_new[j] = vm[~ind2].min()
            
            if verbose > 2:
                if logger:
                    logger.info(f'{fname}: Preprocessing data: inequality data points aggregated (in same grid cells): {n_ineq} points (initial) -> {n_ineq_new} points (final)')
                else:
                    print(f'{fname}: Preprocessing data: inequality data points aggregated (in same grid cells): {n_ineq} points (initial) -> {n_ineq_new} points (final)')

            n_ineq = n_ineq_new
            x_ineq = x_ineq_new
            v_ineq_min = v_ineq_min_new
            v_ineq_max = v_ineq_max_new

            # Check consistency of v_ineq_min and v_ineq_max
            if np.any(v_ineq_max <= v_ineq_min):
                err_msg = f'{fname}: after aggregation (pre-processing): `v_ineq_min` and `v_ineq_max` are not consistent'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
        
        if verbose > 1:
            if logger:
                logger.info(f'{fname}: Preprocessing data done: final number of data points : {n}, inequality data points: {n_ineq}')
            else:
                print(f'{fname}: Preprocessing data done: final number of data points : {n}, inequality data points: {n_ineq}')

    # Set method : computationMode
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    if method == 'simple_kriging':
        computationMode = 3

    elif method == 'ordinary_kriging':
        if verbose > 0:
            if mean is not None:
                if logger:
                    logger.warning(f"{fname}: `mean` is ignored with `method='ordinary_kriging'`")
                else:
                    print(f"{fname}: WARNING: `mean` is ignored with `method='ordinary_kriging'`")
            if var is not None:
                if logger:
                    logger.warning(f"{fname}: `var` is ignored with `method='ordinary_kriging'`")
                else:
                    print(f"{fname}: WARNING: `var` is ignored with `method='ordinary_kriging'`")
        
        computationMode = 2

    else:
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Set parameters for neighborhood
    # - searchRadius, searchRadius_geosClassic
    # - searchRadiusRelative, searchRadiusRelative_geosClassic
    # - nneighborMax, nneighborMax_geosClassic
    if searchRadius is not None:
        # searchRadius is used, searchRadiusRelative is not used
        if searchRadius <= 0.0:
            err_msg = f'{fname}: search radius (isotropic neighborhood) not valid (negative)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)
        
        searchRadius_geosClassic = searchRadius
        searchRadiusRelative_geosClassic = searchRadiusRelative # unused

    else:
        # searchRadius is not used, searchRadiusRelative is used
        searchRadius_geosClassic = 0.0 # to indicate in external C function that search radius is not used
        searchRadiusRelative_geosClassic = searchRadiusRelative 

        # searchRadiusRelative
        if searchRadiusRelative_geosClassic < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
            err_msg = f'{fname}: `searchRadiusRelative` too small (should be at least {geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN})'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if nneighborMax <= 0:
        # (note : nneighborMax = -1 is authorized in Geos-Classic)
        err_msg = f'{fname}: `nneighborMax` should be greater than or equal to 1'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    nneighborMax_geosClassic = nneighborMax

    # searchNeighborhoodSortMode
    if searchNeighborhoodSortMode is None:
        searchNeighborhoodSortMode = 1
        # # Set greatest possible value
        # searchNeighborhoodSortMode = 2
        # if         (alpha is not None and (callable(alpha) or (isinstance(alpha, np.ndarray) and alpha.size > 1))) \
        #         or (beta  is not None and (callable(beta ) or (isinstance(beta,  np.ndarray) and beta.size  > 1))) \
        #         or (gamma is not None and (callable(gamma) or (isinstance(gamma, np.ndarray) and gamma.size > 1))):
        #     searchNeighborhoodSortMode = 1
        # if cov_model_non_stationarity_list is not None:
        #     for cm_ns in cov_model_non_stationarity_list:
        #         if callable(cm_ns[1]) or (isinstance(cm_ns[1],  np.ndarray) and cm_ns[1].size  > 1):
        #             searchNeighborhoodSortMode = 1
        #             break
    elif searchNeighborhoodSortMode == 2:
        if         (alpha is not None and (callable(alpha) or (isinstance(alpha, np.ndarray) and alpha.size > 1))) \
                or (beta  is not None and (callable(beta ) or (isinstance(beta,  np.ndarray) and beta.size  > 1))) \
                or (gamma is not None and (callable(gamma) or (isinstance(gamma, np.ndarray) and gamma.size > 1))):
            err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model (local rotation)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)
        
        if cov_model_non_stationarity_list is not None:
            for cm_ns in cov_model_non_stationarity_list:
                if callable(cm_ns[1]) or (isinstance(cm_ns[1],  np.ndarray) and cm_ns[1].size  > 1):
                    err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model ({cm_ns[0]})'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)
                
    elif searchNeighborhoodSortMode not in (0, 1):
        err_msg = f'{fname}: `searchNeighborhoodSortMode` not valid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Set mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            err_msg = f'{fname}: `mask` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)
    
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
                logger.info(f'{fname}: number of process(es) has been changed (now: nproc={nproc})')
            else:
                print(f'{fname}: number of process(es) has been changed (now: nproc={nproc})')

    # Set number of threads per process: nth
    if nthreads_per_proc is None:
        nthreads_per_proc = -1
    
    if nthreads_per_proc <= 0:
        nth = max(int(np.floor((multiprocessing.cpu_count() + nthreads_per_proc) / nproc)), 1)
    else:
        nth = int(nthreads_per_proc)
        # if verbose > 1 and nth != nthreads_per_proc:
        #     if logger:
        #         logger.info(f'{fname}: number of threads per process has been changed (now: nthreads_per_proc={nth})')
        #     else:
        #         print(f'{fname}: number of threads per process has been changed (now: nthreads_per_proc={nth})')

    if verbose > 0 and nproc * nth > multiprocessing.cpu_count():
        if logger:
            logger.warning(f'{fname}: total number of cpu(s) used ({nproc * nth}) will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: total number of cpu(s) used ({nproc * nth}) will exceed number of cpu(s) of the system...')

    # Set number of process(es) for the step 1a/1b (simulation at inequality data points)
    if nproc_sgs_at_ineq is None:
        nproc_sgs_at_ineq = nproc * nth

    if verbose > 0 and nproc_sgs_at_ineq > multiprocessing.cpu_count():
        if logger:
            logger.warning(f'{fname}: total number of cpu(s) used ({nproc_sgs_at_ineq}) (for simulation at inequality data points) will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: total number of cpu(s) used ({nproc_sgs_at_ineq}) (for simulation at inequality data points) will exceed number of cpu(s) of the system...')

    # Set seed
    if seed is None:
        seed = np.random.randint(1, 1000000)
    seed = int(seed)

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: Computational resources: nproc = {nproc}, nthreads_per_proc = {nth}, nproc_sgs_at_ineq = {nproc_sgs_at_ineq}')
        else:
            print(f'{fname}: Computational resources: nproc = {nproc}, nthreads_per_proc = {nth}, nproc_sgs_at_ineq = {nproc_sgs_at_ineq}')

    # Mode A and B: do SGS simulations at inequality data points
    # ------------
    if n_ineq > 0:
        # Set mean_x, mean_x_ineq from mean
        if mean is not None:
            if d == 1:
                mean_x      = gcm.eval_at_points_1D(mean, x     , nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                mean_x_ineq = gcm.eval_at_points_1D(mean, x_ineq, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                mean_x      = gcm.eval_at_points_2D(mean, x     , nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                mean_x_ineq = gcm.eval_at_points_2D(mean, x_ineq, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                mean_x      = gcm.eval_at_points_3D(mean, x     , nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                mean_x_ineq = gcm.eval_at_points_3D(mean, x_ineq, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        else:
            mean_x      = None
            mean_x_ineq = None

        # Set var_x, var_x_ineq from var
        if var is not None:
            if d == 1:
                var_x      = gcm.eval_at_points_1D(var, x     , nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                var_x_ineq = gcm.eval_at_points_1D(var, x_ineq, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                var_x      = gcm.eval_at_points_2D(var, x     , nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                var_x_ineq = gcm.eval_at_points_2D(var, x_ineq, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                var_x      = gcm.eval_at_points_3D(var, x     , nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                var_x_ineq = gcm.eval_at_points_3D(var, x_ineq, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        else:
            var_x      = None
            var_x_ineq = None
        
        # Rotation: set angles
        # ... alpha_x_ineq from alpha
        if alpha is not None:
            if d == 1:
                alpha_x_ineq = gcm.eval_at_points_1D(alpha, x_ineq, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                alpha_x_ineq = gcm.eval_at_points_2D(alpha, x_ineq, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                alpha_x_ineq = gcm.eval_at_points_3D(alpha, x_ineq, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        else:
            alpha_x_ineq = None

        # ... beta_x_ineq from beta
        if beta is not None:
            if d == 1:
                beta_x_ineq = gcm.eval_at_points_1D(beta, x_ineq, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                beta_x_ineq = gcm.eval_at_points_2D(beta, x_ineq, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                beta_x_ineq = gcm.eval_at_points_3D(beta, x_ineq, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        else:
            beta_x_ineq = None

        # ... gamma_x_ineq from gamma
        if gamma is not None:
            if d == 1:
                gamma_x_ineq = gcm.eval_at_points_1D(gamma, x_ineq, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                gamma_x_ineq = gcm.eval_at_points_2D(gamma, x_ineq, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                gamma_x_ineq = gcm.eval_at_points_3D(gamma, x_ineq, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        else:
            gamma_x_ineq = None

        # Set non-stationary for covariance model
        # ... cov_model_non_stationarity_x_ineq_list from cov_model_non_stationarity_list
        if cov_model_non_stationarity_list is not None:
            cov_model_non_stationarity_x_ineq_list = []
            for cm_ns in cov_model_non_stationarity_list:
                val = cm_ns[1]
                if d == 1:
                    val_x_ineq = gcm.eval_at_points_1D(val, x_ineq, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                elif d == 2:
                    val_x_ineq = gcm.eval_at_points_2D(val, x_ineq, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                else: # d == 3
                    val_x_ineq = gcm.eval_at_points_3D(val, x_ineq, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                if len(cm_ns) == 2:
                    cm_ns_x_ineq = (cm_ns[0], val_x_ineq)
                else : # len(cm_ns) == 3
                    cm_ns_x_ineq = (cm_ns[0], val_x_ineq, cm_ns[2])
                cov_model_non_stationarity_x_ineq_list.append((cm_ns_x_ineq))
        else:
            cov_model_non_stationarity_x_ineq_list = None

        # Do simulations at inequality data points
        if mode_transform_ineq_to_data:
            # Mode A
            # ------
            nsim = transform_ineq_to_data_with_err_nsim
        else:
            # Mode B
            # ------
            nsim = nreal

        if verbose > 1:
            if logger:
                logger.info(f"{fname}: (Step 1.1) Do sgs at inequality data points ({nsim} simulation(s) at {n_ineq} points)...")
            else:
                print(f"{fname}: (Step 1.1) Do sgs at inequality data points ({nsim} simulation(s) at {n_ineq} points)...")
        
        if nproc_sgs_at_ineq != 1:
            v_ineq_sim = gcm.sgs_at_inequality_data_points_mp(
                        x, v, x_ineq, cov_model,
                        v_err_std=v_err_std,
                        v_ineq_min=v_ineq_min, v_ineq_max=v_ineq_max,
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
                        nreal=nsim,
                        seed=seed,
                        verbose=(verbose>2),
                        nproc=nproc_sgs_at_ineq,
                        logger=logger)
        else:
            v_ineq_sim = gcm.sgs_at_inequality_data_points(
                        x, v, x_ineq, cov_model,
                        v_err_std=v_err_std,
                        v_ineq_min=v_ineq_min, v_ineq_max=v_ineq_max,
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
                        nreal=nsim,
                        seed=seed,
                        verbose=(verbose>2),
                        logger=logger)

    # Mode A and B : prepare parameters (common)
    # ------------
    # --- for step 2.
    n_new = n + n_ineq
    x_new = np.vstack((x, x_ineq))
    # v_new # set further
    # v_new_err_std # set further

    if n_new > 0:
        # --- for step 3.
        # Get grid index and single grid index for new data points locations (`x_new`)
        ix, iy, iz = img.pointToGridIndex(*np.hstack((x_new, np.zeros((n_new, 3-d)))).T, sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz)
        ig_x_new = img.gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz)
        _, ig_x_new_ind = np.unique(ig_x_new, return_index=True)

        if d == 1:
            ix = ix[ig_x_new_ind]
            xc = ox + sx * (ix + 0.5)
            x_cc = xc.reshape(-1, 1)
        elif d == 2:
            ix = ix[ig_x_new_ind]
            iy = iy[ig_x_new_ind]
            xc = ox + sx * (ix + 0.5)
            yc = oy + sy * (iy + 0.5)
            x_cc = np.vstack((xc, yc)).T
        else: # d == 3
            ix = ix[ig_x_new_ind]
            iy = iy[ig_x_new_ind]
            iz = iz[ig_x_new_ind]
            xc = ox + sx * (ix + 0.5)
            yc = oy + sy * (iy + 0.5)
            zc = oz + sz * (iz + 0.5)
            x_cc = np.vstack((xc, yc, zc)).T

        # Set mean_x_cc, mean_x_new from mean
        if mean is not None:
            if d == 1:
                mean_x_cc  = gcm.eval_at_points_1D(mean, x_cc,  nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                mean_x_new = gcm.eval_at_points_1D(mean, x_new, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                mean_x_cc  = gcm.eval_at_points_2D(mean, x_cc,  nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                mean_x_new = gcm.eval_at_points_2D(mean, x_new, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                mean_x_cc  = gcm.eval_at_points_3D(mean, x_cc,  nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                mean_x_new = gcm.eval_at_points_3D(mean, x_new, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        else:
            mean_x_cc  = None
            mean_x_new = None

        # Set var_x_cc, var_x_new from var
        if var is not None:
            if d == 1:
                var_x_cc  = gcm.eval_at_points_1D(var, x_cc,  nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                var_x_new = gcm.eval_at_points_1D(var, x_new, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                var_x_cc  = gcm.eval_at_points_2D(var, x_cc,  nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                var_x_new = gcm.eval_at_points_2D(var, x_new, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                var_x_cc  = gcm.eval_at_points_3D(var, x_cc,  nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                var_x_new = gcm.eval_at_points_3D(var, x_new, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        else:
            var_x_cc  = None
            var_x_new = None
       
        # Rotation: set angles
        # ... alpha_x_cc from alpha
        if alpha is not None:
            if d == 1:
                alpha_x_cc = gcm.eval_at_points_1D(alpha, x_cc, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                alpha_x_cc = gcm.eval_at_points_2D(alpha, x_cc, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                alpha_x_cc = gcm.eval_at_points_3D(alpha, x_cc, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        else:
            alpha_x_cc = None

        # ... beta_x_cc from beta
        if beta is not None:
            if d == 1:
                beta_x_cc = gcm.eval_at_points_1D(beta, x_cc, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                beta_x_cc = gcm.eval_at_points_2D(beta, x_cc, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                beta_x_cc = gcm.eval_at_points_3D(beta, x_cc, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        else:
            beta_x_cc = None

        # ... gamma_x_cc from gamma
        if gamma is not None:
            if d == 1:
                gamma_x_cc = gcm.eval_at_points_1D(gamma, x_cc, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                gamma_x_cc = gcm.eval_at_points_2D(gamma, x_cc, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                gamma_x_cc = gcm.eval_at_points_3D(gamma, x_cc, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        else:
            gamma_x_cc = None

        # Set non-stationary for covariance model
        # ... cov_model_non_stationarity_x_cc_list from cov_model_non_stationarity_list
        if cov_model_non_stationarity_list is not None:
            cov_model_non_stationarity_x_cc_list = []
            for cm_ns in cov_model_non_stationarity_list:
                val = cm_ns[1]
                if d == 1:
                    val_x_cc = gcm.eval_at_points_1D(val, x_cc, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                elif d == 2:
                    val_x_cc = gcm.eval_at_points_2D(val, x_cc, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                else: # d == 3
                    val_x_cc = gcm.eval_at_points_3D(val, x_cc, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                if len(cm_ns) == 2:
                    cm_ns_x_cc = (cm_ns[0], val_x_cc)
                else : # len(cm_ns) == 3
                    cm_ns_x_cc = (cm_ns[0], val_x_cc, cm_ns[2])
                cov_model_non_stationarity_x_cc_list.append((cm_ns_x_cc))
        else:
            cov_model_non_stationarity_x_cc_list = None

        # --- for step 4.
        npt = x_cc.shape[0]
        pts = np.hstack((x_cc, np.zeros((npt, 3-d))))
        vv = np.zeros(npt)         # temporary values
        vv_err_std = np.zeros(npt) # temporary values
        dataPointSet = PointSet(
                            npt=npt, nv=5, 
                            val=np.vstack((pts.T, vv, vv_err_std)), 
                            varname=['X', 'Y', 'Z', varname, f'{varname}_err_std'])

        if mask is not None and add_data_point_to_mask:
            # Make a copy of the original mask, to remove value in added mask cell at the end
            mask_original = np.copy(mask)
            # Add cell to mask if needed
            if pts.shape[0]:
                im_tmp = img.imageFromPoints(pts,
                            nx=nx, ny=ny, nz=nz,
                            sx=sx, sy=sy, sz=sz,
                            ox=ox, oy=oy, oz=oz,
                            indicator_var=True,
                            logger=logger)
                mask = 1.0*np.any((im_tmp.val[0], mask), axis=0)
                del(im_tmp)
            recover_mask = True
        
        else:
            recover_mask = False

        del(pts)
    
    else: # n_new = 0
        # --- for step 3.
        x_cc = None
        # --- for step 4.
        dataPointSet = None
        recover_mask = False

    # --- for step 4.
    # Set mean_x_grid from mean
    if mean is not None:
        if d == 1:
            mean_grid = gcm.eval_in_grid_1D(mean, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
        elif d == 2:
            mean_grid = gcm.eval_in_grid_2D(mean, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
        else: # d == 3
            mean_grid = gcm.eval_in_grid_3D(mean, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
    else:
        mean_grid = None

    # Set var_x_grid from var
    if var is not None:
        if d == 1:
            var_grid = gcm.eval_in_grid_1D(var, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
        elif d == 2:
            var_grid = gcm.eval_in_grid_2D(var, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
        else: # d == 3
            var_grid = gcm.eval_in_grid_3D(var, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
    else:
        var_grid = None
    
    # Set non-stationary covariance model (if needed)
    cov_model_grid = gcm.copyCovModel(cov_model)
    # ... from alpha
    if alpha is not None:
        if d == 1:
            vv = gcm.eval_in_grid_1D(alpha, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
        elif d == 2:
            vv = gcm.eval_in_grid_2D(alpha, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
        else: # d == 3
            vv = gcm.eval_in_grid_3D(alpha, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        cov_model_grid.set_alpha(vv)

    # ... from beta
    if beta is not None:
        if d == 1:
            vv = gcm.eval_in_grid_1D(beta, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
        elif d == 2:
            vv = gcm.eval_in_grid_2D(beta, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
        else: # d == 3
            vv = gcm.eval_in_grid_3D(beta, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        cov_model_grid.set_beta(vv)

    # ... from gamma
    if gamma is not None:
        if d == 1:
            vv = gcm.eval_in_grid_1D(gamma, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
        elif d == 2:
            vv = gcm.eval_in_grid_2D(gamma, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
        else: # d == 3
            vv = gcm.eval_in_grid_3D(gamma, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        cov_model_grid.set_gamma(vv)

    # ... from cov_model_non_stationarity_list
    if cov_model_non_stationarity_list is not None:
        for cm_ns in cov_model_non_stationarity_list:
            try:
                method_name = cm_ns[0]
                val = cm_ns[1]
                if len(cm_ns) == 3:
                    kwds = cm_ns[2]
                else:
                    kwds = {}
                if d == 1:
                    arg = gcm.eval_in_grid_1D(val, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                elif d == 2:
                    arg = gcm.eval_in_grid_2D(val, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                else: # d == 3
                    arg = gcm.eval_in_grid_3D(val, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                eval(f'cov_model_grid.{method_name}')(arg, **kwds)
            except:
                err_msg = f'{fname}: cannot apply non-stationarity (in grid) for covariance model (check parameter `cov_model_non_stationarity_list`)'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # Update seed (for simulation in grid)
    seed = seed + 986 # same increment (whatever the number or realization done in sgs above)

    if mode_transform_ineq_to_data or n_ineq == 0:
        # Mode A (or Mode B without any inequality data)
        # ------
        if n_new > 0:
            v_ineq = np.zeros(n_ineq)
            v_ineq_err_std = np.zeros(n_ineq)

            if n_ineq > 0:
                # 1a. Transform inequality data to equality data with error std
                if verbose > 1:
                    if logger:
                        logger.info(f"{fname}: (Step 1.2) Transform inequality data to equality data with error std...")
                    else:
                        print(f"{fname}: (Step 1.2) Transform inequality data to equality data with error std...")

                # Set new data with error at inequality data points 
                for i in range(n_ineq):
                    v_ineq[i], v_ineq_err_std[i] = gcm.values_to_mean_and_err_std(
                                                        v_ineq_sim[:, i], 
                                                        v_min=v_ineq_min[i], 
                                                        v_max=v_ineq_max[i], 
                                                        p=transform_ineq_to_data_with_err_p, 
                                                        def_shift=1.e-5,
                                                        logger=logger)
            elif verbose > 1:
                if logger:
                    logger.info(f"{fname}: (Step 1) No inequality data")
                else:
                    print(f"{fname}: (Step 1) No inequality data")

            # 2a. New dataset
            if verbose > 1:
                if logger:
                    logger.info(f"{fname}: (Step 2) Set new dataset gathering data and inequality data locations...")
                else:
                    print(f"{fname}: (Step 2) Set new dataset gathering data and inequality data locations...")
            v_new = np.hstack((v, v_ineq))
            v_new_err_std = np.hstack((v_err_std, v_ineq_err_std))

            # 3a. Kriging new dataset at the center of grid cells containing at least one data point
            # Do kriging at cell centers x_cc
            if verbose > 1:
                if logger:
                    logger.info(f"{fname}: (Step 3) Do kriging at the center of grid cells containing at least one data point...")
                else:
                    print(f"{fname}: (Step 3) Do kriging at the center of grid cells containing at least one data point...")

            v_cc, v_cc_err_std = gcm.krige(
                                    x_new, v_new, x_cc, cov_model,
                                    v_err_std=v_new_err_std,
                                    method=method,
                                    mean_x=mean_x_new,
                                    mean_xu=mean_x_cc,
                                    var_x=var_x_new,
                                    var_xu=var_x_cc,
                                    alpha_xu=alpha_x_cc,
                                    beta_xu=beta_x_cc,
                                    gamma_xu=gamma_x_cc,
                                    cov_model_non_stationarity_xu_list=cov_model_non_stationarity_x_cc_list,
                                    use_unique_neighborhood=use_unique_neighborhood,
                                    searchRadius=searchRadius,
                                    searchRadiusRelative=searchRadiusRelative,
                                    nneighborMax=nneighborMax,
                                    verbose=(verbose>2),
                                    logger=logger)

            # Update data point set: set values and error std in point set
            dataPointSet.val[3] = v_cc
            dataPointSet.val[4] = v_cc_err_std
        
        elif verbose > 1:
            if logger:
                logger.info(f"{fname}: (Step 1-3 skipped) No data")
            else:
                print(f"{fname}: (Step 1-3 skipped) No data")
        
        # 4a. Do `nreal` SGS realizations on the grid (at cell centers) using data points at cell centers
        if verbose > 1:
            if logger:
                logger.info(f"{fname}: (Step 4) Do sgs ({nreal} realizations) on the grid (at cell centers) using data points at cell centers...")
            else:
                print(f"{fname}: (Step 4) Do sgs ({nreal} realizations) on the grid (at cell centers) using data points at cell centers...")

        if nproc > 1:
            # multiprocessing enabled
            # -----------------------
            # Set the distribution of the realizations over the processes
            # Condider the Euclidean division of nreal by nproc:
            #     nreal = q * nproc + r, with 0 <= r < nproc
            # Then, (q+1) realizations will be done on process 0, 1, ..., r-1, and q realization on process r, ..., nproc-1
            # Define the list real_index_proc of length (nproc+1) such that
            #   real_index_proc[i], ..., real_index_proc[i+1] - 1 : are the realization indices run on process i
            q, r = np.divmod(nreal, nproc)
            real_index_proc = [i*q + min(i, r) for i in range(nproc+1)]

            if verbose > 1:
                if logger:
                    logger.info(  
                        f'{fname}: Geos-Classic running on {nproc} process(es)... [' + \
                        f'VERSION {geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER:s} / ' + \
                        f'BUILD NUMBER {geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER:s} / ' + \
                        f'OpenMP {nth:d} thread(s)]')
                else:
                    print(  
                        f'{fname}: Geos-Classic running on {nproc} process(es)... [' + \
                        f'VERSION {geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER:s} / ' + \
                        f'BUILD NUMBER {geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER:s} / ' + \
                        f'OpenMP {nth:d} thread(s)]')

            outputReportFile_p = None

            # Set pool of nproc workers
            pool = multiprocessing.Pool(nproc)
            out_pool = []
            for i in range(nproc):
                # Set input for i-th process
                nreal_p = real_index_proc[i+1] - real_index_proc[i]
                seed_p = seed + real_index_proc[i]
                
                if outputReportFile is not None:
                    outputReportFile_p = outputReportFile + f'.{i}'
                
                if i == 0 and verbose >= 3:
                    progress_monitor = 4
                else:
                    progress_monitor = 0

                args = (
                        d,
                        cov_model_grid,
                        nx, ny, nz,
                        sx, sy, sz,
                        ox, oy, oz,
                        varname,
                        outputReportFile_p, # adapted to pid
                        computationMode,
                        None,
                        dataPointSet,
                        mask,
                        mean_grid,
                        var_grid,
                        searchRadius_geosClassic,
                        searchRadiusRelative_geosClassic,
                        nneighborMax_geosClassic,
                        searchNeighborhoodSortMode,
                        geosclassic.MPDS_GEOSCLASSIC_NGIBBSSAMPLERPATH_MIN, # unused
                        geosclassic.MPDS_GEOSCLASSIC_NGIBBSSAMPLERPATH_MIN, # unused
                        seed_p,  # adapted to pid
                        nreal_p, # adapted to pid
                        nth
                    )
                kwds = {
                        'progress_monitor':progress_monitor, 
                        'pid':i,
                        'verbose':verbose*(i==0),
                        'logger':logger
                    }

                # Launch geos-classic (i-th process)
                out_pool.append(pool.apply_async(
                    run_MPDSOMPGeosClassicSim, args=args, kwds=kwds
                ))

            # Properly end working process
            pool.close() # Prevents any more tasks from being submitted to the pool,
            pool.join()  # then, wait for the worker processes to exit.

            # Get result from each process
            geosclassic_output_proc = [p.get() for p in out_pool]

            if np.any([out is None for out in geosclassic_output_proc]):
                err_msg = f'{fname}: async `run_MPDSOMPGeosClassicSim` failed'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
 
            # Gather results from every process
            # image
            image_list = []
            for out in geosclassic_output_proc:
                if out['image'] is not None:
                    image_list.append(out['image'])
                    del(out['image'])
            if len(image_list):
                # Gather images and adjust variable names
                image = img.gatherImages(image_list, keep_varname=True, rem_var_from_source=True, treat_image_one_by_one=treat_image_one_by_one, logger=logger)
                ndigit = geosclassic.MPDS_GEOS_CLASSIC_NB_DIGIT_FOR_REALIZATION_NUMBER
                for j in range(image.nv):
                    image.varname[j] = image.varname[j][:-ndigit] + f'{j:0{ndigit}d}'
            else:
                image = None

            # nwarning
            nwarning = int(np.sum([out['nwarning'] for out in geosclassic_output_proc]))
            # warnings
            warnings = list(np.unique(np.hstack([out['warnings'] for out in geosclassic_output_proc])))

            geosclassic_output = {'image':image, 'nwarning':nwarning, 'warnings':warnings}

            if verbose > 1 and geosclassic_output:
                if logger:
                    logger.info(f'{fname}: Geos-Classic ({nproc} process(es)) run complete')
                else:
                    print(f'{fname}: Geos-Classic ({nproc} process(es)) run complete')

        else:
            # multiprocessing not enabled
            # ---------------------------
            if verbose > 1:
                if logger:
                    logger.info(  
                        f'{fname}: Geos-Classic running on 1 process... [' + \
                        f'VERSION {geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER:s} / ' + \
                        f'BUILD NUMBER {geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER:s} / ' + \
                        f'OpenMP {nth:d} thread(s)]')
                else:
                    print(  
                        f'{fname}: Geos-Classic running on 1 process... [' + \
                        f'VERSION {geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER:s} / ' + \
                        f'BUILD NUMBER {geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER:s} / ' + \
                        f'OpenMP {nth:d} thread(s)]')

            if verbose < 3:
                progress_monitor = 0
            else:
                progress_monitor = 4

            args = (
                    d,
                    cov_model_grid,
                    nx, ny, nz,
                    sx, sy, sz,
                    ox, oy, oz,
                    varname,
                    outputReportFile,
                    computationMode,
                    None,
                    dataPointSet,
                    mask,
                    mean_grid,
                    var_grid,
                    searchRadius_geosClassic,
                    searchRadiusRelative_geosClassic,
                    nneighborMax_geosClassic,
                    searchNeighborhoodSortMode,
                    geosclassic.MPDS_GEOSCLASSIC_NGIBBSSAMPLERPATH_MIN, # unused
                    geosclassic.MPDS_GEOSCLASSIC_NGIBBSSAMPLERPATH_MIN, # unused
                    seed,
                    nreal,
                    nth
                )
            kwds = {
                    'progress_monitor':progress_monitor, 
                    'pid':None,
                    'verbose':verbose,
                    'logger':logger
                }
            geosclassic_output = run_MPDSOMPGeosClassicSim(*args, **kwds)

            if verbose > 1 and geosclassic_output:
                if logger:
                    logger.info(f'{fname}: Geos-Classic (1 process) run complete')
                else:
                    print(f'{fname}: Geos-Classic (1 process) run complete')

    else: # mode_transform_ineq_to_data=False and n_ineq > 0
        # Mode B with at least one inequality data point
        # ------
        # Equality data values will change for each realization -> do realizations (one by one)
        v_new_err_std = np.hstack((v_err_std, np.zeros(n_ineq)))

        if nproc > 1:
            # multiprocessing enabled
            # -----------------------
            # Set the distribution of the realizations over the processes
            # Condider the Euclidean division of nreal by nproc:
            #     nreal = q * nproc + r, with 0 <= r < nproc
            # Then, (q+1) realizations will be done on process 0, 1, ..., r-1, and q realization on process r, ..., nproc-1
            # Define the list real_index_proc of length (nproc+1) such that
            #   real_index_proc[i], ..., real_index_proc[i+1] - 1 : are the realization indices run on process i
            q, r = np.divmod(nreal, nproc)
            real_index_proc = [i*q + min(i, r) for i in range(nproc+1)]

            if verbose > 1:
                if logger:
                    logger.info(
                        f'{fname}: (Step 2-4): Krige + Geos-Classic running on {nproc} process(es)... [' + \
                        f'VERSION {geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER:s} / ' + \
                        f'BUILD NUMBER {geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER:s} / ' + \
                        f'OpenMP {nth:d} thread(s)]')
                else:
                    print(
                        f'{fname}: (Step 2-4): Krige + Geos-Classic running on {nproc} process(es)... [' + \
                        f'VERSION {geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER:s} / ' + \
                        f'BUILD NUMBER {geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER:s} / ' + \
                        f'OpenMP {nth:d} thread(s)]')

            # Set pool of nproc workers
            pool = multiprocessing.Pool(nproc)
            out_pool = []
            for i in range(nproc):
                # Set parameters for i-th process
                ir0, ir1 = real_index_proc[i:i+2]
      
           
                if i == 0 and verbose >= 3:
                    progress_monitor = 4
                else:
                    progress_monitor = 0

                args = (
                        ir0, ir1,                   # range of realizations (indices)
                        v_ineq_sim,                 # sim values (at ineq data locations)
                        x_new, v, x_cc, cov_model,  # krige parameters ...
                        v_new_err_std,
                        method,
                        mean_x_new,
                        mean_x_cc,
                        var_x_new,
                        var_x_cc,
                        alpha_x_cc,
                        beta_x_cc,
                        gamma_x_cc,
                        cov_model_non_stationarity_x_cc_list,
                        use_unique_neighborhood,
                        searchRadius,
                        searchRadiusRelative,
                        nneighborMax,
                        d,                          # MPDSOMPGeosClassicSim parameters ... 
                        cov_model_grid,
                        nx, ny, nz,
                        sx, sy, sz,
                        ox, oy, oz,
                        varname,
                        outputReportFile,
                        computationMode,
                        None,
                        dataPointSet,
                        mask,
                        mean_grid,
                        var_grid,
                        searchRadius_geosClassic,
                        searchRadiusRelative_geosClassic,
                        nneighborMax_geosClassic,
                        searchNeighborhoodSortMode,
                        geosclassic.MPDS_GEOSCLASSIC_NGIBBSSAMPLERPATH_MIN, # unused
                        geosclassic.MPDS_GEOSCLASSIC_NGIBBSSAMPLERPATH_MIN, # unused
                        seed,
                        nth,            # for MPDSOMPGeosClassicSim
                    )
                kwds = {
                        'progress_monitor':progress_monitor, 
                        'pid':i,
                        'verbose':(verbose-1)*(i==0),
                        'logger':logger
                    }

                # Launch geos-classic (i-th process)
                out_pool.append(pool.apply_async(
                    _run_krige_and_MPDSOMPGeosClassicSim, args=args, kwds=kwds
                ))

            # Properly end working process
            pool.close() # Prevents any more tasks from being submitted to the pool,
            pool.join()  # then, wait for the worker processes to exit.

            # Get result from each process
            geosclassic_output_proc = [p.get() for p in out_pool]

            if np.any([out is None for out in geosclassic_output_proc]):
                err_msg = f'{fname}: async `_run_krige_and_MPDSOMPGeosClassicSim` failed'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
 
            # Gather results from every process
            # image
            image_list = []
            for out in geosclassic_output_proc:
                if out['image'] is not None:
                    image_list.append(out['image'])
                    del(out['image'])
            if len(image_list):
                # Gather images and adjust variable names
                image = img.gatherImages(image_list, keep_varname=True, rem_var_from_source=True, treat_image_one_by_one=treat_image_one_by_one, logger=logger)
                ndigit = geosclassic.MPDS_GEOS_CLASSIC_NB_DIGIT_FOR_REALIZATION_NUMBER
                for j in range(image.nv):
                    image.varname[j] = image.varname[j][:-ndigit] + f'{j:0{ndigit}d}'
            else:
                image = None

            # nwarning
            nwarning = int(np.sum([out['nwarning'] for out in geosclassic_output_proc]))
            # warnings
            warnings = list(np.unique(np.hstack([out['warnings'] for out in geosclassic_output_proc])))

            geosclassic_output = {'image':image, 'nwarning':nwarning, 'warnings':warnings}

            if verbose > 1 and geosclassic_output:
                if logger:
                    logger.info(f'{fname}: Krige + Geos-Classic ({nproc} process(es)) run complete')
                else:
                    print(f'{fname}: Krige + Geos-Classic ({nproc} process(es)) run complete')
        
        else:
            # multiprocessing not enabled
            # ---------------------------
            if verbose > 1:
                if logger:
                    logger.info(
                        f'{fname}: (Step 2-4): Krige + Geos-Classic running on 1 process... [' + \
                        f'VERSION {geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER:s} / ' + \
                        f'BUILD NUMBER {geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER:s} / ' + \
                        f'OpenMP {nth:d} thread(s)]')
                else:
                    print(
                        f'{fname}: (Step 2-4): Krige + Geos-Classic running on 1 process... [' + \
                        f'VERSION {geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER:s} / ' + \
                        f'BUILD NUMBER {geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER:s} / ' + \
                        f'OpenMP {nth:d} thread(s)]')

            ir0, ir1 = 0, nreal
           
            if verbose >= 3:
                progress_monitor = 4
            else:
                progress_monitor = 0

            args = (
                    ir0, ir1,                   # range of realizations (indices)
                    v_ineq_sim,                 # sim values (at ineq data locations)
                    x_new, v, x_cc, cov_model,  # krige parameters ...
                    v_new_err_std,
                    method,
                    mean_x_new,
                    mean_x_cc,
                    var_x_new,
                    var_x_cc,
                    alpha_x_cc,
                    beta_x_cc,
                    gamma_x_cc,
                    cov_model_non_stationarity_x_cc_list,
                    use_unique_neighborhood,
                    searchRadius,
                    searchRadiusRelative,
                    nneighborMax,
                    d,                          # MPDSOMPGeosClassicSim parameters ... 
                    cov_model_grid,
                    nx, ny, nz,
                    sx, sy, sz,
                    ox, oy, oz,
                    varname,
                    outputReportFile,
                    computationMode,
                    None,
                    dataPointSet,
                    mask,
                    mean_grid,
                    var_grid,
                    searchRadius_geosClassic,
                    searchRadiusRelative_geosClassic,
                    nneighborMax_geosClassic,
                    searchNeighborhoodSortMode,
                    geosclassic.MPDS_GEOSCLASSIC_NGIBBSSAMPLERPATH_MIN, # unused
                    geosclassic.MPDS_GEOSCLASSIC_NGIBBSSAMPLERPATH_MIN, # unused
                    seed,
                    nth,            # for MPDSOMPGeosClassicSim
                )
            kwds = {
                    'progress_monitor':progress_monitor, 
                    'pid':None,
                    'verbose':verbose-1,
                    'logger':logger
                }
            geosclassic_output = _run_krige_and_MPDSOMPGeosClassicSim(*args, **kwds)

            if verbose > 1 and geosclassic_output:
                if logger:
                    logger.info(f'{fname}: Krige + Geos-Classic (1 process) run complete')
                else:
                    print(f'{fname}: Krige + Geos-Classic (1 process) run complete')
        
    if geosclassic_output is not None and recover_mask:
        # Remove the value out of the original mask (using its copy see above)
        geosclassic_output['image'].val[:, mask_original==0.0] = np.nan

    # Show (print) encountered warnings
    if verbose > 0 and geosclassic_output and geosclassic_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({geosclassic_output['nwarning']} times in all):")
            for i, warning_message in enumerate(geosclassic_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def estimate(
        cov_model,
        dimension, 
        spacing=None, 
        origin=None,
        method='ordinary_kriging',
        x=None, 
        v=None, 
        v_err_std=0.0,
        x_ineq=None, 
        v_ineq_min=None, 
        v_ineq_max=None,
        mean=None, 
        var=None,
        alpha=None,
        beta=None,
        gamma=None,
        cov_model_non_stationarity_list=None,
        preprocess_data_and_ineq_in_grid=True,
        preprocess_ineq_less_constrained=True,
        transform_ineq_to_data_with_err_nsim=100,
        transform_ineq_to_data_with_err_p=0.95,
        use_unique_neighborhood=False,
        mask=None,
        add_data_point_to_mask=True,
        searchRadius=None,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        searchNeighborhoodSortMode=1,
        nGibbsSamplerPath=50,
        seed=None,
        outputReportFile=None,
        nthreads=-1,
        nproc_sgs_at_ineq=None,
        verbose=2,
        logger=None):
    """
    Computes kriging estimates and standard deviations in a grid.

    Interpolation takes place in (center of) grid cells, based on simple or
    ordinary kriging. The space dimension (1, 2, or 3) is detected (from the 
    parameter `dimension`).

    Interpolation can be conditioned by
    
    - data: location: `x`, value: `v`, and errors standard deviation: `v_err_std`
    - inequality data: location: `x_ineq`, lower bounds: `v_ineq_min`, \
    and upper bounds `v_ineq_max` (no error)

    Note that if `v_err_std > 0.0`, the data are considered as data with error, the
    error being a zero-mean Gaussian with standard deviation `v_err_std`.

    Before starting the interpolation, the data points and inequality data points can
    be first pre-processed (optional, if `preprocess_data_and_ineq_in_grid=True`) in
    order to have at most one data point or inequality data point per grid cell, by 
    proceeding as follows:
    
    - if one grid cell contains both data point(s) and inequality data point(s), then \
    the inequality data point(s) are removed
    - if one grid cell contains several data points (no inequality data), they \
    are aggregated in that cell into one data point by taking the mean position, the \
    mean value, and the maximal error standard deviation of data points in that cell
    - if one grid cell contains several inequality data points (no equality data) they \
    are aggregated in that cell into one data point by taking the mean position, and \
    if `preprocess_ineq_less_constrained=True`: the minimal lower bound (from `v_ineq_min`), \
    and the maximal upper bound (from `v_ineq_max`), or \
    if `preprocess_ineq_less_constrained=False`: the maximal lower bound (from `v_ineq_min`), \
    and the minimal upper bound (from `v_ineq_max`) \
    i.e. the less (resp. more) contraining bounds are kept if \
    `preprocess_ineq_less_constrained=True` (resp. `preprocess_ineq_less_constrained=False`); \
    note that the minimum or maximum is taken over non-infinite bounds (if any)

    Interpolation steps are the following:

    1a. If there is inequality data, they are first transformed to new data with 
    error: for that, `transform_ineq_to_data_with_err_nsim` simulations at 
    inequality data points are generated, then at each location, the ensemble of values
    is transformed to a new data value with an error standard deviation using the function
    `geone.covModel.values_to_mean_and_err_std`

    2a. A new dataset is formed by grouping the original data (`x`, `v`, with the 
    error std `v_err_std`) and the new data with their own error std from step 1a.

    3a. The dataset from step 2a is then kriged at the center of grid cells containing 
    at least one data point, this results in a dataset at cell centers (values and error 
    std are respectively the kriging estimates and the kriging std)

    4a. The dataset at cell centers from step 3a is then kriged on the grid (at cell 
    centers), using an external C function (Geos-Classic library)
   
    Multiprocessing
    ---------------
    Multiprocessing (parallel processes) may be enabled in the following steps:
    
    - step 1a above, the simulations are done on `nproc_sgs_at_ineq` processes \
    (python) using each one thread; if `nproc_sgs_at_ineq > 1` multiprocessing is used
    
    Parameters
    ----------
    cov_model : :class:`geone.covModel.CovModel<d>D`
        covariance model in 1D or 2D or 3D;      
        note: the covariance model must be stationary, however, non stationarity is
        handled: 

        - local rotation by specifying `alpha` (in 2D or 3D), `beta` (in 3D), `gamma` (in 3D)
        - other non-stationarities by specifying `cov_model_non_stationarity_list` (see below)

    dimension : [sequence of] int(s)
        number of cells along each axis, for simulation in:

        - 1D: `dimension=nx`
        - 2D: `dimension=(nx, ny)`
        - 3D: `dimension=(nx, ny, nz)`

        note : this parameter determines the space dimension (1, 2, or 3)

    spacing : [sequence of] float(s), optional
        cell size along each axis, for simulation in:

        - 1D: `spacing=sx`
        - 2D: `spacing=(sx, sy)`
        - 3D: `spacing=(sx, sy, sz)`

        by default (`None`): 1.0 along each axis

    origin : [sequence of] float(s), optional
        origin of the grid ("corner of the first cell"), for simulation in:

        - 1D: `origin=ox`
        - 2D: `origin=(ox, oy)`
        - 3D: `origin=(ox, oy, oz)`

        by default (`None`): 0.0 along each axis

    method : str {'simple_kriging', 'ordinary_kriging'}, default: 'ordinary_kriging'
        type of kriging

    x : array-like of floats, optional
        data points locations (float coordinates), array of shape `(n, d)`, 
        where `n` is the number of data points and `d` the spce dimension;
        note: `x` is (cast in array and) reshaped according to space dimension
        (i.e. other shape in input can be handled)

    v : 1D array-like of floats of shape (n,), optional
        data values at `x` (`v[i]` is the data value at `x[i]`); 
        note: if one data point, a float is accepted

    v_err_std : 1D array of floats of shape (n,), or float, default: 0.0
        standard deviation of error at data points, array of same length as `v`;
        if `v_err_std` is a float, the same value is used for all data points; 
        this means that at location x[i], the data value is considered as in a Gaussian 
        distribution of mean `v[i]` and standard deviation `v_err_std[i]`

    x_ineq : array-like of floats, optional
        inequality data points locations (float coordinates), array of shape `(n, d)`,
        where `n` is the number of data points and `d` the spce dimension;
        note: `x_ineq` is (cast in array and) reshaped according to space dimension
        (i.e. other shape in input can be handled)

    v_ineq_min : 1D array of floats of shape (n_ineq,), or float, optional
        minimal value (lower bound) at `x_ineq` (`v_ineq_min[i]` is the lower bound
        at `x_ineq[i]`); 
        notes: 
        
        - if `v_ineq_min=None` (default), no minimal value is considered for any \
        inequality data point (even if `n_ineq > 0`);
        - `v_ineq_min[i]` set to `np.nan` or `-np.inf` means that there is \
        no minimal value for point `x_ineq[i]`
        - if one inequality data point, a float is accepted
        
    v_ineq_max : 1D array of floats of shape (n_ineq,), or float, optional
        maximal value (upper bound) at `x_ineq` (`v_ineq_max[i]` is the upper bound
        at `x_ineq[i]`); 
        notes: 
        
        - if `v_ineq_max=None` (default), no maximal value is considered for any \
        inequality data point (even if `n_ineq > 0`);
        - `v_ineq_max[i]` set to `np.nan` or `np.inf` means that there is \
        no maximal value for point `x_ineq[i]`
        - if one inequality data point, a float is accepted

    mean : function (callable), or array-like of floats, or float, optional
        kriging mean value:

        - if a function: function of `d` argument(s) where `d` is the space dimension \
        that returns a value for each location given by their coordinate(s) in argument
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), values at grid cells
        - if a float: same value at every grid cell
        - by default (`None`): the mean of data value (`v`) (0.0 if no data) is \
        considered at every grid cell

        note: if `method=ordinary_kriging`, parameter `mean` is ignored

    var : function (callable), or array-like of floats, or float, optional
        kriging variance value:

        - if a function: function of `d` argument(s) where `d` is the space dimension \
        that returns a value for each location given by their coordinate(s) in argument
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), values at grid cells
        - if a float: same value at every grid cell
        - by default (`None`): not used (use of covariance model only)

        note: if `method=ordinary_kriging`, parameter `var` is ignored

    alpha : function (callable), or array-like of floats, or float, optional
        azimuth angle in degrees (see :class:`geone.covModel.CovModel<d>D`):

        - if a function: function of `d` argument(s) where `d` is the space dimension \
        that returns a value for each location given by their coordinate(s) in argument
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), values at grid cells
        - if a float: same value at every grid cell
        - by default (`None`): not used (use of covariance model only)

        note: not used in 1D

    beta : function (callable), or array-like of floats, or float, optional
        dip angle in degrees (see :class:`geone.covModel.CovModel<d>D`):

        - if a function: function of `d` argument(s) where `d` is the space dimension \
        that returns a value for each location given by their coordinate(s) in argument
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), values at grid cells
        - if a float: same value at every grid cell
        - by default (`None`): not used (use of covariance model only)

        note: not used in 1D or 2D

    gamma : function (callable), or array-like of floats, or float, optional
        plunge angle in degrees (see :class:`geone.covModel.CovModel<d>D`):

        - if a function: function of `d` argument(s) where `d` is the space dimension \
        that returns a value for each location given by their coordinate(s) in argument
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), values at grid cells
        - if a float: same value at every grid cell
        - by default (`None`): not used (use of covariance model only)

        note: not used in 1D or 2D

    cov_model_non_stationarity_list : list, optional
        list to set non-stationarities in covariance model; each entry must be
        a tuple (or list) `cm_ns` of length 2 or 3 with:

        - `cm_ns[0]`: str: the name of the method of `cov_model` to be applied
        - `cm_ns[1]`: function (callable), or array-like of floats, or float: \
        used to set the main parameter passed to the method:
            - if a function: function of `d` argument(s) where `d` is the space dimension \
            that returns a value for each location given by their coordinate(s) in argument
            - if array-like: its size must be equal to the number of grid cells \
            (the array is reshaped if needed), values at grid cells
            - if a float: same value at every grid cell
        - `cm_ns[2]`: dict, optional: keyworkds arguments to be passed to the method

        Examples (where the parameter `arg` is set from `val`):
        
        - `('multiply_w', val)` will apply `cov_model.multiply_w(arg)`; \
        this multipies the weight contribution of every elementary contribution of the \
        covariance model
        - `('multiply_w', val, {'elem_ind':0})` will apply `cov_model.multiply_w(arg, elem_ind=0)`; \
        this multipies the weight contribution of the elementary contribution of index 0 of the \
        covariance model
        - `('multiply_r', val)` will apply `cov_model.multiply_r(arg)`; \
        this multipies the range in all direction of every elementary contribution of the \
        covariance model
        - `('multiply_r', val, {'r_ind':0})` will apply `cov_model.multiply_r(arg, r_ind=0)`; \
        this multipies the range in the first main direction (index 0) of every elementary \
        contribution of the covariance model
        - `('multiply_r', val, {'r_ind':0, 'elem_ind':0})` will apply `cov_model.multiply_r(arg, r_ind=0, elem_ind=0)`; \
        this multipies the range in the first main direction (index 0) of the elementary \
        contribution of index 0 of the covariance model
       
    preprocess_data_and_ineq_in_grid : bool, default: True
        if `True`: data points and inequality data points are first pre-processed
        in order to have at most one data point or inequality data point per grid cell 
        (see above)

    preprocess_ineq_less_constrained : bool, default: True
        the less (resp. more) contrained bounds are kept for aggregated inequality 
        data points if `preprocess_ineq_less_constrained` is `True` (resp. `False`);
        used only if preprocess_data_and_ineq_in_grid=True`

    transform_ineq_to_data_with_err_nsim : int, default: 100
        number of simulations at inequality data points to transform them to
        new data with error (see above)
    
    transform_ineq_to_data_with_err_p : float, default: 0.95
        probability used to transform the ensemble of values at inequality
        data points to new data with error (see above)
    
    use_unique_neighborhood : bool, default: False
        indicates if a unique neighborhood is used (all data points are considered)
        to krige data at cell centers (step 3a above), and to krige on the
        grid (step 4a above)
    
    mask : array-like, optional
        mask value at grid cells (value 1 for simulated cells, value 0 for not
        simulated cells); the size of the array must be equal to the number of
        grid cells (the array is reshaped if needed)

    add_data_point_to_mask : bool, default: True
        - if `True`: any grid cell that contains a data point is added to (the \
        simulated part of) the mask (if present), i.e. mask value at those cells \
        are set to 1; at the end of the computation the "new mask cells" are \
        removed (by setting a missing value (`numpy.nan`) for the variable out of \
        the original mask)
        - if `False`: original mask is kept as given in input, and data point \
        falling out of (the simulated part of) the mask (if present) are ignored

    searchRadius : float, optional
        if specified, i.e. not `None`: radius of the search neighborhood (ellipsoid
        with same radii along each axis), i.e. the data points at distance to the 
        estimated point greater than `searchRadius` are not taken into account 
        in the kriging system; if `searchRadius` is not `None`, then 
        `searchRadiusRelative` is not used;
        by default (`searchRadius=None`): `searchRadiusRelative` is used to 
        define the search ellipsoid;

        note : parameter also considered for:
         
        - simulation at inequality data points (step 1a)
        - kriging to cell centers (step 3a) unless `use_unique_neighborhood=True`

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
        note (Geos-Classic library): if a range r_i is non-stationary over the grid, 
        its maximal value over the grid is considered;

        note : parameter considered for 
        
        - simulation at inequality data points (step 1a)
        - kriging to cell centers (step 3a) unless `use_unique_neighborhood=True`
        - kriging on the grid (step 4a) unless `use_unique_neighborhood=True`

    nneighborMax : int, default: 12
        maximal number of neighbors (data points) taken into account in the
        kriging system, must be greater than or equal to 1; the data points the 
        closest to the estimated point are taken into account (see the parameters 
        `searchRadius`, `searchRadiusRelative`, `searchNeighborhoodSortMode`);

        note : parameter considered for 
        
        - simulation at inequality data points (step 1a)
        - kriging to cell centers (step 3a) unless `use_unique_neighborhood=True`
        - kriging on the grid (step 4a) unless `use_unique_neighborhood=True`

    searchNeighborhoodSortMode : int, default: 1
        indicates how to sort the search neighboorhood cells (neighbors), for 
        external C function (Geos-Classic library); they are sorted in increasing 
        order according to:

        - `searchNeighborhoodSortMode=0`: distance in the usual axes system
        - `searchNeighborhoodSortMode=1`: distance in the axes sytem supporting \
        the covariance model and accounting for anisotropy given by the ranges \
        (if `searchRadius=None`)
        - `searchNeighborhoodSortMode=2`: minus the evaluation of the covariance \
        model

        Notes:

        - if the covariance model is non-stationary (see note for `cov_model`), then \
        `searchNeighborhoodSortMode=2` is not allowed
        - if the covariance model has any non-stationary range or non-stationary \
        angle and `searchNeighborhoodSortMode=1`: "maximal ranges" (adapted to \
        direction from the central cell) are used to compute distance for sorting \
        the neighbors

        note: parameter considered for kriging on the grid (step 4a)
        unless `use_unique_neighborhood=True`

    nGibbsSamplerPath : int, default: 50
        number of Gibbs sampler paths for simulating values at inequality data points
        
    seed : int, optional
        seed for initializing random number generator (for simulation at inequality
        data points)

    outputReportFile : str, optional
        name of the report file (if desired in output); by default (`None`): no
        report file

    nthreads : int, default: -1
        number of thread(s) for external C function (Geos-Classic library);
        if `nthreads = -n <= 0`: `nthreads` is automatically set to `nmax - n` 
        (but at least 1), where nmax is the total number of cpu(s) of the system 
        (retrieved by `os.cpu_count()`); 
        note: if `nthreads=None`, `nthreads=-1` is used

    nproc_sgs_at_ineq : int, optional
        number of process(es) for sgs at inequality data points:

        - if `nproc_sgs_at_ineq = 1`: multiprocessing is not enabled and the function \
        `gcm.sgs_at_inequality_data_points` is used
        - if `nproc_sgs_at_ineq > 1`: multiprocessing is enabled and the function \
        `gcm.sgs_at_inequality_data_points_mp` with parameter `nproc=nproc_sgs_at_ineq` \
        is used

        by default (`None`): `nproc_sgs_at_ineq` is set to `nthreads`

    verbose : int, default: 2
        verbose mode, higher implies more printing (info):

        - 0: no display
        - 1: warnings
        - 2: warnings + basic info
        - 3 (or >2): all information

        note that if an error occurred, it is raised
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    geosclassic_output : dict
        geosclassic output in python, dictionary

        {'image':image, 'nwarning':nwarning, 'warnings':warnings}

        with:

        - image : :class:`geone.img.Img`
            output image, with `image.nv=2` variables (estimates and standard
            deviations);
            note: `image=None` if `mpds_geosClassicOutput->outputImage=NULL`

        - nwarning : int
            total number of warning(s) encountered (same warnings can be counted
            several times)

        - warnings : list of strs
            list of distinct warnings encountered (can be empty)
    """
    fname = 'estimate'

    # Prevent calculation if covariance model is not stationary
    if not cov_model.is_stationary():
        err_msg = f'{fname}: `cov_model` is not stationary: {fname} cannot be applied (use the other paramters for non-stationary covariance)'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Set space dimension: d
    if hasattr(dimension, '__len__'):
        d = len(dimension)
    else:
        # assume dimension is an int, nx
        d = 1

    if d not in (1, 2, 3):
        err_msg = f'{fname}: space dimension not valid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check (or set) argument 'spacing'
    if spacing is None:
        spacing = tuple(np.ones(d))
    else:
        if hasattr(spacing, '__len__') and len(spacing) != d:
            err_msg = f'{fname}: `spacing` of incompatible length'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Check (or set) argument 'origin'
    if origin is None:
        origin = tuple(np.zeros(d))
    else:
        if hasattr(origin, '__len__') and len(origin) != d:
            err_msg = f'{fname}: `origin` of incompatible length'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Set grid geometry
    if d == 1:
        nx, ny, nz = dimension, 1, 1
        sx, sy, sz = spacing, 1.0, 1.0
        ox, oy, oz = origin, 0.0, 0.0
    elif d == 2:
        nx, ny, nz = *dimension, 1
        sx, sy, sz = *spacing, 1.0 
        ox, oy, oz = *origin, 0.0 
    else: # d == 3
        nx, ny, nz = dimension
        sx, sy, sz = spacing
        ox, oy, oz = origin

    # nxy = nx * ny
    # nxyz = nxy * nz

    # Check dimension of cov_model and set if used as omni-directional model
    if isinstance(cov_model, gcm.CovModel1D):
        omni_dir = True
    else:
        if cov_model.__class__.__name__ != f'CovModel{d}D':
            err_msg = f'{fname}: `cov_model` dimension is incompatible with `dimension`'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        omni_dir = False

    # Check parameters for non-stationarity
    # ... rotation given by `alpha`, `beta`, `gamma`
    if alpha is not None:
        if omni_dir:
            err_msg = f'{fname}: `alpha` cannot be used with 1D covariance model'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if beta is not None:
        if omni_dir or d < 3:
            err_msg = f'{fname}: `beta` cannot be used with 1D or 2D covariance model'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)
    
    if gamma is not None:
        if omni_dir or d < 3:
            err_msg = f'{fname}: `gamma` cannot be used with 1D or 2D covariance model'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # ... other non-stationarities given by cov_model_non_stationarity_list
    if cov_model_non_stationarity_list is not None:
        if not isinstance(cov_model_non_stationarity_list, list):
            err_msg = f'{fname}: `cov_model_non_stationarity_list` must be a list if not `None`'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)
        for i, cm_ns in enumerate(cov_model_non_stationarity_list):
            if not hasattr(cm_ns, '__len__') or len(cm_ns) not in (2, 3):
                err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list` not valid: should be a tuple or list of length 2 or 3 containing: method name (str), value(s) or callable (for arg)[, kwds (dict)]'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
            if not isinstance(cm_ns[0], str):
                err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list`, `(method_name, val[, kwds])`: `method_name` not valid: should be a string'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
            if not callable(cm_ns[1]) and not isinstance(cm_ns[1], np.ndarray) and not isinstance(cm_ns[1], float) and not isinstance(cm_ns[1], int):
                err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list`, `(method_name, val[, kwds])`: `val` not valid: should be callable or array or float or int'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
            if len(cm_ns) == 3 and not isinstance(cm_ns[2], dict):
                err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list`, `(method_name, val[, kwds])`: `kwds` not valid: should be a dictionary'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # Set varname
    varname = 'V0'

    # Preparation of data points
    if x is None:
        x = np.empty((0, d), dtype='float')
        n = 0
        if v is not None:
            err_msg = f'{fname}: `x` is None but `v` is not None'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        v = np.array([], dtype='float')

    else:
        x = np.asarray(x, dtype='float').reshape(-1, d) # cast in 2-dimensional array if needed
        n = x.shape[0]

        # Check if all data points are in the grid
        if verbose > 0:
            if d == 1:
                if np.any(x[:, 0] < ox) or np.any(x[:, 0] > ox + nx*sx):
                    if logger:
                        logger.warning(f'{fname}: data point(s) out of the grid')
                    else:
                        print(f'{fname}: WARNING: data point(s) out of the grid')
            elif d == 2:
                if    np.any(x[:, 0] < ox) or np.any(x[:, 0] > ox + nx*sx) \
                   or np.any(x[:, 1] < oy) or np.any(x[:, 1] > oy + ny*sy):
                    if logger:
                        logger.warning(f'{fname}: data point(s) out of the grid')
                    else:
                        print(f'{fname}: WARNING: data point(s) out of the grid')
            else: # d == 3
                if    np.any(x[:, 0] < ox) or np.any(x[:, 0] > ox + nx*sx) \
                   or np.any(x[:, 1] < oy) or np.any(x[:, 1] > oy + ny*sy) \
                   or np.any(x[:, 2] < oz) or np.any(x[:, 2] > oz + nz*sz):
                    if logger:
                        logger.warning(f'{fname}: data point(s) out of the grid')
                    else:
                        print(f'{fname}: WARNING: data point(s) out of the grid')
        
        # Check v
        if v is None:
            err_msg = f'{fname}: `x` is not None but `v` is None'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if v.size != n:
            err_msg = f'{fname}: size of `v` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # ... set standard deviation of data error
    if v_err_std is None:
        v_err_std = 0.0
    v_err_std = np.asarray(v_err_std, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
    if v_err_std.size == 1:
        v_err_std = v_err_std[0] * np.ones(n)
    elif v_err_std.size != n:
        err_msg = f'{fname}: size of `v_err_std` is not valid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Preparation of inequality data points
    if x_ineq is None:
        x_ineq = np.empty((0, d), dtype='float')
        n_ineq = 0
    
    else:
        x_ineq = np.asarray(x_ineq, dtype='float').reshape(-1, d) # cast in 2-dimensional array if needed
        n_ineq = x_ineq.shape[0]

        # Check if all inequality data points are in the grid
        if verbose > 0:
            if d == 1:
                if np.any(x_ineq[:, 0] < ox) or np.any(x_ineq[:, 0] > ox + nx*sx):
                    if logger:
                        logger.warning(f'{fname}: inequality data point(s) out of the grid')
                    else:
                        print(f'{fname}: WARNING: inequality data point(s) out of the grid')
            elif d == 2:
                if    np.any(x_ineq[:, 0] < ox) or np.any(x_ineq[:, 0] > ox + nx*sx) \
                   or np.any(x_ineq[:, 1] < oy) or np.any(x_ineq[:, 1] > oy + ny*sy):
                    if logger:
                        logger.warning(f'{fname}: inequality data point(s) out of the grid')
                    else:
                        print(f'{fname}: WARNING: inequality data point(s) out of the grid')
            else: # d == 3
                if    np.any(x_ineq[:, 0] < ox) or np.any(x_ineq[:, 0] > ox + nx*sx) \
                   or np.any(x_ineq[:, 1] < oy) or np.any(x_ineq[:, 1] > oy + ny*sy) \
                   or np.any(x_ineq[:, 2] < oz) or np.any(x_ineq[:, 2] > oz + nz*sz):
                    if logger:
                        logger.warning(f'{fname}: inequality data point(s) out of the grid')
                    else:
                        print(f'{fname}: WARNING: inequality data point(s) out of the grid')

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
                raise GeosclassicinterfaceError(err_msg)

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
                raise GeosclassicinterfaceError(err_msg)

            v_ineq_max[np.isnan(v_ineq_max)] = np.inf

        # Remove inequality data points with v_ineq_min = -np.inf and v_ineq_max = np.inf
        ind = np.vstack((np.isnan(v_ineq_min), np.isnan(v_ineq_max))).all(axis=0)
        if ind.any():
            x_ineq = x_ineq[~ind]
            v_ineq_min = v_ineq_min[~ind]
            v_ineq_max = v_ineq_max[~ind]
            n_ineq_new = len(x_ineq)
            if verbose > 0:
                if logger:
                    logger.warning(f'{fname}: {n_ineq-n_ineq_new} inequality data points removed (infinite lower and upper bounds))')
                else:
                    print(f'{fname}: WARNING: {n_ineq-n_ineq_new} inequality data points removed (infinite lower and upper bounds))')
            n_ineq = n_ineq_new

        # Check consistency of v_ineq_min and v_ineq_max
        if np.any(v_ineq_max <= v_ineq_min):
            err_msg = f'{fname}: `v_ineq_min` and `v_ineq_max` are not consistent'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Pre-process data and inequality data (if required)
    if preprocess_data_and_ineq_in_grid:
        if n > 0:
            # Get single grid index for data points locations (`x`)
            ix, iy, iz = img.pointToGridIndex(*np.hstack((x, np.zeros((n, 3-d)))).T, sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz)
            ig_x = img.gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz)
            ig_x_unique, ig_x_inv = np.unique(ig_x, return_inverse=True)
        else:
            ig_x = np.array([], dtype='int')
            ig_x_unique = np.array([], dtype='int')

        if n_ineq > 0:
            # Get single grid index for inequality data points locations (`x_ineq`)
            ix, iy, iz = img.pointToGridIndex(*np.hstack((x_ineq, np.zeros((n_ineq, 3-d)))).T, sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz)
            ig_x_ineq = img.gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz)
            ig_x_ineq_unique, ig_x_ineq_inv = np.unique(ig_x_ineq, return_inverse=True)
        else:
            ig_x_ineq = np.array([], dtype='int')
            ig_x_ineq_unique = np.array([], dtype='int')
        
        # Remove inequality data points falling in same cells as data points
        ig_inter, _, ig2 = np.intersect1d(ig_x_unique, ig_x_ineq_unique, assume_unique=True, return_indices=True)
        if len(ig_inter):
            ind = np.vstack([ig_x_ineq == ig_x_ineq_unique[i] for i in ig2]).any(axis=0)
            x_ineq = x_ineq[~ind]
            v_ineq_min = v_ineq_min[~ind]
            v_ineq_max = v_ineq_max[~ind]
            ig_x_ineq = ig_x_ineq[~ind]
            ig_x_ineq_unique, ig_x_ineq_inv = np.unique(ig_x_ineq, return_inverse=True)
            n_ineq_new = len(x_ineq)
            if verbose > 2:
                if logger:
                    logger.info(f'{fname}: Preprocessing data: {n_ineq-n_ineq_new} inequality data points removed (in same grid cell as data points)')
                else:
                    print(f'{fname}: Preprocessing data: {n_ineq-n_ineq_new} inequality data points removed (in same grid cell as data points)')
            n_ineq = n_ineq_new

        # Aggregate data points that fall into the same grid cell
        if len(ig_x) != len(ig_x_unique):
            x         = np.array([x[ig_x_inv==j].mean(axis=0)   for j in range(len(ig_x_unique))])
            v         = np.array([v[ig_x_inv==j].mean()         for j in range(len(ig_x_unique))])
            v_err_std = np.array([v_err_std[ig_x_inv==j].mean() for j in range(len(ig_x_unique))])
            n_new = len(x)
            if verbose > 2:
                if logger:
                    logger.info(f'{fname}: Preprocessing data: data points aggregated (in same grid cells): {n} points (initial) -> {n_new} points (final)')
                else:
                    print(f'{fname}: Preprocessing data: data points aggregated (in same grid cells): {n} points (initial) -> {n_new} points (final)')
            n = n_new

        # Aggregate inequality data points that fall into the same grid cell
        if len(ig_x_ineq) != len(ig_x_ineq_unique):
            n_ineq_new = len(ig_x_ineq_unique)
            x_ineq_new = np.zeros((n_ineq_new, d))
            v_ineq_min_new = np.zeros(n_ineq_new)
            v_ineq_max_new = np.zeros(n_ineq_new)
            for j in range(n_ineq_new):
                ind = ig_x_ineq_inv == j
                x_ineq_new[j] = x_ineq[ind].mean(axis=0)

                vm = v_ineq_min[ind]
                ind2 = np.isinf(vm)
                if ind2.all():
                    v_ineq_min_new[j] = -np.inf
                else:
                    if preprocess_ineq_less_constrained:
                        v_ineq_min_new[j] = vm[~ind2].min()
                    else:
                        v_ineq_min_new[j] = vm[~ind2].max()

                vm = v_ineq_max[ind]
                ind2 = np.isinf(vm)
                if ind2.all():
                    v_ineq_max_new[j] = np.inf
                else:
                    if preprocess_ineq_less_constrained:
                        v_ineq_max_new[j] = vm[~ind2].max()
                    else:
                        v_ineq_max_new[j] = vm[~ind2].min()
            
            if verbose > 2:
                if logger:
                    logger.info(f'{fname}: Preprocessing data: inequality data points aggregated (in same grid cells): {n_ineq} points (initial) -> {n_ineq_new} points (final)')
                else:
                    print(f'{fname}: Preprocessing data: inequality data points aggregated (in same grid cells): {n_ineq} points (initial) -> {n_ineq_new} points (final)')

            n_ineq = n_ineq_new
            x_ineq = x_ineq_new
            v_ineq_min = v_ineq_min_new
            v_ineq_max = v_ineq_max_new

            # Check consistency of v_ineq_min and v_ineq_max
            if np.any(v_ineq_max <= v_ineq_min):
                err_msg = f'{fname}: after aggregation (pre-processing): `v_ineq_min` and `v_ineq_max` are not consistent'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
        
        if verbose > 1:
            if logger:
                logger.info(f'{fname}: Preprocessing data done: final number of data points : {n}, inequality data points: {n_ineq}')
            else:
                print(f'{fname}: Preprocessing data done: final number of data points : {n}, inequality data points: {n_ineq}')

    # Set method : computationMode
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    if method == 'simple_kriging':
        computationMode = 1

    elif method == 'ordinary_kriging':
        if verbose > 0:
            if mean is not None:
                if logger:
                    logger.warning(f"{fname}: `mean` is ignored with `method='ordinary_kriging'`")
                else:
                    print(f"{fname}: WARNING: `mean` is ignored with `method='ordinary_kriging'`")
            if var is not None:
                if logger:
                    logger.warning(f"{fname}: `var` is ignored with `method='ordinary_kriging'`")
                else:
                    print(f"{fname}: WARNING: `var` is ignored with `method='ordinary_kriging'`")
        
        computationMode = 0

    else:
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Set parameters for neighborhood
    # - searchRadius, searchRadius_geosClassic
    # - searchRadiusRelative, searchRadiusRelative_geosClassic
    # - nneighborMax, nneighborMax_geosClassic
    if searchRadius is not None:
        # searchRadius is used, searchRadiusRelative is not used
        if searchRadius <= 0.0:
            err_msg = f'{fname}: search radius (isotropic neighborhood) not valid (negative)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)
        
        searchRadius_geosClassic = searchRadius
        searchRadiusRelative_geosClassic = searchRadiusRelative # unused

    else:
        # searchRadius is not used, searchRadiusRelative is used
        searchRadius_geosClassic = 0.0 # to indicate in external C function that search radius is not used
        searchRadiusRelative_geosClassic = searchRadiusRelative 

        # searchRadiusRelative
        if searchRadiusRelative_geosClassic < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
            err_msg = f'{fname}: `searchRadiusRelative` too small (should be at least {geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN})'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if nneighborMax <= 0:
        # (note : nneighborMax = -1 is authorized in Geos-Classic)
        err_msg = f'{fname}: `nneighborMax` should be greater than or equal to 1'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)
        
    nneighborMax_geosClassic = nneighborMax

    if use_unique_neighborhood:
        searchRadius_geosClassic = 0.0          # to indicate in external C function that search radius is not used
        searchRadiusRelative_geosClassic = -1.0 # to indicate for external C function that unique search neighborhood is used

    # searchNeighborhoodSortMode
    if searchNeighborhoodSortMode is None:
        searchNeighborhoodSortMode = 1
        # # Set greatest possible value
        # searchNeighborhoodSortMode = 2
        # if         (alpha is not None and (callable(alpha) or (isinstance(alpha, np.ndarray) and alpha.size > 1))) \
        #         or (beta  is not None and (callable(beta ) or (isinstance(beta,  np.ndarray) and beta.size  > 1))) \
        #         or (gamma is not None and (callable(gamma) or (isinstance(gamma, np.ndarray) and gamma.size > 1))):
        #     searchNeighborhoodSortMode = 1
        # if cov_model_non_stationarity_list is not None:
        #     for cm_ns in cov_model_non_stationarity_list:
        #         if callable(cm_ns[1]) or (isinstance(cm_ns[1],  np.ndarray) and cm_ns[1].size  > 1):
        #             searchNeighborhoodSortMode = 1
        #             break
    elif searchNeighborhoodSortMode == 2:
        if         (alpha is not None and (callable(alpha) or (isinstance(alpha, np.ndarray) and alpha.size > 1))) \
                or (beta  is not None and (callable(beta ) or (isinstance(beta,  np.ndarray) and beta.size  > 1))) \
                or (gamma is not None and (callable(gamma) or (isinstance(gamma, np.ndarray) and gamma.size > 1))):
            err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model (local rotation)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)
        
        if cov_model_non_stationarity_list is not None:
            for cm_ns in cov_model_non_stationarity_list:
                if callable(cm_ns[1]) or (isinstance(cm_ns[1],  np.ndarray) and cm_ns[1].size  > 1):
                    err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model ({cm_ns[0]})'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)
                
    elif searchNeighborhoodSortMode not in (0, 1):
        err_msg = f'{fname}: `searchNeighborhoodSortMode` not valid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Set mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            err_msg = f'{fname}: `mask` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)
    
    # Set number of threads per process: nth
    if nthreads is None:
        nthreads = -1
    
    if nthreads <= 0:
        nth = max(int(os.cpu_count() + nthreads), 1)
    else:
        nth = int(nthreads)
        # if verbose > 1 and nth != nthreads:
        #     if logger:
        #         logger.info(f'{fname}: number of threads has been changed (now: nthreads={nth})')
        #     else:
        #         print(f'{fname}: number of threads has been changed (now: nthreads={nth})')

    if verbose > 0 and nthreads > os.cpu_count():
        if logger:
            logger.warning(f'{fname}: total number of cpu(s) used ({nthreads}) will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: total number of cpu(s) used ({nthreads}) will exceed number of cpu(s) of the system...')

    # Set number of process(es) for the step 1a (simulation at inequality data points)
    if nproc_sgs_at_ineq is None:
        nproc_sgs_at_ineq = nth

    if verbose > 0 and nproc_sgs_at_ineq > os.cpu_count():
        if logger:
            logger.warning(f'{fname}: total number of cpu(s) used ({nproc_sgs_at_ineq}) (for simulation at inequality data points) will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: total number of cpu(s) used ({nproc_sgs_at_ineq}) (for simulation at inequality data points) will exceed number of cpu(s) of the system...')

    # Set seed
    if seed is None:
        seed = np.random.randint(1, 1000000)
    seed = int(seed)

    if verbose > 1:
        if logger:
            logger.info(f'{fname}: Computational resources: nthreads = {nth}, nproc_sgs_at_ineq = {nproc_sgs_at_ineq}')
        else:
            print(f'{fname}: Computational resources: nthreads = {nth}, nproc_sgs_at_ineq = {nproc_sgs_at_ineq}')

    # Do SGS simulations at inequality data points
    # ------------
    if n_ineq > 0:
        # Set mean_x, mean_x_ineq from mean
        if mean is not None:
            if d == 1:
                mean_x      = gcm.eval_at_points_1D(mean, x     , nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                mean_x_ineq = gcm.eval_at_points_1D(mean, x_ineq, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                mean_x      = gcm.eval_at_points_2D(mean, x     , nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                mean_x_ineq = gcm.eval_at_points_2D(mean, x_ineq, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                mean_x      = gcm.eval_at_points_3D(mean, x     , nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                mean_x_ineq = gcm.eval_at_points_3D(mean, x_ineq, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        else:
            mean_x      = None
            mean_x_ineq = None

        # Set var_x, var_x_ineq from var
        if var is not None:
            if d == 1:
                var_x      = gcm.eval_at_points_1D(var, x     , nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                var_x_ineq = gcm.eval_at_points_1D(var, x_ineq, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                var_x      = gcm.eval_at_points_2D(var, x     , nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                var_x_ineq = gcm.eval_at_points_2D(var, x_ineq, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                var_x      = gcm.eval_at_points_3D(var, x     , nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                var_x_ineq = gcm.eval_at_points_3D(var, x_ineq, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        else:
            var_x      = None
            var_x_ineq = None
        
        # Rotation: set angles
        # ... alpha_x_ineq from alpha
        if alpha is not None:
            if d == 1:
                alpha_x_ineq = gcm.eval_at_points_1D(alpha, x_ineq, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                alpha_x_ineq = gcm.eval_at_points_2D(alpha, x_ineq, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                alpha_x_ineq = gcm.eval_at_points_3D(alpha, x_ineq, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        else:
            alpha_x_ineq = None

        # ... beta_x_ineq from beta
        if beta is not None:
            if d == 1:
                beta_x_ineq = gcm.eval_at_points_1D(beta, x_ineq, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                beta_x_ineq = gcm.eval_at_points_2D(beta, x_ineq, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                beta_x_ineq = gcm.eval_at_points_3D(beta, x_ineq, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        else:
            beta_x_ineq = None

        # ... gamma_x_ineq from gamma
        if gamma is not None:
            if d == 1:
                gamma_x_ineq = gcm.eval_at_points_1D(gamma, x_ineq, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                gamma_x_ineq = gcm.eval_at_points_2D(gamma, x_ineq, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                gamma_x_ineq = gcm.eval_at_points_3D(gamma, x_ineq, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        else:
            gamma_x_ineq = None

        # Set non-stationary for covariance model
        # ... cov_model_non_stationarity_x_ineq_list from cov_model_non_stationarity_list
        if cov_model_non_stationarity_list is not None:
            cov_model_non_stationarity_x_ineq_list = []
            for cm_ns in cov_model_non_stationarity_list:
                val = cm_ns[1]
                if d == 1:
                    val_x_ineq = gcm.eval_at_points_1D(val, x_ineq, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                elif d == 2:
                    val_x_ineq = gcm.eval_at_points_2D(val, x_ineq, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                else: # d == 3
                    val_x_ineq = gcm.eval_at_points_3D(val, x_ineq, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                if len(cm_ns) == 2:
                    cm_ns_x_ineq = (cm_ns[0], val_x_ineq)
                else : # len(cm_ns) == 3
                    cm_ns_x_ineq = (cm_ns[0], val_x_ineq, cm_ns[2])
                cov_model_non_stationarity_x_ineq_list.append((cm_ns_x_ineq))
        else:
            cov_model_non_stationarity_x_ineq_list = None

        # Do simulations at inequality data points
        nsim = transform_ineq_to_data_with_err_nsim

        if verbose > 1:
            if logger:
                logger.info(f"{fname}: (Step 1.1) Do sgs at inequality data points ({nsim} simulation(s) at {n_ineq} points)...")
            else:
                print(f"{fname}: (Step 1.1) Do sgs at inequality data points ({nsim} simulation(s) at {n_ineq} points)...")
        
        if nproc_sgs_at_ineq != 1:
            v_ineq_sim = gcm.sgs_at_inequality_data_points_mp(
                        x, v, x_ineq, cov_model,
                        v_err_std=v_err_std,
                        v_ineq_min=v_ineq_min, v_ineq_max=v_ineq_max,
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
                        nreal=nsim,
                        seed=seed,
                        verbose=(verbose>2),
                        nproc=nproc_sgs_at_ineq,
                        logger=logger)
        else:
            v_ineq_sim = gcm.sgs_at_inequality_data_points(
                        x, v, x_ineq, cov_model,
                        v_err_std=v_err_std,
                        v_ineq_min=v_ineq_min, v_ineq_max=v_ineq_max,
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
                        nreal=nsim,
                        seed=seed,
                        verbose=(verbose>2),
                        logger=logger)

    # --- for step 2. (preparation)
    n_new = n + n_ineq

    if n_new > 0:
        v_ineq = np.zeros(n_ineq)
        v_ineq_err_std = np.zeros(n_ineq)

        if n_ineq > 0:
            # 1a. Transform inequality data to equality data with error std
            if verbose > 1:
                if logger:
                    logger.info(f"{fname}: (Step 1.2) Transform inequality data to equality data with error std...")
                else:
                    print(f"{fname}: (Step 1.2) Transform inequality data to equality data with error std...")

            # Set new data with error at inequality data points 
            for i in range(n_ineq):
                v_ineq[i], v_ineq_err_std[i] = gcm.values_to_mean_and_err_std(
                                                    v_ineq_sim[:, i], 
                                                    v_min=v_ineq_min[i], 
                                                    v_max=v_ineq_max[i], 
                                                    p=transform_ineq_to_data_with_err_p, 
                                                    def_shift=1.e-5,
                                                    logger=logger)
        elif verbose > 1:
            if logger:
                logger.info(f"{fname}: (Step 1) No inequality data")
            else:
                print(f"{fname}: (Step 1) No inequality data")

        # 2a. New dataset
        if verbose > 1:
            if logger:
                logger.info(f"{fname}: (Step 2) Set new dataset gathering data and inequality data locations...")
            else:
                print(f"{fname}: (Step 2) Set new dataset gathering data and inequality data locations...")
                
        x_new = np.vstack((x, x_ineq))
        v_new = np.hstack((v, v_ineq))
        v_new_err_std = np.hstack((v_err_std, v_ineq_err_std))

        # --- for step 3. (preparation)
        # Get grid index and single grid index for new data points locations (`x_new`)
        ix, iy, iz = img.pointToGridIndex(*np.hstack((x_new, np.zeros((n_new, 3-d)))).T, sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz)
        ig_x_new = img.gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz)
        _, ig_x_new_ind = np.unique(ig_x_new, return_index=True)
        
        if d == 1:
            ix = ix[ig_x_new_ind]
            xc = ox + sx * (ix + 0.5)
            x_cc = xc.reshape(-1, 1)
        elif d == 2:
            ix = ix[ig_x_new_ind]
            iy = iy[ig_x_new_ind]
            xc = ox + sx * (ix + 0.5)
            yc = oy + sy * (iy + 0.5)
            x_cc = np.vstack((xc, yc)).T
        else: # d == 3
            ix = ix[ig_x_new_ind]
            iy = iy[ig_x_new_ind]
            iz = iz[ig_x_new_ind]
            xc = ox + sx * (ix + 0.5)
            yc = oy + sy * (iy + 0.5)
            zc = oz + sz * (iz + 0.5)
            x_cc = np.vstack((xc, yc, zc)).T

        # Set mean_x_cc, mean_x_new from mean
        if mean is not None:
            if d == 1:
                mean_x_cc  = gcm.eval_at_points_1D(mean, x_cc,  nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                mean_x_new = gcm.eval_at_points_1D(mean, x_new, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                mean_x_cc  = gcm.eval_at_points_2D(mean, x_cc,  nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                mean_x_new = gcm.eval_at_points_2D(mean, x_new, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                mean_x_cc  = gcm.eval_at_points_3D(mean, x_cc,  nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                mean_x_new = gcm.eval_at_points_3D(mean, x_new, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        else:
            mean_x_cc  = None
            mean_x_new = None

        # Set var_x_cc, var_x_new from var
        if var is not None:
            if d == 1:
                var_x_cc  = gcm.eval_at_points_1D(var, x_cc,  nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                var_x_new = gcm.eval_at_points_1D(var, x_new, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                var_x_cc  = gcm.eval_at_points_2D(var, x_cc,  nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                var_x_new = gcm.eval_at_points_2D(var, x_new, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                var_x_cc  = gcm.eval_at_points_3D(var, x_cc,  nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                var_x_new = gcm.eval_at_points_3D(var, x_new, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        else:
            var_x_cc  = None
            var_x_new = None
       
        # Rotation: set angles
        # ... alpha_x_cc from alpha
        if alpha is not None:
            if d == 1:
                alpha_x_cc = gcm.eval_at_points_1D(alpha, x_cc, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                alpha_x_cc = gcm.eval_at_points_2D(alpha, x_cc, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                alpha_x_cc = gcm.eval_at_points_3D(alpha, x_cc, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        else:
            alpha_x_cc = None

        # ... beta_x_cc from beta
        if beta is not None:
            if d == 1:
                beta_x_cc = gcm.eval_at_points_1D(beta, x_cc, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                beta_x_cc = gcm.eval_at_points_2D(beta, x_cc, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                beta_x_cc = gcm.eval_at_points_3D(beta, x_cc, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        else:
            beta_x_cc = None

        # ... gamma_x_cc from gamma
        if gamma is not None:
            if d == 1:
                gamma_x_cc = gcm.eval_at_points_1D(gamma, x_cc, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                gamma_x_cc = gcm.eval_at_points_2D(gamma, x_cc, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                gamma_x_cc = gcm.eval_at_points_3D(gamma, x_cc, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        else:
            gamma_x_cc = None

        # Set non-stationary for covariance model
        # ... cov_model_non_stationarity_x_cc_list from cov_model_non_stationarity_list
        if cov_model_non_stationarity_list is not None:
            cov_model_non_stationarity_x_cc_list = []
            for cm_ns in cov_model_non_stationarity_list:
                val = cm_ns[1]
                if d == 1:
                    val_x_cc = gcm.eval_at_points_1D(val, x_cc, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                elif d == 2:
                    val_x_cc = gcm.eval_at_points_2D(val, x_cc, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                else: # d == 3
                    val_x_cc = gcm.eval_at_points_3D(val, x_cc, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                if len(cm_ns) == 2:
                    cm_ns_x_cc = (cm_ns[0], val_x_cc)
                else : # len(cm_ns) == 3
                    cm_ns_x_cc = (cm_ns[0], val_x_cc, cm_ns[2])
                cov_model_non_stationarity_x_cc_list.append((cm_ns_x_cc))
        else:
            cov_model_non_stationarity_x_cc_list = None

        # 3a. Kriging new dataset at the center of grid cells containing at least one data point
        # Do kriging at cell centers x_cc
        if verbose > 1:
            if logger:
                logger.info(f"{fname}: (Step 3) Do kriging at the center of grid cells containing at least one data point...")
            else:
                print(f"{fname}: (Step 3) Do kriging at the center of grid cells containing at least one data point...")

        v_cc, v_cc_err_std = gcm.krige(
                                x_new, v_new, x_cc, cov_model,
                                v_err_std=v_new_err_std,
                                method=method,
                                mean_x=mean_x_new,
                                mean_xu=mean_x_cc,
                                var_x=var_x_new,
                                var_xu=var_x_cc,
                                alpha_xu=alpha_x_cc,
                                beta_xu=beta_x_cc,
                                gamma_xu=gamma_x_cc,
                                cov_model_non_stationarity_xu_list=cov_model_non_stationarity_x_cc_list,
                                use_unique_neighborhood=use_unique_neighborhood,
                                searchRadius=searchRadius,
                                searchRadiusRelative=searchRadiusRelative,
                                nneighborMax=nneighborMax,
                                verbose=(verbose>2),
                                logger=logger)

        # --- for step 4. (preparation)
        npt = x_cc.shape[0]
        pts = np.hstack((x_cc, np.zeros((npt, 3-d))))
        dataPointSet = PointSet(
                            npt=npt, nv=5, 
                            val=np.vstack((pts.T, v_cc, v_cc_err_std)), 
                            varname=['X', 'Y', 'Z', varname, f'{varname}_err_std'])

        if mask is not None and add_data_point_to_mask:
            # Make a copy of the original mask, to remove value in added mask cell at the end
            mask_original = np.copy(mask)
            # Add cell to mask if needed
            if pts.shape[0]:
                im_tmp = img.imageFromPoints(pts,
                            nx=nx, ny=ny, nz=nz,
                            sx=sx, sy=sy, sz=sz,
                            ox=ox, oy=oy, oz=oz,
                            indicator_var=True,
                            logger=logger)
                mask = 1.0*np.any((im_tmp.val[0], mask), axis=0)
                del(im_tmp)
            recover_mask = True
        
        else:
            recover_mask = False

        del(pts)

    else: # n_new = 0 : no data
        if verbose > 1:
            if logger:
                logger.info(f"{fname}: (Step 1-3 skipped) No data")
            else:
                print(f"{fname}: (Step 1-3 skipped) No data")
        
        # --- for step 4. (preparation)
        dataPointSet = None
        recover_mask = False

    # --- for step 4. (preparation)
    # Set mean_x_grid from mean
    if mean is not None:
        if d == 1:
            mean_grid = gcm.eval_in_grid_1D(mean, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
        elif d == 2:
            mean_grid = gcm.eval_in_grid_2D(mean, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
        else: # d == 3
            mean_grid = gcm.eval_in_grid_3D(mean, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
    else:
        mean_grid = None

    # Set var_x_grid from var
    if var is not None:
        if d == 1:
            var_grid = gcm.eval_in_grid_1D(var, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
        elif d == 2:
            var_grid = gcm.eval_in_grid_2D(var, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
        else: # d == 3
            var_grid = gcm.eval_in_grid_3D(var, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
    else:
        var_grid = None
    
    # Set non-stationary covariance model (if needed)
    cov_model_grid = gcm.copyCovModel(cov_model)
    # ... from alpha
    if alpha is not None:
        if d == 1:
            vv = gcm.eval_in_grid_1D(alpha, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
        elif d == 2:
            vv = gcm.eval_in_grid_2D(alpha, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
        else: # d == 3
            vv = gcm.eval_in_grid_3D(alpha, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        cov_model_grid.set_alpha(vv)

    # ... from beta
    if beta is not None:
        if d == 1:
            vv = gcm.eval_in_grid_1D(beta, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
        elif d == 2:
            vv = gcm.eval_in_grid_2D(beta, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
        else: # d == 3
            vv = gcm.eval_in_grid_3D(beta, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        cov_model_grid.set_beta(vv)

    # ... from gamma
    if gamma is not None:
        if d == 1:
            vv = gcm.eval_in_grid_1D(gamma, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
        elif d == 2:
            vv = gcm.eval_in_grid_2D(gamma, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
        else: # d == 3
            vv = gcm.eval_in_grid_3D(gamma, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
        cov_model_grid.set_gamma(vv)

    # ... from cov_model_non_stationarity_list
    if cov_model_non_stationarity_list is not None:
        for cm_ns in cov_model_non_stationarity_list:
            try:
                method_name = cm_ns[0]
                val = cm_ns[1]
                if len(cm_ns) == 3:
                    kwds = cm_ns[2]
                else:
                    kwds = {}
                if d == 1:
                    arg = gcm.eval_in_grid_1D(val, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                elif d == 2:
                    arg = gcm.eval_in_grid_2D(val, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                else: # d == 3
                    arg = gcm.eval_in_grid_3D(val, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                eval(f'cov_model_grid.{method_name}')(arg, **kwds)
            except:
                err_msg = f'{fname}: cannot apply non-stationarity (in grid) for covariance model (check parameter `cov_model_non_stationarity_list`)'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # 4a. Do kriging on the grid (at cell centers) using data points at cell centers
    if verbose > 1:
        if logger:
            logger.info(f"{fname}: (Step 4) Do kriging on the grid (at cell centers) using data points at cell centers...")
        else:
            print(f"{fname}: (Step 4) Do kriging on the grid (at cell centers) using data points at cell centers...")

    if verbose > 1:
        if logger:
            logger.info(  
                f'{fname}: Geos-Classic running on 1 process... [' + \
                f'VERSION {geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER:s} / ' + \
                f'BUILD NUMBER {geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER:s} / ' + \
                f'OpenMP {nth:d} thread(s)]')
        else:
            print(  
                f'{fname}: Geos-Classic running on 1 process... [' + \
                f'VERSION {geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER:s} / ' + \
                f'BUILD NUMBER {geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER:s} / ' + \
                f'OpenMP {nth:d} thread(s)]')

    if verbose < 3:
        progress_monitor = 0
    else:
        progress_monitor = 4

    args = (
            d,
            cov_model_grid,
            nx, ny, nz,
            sx, sy, sz,
            ox, oy, oz,
            varname,
            outputReportFile,
            computationMode,
            None,
            dataPointSet,
            mask,
            mean_grid,
            var_grid,
            searchRadius_geosClassic,
            searchRadiusRelative_geosClassic,
            nneighborMax_geosClassic,
            searchNeighborhoodSortMode,
            0,
            0,
            0,
            0,
            nth
        )
    kwds = {
            'progress_monitor':progress_monitor, 
            'pid':None,
            'verbose':verbose,
            'logger':logger
        }
    geosclassic_output = run_MPDSOMPGeosClassicSim(*args, **kwds)

    if verbose > 1 and geosclassic_output:
        if logger:
            logger.info(f'{fname}: Geos-Classic (1 process) run complete')
        else:
            print(f'{fname}: Geos-Classic (1 process) run complete')

    if geosclassic_output is not None and recover_mask:
        # Remove the value out of the original mask (using its copy see above)
        geosclassic_output['image'].val[:, mask_original==0.0] = np.nan

    # Show (print) encountered warnings
    if verbose > 0 and geosclassic_output and geosclassic_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({geosclassic_output['nwarning']} times in all):")
            for i, warning_message in enumerate(geosclassic_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def fill_mpds_geosClassicIndicatorInput(
        use_same_cov_model_for_all_categories,
        space_dim,
        ncategory,
        categoryValue,
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
        probability,
        searchRadius,
        searchRadiusRelative,
        nneighborMax,
        searchNeighborhoodSortMode,
        seed,
        nreal,
        logger=None):
    """
    Fills a mpds_geosClassicIndicatorInput C structure from given parameters.

    This function should not be called directly, it is used in other functions
    of this module.

    Parameters
    ----------
    use_same_cov_model_for_all_categories : bool
        indicates if the same covariance model is used for all categories;
        the parameters `cov_model` consists of a single entry (if `True`)
        or a list of `ncategory` entries (if `False`)

    space_dim : int
        space dimension (1, 2, or 3)

    ncategory : int
        number of categories

    categoryValue : array-like
        category values
        
    cov_model : [list of] :class:`geone.covModel.CovModel<d>D`
        covariance model (see `use_same_cov_model_for_all_categories`)

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

    varname : str
        variable name

    outputReportFile : str, optional
        name of the report file (if desired in output); by default (`None`): no
        report file

    computationMode : int
        computation mode:

        - `computationMode=0`: estimation, ordinary kriging
        - `computationMode=1`: estimation, simple kriging
        - `computationMode=2`: simulation, ordinary kriging
        - `computationMode=3`: simulation, simple kriging

    dataImage : sequence of :class:`geone.img.Img`, or `None`
        list of data image(s)

    dataPointSet : sequence of :class:`geone.img.PointSet`, or `None`
        list of data point set(s)

    mask : array-like, or `None`
        mask value in grid cells

    probability : sequence of floats, or sequence of array-like, or `None`
        probability (mean) for each category

    searchRadius : sequence of floats
        searchRadius parameter

    searchRadiusRelative : sequence of floats
        searchRadiusRelative parameter

    nneighborMax : sequence of ints
        nneighborMax parameter

    searchNeighborhoodSortMode : sequence of ints
        searchNeighborhoodSortMode parameter

    seed : int
        seed parameter

    nreal : int
        nreal parameter
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    mpds_geosClassicIndicatorInput : \\(MPDS_GEOSCLASSICINDICATORINPUT \\*\\)
        geosclassicIndicator input in C, intended for "GeosClassicIndicatorSim"
        C program
    """
    fname = 'fill_mpds_geosClassicIndicatorInput'

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

    if use_same_cov_model_for_all_categories:
        cov_model = ncategory * [cov_model]

    for i, cov_model_i in enumerate(cov_model):
        cov_model_c = geosclassic.malloc_MPDS_COVMODEL()
        geosclassic.MPDSGeosClassicInitCovModel(cov_model_c)

        try:
            if space_dim==1:
                cov_model_c = covModel1D_py2C(cov_model_i, nx, ny, nz, sx, sy, sz, ox, oy, oz, logger=logger)
            elif space_dim==2:
                if isinstance(cov_model_i, gcm.CovModel1D):
                    cov_model_i = gcm.covModel1D_to_covModel2D(cov_model_i) # convert model 1D in 2D
                    # -> will not be modified cov_model at exit
                cov_model_c = covModel2D_py2C(cov_model_i, nx, ny, nz, sx, sy, sz, ox, oy, oz, logger=logger)
            elif space_dim==3:
                if isinstance(cov_model_i, gcm.CovModel1D):
                    cov_model_i = gcm.covModel1D_to_covModel3D(cov_model_i) # convert model 1D in 3D
                    # -> will not be modified cov_model at exit
                cov_model_c = covModel3D_py2C(cov_model_i, nx, ny, nz, sx, sy, sz, ox, oy, oz, logger=logger)

        except Exception as exc:
            # Free memory on C side
            geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
            geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)
            # Raise error
            err_msg = f'{fname}: cannot fill mpds_geosClassicIndicatorInput C structure (covModel)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

        geosclassic.MPDS_COVMODEL_array_setitem(mpds_geosClassicIndicatorInput.covModel, i, cov_model_c)

    # mpds_geosClassicIndicatorInput.searchRadius
    searchRadius = np.asarray(searchRadius, dtype='float')
    mpds_geosClassicIndicatorInput.searchRadius = geosclassic.new_real_array(int(ncategory))
    geosclassic.mpds_set_real_vector_from_array(
        mpds_geosClassicIndicatorInput.searchRadius, 0,
        np.asarray(searchRadius).reshape(int(ncategory)))

    # mpds_geosClassicIndicatorInput.searchRadiusRelative
    searchRadiusRelative = np.asarray(searchRadiusRelative, dtype='float')
    mpds_geosClassicIndicatorInput.searchRadiusRelative = geosclassic.new_real_array(int(ncategory))
    geosclassic.mpds_set_real_vector_from_array(
        mpds_geosClassicIndicatorInput.searchRadiusRelative, 0,
        np.asarray(searchRadiusRelative).reshape(int(ncategory)))

    # mpds_geosClassicIndicatorInput.nneighborMax
    nneighborMax = np.asarray(nneighborMax, dtype='intc')
    mpds_geosClassicIndicatorInput.nneighborMax = geosclassic.new_int_array(int(ncategory))
    geosclassic.mpds_set_int_vector_from_array(
        mpds_geosClassicIndicatorInput.nneighborMax, 0,
        np.asarray(nneighborMax).reshape(int(ncategory)))

    # mpds_geosClassicIndicatorInput.searchNeighborhoodSortMode
    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode, dtype='intc')
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
            try:
                im_c = img_py2C(dataIm, logger=logger)
            except Exception as exc:
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
                geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)
                # Raise error
                err_msg = f'{fname}: cannot fill mpds_geosClassicIndicatorInput C structure (dataImage)'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

            geosclassic.MPDS_IMAGE_array_setitem(mpds_geosClassicIndicatorInput.dataImage, i, im_c)

    # mpds_geosClassicIndicatorInput.ndataPointSet and mpds_geosClassicIndicatorInput.dataPointSet
    if dataPointSet is None:
        mpds_geosClassicIndicatorInput.ndataPointSet = 0
    else:
        dataPointSet = np.asarray(dataPointSet).reshape(-1)
        n = len(dataPointSet)
        mpds_geosClassicIndicatorInput.ndataPointSet = n
        mpds_geosClassicIndicatorInput.dataPointSet = geosclassic.new_MPDS_POINTSET_array(n)
        for i, dataPS in enumerate(dataPointSet):
            try:
                ps_c = ps_py2C(dataPS, logger=logger)
            except Exception as exc:
                # Free memory on C side
                geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
                geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)
                # Raise error
                err_msg = f'{fname}: cannot fill mpds_geosClassicIndicatorInput C structure (dataPointSet)'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

            geosclassic.MPDS_POINTSET_array_setitem(mpds_geosClassicIndicatorInput.dataPointSet, i, ps_c)

    # mpds_geosClassicIndicatorInput.maskImageFlag and mpds_geosClassicIndicatorInput.maskImage
    if mask is None:
        mpds_geosClassicIndicatorInput.maskImageFlag = geosclassic.FALSE
    else:
        mpds_geosClassicIndicatorInput.maskImageFlag = geosclassic.TRUE
        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=1, val=mask,
                 logger=logger)
        try:
            mpds_geosClassicIndicatorInput.maskImage = img_py2C(im, logger=logger)
        except Exception as exc:
            # Free memory on C side
            geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
            geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)
            # Raise error
            err_msg = f'{fname}: cannot fill mpds_geosClassicIndicatorInput C structure (mask)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

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
                 nv=ncategory, val=probability,
                 logger=logger)
        try:
            mpds_geosClassicIndicatorInput.probabilityImage = img_py2C(im, logger=logger)
        except Exception as exc:
            # Free memory on C side
            geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
            geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)
            # Raise error
            err_msg = f'{fname}: cannot fill mpds_geosClassicIndicatorInput C structure (probabilityImage)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

    else:
        # Free memory on C side
        geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
        geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)
        # Raise error
        err_msg = f'{fname}: cannot fill mpds_geosClassicIndicatorInput C structure (`probability` not compatible with simulation grid)'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # mpds_geosClassicIndicatorInput.seed
    if seed is None:
        seed = np.random.randint(1, 1000000)
    mpds_geosClassicIndicatorInput.seed = int(seed)

    # mpds_geosClassicIndicatorInput.seedIncrement
    mpds_geosClassicIndicatorInput.seedIncrement = 1

    # mpds_geosClassicIndicatorInput.nrealization
    mpds_geosClassicIndicatorInput.nrealization = int(nreal)

    return mpds_geosClassicIndicatorInput
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def run_MPDSOMPGeosClassicIndicatorSim(
        use_same_cov_model_for_all_categories,
        space_dim,
        ncategory,
        categoryValue,
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
        probability,
        searchRadius,
        searchRadiusRelative,
        nneighborMax,
        searchNeighborhoodSortMode,
        seed,
        nreal,
        nthreads,
        progress_monitor=0,
        pid=None,
        verbose=0,
        logger=None):
    """
    Launches external C function `MPDSOMPGeosClassicIndicatorSim` (Geos-Classic library).

    This function should not be called directly, it is used in other functions
    of this module.

    All the parameters (except `use_same_cov_model_for_all_categories`, and `nthreads`, 
    `progress_monitor`, `pid`, `verbose` and `logger`) are the parameters for the input structure 
    `mpds_geosClassicInput` (C); `use_same_cov_model_for_all_categories` indicates if 
    the covariance model is given as a single entry or per category;
    `nthreads` is the number of threads used by the function `MPDSOMPGeosClassicSim`; 
    `progress_monitor` is an integer specifying the function used to update the progress 
    monitor (0 : no output, 1: warning only, 4: more detailed); 
    `pid` is the process id of the caller (int, used with multiprocessing);
    `verbose` is the verbosity (int).

    The function returns a dictionary `geosclassic_output` with:
    - `geosclassic_output['image']` : (class:`geone.img.Img`) output image
    - `geosclassic_output['nwarning']` : (int) total number of warnings encountered
    - `geosclassic_output['warnings']` : (list of strs) list of distinct warnings encountered
    """
    fname = 'run_MPDSOMPGeosClassicIndicatorSim'
    if pid is not None:
        fname = f'{fname} [pid={pid}]'

    if verbose > 2:
        if logger:
            logger.info(f'{fname}: Filling C structure for Geos-Classic program')
        else:
            print(f'{fname}: Filling C structure for Geos-Classic program')

    # --- Fill mpds_geosClassicIndicatorInput structure (C)
    try:
        mpds_geosClassicIndicatorInput = fill_mpds_geosClassicIndicatorInput(
                use_same_cov_model_for_all_categories,
                space_dim,
                ncategory,
                categoryValue,
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
                probability,
                searchRadius,
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                seed,
                nreal,
                logger=logger)

    except Exception as exc:
        err_msg = f'{fname}: cannot fill mpds_geosClassicIndicatorInput C structure'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg) from exc

    # --- Prepare mpds_geosClassicIOutput structure (C)
    # Allocate mpds_geosClassicOutput
    mpds_geosClassicOutput = geosclassic.malloc_MPDS_GEOSCLASSICOUTPUT()

    # Init mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicInitGeosClassicOutput(mpds_geosClassicOutput)

    # --- Set progress monitor
    mpds_progressMonitor = geosclassic.malloc_MPDS_PROGRESSMONITOR()
    geosclassic.MPDSInitProgressMonitor(mpds_progressMonitor)

    # Set function to update progress monitor
    # note: according to geosclassic.MPDS_SHOW_PROGRESS_MONITOR set to 4 for compilation of 
    #   py module, the function
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr
    # should be used, but the following function can also be used:
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr: no output
    #    mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor1_ptr: warning only
    if progress_monitor == 0:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
    elif progress_monitor == 1:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor1_ptr
    # elif progress_monitor == 2:
    #     mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor2_ptr
    # elif progress_monitor == 3:
    #     mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor3_ptr
    elif progress_monitor == 4:
        mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr
    else:
        err_msg = f'{fname}: `progress_monitor` not valid (should be 0, 1, or 4)'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    if verbose > 1:
        if logger:
            logger.info(  
                f'{fname}: Geos-Classic running... [' + \
                f'VERSION {geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER:s} / ' + \
                f'BUILD NUMBER {geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER:s} / ' + \
                f'OpenMP {nthreads:d} thread(s)]')
        else:
            print(  
                f'{fname}: Geos-Classic running... [' + \
                f'VERSION {geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER:s} / ' + \
                f'BUILD NUMBER {geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER:s} / ' + \
                f'OpenMP {nthreads:d} thread(s)]')
            sys.stdout.flush()
            sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicIndicatorSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nthreads)

    # Free memory on C side: mpds_geosClassicIndicatorInput
    geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
    geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)

    if err:
        # Free memory on C side: mpds_geosClassicOutput
        geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
        geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)
        # Free memory on C side: mpds_progressMonitor
        geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)
        # Raise error
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        err_msg = f'{fname}: {err_message}'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor, logger=logger)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if verbose > 1 and geosclassic_output:
        if logger:
            logger.info(f'{fname}: Geos-Classic run complete')
        else:
            print(f'{fname}: Geos-Classic run complete')

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def simulateIndicator(
        category_values,
        cov_model,
        dimension, 
        spacing=None, 
        origin=None,
        method='ordinary_kriging',
        x=None, 
        v=None,
        probability=None,
        alpha=None,
        beta=None,
        gamma=None,
        cov_model_non_stationarity_list=None,
        mask=None,
        add_data_point_to_mask=True,
        searchRadius=None,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        searchNeighborhoodSortMode=1,
        seed=None,
        nreal=1,
        outputReportFile=None,
        nproc=-1,
        nthreads_per_proc=-1,
        treat_image_one_by_one=False,
        verbose=2,
        logger=None):
    """
    Generates indicator simulations (Sequential Indicator Simulation, SIS) in a grid.

    A simulation takes place in (center of) grid cells, based on simple or
    ordinary kriging of the indicator variables of the categories. The space dimension 
    (1, 2, or 3) is detected (from the parameter `dimension`).

    The computation relies on a covariance model for each category value (indicator).
    The parameter `cov_model` is either one single covariance model (used for all categories)
    a list of `ncategory` covariance models (one per category). Accordingly, the parameters 
    `alpha`, `beta`, `gamma`, `cov_model_non_stationarity_list` are either given as one
    single entry or a list of `ncategory` entries (if not `None`).

    The conditioning data are treated as follows: (one of) the most frequent categories 
    of the data point(s) falling in the same grid cell is attributed to that cell.

    Parameters
    ----------
    category_values : 1D array-like
        sequence of category values; let `ncategory` be the number of categories,
        then:

        - if `ncategory=1`: the unique category value given must not be equal to \
        zero; it is used for a binary case with values "unique category value" \
        and 0, where 0 indicates the absence of the considered medium; the \
        conditioning data values should be equal to"unique category value" or 0
        - if `ncategory>=2`: it is used for a multi-category case with given \
        category values (distinct); the conditioning data values should be in the \
        `category_values`

    cov_model : [sequence of] :class:`geone.covModel.CovModel<d>D`
        sequence of same length as `category_values` of covariance model in 1D,
        or a unique covariance model in 1D (recycled):
        covariance model for each category;
        covariance model in 1D or 2D or 3D;      
        note: the covariance model must be stationary, however, non stationarity is
        handled: 

        - local rotation by specifying `alpha` (in 2D or 3D), `beta` (in 3D), `gamma` (in 3D)
        - other non-stationarities by specifying `cov_model_non_stationarity_list` (see below)

    dimension : [sequence of] int(s)
        number of cells along each axis, for simulation in:

        - 1D: `dimension=nx`
        - 2D: `dimension=(nx, ny)`
        - 3D: `dimension=(nx, ny, nz)`

        note : this parameter determines the space dimension (1, 2, or 3)

    spacing : [sequence of] float(s), optional
        cell size along each axis, for simulation in:

        - 1D: `spacing=sx`
        - 2D: `spacing=(sx, sy)`
        - 3D: `spacing=(sx, sy, sz)`

        by default (`None`): 1.0 along each axis

    origin : [sequence of] float(s), optional
        origin of the grid ("corner of the first cell"), for simulation in:

        - 1D: `origin=ox`
        - 2D: `origin=(ox, oy)`
        - 3D: `origin=(ox, oy, oz)`

        by default (`None`): 0.0 along each axis
        
    method : str {'simple_kriging', 'ordinary_kriging'}, default: 'ordinary_kriging'
        type of kriging

    x : 1D array-like of floats, optional
        data points locations (float coordinates); note: if one point, a float
        is accepted

    v : 1D array-like of floats, optional
        data values at `x` (`v[i]` is the data value at `x[i]`), array of same
        length as `x` (or float if one point)

    probability : array-like of floats, optional
        probability for each category:

        - sequence of same length as `category_values`: \
        probability[i]: probability (proportion, kriging mean value for the \
        indicator variable) for category `category_values[i]`, used for \
        every grid cell
        - array-like of size `ncategory * ngrid_cells`, where `ncategory` is the \
        length of `category_values` and `ngrid_cells` is the number of grid \
        cells (the array is reshaped if needed): first `ngrid_cells` values are \
        the probabilities (proportions, kriging mean values for the indicator \
        variable) for the first category at grid cells, etc. \
        (for non-stationary probailities / proportions)

        By default (`None`): proportion of each category computed from the
        data values (`v`) are used for every grid cell (uniform probabilities
        if no data point)

        Note: for ordinary kriging (`method='ordinary_kriging'`), it is used for
        case with no neighbor

    alpha : function (callable), or array-like of floats, or float, optional
        if specified: either a single entry or a list of entries of lenght `ncategory`
        (the length of `category_values`, according to the parameter `cov_model` (see 
        above);
        azimuth angle in degrees (see :class:`geone.covModel.CovModel<d>D`):

        - if a function: function of `d` argument(s) where `d` is the space dimension \
        that returns a value for each location given by their coordinate(s) in argument
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), values at grid cells
        - if a float: same value at every grid cell
        - by default (`None`): not used (use of covariance model only)

        note: not used in 1D

    beta : function (callable), or array-like of floats, or float, optional
        if specified: either a single entry or a list of entries of lenght `ncategory`
        (the length of `category_values`, according to the parameter `cov_model` (see 
        above);
        dip angle in degrees (see :class:`geone.covModel.CovModel<d>D`):

        - if a function: function of `d` argument(s) where `d` is the space dimension \
        that returns a value for each location given by their coordinate(s) in argument
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), values at grid cells
        - if a float: same value at every grid cell
        - by default (`None`): not used (use of covariance model only)

        note: not used in 1D or 2D

    gamma : function (callable), or array-like of floats, or float, optional
        if specified: either a single entry or a list of entries of lenght `ncategory`
        (the length of `category_values`, according to the parameter `cov_model` (see 
        above);
        plunge angle in degrees (see :class:`geone.covModel.CovModel<d>D`):

        - if a function: function of `d` argument(s) where `d` is the space dimension \
        that returns a value for each location given by their coordinate(s) in argument
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), values at grid cells
        - if a float: same value at every grid cell
        - by default (`None`): not used (use of covariance model only)

        note: not used in 1D or 2D

    cov_model_non_stationarity_list : list, optional
        if specified: either a single entry or a list of entries of lenght `ncategory`
        (the length of `category_values`, according to the parameter `cov_model` (see 
        above);
        list to set non-stationarities in covariance model; each entry must be
        a tuple (or list) `cm_ns` of length 2 or 3 with:

        - `cm_ns[0]`: str: the name of the method of `cov_model` to be applied
        - `cm_ns[1]`: function (callable), or array-like of floats, or float: \
        used to set the main parameter passed to the method:
            - if a function: function of `d` argument(s) where `d` is the space dimension \
            that returns a value for each location given by their coordinate(s) in argument
            - if array-like: its size must be equal to the number of grid cells \
            (the array is reshaped if needed), values at grid cells
            - if a float: same value at every grid cell
        - `cm_ns[2]`: dict, optional: keyworkds arguments to be passed to the method

        Examples (where the parameter `arg` is set from `val`):
        
        - `('multiply_w', val)` will apply `cov_model.multiply_w(arg)`; \
        this multipies the weight contribution of every elementary contribution of the \
        covariance model
        - `('multiply_w', val, {'elem_ind':0})` will apply `cov_model.multiply_w(arg, elem_ind=0)`; \
        this multipies the weight contribution of the elementary contribution of index 0 of the \
        covariance model
        - `('multiply_r', val)` will apply `cov_model.multiply_r(arg)`; \
        this multipies the range in all direction of every elementary contribution of the \
        covariance model
        - `('multiply_r', val, {'r_ind':0})` will apply `cov_model.multiply_r(arg, r_ind=0)`; \
        this multipies the range in the first main direction (index 0) of every elementary \
        contribution of the covariance model
        - `('multiply_r', val, {'r_ind':0, 'elem_ind':0})` will apply `cov_model.multiply_r(arg, r_ind=0, elem_ind=0)`; \
        this multipies the range in the first main direction (index 0) of the elementary \
        contribution of index 0 of the covariance model

    mask : array-like, optional
        mask value at grid cells (value 1 for simulated cells, value 0 for not
        simulated cells); the size of the array must be equal to the number of
        grid cells (the array is reshaped if needed)

    add_data_point_to_mask : bool, default: True
        - if `True`: any grid cell that contains a data point is added to (the \
        simulated part of) the mask (if present), i.e. mask value at those cells \
        are set to 1; at the end of the computation the "new mask cells" are \
        removed (by setting a missing value (`numpy.nan`) for the variable out of \
        the original mask)
        - if `False`: original mask is kept as given in input, and data point \
        falling out of (the simulated part of) the mask (if present) are ignored

    searchRadius : [sequence of] float(s), optional
        sequence of floats of same length as `category_values`, or
        a unique float (recycled); one parameter per category:
        if specified, i.e. not `None`: radius of the search neighborhood (ellipsoid
        with same radii along each axis), i.e. the data points at distance to the 
        estimated point greater than `searchRadius` are not taken into account 
        in the kriging system; if `searchRadius` is not `None`, then 
        `searchRadiusRelative` is not used;
        by default (`searchRadius=None`): `searchRadiusRelative` is used to 
        define the search ellipsoid;

    searchRadiusRelative : [sequence of] float(s), default: 1.2
        sequence of floats of same length as `category_values`, or
        a unique float (recycled); one parameter per category:
        used only if `searchRadius` is `None`;
        indicates how the search ellipsoid is limited (should be positive): let
        r_i be the ranges of the covariance model along its main axes, when
        estimating/simulating a cell x, a cell y is taken into account iff it is
        within the ellipsoid centered at x of half axes equal to
        `searchRadiusRelative` * r_i; 
        (note that the distances to the central node are computed in the axes 
        sytem supporting the covariance model and accounting for anisotropy given 
        by the ranges)
        note (Geos-Classic library): if a range r_i is non-stationary over the grid, 
        its maximal value over the grid is considered;

    nneighborMax : [sequence of] int(s), default: 12
        sequence of ints of same length as `category_values`, or
        a unique int (recycled); one parameter per category:
        maximal number of neighbors (informed cells) taken into account in the
        kriging system, must be greater than or equal to 1; the informed cells the 
        closest to the estimated cell are taken into account (see the parameters 
        `searchRadius`, `searchRadiusRelative`, `searchNeighborhoodSortMode`);

    searchNeighborhoodSortMode : [sequence of] int(s), default: 1
        sequence of ints of same length as `category_values`, or
        a unique int (recycled); one parameter per category:
        indicates how to sort the search neighboorhood cells (neighbors), for 
        external C function (Geos-Classic library); they are sorted in increasing 
        order according to:

        - `searchNeighborhoodSortMode=0`: distance in the usual axes system
        - `searchNeighborhoodSortMode=1`: distance in the axes sytem supporting \
        the covariance model and accounting for anisotropy given by the ranges \
        (if `searchRadius=None`)
        - `searchNeighborhoodSortMode=2`: minus the evaluation of the covariance \
        model

        Notes:

        - if the covariance model is non-stationary (see note for `cov_model`), then \
        `searchNeighborhoodSortMode=2` is not allowed
        - if the covariance model has any non-stationary range or non-stationary \
        angle and `searchNeighborhoodSortMode=1`: "maximal ranges" (adapted to \
        direction from the central cell) are used to compute distance for sorting \
        the neighbors

    seed : int, optional
        seed for initializing random number generator

    nreal : int, default: 1
        number of realizations

    outputReportFile : str, optional
        name of the report file (if desired in output); by default (`None`): no
        report file

    nproc : int, default: -1
        number of process(es) for external C function (Geos-Classic library): 
        a negative number, -n <= 0, can be specified to use the total number of cpu(s) 
        of the system except n; `nproc` is finally at maximum equal to `nreal` but at 
        least 1 by applying:
        
        - if `nproc >= 1`, then `nproc = max(min(nproc, nreal), 1)` is used
        - if `nproc = -n <= 0`, then `nproc = max(min(nmax-n, nreal), 1)` is used, \
        where nmax is the total number of cpu(s) of the system (retrieved by \
        `multiprocessing.cpu_count()`)

        note: if `nproc=None`, `nproc=-1` is used

    nthreads_per_proc : int, default: -1
        number of thread(s) per process for external C function (Geos-Classic library);
        if `nthreads_per_proc = -n <= 0`: `nthreads_per_proc` is automatically 
        computed as the maximal integer (but at least 1) such that 
        `nproc*nthreads_per_proc <= nmax-n`, where nmax is the total number of cpu(s)
        of the system (retrieved by `multiprocessing.cpu_count()`); 
        
        note: if `nthreads_per_proc=None`, `nthreads_per_proc=-1` is used

    treat_image_one_by_one : bool, default: False
        used only if multiprocessing is enabled  with `nproc > 1`;
        keyword argument passed to the function :func:`geone.img.gatherImages`:

        - if `True`: images (result of each process) are gathered one by one, \
        i.e. the variables of each image are inserted in an output image one by \
        one and removed from the source (slower, may save memory)
        - if `False`: images (result of each process) are gathered at once, \
        i.e. the variables of all images are inserted in an output image at once, \
        and then removed (faster)

    verbose : int, default: 2
        verbose mode, higher implies more printing (info):

        - 0: no display
        - 1: warnings
        - 2: warnings + basic info
        - 3 (or >2): all information

        note that if an error occurred, it is raised
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    geosclassic_output : dict
        geosclassic output in python, dictionary

        {'image':image, 'nwarning':nwarning, 'warnings':warnings}

        with:

        - image : :class:`geone.img.Img`
            output image, with `image.nv=nreal` variables (simulations);
            note: `image=None` if `mpds_geosClassicOutput->outputImage=NULL`

        - nwarning : int
            total number of warning(s) encountered (same warnings can be counted
            several times)

        - warnings : list of strs
            list of distinct warnings encountered (can be empty)
    """
    fname = 'simulateIndicator'

    # Number of realization : nreal
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose > 0:
            if logger:
                logger.warning(f'{fname}: `nreal` <= 0: `None` is returned')
            else:
                print(f'{fname}: WARNING: `nreal` <= 0: `None` is returned')
        return None

    # Set space dimension: d
    if hasattr(dimension, '__len__'):
        d = len(dimension)
    else:
        # assume dimension is an int, nx
        d = 1

    if d not in (1, 2, 3):
        err_msg = f'{fname}: space dimension not valid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check (or set) argument 'spacing'
    if spacing is None:
        spacing = tuple(np.ones(d))
    else:
        if hasattr(spacing, '__len__') and len(spacing) != d:
            err_msg = f'{fname}: `spacing` of incompatible length'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Check (or set) argument 'origin'
    if origin is None:
        origin = tuple(np.zeros(d))
    else:
        if hasattr(origin, '__len__') and len(origin) != d:
            err_msg = f'{fname}: `origin` of incompatible length'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Set grid geometry
    if d == 1:
        nx, ny, nz = dimension, 1, 1
        sx, sy, sz = spacing, 1.0, 1.0
        ox, oy, oz = origin, 0.0, 0.0
    elif d == 2:
        nx, ny, nz = *dimension, 1
        sx, sy, sz = *spacing, 1.0 
        ox, oy, oz = *origin, 0.0 
    else: # d == 3
        nx, ny, nz = dimension
        sx, sy, sz = spacing
        ox, oy, oz = origin

    nxy = nx * ny
    nxyz = nxy * nz

    # Check category values and compute number of category (ncategory
    try:
        category_values = np.asarray(category_values, dtype='float').reshape(-1)
    except:
        err_msg = f'{fname}: `category_values` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    ncategory = len(category_values)
    if ncategory <= 0:
        err_msg = f'{fname}: `category_values` is empty'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check dimension of cov_model for each category and set if used as omni-directional model
    cov_model = list(np.asarray(cov_model).reshape(-1))
    if len(cov_model) == 1:
        use_same_cov_model_for_all_categories = True
        cov_model = cov_model[0]
    elif len(cov_model) != ncategory:
        err_msg = f'{fname}: `cov_model_for_category` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)
    else:
        use_same_cov_model_for_all_categories = False

    if use_same_cov_model_for_all_categories:
        # Same covariance model is used for all categories
        # ------------------------------------------------
        # -> "single entries" for the parameters below ...
        # Check dimension of cov_model and set if used as omni-directional model
        if isinstance(cov_model, gcm.CovModel1D):
            omni_dir = True
        else:
            if cov_model.__class__.__name__ != f'CovModel{d}D':
                err_msg = f'{fname}: `cov_model` dimension is incompatible with `dimension`'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            omni_dir = False

        # Prevent calculation if covariance model is not stationary
        if not cov_model.is_stationary():
            err_msg = f'{fname}: `cov_model` is not stationary: {fname} cannot be applied (use the other paramters for non-stationary covariance)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # Check parameters for non-stationarity
        # ... rotation given by `alpha`, `beta`, `gamma`
        if alpha is not None:
            if omni_dir:
                err_msg = f'{fname}: `alpha` cannot be used with 1D covariance model'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
            if not callable(alpha) and not isinstance(alpha, np.ndarray) and not isinstance(alpha, float) and not isinstance(alpha, int):
                err_msg = f'{fname}: `alpha` not valid: should be callable or array or float or int'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

        if beta is not None:
            if omni_dir or d < 3:
                err_msg = f'{fname}: `beta` cannot be used with 1D or 2D covariance model'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
            if not callable(beta) and not isinstance(beta, np.ndarray) and not isinstance(beta, float) and not isinstance(beta, int):
                err_msg = f'{fname}: `beta` not valid: should be callable or array or float or int'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
        
        if gamma is not None:
            if omni_dir or d < 3:
                err_msg = f'{fname}: `gamma` cannot be used with 1D or 2D covariance model'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
            if not callable(gamma) and not isinstance(gamma, np.ndarray) and not isinstance(gamma, float) and not isinstance(gamma, int):
                err_msg = f'{fname}: `gamma` not valid: should be callable or array or float or int'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

        # ... other non-stationarities given by cov_model_non_stationarity_list
        if cov_model_non_stationarity_list is not None:
            if not isinstance(cov_model_non_stationarity_list, list):
                err_msg = f'{fname}: `cov_model_non_stationarity_list` must be a list if not `None`'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
            for i, cm_ns in enumerate(cov_model_non_stationarity_list):
                if not hasattr(cm_ns, '__len__') or len(cm_ns) not in (2, 3):
                    err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list` not valid: should be a tuple or list of length 2 or 3 containing: method name (str), value(s) or callable (for arg)[, kwds (dict)]'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)
                if not isinstance(cm_ns[0], str):
                    err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list`, `(method_name, val[, kwds])`: `method_name` not valid: should be a string'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)
                if not callable(cm_ns[1]) and not isinstance(cm_ns[1], np.ndarray) and not isinstance(cm_ns[1], float) and not isinstance(cm_ns[1], int):
                    err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list`, `(method_name, val[, kwds])`: `val` not valid: should be callable or array or float or int'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)
                if len(cm_ns) == 3 and not isinstance(cm_ns[2], dict):
                    err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list`, `(method_name, val[, kwds])`: `kwds` not valid: should be a dictionary'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

    else:
        # One covariance model per categories (ncategory at all)
        # ------------------------------------------------------
        # -> "multiple entries" for the parameters below ...
        # Check dimension of cov_model and set if used as omni-directional model
        omni_dir = np.zeros(ncategory, dtype='bool')
        for k, cov_model_k in enumerate(cov_model):
            if isinstance(cov_model_k, gcm.CovModel1D):
                omni_dir[k] = True
            elif cov_model_k.__class__.__name__ != f'CovModel{d}D':
                err_msg = f'{fname}: `cov_model` (index {k}) dimension is incompatible with `dimension`'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            # Prevent calculation if covariance model is not stationary
            if not cov_model_k.is_stationary():
                err_msg = f'{fname}: `cov_model` (index {k}) is not stationary: {fname} cannot be applied (use the other paramters for non-stationary covariance)'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

        # Check parameters for non-stationarity
        # ... rotation given by `alpha`, `beta`, `gamma`
        # alpha
        if alpha is None:
            alpha = ncategory * [None]
        
        elif not isinstance(alpha, list) or len(alpha) != ncategory:
            err_msg = f'{fname}: `alpha` should be a list of length equal to the number of categories (when covariance model is set per category)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        else:
            for k, alpha_k in enumerate(alpha):
                if omni_dir:
                    err_msg = f'{fname}: `alpha` (index {k}) cannot be used with 1D covariance model'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)
                if not callable(alpha_k) and not isinstance(alpha_k, np.ndarray) and not isinstance(alpha_k, float) and not isinstance(alpha_k, int):
                    err_msg = f'{fname}: `alpha` (index {k}) not valid: should be callable or array or float or int'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

        # beta
        if beta is None:
            beta = ncategory * [None]
        
        elif not isinstance(beta, list) or len(beta) != ncategory:
            err_msg = f'{fname}: `beta` should be a list of length equal to the number of categories (when covariance model is set per category)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        else:
            if not isinstance(beta, list) or len(beta) != ncategory:
                err_msg = f'{fname}: `beta` should be a list of length equal to the number of categories (when covariance model is set per category)'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
            
            for k, beta_k in enumerate(beta):
                if omni_dir or d < 3:
                    err_msg = f'{fname}: `beta` (index {k}) cannot be used with 1D or 2D covariance model'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)
                if not callable(beta_k) and not isinstance(beta_k, np.ndarray) and not isinstance(beta_k, float) and not isinstance(beta_k, int):
                    err_msg = f'{fname}: `beta` (index {k}) not valid: should be callable or array or float or int'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

        # gamma
        if gamma is None:
            gamma = ncategory * [None]
        
        elif not isinstance(gamma, list) or len(gamma) != ncategory:
            err_msg = f'{fname}: `gamma` should be a list of length equal to the number of categories (when covariance model is set per category)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        else:
            if not isinstance(gamma, list) or len(gamma) != ncategory:
                err_msg = f'{fname}: `gamma` should be a list of length equal to the number of categories (when covariance model is set per category)'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
            for k, gamma_k in enumerate(gamma):
                if omni_dir or d < 3:
                    err_msg = f'{fname}: `gamma` (index {k}) cannot be used with 1D or 2D covariance model'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)
                if not callable(gamma_k) and not isinstance(gamma_k, np.ndarray) and not isinstance(gamma_k, float) and not isinstance(gamma_k, int):
                    err_msg = f'{fname}: `gamma` (index {k}) not valid: should be callable or array or float or int'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

        # ... other non-stationarities given by cov_model_non_stationarity_list
        if cov_model_non_stationarity_list is None:
            cov_model_non_stationarity_list = ncategory * [None]
        
        elif not isinstance(cov_model_non_stationarity_list, list) or len(cov_model_non_stationarity_list) != ncategory:
            err_msg = f'{fname}: `cov_model_non_stationarity_list` should be a list (of list) of length equal to the number of categories (when covariance model is set per category)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        else:
            for k, cov_model_non_stationarity_list_k in enumerate(cov_model_non_stationarity_list):
                if cov_model_non_stationarity_list_k is not None:
                    if not isinstance(cov_model_non_stationarity_list_k, list):
                        err_msg = f'{fname}: `cov_model_non_stationarity_list` (index {k}) must be a list if not `None`'
                        if logger: logger.error(err_msg)
                        raise GeosclassicinterfaceError(err_msg)
                    
                    for i, cm_ns in enumerate(cov_model_non_stationarity_list_k):
                        if not hasattr(cm_ns, '__len__') or len(cm_ns) not in (2, 3):
                            err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list` (index {k}) not valid: should be a tuple or list of length 2 or 3 containing: method name (str), value(s) or callable (for arg)[, kwds (dict)]'
                            if logger: logger.error(err_msg)
                            raise GeosclassicinterfaceError(err_msg)
                        if not isinstance(cm_ns[0], str):
                            err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list` (index {k}), `(method_name, val[, kwds])`: `method_name` not valid: should be a string'
                            if logger: logger.error(err_msg)
                            raise GeosclassicinterfaceError(err_msg)
                        if not callable(cm_ns[1]) and not isinstance(cm_ns[1], np.ndarray) and not isinstance(cm_ns[1], float) and not isinstance(cm_ns[1], int):
                            err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list` (index {k}), `(method_name, val[, kwds])`: `val` not valid: should be callable or array or float or int'
                            if logger: logger.error(err_msg)
                            raise GeosclassicinterfaceError(err_msg)
                        if len(cm_ns) == 3 and not isinstance(cm_ns[2], dict):
                            err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list` (index {k}), `(method_name, val[, kwds])`: `kwds` not valid: should be a dictionary'
                            if logger: logger.error(err_msg)
                            raise GeosclassicinterfaceError(err_msg)

    # Set parameters for neighborhood
    # - searchRadius, searchRadius_geosClassic
    # - searchRadiusRelative, searchRadiusRelative_geosClassic
    # - nneighborMax, nneighborMax_geosClassic
    # - searchNeighborhoodSortMode
    if searchRadius is None:
        searchRadius = ncategory * [None]
    elif isinstance(searchRadius, float) or isinstance(searchRadius, int):
        searchRadius = ncategory * [searchRadius]
    elif not isinstance(searchRadius, list) or len(searchRadius) != ncategory:
        err_msg = f'{fname}: `searchRadius` should be `None`, a single entry or a list of length equal to the number of categories'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)
    
    if searchRadiusRelative is None:
        searchRadiusRelative = ncategory * [None]
    elif isinstance(searchRadiusRelative, float) or isinstance(searchRadiusRelative, int):
        searchRadiusRelative = ncategory * [searchRadiusRelative]
    elif not isinstance(searchRadiusRelative, list) or len(searchRadiusRelative) != ncategory:
        err_msg = f'{fname}: `searchRadiusRelative` should be a list of length equal to the number of categories'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)
    
    if isinstance(nneighborMax, int):
        nneighborMax = ncategory * [nneighborMax]          
    elif not isinstance(nneighborMax, list) or len(nneighborMax) != ncategory:
        err_msg = f'{fname}: `nneighborMax` should be a single entry or a list of length equal to the number of categories'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # searchNeighborhoodSortMode
    if searchNeighborhoodSortMode is None:
        searchNeighborhoodSortMode = ncategory * [1]
    elif isinstance(searchNeighborhoodSortMode, int):
        searchNeighborhoodSortMode = ncategory * [searchNeighborhoodSortMode]
    elif not isinstance(searchNeighborhoodSortMode, list) or len(searchNeighborhoodSortMode) != ncategory:
        err_msg = f'{fname}: `searchNeighborhoodSortMode` should be a single entry or a list of length equal to the number of categories'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    searchRadius_geosClassic = ncategory * [0.0]
    searchRadiusRelative_geosClassic = ncategory * [0.0]
    nneighborMax_geosClassic = ncategory * [0]
    for k in range(ncategory):
        if searchRadius[k] is not None:
            # searchRadius[k] is used, searchRadiusRelative[k] is not used
            if searchRadius[k] <= 0.0:
                err_msg = f'{fname}: search radius (index {k}) (isotropic neighborhood) not valid (negative)'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
            
            searchRadius_geosClassic[k] = searchRadius[k]
            searchRadiusRelative_geosClassic[k] = searchRadiusRelative[k] # unused

        else:
            # searchRadius[k] is not used, searchRadiusRelative is used
            searchRadius_geosClassic[k] = 0.0 # to indicate in external C function that search radius is not used
            searchRadiusRelative_geosClassic[k] = searchRadiusRelative[k]

            # searchRadiusRelative
            if searchRadiusRelative_geosClassic[k] < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
                err_msg = f'{fname}: `searchRadiusRelative` (index {k}) too small (should be at least {geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN})'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

        if nneighborMax[k] <= 0:
            # (note : nneighborMax = -1 is authorized in Geos-Classic)
            err_msg = f'{fname}: `nneighborMax` (index {k}) should be greater than or equal to 1'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)
        
        nneighborMax_geosClassic[k] = nneighborMax[k]
        
        if searchNeighborhoodSortMode[k] is None:
            searchNeighborhoodSortMode[k] = 1
            # # Set greatest possible value
            # searchNeighborhoodSortMode[k] = 2
            # if use_same_cov_model_for_all_categories:
            #     if         (alpha is not None and (callable(alpha) or (isinstance(alpha, np.ndarray) and alpha.size > 1))) \
            #             or (beta  is not None and (callable(beta ) or (isinstance(beta,  np.ndarray) and beta.size  > 1))) \
            #             or (gamma is not None and (callable(gamma) or (isinstance(gamma, np.ndarray) and gamma.size > 1))):
            #         searchNeighborhoodSortMode[k] = 1
            # else:
            #     if         (alpha[k] is not None and (callable(alpha[k]) or (isinstance(alpha[k], np.ndarray) and alpha[k].size > 1))) \
            #             or (beta[k]  is not None and (callable(beta[k] ) or (isinstance(beta[k],  np.ndarray) and beta[k].size  > 1))) \
            #             or (gamma[k] is not None and (callable(gamma[k]) or (isinstance(gamma[k], np.ndarray) and gamma[k].size > 1))):
            #         searchNeighborhoodSortMode[k] = 1
            # if cov_model_non_stationarity_list[k] is not None:
            #     for cm_ns in cov_model_non_stationarity_list[k]:
            #         if callable(cm_ns[1]) or (isinstance(cm_ns[1],  np.ndarray) and cm_ns[1].size  > 1):
            #             searchNeighborhoodSortMode[k] = 1
            #             break
        elif searchNeighborhoodSortMode[k] == 2:
            if use_same_cov_model_for_all_categories:
                if         (alpha is not None and (callable(alpha) or (isinstance(alpha, np.ndarray) and alpha.size > 1))) \
                        or (beta  is not None and (callable(beta ) or (isinstance(beta,  np.ndarray) and beta.size  > 1))) \
                        or (gamma is not None and (callable(gamma) or (isinstance(gamma, np.ndarray) and gamma.size > 1))):
                    err_msg = f'{fname}: `searchNeighborhoodSortMode=2` (index {k}) not allowed with non-stationary covariance model (local rotation)'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)
            else:
                if         (alpha[k] is not None and (callable(alpha[k]) or (isinstance(alpha[k], np.ndarray) and alpha[k].size > 1))) \
                        or (beta[k]  is not None and (callable(beta[k] ) or (isinstance(beta[k],  np.ndarray) and beta[k].size  > 1))) \
                        or (gamma[k] is not None and (callable(gamma[k]) or (isinstance(gamma[k], np.ndarray) and gamma[k].size > 1))):
                    err_msg = f'{fname}: `searchNeighborhoodSortMode=2` (index {k}) not allowed with non-stationary covariance model (local rotation)'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)
            
            if cov_model_non_stationarity_list[k] is not None:
                for cm_ns in cov_model_non_stationarity_list[k]:
                    if callable(cm_ns[1]) or (isinstance(cm_ns[1],  np.ndarray) and cm_ns[1].size  > 1):
                        err_msg = f'{fname}: `searchNeighborhoodSortMode=2` (index {k}) not allowed with non-stationary covariance model ({cm_ns[0]})'
                        if logger: logger.error(err_msg)
                        raise GeosclassicinterfaceError(err_msg)
                    
        elif searchNeighborhoodSortMode[k] not in (0, 1):
            err_msg = f'{fname}: `searchNeighborhoodSortMode` (index {k}) not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Set varname
    varname = 'V0'

    # Set probability
    if probability is not None:
        # if method == 'ordinary_kriging':
        #     err_msg = f"{fname}: specifying 'probability' not allowed with ordinary kriging"
        #     if logger: logger.error(err_msg)
        #     raise GeosclassicinterfaceError(err_msg)
        probability = np.asarray(probability, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if probability.size not in (ncategory, ncategory*nxyz):
            err_msg = f'{fname}: size of `probability` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Set method : computationMode
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    if method == 'simple_kriging':
        computationMode = 3
    elif method == 'ordinary_kriging':
        computationMode = 2
    else:
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Set dataPointSet from data points: x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, d) # cast in 2-dimensional array if needed

        # Check if all data points are in the grid
        if verbose > 0:
            if d == 1:
                if np.any(x[:, 0] < ox) or np.any(x[:, 0] > ox + nx*sx):
                    if logger:
                        logger.warning(f'{fname}: data point(s) out of the grid')
                    else:
                        print(f'{fname}: WARNING: data point(s) out of the grid')
            elif d == 2:
                if    np.any(x[:, 0] < ox) or np.any(x[:, 0] > ox + nx*sx) \
                   or np.any(x[:, 1] < oy) or np.any(x[:, 1] > oy + ny*sy):
                    if logger:
                        logger.warning(f'{fname}: data point(s) out of the grid')
                    else:
                        print(f'{fname}: WARNING: data point(s) out of the grid')
            else: # d == 3
                if    np.any(x[:, 0] < ox) or np.any(x[:, 0] > ox + nx*sx) \
                   or np.any(x[:, 1] < oy) or np.any(x[:, 1] > oy + ny*sy) \
                   or np.any(x[:, 2] < oz) or np.any(x[:, 2] > oz + nz*sz):
                    if logger:
                        logger.warning(f'{fname}: data point(s) out of the grid')
                    else:
                        print(f'{fname}: WARNING: data point(s) out of the grid')

        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        if ncategory > 1:
            if not np.all([vi in category_values for vi in v]):
                err_msg = f'{fname}: `v` contains an invalid category value (not in `category_values`)'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
        else:
            if not np.all([vi in np.hstack((category_values, np.array([0.0]))) for vi in v]):
                err_msg = f'{fname}: `v` contains an invalid category value (not in `category_values` nor 0.0)'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

        # Aggregate data on grid by taking the most frequent value in grid cell
        xx, yy, zz = np.hstack((x, np.zeros((len(x), 3-d)))).T
        try:
            xx_agg, yy_agg, zz_agg, v_agg = img.aggregateDataPointsWrtGrid(
                                                xx, yy, zz, v,
                                                nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                op='most_freq',
                                                logger=logger)
        except Exception as exc:
            err_msg = f"{fname}: data aggregation ('most_freq') failed"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

        if len(xx_agg) == 0:
            err_msg = f'{fname}: no data point in grid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        dataPointSet = PointSet(npt=v_agg.shape[0], nv=4, val=np.vstack((xx_agg, yy_agg, zz_agg, v_agg)), varname=['X', 'Y', 'Z', varname])
    else:
        dataPointSet = None

    # Set mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            err_msg = f'{fname}: `mask` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if mask is not None and add_data_point_to_mask and len(dataPointSet):
        # Make a copy of the original mask, to remove value in added mask cell at the end
        mask_original = np.copy(mask)
        # Add cell to mask if needed
        if dataPointSet[0].npt:
            im_tmp = img.imageFromPoints(dataPointSet[0].val[:3, :].T,
                        nx=nx, ny=ny, nz=nz,
                        sx=sx, sy=sy, sz=sz,
                        ox=ox, oy=oy, oz=oz,
                        indicator_var=True,
                        logger=logger)
            mask = 1.0*np.any((im_tmp.val[0], mask), axis=0)
            del(im_tmp)
        recover_mask = True
    
    else:
        recover_mask = False

    if use_same_cov_model_for_all_categories:
        # Same covariance model is used for all categories
        # ------------------------------------------------
        # Set non-stationary covariance model (if needed)
        cov_model_grid = gcm.copyCovModel(cov_model)
        # ... from alpha
        if alpha is not None:
            if d == 1:
                vv = gcm.eval_in_grid_1D(alpha, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                vv = gcm.eval_in_grid_2D(alpha, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                vv = gcm.eval_in_grid_3D(alpha, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
            cov_model_grid.set_alpha(vv)

        # ... from beta
        if beta is not None:
            if d == 1:
                vv = gcm.eval_in_grid_1D(beta, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                vv = gcm.eval_in_grid_2D(beta, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                vv = gcm.eval_in_grid_3D(beta, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
            cov_model_grid.set_beta(vv)

        # ... from gamma
        if gamma is not None:
            if d == 1:
                vv = gcm.eval_in_grid_1D(gamma, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                vv = gcm.eval_in_grid_2D(gamma, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                vv = gcm.eval_in_grid_3D(gamma, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
            cov_model_grid.set_gamma(vv)

        # ... from cov_model_non_stationarity_list
        if cov_model_non_stationarity_list is not None:
            for cm_ns in cov_model_non_stationarity_list:
                try:
                    method_name = cm_ns[0]
                    val = cm_ns[1]
                    if len(cm_ns) == 3:
                        kwds = cm_ns[2]
                    else:
                        kwds = {}
                    if d == 1:
                        arg = gcm.eval_in_grid_1D(val, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                    elif d == 2:
                        arg = gcm.eval_in_grid_2D(val, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                    else: # d == 3
                        arg = gcm.eval_in_grid_3D(val, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                    eval(f'cov_model_grid.{method_name}')(arg, **kwds)
                except:
                    err_msg = f'{fname}: cannot apply non-stationarity (in grid) for covariance model (check parameter `cov_model_non_stationarity_list`)'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)
    else:
        # One covariance model per categories (ncategory at all)
        # ------------------------------------------------------
        # Set non-stationary covariance model (if needed)
        cov_model_grid = [gcm.copyCovModel(cov_model_k) for cov_model_k in cov_model]
        for k in range(ncategory):
            # ... from alpha
            if alpha[k] is not None:
                if d == 1:
                    vv = gcm.eval_in_grid_1D(alpha[k], nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                elif d == 2:
                    vv = gcm.eval_in_grid_2D(alpha[k], nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                else: # d == 3
                    vv = gcm.eval_in_grid_3D(alpha[k], nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                cov_model_grid[k].set_alpha(vv)

            # ... from beta
            if beta[k] is not None:
                if d == 1:
                    vv = gcm.eval_in_grid_1D(beta[k], nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                elif d == 2:
                    vv = gcm.eval_in_grid_2D(beta[k], nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                else: # d == 3
                    vv = gcm.eval_in_grid_3D(beta[k], nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                cov_model_grid[k].set_beta(vv)

            # ... from gamma
            if gamma[k] is not None:
                if d == 1:
                    vv = gcm.eval_in_grid_1D(gamma[k], nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                elif d == 2:
                    vv = gcm.eval_in_grid_2D(gamma[k], nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                else: # d == 3
                    vv = gcm.eval_in_grid_3D(gamma[k], nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                cov_model_grid[k].set_gamma(vv)

            # ... from cov_model_non_stationarity_list
            if cov_model_non_stationarity_list[k] is not None:
                for cm_ns in cov_model_non_stationarity_list[k]:
                    try:
                        method_name = cm_ns[0]
                        val = cm_ns[1]
                        if len(cm_ns) == 3:
                            kwds = cm_ns[2]
                        else:
                            kwds = {}
                        if d == 1:
                            arg = gcm.eval_in_grid_1D(val, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                        elif d == 2:
                            arg = gcm.eval_in_grid_2D(val, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                        else: # d == 3
                            arg = gcm.eval_in_grid_3D(val, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                        eval(f'cov_model_grid[k].{method_name}')(arg, **kwds)
                    except:
                        err_msg = f'{fname}: cannot apply non-stationarity (in grid) for covariance model (index {k}) (check parameter `cov_model_non_stationarity_list`)'
                        if logger: logger.error(err_msg)
                        raise GeosclassicinterfaceError(err_msg)

    # Set seed
    if seed is None:
        seed = np.random.randint(1, 1000000)
    seed = int(seed)

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
                logger.info(f'{fname}: number of process(es) has been changed (now: nproc={nproc})')
            else:
                print(f'{fname}: number of process(es) has been changed (now: nproc={nproc})')

    # Set number of threads per process: nth
    if nthreads_per_proc is None:
        nthreads_per_proc = -1
    
    if nthreads_per_proc <= 0:
        nth = max(int(np.floor((multiprocessing.cpu_count() + nthreads_per_proc) / nproc)), 1)
    else:
        nth = int(nthreads_per_proc)
        # if verbose > 1 and nth != nthreads_per_proc:
        #     if logger:
        #         logger.info(f'{fname}: number of threads per process has been changed (now: nthreads_per_proc={nth})')
        #     else:
        #         print(f'{fname}: number of threads per process has been changed (now: nthreads_per_proc={nth})')

    if verbose > 0 and nproc * nth > multiprocessing.cpu_count():
        if logger:
            logger.warning(f'{fname}: total number of cpu(s) used ({nproc * nth}) will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: total number of cpu(s) used ({nproc * nth}) will exceed number of cpu(s) of the system...')

    if nproc > 1:
        # multiprocessing enabled
        # -----------------------
        # Set the distribution of the realizations over the processes
        # Condider the Euclidean division of nreal by nproc:
        #     nreal = q * nproc + r, with 0 <= r < nproc
        # Then, (q+1) realizations will be done on process 0, 1, ..., r-1, and q realization on process r, ..., nproc-1
        # Define the list real_index_proc of length (nproc+1) such that
        #   real_index_proc[i], ..., real_index_proc[i+1] - 1 : are the realization indices run on process i
        q, r = np.divmod(nreal, nproc)
        real_index_proc = [i*q + min(i, r) for i in range(nproc+1)]

        if verbose > 1:
            if logger:
                logger.info(  
                    f'{fname}: Geos-Classic running on {nproc} process(es)... [' + \
                    f'VERSION {geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER:s} / ' + \
                    f'BUILD NUMBER {geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER:s} / ' + \
                    f'OpenMP {nth:d} thread(s)]')
            else:
                print(  
                    f'{fname}: Geos-Classic running on {nproc} process(es)... [' + \
                    f'VERSION {geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER:s} / ' + \
                    f'BUILD NUMBER {geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER:s} / ' + \
                    f'OpenMP {nth:d} thread(s)]')

        outputReportFile_p = None

        # Set pool of nproc workers
        pool = multiprocessing.Pool(nproc)
        out_pool = []
        for i in range(nproc):
            # Set input for i-th process
            nreal_p = real_index_proc[i+1] - real_index_proc[i]
            seed_p = seed + real_index_proc[i]
            
            if outputReportFile is not None:
                outputReportFile_p = outputReportFile + f'.{i}'
            
            if i == 0 and verbose >= 3:
                progress_monitor = 4
            else:
                progress_monitor = 0

            args = (
                    use_same_cov_model_for_all_categories,
                    d,
                    ncategory,
                    category_values,
                    cov_model_grid,
                    nx, ny, nz,
                    sx, sy, sz,
                    ox, oy, oz,
                    varname,
                    outputReportFile_p, # adapted to pid
                    computationMode,
                    None,
                    dataPointSet,
                    mask,
                    probability,
                    searchRadius_geosClassic,
                    searchRadiusRelative_geosClassic,
                    nneighborMax_geosClassic,
                    searchNeighborhoodSortMode,
                    seed_p,  # adapted to pid
                    nreal_p, # adapted to pid
                    nth
                )
            kwds = {
                    'progress_monitor':progress_monitor, 
                    'pid':i,
                    'verbose':verbose*(i==0),
                    'logger':logger
                }

            # Launch geos-classic (i-th process)
            out_pool.append(pool.apply_async(
                run_MPDSOMPGeosClassicIndicatorSim, args=args, kwds=kwds
            ))

        # Properly end working process
        pool.close() # Prevents any more tasks from being submitted to the pool,
        pool.join()  # then, wait for the worker processes to exit.

        # Get result from each process
        geosclassic_output_proc = [p.get() for p in out_pool]

        if np.any([out is None for out in geosclassic_output_proc]):
            err_msg = f'{fname}: async `run_MPDSOMPGeosClassicIndicatorSim` failed'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # Gather results from every process
        # image
        image_list = []
        for out in geosclassic_output_proc:
            if out['image'] is not None:
                image_list.append(out['image'])
                del(out['image'])
        if len(image_list):
            # Gather images and adjust variable names
            image = img.gatherImages(image_list, keep_varname=True, rem_var_from_source=True, treat_image_one_by_one=treat_image_one_by_one, logger=logger)
            ndigit = geosclassic.MPDS_GEOS_CLASSIC_NB_DIGIT_FOR_REALIZATION_NUMBER
            for j in range(image.nv):
                image.varname[j] = image.varname[j][:-ndigit] + f'{j:0{ndigit}d}'
        else:
            image = None

        # nwarning
        nwarning = int(np.sum([out['nwarning'] for out in geosclassic_output_proc]))
        # warnings
        warnings = list(np.unique(np.hstack([out['warnings'] for out in geosclassic_output_proc])))

        geosclassic_output = {'image':image, 'nwarning':nwarning, 'warnings':warnings}

        if verbose > 1 and geosclassic_output:
            if logger:
                logger.info(f'{fname}: Geos-Classic ({nproc} process(es)) run complete')
            else:
                print(f'{fname}: Geos-Classic ({nproc} process(es)) run complete')

    else:
        # multiprocessing not enabled
        # ---------------------------
        if verbose > 1:
            if logger:
                logger.info(  
                    f'{fname}: Geos-Classic running on 1 process... [' + \
                    f'VERSION {geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER:s} / ' + \
                    f'BUILD NUMBER {geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER:s} / ' + \
                    f'OpenMP {nth:d} thread(s)]')
            else:
                print(  
                    f'{fname}: Geos-Classic running on 1 process... [' + \
                    f'VERSION {geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER:s} / ' + \
                    f'BUILD NUMBER {geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER:s} / ' + \
                    f'OpenMP {nth:d} thread(s)]')

        if verbose < 3:
            progress_monitor = 0
        else:
            progress_monitor = 4

        args = (
                use_same_cov_model_for_all_categories,
                d,
                ncategory,
                category_values,
                cov_model_grid,
                nx, ny, nz,
                sx, sy, sz,
                ox, oy, oz,
                varname,
                outputReportFile,
                computationMode,
                None,
                dataPointSet,
                mask,
                probability,
                searchRadius_geosClassic,
                searchRadiusRelative_geosClassic,
                nneighborMax_geosClassic,
                searchNeighborhoodSortMode,
                seed,
                nreal,
                nth
            )
        kwds = {
                'progress_monitor':progress_monitor, 
                'pid':None,
                'verbose':verbose,
                'logger':logger
            }
        geosclassic_output = run_MPDSOMPGeosClassicIndicatorSim(*args, **kwds)

        if verbose > 1 and geosclassic_output:
            if logger:
                logger.info(f'{fname}: Geos-Classic (1 process) run complete')
            else:
                print(f'{fname}: Geos-Classic (1 process) run complete')

    if geosclassic_output is not None and recover_mask:
        # Remove the value out of the original mask (using its copy see above)
        geosclassic_output['image'].val[:, mask_original==0.0] = np.nan

    # Show (print) encountered warnings
    if verbose > 0 and geosclassic_output and geosclassic_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({geosclassic_output['nwarning']} times in all):")
            for i, warning_message in enumerate(geosclassic_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def estimateIndicator(
        category_values,
        cov_model,
        dimension, 
        spacing=None, 
        origin=None,
        method='ordinary_kriging',
        x=None, 
        v=None,
        probability=None,
        alpha=None,
        beta=None,
        gamma=None,
        cov_model_non_stationarity_list=None,
        mask=None,
        add_data_point_to_mask=True,
        use_unique_neighborhood=False,
        searchRadius=None,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        searchNeighborhoodSortMode=1,
        outputReportFile=None,
        nthreads=-1,
        verbose=2,
        logger=None):
    """
    Generates estimate probabilities / proportions of categories (indicators) in a grid.

    Interpolation takes place in (center of) grid cells, based on simple or
    ordinary kriging. The space dimension (1, 2, or 3) is detected (from the 
    parameter `dimension`).

    The computation relies on a covariance model for each category value (indicator).
    The parameter `cov_model` is either one single covariance model (used for all categories)
    a list of `ncategory` covariance models (one per category). Accordingly, the parameters 
    `alpha`, `beta`, `gamma`, `cov_model_non_stationarity_list` are either given as one
    single entry or a list of `ncategory` entries (if not `None`).

    The conditioning data are treated as follows: (one of) the most frequent categories 
    of the data point(s) falling in the same grid cell is attributed to that cell.

    Parameters
    ----------
    category_values : 1D array-like
        sequence of category values; let `ncategory` be the number of categories,
        then:

        - if `ncategory=1`: the unique category value given must not be equal to \
        zero; it is used for a binary case with values "unique category value" \
        and 0, where 0 indicates the absence of the considered medium; the \
        conditioning data values should be equal to"unique category value" or 0
        - if `ncategory>=2`: it is used for a multi-category case with given \
        category values (distinct); the conditioning data values should be in the \
        `category_values`

    cov_model : [sequence of] :class:`geone.covModel.CovModel<d>D`
        sequence of same length as `category_values` of covariance model in 1D,
        or a unique covariance model in 1D (recycled):
        covariance model for each category;
        covariance model in 1D or 2D or 3D;      
        note: the covariance model must be stationary, however, non stationarity is
        handled: 

        - local rotation by specifying `alpha` (in 2D or 3D), `beta` (in 3D), `gamma` (in 3D)
        - other non-stationarities by specifying `cov_model_non_stationarity_list` (see below)

    dimension : [sequence of] int(s)
        number of cells along each axis, for simulation in:

        - 1D: `dimension=nx`
        - 2D: `dimension=(nx, ny)`
        - 3D: `dimension=(nx, ny, nz)`

        note : this parameter determines the space dimension (1, 2, or 3)

    spacing : [sequence of] float(s), optional
        cell size along each axis, for simulation in:

        - 1D: `spacing=sx`
        - 2D: `spacing=(sx, sy)`
        - 3D: `spacing=(sx, sy, sz)`

        by default (`None`): 1.0 along each axis

    origin : [sequence of] float(s), optional
        origin of the grid ("corner of the first cell"), for simulation in:

        - 1D: `origin=ox`
        - 2D: `origin=(ox, oy)`
        - 3D: `origin=(ox, oy, oz)`

        by default (`None`): 0.0 along each axis
        
    method : str {'simple_kriging', 'ordinary_kriging'}, default: 'ordinary_kriging'
        type of kriging

    x : 1D array-like of floats, optional
        data points locations (float coordinates); note: if one point, a float
        is accepted

    v : 1D array-like of floats, optional
        data values at `x` (`v[i]` is the data value at `x[i]`), array of same
        length as `x` (or float if one point)

    probability : array-like of floats, optional
        probability for each category:

        - sequence of same length as `category_values`: \
        probability[i]: probability (proportion, kriging mean value for the \
        indicator variable) for category `category_values[i]`, used for \
        every grid cell
        - array-like of size `ncategory * ngrid_cells`, where `ncategory` is the \
        length of `category_values` and `ngrid_cells` is the number of grid \
        cells (the array is reshaped if needed): first `ngrid_cells` values are \
        the probabilities (proportions, kriging mean values for the indicator \
        variable) for the first category at grid cells, etc. \
        (for non-stationary probailities / proportions)

        By default (`None`): proportion of each category computed from the
        data values (`v`) are used for every grid cell (uniform probabilities
        if no data point)

        Note: for ordinary kriging (`method='ordinary_kriging'`), it is used for
        case with no neighbor

    alpha : function (callable), or array-like of floats, or float, optional
        if specified: either a single entry or a list of entries of lenght `ncategory`
        (the length of `category_values`, according to the parameter `cov_model` (see 
        above);
        azimuth angle in degrees (see :class:`geone.covModel.CovModel<d>D`):

        - if a function: function of `d` argument(s) where `d` is the space dimension \
        that returns a value for each location given by their coordinate(s) in argument
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), values at grid cells
        - if a float: same value at every grid cell
        - by default (`None`): not used (use of covariance model only)

        note: not used in 1D

    beta : function (callable), or array-like of floats, or float, optional
        if specified: either a single entry or a list of entries of lenght `ncategory`
        (the length of `category_values`, according to the parameter `cov_model` (see 
        above);
        dip angle in degrees (see :class:`geone.covModel.CovModel<d>D`):

        - if a function: function of `d` argument(s) where `d` is the space dimension \
        that returns a value for each location given by their coordinate(s) in argument
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), values at grid cells
        - if a float: same value at every grid cell
        - by default (`None`): not used (use of covariance model only)

        note: not used in 1D or 2D

    gamma : function (callable), or array-like of floats, or float, optional
        if specified: either a single entry or a list of entries of lenght `ncategory`
        (the length of `category_values`, according to the parameter `cov_model` (see 
        above);
        plunge angle in degrees (see :class:`geone.covModel.CovModel<d>D`):

        - if a function: function of `d` argument(s) where `d` is the space dimension \
        that returns a value for each location given by their coordinate(s) in argument
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), values at grid cells
        - if a float: same value at every grid cell
        - by default (`None`): not used (use of covariance model only)

        note: not used in 1D or 2D

    cov_model_non_stationarity_list : list, optional
        if specified: either a single entry or a list of entries of lenght `ncategory`
        (the length of `category_values`, according to the parameter `cov_model` (see 
        above);
        list to set non-stationarities in covariance model; each entry must be
        a tuple (or list) `cm_ns` of length 2 or 3 with:

        - `cm_ns[0]`: str: the name of the method of `cov_model` to be applied
        - `cm_ns[1]`: function (callable), or array-like of floats, or float: \
        used to set the main parameter passed to the method:
            - if a function: function of `d` argument(s) where `d` is the space dimension \
            that returns a value for each location given by their coordinate(s) in argument
            - if array-like: its size must be equal to the number of grid cells \
            (the array is reshaped if needed), values at grid cells
            - if a float: same value at every grid cell
        - `cm_ns[2]`: dict, optional: keyworkds arguments to be passed to the method

        Examples (where the parameter `arg` is set from `val`):
        
        - `('multiply_w', val)` will apply `cov_model.multiply_w(arg)`; \
        this multipies the weight contribution of every elementary contribution of the \
        covariance model
        - `('multiply_w', val, {'elem_ind':0})` will apply `cov_model.multiply_w(arg, elem_ind=0)`; \
        this multipies the weight contribution of the elementary contribution of index 0 of the \
        covariance model
        - `('multiply_r', val)` will apply `cov_model.multiply_r(arg)`; \
        this multipies the range in all direction of every elementary contribution of the \
        covariance model
        - `('multiply_r', val, {'r_ind':0})` will apply `cov_model.multiply_r(arg, r_ind=0)`; \
        this multipies the range in the first main direction (index 0) of every elementary \
        contribution of the covariance model
        - `('multiply_r', val, {'r_ind':0, 'elem_ind':0})` will apply `cov_model.multiply_r(arg, r_ind=0, elem_ind=0)`; \
        this multipies the range in the first main direction (index 0) of the elementary \
        contribution of index 0 of the covariance model

    mask : array-like, optional
        mask value at grid cells (value 1 for simulated cells, value 0 for not
        simulated cells); the size of the array must be equal to the number of
        grid cells (the array is reshaped if needed)

    add_data_point_to_mask : bool, default: True
        - if `True`: any grid cell that contains a data point is added to (the \
        simulated part of) the mask (if present), i.e. mask value at those cells \
        are set to 1; at the end of the computation the "new mask cells" are \
        removed (by setting a missing value (`numpy.nan`) for the variable out of \
        the original mask)
        - if `False`: original mask is kept as given in input, and data point \
        falling out of (the simulated part of) the mask (if present) are ignored

    use_unique_neighborhood : [sequence of] bool(s), default: False
        sequence of bools of same length as `category_values`, or
        a unique bool (recycled); one parameter per category:
        indicates if a unique neighborhood is used (all data points are considered)
        for kriging, in that case (`True`), parameters `searchRadius`, 
        `searchRadiusRelative`, `nneighborMax`, `searchNeighborhoodSortMode` are
        not used

    searchRadius : [sequence of] float(s), optional
        sequence of floats of same length as `category_values`, or
        a unique float (recycled); one parameter per category:
        if specified, i.e. not `None`: radius of the search neighborhood (ellipsoid
        with same radii along each axis), i.e. the data points at distance to the 
        estimated point greater than `searchRadius` are not taken into account 
        in the kriging system; if `searchRadius` is not `None`, then 
        `searchRadiusRelative` is not used;
        by default (`searchRadius=None`): `searchRadiusRelative` is used to 
        define the search ellipsoid;
        note: this parameter is ignored if `use_unique_neighborhood=True`

    searchRadiusRelative : [sequence of] float(s), default: 1.2
        sequence of floats of same length as `category_values`, or
        a unique float (recycled); one parameter per category:
        used only if `searchRadius` is `None`;
        indicates how the search ellipsoid is limited (should be positive): let
        r_i be the ranges of the covariance model along its main axes, when
        estimating/simulating a cell x, a cell y is taken into account iff it is
        within the ellipsoid centered at x of half axes equal to
        `searchRadiusRelative` * r_i; 
        (note that the distances to the central node are computed in the axes 
        sytem supporting the covariance model and accounting for anisotropy given 
        by the ranges)
        note (Geos-Classic library): if a range r_i is non-stationary over the grid, 
        its maximal value over the grid is considered;
        note: this parameter is ignored if `use_unique_neighborhood=True`

    nneighborMax : [sequence of] int(s), default: 12
        sequence of ints of same length as `category_values`, or
        a unique int (recycled); one parameter per category:
        maximal number of neighbors (informed cells) taken into account in the
        kriging system, must be greater than or equal to 1; the informed cells the 
        closest to the estimated cell are taken into account (see the parameters 
        `searchRadius`, `searchRadiusRelative`, `searchNeighborhoodSortMode`);
        note: this parameter is ignored if `use_unique_neighborhood=True`

    searchNeighborhoodSortMode : [sequence of] int(s), default: 1
        sequence of ints of same length as `category_values`, or
        a unique int (recycled); one parameter per category:
        indicates how to sort the search neighboorhood cells (neighbors), for 
        external C function (Geos-Classic library); they are sorted in increasing 
        order according to:

        - `searchNeighborhoodSortMode=0`: distance in the usual axes system
        - `searchNeighborhoodSortMode=1`: distance in the axes sytem supporting \
        the covariance model and accounting for anisotropy given by the ranges \
        (if `searchRadius=None`)
        - `searchNeighborhoodSortMode=2`: minus the evaluation of the covariance \
        model

        Notes:

        - if the covariance model is non-stationary (see note for `cov_model`), then \
        `searchNeighborhoodSortMode=2` is not allowed
        - if the covariance model has any non-stationary range or non-stationary \
        angle and `searchNeighborhoodSortMode=1`: "maximal ranges" (adapted to \
        direction from the central cell) are used to compute distance for sorting \
        the neighbors

    outputReportFile : str, optional
        name of the report file (if desired in output); by default (`None`): no
        report file

    nthreads : int, default: -1
        number of thread(s) for external C function (Geos-Classic library);
        if `nthreads = -n <= 0`: `nthreads` is automatically set to `nmax - n` 
        (but at least 1), where nmax is the total number of cpu(s) of the system 
        (retrieved by `os.cpu_count()`); 
        not: if `nthreads=None`, `nthreads=-1` is used

    verbose : int, default: 2
        verbose mode, higher implies more printing (info):

        - 0: no display
        - 1: warnings
        - 2: warnings + basic info
        - 3 (or >2): all information

        note that if an error occurred, it is raised
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    geosclassic_output : dict
        geosclassic output in python, dictionary

        {'image':image, 'nwarning':nwarning, 'warnings':warnings}

        with:

        - image : :class:`geone.img.Img`
            output image, with `image.nv=ncategory` variables (probability /
            proportion estimates, of each category);
            note: `image=None` if `mpds_geosClassicOutput->outputImage=NULL`

        - nwarning : int
            total number of warning(s) encountered (same warnings can be counted
            several times)

        - warnings : list of strs
            list of distinct warnings encountered (can be empty)
    """
    fname = 'estimateIndicator'

    # Set space dimension: d
    if hasattr(dimension, '__len__'):
        d = len(dimension)
    else:
        # assume dimension is an int, nx
        d = 1

    if d not in (1, 2, 3):
        err_msg = f'{fname}: space dimension not valid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check (or set) argument 'spacing'
    if spacing is None:
        spacing = tuple(np.ones(d))
    else:
        if hasattr(spacing, '__len__') and len(spacing) != d:
            err_msg = f'{fname}: `spacing` of incompatible length'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Check (or set) argument 'origin'
    if origin is None:
        origin = tuple(np.zeros(d))
    else:
        if hasattr(origin, '__len__') and len(origin) != d:
            err_msg = f'{fname}: `origin` of incompatible length'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Set grid geometry
    if d == 1:
        nx, ny, nz = dimension, 1, 1
        sx, sy, sz = spacing, 1.0, 1.0
        ox, oy, oz = origin, 0.0, 0.0
    elif d == 2:
        nx, ny, nz = *dimension, 1
        sx, sy, sz = *spacing, 1.0 
        ox, oy, oz = *origin, 0.0 
    else: # d == 3
        nx, ny, nz = dimension
        sx, sy, sz = spacing
        ox, oy, oz = origin

    nxy = nx * ny
    nxyz = nxy * nz

    # Check category values and compute number of category (ncategory
    try:
        category_values = np.asarray(category_values, dtype='float').reshape(-1)
    except:
        err_msg = f'{fname}: `category_values` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    ncategory = len(category_values)
    if ncategory <= 0:
        err_msg = f'{fname}: `category_values` is empty'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check dimension of cov_model for each category and set if used as omni-directional model
    cov_model = list(np.asarray(cov_model).reshape(-1))
    if len(cov_model) == 1:
        use_same_cov_model_for_all_categories = True
        cov_model = cov_model[0]
    elif len(cov_model) != ncategory:
        err_msg = f'{fname}: `cov_model_for_category` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)
    else:
        use_same_cov_model_for_all_categories = False

    if use_same_cov_model_for_all_categories:
        # Same covariance model is used for all categories
        # ------------------------------------------------
        # -> "single entries" for the parameters below ...
        # Check dimension of cov_model and set if used as omni-directional model
        if isinstance(cov_model, gcm.CovModel1D):
            omni_dir = True
        else:
            if cov_model.__class__.__name__ != f'CovModel{d}D':
                err_msg = f'{fname}: `cov_model` dimension is incompatible with `dimension`'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            omni_dir = False

        # Prevent calculation if covariance model is not stationary
        if not cov_model.is_stationary():
            err_msg = f'{fname}: `cov_model` is not stationary: {fname} cannot be applied (use the other paramters for non-stationary covariance)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # Check parameters for non-stationarity
        # ... rotation given by `alpha`, `beta`, `gamma`
        if alpha is not None:
            if omni_dir:
                err_msg = f'{fname}: `alpha` cannot be used with 1D covariance model'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
            if not callable(alpha) and not isinstance(alpha, np.ndarray) and not isinstance(alpha, float) and not isinstance(alpha, int):
                err_msg = f'{fname}: `alpha` not valid: should be callable or array or float or int'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

        if beta is not None:
            if omni_dir or d < 3:
                err_msg = f'{fname}: `beta` cannot be used with 1D or 2D covariance model'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
            if not callable(beta) and not isinstance(beta, np.ndarray) and not isinstance(beta, float) and not isinstance(beta, int):
                err_msg = f'{fname}: `beta` not valid: should be callable or array or float or int'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
        
        if gamma is not None:
            if omni_dir or d < 3:
                err_msg = f'{fname}: `gamma` cannot be used with 1D or 2D covariance model'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
            if not callable(gamma) and not isinstance(gamma, np.ndarray) and not isinstance(gamma, float) and not isinstance(gamma, int):
                err_msg = f'{fname}: `gamma` not valid: should be callable or array or float or int'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

        # ... other non-stationarities given by cov_model_non_stationarity_list
        if cov_model_non_stationarity_list is not None:
            if not isinstance(cov_model_non_stationarity_list, list):
                err_msg = f'{fname}: `cov_model_non_stationarity_list` must be a list if not `None`'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
            for i, cm_ns in enumerate(cov_model_non_stationarity_list):
                if not hasattr(cm_ns, '__len__') or len(cm_ns) not in (2, 3):
                    err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list` not valid: should be a tuple or list of length 2 or 3 containing: method name (str), value(s) or callable (for arg)[, kwds (dict)]'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)
                if not isinstance(cm_ns[0], str):
                    err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list`, `(method_name, val[, kwds])`: `method_name` not valid: should be a string'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)
                if not callable(cm_ns[1]) and not isinstance(cm_ns[1], np.ndarray) and not isinstance(cm_ns[1], float) and not isinstance(cm_ns[1], int):
                    err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list`, `(method_name, val[, kwds])`: `val` not valid: should be callable or array or float or int'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)
                if len(cm_ns) == 3 and not isinstance(cm_ns[2], dict):
                    err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list`, `(method_name, val[, kwds])`: `kwds` not valid: should be a dictionary'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

    else:
        # One covariance model per categories (ncategory at all)
        # ------------------------------------------------------
        # -> "multiple entries" for the parameters below ...
        # Check dimension of cov_model and set if used as omni-directional model
        omni_dir = np.zeros(ncategory, dtype='bool')
        for k, cov_model_k in enumerate(cov_model):
            if isinstance(cov_model_k, gcm.CovModel1D):
                omni_dir[k] = True
            elif cov_model_k.__class__.__name__ != f'CovModel{d}D':
                err_msg = f'{fname}: `cov_model` (index {k}) dimension is incompatible with `dimension`'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            # Prevent calculation if covariance model is not stationary
            if not cov_model_k.is_stationary():
                err_msg = f'{fname}: `cov_model` (index {k}) is not stationary: {fname} cannot be applied (use the other paramters for non-stationary covariance)'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

        # Check parameters for non-stationarity
        # ... rotation given by `alpha`, `beta`, `gamma`
        # alpha
        if alpha is None:
            alpha = ncategory * [None]
        
        elif not isinstance(alpha, list) or len(alpha) != ncategory:
            err_msg = f'{fname}: `alpha` should be a list of length equal to the number of categories (when covariance model is set per category)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        else:
            for k, alpha_k in enumerate(alpha):
                if omni_dir:
                    err_msg = f'{fname}: `alpha` (index {k}) cannot be used with 1D covariance model'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)
                if not callable(alpha_k) and not isinstance(alpha_k, np.ndarray) and not isinstance(alpha_k, float) and not isinstance(alpha_k, int):
                    err_msg = f'{fname}: `alpha` (index {k}) not valid: should be callable or array or float or int'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

        # beta
        if beta is None:
            beta = ncategory * [None]
        
        elif not isinstance(beta, list) or len(beta) != ncategory:
            err_msg = f'{fname}: `beta` should be a list of length equal to the number of categories (when covariance model is set per category)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        else:
            if not isinstance(beta, list) or len(beta) != ncategory:
                err_msg = f'{fname}: `beta` should be a list of length equal to the number of categories (when covariance model is set per category)'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
            
            for k, beta_k in enumerate(beta):
                if omni_dir or d < 3:
                    err_msg = f'{fname}: `beta` (index {k}) cannot be used with 1D or 2D covariance model'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)
                if not callable(beta_k) and not isinstance(beta_k, np.ndarray) and not isinstance(beta_k, float) and not isinstance(beta_k, int):
                    err_msg = f'{fname}: `beta` (index {k}) not valid: should be callable or array or float or int'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

        # gamma
        if gamma is None:
            gamma = ncategory * [None]
        
        elif not isinstance(gamma, list) or len(gamma) != ncategory:
            err_msg = f'{fname}: `gamma` should be a list of length equal to the number of categories (when covariance model is set per category)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        else:
            if not isinstance(gamma, list) or len(gamma) != ncategory:
                err_msg = f'{fname}: `gamma` should be a list of length equal to the number of categories (when covariance model is set per category)'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
            for k, gamma_k in enumerate(gamma):
                if omni_dir or d < 3:
                    err_msg = f'{fname}: `gamma` (index {k}) cannot be used with 1D or 2D covariance model'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)
                if not callable(gamma_k) and not isinstance(gamma_k, np.ndarray) and not isinstance(gamma_k, float) and not isinstance(gamma_k, int):
                    err_msg = f'{fname}: `gamma` (index {k}) not valid: should be callable or array or float or int'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

        # ... other non-stationarities given by cov_model_non_stationarity_list
        if cov_model_non_stationarity_list is None:
            cov_model_non_stationarity_list = ncategory * [None]
        
        elif not isinstance(cov_model_non_stationarity_list, list) or len(cov_model_non_stationarity_list) != ncategory:
            err_msg = f'{fname}: `cov_model_non_stationarity_list` should be a list (of list) of length equal to the number of categories (when covariance model is set per category)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        else:
            for k, cov_model_non_stationarity_list_k in enumerate(cov_model_non_stationarity_list):
                if cov_model_non_stationarity_list_k is not None:
                    if not isinstance(cov_model_non_stationarity_list_k, list):
                        err_msg = f'{fname}: `cov_model_non_stationarity_list` (index {k}) must be a list if not `None`'
                        if logger: logger.error(err_msg)
                        raise GeosclassicinterfaceError(err_msg)
                    
                    for i, cm_ns in enumerate(cov_model_non_stationarity_list_k):
                        if not hasattr(cm_ns, '__len__') or len(cm_ns) not in (2, 3):
                            err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list` (index {k}) not valid: should be a tuple or list of length 2 or 3 containing: method name (str), value(s) or callable (for arg)[, kwds (dict)]'
                            if logger: logger.error(err_msg)
                            raise GeosclassicinterfaceError(err_msg)
                        if not isinstance(cm_ns[0], str):
                            err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list` (index {k}), `(method_name, val[, kwds])`: `method_name` not valid: should be a string'
                            if logger: logger.error(err_msg)
                            raise GeosclassicinterfaceError(err_msg)
                        if not callable(cm_ns[1]) and not isinstance(cm_ns[1], np.ndarray) and not isinstance(cm_ns[1], float) and not isinstance(cm_ns[1], int):
                            err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list` (index {k}), `(method_name, val[, kwds])`: `val` not valid: should be callable or array or float or int'
                            if logger: logger.error(err_msg)
                            raise GeosclassicinterfaceError(err_msg)
                        if len(cm_ns) == 3 and not isinstance(cm_ns[2], dict):
                            err_msg = f'{fname}: entry {i} of `cov_model_non_stationarity_list` (index {k}), `(method_name, val[, kwds])`: `kwds` not valid: should be a dictionary'
                            if logger: logger.error(err_msg)
                            raise GeosclassicinterfaceError(err_msg)

    # Set parameters for neighborhood
    # - searchRadius, searchRadius_geosClassic
    # - searchRadiusRelative, searchRadiusRelative_geosClassic
    # - nneighborMax, nneighborMax_geosClassic
    # - searchNeighborhoodSortMode
    if isinstance(use_unique_neighborhood, bool):
        use_unique_neighborhood = ncategory * [use_unique_neighborhood]
    elif not isinstance(use_unique_neighborhood, list) or len(use_unique_neighborhood) != ncategory:
        err_msg = f'{fname}: `use_unique_neighborhood` should be a single entry or a list of length equal to the number of categories'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)
    
    if searchRadius is None:
        searchRadius = ncategory * [None]
    elif isinstance(searchRadius, float) or isinstance(searchRadius, int):
        searchRadius = ncategory * [searchRadius]
    elif not isinstance(searchRadius, list) or len(searchRadius) != ncategory:
        err_msg = f'{fname}: `searchRadius` should be `None`, a single entry or a list of length equal to the number of categories'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)
    
    if searchRadiusRelative is None:
        searchRadiusRelative = ncategory * [None]
    elif isinstance(searchRadiusRelative, float) or isinstance(searchRadiusRelative, int):
        searchRadiusRelative = ncategory * [searchRadiusRelative]
    elif not isinstance(searchRadiusRelative, list) or len(searchRadiusRelative) != ncategory:
        err_msg = f'{fname}: `searchRadiusRelative` should be a list of length equal to the number of categories'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)
    
    if isinstance(nneighborMax, int):
        nneighborMax = ncategory * [nneighborMax]          
    elif not isinstance(nneighborMax, list) or len(nneighborMax) != ncategory:
        err_msg = f'{fname}: `nneighborMax` should be a single entry or a list of length equal to the number of categories'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    if searchNeighborhoodSortMode is None:
        searchNeighborhoodSortMode = ncategory * [1]
    elif isinstance(searchNeighborhoodSortMode, int):
        searchNeighborhoodSortMode = ncategory * [searchNeighborhoodSortMode]
    elif not isinstance(searchNeighborhoodSortMode, list) or len(searchNeighborhoodSortMode) != ncategory:
        err_msg = f'{fname}: `searchNeighborhoodSortMode` should be a single entry or a list of length equal to the number of categories'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    searchRadius_geosClassic = ncategory * [0.0]
    searchRadiusRelative_geosClassic = ncategory * [0.0]
    nneighborMax_geosClassic = ncategory * [0]
    for k in range(ncategory):
        if use_unique_neighborhood[k]:
            searchRadius_geosClassic[k] = 0.0          # to indicate in external C function that search radius is not used
            searchRadiusRelative_geosClassic[k] = -1.0 # to indicate for external C function that unique search neighborhood is used
            nneighborMax_geosClassic[k] = 1            # not used
            searchNeighborhoodSortMode[k] = 1          # not used
        else:
            if searchRadius[k] is not None:
                # searchRadius[k] is used, searchRadiusRelative[k] is not used
                if searchRadius[k] <= 0.0:
                    err_msg = f'{fname}: search radius (index {k}) (isotropic neighborhood) not valid (negative)'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)
                
                searchRadius_geosClassic[k] = searchRadius[k]
                searchRadiusRelative_geosClassic[k] = searchRadiusRelative[k] # unused

            else:
                # searchRadius[k] is not used, searchRadiusRelative is used
                searchRadius_geosClassic[k] = 0.0 # to indicate in external C function that search radius is not used
                searchRadiusRelative_geosClassic[k] = searchRadiusRelative[k]

                # searchRadiusRelative
                if searchRadiusRelative_geosClassic[k] < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
                    err_msg = f'{fname}: `searchRadiusRelative` (index {k}) too small (should be at least {geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN})'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

            if nneighborMax[k] <= 0:
                # (note : nneighborMax = -1 is authorized in Geos-Classic)
                err_msg = f'{fname}: `nneighborMax` (index {k}) should be greater than or equal to 1'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
            
            nneighborMax_geosClassic[k] = nneighborMax[k]

            if searchNeighborhoodSortMode[k] is None:
                searchNeighborhoodSortMode[k] = 1
                # # Set greatest possible value
                # searchNeighborhoodSortMode[k] = 2
                # if use_same_cov_model_for_all_categories:
                #     if         (alpha is not None and (callable(alpha) or (isinstance(alpha, np.ndarray) and alpha.size > 1))) \
                #             or (beta  is not None and (callable(beta ) or (isinstance(beta,  np.ndarray) and beta.size  > 1))) \
                #             or (gamma is not None and (callable(gamma) or (isinstance(gamma, np.ndarray) and gamma.size > 1))):
                #         searchNeighborhoodSortMode[k] = 1
                # else:
                #     if         (alpha[k] is not None and (callable(alpha[k]) or (isinstance(alpha[k], np.ndarray) and alpha[k].size > 1))) \
                #             or (beta[k]  is not None and (callable(beta[k] ) or (isinstance(beta[k],  np.ndarray) and beta[k].size  > 1))) \
                #             or (gamma[k] is not None and (callable(gamma[k]) or (isinstance(gamma[k], np.ndarray) and gamma[k].size > 1))):
                #         searchNeighborhoodSortMode[k] = 1
                # if cov_model_non_stationarity_list[k] is not None:
                #     for cm_ns in cov_model_non_stationarity_list[k]:
                #         if callable(cm_ns[1]) or (isinstance(cm_ns[1],  np.ndarray) and cm_ns[1].size  > 1):
                #             searchNeighborhoodSortMode[k] = 1
                #             break
            elif searchNeighborhoodSortMode[k] == 2:
                if use_same_cov_model_for_all_categories:
                    if         (alpha is not None and (callable(alpha) or (isinstance(alpha, np.ndarray) and alpha.size > 1))) \
                            or (beta  is not None and (callable(beta ) or (isinstance(beta,  np.ndarray) and beta.size  > 1))) \
                            or (gamma is not None and (callable(gamma) or (isinstance(gamma, np.ndarray) and gamma.size > 1))):
                        err_msg = f'{fname}: `searchNeighborhoodSortMode=2` (index {k}) not allowed with non-stationary covariance model (local rotation)'
                        if logger: logger.error(err_msg)
                        raise GeosclassicinterfaceError(err_msg)
                else:
                    if         (alpha[k] is not None and (callable(alpha[k]) or (isinstance(alpha[k], np.ndarray) and alpha[k].size > 1))) \
                            or (beta[k]  is not None and (callable(beta[k] ) or (isinstance(beta[k],  np.ndarray) and beta[k].size  > 1))) \
                            or (gamma[k] is not None and (callable(gamma[k]) or (isinstance(gamma[k], np.ndarray) and gamma[k].size > 1))):
                        err_msg = f'{fname}: `searchNeighborhoodSortMode=2` (index {k}) not allowed with non-stationary covariance model (local rotation)'
                        if logger: logger.error(err_msg)
                        raise GeosclassicinterfaceError(err_msg)
                
                if cov_model_non_stationarity_list[k] is not None:
                    for cm_ns in cov_model_non_stationarity_list[k]:
                        if callable(cm_ns[1]) or (isinstance(cm_ns[1],  np.ndarray) and cm_ns[1].size  > 1):
                            err_msg = f'{fname}: `searchNeighborhoodSortMode=2` (index {k}) not allowed with non-stationary covariance model ({cm_ns[0]})'
                            if logger: logger.error(err_msg)
                            raise GeosclassicinterfaceError(err_msg)
                        
            elif searchNeighborhoodSortMode[k] not in (0, 1):
                err_msg = f'{fname}: `searchNeighborhoodSortMode` (index {k}) not valid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # Set varname
    varname = 'V0'

    # Set probability
    if probability is not None:
        # if method == 'ordinary_kriging':
        #     err_msg = f"{fname}: specifying 'probability' not allowed with ordinary kriging"
        #     if logger: logger.error(err_msg)
        #     raise GeosclassicinterfaceError(err_msg)
        probability = np.asarray(probability, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if probability.size not in (ncategory, ncategory*nxyz):
            err_msg = f'{fname}: size of `probability` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Set method : computationMode
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    if method == 'simple_kriging':
        computationMode = 1
    elif method == 'ordinary_kriging':
        computationMode = 0
    else:
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Set dataPointSet from data points: x, v
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, d) # cast in 2-dimensional array if needed

        # Check if all data points are in the grid
        if verbose > 0:
            if d == 1:
                if np.any(x[:, 0] < ox) or np.any(x[:, 0] > ox + nx*sx):
                    if logger:
                        logger.warning(f'{fname}: data point(s) out of the grid')
                    else:
                        print(f'{fname}: WARNING: data point(s) out of the grid')
            elif d == 2:
                if    np.any(x[:, 0] < ox) or np.any(x[:, 0] > ox + nx*sx) \
                   or np.any(x[:, 1] < oy) or np.any(x[:, 1] > oy + ny*sy):
                    if logger:
                        logger.warning(f'{fname}: data point(s) out of the grid')
                    else:
                        print(f'{fname}: WARNING: data point(s) out of the grid')
            else: # d == 3
                if    np.any(x[:, 0] < ox) or np.any(x[:, 0] > ox + nx*sx) \
                   or np.any(x[:, 1] < oy) or np.any(x[:, 1] > oy + ny*sy) \
                   or np.any(x[:, 2] < oz) or np.any(x[:, 2] > oz + nz*sz):
                    if logger:
                        logger.warning(f'{fname}: data point(s) out of the grid')
                    else:
                        print(f'{fname}: WARNING: data point(s) out of the grid')

        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        if ncategory > 1:
            if not np.all([vi in category_values for vi in v]):
                err_msg = f'{fname}: `v` contains an invalid category value (not in `category_values`)'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)
        else:
            if not np.all([vi in np.hstack((category_values, np.array([0.0]))) for vi in v]):
                err_msg = f'{fname}: `v` contains an invalid category value (not in `category_values` nor 0.0)'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

        # Aggregate data on grid by taking the most frequent value in grid cell
        xx, yy, zz = np.hstack((x, np.zeros((len(x), 3-d)))).T
        try:
            xx_agg, yy_agg, zz_agg, v_agg = img.aggregateDataPointsWrtGrid(
                                                xx, yy, zz, v,
                                                nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                op='most_freq',
                                                logger=logger)
        except Exception as exc:
            err_msg = f"{fname}: data aggregation ('most_freq') failed"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

        if len(xx_agg) == 0:
            err_msg = f'{fname}: no data point in grid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        dataPointSet = PointSet(npt=v_agg.shape[0], nv=4, val=np.vstack((xx_agg, yy_agg, zz_agg, v_agg)), varname=['X', 'Y', 'Z', varname])
    else:
        dataPointSet = None

    # Set mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            err_msg = f'{fname}: `mask` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if mask is not None and add_data_point_to_mask and len(dataPointSet):
        # Make a copy of the original mask, to remove value in added mask cell at the end
        mask_original = np.copy(mask)
        # Add cell to mask if needed
        if dataPointSet[0].npt:
            im_tmp = img.imageFromPoints(dataPointSet[0].val[:3, :].T,
                        nx=nx, ny=ny, nz=nz,
                        sx=sx, sy=sy, sz=sz,
                        ox=ox, oy=oy, oz=oz,
                        indicator_var=True,
                        logger=logger)
            mask = 1.0*np.any((im_tmp.val[0], mask), axis=0)
            del(im_tmp)
        recover_mask = True
    
    else:
        recover_mask = False

    if use_same_cov_model_for_all_categories:
        # Same covariance model is used for all categories
        # ------------------------------------------------
        # Set non-stationary covariance model (if needed)
        cov_model_grid = gcm.copyCovModel(cov_model)
        # ... from alpha
        if alpha is not None:
            if d == 1:
                vv = gcm.eval_in_grid_1D(alpha, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                vv = gcm.eval_in_grid_2D(alpha, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                vv = gcm.eval_in_grid_3D(alpha, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
            cov_model_grid.set_alpha(vv)

        # ... from beta
        if beta is not None:
            if d == 1:
                vv = gcm.eval_in_grid_1D(beta, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                vv = gcm.eval_in_grid_2D(beta, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                vv = gcm.eval_in_grid_3D(beta, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
            cov_model_grid.set_beta(vv)

        # ... from gamma
        if gamma is not None:
            if d == 1:
                vv = gcm.eval_in_grid_1D(gamma, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
            elif d == 2:
                vv = gcm.eval_in_grid_2D(gamma, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
            else: # d == 3
                vv = gcm.eval_in_grid_3D(gamma, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
            cov_model_grid.set_gamma(vv)

        # ... from cov_model_non_stationarity_list
        if cov_model_non_stationarity_list is not None:
            for cm_ns in cov_model_non_stationarity_list:
                try:
                    method_name = cm_ns[0]
                    val = cm_ns[1]
                    if len(cm_ns) == 3:
                        kwds = cm_ns[2]
                    else:
                        kwds = {}
                    if d == 1:
                        arg = gcm.eval_in_grid_1D(val, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                    elif d == 2:
                        arg = gcm.eval_in_grid_2D(val, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                    else: # d == 3
                        arg = gcm.eval_in_grid_3D(val, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                    eval(f'cov_model_grid.{method_name}')(arg, **kwds)
                except:
                    err_msg = f'{fname}: cannot apply non-stationarity (in grid) for covariance model (check parameter `cov_model_non_stationarity_list`)'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)
    else:
        # One covariance model per categories (ncategory at all)
        # ------------------------------------------------------
        # Set non-stationary covariance model (if needed)
        cov_model_grid = [gcm.copyCovModel(cov_model_k) for cov_model_k in cov_model]
        for k in range(ncategory):
            # ... from alpha
            if alpha[k] is not None:
                if d == 1:
                    vv = gcm.eval_in_grid_1D(alpha[k], nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                elif d == 2:
                    vv = gcm.eval_in_grid_2D(alpha[k], nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                else: # d == 3
                    vv = gcm.eval_in_grid_3D(alpha[k], nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                cov_model_grid[k].set_alpha(vv)

            # ... from beta
            if beta[k] is not None:
                if d == 1:
                    vv = gcm.eval_in_grid_1D(beta[k], nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                elif d == 2:
                    vv = gcm.eval_in_grid_2D(beta[k], nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                else: # d == 3
                    vv = gcm.eval_in_grid_3D(beta[k], nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                cov_model_grid[k].set_beta(vv)

            # ... from gamma
            if gamma[k] is not None:
                if d == 1:
                    vv = gcm.eval_in_grid_1D(gamma[k], nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                elif d == 2:
                    vv = gcm.eval_in_grid_2D(gamma[k], nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                else: # d == 3
                    vv = gcm.eval_in_grid_3D(gamma[k], nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                cov_model_grid[k].set_gamma(vv)

            # ... from cov_model_non_stationarity_list
            if cov_model_non_stationarity_list[k] is not None:
                for cm_ns in cov_model_non_stationarity_list[k]:
                    try:
                        method_name = cm_ns[0]
                        val = cm_ns[1]
                        if len(cm_ns) == 3:
                            kwds = cm_ns[2]
                        else:
                            kwds = {}
                        if d == 1:
                            arg = gcm.eval_in_grid_1D(val, nx,         sx,         ox,         return_float_if_unique=True, logger=logger)
                        elif d == 2:
                            arg = gcm.eval_in_grid_2D(val, nx, ny,     sx, sy,     ox, oy,     return_float_if_unique=True, logger=logger)
                        else: # d == 3
                            arg = gcm.eval_in_grid_3D(val, nx, ny, nz, sx, sy, sz, ox, oy, oz, return_float_if_unique=True, logger=logger)
                        eval(f'cov_model_grid[k].{method_name}')(arg, **kwds)
                    except:
                        err_msg = f'{fname}: cannot apply non-stationarity (in grid) for covariance model (index {k}) (check parameter `cov_model_non_stationarity_list`)'
                        if logger: logger.error(err_msg)
                        raise GeosclassicinterfaceError(err_msg)

    # Set number of threads per process: nth - GeosClassic-parameter
    if nthreads is None:
        nthreads = -1
    
    if nthreads <= 0:
        nth = max(int(os.cpu_count() + nthreads), 1)
    else:
        nth = int(nthreads)
        # if verbose > 1 and nth != nthreads:
        #     if logger:
        #         logger.info(f'{fname}: number of threads has been changed (now: nthreads={nth})')
        #     else:
        #         print(f'{fname}: number of threads has been changed (now: nthreads={nth})')

    if verbose > 0 and nthreads > os.cpu_count():
        if logger:
            logger.warning(f'{fname}: total number of cpu(s) used ({nthreads}) will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: total number of cpu(s) used ({nthreads}) will exceed number of cpu(s) of the system...')

    if verbose > 1:
        if logger:
            logger.info(  
                f'{fname}: Geos-Classic running on 1 process... [' + \
                f'VERSION {geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER:s} / ' + \
                f'BUILD NUMBER {geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER:s} / ' + \
                f'OpenMP {nth:d} thread(s)]')
        else:
            print(  
                f'{fname}: Geos-Classic running on 1 process... [' + \
                f'VERSION {geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER:s} / ' + \
                f'BUILD NUMBER {geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER:s} / ' + \
                f'OpenMP {nth:d} thread(s)]')

    if verbose < 3:
        progress_monitor = 0
    else:
        progress_monitor = 4

    args = (
            use_same_cov_model_for_all_categories,
            d,
            ncategory,
            category_values,
            cov_model_grid,
            nx, ny, nz,
            sx, sy, sz,
            ox, oy, oz,
            varname,
            outputReportFile,
            computationMode,
            None,
            dataPointSet,
            mask,
            probability,
            searchRadius_geosClassic,
            searchRadiusRelative_geosClassic,
            nneighborMax_geosClassic,
            searchNeighborhoodSortMode,
            0,
            0,
            nth
        )
    kwds = {
            'progress_monitor':progress_monitor, 
            'pid':None,
            'verbose':verbose,
            'logger':logger
        }
    geosclassic_output = run_MPDSOMPGeosClassicIndicatorSim(*args, **kwds)

    if verbose > 1 and geosclassic_output:
        if logger:
            logger.info(f'{fname}: Geos-Classic (1 process) run complete')
        else:
            print(f'{fname}: Geos-Classic (1 process) run complete')

    if geosclassic_output is not None and recover_mask:
        # Remove the value out of the original mask (using its copy see above)
        geosclassic_output['image'].val[:, mask_original==0.0] = np.nan

    # Show (print) encountered warnings
    if verbose > 0 and geosclassic_output and geosclassic_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({geosclassic_output['nwarning']} times in all):")
            for i, warning_message in enumerate(geosclassic_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imgDistanceImage(
        input_image,
        distance_type='L2',
        distance_negative=False,
        nthreads=-1,
        verbose=0,
        logger=None):
    """
    Computes distance to a given subset in an image.

    This function computes the image of the distances to the set of non zero
    values in the input image. The distances are computed for each variable v
    over the image grid: distance to the set S = {v!=0}. Distance is equal to
    zero for all cells in S if the keyword argument `distance_negative=False`
    (default). If `distance_negative=True`, the distance to the border of S is
    computed for the cells in the interior of S (i.e. in S but not on the
    border), and the opposite (negative) value is retrieved for that cells.
    The output image has the same number of variable(s) and the same size
    (grid geometry) as the input image.

    Parameters
    ----------
    input_image : :class:`geone.img.Img`
        input image

    distance_type : str {'L1', 'L2'}, default: 'L2'
        type of distance

    distance_negative : bool, default: True
        - if `True`: negative distance are retrieved for the grid cells in S={v!=0} \
        (where v denotes the value of a variable in the input image) (distance \
        set to zero on the border of S), see above
        - if `False`: distance is set to zero for all grid cells in S={v!=0} \
        (where v denotes the value of a variable in the input image), see above

    nthreads : int, default: -1
        number of thread(s) to use for C program;
        `nthreads = -n <= 0`: maximal number of threads of the system except n
        (but at least 1)

    verbose : int, default: 0
        verbose mode, higher implies more printing (info)
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    output_image : :class:`geone.img.Img`
        image with same grid as the input image and same number of variable,
        distance to the set S={v!=0} (where v denotes the value of a variable
        in the input image), for each variable

    References
    ----------
    - A. Meijster, J. B. T. M. Roerdink, W. H. Hesselink (2000), \
    A General Algorithm for Computing Distance Transforms in Linear Time, \
    in book: "Mathematical Morphology and its Applications to Image and Signal Processing", \
    Springer US, pp. 331-340, \
    `doi:10.1007/0-306-47025-X_36 <https://doi.org/10.1007/0-306-47025-X_36>`_
    """
    fname = 'imgDistanceImage'

    # Check
    if distance_type not in ('L1', 'L2'):
        err_msg = f'{fname}: unknown `distance_type`'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Set input image "in C"
    try:
        input_image_c = img_py2C(input_image, logger=logger)
    except Exception as exc:
        err_msg = f'{fname}: cannot convert input image from python to C'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg) from exc

    # Allocate output image "in C"
    output_image_c = geosclassic.malloc_MPDS_IMAGE()
    geosclassic.MPDSInitImage(output_image_c)

    # Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose > 0 and nth > os.cpu_count():
        if logger:
            logger.warning(f'{fname}: number of threads used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: number of threads used will exceed number of cpu(s) of the system...')

    # Compute distances (launch C code)
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
        # Free memory on C side: input_image_c
        geosclassic.MPDSFreeImage(input_image_c)
        geosclassic.free_MPDS_IMAGE(input_image_c)
        # Free memory on C side: output_image_c
        geosclassic.MPDSFreeImage(output_image_c)
        geosclassic.free_MPDS_IMAGE(output_image_c)
        # Raise error
        err_msg = f'{fname}: `distance_type` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Retrieve output image "in python"
    if err:
        # Free memory on C side: input_image_c
        geosclassic.MPDSFreeImage(input_image_c)
        geosclassic.free_MPDS_IMAGE(input_image_c)
        # Free memory on C side: output_image_c
        geosclassic.MPDSFreeImage(output_image_c)
        geosclassic.free_MPDS_IMAGE(output_image_c)
        # Raise error
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        err_msg = f'{fname}: {err_message}'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    output_image = img_C2py(output_image_c, logger=logger)

    # Free memory on C side: input_image_c
    geosclassic.MPDSFreeImage(input_image_c)
    geosclassic.free_MPDS_IMAGE(input_image_c)

    # Free memory on C side: output_image_c
    geosclassic.MPDSFreeImage(output_image_c)
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
        connect_type='connect_face',
        logger=None):
    """
    Computes the geobody image (map) for one variable of the input image.

    For the considered variable v and any location x in the input image, an
    indicator I is defined as

    * I(x) = 1 if v(x) is between `bound_inf` and `bound_sup`
    * I(x) = 0 otherwise

    Then lower (resp. upper) bound `bound_inf` (resp. `bound_sup`) is exluded
    from the set I=1 if `bound_inf_excluded=True` (resp.
    `bound_sup_excluded=True`) is True or included if `bound_inf_excluded=False`
    (resp. `bound_sup_excluded=False`); hence:

    .. list-table::
        :widths: 25 45
        :header-rows: 1

        *   - `bound_inf_excluded, bound_sup_excluded`
            - indicator variable :math:`I(x)`
        *   - `True, True (default)`
            - :math:`I(x) = 1 \\iff bound\\_inf < v(x) < bound\\_sup`
        *   - `True, False`
            - :math:`I(x) = 1 \\iff bound\\_inf < v(x) \\leqslant bound\\_sup`
        *   - `False, True`
            - :math:`I(x) = 1 \\iff bound\\_inf \\leqslant v(x) < bound\\_sup`
        *   - `False, False`
            - :math:`I(x) = 1 \\iff bound\\_inf \\leqslant v(x) \\leqslant bound\\_sup`

    If `complementary_set=True`, the variable IC(x) = 1 - I(x) is used
    instead of variable I, i.e. the set I=0 and I=1 are swapped.

    The geobody image (map) is computed for the indicator variable I, which
    consists in labelling the connected components from 1 to n, i.e.

    * C(x) = 0     if I(x) = 0
    * C(x) = k > 0 if I(x) = 1 and x is in the k-th connected component

    Two cells x and y in the grid are said connected, :math:`x \\leftrightarrow y`,
    if there exists a path between x and y going composed of adjacent cells, within
    the set I=1. Following this definition, we have

    * :math:`x \\leftrightarrow y \\iff C(x) = C(y) > 0`

    The definition of adjacent cells is set according to the parameter
    `connect_type`:

    .. list-table::
        :widths: 25 45
        :header-rows: 1

        *   - `connect_type`
            - two grid cells are adjacent if they have
        *   - 'connect_face' (default)
            - a common face
        *   - 'connect_face_edge'
            - a common face or a common edge
        *   - 'connect_face_edge_corner'
            - a common face or a common edge or a common corner

    Parameters
    ----------
    input_image : :class:`geone.img.Img`
        input image

    var_index : int, default: 0
        index of the considered variable in input image

    bound_inf : float, default: 0.0
        lower bound of the interval defining the indicator variable

    bound_sup : float, optional
        upper bound of the interval defining the indicator variable;
        by default (`None`): `bound_sup=numpy.inf` is used (no upper bound)

    bound_inf_excluded : bool, default: True
        lower bound is excluded from the interval defining the indicator
        variable (`True`) or included (`False`)

    bound_sup_excluded : bool, default: True
        upper bound is excluded from the interval defining the indicator
        variable (`True`) or included (`False`)

    complementary_set : bool, default: False
        - if `True`: the complementary indicator variable (IC = 1-I, see above) \
        is used
        - if `False`: the indicator variable I (see above) is used

    connect_type : str {'connect_face', 'connect_face_edge', \
            'connect_face_edge_corner'}, default: 'connect_face'
        indicates which definition of adjacent cells is used (see above)
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    output_image : :class:`geone.img.Img`
        image with same grid as the input image and one variable, the geobody
        label (see above)

    References
    ----------
    - Hoshen and Kopelman (1976), \
    Percolation and cluster distribution. I. Cluster multiple labeling technique and critical concentration algorithm. \
    Physical Review B, 14(8):3438-3445, \
    `doi:10.1103/PhysRevB.14.3438 <https://doi.org/10.1103/PhysRevB.14.3438>`_
    """
    # Two cells x and y in the grid are said connected, x <-> y, if there exists
    # a path between x and y going composed of adjacent cells, within the set I=1.
    # Following this definition, we have
    # * x <-> y iff C(x) = C(y) > 0
    fname = 'imgGeobodyImage'

    # Check
    if connect_type not in ('connect_face', 'connect_face_edge', 'connect_face_edge_corner'):
        err_msg = f'{fname}: unknown `connect_type`'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    if var_index < 0 or var_index >= input_image.nv:
        err_msg = f'{fname}: `var_index` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    if bound_sup is None:
        bound_sup = 1. + np.nanmax(input_image.val[var_index])

    # Set C function to launch for computing geobody image
    if connect_type == 'connect_face':
        g = geosclassic.MPDSImageGeobody6
    elif connect_type == 'connect_face_edge':
        g = geosclassic.MPDSImageGeobody18
    elif connect_type == 'connect_face_edge_corner':
        g = geosclassic.MPDSImageGeobody26
    else:
        err_msg = f'{fname}: `connect_type` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Set input image "in C"
    try:
        input_image_c = img_py2C(input_image, logger=logger)
    except Exception as exc:
        err_msg = f'{fname}: cannot convert input image from python to C'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg) from exc

    # Allocate variable in C
    rangeValueMin_c = geosclassic.new_real_array(1)
    geosclassic.mpds_set_real_vector_from_array(rangeValueMin_c, 0, np.array([bound_inf], dtype='float'))

    rangeValueMax_c = geosclassic.new_real_array(1)
    geosclassic.mpds_set_real_vector_from_array(rangeValueMax_c, 0, np.array([bound_sup], dtype='float'))

    ngeobody_c = geosclassic.new_int_array(1)

    # Allocate output image "in C"
    output_image_c = geosclassic.malloc_MPDS_IMAGE()
    geosclassic.MPDSInitImage(output_image_c)

    # Launch C function
    err = g(input_image_c, output_image_c, var_index,
            complementary_set,
            1, rangeValueMin_c, rangeValueMax_c, bound_inf_excluded, bound_sup_excluded,
            ngeobody_c)

    # Retrieve output image "in python"
    if err:
        # Free memory on C side: input_image_c
        geosclassic.MPDSFreeImage(input_image_c)
        geosclassic.free_MPDS_IMAGE(input_image_c)
        # Free memory on C side: output_image_c
        geosclassic.MPDSFreeImage(output_image_c)
        geosclassic.free_MPDS_IMAGE(output_image_c)
        # Free memory on C side: rangeValueMin_c, rangeValueMax_c, ngeobody_c
        geosclassic.delete_real_array(rangeValueMin_c)
        geosclassic.delete_real_array(rangeValueMax_c)
        geosclassic.delete_int_array(ngeobody_c)
        # Raise error
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        err_msg = f'{fname}: {err_message}'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    output_image = img_C2py(output_image_c, logger=logger)
    # # Retrieve the number of geobody (not used, this is simple the max of the output image (max label))
    # ngeobody = np.zeros(1, dtype='intc') # 'intc' for C-compatibility
    # geosclassic.mpds_get_array_from_int_vector(ngeobody_c, 0, ngeobody)
    # ngeobody = ngeobody[0]

    # Free memory on C side: input_image_c
    geosclassic.MPDSFreeImage(input_image_c)
    geosclassic.free_MPDS_IMAGE(input_image_c)

    # Free memory on C side: output_image_c
    geosclassic.MPDSFreeImage(output_image_c)
    geosclassic.free_MPDS_IMAGE(output_image_c)

    # Free memory on C side: rangeValueMin_c, rangeValueMax_c, ngeobody_c
    geosclassic.delete_real_array(rangeValueMin_c)
    geosclassic.delete_real_array(rangeValueMax_c)
    geosclassic.delete_int_array(ngeobody_c)

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
        show_progress=None,
        nthreads=-1,
        verbose=0,
        logger=None):
    """
    Computes two-point statistics image (map) for one variable of the input image.

    Two-point statistics g(h) as function of lag vector h are computed. Let v(x)
    the value of the considered variable at grid cell x; the available two-point
    statistics (according to parameter `stat_type`) are:

    .. list-table::
        :widths: 25 45
        :header-rows: 1

        *   - `stat_type`
            - Two-point statistics
        *   - correlogram
            - :math:`g(h) = \\operatorname{cor}(v(x), v(x+h)) \\text{ (linear correlation)}`
        *   - connectivity_func0
            - :math:`g(h) = \\mathbb{P}\\left(v(x)=v(x+h) > 0\\right)`
        *   - connectivity_func1
            - :math:`g(h) = \\mathbb{P}\\left(v(x)=v(x+h) > 0 \\ \\vert\\  v(x) > 0\\right)`
        *   - connectivity_func2
            - :math:`g(h) = \\mathbb{P}\\left(v(x)=v(x+h) > 0 \\ \\vert\\  v(x) > 0, v(x+h) > 0\\right)`
        *   - covariance (default)
            - :math:`g(h) = \\operatorname{cov}(v(x), v(x+h))`
        *   - covariance_not_centered
            - :math:`g(h) = \\mathbb{E}\\left[v(x)*v(x+h)\\right]`
        *   - transiogram
            - :math:`g(h) = \\mathbb{P}\\left(v(x+h) > 0 \\ \\vert\\  v(x) > 0\\right)`
        *   - variogram
            - :math:`g(h) = 1/2 \\cdot \\mathbb{E}\\left[(v(x)-v(x+h))^2\\right]`

    A transiogram can be applied on a binary variable.

    A connectivity function (connectivity_func[012]) should be applied on
    a variable consisting of geobody (connected component) labels,
    i.e. input_image should be the output image returned by the function
    `imgGeobodyImage`;
    in that case, denoting I(x) is the indicator variable defined as
    :math:`I(x) = 1 \\iff v(x)>0`, the variable v is the geobody
    label for the indicator variable I an we have the relations

    .. math::
        \\begin{array}{l}
            \\mathbb{P}\\left(v(x) = v(x+h) > 0\\right) \\\\[2mm]
            \\quad = \\mathbb{P}\\left(v(x)=v(x+h) > 0 \\ \\vert\\  v(x) > 0, v(x+h) > 0\\right) \\cdot \\mathbb{P}\\left(v(x) > 0, v(x+h) > 0\\right) \\\\[2mm]
            \\quad = \\mathbb{P}\\left(v(x)=v(x+h) > 0 \\ \\vert\\  v(x) > 0, v(x+h) > 0\\right) \\cdot \\mathbb{P}\\left(I(x) \\cdot I(x+h)\\right) \\\\[2mm]
            \\quad = \\mathbb{P}\\left(v(x)=v(x+h) > 0 \\ \\vert\\  v(x) > 0, v(x+h) > 0\\right) \\cdot \\mathbb{E}\\left(I(x) \\cdot I(x+h)\\right)
        \\end{array}

    i.e.

    .. math::
        \\mathbb{P}(x \\leftrightarrow x+h) = \\mathbb{P}\\left(x \\leftrightarrow x+h \\ \\vert \\ x, x+h \\in \\{I=1\\}\\right) \\cdot \\mathbb{E}(I(x) \\cdot I(x+h))

    that is "connectivity_func0(v) = connectivity_func2(v)*covariance_not_centered(I)"
    (see definition of "is connected to" (:math:`\\leftrightarrow`) in the function
    :func:`geone.geosclassicinterface.imgGeobodyImage`).

    The output image has one variable and its grid is defined according the
    considered lags h defined according to the parameters:

    * `hx_min`, `hx_max`, `hx_step`
    * `hy_min`, `hy_max`, `hy_step`
    * `hz_min`, `hz_max`, `hz_step`

    The minimal (resp. maximal) lag in x direction, expressed in number of cells
    (in the input image), is given by `hx_min` (resp. `hx_max`); every `hx_step`
    cells from `hx_min` up to at most `hx_max` are considered as lag in x
    direction. Hence, the output image grid will have
    `1 + (hx_max-hx_min)//hx_step` cells in x direction.
    This is similar for y and z direction.

    For example, `hx_min=-10, hx_max=10, hx_step=2` implies lags in x direction
    of -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10 cells (in input image).

    Parameters
    ----------
    input_image : :class:`geone.img.Img`
        input image

    var_index : int, default: 0
        index of the considered variable in input image

    hx_min : int, optional
        min lag along x axis, expressed in number of cells;
        by default (`None`): `hx_min = - (input_image.nx // 2)` is used

    hx_max : int, optional
        max lag along x axis, expressed in number of cells;
        by default (`None`): `hx_max = input_image.nx // 2` is used

    hx_step : int, optional
        step for lags along x axis, expressed in number of cells;
        by default (`None`): `hx_step = 1` is used

    hy_min : int, optional
        min lag along y axis, expressed in number of cells;
        by default (`None`): `hy_min = - (input_image.ny // 2)` is used

    hy_max : int, optional
        max lag along y axis, expressed in number of cells;
        by default (`None`): `hy_max = input_image.ny // 2` is used

    hy_step : int, optional
        step for lags along y axis, expressed in number of cells;
        by default (`None`): `hy_step = 1` is used

    hz_min : int, optional
        min lag along z axis, expressed in number of cells;
        by default (`None`): `hz_min = - (input_image.nz // 2)` is used

    hz_max : int, optional
        max lag along z axis, expressed in number of cells;
        by default (`None`): `hz_max = input_image.nz // 2` is used

    hz_step : int, optional
        step for lags along z axis, expressed in number of cells;
        by default (`None`): `hz_step = 1` is used

    stat_type : str {'correlogram', \
                'connectivity_func0', 'connectivity_func1', 'connectivity_func2', \
                'covariance', 'covariance_not_centered', \
                'transiogram', 'variogram'}, default: 'covariance'
        type of two-point statistics  (see above);
        for type 'connectivity_func[012]', the input image is assumed to be a
        geobody image (see above)

    show_progress : bool, optional
        deprecated, use `verbose` instead;

        - if `show_progress=False`, `verbose` is set to 1 (overwritten)
        - if `show_progress=True`, `verbose` is set to 2 (overwritten)
        - if `show_progress=None` (default): not used

    nthreads : int, default: -1
        number of thread(s) to use for C program;
        `nthreads = -n <= 0`: maximal number of threads of the system except n
        (but at least 1)

    verbose : int, default: 0
        verbose mode, higher implies more printing (info)
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    output_image : :class:`geone.img.Img`
        two-point statistics image (one variable)

    References
    ----------
    - P. Renard, D. Allard (2013), \
    Connectivity metrics for subsurface flow and transport. \
    Advances in Water Resources 51:168-196, \
    `doi:10.1016/j.advwatres.2011.12.001 <https://doi.org/10.1016/j.advwatres.2011.12.001>`_
    """
    # * correlogram            : g(h) = cor(v(x), v(x+h)) (linear correlation)
    # * connectivity_func0     : g(h) = P(v(x)=v(x+h) > 0)
    # * connectivity_func1     : g(h) = P(v(x)=v(x+h) > 0 | v(x) > 0)
    # * connectivity_func2     : g(h) = P(v(x)=v(x+h) > 0 | v(x) > 0, v(x+h) > 0)
    # * covariance (default)   : g(h) = cov(v(x), v(x+h))
    # * covariance_not_centered: g(h) = E[v(x)*v(x+h)]
    # * transiogram            : g(h) = P(v(x+h) > 0 | v(x) > 0)
    # * variogram              : g(h) = 0.5 * E[(v(x)-v(x+h))**2]

    # * P(v(x) = v(x+h) > 0)
    #         = P(v(x)=v(x+h) > 0 | v(x) > 0, v(x+h) > 0) * P(v(x) > 0, v(x+h) > 0)
    #         = P(v(x)=v(x+h) > 0 | v(x) > 0, v(x+h) > 0) * P(I(x)*I(x+h))
    #         = P(v(x)=v(x+h) > 0 | v(x) > 0, v(x+h) > 0) * E(I(x)*I(x+h))
    # i.e.
    # * P(x <-> x+h) = P(x <-> x+h | x, x+h in {I=1}) * E(I(x)*I(x+h))
    fname = 'imgTwoPointStatisticsImage'

    # Set verbose mode according to show_progress (if given)
    if show_progress is not None:
        if show_progress:
            verbose = 2
        else:
            verbose = 1

    # Check
    if stat_type not in ('correlogram',
                         'connectivity_func0',
                         'connectivity_func1',
                         'connectivity_func2',
                         'covariance',
                         'covariance_not_centered',
                         'transiogram',
                         'variogram'):
        err_msg = f'{fname}: unknown `stat_type`'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    if var_index < 0 or var_index >= input_image.nv:
        err_msg = f'{fname}: `var_index` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Prepare parameters
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

    # Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose > 0 and nth > os.cpu_count():
        if logger:
            logger.warning(f'{fname}: number of threads used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: number of threads used will exceed number of cpu(s) of the system...')

    # Set C function to launch for computing two-point statistics
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
        err_msg = f'{fname}: `stat_type` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Set input image "in C"
    try:
        input_image_c = img_py2C(input_image, logger=logger)
    except Exception as exc:
        err_msg = f'{fname}: cannot convert input image from python to C'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg) from exc

    # Allocate output image "in C"
    output_image_c = geosclassic.malloc_MPDS_IMAGE()
    geosclassic.MPDSInitImage(output_image_c)

    # Launch C function
    err = g(input_image_c, output_image_c, var_index,
            hx_min, hx_max, hx_step,
            hy_min, hy_max, hy_step,
            hz_min, hz_max, hz_step,
            verbose > 1, nth)

    # Retrieve output image "in python"
    if err:
        # Free memory on C side: input_image_c
        geosclassic.MPDSFreeImage(input_image_c)
        geosclassic.free_MPDS_IMAGE(input_image_c)
        # Free memory on C side: output_image_c
        geosclassic.MPDSFreeImage(output_image_c)
        geosclassic.free_MPDS_IMAGE(output_image_c)
        # Raise error
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        err_msg = f'{fname}: {err_message}'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    output_image = img_C2py(output_image_c, logger=logger)

    # Free memory on C side: input_image_c
    geosclassic.MPDSFreeImage(input_image_c)
    geosclassic.free_MPDS_IMAGE(input_image_c)

    # Free memory on C side: output_image_c
    geosclassic.MPDSFreeImage(output_image_c)
    geosclassic.free_MPDS_IMAGE(output_image_c)

    return output_image
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imgConnectivityGammaValue(
        input_image,
        var_index=0,
        geobody_image_in_input=False,
        complementary_set=False,
        connect_type='connect_face',
        logger=None):
    """
    Computes the Gamma value for one variable v of the input image.

    The Gamma (:math:`\\Gamma`) value is defined as

    .. math::
        \\Gamma = \\frac{1}{m^2} \\sum_{i=1}^N n(i)^2

    where

    * :math:`N` is the number of connected components (geobodies) of the set {v>0}
    * :math:`n(i)` is the size (number of cells) in the i-th connected component
    * :math:`m` is the size (number of cells) of the set {v>0}

    Note that the Gamma value is set to 1.0 if N = 0.

    The definition of adjacent cells, required to compute the connected
    components, is set according to the parameter `connect_type`:

    .. list-table::
        :widths: 25 45
        :header-rows: 1

        *   - `connect_type`
            - two grid cells are adjacent if they have
        *   - 'connect_face' (default)
            - a common face
        *   - 'connect_face_edge'
            - a common face or a common edge
        *   - 'connect_face_edge_corner'
            - a common face or a common edge or a common corner

    Parameters
    ----------
    input_image : :class:`geone.img.Img`
        input image

    var_index : int, default: 0
        index of the considered variable in input image

    geobody_image_in_input : bool, default: False
        - if `True`: the input image is already a geobody image (variable \
        `var_index` is the geobody label); in this case the parameters \
        `complementary_set` and `connect_type` are not used
        - `False`: the geobody image for the indicator variable {v>0} (with v \
        the variable of index `var_index`) is computed

    complementary_set : bool, default: False
        - if `True`: the complementary indicator variable (IC = 1-I, see above) \
        is used
        - if `False`: the indicator variable I (see above) is used

    connect_type : str {'connect_face', 'connect_face_edge', \
            'connect_face_edge_corner'}, default: 'connect_face'
        indicates which definition of adjacent cells is used (see above)
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    gamma : float
        Gamma value (see above)

    Notes
    -----
    The Gamma value is a global indicator of the connectivity for the binary
    image of variable I


    References
    ----------
    - P. Renard, D. Allard (2013), \
    Connectivity metrics for subsurface flow and transport. \
    Advances in Water Resources 51:168-196, \
    `doi:10.1016/j.advwatres.2011.12.001 <https://doi.org/10.1016/j.advwatres.2011.12.001>`_
    """
    fname = 'imgConnectivityGammaValue'

    # --- Check and prepare
    if var_index < 0 or var_index >= input_image.nv:
        err_msg = f'{fname}: `var_index` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    if not geobody_image_in_input and connect_type not in ('connect_face', 'connect_face_edge', 'connect_face_edge_corner'):
        err_msg = f'{fname}: unknown `connect_type`'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Compute geobody image
    if not geobody_image_in_input:
        try:
            im_geobody = imgGeobodyImage(
                    input_image,
                    var_index,
                    bound_inf=0.0,
                    bound_sup=None,
                    bound_inf_excluded=True,
                    bound_sup_excluded=True,
                    complementary_set=complementary_set,
                    connect_type=connect_type,
                    logger=logger)
            
        except Exception as exc:
            err_msg = f'{fname}: cannot compute geobody image'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

        iv = 0
    else:
        im_geobody = input_image
        iv = var_index

    # Compute Gamma value
    ngeo = int(im_geobody.val[iv].max())
    if ngeo == 0:
        gamma = 1.0
    else:
        gamma = np.sum(np.array([float(np.sum(im_geobody.val[iv] == i))**2 for i in np.arange(1, ngeo+1)])) / float(np.sum(im_geobody.val[iv] != 0))**2

    return gamma
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imgConnectivityGammaCurves(
        input_image,
        threshold_min=None,
        threshold_max=None,
        nthreshold=50,
        connect_type='connect_face',
        show_progress=None,
        nthreads=-1,
        verbose=0,
        logger=None):
    """
    Computes Gamma curves for an input image with one continuous variable.

    For a threshold t, we consider the indicator variable I(t) defined as

    * :math:`I(t)(x) = 1 \\iff v(x) \\leqslant t`

    and we compute

    * :math:`\\Gamma(t) = \\frac{1}{m(t)^2} \\sum_{i=1}^{N(t)} n(t, i)^2`

    where

    * :math`N(t)` is the number of connected components (geobodies) of the set {I(t)=1}
    * :math`n(t, i) is the size (number of cells) in the i-th connected component
    * :math`m(t)` is the size (number of cells) of the set {I(t)=1}

    Note: gamma(t) is set to 1.0 if N = 0.

    We also compute :math:`\\Gamma_C(t)`, the gamma value for the complementary set
    {IC(t)=1} where IC(t)(x) = 1 - I(t)(x).

    This is repeated for different threshold values t, which gives the curves
    Gamma(t) (i.e. :math:`\\Gamma(t)`) and GammaC(t) (i.e. :math:`\\Gamma_C(t)`).

    The definition of adjacent cells, required to compute the connected
    components, is set according to the parameter `connect_type`:

    .. list-table::
        :widths: 25 45
        :header-rows: 1

        *   - `connect_type`
            - two grid cells are adjacent if they have
        *   - 'connect_face' (default)
            - a common face
        *   - 'connect_face_edge'
            - a common face or a common edge
        *   - 'connect_face_edge_corner'
            - a common face or a common edge or a common corner

    Parameters
    ----------
    input_image : :class:`geone.img.Img`
        input image, it should have only one variable

    threshold_min : float, optional
        minimal value of the threshold;
        by default (`None`): min of the input variable values minus 1.e-10

    threshold_max : float, optional
        maximal value of the threshold;
        by default (`None`): max of the input variable values plus 1.e-10

    nthreshod : int, default: 50
        number of thresholds considered, the threshold values are
        `numpy.linspace(threshold_min, threshold_max, nthreshold)`

    connect_type : str {'connect_face', 'connect_face_edge', \
            'connect_face_edge_corner'}, default: 'connect_face'
        indicates which definition of adjacent cells is used (see above)

    show_progress : bool, optional
        deprecated, use `verbose` instead;

        - if `show_progress=False`, `verbose` is set to 1 (overwritten)
        - if `show_progress=True`, `verbose` is set to 2 (overwritten)
        - if `show_progress=None` (default): not used

    nthreads : int, default: -1
        number of thread(s) to use for C program;
        `nthreads = -n <= 0`: maximal number of threads of the system except n
        (but at least 1)

    verbose : int, default: 0
        verbose mode, higher implies more printing (info)
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    out_array : 2D array of floats of shape (ntrheshold, 3)
        the columns correspond to: the threshold values, the Gamma values, and
        the GammaC values, i.e.:

        - `out_array[i, 0]`:\
        `numpy.linspace(threshold_min, threshold_max, nthreshold)`
        - `out_array[i, 1]`: Gamma(out_array[i, 0])
        - `out_array[i, 2]`: GammaC(out_array[i, 0])

    Notes
    -----
    The Gamma value Gamma(t) (resp. GammaC(t)) is a global indicator of the
    connectivity for the binary variable I(t) (resp. IC(t)).

    References
    ----------
    - P. Renard, D. Allard (2013), \
    Connectivity metrics for subsurface flow and transport. \
    Advances in Water Resources 51:168-196, \
    `doi:10.1016/j.advwatres.2011.12.001 <https://doi.org/10.1016/j.advwatres.2011.12.001>`_
    """
    fname = 'imgConnectivityGammaCurves'

    # Set verbose mode according to show_progress (if given)
    if show_progress is not None:
        if show_progress:
            verbose = 2
        else:
            verbose = 1

    # Check and prepare
    if input_image.nv != 1:
        err_msg = f'{fname}: input image must have one variable only'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    if threshold_min is None:
        threshold_min = np.nanmin(input_image.val) - 1.e-10

    if threshold_max is None:
        threshold_max = np.nanmax(input_image.val) + 1.e-10

    if threshold_min > threshold_max:
        err_msg = f'{fname}: `threshold_min` is greater than `threshold_max`'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    if nthreshold < 0:
        err_msg = f'{fname}: `nthreshold` is negative'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    elif nthreshold == 1:
        threshold_step = 1.0
    else:
        threshold_step = (threshold_max - threshold_min) / (nthreshold - 1)

    if threshold_step < geosclassic.MPDS_EPSILON:
        err_msg = f'{fname}: threshold step too small'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    if connect_type not in ('connect_face', 'connect_face_edge', 'connect_face_edge_corner'):
        err_msg = f'{fname}: unknown `connect_type`'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose > 0 and nth > os.cpu_count():
        if logger:
            logger.warning(f'{fname}: number of threads used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: number of threads used will exceed number of cpu(s) of the system...')

    # Set C function to launch for computing Gamma curves
    if connect_type == 'connect_face':
        g = geosclassic.MPDSOMPImageConnectivity6GlobalIndicatorCurve
    elif connect_type == 'connect_face_edge':
        g = geosclassic.MPDSOMPImageConnectivity18GlobalIndicatorCurve
    elif connect_type == 'connect_face_edge_corner':
        g = geosclassic.MPDSOMPImageConnectivity26GlobalIndicatorCurve
    else:
        err_msg = f'{fname}: `connect_type` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Set input image "in C"
    try:
        input_image_c = img_py2C(input_image, logger=logger)
    except Exception as exc:
        err_msg = f'{fname}: cannot convert input image from python to C'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg) from exc

    # Allocate output variable in C
    threshold_c = geosclassic.new_real_array(nthreshold)
    gamma_c = geosclassic.new_real_array(nthreshold)
    gammaC_c = geosclassic.new_real_array(nthreshold)

    # Launch C function
    err = g(input_image_c, nthreshold, threshold_min, threshold_step,
            threshold_c, gamma_c, gammaC_c,
            verbose > 1, nth)

    # Retrieve output "in python"
    if err:
        # Free memory on C side: input_image_c
        geosclassic.MPDSFreeImage(input_image_c)
        geosclassic.free_MPDS_IMAGE(input_image_c)
        # Free memory on C side: threshold_c, gamma_c, gammaC_c
        geosclassic.delete_real_array(threshold_c)
        geosclassic.delete_real_array(gamma_c)
        geosclassic.delete_real_array(gammaC_c)
        # Raise error
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        err_msg = f'{fname}: {err_message}'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    threshold = np.zeros(nthreshold)
    geosclassic.mpds_get_array_from_real_vector(threshold_c, 0, threshold)

    gamma = np.zeros(nthreshold)
    geosclassic.mpds_get_array_from_real_vector(gamma_c, 0, gamma)

    gammaC = np.zeros(nthreshold)
    geosclassic.mpds_get_array_from_real_vector(gammaC_c, 0, gammaC)

    out_array = np.array((threshold, gamma, gammaC)).reshape(3, -1).T

    # Free memory on C side: input_image_c
    geosclassic.MPDSFreeImage(input_image_c)
    geosclassic.free_MPDS_IMAGE(input_image_c)

    # Free memory on C side: threshold_c, gamma_c, gammaC_c
    geosclassic.delete_real_array(threshold_c)
    geosclassic.delete_real_array(gamma_c)
    geosclassic.delete_real_array(gammaC_c)

    return out_array
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imgConnectivityEulerNumber(
        input_image,
        var_index=0,
        geobody_image_in_input=False,
        complementary_set=False,
        nthreads=-1,
        verbose=0,
        logger=None):
    """
    Computes the Euler number for one variable v of the input image.

    The Euler number is defined, for the 3D image grid, as

    * E = #{connected components (geobodies)} + #{holes} - #{handles}

    for the set {v>0}, i.e. the indicator variable
    :math:`I(x) = 1 \\iff v(x) > 0`, is considered.

    The Euler number E can be computed by the formula:

    * E = sum_{i=1,...,N} (e0(i) - e1(i) + e2(i) - e3(i)),

    where

    - N is the number of connected component (geobodies) in the set {I=1}
    - for a geobody i:
        * e0(i) : the number of vertices (dim 0) in the i-th geobody
        * e1(i) : the number of edges (dim 1) in the i-th geobody
        * e2(i) : the number of faces (dim 2) in the i-th geobody
        * e3(i) : the number of volumes (dim 3) in the i-th geobody

        where vertices, edges, faces, and volumes of each grid cell
        (3D parallelepiped element) are considered.

    Note that the connected components are computed considering two cells as
    adjacent as soon as they have a common face (`connect_type='connect_face'`
    for the computation of the geobody image (see function :func:`imgGeobodyImage`).

    Parameters
    ----------
    input_image : :class:`geone.img.Img`
        input image

    var_index : int, default: 0
        index of the considered variable in input image

    geobody_image_in_input : bool, default: False
        - if `True`: the input image is already a geobody image (variable \
        `var_index` is the geobody label); in this case the parameter \
        `complementary_set` is not used
        - if `False`: the geobody image for the indicator variable {v>0} (with v \
        the variable of index `var_index`) is computed

    complementary_set : bool, default: False
        - if `True`: the complementary indicator variable (IC = 1-I, see above) \
        is used
        - if `False`: the indicator variable I (see above) is used

    nthreads : int, default: -1
        number of thread(s) to use for C program;
        `nthreads = -n <= 0`: maximal number of threads of the system except n
        (but at least 1)

    verbose : int, default: 0
        verbose mode, higher implies more printing (info)
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    euler_number : float
        Euler number (see above)

    References
    ----------
    - P. Renard, D. Allard (2013), \
    Connectivity metrics for subsurface flow and transport. \
    Advances in Water Resources 51:168-196, \
    `doi:10.1016/j.advwatres.2011.12.001 <https://doi.org/10.1016/j.advwatres.2011.12.001>`_
    """
    fname = 'imgConnectivityEulerNumber'

    # --- Check and prepare
    if var_index < 0 or var_index >= input_image.nv:
        err_msg = f'{fname}: `var_index` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Compute geobody image
    if not geobody_image_in_input:
        try:
            im_geobody = imgGeobodyImage(
                    input_image,
                    var_index,
                    bound_inf=0.0,
                    bound_sup=None,
                    bound_inf_excluded=True,
                    bound_sup_excluded=True,
                    complementary_set=complementary_set,
                    connect_type='connect_face',
                    logger=logger)
            
        except Exception as exc:
            err_msg = f'{fname}: cannot compute geobody image'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc
        iv = 0
    else:
        im_geobody = input_image
        iv = var_index

    # Compute Euler Number
    # Set geobody image "in C"
    try:
        im_geobody_c = img_py2C(im_geobody, logger=logger)
    except Exception as exc:
        err_msg = f'{fname}: cannot convert geobody image from python to C'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg) from exc

    # Allocate euler number "in C"
    euler_number_c = geosclassic.new_int_array(1)

    # Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose > 0 and nth > os.cpu_count():
        if logger:
            logger.warning(f'{fname}: number of threads used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: number of threads used will exceed number of cpu(s) of the system...')

    # Compute Euler number (launch C code)
    err = geosclassic.MPDSOMPImageConnectivityEulerNumber(im_geobody_c, var_index, euler_number_c, nth)

    # Retrieve output "in python"
    if err:
        # Free memory on C side: im_geobody_c
        geosclassic.MPDSFreeImage(im_geobody_c)
        geosclassic.free_MPDS_IMAGE(im_geobody_c)
        # Free memory on C side: euler_number_c
        geosclassic.delete_int_array(euler_number_c)
        # Raise error
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        err_msg = f'{fname}: {err_message}'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    euler_number = np.zeros(1, dtype='intc') # 'intc' for C-compatibility
    geosclassic.mpds_get_array_from_int_vector(euler_number_c, 0, euler_number)
    euler_number = euler_number[0]

    # Free memory on C side: im_geobody_c
    geosclassic.MPDSFreeImage(im_geobody_c)
    geosclassic.free_MPDS_IMAGE(im_geobody_c)

    # Free memory on C side: euler_number_c
    geosclassic.delete_int_array(euler_number_c)

    return euler_number
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def imgConnectivityEulerNumberCurves(
        input_image,
        threshold_min=None,
        threshold_max=None,
        nthreshold=50,
        show_progress=None,
        nthreads=-1,
        verbose=0,
        logger=None):
    """
    Computes the curves of Euler number for one variable v of the input image.

    For a threshold t, we consider the indicator variable I(t) defined as

    * :math:`I(t)(x) = 1 \\iff v(x) \\leqslant t`

    and we compute the Euler number

    * E(t) = #{connected components (geobodies)} + #{holes} - #{handles}

    for the set {I(t)=1}; we compute also EC(t), the Euler number for the
    complementary set {IC(t)=1} where IC(t)(x) = 1 - I(t)(x)

    This is repeated for different threshold values t, which gives the curves
    of Euler numbers E(t) and EC(t).

    See function :func:`imgConnectivityEulerNumber` for details about Euler number.

    Note that the connected components are computed considering two cells as
    adjacent as soon as they have a common face (`connect_type='connect_face'`
    for the computation of the geobody image (see function :func:`imgGeobodyImage`).

    Parameters
    ----------
    input_image : :class:`geone.img.Img`
        input image, it should have only one variable

    threshold_min : float, optional
        minimal value of the threshold;
        by default (`None`): min of the input variable values minus 1.e-10

    threshold_max : float, optional
        maximal value of the threshold;
        by default (`None`): max of the input variable values plus 1.e-10

    nthreshod : int, default: 50
        number of thresholds considered, the threshold values are
        `numpy.linspace(threshold_min, threshold_max, nthreshold)`

    show_progress : bool, optional
        deprecated, use `verbose` instead;

        - if `show_progress=False`, `verbose` is set to 1 (overwritten)
        - if `show_progress=True`, `verbose` is set to 2 (overwritten)
        - if `show_progress=None` (default): not used

    nthreads : int, default: -1
        number of thread(s) to use for C program;
        `nthreads = -n <= 0`: maximal number of threads of the system except n
        (but at least 1)

    verbose : int, default: 0
        verbose mode, higher implies more printing (info)
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    out_array : 2D array of floats of shape (ntrheshold, 3)
        the columns correspond to: the threshold values, the Euler numbers E,
        and the Euler numbers EC, i.e.:

        - `out_array[i, 0]`:\
        `numpy.linspace(threshold_min, threshold_max, nthreshold)`
        - `out_array[i, 1]`: E(out_array[i, 0])
        - `out_array[i, 2]`: EC(out_array[i, 0])
    """
    fname = 'imgConnectivityEulerNumberCurves'

    # Set verbose mode according to show_progress (if given)
    if show_progress is not None:
        if show_progress:
            verbose = 2
        else:
            verbose = 1

    # Check and prepare
    if input_image.nv != 1:
        err_msg = f'{fname}: input image must have one variable only'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    if threshold_min is None:
        threshold_min = np.nanmin(input_image.val) - 1.e-10

    if threshold_max is None:
        threshold_max = np.nanmax(input_image.val) + 1.e-10

    if threshold_min > threshold_max:
        err_msg = f'{fname}: `threshold_min` is greater than `threshold_max`'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    if nthreshold < 0:
        err_msg = f'{fname}: `nthreshold` is negative'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    elif nthreshold == 1:
        threshold_step = 1.0
    else:
        threshold_step = (threshold_max - threshold_min) / (nthreshold - 1)

    if threshold_step < geosclassic.MPDS_EPSILON:
        err_msg = f'{fname}: threshold step too small'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Set input image "in C"
    try:
        input_image_c = img_py2C(input_image, logger=logger)
    except Exception as exc:
        err_msg = f'{fname}: cannot convert input image from python to C'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg) from exc

    # Allocate output variable in C
    threshold_c = geosclassic.new_real_array(nthreshold)
    euler_number_c = geosclassic.new_int_array(nthreshold)
    euler_numberC_c = geosclassic.new_int_array(nthreshold)

    # Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose > 0 and nth > os.cpu_count():
        if logger:
            logger.warning(f'{fname}: number of threads used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: number of threads used will exceed number of cpu(s) of the system...')

    # Compute Euler number curves (launch C code)
    err = geosclassic.MPDSOMPImageConnectivity6EulerNumberCurve(
            input_image_c, nthreshold, threshold_min, threshold_step,
            threshold_c, euler_number_c, euler_numberC_c,
            verbose > 1, nth)

    # Retrieve output "in python"
    if err:
        # Free memory on C side: input_image_c
        geosclassic.MPDSFreeImage(input_image_c)
        geosclassic.free_MPDS_IMAGE(input_image_c)
        # Free memory on C side: threshold_c, gamma_c, gammaC_c
        geosclassic.delete_real_array(threshold_c)
        geosclassic.delete_int_array(euler_number_c)
        geosclassic.delete_int_array(euler_numberC_c)
        # Raise error
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        err_msg = f'{fname}: {err_message}'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    threshold = np.zeros(nthreshold)
    geosclassic.mpds_get_array_from_real_vector(threshold_c, 0, threshold)

    euler_number = np.zeros(nthreshold, dtype='intc') # 'intc' for C-compatibility
    geosclassic.mpds_get_array_from_int_vector(euler_number_c, 0, euler_number)

    euler_numberC = np.zeros(nthreshold, dtype='intc') # 'intc' for C-compatibility
    geosclassic.mpds_get_array_from_int_vector(euler_numberC_c, 0, euler_numberC)

    out_array = np.array((threshold, euler_number, euler_numberC)).reshape(3, -1).T

    # Free memory on C side: input_image_c
    geosclassic.MPDSFreeImage(input_image_c)
    geosclassic.free_MPDS_IMAGE(input_image_c)

    # Free memory on C side: threshold_c, gamma_c, gammaC_c
    geosclassic.delete_real_array(threshold_c)
    geosclassic.delete_int_array(euler_number_c)
    geosclassic.delete_int_array(euler_numberC_c)

    return out_array
# ----------------------------------------------------------------------------

# ============================================================================
# Below : old version - "deprecated", but kept for compatibility
# ============================================================================

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
        aggregate_data_op=None,
        aggregate_data_op_kwargs=None,
        aggregate_data_ineqMin_op='max',
        aggregate_data_ineqMin_op_kwargs=None,
        aggregate_data_ineqMax_op='min',
        aggregate_data_ineqMax_op_kwargs=None,
        mask=None,
        add_data_point_to_mask=True,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        nGibbsSamplerPathMin=50,
        nGibbsSamplerPathMax=200,
        seed=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2,
        logger=None):
    """
    Generates 1D simulations (Sequential Gaussian Simulation, SGS).

    A simulation takes place in (center of) grid cells, based on simple or
    ordinary kriging.

    Parameters
    ----------
    cov_model : :class:`geone.covModel.CovModel1D`
        covariance model in 1D

    dimension : int
        `dimension=nx`, number of cells in the 1D simulation grid

    spacing : float, default: 1.0
        `spacing=sx`, cell size

    origin : float, default: 0.0
        `origin=ox`, origin of the 1D simulation grid (left border)

    method : str {'simple_kriging', 'ordinary_kriging'}, default: 'simple_kriging'
        type of kriging

    nreal : int, default: 1
        number of realizations

    mean : function (callable), or array-like of floats, or float, optional
        kriging mean value:

        - if a function: function of one argument (xi) that returns the mean at \
        location xi
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), mean values at grid cells (for \
        non-stationary mean)
        - if a float: same mean value at every grid cell
        - by default (`None`): the mean of data value (`v`) (0.0 if no data) is \
        considered at every grid cell

    var : function (callable), or array-like of floats, or float, optional
        kriging variance value:

        - if a function: function of one argument (xi) that returns the variance \
        at location xi
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), variance values at grid cells (for \
        non-stationary variance)
        - if a float: same variance value at every grid cell
        - by default (`None`): not used (use of covariance model only)

    x : 1D array-like of floats, optional
        data points locations (float coordinates); note: if one point, a float
        is accepted

    v : 1D array-like of floats, optional
        data values at `x` (`v[i]` is the data value at `x[i]`), array of same
        length as `x` (or float if one point)

    xIneqMin : 1D array-like of floats, optional
        data points locations (float coordinates), for inequality data with
        lower bound; note: if one point, a float is accepted

    vIneqMin : 1D array-like of floats, optional
        inequality data values, lower bounds, at `xIneqMin` (`vIneqMin[i]` is the
        data value at `xIneqMin[i]`), array of same length as `xIneqMin` (or
        float if one point)

    xIneqMax : 1D array-like of floats, optional
        data points locations (float coordinates), for inequality data with
        upper bound; note: if one point, a float is accepted

    vIneqMax : 1D array-like of floats, optional
        inequality data values, upper bounds, at `xIneqMax` (`vIneqMax[i]` is the
        data value at `xIneqMax[i]`), array of same length as `xIneqMax`  (or
        float if one point)

    aggregate_data_op : str {'sgs', 'krige', 'min', 'max', 'mean', 'quantile', \
                        'most_freq', 'random'}, optional
        operation used to aggregate data points falling in the same grid cells

        - if `aggregate_data_op='sgs'`: function :func:`geone.covModel.sgs` is used \
        with the covariance model `cov_model` given in arguments, as well as \
        the parameter `nneighborMax` given in arguments unless it is given \
        in `aggregate_data_op_kwargs`
        - if `aggregate_data_op='krige'`: function :func:`geone.covModel.krige` is used \
        with the covariance model `cov_model` given in arguments, as well as \
        the parameters `use_unique_neighborhood`, `nneighborMax` given in \
        arguments unless they are given in `aggregate_data_op_kwargs`
        - if `aggregate_data_op='most_freq'`: most frequent value is selected \
        (smallest one if more than one value with the maximal frequence)
        - if `aggregate_data_op='random'`: value from a random point is selected \
        - otherwise: the function `numpy.<aggregate_data_op>` is used with the \
        additional parameters given by `aggregate_data_op_kwargs`, note that, e.g. \
        `aggregate_data_op='quantile'` requires the additional parameter \
        `q=<quantile_to_compute>`

        Note: if `aggregate_data_op='sgs'` or `aggregate_data_op='random'`, the
        aggregation is done for each realization (simulation), i.e. each simulation
        on the grid starts with a new set of values in conditioning grid cells

        By default: if covariance model has stationary ranges and weight (sill),
        `aggregate_data_op='sgs'` is used, otherwise `aggregate_data_op='mean'`

    aggregate_data_op_kwargs : dict, optional
        keyword arguments to be passed to `geone.covModel.sgs`,
        `geone.covModel.krige`, or `numpy.<aggregate_data_op>`, according to
        the parameter `aggregate_data_op`

    aggregate_data_ineqMin_op : str {'min', 'max', 'mean', 'quantile', \
                        'most_freq', 'random'}, default: 'max'
        operation used to aggregate inequality (min, lower boudns) data points
        falling in the same grid cells:

        - if `aggregate_data_ineqMin_op='most_freq'`: most frequent value is \
        selected (smallest one if more than one value with the maximal frequence)
        - if `aggregate_data_ineqMin_op='random'`: value from a random point is \
        selected
        - otherwise: the function `numpy.<aggregate_data_ineqMin_op>` is used with \
        the additional parameters given by `aggregate_data_ineqMin_op_kwargs`, \
        note that, e.g. `aggregate_data_ineqMin_op='quantile'` requires the \
        additional parameter `q=<quantile_to_compute>`

        Note: in any case, the aggregation is done once, i.e. same inequality
        values are used for each simulation on the grid

    aggregate_data_ineqMin_op_kwargs : dict, optional
        keyword arguments to be passed to `numpy.<aggregate_data_ineqMin_op>`,
        according to the parameter `aggregate_data_ineqMin_op`

    aggregate_data_ineqMax_op : str {'min', 'max', 'mean', 'quantile', \
                        'most_freq', 'random'}, default: 'min'
        operation used to aggregate inequality (min, lower boudns) data points
        falling in the same grid cells:

        - if `aggregate_data_ineqMax_op='most_freq'`: most frequent value is \
        selected (smallest one if more than one value with the maximal frequence)
        - if `aggregate_data_ineqMax_op='random'`: value from a random point is \
        selected
        - otherwise: the function `numpy.<aggregate_data_ineqMax_op>` is used with \
        the additional parameters given by `aggregate_data_ineqMax_op_kwargs`, \
        note that, e.g. `aggregate_data_ineqMax_op='quantile'` requires the \
        additional parameter `q=<quantile_to_compute>`

        Note: in any case, the aggregation is done once, i.e. same inequality
        values are used for each simulation on the grid

    aggregate_data_ineqMax_op_kwargs : dict, optional
        keyword arguments to be passed to `numpy.<aggregate_data_ineqMax_op>`,
        according to the parameter `aggregate_data_ineqMax_op`

    mask : array-like, optional
        mask value at grid cells (value 1 for simulated cells, value 0 for not
        simulated cells); the size of the array must be equal to the number of
        grid cells (the array is reshaped if needed)

    add_data_point_to_mask : bool, default: True
        - if `True`: any grid cell that contains a data point is added to (the \
        simulated part of) the mask (if present), i.e. mask value at those cells \
        are set to 1; at the end of the computation the "new mask cells" are \
        removed (by setting a missing value (`numpy.nan`) for the variable out of \
        the original mask)
        - if `False`: original mask is kept as given in input, and data point \
        falling out of (the simulated part of) the mask (if present) are ignored

    searchRadiusRelative : float, default: 1.2
        indicates how the search ellipsoid is limited (should be positive): let
        r_i be the ranges of the covariance model along its main axes, when
        estimating/simulating a cell x, a cell y is taken into account iff it is
        within the ellipsoid centered at x of half axes equal to
        `searchRadiusRelative` * r_i; 
        (note that the distances to the central node are computed in the axes 
        sytem supporting the covariance model and accounting for anisotropy given 
        by the ranges)
        note (Geos-Classic library): if a range r_i is non-stationary over the grid, 
        its maximal value over the grid is considered;

    nneighborMax : int, default: 12
        maximum number of cells retrieved from the search ellipsoid (when
        estimating/simulating a cell), `nneighborMax=-1` for unlimited

    searchNeighborhoodSortMode : int, optional
        indicates how to sort the search neighboorhood cells (neighbors); they
        are sorted in increasing order according to:

        - `searchNeighborhoodSortMode=0`: distance in the usual axes system
        - `searchNeighborhoodSortMode=1`: distance in the axes sytem supporting \
        the covariance model and accounting for anisotropy given by the ranges
        - `searchNeighborhoodSortMode=2`: minus the evaluation of the covariance \
        model

        Notes:

        - if the covariance model has any non-stationary parameter, then \
        `searchNeighborhoodSortMode=2` is not allowed
        - if the covariance model has any non-stationary range or non-stationary \
        angle and `searchNeighborhoodSortMode=1`: "maximal ranges" (adapted to \
        direction from the central cell) are used to compute distance for sorting \
        the neighbors

        By default (`None`): the greatest possible value is used (i.e. 2 for
        stationary covariance model, or 1 otherwise)

    nGibbsSamplerPathMin: int, default: 50
        see `nGibbsSamplerPathMax`

    nGibbsSamplerPathMax: int, default: 200
        `nGibbsSamplerPathMin` and `nGibbsSamplerPathMax` are the mini and max number
        of Gibbs sampler paths to deal with inequality data; the conditioning locations
        with inequality data are first simulated (based on truncated gaussian
        distribution) sequentially; then, these locations are re-simulated following a
        new path as many times as needed, but the total number of paths will be between
        `nGibbsSamplerPathMin` and `nGibbsSamplerPathMax`

    seed : int, optional
        seed for initializing random number generator

    outputReportFile : str, optional
        name of the report file (if desired in output); by default (`None`): no
        report file

    nthreads : int, default: -1
        number of thread(s) to use for "GeosClassicSim" C program;
        `nthreads = -n <= 0`: maximal number of threads of the system except n
        (but at least 1)

    verbose : int, default: 2
        verbose mode, higher implies more printing (info):

        - 0: no display
        - 1: warnings
        - 2: warnings + basic info
        - 3 (or >2): all information

        note that if an error occurred, it is raised
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    geosclassic_output : dict
        geosclassic output in python, dictionary

        {'image':image, 'nwarning':nwarning, 'warnings':warnings}

        with:

        - image : :class:`geone.img.Img`
            output image, with `image.nv=nreal` variables (simulations);
            note: `image=None` if `mpds_geosClassicOutput->outputImage=NULL`

        - nwarning : int
            total number of warning(s) encountered (same warnings can be counted
            several times)

        - warnings : list of strs
            list of distinct warnings encountered (can be empty)
    """
    fname = 'simulate1D'

    if logger:
        logger.warning(f'{fname}: deprecated function, use function `simulate` instead')
    else:
        print(f'{fname}: WARNING: deprecated function, use function `simulate` instead')

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
    # nreal
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose > 0:
            if logger:
                logger.warning(f'{fname}: `nreal` <= 0: `None` is returned')
            else:
                print(f'{fname}: WARNING: `nreal` <= 0: `None` is returned')
        return None

    # cov_model
    if not isinstance(cov_model, gcm.CovModel1D):
        err_msg = f'{fname}: `cov_model` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    for el in cov_model.elem:
        # weight
        w = el[1]['w']
        if np.size(w) != 1 and np.size(w) != nxyz:
            err_msg = f"{fname}: `cov_model`: weight ('w') not compatible with simulation grid"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # ranges
        if 'r' in el[1].keys():
            r  = el[1]['r']
            if np.size(r) != 1 and np.size(r) != nxyz:
                err_msg = f"{fname}: `cov_model`: range ('r') not compatible with simulation grid"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

        # additional parameter (s)
        if 's' in el[1].keys():
            s  = el[1]['s']
            if np.size(s) != 1 and np.size(s) != nxyz:
                err_msg = f"{fname}: `cov_model`: parameter ('s') not compatible with simulation grid"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # aggregate_data_op (default)
    if aggregate_data_op is None:
        if not cov_model.is_weight_stationary() or not cov_model.is_range_stationary():
            aggregate_data_op = 'mean'
        else:
            aggregate_data_op = 'sgs'

    if aggregate_data_op_kwargs is None:
        aggregate_data_op_kwargs = {}

    if aggregate_data_ineqMin_op_kwargs is None:
        aggregate_data_ineqMin_op_kwargs = {}

    if aggregate_data_ineqMax_op_kwargs is None:
        aggregate_data_ineqMax_op_kwargs = {}

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    if method == 'simple_kriging':
        computationMode = 3
    elif method == 'ordinary_kriging':
        computationMode = 2
    else:
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - searchRadiusRelative
    if searchRadiusRelative < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
        err_msg = f'{fname}: `searchRadiusRelative` too small (should be at least {geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN})'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - nneighborMax
    if nneighborMax != -1 and nneighborMax <= 0:
        err_msg = f'{fname}: `nneighborMax` should be greater than 0 or equal to -1 (unlimited)'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - searchNeighborhoodSortMode
    if searchNeighborhoodSortMode is None:
        # set greatest possible value
        if cov_model.is_stationary():
            searchNeighborhoodSortMode = 2
        else:
            searchNeighborhoodSortMode = 1
    else:
        if searchNeighborhoodSortMode == 2:
            if not cov_model.is_stationary():
                err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # if searchNeighborhoodSortMode is None:
    #     # set greatest possible value
    #     if cov_model.is_stationary():
    #         searchNeighborhoodSortMode = 2
    #     elif cov_model.is_orientation_stationary() and cov_model.is_range_stationary():
    #         searchNeighborhoodSortMode = 1
    #     else:
    #         searchNeighborhoodSortMode = 0
    # else:
    #     if searchNeighborhoodSortMode == 2:
    #         if not cov_model.is_stationary():
    #             err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
    #             if logger: logger.error(err_msg)
    #             raise GeosclassicinterfaceError(err_msg)
    #     elif searchNeighborhoodSortMode == 1:
    #         if not cov_model.is_orientation_stationary() or not cov_model.is_range_stationary():
    #             err_msg = f'{fname}: `searchNeighborhoodSortMode=1` not allowed with non-stationary range or non-stationary orientation in covariance model'
    #             if logger: logger.error(err_msg)
    #             raise GeosclassicinterfaceError(err_msg)

    # Preparation of data points
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 1) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if xIneqMin is not None:
        xIneqMin = np.asarray(xIneqMin, dtype='float').reshape(-1, 1) # cast in 2-dimensional array if needed
        vIneqMin = np.asarray(vIneqMin, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(vIneqMin) != xIneqMin.shape[0]:
            err_msg = f'{fname}: length of `vIneqMin` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if xIneqMax is not None:
        xIneqMax = np.asarray(xIneqMax, dtype='float').reshape(-1, 1) # cast in 2-dimensional array if needed
        vIneqMax = np.asarray(vIneqMax, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(vIneqMax) != xIneqMax.shape[0]:
            err_msg = f'{fname}: length of `vIneqMax` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Check parameters - mean
    mean_x = mean
    if mean is not None:
        # if method == 'ordinary_kriging':
        #     err_msg = f'{fname}: specifying `mean` not allowed with ordinary kriging'
        #     if logger: logger.error(err_msg)
        #     raise GeosclassicinterfaceError(err_msg)

        if callable(mean):
            if x is not None:
                mean_x = mean(x[:, 0])
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            mean = mean(xi) # replace function 'mean' by its evaluation on the grid
        else:
            mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if mean.size == 1:
                if x is not None:
                    mean_x = mean
            elif mean.size == nxyz:
                # mean = mean.reshape(nx)
                if x is not None:
                    mean_x = img.Img_interp_func(img.Img(nx, 1, 1, sx, 1., 1., ox, 0., 0., nv=1, val=mean, logger=logger), iy=0, iz=0, logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `mean` is not valid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # Check parameters - var
    var_x = var
    if var is not None:
        if method == 'ordinary_kriging':
            err_msg = f'{fname}: specifying `var` not allowed with ordinary kriging'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        if callable(var):
            if x is not None:
                var_x = var(x[:, 0])
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            var = var(xi) # replace function 'var' by its evaluation on the grid
        else:
            var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if var.size == 1:
                if x is not None:
                    var_x = var
            elif var.size == nxyz:
                # var = var.reshape(nx)
                if x is not None:
                    var_x = img.Img_interp_func(img.Img(nx, 1, 1, sx, 1., 1., ox, 0., 0., nv=1, val=var, logger=logger), iy=0, iz=0, logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `var` is not valid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # Prepare seed
    if seed is None:
        seed = np.random.randint(1, 1000000)
    seed = int(seed)

    # data points: x, v, xIneqMin, vIneqMin, xIneqMax, vIneqMax
    dataPointSet = []

    # data point set from x, v
    aggregate_data_by_simul = False
    if x is not None:
        if aggregate_data_op == 'krige' or aggregate_data_op == 'sgs':
            if not cov_model.is_weight_stationary() or not cov_model.is_range_stationary():
                err_msg = f"{fname}: covariance model with non-stationary weight or range cannot be used with `aggregate_data_op`='{aggregate_data_op}'"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            cov_model_agg = cov_model
            # Get grid cell with at least one data point:
            # x_agg: 2D array, each row contains the coordinates of the center of such cell
            im_tmp = img.imageFromPoints(x, values=None, varname=None,
                                         nx=nx, sx=sx, ox=ox,
                                         indicator_var=True, 
                                         count_var=False,
                                         logger=logger)
            ind_agg = np.where(im_tmp.val[0])
            if len(ind_agg[0]) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            x_agg = im_tmp.xx()[ind_agg].reshape(-1, 1)
            # x_agg = im_tmp.xx()[*ind_agg].reshape(-1, 1) # ok from python 3.11 only ?
            ind_agg = ind_agg[2:] # remove index along z and y axes
            del(im_tmp)
            # Compute
            # - kriging estimate (v_agg) and kriging std (v_agg_std) at x_agg,
            # - or nreal simulation(s) (v_agg) at x_agg
            if mean is not None and mean.size > 1:
                mean_x_agg = mean[ind_agg]
                # mean_x_agg = mean[*ind_agg]
            else:
                mean_x_agg = mean
            if var is not None and var.size > 1:
                var_x_agg = var[ind_agg]
                # var_x_agg = var[*ind_agg]
            else:
                var_x_agg = var
            # Set parameters `nneighborMax` from the arguments if not given in `aggregate_data_op_kwargs`
            if 'nneighborMax' not in aggregate_data_op_kwargs.keys():
                aggregate_data_op_kwargs['nneighborMax'] = nneighborMax
            if aggregate_data_op == 'krige':
                try:
                    v_agg, v_agg_std = gcm.krige(x, v, x_agg, cov_model_agg, method=method,
                                                 mean_x=mean_x, mean_xu=mean_x_agg,
                                                 var_x=var_x, var_xu=var_x_agg,
                                                 verbose=0, logger=logger, 
                                                 **aggregate_data_op_kwargs)
                except Exception as exc:
                    err_msg = f'{fname}: kriging error'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg) from exc

            else:
                aggregate_data_by_simul = True
                try:
                    v_agg = gcm.sgs(x, v, x_agg, cov_model_agg, method=method,
                                    mean_x=mean_x, mean_xu=mean_x_agg,
                                    var_x=var_x, var_xu=var_x_agg,
                                    nreal=nreal, seed=seed,
                                    verbose=0, logger=logger,
                                    **aggregate_data_op_kwargs)
                except Exception as exc:
                    err_msg = f'{fname}: sgs error'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg) from exc

            xx_agg = x_agg[:, 0]
            yy_agg = np.ones_like(xx_agg) * oy + 0.5 * sy
            zz_agg = np.ones_like(xx_agg) * oz + 0.5 * sz

        elif aggregate_data_op == 'random':
            aggregate_data_by_simul = True
            # Aggregate data on grid cell by taking random point
            xx = x[:, 0]
            yy = np.ones_like(xx) * oy + 0.5 * sy
            zz = np.ones_like(xx) * oz + 0.5 * sz
            # first realization of v_agg
            try:
                xx_agg, yy_agg, zz_agg, v_agg, i_inv = img.aggregateDataPointsWrtGrid(
                                                            xx, yy, zz, v,
                                                            nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                            op=aggregate_data_op, 
                                                            return_inverse=True,
                                                            logger=logger,
                                                            **aggregate_data_op_kwargs)
            except Exception as exc:
                err_msg = f'{fname}: data aggregation'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

            if len(xx_agg) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            # next realizations of v_agg
            v_agg = np.vstack((v_agg, np.zeros((nreal-1, v_agg.size))))
            for i in range(1, nreal):
                v_agg[i] = [v[np.random.choice(np.where(i_inv==j)[0])] for j in range(len(xx_agg))]

        else:
            # Aggregate data on grid cell by using the given operation
            xx = x[:, 0]
            yy = np.ones_like(xx) * oy + 0.5 * sy
            zz = np.ones_like(xx) * oz + 0.5 * sz
            try:
                xx_agg, yy_agg, zz_agg, v_agg = img.aggregateDataPointsWrtGrid(
                                                    xx, yy, zz, v,
                                                    nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                    op=aggregate_data_op, 
                                                    logger=logger,
                                                    **aggregate_data_op_kwargs)
            except Exception as exc:
                err_msg = f"{fname}: data aggregation (`aggregate_data_op='{aggregate_data_op}'`) failed"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

            if len(xx_agg) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

        if xIneqMin is not None or xIneqMax is not None:
            # Get single grid index for data points
            ix, iy, iz = img.pointToGridIndex(xx_agg, yy_agg, zz_agg,
                                              sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz)
            ig = img.gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz)

        if not aggregate_data_by_simul:
            dataPointSet.append(
                PointSet(npt=v_agg.shape[0], nv=4, val=np.array((xx_agg, yy_agg, zz_agg, v_agg)), varname=['X', 'Y', 'Z', varname])
                )
        else:
            # Integrate data points from sgs index 0
            dataPointSet.append(
                PointSet(npt=v_agg.shape[1], nv=4, val=np.array((xx_agg, yy_agg, zz_agg, v_agg[0])), varname=['X', 'Y', 'Z', varname])
                )

    # data point set from xIneqMin, vIneqMin
    if xIneqMin is not None:
        # Aggregate data on grid cell by using the given operation
        xx = xIneqMin[:, 0]
        yy = np.ones_like(xx) * oy + 0.5 * sy
        zz = np.ones_like(xx) * oz + 0.5 * sz
        try:
            xx_ineqMin_agg, yy_ineqMin_agg, zz_ineqMin_agg, v_ineqMin_agg = img.aggregateDataPointsWrtGrid(
                                                xx, yy, zz, vIneqMin,
                                                nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                op=aggregate_data_ineqMin_op,
                                                logger=logger,
                                                **aggregate_data_ineqMin_op_kwargs)
        except Exception as exc:
            err_msg = f"{fname}: inequality data (min) aggregation (`aggregate_data_op='{aggregate_data_ineqMin_op}'`) failed"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

        if x is not None:
            # Get single grid index for inequality (min) data points
            ix, iy, iz = img.pointToGridIndex(xx_ineqMin_agg, yy_ineqMin_agg, zz_ineqMin_agg,
                                              sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz)
            ig_ineqMin = img.gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz)

            # Get single grid index with points from `x` and `xIneqMin`
            ig_inter, ig1, ig2 = np.intersect1d(ig, ig_ineqMin, assume_unique=True, return_indices=True)

            if ig_inter.size:
                if verbose > 0:
                    if logger:
                        logger.warning(f'{fname}: {ig_inter.size} grid cell(s) have both "inequality (min)" and "equality" data: inequlity data has been removed')
                    else:
                        print(f'{fname}: WARNING: {ig_inter.size} grid cell(s) have both "inequality (min)" and "equality" data: inequlity data has been removed')
                    if not aggregate_data_by_simul:
                        ninconsistent = (v_agg[ig1] < v_ineqMin_agg[ig2]).sum()
                        if ninconsistent:
                            if logger:
                                logger.warning(f'{fname}: {ninconsistent} "inequality (min)" found')
                            else:
                                print(f'{fname}: WARNING: {ninconsistent} "inequality (min)" found')
                # Remove redundant points from inequality dataset
                xx_ineqMin_agg = np.delete(xx_ineqMin_agg, ig2)
                yy_ineqMin_agg = np.delete(yy_ineqMin_agg, ig2)
                zz_ineqMin_agg = np.delete(zz_ineqMin_agg, ig2)
                v_ineqMin_agg = np.delete(v_ineqMin_agg, ig2)

        if v_ineqMin_agg.shape[0]:
            dataPointSet.append(
                PointSet(npt=v_ineqMin_agg.shape[0], nv=4, val=np.array((xx_ineqMin_agg, yy_ineqMin_agg, zz_ineqMin_agg, v_ineqMin_agg)), varname=['X', 'Y', 'Z', f'{varname}_min'])
                )
        else:
            if verbose > 0:
                if logger:
                    logger.warning(f'{fname}: no inequality (min) data point in grid')
                else:
                    print(f'{fname}: WARNING: no inequality (min) data point in grid')
            xIneqMin = None

    # data point set from xIneqMax, vIneqMax
    if xIneqMax is not None:
        # Aggregate data on grid cell by using the given operation
        xx = xIneqMax[:, 0]
        yy = np.ones_like(xx) * oy + 0.5 * sy
        zz = np.ones_like(xx) * oz + 0.5 * sz
        try:
            xx_ineqMax_agg, yy_ineqMax_agg, zz_ineqMax_agg, v_ineqMax_agg = img.aggregateDataPointsWrtGrid(
                                                xx, yy, zz, vIneqMax,
                                                nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                op=aggregate_data_ineqMax_op, 
                                                logger=logger,
                                                **aggregate_data_ineqMax_op_kwargs)
        except Exception as exc:
            err_msg = f"{fname}: inequality data (max) aggregation (`aggregate_data_op='{aggregate_data_ineqMax_op}'`) failed"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

        if x is not None:
            # Get single grid index for inequality (max) data points
            ix, iy, iz = img.pointToGridIndex(xx_ineqMax_agg, yy_ineqMax_agg, zz_ineqMax_agg,
                                              sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz)
            ig_ineqMax = img.gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz)

            # Get single grid index with points from `x` and `xIneqMax`
            ig_inter, ig1, ig2 = np.intersect1d(ig, ig_ineqMax, assume_unique=True, return_indices=True)

            if ig_inter.size:
                if verbose > 0:
                    if logger:
                        logger.warning(f'{fname}: {ig_inter.size} grid cell(s) have both "inequality (max)" and "equality" data: inequlity data has been removed')
                    else:
                        print(f'{fname}: WARNING: {ig_inter.size} grid cell(s) have both "inequality (max)" and "equality" data: inequlity data has been removed')
                    if not aggregate_data_by_simul:
                        ninconsistent = (v_agg[ig1] > v_ineqMax_agg[ig2]).sum()
                        if ninconsistent:
                            if logger:
                                logger.warning(f'{fname}: {ninconsistent} "inequality (max)" found')
                            else:
                                print(f'{fname}: WARNING: {ninconsistent} "inequality (max)" found')
                # Remove redundant points from inequality dataset
                xx_ineqMax_agg = np.delete(xx_ineqMax_agg, ig2)
                yy_ineqMax_agg = np.delete(yy_ineqMax_agg, ig2)
                zz_ineqMax_agg = np.delete(zz_ineqMax_agg, ig2)
                v_ineqMax_agg = np.delete(v_ineqMax_agg, ig2)

        if xIneqMin is not None:
            # Get single grid index for inequality (min) data points
            ix, iy, iz = img.pointToGridIndex(xx_ineqMin_agg, yy_ineqMin_agg, zz_ineqMin_agg,
                                              sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz)
            ig_ineqMin = img.gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz)

            # Get single grid index for inequality (max) data points
            ix, iy, iz = img.pointToGridIndex(xx_ineqMax_agg, yy_ineqMax_agg, zz_ineqMax_agg,
                                              sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz)
            ig_ineqMax = img.gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz)

            # Get single grid index with points from `xIneqMin` and `xIneqMax`
            ig_inter, ig1, ig2 = np.intersect1d(ig_ineqMin, ig_ineqMax, assume_unique=True, return_indices=True)

            if ig_inter.size:
                ii = np.where(v_ineqMin_agg[ig1] > v_ineqMax_agg[ig2])[0]
                if len(ii):
                    err_msg = f'{fname}: {len(ii)} grid cell(s) have inconsistent "inequality min" and "inequality max" data'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)
                    # if verbose > 0:
                    #     if logger:
                    #         logger.warning(f'{fname}: {len(ii)} grid cell(s) have inconsistent "inequality min" and "inequality max" data: inequlity max data has been removed')
                    #     else:
                    #         print(f'{fname}: WARNING: {len(ii)} grid cell(s) have inconsistent "inequality min" and "inequality max" data: inequlity max data has been removed')
                    # ig2 = ig2[ii]
                    # # Remove inconsistent inequality max
                    # xx_ineqMax_agg = np.delete(xx_ineqMax_agg, ig2)
                    # yy_ineqMax_agg = np.delete(yy_ineqMax_agg, ig2)
                    # zz_ineqMax_agg = np.delete(zz_ineqMax_agg, ig2)
                    # v_ineqMax_agg = np.delete(v_ineqMax_agg, ig2)

        if v_ineqMax_agg.shape[0]:
            dataPointSet.append(
                PointSet(npt=v_ineqMax_agg.shape[0], nv=4, val=np.array((xx_ineqMax_agg, yy_ineqMax_agg, zz_ineqMax_agg, v_ineqMax_agg)), varname=['X', 'Y', 'Z', f'{varname}_max'])
                )
        else:
            if verbose > 0:
                if logger:
                    logger.warning(f'{fname}: no inequality (max) data point in grid')
                else:
                    print(f'{fname}: WARNING: no inequality (max) data point in grid')
            xIneqMax = None

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            err_msg = f'{fname}: `mask` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if mask is not None and add_data_point_to_mask:
        # Make a copy of the original mask, to remove value in added mask cell at the end
        mask_original = np.copy(mask)
        # Add cell to mask if needed
        pts = np.zeros((0,3))
        if x is not None:
            pts = np.vstack((pts, np.array((xx_agg, yy_agg, zz_agg)).T))
        if xIneqMin is not None:
            pts = np.vstack((pts, np.array((xx_ineqMin_agg, yy_ineqMin_agg, zz_ineqMin_agg)).T))
        if xIneqMax is not None:
            pts = np.vstack((pts, np.array((xx_ineqMax_agg, yy_ineqMax_agg, zz_ineqMax_agg)).T))
        if pts.shape[0]:
            im_tmp = img.imageFromPoints(pts,
                        nx=nx, ny=ny, nz=nz,
                        sx=sx, sy=sy, sz=sz,
                        ox=ox, oy=oy, oz=oz,
                        indicator_var=True,
                        logger=logger)
            mask = 1.0*np.any((im_tmp.val[0], mask), axis=0)
            del(im_tmp)
        del(pts)

    # Prepare seed (for simulation in grid)
    seed = seed + 986 # same increment (whatever the number or realization done in sgs above)

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose > 0 and nth > os.cpu_count():
        if logger:
            logger.warning(f'{fname}: number of threads used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: number of threads used will exceed number of cpu(s) of the system...')

    if not aggregate_data_by_simul:
        # --- Fill mpds_geosClassicInput structure (C)
        try:
            mpds_geosClassicInput = fill_mpds_geosClassicInput(
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
                    0.0,
                    searchRadiusRelative,
                    nneighborMax,
                    searchNeighborhoodSortMode,
                    nGibbsSamplerPathMin,
                    nGibbsSamplerPathMax,
                    seed,
                    nreal,
                    logger=logger)
            
        except Exception as exc:
            err_msg = f'{fname}: cannot fill mpds_geosClassicInput C structure'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

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

        # # --- Set number of threads
        # if nthreads <= 0:
        #     nth = max(os.cpu_count() + nthreads, 1)
        # else:
        #     nth = nthreads

        if verbose > 1:
            if logger:
                logger.info('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            else:
                print('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
                sys.stdout.flush()
                sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

        # --- Launch "GeosClassicSim" (launch C code)
        # err = geosclassic.MPDSGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
        err = geosclassic.MPDSOMPGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

        # Free memory on C side: mpds_geosClassicInput
        geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
        geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)

        if err:
            # Free memory on C side: mpds_geosClassicOutput
            geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
            geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)
            # Free memory on C side: mpds_progressMonitor
            geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)
            # Raise error
            err_message = geosclassic.mpds_get_error_message(-err)
            err_message = err_message.replace('\n', '')
            err_msg = f'{fname}: {err_message}'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor, logger=logger)

        # Free memory on C side: mpds_geosClassicOutput
        geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
        geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

        # Free memory on C side: mpds_progressMonitor
        geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    else:
        # Equality data values will change for each realization
        if verbose > 1:
            if logger:
                logger.info('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            else:
                print('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
                sys.stdout.flush()
                sys.stdout.flush() # twice!, so that the previous print is flushed before launching geos-classic...

        # Initialization of image and warnings for storing results
        image = Img(nx=nx, ny=ny, nz=nz, sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz, nv=nreal, val=np.nan, logger=logger)
        nwarning = 0
        warnings = []
        outputReportFile_ir = None # default
        for ir in range(nreal):
            if ir > 0:
                # Set equality data values for realization index ir
                dataPointSet[0].val[3] = v_agg[ir]

            if outputReportFile is not None:
                outputReportFile_ir = outputReportFile + f'.{ir}'

            # --- Fill mpds_geosClassicInput structure (C)
            try:
                mpds_geosClassicInput = fill_mpds_geosClassicInput(
                        space_dim,
                        cov_model,
                        nx, ny, nz,
                        sx, sy, sz,
                        ox, oy, oz,
                        varname,
                        outputReportFile_ir,
                        computationMode,
                        None,
                        dataPointSet,
                        mask,
                        mean,
                        var,
                        0.0,
                        searchRadiusRelative,
                        nneighborMax,
                        searchNeighborhoodSortMode,
                        nGibbsSamplerPathMin,
                        nGibbsSamplerPathMax,
                        seed+ir, # seed for realization index ir
                        1, # one real
                        logger=logger)
                
            except Exception as exc:
                err_msg = f'{fname}: cannot fill mpds_geosClassicInput C structure'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

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
            mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
            # if verbose < 3:
            #     mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
            # else:
            #     mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr
            #
            # if verbose > 1:
            #     if logger:
            #         logger.info('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            #     else:
            #         print('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            #         sys.stdout.flush()
            #         sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

            # --- Launch "GeosClassicSim" (launch C code)
            # err = geosclassic.MPDSGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
            err = geosclassic.MPDSOMPGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

            # Free memory on C side: mpds_geosClassicInput
            geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
            geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)

            if err:
                # Free memory on C side: mpds_geosClassicOutput
                geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
                geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)
                # Free memory on C side: mpds_progressMonitor
                geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)
                # Raise error
                err_message = geosclassic.mpds_get_error_message(-err)
                err_message = err_message.replace('\n', '')
                err_msg = f'{fname}: {err_message}'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor, logger=logger)

            # Free memory on C side: mpds_geosClassicOutput
            geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
            geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

            # Free memory on C side: mpds_progressMonitor
            geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

            image.val[ir] = geosclassic_output['image'].val[0]
            nwarning = nwarning + int(geosclassic_output['nwarning'])
            warnings.extend(geosclassic_output['warnings'])

            del(geosclassic_output)

        # Remove duplicated warnings
        warnings = list(np.unique(warnings))

        # Rename variables
        ndigit = geosclassic.MPDS_GEOS_CLASSIC_NB_DIGIT_FOR_REALIZATION_NUMBER
        for j in range(image.nv):
            image.varname[j] = image.varname[j][:-ndigit] + f'{j:0{ndigit}d}'

        # Set geosclassic_output
        geosclassic_output = {'image':image, 'nwarning':nwarning, 'warnings':warnings}

    if geosclassic_output is not None and mask is not None and add_data_point_to_mask:
        # Remove the value out of the original mask (using its copy see above)
        geosclassic_output['image'].val[:, mask_original==0.0] = np.nan

    if verbose > 1 and geosclassic_output:
        if logger:
            logger.info(f'{fname}: Geos-Classic run complete')
        else:
            print(f'{fname}: Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose > 0 and geosclassic_output and geosclassic_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({geosclassic_output['nwarning']} times in all):")
            for i, warning_message in enumerate(geosclassic_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

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
        aggregate_data_op=None,
        aggregate_data_op_kwargs=None,
        aggregate_data_ineqMin_op='max',
        aggregate_data_ineqMin_op_kwargs=None,
        aggregate_data_ineqMax_op='min',
        aggregate_data_ineqMax_op_kwargs=None,
        mask=None,
        add_data_point_to_mask=True,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        nGibbsSamplerPathMin=50,
        nGibbsSamplerPathMax=200,
        seed=None,
        outputReportFile=None,
        treat_image_one_by_one=False,
        nproc=None, nthreads_per_proc=None,
        verbose=2,
        logger=None):
    """
    Computes the same as the function :func:`simulate1D`, using multiprocessing.

    All the parameters are the same as those of the function :func:`simulate1D`,
    except `nthreads` that is replaced by the parameters `nproc` and
    `nthreads_per_proc`, and an extra parameter `treat_image_one_by_one`.

    This function launches multiple processes (based on `multiprocessing`
    package):

    - `nproc` parallel processes using each one `nthreads_per_proc` threads \
    are launched [parallel calls of the function :func:`simulate1D`]
    - the set of realizations (specified by `nreal`) is distributed in a \
    balanced way over the processes
    - in terms of resources, this implies the use of `nproc*nthreads_per_proc` \
    cpu(s)

    See function :func:`simulate1D`.

    **Parameters (new)**
    --------------------
    nproc : int, optional
        number of process(es); by default (`None`):
        `nproc` is set to `min(nmax-1, nreal)` (but at least 1), where nmax is
        the total number of cpu(s) of the system (retrieved by
        `multiprocessing.cpu_count()`)

    nthreads_per_proc : int, optional
        number of thread(s) per process (should be > 0); by default (`None`):
        `nthreads_per_proc` is automatically computed as the maximal integer
        (but at least 1) such that `nproc*nthreads_per_proc <= nmax-1`, where
        nmax is the total number of cpu(s) of the system (retrieved by
        `multiprocessing.cpu_count()`)

    treat_image_one_by_one : bool, default: False
        keyword argument passed to the function :func:`geone.img.gatherImages`:

        - if `True`: images (result of each process) are gathered one by one, \
        i.e. the variables of each image are inserted in an output image one by \
        one and removed from the source (slower, may save memory)
        - if `False`: images (result of each process) are gathered at once, \
        i.e. the variables of all images are inserted in an output image at once, \
        and then removed (faster)
    """
    fname = 'simulate1D_mp'

    if logger:
        logger.warning(f'{fname}: deprecated function, use function `simulate` instead')
    else:
        print(f'{fname}: WARNING: deprecated function, use function `simulate` instead')

    # Set number of process(es): nproc
    if nproc is None:
        nproc = max(min(multiprocessing.cpu_count()-1, nreal), 1)
    else:
        nproc_tmp = nproc
        nproc = max(min(int(nproc), nreal), 1)
        if verbose > 1 and nproc != nproc_tmp:
            if logger:
                logger.info(f'{fname}: number of process(es) has been changed (now: nproc={nproc})')
            else:
                print(f'{fname}: number of process(es) has been changed (now: nproc={nproc})')

    # Set number of threads per process: nth
    if nthreads_per_proc is None:
        nth = max(int(np.floor((multiprocessing.cpu_count()-1) / nproc)), 1)
    else:
        nth = max(int(nthreads_per_proc), 1)
        if verbose > 1 and nth != nthreads_per_proc:
            if logger:
                logger.info(f'{fname}: number of threads per process has been changed (now: nthreads_per_proc={nth})')
            else:
                print(f'{fname}: number of threads per process has been changed (now: nthreads_per_proc={nth})')

    if verbose > 0 and nproc * nth > multiprocessing.cpu_count():
        if logger:
            logger.warning(f'{fname}: total number of cpu(s) used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: total number of cpu(s) used will exceed number of cpu(s) of the system...')

    # Set the distribution of the realizations over the processes
    # Condider the Euclidean division of nreal by nproc:
    #     nreal = q * nproc + r, with 0 <= r < nproc
    # Then, (q+1) realizations will be done on process 0, 1, ..., r-1, and q realization on process r, ..., nproc-1
    # Define the list real_index_proc of length (nproc+1) such that
    #   real_index_proc[i], ..., real_index_proc[i+1] - 1 : are the realization indices run on process i
    q, r = np.divmod(nreal, nproc)
    real_index_proc = [i*q + min(i, r) for i in range(nproc+1)]

    if verbose > 1:
        if logger:
            logger.info('{}: Geos-Classic running on {} process(es)... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, nproc, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        else:
            print('{}: Geos-Classic running on {} process(es)... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, nproc, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            sys.stdout.flush()
            sys.stdout.flush() # twice!, so that the previous print is flushed before launching geos-classic...

    # Prepare seed
    if seed is None:
        seed = np.random.randint(1, 1000000)
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
        verbose_p = 0
        # if i==0:
        #     verbose_p = min(verbose, 1) # allow to print warnings for process i
        # else:
        #     verbose_p = 0
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
                aggregate_data_op,
                aggregate_data_op_kwargs,
                aggregate_data_ineqMin_op,
                aggregate_data_ineqMin_op_kwargs,
                aggregate_data_ineqMax_op,
                aggregate_data_ineqMax_op_kwargs,
                mask,
                add_data_point_to_mask,
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                nGibbsSamplerPathMin,
                nGibbsSamplerPathMax,
                seed_p,                      # seed (adjusted)
                outputReportFile_p,          # outputReportFile (adjusted)
                nth,                         # nthreads
                verbose_p),                  # verbose (adjusted)
                kwds={'logger':logger})
            )

    # Properly end working process
    pool.close() # Prevents any more tasks from being submitted to the pool,
    pool.join()  # then, wait for the worker processes to exit.

    # Get result from each process
    geosclassic_output_proc = [p.get() for p in out_pool]

    if np.any([out is None for out in geosclassic_output_proc]):
        return None

    # Gather results from every process
    # image
    image = []
    for out in geosclassic_output_proc:
        if out['image'] is not None:
            image.append(out['image'])
            del(out['image'])
    if len(image) == 0:
        image = None
    # Gather images and adjust variable names
    all_image = img.gatherImages(image, keep_varname=True, rem_var_from_source=True, treat_image_one_by_one=treat_image_one_by_one, logger=logger)
    ndigit = geosclassic.MPDS_GEOS_CLASSIC_NB_DIGIT_FOR_REALIZATION_NUMBER
    for j in range(all_image.nv):
        all_image.varname[j] = all_image.varname[j][:-ndigit] + f'{j:0{ndigit}d}'

    # nwarning
    nwarning = np.sum([out['nwarning'] for out in geosclassic_output_proc])
    # warnings
    warnings = list(np.unique(np.hstack([out['warnings'] for out in geosclassic_output_proc])))

    geosclassic_output = {'image':all_image, 'nwarning':nwarning, 'warnings':warnings}

    if verbose > 1 and geosclassic_output:
        if logger:
            logger.info(f'{fname}: Geos-Classic run complete (all process(es))')
        else:
            print(f'{fname}: Geos-Classic run complete (all process(es))')

    # Show (print) encountered warnings
    if verbose > 0 and geosclassic_output and geosclassic_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({geosclassic_output['nwarning']} times in all):")
            for i, warning_message in enumerate(geosclassic_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

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
        aggregate_data_op=None,
        aggregate_data_op_kwargs=None,
        aggregate_data_ineqMin_op='max',
        aggregate_data_ineqMin_op_kwargs=None,
        aggregate_data_ineqMax_op='min',
        aggregate_data_ineqMax_op_kwargs=None,
        mask=None,
        add_data_point_to_mask=True,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        nGibbsSamplerPathMin=50,
        nGibbsSamplerPathMax=200,
        seed=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2,
        logger=None):
    """
    Generates 2D simulations (Sequential Gaussian Simulation, SGS).

    A simulation takes place in (center of) grid cells, based on simple or
    ordinary kriging.

    Parameters
    ----------
    cov_model : :class:`geone.covModel.CovModel2D`
        covariance model in 2D

    dimension : 2-tuple of ints
        `dimension=(nx, ny)`, number of cells in the 2D simulation grid along
        each axis

    spacing : 2-tuple of floats, default: (1.0, 1.0)
        `spacing=(sx, sy)`, cell size along each axis

    origin : 2-tuple of floats, default: (0.0, 0.0)
        `origin=(ox, oy)`, origin of the 2D simulation grid (lower-left corner)

    method : str {'simple_kriging', 'ordinary_kriging'}, default: 'simple_kriging'
        type of kriging

    nreal : int, default: 1
        number of realizations

    mean : function (callable), or array-like of floats, or float, optional
        kriging mean value:

        - if a function: function of two arguments (xi, yi) that returns the mean \
        at location (xi, yi)
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), mean values at grid cells (for \
        non-stationary mean)
        - if a float: same mean value at every grid cell
        - by default (`None`): the mean of data value (`v`) (0.0 if no data) is \
        considered at every grid cell

    var : function (callable), or array-like of floats, or float, optional
        kriging variance value:

        - if a function: function of two arguments (xi, yi) that returns the \
        variance at location (xi, yi)
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), variance values at grid cells (for \
        non-stationary variance)
        - if a float: same variance value at every grid cell
        - by default (`None`): not used (use of covariance model only)

    x : 2D array of floats of shape (n, 2), optional
        data points locations, with n the number of data points, each row of `x`
        is the float coordinates of one data point; note: if n=1, a 1D array of
        shape (2,) is accepted

    v : 1D array of floats of shape (n,), optional
        data values at `x` (`v[i]` is the data value at `x[i]`)

    xIneqMin : 2D array of floats of shape (nIneqMin, 2), optional
        data points locations, for inequality data with lower bound, with
        nIneqMin the number of data points, each row of `xIneqMin` is the float
        coordinates of one data point; note: if nIneqMin=1, a 1D array of
        shape (2,) is accepted

    vIneqMin : 1D array of floats of shape (nIneqMin,), optional
        inequality data values, lower bounds, at `xIneqMin` (`vIneqMin[i]` is the
        data value at `xIneqMin[i]`)

    xIneqMax : 2D array of floats of shape (nIneqMax, 2), optional
        data points locations, for inequality data with upper bound, with
        nIneqMax the number of data points, each row of `xIneqMax` is the float
        coordinates of one data point; note: if nIneqMax=1, a 1D array of
        shape (2,) is accepted

    vIneqMax : 1D array of floats of shape (nIneqMax,), optional
        inequality data values, upper bounds, at `xIneqMax` (`vIneqMax[i]` is the
        data value at `xIneqMax[i]`)

    aggregate_data_op : str {'sgs', 'krige', 'min', 'max', 'mean', 'quantile', \
                        'most_freq', 'random'}, optional
        operation used to aggregate data points falling in the same grid cells

        - if `aggregate_data_op='sgs'`: function :func:`geone.covModel.sgs` is used \
        with the covariance model `cov_model` given in arguments, as well as \
        the parameter `nneighborMax` given in arguments unless it is given \
        in `aggregate_data_op_kwargs`
        - if `aggregate_data_op='krige'`: function :func:`geone.covModel.krige` is used \
        with the covariance model `cov_model` given in arguments, as well as \
        the parameters `use_unique_neighborhood`, `nneighborMax` given in \
        arguments unless they are given in `aggregate_data_op_kwargs`
        - if `aggregate_data_op='most_freq'`: most frequent value is selected \
        (smallest one if more than one value with the maximal frequence)
        - if `aggregate_data_op='random'`: value from a random point is selected \
        - otherwise: the function `numpy.<aggregate_data_op>` is used with the \
        additional parameters given by `aggregate_data_op_kwargs`, note that, e.g. \
        `aggregate_data_op='quantile'` requires the additional parameter \
        `q=<quantile_to_compute>`

        Note: if `aggregate_data_op='sgs'` or `aggregate_data_op='random'`, the
        aggregation is done for each realization (simulation), i.e. each simulation
        on the grid starts with a new set of values in conditioning grid cells

        By default: if covariance model has stationary ranges and weight (sill),
        `aggregate_data_op='sgs'` is used, otherwise `aggregate_data_op='mean'`

    aggregate_data_op_kwargs : dict, optional
        keyword arguments to be passed to `geone.covModel.sgs`,
        `geone.covModel.krige`, or `numpy.<aggregate_data_op>`, according to
        the parameter `aggregate_data_op`

    aggregate_data_ineqMin_op : str {'min', 'max', 'mean', 'quantile', \
                        'most_freq', 'random'}, default: 'max'
        operation used to aggregate inequality (min, lower boudns) data points
        falling in the same grid cells:

        - if `aggregate_data_ineqMin_op='most_freq'`: most frequent value is \
        selected (smallest one if more than one value with the maximal frequence)
        - if `aggregate_data_ineqMin_op='random'`: value from a random point is \
        selected
        - otherwise: the function `numpy.<aggregate_data_ineqMin_op>` is used with \
        the additional parameters given by `aggregate_data_ineqMin_op_kwargs`, \
        note that, e.g. `aggregate_data_ineqMin_op='quantile'` requires the \
        additional parameter `q=<quantile_to_compute>`

        Note: in any case, the aggregation is done once, i.e. same inequality
        values are used for each simulation on the grid

    aggregate_data_ineqMin_op_kwargs : dict, optional
        keyword arguments to be passed to `numpy.<aggregate_data_ineqMin_op>`,
        according to the parameter `aggregate_data_ineqMin_op`

    aggregate_data_ineqMax_op : str {'min', 'max', 'mean', 'quantile', \
                        'most_freq', 'random'}, default: 'min'
        operation used to aggregate inequality (min, lower boudns) data points
        falling in the same grid cells:

        - if `aggregate_data_ineqMax_op='most_freq'`: most frequent value is \
        selected (smallest one if more than one value with the maximal frequence)
        - if `aggregate_data_ineqMax_op='random'`: value from a random point is \
        selected
        - otherwise: the function `numpy.<aggregate_data_ineqMax_op>` is used with \
        the additional parameters given by `aggregate_data_ineqMax_op_kwargs`, \
        note that, e.g. `aggregate_data_ineqMax_op='quantile'` requires the \
        additional parameter `q=<quantile_to_compute>`

        Note: in any case, the aggregation is done once, i.e. same inequality
        values are used for each simulation on the grid

    aggregate_data_ineqMax_op_kwargs : dict, optional
        keyword arguments to be passed to `numpy.<aggregate_data_ineqMax_op>`,
        according to the parameter `aggregate_data_ineqMax_op`

    mask : array-like, optional
        mask value at grid cells (value 1 for simulated cells, value 0 for not
        simulated cells); the size of the array must be equal to the number of
        grid cells (the array is reshaped if needed)

    add_data_point_to_mask : bool, default: True
        - if `True`: any grid cell that contains a data point is added to (the \
        simulated part of) the mask (if present), i.e. mask value at those cells \
        are set to 1; at the end of the computation the "new mask cells" are \
        removed (by setting a missing value (`numpy.nan`) for the variable out of \
        the original mask)
        - if `False`: original mask is kept as given in input, and data point \
        falling out of (the simulated part of) the mask (if present) are ignored

    searchRadiusRelative : float, default: 1.2
        indicates how the search ellipsoid is limited (should be positive): let
        r_i be the ranges of the covariance model along its main axes, when
        estimating/simulating a cell x, a cell y is taken into account iff it is
        within the ellipsoid centered at x of half axes equal to
        `searchRadiusRelative` * r_i; 
        (note that the distances to the central node are computed in the axes 
        sytem supporting the covariance model and accounting for anisotropy given 
        by the ranges)
        note (Geos-Classic library): if a range r_i is non-stationary over the grid, 
        its maximal value over the grid is considered;

    nneighborMax : int, default: 12
        maximum number of cells retrieved from the search ellipsoid (when
        estimating/simulating a cell), `nneighborMax=-1` for unlimited

    searchNeighborhoodSortMode : int, optional
        indicates how to sort the search neighboorhood cells (neighbors); they
        are sorted in increasing order according to:

        - `searchNeighborhoodSortMode=0`: distance in the usual axes system
        - `searchNeighborhoodSortMode=1`: distance in the axes sytem supporting \
        the covariance model and accounting for anisotropy given by the ranges
        - `searchNeighborhoodSortMode=2`: minus the evaluation of the covariance \
        model

        Notes:

        - if the covariance model has any non-stationary parameter, then \
        `searchNeighborhoodSortMode=2` is not allowed
        - if the covariance model has any non-stationary range or non-stationary \
        angle and `searchNeighborhoodSortMode=1`: "maximal ranges" (adapted to \
        direction from the central cell) are used to compute distance for sorting \
        the neighbors

        By default (`None`): the greatest possible value is used (i.e. 2 for
        stationary covariance model, or 1 otherwise)

    nGibbsSamplerPathMin: int, default: 50
        see `nGibbsSamplerPathMax`

    nGibbsSamplerPathMax: int, default: 200
        `nGibbsSamplerPathMin` and `nGibbsSamplerPathMax` are the mini and max number
        of Gibbs sampler paths to deal with inequality data; the conditioning locations
        with inequality data are first simulated (based on truncated gaussian
        distribution) sequentially; then, these locations are re-simulated following a
        new path as many times as needed, but the total number of paths will be between
        `nGibbsSamplerPathMin` and `nGibbsSamplerPathMax`

    seed : int, optional
        seed for initializing random number generator

    outputReportFile : str, optional
        name of the report file (if desired in output); by default (`None`): no
        report file

    nthreads : int, default: -1
        number of thread(s) to use for "GeosClassicSim" C program;
        `nthreads = -n <= 0`: maximal number of threads of the system except n
        (but at least 1)

    verbose : int, default: 2
        verbose mode, higher implies more printing (info):

        - 0: no display
        - 1: warnings
        - 2: warnings + basic info
        - 3 (or >2): all information

        note that if an error occurred, it is raised
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    geosclassic_output : dict
        geosclassic output in python, dictionary

        {'image':image, 'nwarning':nwarning, 'warnings':warnings}

        with:

        - image : :class:`geone.img.Img`
            output image, with `image.nv=nreal` variables (simulations);
            note: `image=None` if `mpds_geosClassicOutput->outputImage=NULL`

        - nwarning : int
            total number of warning(s) encountered (same warnings can be counted
            several times)

        - warnings : list of strs
            list of distinct warnings encountered (can be empty)
    """
    fname = 'simulate2D'

    if logger:
        logger.warning(f'{fname}: deprecated function, use function `simulate` instead')
    else:
        print(f'{fname}: WARNING: deprecated function, use function `simulate` instead')

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
    # nreal
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose > 0:
            if logger:
                logger.warning(f'{fname}: `nreal` <= 0: `None` is returned')
            else:
                print(f'{fname}: WARNING: `nreal` <= 0: `None` is returned')
        return None

    # cov_model
    if isinstance(cov_model, gcm.CovModel1D):
        cov_model = gcm.covModel1D_to_covModel2D(cov_model) # convert model 1D in 2D
            # -> will not be modified cov_model at exit

    if not isinstance(cov_model, gcm.CovModel2D):
        err_msg = f'{fname}: `cov_model` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    for el in cov_model.elem:
        # weight
        w = el[1]['w']
        if np.size(w) != 1 and np.size(w) != nxyz:
            err_msg = f"{fname}: `cov_model`: weight ('w') not compatible with simulation grid"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # ranges
        if 'r' in el[1].keys():
            for r in el[1]['r']:
                if np.size(r) != 1 and np.size(r) != nxyz:
                    err_msg = f"{fname}: `cov_model`: range ('r') not compatible with simulation grid"
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

        # additional parameter (s)
        if 's' in el[1].keys():
            s  = el[1]['s']
            if np.size(s) != 1 and np.size(s) != nxyz:
                err_msg = f"{fname}: `cov_model`: parameter ('s') not compatible with simulation grid"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # alpha
    angle = cov_model.alpha
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        err_msg = f"{fname}: `cov_model`: angle ('alpha') not compatible with simulation grid"
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # aggregate_data_op (default)
    if aggregate_data_op is None:
        if not cov_model.is_weight_stationary() or not cov_model.is_range_stationary():
            aggregate_data_op = 'mean'
        else:
            aggregate_data_op = 'sgs'

    if aggregate_data_op_kwargs is None:
        aggregate_data_op_kwargs = {}

    if aggregate_data_ineqMin_op_kwargs is None:
        aggregate_data_ineqMin_op_kwargs = {}

    if aggregate_data_ineqMax_op_kwargs is None:
        aggregate_data_ineqMax_op_kwargs = {}

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    if method == 'simple_kriging':
        computationMode = 3
    elif method == 'ordinary_kriging':
        computationMode = 2
    else:
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - searchRadiusRelative
    if searchRadiusRelative < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
        err_msg = f'{fname}: `searchRadiusRelative` too small (should be at least {geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN})'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - nneighborMax
    if nneighborMax != -1 and nneighborMax <= 0:
        err_msg = f'{fname}: `nneighborMax` should be greater than 0 or equal to -1 (unlimited)'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - searchNeighborhoodSortMode
    if searchNeighborhoodSortMode is None:
        # set greatest possible value
        if cov_model.is_stationary():
            searchNeighborhoodSortMode = 2
        else:
            searchNeighborhoodSortMode = 1
    else:
        if searchNeighborhoodSortMode == 2:
            if not cov_model.is_stationary():
                err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # if searchNeighborhoodSortMode is None:
    #     # set greatest possible value
    #     if cov_model.is_stationary():
    #         searchNeighborhoodSortMode = 2
    #     elif cov_model.is_orientation_stationary() and cov_model.is_range_stationary():
    #         searchNeighborhoodSortMode = 1
    #     else:
    #         searchNeighborhoodSortMode = 0
    # else:
    #     if searchNeighborhoodSortMode == 2:
    #         if not cov_model.is_stationary():
    #             err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
    #             if logger: logger.error(err_msg)
    #             raise GeosclassicinterfaceError(err_msg)
    #     elif searchNeighborhoodSortMode == 1:
    #         if not cov_model.is_orientation_stationary() or not cov_model.is_range_stationary():
    #             err_msg = f'{fname}: `searchNeighborhoodSortMode=1` not allowed with non-stationary range or non-stationary orientation in covariance model'
    #             if logger: logger.error(err_msg)
    #             raise GeosclassicinterfaceError(err_msg)

    # Preparation of data points
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 2) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if xIneqMin is not None:
        xIneqMin = np.asarray(xIneqMin, dtype='float').reshape(-1, 2) # cast in 2-dimensional array if needed
        vIneqMin = np.asarray(vIneqMin, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(vIneqMin) != xIneqMin.shape[0]:
            err_msg = f'{fname}: length of `vIneqMin` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if xIneqMax is not None:
        xIneqMax = np.asarray(xIneqMax, dtype='float').reshape(-1, 2) # cast in 2-dimensional array if needed
        vIneqMax = np.asarray(vIneqMax, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(vIneqMax) != xIneqMax.shape[0]:
            err_msg = f'{fname}: length of `vIneqMax` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Check parameters - mean
    mean_x = mean
    if mean is not None:
        # if method == 'ordinary_kriging':
        #     err_msg = f'{fname}: specifying `mean` not allowed with ordinary kriging'
        #     if logger: logger.error(err_msg)
        #     raise GeosclassicinterfaceError(err_msg)

        if callable(mean):
            if x is not None:
                mean_x = mean(x[:, 0], x[:, 1])
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            yyi, xxi = np.meshgrid(yi, xi, indexing='ij')
            mean = mean(xxi, yyi) # replace function 'mean' by its evaluation on the grid
        else:
            mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if mean.size == 1:
                if x is not None:
                    mean_x = mean
            elif mean.size == nxyz:
                mean = mean.reshape(ny, nx)
                if x is not None:
                    mean_x = img.Img_interp_func(img.Img(nx, ny, 1, sx, sy, 1., ox, oy, 0., nv=1, val=mean, logger=logger), iz=0, logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `mean` is not valid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # Check parameters - var
    var_x = var
    if var is not None:
        if method == 'ordinary_kriging':
            err_msg = f'{fname}: specifying `var` not allowed with ordinary kriging'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        if callable(var):
            if x is not None:
                var_x = var(x[:, 0], x[:, 1])
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            yyi, xxi = np.meshgrid(yi, xi, indexing='ij')
            var = var(xxi, yyi) # replace function 'var' by its evaluation on the grid
        else:
            var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if var.size == 1:
                if x is not None:
                    var_x = var
            elif var.size == nxyz:
                var = var.reshape(ny, nx)
                if x is not None:
                    var_x = img.Img_interp_func(img.Img(nx, ny, 1, sx, sy, 1., ox, oy, 0., nv=1, val=var, logger=logger), iz=0, logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `var` is not valid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # Prepare seed
    if seed is None:
        seed = np.random.randint(1, 1000000)
    seed = int(seed)

    # data points: x, v, xIneqMin, vIneqMin, xIneqMax, vIneqMax
    dataPointSet = []

    # data point set from x, v
    aggregate_data_by_simul = False
    if x is not None:
        if aggregate_data_op == 'krige' or aggregate_data_op == 'sgs':
            if not cov_model.is_weight_stationary() or not cov_model.is_range_stationary():
                err_msg = f"{fname}: covariance model with non-stationary weight or range cannot be used with `aggregate_data_op`='{aggregate_data_op}'"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            if cov_model.is_orientation_stationary():
                cov_model_agg = cov_model
            else:
                cov_model_agg = gcm.copyCovModel(cov_model)
                cov_model_agg.set_alpha(0.0)
            # Get grid cell with at least one data point:
            # x_agg: 2D array, each row contains the coordinates of the center of such cell
            im_tmp = img.imageFromPoints(x, values=None, varname=None,
                                         nx=nx, ny=ny, sx=sx, sy=sy, ox=ox, oy=oy,
                                         indicator_var=True, 
                                         count_var=False,
                                         logger=logger)
            ind_agg = np.where(im_tmp.val[0])
            if len(ind_agg[0]) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            x_agg = np.array((im_tmp.xx()[ind_agg].reshape(-1), im_tmp.yy()[ind_agg].reshape(-1))).T
            # x_agg = np.array((im_tmp.xx()[*ind_agg].reshape(-1), im_tmp.yy()[*ind_agg].reshape(-1))).T
            ind_agg = ind_agg[1:] # remove index along z axis
            del(im_tmp)
            # Compute
            # - kriging estimate (v_agg) and kriging std (v_agg_std) at x_agg,
            # - or nreal simulation(s) (v_agg) at x_agg
            if mean is not None and mean.size > 1:
                mean_x_agg = mean[ind_agg]
                # mean_x_agg = mean[*ind_agg]
            else:
                mean_x_agg = mean
            if var is not None and var.size > 1:
                var_x_agg = var[ind_agg]
                # var_x_agg = var[*ind_agg]
            else:
                var_x_agg = var
            if isinstance(cov_model.alpha, np.ndarray) and cov_model.alpha.size == nxyz:
                alpha_x_agg = cov_model.alpha.reshape(ny, nx)[ind_agg]
                # alpha_x_agg = cov_model.alpha.reshape(ny, nx)[*ind_agg]
            else:
                alpha_x_agg = cov_model.alpha
            # Set parameters `nneighborMax` from the arguments if not given in `aggregate_data_op_kwargs`
            if 'nneighborMax' not in aggregate_data_op_kwargs.keys():
                aggregate_data_op_kwargs['nneighborMax'] = nneighborMax
            if aggregate_data_op == 'krige':
                try:
                    v_agg, v_agg_std = gcm.krige(x, v, x_agg, cov_model_agg, method=method,
                                                 mean_x=mean_x, mean_xu=mean_x_agg,
                                                 var_x=var_x, var_xu=var_x_agg,
                                                 alpha_xu=alpha_x_agg,
                                                 verbose=0, logger=logger,
                                                 **aggregate_data_op_kwargs)
                except Exception as exc:
                    err_msg = f'{fname}: kriging error'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg) from exc

            else:
                aggregate_data_by_simul = True
                try:
                    v_agg = gcm.sgs(x, v, x_agg, cov_model_agg, method=method,
                                    mean_x=mean_x, mean_xu=mean_x_agg,
                                    var_x=var_x, var_xu=var_x_agg,
                                    alpha_xu=alpha_x_agg,
                                    nreal=nreal, seed=seed,
                                    verbose=0, logger=logger,
                                    **aggregate_data_op_kwargs)
                except Exception as exc:
                    err_msg = f'{fname}: sgs error'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg) from exc

            xx_agg, yy_agg = x_agg.T
            zz_agg = np.ones_like(xx_agg) * oz + 0.5 * sz
        elif aggregate_data_op == 'random':
            aggregate_data_by_simul = True
            # Aggregate data on grid cell by taking random point
            xx, yy = x.T
            zz = np.ones_like(xx) * oz + 0.5 * sz
            # first realization of v_agg
            try:
                xx_agg, yy_agg, zz_agg, v_agg, i_inv = img.aggregateDataPointsWrtGrid(
                                                            xx, yy, zz, v,
                                                            nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                            op=aggregate_data_op, 
                                                            return_inverse=True,
                                                            logger=logger,
                                                            **aggregate_data_op_kwargs)
            except Exception as exc:
                err_msg = f'{fname}: data aggregation'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

            if len(xx_agg) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            # next realizations of v_agg
            v_agg = np.vstack((v_agg, np.zeros((nreal-1, v_agg.size))))
            for i in range(1, nreal):
                v_agg[i] = [v[np.random.choice(np.where(i_inv==j)[0])] for j in range(len(xx_agg))]
        else:
            # Aggregate data on grid cell by using the given operation
            xx, yy = x.T
            zz = np.ones_like(xx) * oz + 0.5 * sz
            try:
                xx_agg, yy_agg, zz_agg, v_agg = img.aggregateDataPointsWrtGrid(
                                                    xx, yy, zz, v,
                                                    nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                    op=aggregate_data_op, 
                                                    logger=logger,
                                                    **aggregate_data_op_kwargs)
            except Exception as exc:
                err_msg = f"{fname}: data aggregation (`aggregate_data_op='{aggregate_data_op}'`) failed"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

            if len(xx_agg) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

        if xIneqMin is not None or xIneqMax is not None:
            # Get single grid index for data points
            ix, iy, iz = img.pointToGridIndex(xx_agg, yy_agg, zz_agg,
                                              sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz)
            ig = img.gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz)

        if not aggregate_data_by_simul:
            dataPointSet.append(
                PointSet(npt=v_agg.shape[0], nv=4, val=np.array((xx_agg, yy_agg, zz_agg, v_agg)), varname=['X', 'Y', 'Z', varname])
                )
        else:
            # Integrate data points from sgs index 0
            dataPointSet.append(
                PointSet(npt=v_agg.shape[1], nv=4, val=np.array((xx_agg, yy_agg, zz_agg, v_agg[0])), varname=['X', 'Y', 'Z', varname])
                )

    # data point set from xIneqMin, vIneqMin
    if xIneqMin is not None:
        # Aggregate data on grid cell by using the given operation
        xx, yy = xIneqMin.T
        zz = np.ones_like(xx) * oz + 0.5 * sz
        try:
            xx_ineqMin_agg, yy_ineqMin_agg, zz_ineqMin_agg, v_ineqMin_agg = img.aggregateDataPointsWrtGrid(
                                                xx, yy, zz, vIneqMin,
                                                nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                op=aggregate_data_ineqMin_op, 
                                                logger=logger,
                                                **aggregate_data_ineqMin_op_kwargs)
        except Exception as exc:
            err_msg = f"{fname}: inequality data (min) aggregation (`aggregate_data_op='{aggregate_data_ineqMin_op}'`) failed"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

        if x is not None:
            # Get single grid index for inequality (min) data points
            ix, iy, iz = img.pointToGridIndex(xx_ineqMin_agg, yy_ineqMin_agg, zz_ineqMin_agg,
                                              sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz)
            ig_ineqMin = img.gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz)

            # Get single grid index with points from `x` and `xIneqMin`
            ig_inter, ig1, ig2 = np.intersect1d(ig, ig_ineqMin, assume_unique=True, return_indices=True)

            if ig_inter.size:
                if verbose > 0:
                    if logger:
                        logger.warning(f'{fname}: {ig_inter.size} grid cell(s) have both "inequality (min)" and "equality" data: inequlity data has been removed')
                    else:
                        print(f'{fname}: WARNING: {ig_inter.size} grid cell(s) have both "inequality (min)" and "equality" data: inequlity data has been removed')
                    if not aggregate_data_by_simul:
                        ninconsistent = (v_agg[ig1] < v_ineqMin_agg[ig2]).sum()
                        if ninconsistent:
                            if logger:
                                logger.warning(f'{fname}: {ninconsistent} "inequality (min)" found')
                            else:
                                print(f'{fname}: WARNING: {ninconsistent} "inequality (min)" found')
                # Remove redundant points from inequality dataset
                xx_ineqMin_agg = np.delete(xx_ineqMin_agg, ig2)
                yy_ineqMin_agg = np.delete(yy_ineqMin_agg, ig2)
                zz_ineqMin_agg = np.delete(zz_ineqMin_agg, ig2)
                v_ineqMin_agg = np.delete(v_ineqMin_agg, ig2)

        if v_ineqMin_agg.shape[0]:
            dataPointSet.append(
                PointSet(npt=v_ineqMin_agg.shape[0], nv=4, val=np.array((xx_ineqMin_agg, yy_ineqMin_agg, zz_ineqMin_agg, v_ineqMin_agg)), varname=['X', 'Y', 'Z', f'{varname}_min'])
                )
        else:
            if verbose > 0:
                if logger:
                    logger.warning(f'{fname}: no inequality (min) data point in grid')
                else:
                    print(f'{fname}: WARNING: no inequality (min) data point in grid')
            xIneqMin = None

    # data point set from xIneqMax, vIneqMax
    if xIneqMax is not None:
        # Aggregate data on grid cell by using the given operation
        xx, yy = xIneqMax.T
        zz = np.ones_like(xx) * oz + 0.5 * sz
        try:
            xx_ineqMax_agg, yy_ineqMax_agg, zz_ineqMax_agg, v_ineqMax_agg = img.aggregateDataPointsWrtGrid(
                                                xx, yy, zz, vIneqMax,
                                                nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                op=aggregate_data_ineqMax_op, 
                                                logger=logger,
                                                **aggregate_data_ineqMax_op_kwargs)
        except Exception as exc:
            err_msg = f"{fname}: inequality data (max) aggregation (`aggregate_data_op='{aggregate_data_ineqMax_op}'`) failed"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

        if x is not None:
            # Get single grid index for inequality (max) data points
            ix, iy, iz = img.pointToGridIndex(xx_ineqMax_agg, yy_ineqMax_agg, zz_ineqMax_agg,
                                              sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz)
            ig_ineqMax = img.gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz)

            # Get single grid index with points from `x` and `xIneqMax`
            ig_inter, ig1, ig2 = np.intersect1d(ig, ig_ineqMax, assume_unique=True, return_indices=True)

            if ig_inter.size:
                if verbose > 0:
                    if logger:
                        logger.warning(f'{fname}: {ig_inter.size} grid cell(s) have both "inequality (max)" and "equality" data: inequlity data has been removed')
                    else:
                        print(f'{fname}: WARNING: {ig_inter.size} grid cell(s) have both "inequality (max)" and "equality" data: inequlity data has been removed')
                    if not aggregate_data_by_simul:
                        ninconsistent = (v_agg[ig1] > v_ineqMax_agg[ig2]).sum()
                        if ninconsistent:
                            if logger:
                                logger.warning(f'{fname}: {ninconsistent} "inequality (max)" found')
                            else:
                                print(f'{fname}: WARNING: {ninconsistent} "inequality (max)" found')
                # Remove redundant points from inequality dataset
                xx_ineqMax_agg = np.delete(xx_ineqMax_agg, ig2)
                yy_ineqMax_agg = np.delete(yy_ineqMax_agg, ig2)
                zz_ineqMax_agg = np.delete(zz_ineqMax_agg, ig2)
                v_ineqMax_agg = np.delete(v_ineqMax_agg, ig2)

        if xIneqMin is not None:
            # Get single grid index for inequality (min) data points
            ix, iy, iz = img.pointToGridIndex(xx_ineqMin_agg, yy_ineqMin_agg, zz_ineqMin_agg,
                                              sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz)
            ig_ineqMin = img.gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz)

            # Get single grid index for inequality (max) data points
            ix, iy, iz = img.pointToGridIndex(xx_ineqMax_agg, yy_ineqMax_agg, zz_ineqMax_agg,
                                              sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz)
            ig_ineqMax = img.gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz)

            # Get single grid index with points from `xIneqMin` and `xIneqMax`
            ig_inter, ig1, ig2 = np.intersect1d(ig_ineqMin, ig_ineqMax, assume_unique=True, return_indices=True)

            if ig_inter.size:
                ii = np.where(v_ineqMin_agg[ig1] > v_ineqMax_agg[ig2])[0]
                if len(ii):
                    err_msg = f'{fname}: {len(ii)} grid cell(s) have inconsistent "inequality min" and "inequality max" data'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)
                    # if verbose > 0:
                    #     if logger:
                    #         logger.warning(f'{fname}: {len(ii)} grid cell(s) have inconsistent "inequality min" and "inequality max" data: inequlity max data has been removed')
                    #     else:
                    #         print(f'{fname}: WARNING: {len(ii)} grid cell(s) have inconsistent "inequality min" and "inequality max" data: inequlity max data has been removed')
                    # ig2 = ig2[ii]
                    # # Remove inconsistent inequality max
                    # xx_ineqMax_agg = np.delete(xx_ineqMax_agg, ig2)
                    # yy_ineqMax_agg = np.delete(yy_ineqMax_agg, ig2)
                    # zz_ineqMax_agg = np.delete(zz_ineqMax_agg, ig2)
                    # v_ineqMax_agg = np.delete(v_ineqMax_agg, ig2)

        if v_ineqMax_agg.shape[0]:
            dataPointSet.append(
                PointSet(npt=v_ineqMax_agg.shape[0], nv=4, val=np.array((xx_ineqMax_agg, yy_ineqMax_agg, zz_ineqMax_agg, v_ineqMax_agg)), varname=['X', 'Y', 'Z', f'{varname}_max'])
                )
        else:
            if verbose > 0:
                if logger:
                    logger.warning(f'{fname}: no inequality (max) data point in grid')
                else:
                    print(f'{fname}: WARNING: no inequality (max) data point in grid')
            xIneqMax = None

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            err_msg = f'{fname}: `mask` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if mask is not None and add_data_point_to_mask:
        # Make a copy of the original mask, to remove value in added mask cell at the end
        mask_original = np.copy(mask)
        # Add cell to mask if needed
        pts = np.zeros((0,3))
        if x is not None:
            pts = np.vstack((pts, np.array((xx_agg, yy_agg, zz_agg)).T))
        if xIneqMin is not None:
            pts = np.vstack((pts, np.array((xx_ineqMin_agg, yy_ineqMin_agg, zz_ineqMin_agg)).T))
        if xIneqMax is not None:
            pts = np.vstack((pts, np.array((xx_ineqMax_agg, yy_ineqMax_agg, zz_ineqMax_agg)).T))
        if pts.shape[0]:
            im_tmp = img.imageFromPoints(pts,
                        nx=nx, ny=ny, nz=nz,
                        sx=sx, sy=sy, sz=sz,
                        ox=ox, oy=oy, oz=oz,
                        indicator_var=True,
                        logger=logger)
            mask = 1.0*np.any((im_tmp.val[0], mask), axis=0)
            del(im_tmp)
        del(pts)

    # Prepare seed (for simulation in grid)
    seed = seed + 986 # same increment (whatever the number or realization done in sgs above)

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose > 0 and nth > os.cpu_count():
        if logger:
            logger.warning(f'{fname}: number of threads used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: number of threads used will exceed number of cpu(s) of the system...')

    if not aggregate_data_by_simul:
        # --- Fill mpds_geosClassicInput structure (C)
        try:
            mpds_geosClassicInput = fill_mpds_geosClassicInput(
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
                    0.0,
                    searchRadiusRelative,
                    nneighborMax,
                    searchNeighborhoodSortMode,
                    nGibbsSamplerPathMin,
                    nGibbsSamplerPathMax,
                    seed,
                    nreal,
                    logger=logger)

        except Exception as exc:
            err_msg = f'{fname}: cannot fill mpds_geosClassicInput C structure'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

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

        # # --- Set number of threads
        # if nthreads <= 0:
        #     nth = max(os.cpu_count() + nthreads, 1)
        # else:
        #     nth = nthreads

        if verbose > 1:
            if logger:
                logger.info('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            else:
                print('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
                sys.stdout.flush()
                sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

        # --- Launch "GeosClassicSim" (launch C code)
        # err = geosclassic.MPDSGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
        err = geosclassic.MPDSOMPGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

        # Free memory on C side: mpds_geosClassicInput
        geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
        geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)

        if err:
            # Free memory on C side: mpds_geosClassicOutput
            geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
            geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)
            # Free memory on C side: mpds_progressMonitor
            geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)
            # Raise error
            err_message = geosclassic.mpds_get_error_message(-err)
            err_message = err_message.replace('\n', '')
            err_msg = f'{fname}: {err_message}'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor, logger=logger)

        # Free memory on C side: mpds_geosClassicOutput
        geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
        geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

        # Free memory on C side: mpds_progressMonitor
        geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    else:
        # Equality data values will change for each realization
        if verbose > 1:
            if logger:
                logger.info('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            else:
                print('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
                sys.stdout.flush()
                sys.stdout.flush() # twice!, so that the previous print is flushed before launching geos-classic...

        # Initialization of image and warnings for storing results
        image = Img(nx=nx, ny=ny, nz=nz, sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz, nv=nreal, val=np.nan, logger=logger)
        nwarning = 0
        warnings = []
        outputReportFile_ir = None # default
        for ir in range(nreal):
            if ir > 0:
                # Set equality data values for realization index ir
                dataPointSet[0].val[3] = v_agg[ir]

            if outputReportFile is not None:
                outputReportFile_ir = outputReportFile + f'.{ir}'

            # --- Fill mpds_geosClassicInput structure (C)
            try:
                mpds_geosClassicInput = fill_mpds_geosClassicInput(
                        space_dim,
                        cov_model,
                        nx, ny, nz,
                        sx, sy, sz,
                        ox, oy, oz,
                        varname,
                        outputReportFile_ir,
                        computationMode,
                        None,
                        dataPointSet,
                        mask,
                        mean,
                        var,
                        0.0,
                        searchRadiusRelative,
                        nneighborMax,
                        searchNeighborhoodSortMode,
                        nGibbsSamplerPathMin,
                        nGibbsSamplerPathMax,
                        seed+ir, # seed for realization index ir
                        1, # one real
                        logger=logger)
                
            except Exception as exc:
                err_msg = f'{fname}: cannot fill mpds_geosClassicInput C structure'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

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
            mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
            # if verbose < 3:
            #     mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
            # else:
            #     mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr
            #
            # if verbose > 1:
            #     if logger:
            #         logger.info('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            #     else:
            #         print('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            #         sys.stdout.flush()
            #         sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

            # --- Launch "GeosClassicSim" (launch C code)
            # err = geosclassic.MPDSGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
            err = geosclassic.MPDSOMPGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

            # Free memory on C side: mpds_geosClassicInput
            geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
            geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)

            if err:
                # Free memory on C side: mpds_geosClassicOutput
                geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
                geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)
                # Free memory on C side: mpds_progressMonitor
                geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)
                # Raise error
                err_message = geosclassic.mpds_get_error_message(-err)
                err_message = err_message.replace('\n', '')
                err_msg = f'{fname}: {err_message}'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor, logger=logger)

            # Free memory on C side: mpds_geosClassicOutput
            geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
            geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

            # Free memory on C side: mpds_progressMonitor
            geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

            image.val[ir] = geosclassic_output['image'].val[0]
            nwarning = nwarning + int(geosclassic_output['nwarning'])
            warnings.extend(geosclassic_output['warnings'])

            del(geosclassic_output)

        # Remove duplicated warnings
        warnings = list(np.unique(warnings))

        # Rename variables
        ndigit = geosclassic.MPDS_GEOS_CLASSIC_NB_DIGIT_FOR_REALIZATION_NUMBER
        for j in range(image.nv):
            image.varname[j] = image.varname[j][:-ndigit] + f'{j:0{ndigit}d}'

        # Set geosclassic_output
        geosclassic_output = {'image':image, 'nwarning':nwarning, 'warnings':warnings}

    if geosclassic_output is not None and mask is not None and add_data_point_to_mask:
        # Remove the value out of the original mask (using its copy see above)
        geosclassic_output['image'].val[:, mask_original==0.0] = np.nan

    if verbose > 1 and geosclassic_output:
        if logger:
            logger.info(f'{fname}: Geos-Classic run complete')
        else:
            print(f'{fname}: Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose > 0 and geosclassic_output and geosclassic_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({geosclassic_output['nwarning']} times in all):")
            for i, warning_message in enumerate(geosclassic_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

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
        aggregate_data_op=None,
        aggregate_data_op_kwargs=None,
        aggregate_data_ineqMin_op='max',
        aggregate_data_ineqMin_op_kwargs=None,
        aggregate_data_ineqMax_op='min',
        aggregate_data_ineqMax_op_kwargs=None,
        mask=None,
        add_data_point_to_mask=True,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        nGibbsSamplerPathMin=50,
        nGibbsSamplerPathMax=200,
        seed=None,
        outputReportFile=None,
        treat_image_one_by_one=False,
        nproc=None, nthreads_per_proc=None,
        verbose=2,
        logger=None):
    """
    Computes the same as the function :func:`simulate2D`, using multiprocessing.

    All the parameters are the same as those of the function :func:`simulate2D`,
    except `nthreads` that is replaced by the parameters `nproc` and
    `nthreads_per_proc`, and an extra parameter `treat_image_one_by_one`.

    This function launches multiple processes (based on `multiprocessing`
    package):

    - `nproc` parallel processes using each one `nthreads_per_proc` threads \
    are launched [parallel calls of the function :func:`simulate2D`]
    - the set of realizations (specified by `nreal`) is distributed in a \
    balanced way over the processes
    - in terms of resources, this implies the use of `nproc*nthreads_per_proc` \
    cpu(s)

    See function :func:`simulate2D`.

    **Parameters (new)**
    --------------------
    nproc : int, optional
        number of process(es); by default (`None`):
        `nproc` is set to `min(nmax-1, nreal)` (but at least 1), where nmax is
        the total number of cpu(s) of the system (retrieved by
        `multiprocessing.cpu_count()`)

    nthreads_per_proc : int, optional
        number of thread(s) per process (should be > 0); by default (`None`):
        `nthreads_per_proc` is automatically computed as the maximal integer
        (but at least 1) such that `nproc*nthreads_per_proc <= nmax-1`, where
        nmax is the total number of cpu(s) of the system (retrieved by
        `multiprocessing.cpu_count()`)

    treat_image_one_by_one : bool, default: False
        keyword argument passed to the function :func:`geone.img.gatherImages`:

        - if `True`: images (result of each process) are gathered one by one, \
        i.e. the variables of each image are inserted in an output image one by \
        one and removed from the source (slower, may save memory)
        - if `False`: images (result of each process) are gathered at once, \
        i.e. the variables of all images are inserted in an output image at once, \
        and then removed (faster)
    """
    fname = 'simulate2D_mp'

    if logger:
        logger.warning(f'{fname}: deprecated function, use function `simulate` instead')
    else:
        print(f'{fname}: WARNING: deprecated function, use function `simulate` instead')

    # Set number of process(es): nproc
    if nproc is None:
        nproc = max(min(multiprocessing.cpu_count()-1, nreal), 1)
    else:
        nproc_tmp = nproc
        nproc = max(min(int(nproc), nreal), 1)
        if verbose > 1 and nproc != nproc_tmp:
            if logger:
                logger.info(f'{fname}: number of process(es) has been changed (now: nproc={nproc})')
            else:
                print(f'{fname}: number of process(es) has been changed (now: nproc={nproc})')

    # Set number of threads per process: nth
    if nthreads_per_proc is None:
        nth = max(int(np.floor((multiprocessing.cpu_count()-1) / nproc)), 1)
    else:
        nth = max(int(nthreads_per_proc), 1)
        if verbose > 1 and nth != nthreads_per_proc:
            if logger:
                logger.info(f'{fname}: number of threads per process has been changed (now: nthreads_per_proc={nth})')
            else:
                print(f'{fname}: number of threads per process has been changed (now: nthreads_per_proc={nth})')

    if verbose > 0 and nproc * nth > multiprocessing.cpu_count():
        if logger:
            logger.warning(f'{fname}: total number of cpu(s) used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: total number of cpu(s) used will exceed number of cpu(s) of the system...')

    # Set the distribution of the realizations over the processes
    # Condider the Euclidean division of nreal by nproc:
    #     nreal = q * nproc + r, with 0 <= r < nproc
    # Then, (q+1) realizations will be done on process 0, 1, ..., r-1, and q realization on process r, ..., nproc-1
    # Define the list real_index_proc of length (nproc+1) such that
    #   real_index_proc[i], ..., real_index_proc[i+1] - 1 : are the realization indices run on process i
    q, r = np.divmod(nreal, nproc)
    real_index_proc = [i*q + min(i, r) for i in range(nproc+1)]

    if verbose > 1:
        if logger:
            logger.info('{}: Geos-Classic running on {} process(es)... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, nproc, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        else:
            print('{}: Geos-Classic running on {} process(es)... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, nproc, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            sys.stdout.flush()
            sys.stdout.flush() # twice!, so that the previous print is flushed before launching geos-classic...

    # Prepare seed
    if seed is None:
        seed = np.random.randint(1, 1000000)
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
        verbose_p = 0
        # if i==0:
        #     verbose_p = min(verbose, 1) # allow to print warnings for process i
        # else:
        #     verbose_p = 0
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
                aggregate_data_op,
                aggregate_data_op_kwargs,
                aggregate_data_ineqMin_op,
                aggregate_data_ineqMin_op_kwargs,
                aggregate_data_ineqMax_op,
                aggregate_data_ineqMax_op_kwargs,
                mask,
                add_data_point_to_mask,
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                nGibbsSamplerPathMin,
                nGibbsSamplerPathMax,
                seed_p,                      # seed (adjusted)
                outputReportFile_p,          # outputReportFile (adjusted)
                nth,                         # nthreads
                verbose_p),                  # verbose (adjusted)
                kwds={'logger':logger})
            )

    # Properly end working process
    pool.close() # Prevents any more tasks from being submitted to the pool,
    pool.join()  # then, wait for the worker processes to exit.

    # Get result from each process
    geosclassic_output_proc = [p.get() for p in out_pool]

    if np.any([out is None for out in geosclassic_output_proc]):
        return None

    # Gather results from every process
    # image
    image = []
    for out in geosclassic_output_proc:
        if out['image'] is not None:
            image.append(out['image'])
            del(out['image'])
    if len(image) == 0:
        image = None
    # Gather images and adjust variable names
    all_image = img.gatherImages(image, keep_varname=True, rem_var_from_source=True, treat_image_one_by_one=treat_image_one_by_one, logger=logger)
    ndigit = geosclassic.MPDS_GEOS_CLASSIC_NB_DIGIT_FOR_REALIZATION_NUMBER
    for j in range(all_image.nv):
        all_image.varname[j] = all_image.varname[j][:-ndigit] + f'{j:0{ndigit}d}'

    # nwarning
    nwarning = np.sum([out['nwarning'] for out in geosclassic_output_proc])
    # warnings
    warnings = list(np.unique(np.hstack([out['warnings'] for out in geosclassic_output_proc])))

    geosclassic_output = {'image':all_image, 'nwarning':nwarning, 'warnings':warnings}

    if verbose > 1 and geosclassic_output:
        if logger:
            logger.info(f'{fname}: Geos-Classic run complete (all process(es))')
        else:
            print(f'{fname}: Geos-Classic run complete (all process(es))')

    # Show (print) encountered warnings
    if verbose > 0 and geosclassic_output and geosclassic_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({geosclassic_output['nwarning']} times in all):")
            for i, warning_message in enumerate(geosclassic_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

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
        aggregate_data_op=None,
        aggregate_data_op_kwargs=None,
        aggregate_data_ineqMin_op='max',
        aggregate_data_ineqMin_op_kwargs=None,
        aggregate_data_ineqMax_op='min',
        aggregate_data_ineqMax_op_kwargs=None,
        mask=None,
        add_data_point_to_mask=True,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        nGibbsSamplerPathMin=50,
        nGibbsSamplerPathMax=200,
        seed=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2,
        logger=None):
    """
    Generates 3D simulations (Sequential Gaussian Simulation, SGS).

    A simulation takes place in (center of) grid cells, based on simple or
    ordinary kriging.

    Parameters
    ----------
    cov_model : :class:`geone.covModel.CovModel3D`
        covariance model in 3D

    dimension : 3-tuple of ints
        `dimension=(nx, ny, nz)`, number of cells in the 3D simulation grid along
        each axis

    spacing : 3-tuple of floats, default: (1.0,1.0, 1.0)
        `spacing=(sx, sy, sz)`, cell size along each axis

    origin : 3-tuple of floats, default: (0.0, 0.0, 0.0)
        `origin=(ox, oy, oz)`, origin of the 3D simulation grid (bottom-lower-left
        corner)

    method : str {'simple_kriging', 'ordinary_kriging'}, default: 'simple_kriging'
        type of kriging

    nreal : int, default: 1
        number of realizations

    mean : function (callable), or array-like of floats, or float, optional
        kriging mean value:

        - if a function: function of three arguments (xi, yi, zi) that returns \
        the mean at location (xi, yi, zi)
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), mean values at grid cells (for \
        non-stationary mean)
        - if a float: same mean value at every grid cell
        - by default (`None`): the mean of data value (`v`) (0.0 if no data) is \
        considered at every grid cell

    var : function (callable), or array-like of floats, or float, optional
        kriging variance value:

        - if a function: function of three arguments (xi, yi, yi) that returns \
        the variance at location (xi, yi, zi)
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), variance values at grid cells (for \
        non-stationary variance)
        - if a float: same variance value at every grid cell
        - by default (`None`): not used (use of covariance model only)

    x : 2D array of floats of shape (n, 3), optional
        data points locations, with n the number of data points, each row of `x`
        is the float coordinates of one data point; note: if n=1, a 1D array of
        shape (3,) is accepted

    v : 1D array of floats of shape (n,), optional
        data values at `x` (`v[i]` is the data value at `x[i]`)

    xIneqMin : 2D array of floats of shape (nIneqMin, 3), optional
        data points locations, for inequality data with lower bound, with
        nIneqMin the number of data points, each row of `xIneqMin` is the float
        coordinates of one data point; note: if nIneqMin=1, a 1D array of
        shape (3,) is accepted

    vIneqMin : 1D array of floats of shape (nIneqMin,), optional
        inequality data values, lower bounds, at `xIneqMin` (`vIneqMin[i]` is the
        data value at `xIneqMin[i]`)

    xIneqMax : 2D array of floats of shape (nIneqMax, 3), optional
        data points locations, for inequality data with upper bound, with
        nIneqMax the number of data points, each row of `xIneqMax` is the float
        coordinates of one data point; note: if nIneqMax=1, a 1D array of
        shape (3,) is accepted

    vIneqMax : 1D array of floats of shape (nIneqMax,), optional
        inequality data values, upper bounds, at `xIneqMax` (`vIneqMax[i]` is the
        data value at `xIneqMax[i]`)

    aggregate_data_op : str {'sgs', 'krige', 'min', 'max', 'mean', 'quantile', \
                        'most_freq', 'random'}, optional
        operation used to aggregate data points falling in the same grid cells

        - if `aggregate_data_op='sgs'`: function :func:`geone.covModel.sgs` is used \
        with the covariance model `cov_model` given in arguments, as well as \
        the parameter `nneighborMax` given in arguments unless it is given \
        in `aggregate_data_op_kwargs`
        - if `aggregate_data_op='krige'`: function :func:`geone.covModel.krige` is used \
        with the covariance model `cov_model` given in arguments, as well as \
        the parameters `use_unique_neighborhood`, `nneighborMax` given in \
        arguments unless they are given in `aggregate_data_op_kwargs`
        - if `aggregate_data_op='most_freq'`: most frequent value is selected \
        (smallest one if more than one value with the maximal frequence)
        - if `aggregate_data_op='random'`: value from a random point is selected \
        - otherwise: the function `numpy.<aggregate_data_op>` is used with the \
        additional parameters given by `aggregate_data_op_kwargs`, note that, e.g. \
        `aggregate_data_op='quantile'` requires the additional parameter \
        `q=<quantile_to_compute>`

        Note: if `aggregate_data_op='sgs'` or `aggregate_data_op='random'`, the
        aggregation is done for each realization (simulation), i.e. each simulation
        on the grid starts with a new set of values in conditioning grid cells

        By default: if covariance model has stationary ranges and weight (sill),
        `aggregate_data_op='sgs'` is used, otherwise `aggregate_data_op='mean'`

    aggregate_data_op_kwargs : dict, optional
        keyword arguments to be passed to `geone.covModel.sgs`,
        `geone.covModel.krige`, or `numpy.<aggregate_data_op>`, according to
        the parameter `aggregate_data_op`

    aggregate_data_ineqMin_op : str {'min', 'max', 'mean', 'quantile', \
                        'most_freq', 'random'}, default: 'max'
        operation used to aggregate inequality (min, lower boudns) data points
        falling in the same grid cells:

        - if `aggregate_data_ineqMin_op='most_freq'`: most frequent value is \
        selected (smallest one if more than one value with the maximal frequence)
        - if `aggregate_data_ineqMin_op='random'`: value from a random point is \
        selected
        - otherwise: the function `numpy.<aggregate_data_ineqMin_op>` is used with \
        the additional parameters given by `aggregate_data_ineqMin_op_kwargs`, \
        note that, e.g. `aggregate_data_ineqMin_op='quantile'` requires the \
        additional parameter `q=<quantile_to_compute>`

        Note: in any case, the aggregation is done once, i.e. same inequality
        values are used for each simulation on the grid

    aggregate_data_ineqMin_op_kwargs : dict, optional
        keyword arguments to be passed to `numpy.<aggregate_data_ineqMin_op>`,
        according to the parameter `aggregate_data_ineqMin_op`

    aggregate_data_ineqMax_op : str {'min', 'max', 'mean', 'quantile', \
                        'most_freq', 'random'}, default: 'min'
        operation used to aggregate inequality (min, lower boudns) data points
        falling in the same grid cells:

        - if `aggregate_data_ineqMax_op='most_freq'`: most frequent value is \
        selected (smallest one if more than one value with the maximal frequence)
        - if `aggregate_data_ineqMax_op='random'`: value from a random point is \
        selected
        - otherwise: the function `numpy.<aggregate_data_ineqMax_op>` is used with \
        the additional parameters given by `aggregate_data_ineqMax_op_kwargs`, \
        note that, e.g. `aggregate_data_ineqMax_op='quantile'` requires the \
        additional parameter `q=<quantile_to_compute>`

        Note: in any case, the aggregation is done once, i.e. same inequality
        values are used for each simulation on the grid

    aggregate_data_ineqMax_op_kwargs : dict, optional
        keyword arguments to be passed to `numpy.<aggregate_data_ineqMax_op>`,
        according to the parameter `aggregate_data_ineqMax_op`

    mask : array-like, optional
        mask value at grid cells (value 1 for simulated cells, value 0 for not
        simulated cells); the size of the array must be equal to the number of
        grid cells (the array is reshaped if needed)

    add_data_point_to_mask : bool, default: True
        - if `True`: any grid cell that contains a data point is added to (the \
        simulated part of) the mask (if present), i.e. mask value at those cells \
        are set to 1; at the end of the computation the "new mask cells" are \
        removed (by setting a missing value (`numpy.nan`) for the variable out of \
        the original mask)
        - if `False`: original mask is kept as given in input, and data point \
        falling out of (the simulated part of) the mask (if present) are ignored

    searchRadiusRelative : float, default: 1.2
        indicates how the search ellipsoid is limited (should be positive): let
        r_i be the ranges of the covariance model along its main axes, when
        estimating/simulating a cell x, a cell y is taken into account iff it is
        within the ellipsoid centered at x of half axes equal to
        `searchRadiusRelative` * r_i; 
        (note that the distances to the central node are computed in the axes 
        sytem supporting the covariance model and accounting for anisotropy given 
        by the ranges)
        note (Geos-Classic library): if a range r_i is non-stationary over the grid, 
        its maximal value over the grid is considered;

    nneighborMax : int, default: 12
        maximum number of cells retrieved from the search ellipsoid (when
        estimating/simulating a cell), `nneighborMax=-1` for unlimited

    searchNeighborhoodSortMode : int, optional
        indicates how to sort the search neighboorhood cells (neighbors); they
        are sorted in increasing order according to:

        - `searchNeighborhoodSortMode=0`: distance in the usual axes system
        - `searchNeighborhoodSortMode=1`: distance in the axes sytem supporting \
        the covariance model and accounting for anisotropy given by the ranges
        - `searchNeighborhoodSortMode=2`: minus the evaluation of the covariance \
        model

        Notes:

        - if the covariance model has any non-stationary parameter, then \
        `searchNeighborhoodSortMode=2` is not allowed
        - if the covariance model has any non-stationary range or non-stationary \
        angle and `searchNeighborhoodSortMode=1`: "maximal ranges" (adapted to \
        direction from the central cell) are used to compute distance for sorting \
        the neighbors

        By default (`None`): the greatest possible value is used (i.e. 2 for
        stationary covariance model, or 1 otherwise)

    nGibbsSamplerPathMin: int, default: 50
        see `nGibbsSamplerPathMax`

    nGibbsSamplerPathMax: int, default: 200
        `nGibbsSamplerPathMin` and `nGibbsSamplerPathMax` are the mini and max number
        of Gibbs sampler paths to deal with inequality data; the conditioning locations
        with inequality data are first simulated (based on truncated gaussian
        distribution) sequentially; then, these locations are re-simulated following a
        new path as many times as needed, but the total number of paths will be between
        `nGibbsSamplerPathMin` and `nGibbsSamplerPathMax`

    seed : int, optional
        seed for initializing random number generator

    outputReportFile : str, optional
        name of the report file (if desired in output); by default (`None`): no
        report file

    nthreads : int, default: -1
        number of thread(s) to use for "GeosClassicSim" C program;
        `nthreads = -n <= 0`: maximal number of threads of the system except n
        (but at least 1)

    verbose : int, default: 2
        verbose mode, higher implies more printing (info):

        - 0: no display
        - 1: warnings
        - 2: warnings + basic info
        - 3 (or >2): all information

        note that if an error occurred, it is raised
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    geosclassic_output : dict
        geosclassic output in python, dictionary

        {'image':image, 'nwarning':nwarning, 'warnings':warnings}

        with:

        - image : :class:`geone.img.Img`
            output image, with `image.nv=nreal` variables (simulations);
            note: `image=None` if `mpds_geosClassicOutput->outputImage=NULL`

        - nwarning : int
            total number of warning(s) encountered (same warnings can be counted
            several times)

        - warnings : list of strs
            list of distinct warnings encountered (can be empty)
    """
    fname = 'simulate3D'

    if logger:
        logger.warning(f'{fname}: deprecated function, use function `simulate` instead')
    else:
        print(f'{fname}: WARNING: deprecated function, use function `simulate` instead')

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
    # nreal
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose > 0:
            if logger:
                logger.warning(f'{fname}: `nreal` <= 0: `None` is returned')
            else:
                print(f'{fname}: WARNING: `nreal` <= 0: `None` is returned')
        return None

    # cov_model
    if isinstance(cov_model, gcm.CovModel1D):
        cov_model = gcm.covModel1D_to_covModel3D(cov_model) # convert model 1D in 3D
            # -> will not be modified cov_model at exit

    if not isinstance(cov_model, gcm.CovModel3D):
        err_msg = f'{fname}: `cov_model` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    for el in cov_model.elem:
        # weight
        w = el[1]['w']
        if np.size(w) != 1 and np.size(w) != nxyz:
            err_msg = f"{fname}: `cov_model`: weight ('w') not compatible with simulation grid"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # ranges
        if 'r' in el[1].keys():
            for r in el[1]['r']:
                if np.size(r) != 1 and np.size(r) != nxyz:
                    err_msg = f"{fname}: `cov_model`: range ('r') not compatible with simulation grid"
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

        # additional parameter (s)
        if 's' in el[1].keys():
            s  = el[1]['s']
            if np.size(s) != 1 and np.size(s) != nxyz:
                err_msg = f"{fname}: `cov_model`: parameter ('s') not compatible with simulation grid"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # alpha
    angle = cov_model.alpha
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        err_msg = f"{fname}: `cov_model`: angle ('alpha') not compatible with simulation grid"
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # beta
    angle = cov_model.beta
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        err_msg = f"{fname}: `cov_model`: angle ('beta') not compatible with simulation grid"
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # gamma
    angle = cov_model.gamma
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        err_msg = f"{fname}: `cov_model`: angle ('gamma') not compatible with simulation grid"
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # aggregate_data_op (default)
    if aggregate_data_op is None:
        if not cov_model.is_weight_stationary() or not cov_model.is_range_stationary():
            aggregate_data_op = 'mean'
        else:
            aggregate_data_op = 'sgs'

    if aggregate_data_op_kwargs is None:
        aggregate_data_op_kwargs = {}

    if aggregate_data_ineqMin_op_kwargs is None:
        aggregate_data_ineqMin_op_kwargs = {}

    if aggregate_data_ineqMax_op_kwargs is None:
        aggregate_data_ineqMax_op_kwargs = {}

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    if method == 'simple_kriging':
        computationMode = 3
    elif method == 'ordinary_kriging':
        computationMode = 2
    else:
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - searchRadiusRelative
    if searchRadiusRelative < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
        err_msg = f'{fname}: `searchRadiusRelative` too small (should be at least {geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN})'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - nneighborMax
    if nneighborMax != -1 and nneighborMax <= 0:
        err_msg = f'{fname}: `nneighborMax` should be greater than 0 or equal to -1 (unlimited)'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - searchNeighborhoodSortMode
    if searchNeighborhoodSortMode is None:
        # set greatest possible value
        if cov_model.is_stationary():
            searchNeighborhoodSortMode = 2
        else:
            searchNeighborhoodSortMode = 1
    else:
        if searchNeighborhoodSortMode == 2:
            if not cov_model.is_stationary():
                err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # if searchNeighborhoodSortMode is None:
    #     # set greatest possible value
    #     if cov_model.is_stationary():
    #         searchNeighborhoodSortMode = 2
    #     elif cov_model.is_orientation_stationary() and cov_model.is_range_stationary():
    #         searchNeighborhoodSortMode = 1
    #     else:
    #         searchNeighborhoodSortMode = 0
    # else:
    #     if searchNeighborhoodSortMode == 2:
    #         if not cov_model.is_stationary():
    #             err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
    #             if logger: logger.error(err_msg)
    #             raise GeosclassicinterfaceError(err_msg)
    #     elif searchNeighborhoodSortMode == 1:
    #         if not cov_model.is_orientation_stationary() or not cov_model.is_range_stationary():
    #             err_msg = f'{fname}: `searchNeighborhoodSortMode=1` not allowed with non-stationary range or non-stationary orientation in covariance model'
    #             if logger: logger.error(err_msg)
    #             raise GeosclassicinterfaceError(err_msg)

    # Preparation of data points
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 3) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if xIneqMin is not None:
        xIneqMin = np.asarray(xIneqMin, dtype='float').reshape(-1, 3) # cast in 2-dimensional array if needed
        vIneqMin = np.asarray(vIneqMin, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(vIneqMin) != xIneqMin.shape[0]:
            err_msg = f'{fname}: length of `vIneqMin` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if xIneqMax is not None:
        xIneqMax = np.asarray(xIneqMax, dtype='float').reshape(-1, 3) # cast in 2-dimensional array if needed
        vIneqMax = np.asarray(vIneqMax, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(vIneqMax) != xIneqMax.shape[0]:
            err_msg = f'{fname}: length of `vIneqMax` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Check parameters - mean
    mean_x = mean
    if mean is not None:
        # if method == 'ordinary_kriging':
        #     err_msg = f'{fname}: specifying `mean` not allowed with ordinary kriging'
        #     if logger: logger.error(err_msg)
        #     raise GeosclassicinterfaceError(err_msg)

        if callable(mean):
            if x is not None:
                mean_x = mean(x[:, 0], x[:, 1], x[:, 2])
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            zi = oz + sz*(0.5+np.arange(nz)) # z-coordinate of cell center
            zzi, yyi, xxi = np.meshgrid(zi, yi, xi, indexing='ij')
            mean = mean(xxi, yyi, zzi) # replace function 'mean' by its evaluation on the grid
        else:
            mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if mean.size == 1:
                if x is not None:
                    mean_x = mean
            elif mean.size == nxyz:
                mean = mean.reshape(nz, ny, nx)
                if x is not None:
                    mean_x = img.Img_interp_func(img.Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv=1, val=mean, logger=logger), logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `mean` is not valid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # Check parameters - var
    var_x = var
    if var is not None:
        if method == 'ordinary_kriging':
            err_msg = f'{fname}: specifying `var` not allowed with ordinary kriging'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        if callable(var):
            if x is not None:
                var_x = var(x[:, 0], x[:, 1], x[:, 2])
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            zi = oz + sz*(0.5+np.arange(nz)) # z-coordinate of cell center
            zzi, yyi, xxi = np.meshgrid(zi, yi, xi, indexing='ij')
            var = var(xxi, yyi, zzi) # replace function 'var' by its evaluation on the grid
        else:
            var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if var.size == 1:
                if x is not None:
                    var_x = var
            elif var.size == nxyz:
                var = var.reshape(nz, ny, nx)
                if x is not None:
                    var_x = img.Img_interp_func(img.Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv=1, val=var, logger=logger), logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `var` is not valid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # Prepare seed
    if seed is None:
        seed = np.random.randint(1, 1000000)
    seed = int(seed)

    # data points: x, v, xIneqMin, vIneqMin, xIneqMax, vIneqMax
    dataPointSet = []

    # data point set from x, v
    aggregate_data_by_simul = False
    if x is not None:
        if aggregate_data_op == 'krige' or aggregate_data_op == 'sgs':
            if not cov_model.is_weight_stationary() or not cov_model.is_range_stationary():
                err_msg = f"{fname}: covariance model with non-stationary weight or range cannot be used with `aggregate_data_op`='{aggregate_data_op}'"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            if cov_model.is_orientation_stationary():
                cov_model_agg = cov_model
            else:
                cov_model_agg = gcm.copyCovModel(cov_model)
                cov_model_agg.set_alpha(0.0)
                cov_model_agg.set_beta(0.0)
                cov_model_agg.set_gamma(0.0)
            # Get grid cell with at least one data point:
            # x_agg: 2D array, each row contains the coordinates of the center of such cell
            im_tmp = img.imageFromPoints(x, values=None, varname=None,
                                         nx=nx, ny=ny, nz=nz, sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz,
                                         indicator_var=True, 
                                         count_var=False,
                                         logger=logger)
            ind_agg = np.where(im_tmp.val[0])
            if len(ind_agg[0]) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            x_agg = np.array((im_tmp.xx()[ind_agg].reshape(-1), im_tmp.yy()[ind_agg].reshape(-1), im_tmp.zz()[ind_agg].reshape(-1))).T
            # x_agg = np.array((im_tmp.xx()[*ind_agg].reshape(-1), im_tmp.yy()[*ind_agg].reshape(-1), im_tmp.zz()[*ind_agg].reshape(-1))).T
            del(im_tmp)
            # Compute
            # - kriging estimate (v_agg) and kriging std (v_agg_std) at x_agg,
            # - or nreal simulation(s) (v_agg) at x_agg
            if mean is not None and mean.size > 1:
                mean_x_agg = mean[ind_agg]
                # mean_x_agg = mean[*ind_agg]
            else:
                mean_x_agg = mean
            if var is not None and var.size > 1:
                var_x_agg = var[ind_agg]
                # var_x_agg = var[*ind_agg]
            else:
                var_x_agg = var
            if isinstance(cov_model.alpha, np.ndarray) and cov_model.alpha.size == nxyz:
                alpha_x_agg = cov_model.alpha[ind_agg]
                # alpha_x_agg = cov_model.alpha[*ind_agg]
            else:
                alpha_x_agg = cov_model.alpha
            if isinstance(cov_model.beta, np.ndarray) and cov_model.beta.size == nxyz:
                beta_x_agg = cov_model.beta[ind_agg]
                # beta_x_agg = cov_model.beta[*ind_agg]
            else:
                beta_x_agg = cov_model.beta
            if isinstance(cov_model.gamma, np.ndarray) and cov_model.gamma.size == nxyz:
                gamma_x_agg = cov_model.gamma[ind_agg]
                # gamma_x_agg = cov_model.gamma[*ind_agg]
            else:
                gamma_x_agg = cov_model.gamma
            # Set parameters `nneighborMax` from the arguments if not given in `aggregate_data_op_kwargs`
            if 'nneighborMax' not in aggregate_data_op_kwargs.keys():
                aggregate_data_op_kwargs['nneighborMax'] = nneighborMax
            if aggregate_data_op == 'krige':
                try:
                    v_agg, v_agg_std = gcm.krige(x, v, x_agg, cov_model_agg, method=method,
                                                 mean_x=mean_x, mean_xu=mean_x_agg,
                                                 var_x=var_x, var_xu=var_x_agg,
                                                 alpha_xu=alpha_x_agg, beta_xu=beta_x_agg, gamma_xu=gamma_x_agg,
                                                 verbose=0, logger=logger,
                                                 **aggregate_data_op_kwargs)
                except Exception as exc:
                    err_msg = f'{fname}: kriging error'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg) from exc

            else:
                aggregate_data_by_simul = True
                try:
                    v_agg = gcm.sgs(x, v, x_agg, cov_model_agg, method=method,
                                    mean_x=mean_x, mean_xu=mean_x_agg,
                                    var_x=var_x, var_xu=var_x_agg,
                                    alpha_xu=alpha_x_agg, beta_xu=beta_x_agg, gamma_xu=gamma_x_agg,
                                    nreal=nreal, seed=seed,
                                    verbose=0, logger=logger,
                                    **aggregate_data_op_kwargs)
                except Exception as exc:
                    err_msg = f'{fname}: sgs error'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg) from exc

            xx_agg, yy_agg, zz_agg = x_agg.T
        elif aggregate_data_op == 'random':
            aggregate_data_by_simul = True
            # Aggregate data on grid cell by taking random point
            xx, yy, zz = x.T
            # first realization of v_agg
            try:
                xx_agg, yy_agg, zz_agg, v_agg, i_inv = img.aggregateDataPointsWrtGrid(
                                                        xx, yy, zz, v,
                                                        nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                        op=aggregate_data_op, 
                                                        return_inverse=True,
                                                        logger=logger,
                                                        **aggregate_data_op_kwargs)
            except Exception as exc:
                err_msg = f'{fname}: data aggregation'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

            if len(xx_agg) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            # next realizations of v_agg
            v_agg = np.vstack((v_agg, np.zeros((nreal-1, v_agg.size))))
            for i in range(1, nreal):
                v_agg[i] = [v[np.random.choice(np.where(i_inv==j)[0])] for j in range(len(xx_agg))]
        else:
            # Aggregate data on grid cell by using the given operation
            xx, yy, zz = x.T
            try:
                xx_agg, yy_agg, zz_agg, v_agg = img.aggregateDataPointsWrtGrid(
                                                    xx, yy, zz, v,
                                                    nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                    op=aggregate_data_op, 
                                                    logger=logger,
                                                    **aggregate_data_op_kwargs)
            except Exception as exc:
                err_msg = f"{fname}: data aggregation (`aggregate_data_op='{aggregate_data_op}'`) failed"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

            if len(xx_agg) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

        if xIneqMin is not None or xIneqMax is not None:
            # Get single grid index for data points
            ix, iy, iz = img.pointToGridIndex(xx_agg, yy_agg, zz_agg,
                                              sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz)
            ig = img.gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz)

        if not aggregate_data_by_simul:
            dataPointSet.append(
                PointSet(npt=v_agg.shape[0], nv=4, val=np.array((xx_agg, yy_agg, zz_agg, v_agg)), varname=['X', 'Y', 'Z', varname])
                )
        else:
            # Integrate data points from sgs index 0
            dataPointSet.append(
                PointSet(npt=v_agg.shape[1], nv=4, val=np.array((xx_agg, yy_agg, zz_agg, v_agg[0])), varname=['X', 'Y', 'Z', varname])
                )

    # data point set from xIneqMin, vIneqMin
    if xIneqMin is not None:
        # Aggregate data on grid cell by using the given operation
        xx, yy, zz = xIneqMin.T
        try:
            xx_ineqMin_agg, yy_ineqMin_agg, zz_ineqMin_agg, v_ineqMin_agg = img.aggregateDataPointsWrtGrid(
                                                xx, yy, zz, vIneqMin,
                                                nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                op=aggregate_data_ineqMin_op, 
                                                logger=logger,
                                                **aggregate_data_ineqMin_op_kwargs)
        except Exception as exc:
            err_msg = f"{fname}: inequality data (min) aggregation (`aggregate_data_op='{aggregate_data_ineqMin_op}'`) failed"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

        if x is not None:
            # Get single grid index for inequality (min) data points
            ix, iy, iz = img.pointToGridIndex(xx_ineqMin_agg, yy_ineqMin_agg, zz_ineqMin_agg,
                                              sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz)
            ig_ineqMin = img.gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz)

            # Get single grid index with points from `x` and `xIneqMin`
            ig_inter, ig1, ig2 = np.intersect1d(ig, ig_ineqMin, assume_unique=True, return_indices=True)

            if ig_inter.size:
                if verbose > 0:
                    if logger:
                        logger.warning(f'{fname}: {ig_inter.size} grid cell(s) have both "inequality (min)" and "equality" data: inequlity data has been removed')
                    else:
                        print(f'{fname}: WARNING: {ig_inter.size} grid cell(s) have both "inequality (min)" and "equality" data: inequlity data has been removed')
                    if not aggregate_data_by_simul:
                        ninconsistent = (v_agg[ig1] < v_ineqMin_agg[ig2]).sum()
                        if ninconsistent:
                            if logger:
                                logger.warning(f'{fname}: {ninconsistent} "inequality (min)" found')
                            else:
                                print(f'{fname}: WARNING: {ninconsistent} "inequality (min)" found')
                # Remove redundant points from inequality dataset
                xx_ineqMin_agg = np.delete(xx_ineqMin_agg, ig2)
                yy_ineqMin_agg = np.delete(yy_ineqMin_agg, ig2)
                zz_ineqMin_agg = np.delete(zz_ineqMin_agg, ig2)
                v_ineqMin_agg = np.delete(v_ineqMin_agg, ig2)

        if v_ineqMin_agg.shape[0]:
            dataPointSet.append(
                PointSet(npt=v_ineqMin_agg.shape[0], nv=4, val=np.array((xx_ineqMin_agg, yy_ineqMin_agg, zz_ineqMin_agg, v_ineqMin_agg)), varname=['X', 'Y', 'Z', f'{varname}_min'])
                )
        else:
            if verbose > 0:
                if logger:
                    logger.warning(f'{fname}: no inequality (min) data point in grid')
                else:
                    print(f'{fname}: WARNING: no inequality (min) data point in grid')
            xIneqMin = None

    # data point set from xIneqMax, vIneqMax
    if xIneqMax is not None:
        # Aggregate data on grid cell by using the given operation
        xx, yy, zz = xIneqMax.T
        try:
            xx_ineqMax_agg, yy_ineqMax_agg, zz_ineqMax_agg, v_ineqMax_agg = img.aggregateDataPointsWrtGrid(
                                                xx, yy, zz, vIneqMax,
                                                nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                op=aggregate_data_ineqMax_op, 
                                                logger=logger,
                                                **aggregate_data_ineqMax_op_kwargs)
        except Exception as exc:
            err_msg = f"{fname}: inequality data (max) aggregation (`aggregate_data_op='{aggregate_data_ineqMax_op}'`) failed"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

        if x is not None:
            # Get single grid index for inequality (max) data points
            ix, iy, iz = img.pointToGridIndex(xx_ineqMax_agg, yy_ineqMax_agg, zz_ineqMax_agg,
                                              sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz)
            ig_ineqMax = img.gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz)

            # Get single grid index with points from `x` and `xIneqMax`
            ig_inter, ig1, ig2 = np.intersect1d(ig, ig_ineqMax, assume_unique=True, return_indices=True)

            if ig_inter.size:
                if verbose > 0:
                    if logger:
                        logger.warning(f'{fname}: {ig_inter.size} grid cell(s) have both "inequality (max)" and "equality" data: inequlity data has been removed')
                    else:
                        print(f'{fname}: WARNING: {ig_inter.size} grid cell(s) have both "inequality (max)" and "equality" data: inequlity data has been removed')
                    if not aggregate_data_by_simul:
                        ninconsistent = (v_agg[ig1] > v_ineqMax_agg[ig2]).sum()
                        if ninconsistent:
                            if logger:
                                logger.warning(f'{fname}: {ninconsistent} "inequality (max)" found')
                            else:
                                print(f'{fname}: WARNING: {ninconsistent} "inequality (max)" found')
                # Remove redundant points from inequality dataset
                xx_ineqMax_agg = np.delete(xx_ineqMax_agg, ig2)
                yy_ineqMax_agg = np.delete(yy_ineqMax_agg, ig2)
                zz_ineqMax_agg = np.delete(zz_ineqMax_agg, ig2)
                v_ineqMax_agg = np.delete(v_ineqMax_agg, ig2)

        if xIneqMin is not None:
            # Get single grid index for inequality (min) data points
            ix, iy, iz = img.pointToGridIndex(xx_ineqMin_agg, yy_ineqMin_agg, zz_ineqMin_agg,
                                              sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz)
            ig_ineqMin = img.gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz)

            # Get single grid index for inequality (max) data points
            ix, iy, iz = img.pointToGridIndex(xx_ineqMax_agg, yy_ineqMax_agg, zz_ineqMax_agg,
                                              sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz)
            ig_ineqMax = img.gridIndexToSingleGridIndex(ix, iy, iz, nx, ny, nz)

            # Get single grid index with points from `xIneqMin` and `xIneqMax`
            ig_inter, ig1, ig2 = np.intersect1d(ig_ineqMin, ig_ineqMax, assume_unique=True, return_indices=True)

            if ig_inter.size:
                ii = np.where(v_ineqMin_agg[ig1] > v_ineqMax_agg[ig2])[0]
                if len(ii):
                    err_msg = f'{fname}: {len(ii)} grid cell(s) have inconsistent "inequality min" and "inequality max" data'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)
                    # if verbose > 0:
                    #     if logger:
                    #         logger.warning(f'{fname}: {len(ii)} grid cell(s) have inconsistent "inequality min" and "inequality max" data: inequlity max data has been removed')
                    #     else:
                    #         print(f'{fname}: WARNING: {len(ii)} grid cell(s) have inconsistent "inequality min" and "inequality max" data: inequlity max data has been removed')
                    # ig2 = ig2[ii]
                    # # Remove inconsistent inequality max
                    # xx_ineqMax_agg = np.delete(xx_ineqMax_agg, ig2)
                    # yy_ineqMax_agg = np.delete(yy_ineqMax_agg, ig2)
                    # zz_ineqMax_agg = np.delete(zz_ineqMax_agg, ig2)
                    # v_ineqMax_agg = np.delete(v_ineqMax_agg, ig2)

        if v_ineqMax_agg.shape[0]:
            dataPointSet.append(
                PointSet(npt=v_ineqMax_agg.shape[0], nv=4, val=np.array((xx_ineqMax_agg, yy_ineqMax_agg, zz_ineqMax_agg, v_ineqMax_agg)), varname=['X', 'Y', 'Z', f'{varname}_max'])
                )
        else:
            if verbose > 0:
                if logger:
                    logger.warning(f'{fname}: no inequality (max) data point in grid')
                else:
                    print(f'{fname}: WARNING: no inequality (max) data point in grid')
            xIneqMax = None

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            err_msg = f'{fname}: `mask` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if mask is not None and add_data_point_to_mask:
        # Make a copy of the original mask, to remove value in added mask cell at the end
        mask_original = np.copy(mask)
        # Add cell to mask if needed
        pts = np.zeros((0,3))
        if x is not None:
            pts = np.vstack((pts, np.array((xx_agg, yy_agg, zz_agg)).T))
        if xIneqMin is not None:
            pts = np.vstack((pts, np.array((xx_ineqMin_agg, yy_ineqMin_agg, zz_ineqMin_agg)).T))
        if xIneqMax is not None:
            pts = np.vstack((pts, np.array((xx_ineqMax_agg, yy_ineqMax_agg, zz_ineqMax_agg)).T))
        if pts.shape[0]:
            im_tmp = img.imageFromPoints(pts,
                        nx=nx, ny=ny, nz=nz,
                        sx=sx, sy=sy, sz=sz,
                        ox=ox, oy=oy, oz=oz,
                        indicator_var=True,
                        logger=logger)
            mask = 1.0*np.any((im_tmp.val[0], mask), axis=0)
            del(im_tmp)
        del(pts)

    # Prepare seed (for simulation in grid)
    seed = seed + 986 # same increment (whatever the number or realization done in sgs above)

    # --- Set number of threads
    if nthreads <= 0:
        nth = max(os.cpu_count() + nthreads, 1)
    else:
        nth = nthreads

    if verbose > 0 and nth > os.cpu_count():
        if logger:
            logger.warning(f'{fname}: number of threads used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: number of threads used will exceed number of cpu(s) of the system...')

    if not aggregate_data_by_simul:
        # --- Fill mpds_geosClassicInput structure (C)
        try:
            mpds_geosClassicInput = fill_mpds_geosClassicInput(
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
                    0.0,
                    searchRadiusRelative,
                    nneighborMax,
                    searchNeighborhoodSortMode,
                    nGibbsSamplerPathMin,
                    nGibbsSamplerPathMax,
                    seed,
                    nreal,
                    logger=logger)
            
        except Exception as exc:
            err_msg = f'{fname}: cannot fill mpds_geosClassicInput C structure'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

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

        # # --- Set number of threads
        # if nthreads <= 0:
        #     nth = max(os.cpu_count() + nthreads, 1)
        # else:
        #     nth = nthreads

        if verbose > 1:
            if logger:
                logger.info('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            else:
                print('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
                sys.stdout.flush()
                sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

        # --- Launch "GeosClassicSim" (launch C code)
        # err = geosclassic.MPDSGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
        err = geosclassic.MPDSOMPGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

        # Free memory on C side: mpds_geosClassicInput
        geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
        geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)

        if err:
            # Free memory on C side: mpds_geosClassicOutput
            geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
            geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)
            # Free memory on C side: mpds_progressMonitor
            geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)
            # Raise error
            err_message = geosclassic.mpds_get_error_message(-err)
            err_message = err_message.replace('\n', '')
            err_msg = f'{fname}: {err_message}'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor, logger=logger)

        # Free memory on C side: mpds_geosClassicOutput
        geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
        geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

        # Free memory on C side: mpds_progressMonitor
        geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    else:
        # Equality data values will change for each realization
        if verbose > 1:
            if logger:
                logger.info('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            else:
                print('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
                sys.stdout.flush()
                sys.stdout.flush() # twice!, so that the previous print is flushed before launching geos-classic...

        # Initialization of image and warnings for storing results
        image = Img(nx=nx, ny=ny, nz=nz, sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz, nv=nreal, val=np.nan, logger=logger)
        nwarning = 0
        warnings = []
        outputReportFile_ir = None # default
        for ir in range(nreal):
            if ir > 0:
                # Set equality data values for realization index ir
                dataPointSet[0].val[3] = v_agg[ir]

            if outputReportFile is not None:
                outputReportFile_ir = outputReportFile + f'.{ir}'

            # --- Fill mpds_geosClassicInput structure (C)
            try:
                mpds_geosClassicInput = fill_mpds_geosClassicInput(
                        space_dim,
                        cov_model,
                        nx, ny, nz,
                        sx, sy, sz,
                        ox, oy, oz,
                        varname,
                        outputReportFile_ir,
                        computationMode,
                        None,
                        dataPointSet,
                        mask,
                        mean,
                        var,
                        0.0,
                        searchRadiusRelative,
                        nneighborMax,
                        searchNeighborhoodSortMode,
                        nGibbsSamplerPathMin,
                        nGibbsSamplerPathMax,
                        seed+ir, # seed for realization index ir
                        1, # one real
                        logger=logger)
                
            except Exception as exc:
                err_msg = f'{fname}: cannot fill mpds_geosClassicInput C structure'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

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
            mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
            # if verbose < 3:
            #     mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor0_ptr
            # else:
            #     mpds_updateProgressMonitor = geosclassic.MPDSUpdateProgressMonitor4_ptr
            #
            # if verbose > 1:
            #     if logger:
            #         logger.info('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            #     else:
            #         print('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            #         sys.stdout.flush()
            #         sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

            # --- Launch "GeosClassicSim" (launch C code)
            # err = geosclassic.MPDSGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
            err = geosclassic.MPDSOMPGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

            # Free memory on C side: mpds_geosClassicInput
            geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
            geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)

            if err:
                # Free memory on C side: mpds_geosClassicOutput
                geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
                geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)
                # Free memory on C side: mpds_progressMonitor
                geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)
                # Raise error
                err_message = geosclassic.mpds_get_error_message(-err)
                err_message = err_message.replace('\n', '')
                err_msg = f'{fname}: {err_message}'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor, logger=logger)

            # Free memory on C side: mpds_geosClassicOutput
            geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
            geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

            # Free memory on C side: mpds_progressMonitor
            geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

            image.val[ir] = geosclassic_output['image'].val[0]
            nwarning = nwarning + int(geosclassic_output['nwarning'])
            warnings.extend(geosclassic_output['warnings'])

            del(geosclassic_output)

        # Remove duplicated warnings
        warnings = list(np.unique(warnings))

        # Rename variables
        ndigit = geosclassic.MPDS_GEOS_CLASSIC_NB_DIGIT_FOR_REALIZATION_NUMBER
        for j in range(image.nv):
            image.varname[j] = image.varname[j][:-ndigit] + f'{j:0{ndigit}d}'

        # Set geosclassic_output
        geosclassic_output = {'image':image, 'nwarning':nwarning, 'warnings':warnings}

    if geosclassic_output is not None and mask is not None and add_data_point_to_mask:
        # Remove the value out of the original mask (using its copy see above)
        geosclassic_output['image'].val[:, mask_original==0.0] = np.nan

    if verbose > 1 and geosclassic_output:
        if logger:
            logger.info(f'{fname}: Geos-Classic run complete')
        else:
            print(f'{fname}: Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose > 0 and geosclassic_output and geosclassic_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({geosclassic_output['nwarning']} times in all):")
            for i, warning_message in enumerate(geosclassic_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

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
        aggregate_data_op=None,
        aggregate_data_op_kwargs=None,
        aggregate_data_ineqMin_op='max',
        aggregate_data_ineqMin_op_kwargs=None,
        aggregate_data_ineqMax_op='min',
        aggregate_data_ineqMax_op_kwargs=None,
        mask=None,
        add_data_point_to_mask=True,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        nGibbsSamplerPathMin=50,
        nGibbsSamplerPathMax=200,
        seed=None,
        outputReportFile=None,
        treat_image_one_by_one=False,
        nproc=None, nthreads_per_proc=None,
        verbose=2, logger=None):
    """
    Computes the same as the function :func:`simulate3D`, using multiprocessing.

    All the parameters are the same as those of the function :func:`simulate3D`,
    except `nthreads` that is replaced by the parameters `nproc` and
    `nthreads_per_proc`, and an extra parameter `treat_image_one_by_one`.

    This function launches multiple processes (based on `multiprocessing`
    package):

    - `nproc` parallel processes using each one `nthreads_per_proc` threads \
    are launched [parallel calls of the function :func:`simulate3D`]
    - the set of realizations (specified by `nreal`) is distributed in a \
    balanced way over the processes
    - in terms of resources, this implies the use of `nproc*nthreads_per_proc` \
    cpu(s)

    See function :func:`simulate3D`.

    **Parameters (new)**
    --------------------
    nproc : int, optional
        number of process(es); by default (`None`):
        `nproc` is set to `min(nmax-1, nreal)` (but at least 1), where nmax is
        the total number of cpu(s) of the system (retrieved by
        `multiprocessing.cpu_count()`)

    nthreads_per_proc : int, optional
        number of thread(s) per process (should be > 0); by default (`None`):
        `nthreads_per_proc` is automatically computed as the maximal integer
        (but at least 1) such that `nproc*nthreads_per_proc <= nmax-1`, where
        nmax is the total number of cpu(s) of the system (retrieved by
        `multiprocessing.cpu_count()`)

    treat_image_one_by_one : bool, default: False
        keyword argument passed to the function :func:`geone.img.gatherImages`:

        - if `True`: images (result of each process) are gathered one by one, \
        i.e. the variables of each image are inserted in an output image one by \
        one and removed from the source (slower, may save memory)
        - if `False`: images (result of each process) are gathered at once, \
        i.e. the variables of all images are inserted in an output image at once, \
        and then removed (faster)
    """
    fname = 'simulate3D_mp'

    if logger:
        logger.warning(f'{fname}: deprecated function, use function `simulate` instead')
    else:
        print(f'{fname}: WARNING: deprecated function, use function `simulate` instead')

    # Set number of process(es): nproc
    if nproc is None:
        nproc = max(min(multiprocessing.cpu_count()-1, nreal), 1)
    else:
        nproc_tmp = nproc
        nproc = max(min(int(nproc), nreal), 1)
        if verbose > 1 and nproc != nproc_tmp:
            if logger:
                logger.info(f'{fname}: number of process(es) has been changed (now: nproc={nproc})')
            else:
                print(f'{fname}: number of process(es) has been changed (now: nproc={nproc})')

    # Set number of threads per process: nth
    if nthreads_per_proc is None:
        nth = max(int(np.floor((multiprocessing.cpu_count()-1) / nproc)), 1)
    else:
        nth = max(int(nthreads_per_proc), 1)
        if verbose > 1 and nth != nthreads_per_proc:
            if logger:
                logger.info(f'{fname}: number of threads per process has been changed (now: nthreads_per_proc={nth})')
            else:
                print(f'{fname}: number of threads per process has been changed (now: nthreads_per_proc={nth})')

    if verbose > 0 and nproc * nth > multiprocessing.cpu_count():
        if logger:
            logger.warning(f'{fname}: total number of cpu(s) used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: total number of cpu(s) used will exceed number of cpu(s) of the system...')

    # Set the distribution of the realizations over the processes
    # Condider the Euclidean division of nreal by nproc:
    #     nreal = q * nproc + r, with 0 <= r < nproc
    # Then, (q+1) realizations will be done on process 0, 1, ..., r-1, and q realization on process r, ..., nproc-1
    # Define the list real_index_proc of length (nproc+1) such that
    #   real_index_proc[i], ..., real_index_proc[i+1] - 1 : are the realization indices run on process i
    q, r = np.divmod(nreal, nproc)
    real_index_proc = [i*q + min(i, r) for i in range(nproc+1)]

    if verbose > 1:
        if logger:
            logger.info('{}: Geos-Classic running on {} process(es)... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, nproc, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        else:
            print('{}: Geos-Classic running on {} process(es)... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, nproc, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            sys.stdout.flush()
            sys.stdout.flush() # twice!, so that the previous print is flushed before launching geos-classic...

    # Prepare seed
    if seed is None:
        seed = np.random.randint(1, 1000000)
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
        verbose_p = 0
        # if i==0:
        #     verbose_p = min(verbose, 1) # allow to print warnings for process i
        # else:
        #     verbose_p = 0
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
                aggregate_data_op,
                aggregate_data_op_kwargs,
                aggregate_data_ineqMin_op,
                aggregate_data_ineqMin_op_kwargs,
                aggregate_data_ineqMax_op,
                aggregate_data_ineqMax_op_kwargs,
                mask,
                add_data_point_to_mask,
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                nGibbsSamplerPathMin,
                nGibbsSamplerPathMax,
                seed_p,                      # seed (adjusted)
                outputReportFile_p,          # outputReportFile (adjusted)
                nth,                         # nthreads
                verbose_p),                  # verbose (adjusted)
                kwds={'logger':logger})
            )

    # Properly end working process
    pool.close() # Prevents any more tasks from being submitted to the pool,
    pool.join()  # then, wait for the worker processes to exit.

    # Get result from each process
    geosclassic_output_proc = [p.get() for p in out_pool]

    if np.any([out is None for out in geosclassic_output_proc]):
        return None

    # Gather results from every process
    # image
    image = []
    for out in geosclassic_output_proc:
        if out['image'] is not None:
            image.append(out['image'])
            del(out['image'])
    if len(image) == 0:
        image = None
    # Gather images and adjust variable names
    all_image = img.gatherImages(image, keep_varname=True, rem_var_from_source=True, treat_image_one_by_one=treat_image_one_by_one, logger=logger)
    ndigit = geosclassic.MPDS_GEOS_CLASSIC_NB_DIGIT_FOR_REALIZATION_NUMBER
    for j in range(all_image.nv):
        all_image.varname[j] = all_image.varname[j][:-ndigit] + f'{j:0{ndigit}d}'

    # nwarning
    nwarning = np.sum([out['nwarning'] for out in geosclassic_output_proc])
    # warnings
    warnings = list(np.unique(np.hstack([out['warnings'] for out in geosclassic_output_proc])))

    geosclassic_output = {'image':all_image, 'nwarning':nwarning, 'warnings':warnings}

    if verbose > 1 and geosclassic_output:
        if logger:
            logger.info(f'{fname}: Geos-Classic run complete (all process(es))')
        else:
            print(f'{fname}: Geos-Classic run complete (all process(es))')

    # Show (print) encountered warnings
    if verbose > 0 and geosclassic_output and geosclassic_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({geosclassic_output['nwarning']} times in all):")
            for i, warning_message in enumerate(geosclassic_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def estimate1D(
        cov_model,
        dimension, spacing=1.0, origin=0.0,
        method='simple_kriging',
        mean=None, var=None,
        x=None, v=None,
        aggregate_data_op=None,
        aggregate_data_op_kwargs=None,
        mask=None,
        add_data_point_to_mask=True,
        use_unique_neighborhood=False,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2,
        logger=None):
    """
    Computes kriging estimates and standard deviations in 1D.

    Interpolation takes place in (center of) grid cells, based on simple or
    ordinary kriging.

    Parameters
    ----------
    cov_model : :class:`geone.covModel.CovModel1D`
        covariance model in 1D

    dimension : int
        `dimension=nx`, number of cells in the 1D simulation grid

    spacing : float, default: 1.0
        `spacing=sx`, cell size

    origin : float, default: 0.0
        `origin=ox`, origin of the 1D simulation grid (left border)

    method : str {'simple_kriging', 'ordinary_kriging'}, default: 'simple_kriging'
        type of kriging

    mean : function (callable), or array-like of floats, or float, optional
        kriging mean value:

        - if a function: function of one argument (xi) that returns the mean at \
        location xi
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), mean values at grid cells (for \
        non-stationary mean)
        - if a float: same mean value at every grid cell
        - by default (`None`): the mean of data value (`v`) (0.0 if no data) is \
        considered at every grid cell

    var : function (callable), or array-like of floats, or float, optional
        kriging variance value:

        - if a function: function of one argument (xi) that returns the variance \
        at location xi
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), variance values at grid cells (for \
        non-stationary variance)
        - if a float: same variance value at every grid cell
        - by default (`None`): not used (use of covariance model only)

    x : 1D array-like of floats, optional
        data points locations (float coordinates); note: if one point, a float
        is accepted

    v : 1D array-like of floats, optional
        data values at `x` (`v[i]` is the data value at `x[i]`), array of same
        length as `x` (or float if one point)

    aggregate_data_op : str {'krige', 'min', 'max', 'mean', 'quantile', \
                        'most_freq', 'random'}, optional
        operation used to aggregate data points falling in the same grid cells

        - if `aggregate_data_op='krige'`: function :func:`geone.covModel.krige` is used \
        with the covariance model `cov_model` given in arguments, as well as \
        the parameters `use_unique_neighborhood`, `nneighborMax` given in \
        arguments unless they are given in `aggregate_data_op_kwargs`
        - if `aggregate_data_op='most_freq'`: most frequent value is selected \
        (smallest one if more than one value with the maximal frequence)
        - if `aggregate_data_op='random'`: value from a random point is selected
        - otherwise: the function `numpy.<aggregate_data_op>` is used with the \
        additional parameters given by `aggregate_data_op_kwargs`, note that, e.g. \
        `aggregate_data_op='quantile'` requires the additional parameter \
        `q=<quantile_to_compute>`

        By default: if covariance model has stationary ranges and weight (sill),
        `aggregate_data_op='krige'` is used, otherwise `aggregate_data_op='mean'`

    aggregate_data_op_kwargs : dict, optional
        keyword arguments to be passed to `geone.covModel.krige` or
        `numpy.<aggregate_data_op>`, according to the parameter
        `aggregate_data_op`

    mask : array-like, optional
        mask value at grid cells (value 1 for simulated cells, value 0 for not
        simulated cells); the size of the array must be equal to the number of
        grid cells (the array is reshaped if needed)

    add_data_point_to_mask : bool, default: True
        - if `True`: any grid cell that contains a data point is added to (the \
        simulated part of) the mask (if present), i.e. mask value at those cells \
        are set to 1; at the end of the computation the "new mask cells" are \
        removed (by setting a missing value (`numpy.nan`) for the variable out of \
        the original mask)
        - if `False`: original mask is kept as given in input, and data point \
        falling out of (the simulated part of) the mask (if present) are ignored

    use_unique_neighborhood : bool, default: False
        indicates if a unique neighborhood is used:

        - if `True`: all data points are taken into account for computing \
        estimates and standard deviations; in this case: parameters \
        `searchRadiusRelative`, `nneighborMax`, `searchNeighborhoodSortMode` are \
        not used
        - if `False`: only data points within a search ellipsoid are taken into \
        account for computing estimates and standard deviations (see parameters \
        `searchRadiusRelative`, `nneighborMax`, `searchNeighborhoodSortMode`)

    searchRadiusRelative : float, default: 1.2
        indicates how the search ellipsoid is limited (should be positive): let
        r_i be the ranges of the covariance model along its main axes, when
        estimating/simulating a cell x, a cell y is taken into account iff it is
        within the ellipsoid centered at x of half axes equal to
        `searchRadiusRelative` * r_i; 
        (note that the distances to the central node are computed in the axes 
        sytem supporting the covariance model and accounting for anisotropy given 
        by the ranges)
        note (Geos-Classic library): if a range r_i is non-stationary over the grid, 
        its maximal value over the grid is considered;

    nneighborMax : int, default: 12
        maximum number of cells retrieved from the search ellipsoid (when
        estimating/simulating a cell), `nneighborMax=-1` for unlimited

    searchNeighborhoodSortMode : int, optional
        indicates how to sort the search neighboorhood cells (neighbors); they
        are sorted in increasing order according to:

        - `searchNeighborhoodSortMode=0`: distance in the usual axes system
        - `searchNeighborhoodSortMode=1`: distance in the axes sytem supporting \
        the covariance model and accounting for anisotropy given by the ranges
        - `searchNeighborhoodSortMode=2`: minus the evaluation of the covariance \
        model

        Notes:

        - if the covariance model has any non-stationary parameter, then \
        `searchNeighborhoodSortMode=2` is not allowed
        - if the covariance model has any non-stationary range or non-stationary \
        angle and `searchNeighborhoodSortMode=1`: "maximal ranges" (adapted to \
        direction from the central cell) are used to compute distance for sorting \
        the neighbors

        By default (`None`): the greatest possible value is used (i.e. 2 for
        stationary covariance model, or 1 otherwise)

    outputReportFile : str, optional
        name of the report file (if desired in output); by default (`None`): no
        report file

    nthreads : int, default: -1
        number of thread(s) to use for "GeosClassicSim" C program;
        `nthreads = -n <= 0`: maximal number of threads of the system except n
        (but at least 1)

    verbose : int, default: 2
        verbose mode, higher implies more printing (info):

        - 0: no display
        - 1: warnings
        - 2: warnings + basic info
        - 3 (or >2): all information

        note that if an error occurred, it is raised
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    geosclassic_output : dict
        geosclassic output in python, dictionary

        {'image':image, 'nwarning':nwarning, 'warnings':warnings}

        with:

        - image : :class:`geone.img.Img`
            output image, with `image.nv=2` variables (estimates and standard
            deviations);
            note: `image=None` if `mpds_geosClassicOutput->outputImage=NULL`

        - nwarning : int
            total number of warning(s) encountered (same warnings can be counted
            several times)

        - warnings : list of strs
            list of distinct warnings encountered (can be empty)
    """
    fname = 'estimate1D'

    if logger:
        logger.warning(f'{fname}: deprecated function, use function `estimate` instead')
    else:
        print(f'{fname}: WARNING: deprecated function, use function `estimate` instead')

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
        err_msg = f'{fname}: `cov_model` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    for el in cov_model.elem:
        # weight
        w = el[1]['w']
        if np.size(w) != 1 and np.size(w) != nxyz:
            err_msg = f"{fname}: `cov_model`: weight ('w') not compatible with simulation grid"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # ranges
        if 'r' in el[1].keys():
            r  = el[1]['r']
            if np.size(r) != 1 and np.size(r) != nxyz:
                err_msg = f"{fname}: `cov_model`: range ('r') not compatible with simulation grid"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

        # additional parameter (s)
        if 's' in el[1].keys():
            s  = el[1]['s']
            if np.size(s) != 1 and np.size(s) != nxyz:
                err_msg = f"{fname}: `cov_model`: parameter ('s') not compatible with simulation grid"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # aggregate_data_op (default)
    if aggregate_data_op is None:
        if not cov_model.is_weight_stationary() or not cov_model.is_range_stationary():
            aggregate_data_op = 'mean'
        else:
            aggregate_data_op = 'krige'

    if aggregate_data_op_kwargs is None:
        aggregate_data_op_kwargs = {}

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    if method == 'simple_kriging':
        computationMode = 1
    elif method == 'ordinary_kriging':
        computationMode = 0
    else:
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # If unique neighborhood is used, set searchRadiusRelative to -1
    #    (and initialize nneighborMax, searchNeighborhoodSortMode (unused))
    if use_unique_neighborhood:
        searchRadiusRelative = -1.0
        nneighborMax = 1
        searchNeighborhoodSortMode = 0

    else:
        # Check parameters - searchRadiusRelative
        if searchRadiusRelative < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
            err_msg = f'{fname}: `searchRadiusRelative` too small (should be at least {geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN})'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # Check parameters - nneighborMax
        if nneighborMax != -1 and nneighborMax <= 0:
            err_msg = f'{fname}: `nneighborMax` should be greater than 0 or equal to -1 (unlimited)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # Check parameters - searchNeighborhoodSortMode
        if searchNeighborhoodSortMode is None:
            # set greatest possible value
            if cov_model.is_stationary():
                searchNeighborhoodSortMode = 2
            else:
                searchNeighborhoodSortMode = 1
        else:
            if searchNeighborhoodSortMode == 2:
                if not cov_model.is_stationary():
                    err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

        # if searchNeighborhoodSortMode is None:
        #     # set greatest possible value
        #     if cov_model.is_stationary():
        #         searchNeighborhoodSortMode = 2
        #     elif cov_model.is_orientation_stationary() and cov_model.is_range_stationary():
        #         searchNeighborhoodSortMode = 1
        #     else:
        #         searchNeighborhoodSortMode = 0
        # else:
        #     if searchNeighborhoodSortMode == 2:
        #         if not cov_model.is_stationary():
        #             err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
        #             if logger: logger.error(err_msg)
        #             raise GeosclassicinterfaceError(err_msg)
        #     elif searchNeighborhoodSortMode == 1:
        #         if not cov_model.is_orientation_stationary() or not cov_model.is_range_stationary():
        #             err_msg = f'{fname}: `searchNeighborhoodSortMode=1` not allowed with non-stationary range or non-stationary orientation in covariance model'
        #             if logger: logger.error(err_msg)
        #             raise GeosclassicinterfaceError(err_msg)

    # Preparation of data points
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 1) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Check parameters - mean
    mean_x = mean
    if mean is not None:
        # if method == 'ordinary_kriging':
        #     err_msg = f'{fname}: specifying `mean` not allowed with ordinary kriging'
        #     if logger: logger.error(err_msg)
        #     raise GeosclassicinterfaceError(err_msg)

        if callable(mean):
            if x is not None:
                mean_x = mean(x[:, 0])
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            mean = mean(xi) # replace function 'mean' by its evaluation on the grid
        else:
            mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if mean.size == 1:
                if x is not None:
                    mean_x = mean
            elif mean.size == nxyz:
                # mean = mean.reshape(nx)
                if x is not None:
                    mean_x = img.Img_interp_func(img.Img(nx, 1, 1, sx, 1., 1., ox, 0., 0., nv=1, val=mean, logger=logger), iy=0, iz=0, logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `mean` is not valid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # Check parameters - var
    var_x = var
    if var is not None:
        if method == 'ordinary_kriging':
            err_msg = f'{fname}: specifying `var` not allowed with ordinary kriging'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        if callable(var):
            if x is not None:
                var_x = var(x[:, 0])
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            var = var(xi) # replace function 'var' by its evaluation on the grid
        else:
            var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if var.size == 1:
                if x is not None:
                    var_x = var
            elif var.size == nxyz:
                # var = var.reshape(nx)
                if x is not None:
                    var_x = img.Img_interp_func(img.Img(nx, 1, 1, sx, 1., 1., ox, 0., 0., nv=1, val=var, logger=logger), iy=0, iz=0, logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `var` is not valid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # data points: x, v
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        if aggregate_data_op == 'krige':
            if not cov_model.is_weight_stationary() or not cov_model.is_range_stationary():
                err_msg = f"{fname}: covariance model with non-stationary weight or range cannot be used with `aggregate_data_op`='{aggregate_data_op}'"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            cov_model_agg = cov_model
            # Get grid cell with at least one data point:
            # x_agg: 2D array, each row contains the coordinates of the center of such cell
            im_tmp = img.imageFromPoints(x, values=None, varname=None,
                                         nx=nx, sx=sx, ox=ox,
                                         indicator_var=True, 
                                         count_var=False,
                                         logger=logger)
            ind_agg = np.where(im_tmp.val[0])
            if len(ind_agg[0]) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            x_agg = im_tmp.xx()[ind_agg].reshape(-1, 1)
            # x_agg = im_tmp.xx()[*ind_agg].reshape(-1, 1)
            ind_agg = ind_agg[2:] # remove index along z and y axes
            del(im_tmp)
            # Compute kriging estimate (v_agg) and kriging std (v_agg_std) at x_agg
            if mean is not None and mean.size > 1:
                mean_x_agg = mean[ind_agg]
                # mean_x_agg = mean[*ind_agg]
            else:
                mean_x_agg = mean
            if var is not None and var.size > 1:
                var_x_agg = var[ind_agg]
                # var_x_agg = var[*ind_agg]
            else:
                var_x_agg = var
            # Set parameters `use_unique_neighborhood` and `nneighborMax`
            # from the arguments if not given in `aggregate_data_op_kwargs`
            if 'use_unique_neighborhood' not in aggregate_data_op_kwargs.keys():
                aggregate_data_op_kwargs['use_unique_neighborhood'] = use_unique_neighborhood
            if 'nneighborMax' not in aggregate_data_op_kwargs.keys():
                aggregate_data_op_kwargs['nneighborMax'] = nneighborMax
            try:
                v_agg, v_agg_std = gcm.krige(x, v, x_agg, cov_model_agg, method=method,
                                             mean_x=mean_x, mean_xu=mean_x_agg,
                                             var_x=var_x, var_xu=var_x_agg,
                                             verbose=0, logger=logger,
                                             **aggregate_data_op_kwargs)
            except Exception as exc:
                err_msg = f'{fname}: kriging error'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

            xx_agg = x_agg[:, 0]
            yy_agg = np.ones_like(xx_agg) * oy + 0.5 * sy
            zz_agg = np.ones_like(xx_agg) * oz + 0.5 * sz
            
            dataPointSet.append(
                PointSet(npt=v_agg.shape[0], nv=5, val=np.array((xx_agg, yy_agg, zz_agg, v_agg, v_agg_std)), varname=['X', 'Y', 'Z', varname, f'{varname}_err_std'])
                )

        else:
            # Aggregate data on grid cell by using the given operation
            xx = x[:, 0]
            yy = np.ones_like(xx) * oy + 0.5 * sy
            zz = np.ones_like(xx) * oz + 0.5 * sz
            try:
                xx_agg, yy_agg, zz_agg, v_agg = img.aggregateDataPointsWrtGrid(
                                                    xx, yy, zz, v,
                                                    nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                    op=aggregate_data_op, 
                                                    logger=logger,
                                                    **aggregate_data_op_kwargs)
            except Exception as exc:
                err_msg = f"{fname}: data aggregation (`aggregate_data_op='{aggregate_data_op}'`) failed"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

            if len(xx_agg) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            dataPointSet.append(
                PointSet(npt=v_agg.shape[0], nv=4, val=np.array((xx_agg, yy_agg, zz_agg, v_agg)), varname=['X', 'Y', 'Z', varname])
                )
        # dataPointSet.append(
        #     PointSet(npt=v_agg.shape[0], nv=4, val=np.array((xx_agg, yy_agg, zz_agg, v_agg)), varname=['X', 'Y', 'Z', varname])
        #     )

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            err_msg = f'{fname}: `mask` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if mask is not None and add_data_point_to_mask:
        # Make a copy of the original mask, to remove value in added mask cell at the end
        mask_original = np.copy(mask)
        # Add cell to mask if needed
        pts = np.zeros((0,3))
        if x is not None:
            pts = np.vstack((pts, np.array((xx_agg, yy_agg, zz_agg)).T))
        if pts.shape[0]:
            im_tmp = img.imageFromPoints(pts,
                        nx=nx, ny=ny, nz=nz,
                        sx=sx, sy=sy, sz=sz,
                        ox=ox, oy=oy, oz=oz,
                        indicator_var=True,
                        logger=logger)
            mask = 1.0*np.any((im_tmp.val[0], mask), axis=0)
            del(im_tmp)
        del(pts)

    # --- Fill mpds_geosClassicInput structure (C)
    try:
        mpds_geosClassicInput = fill_mpds_geosClassicInput(
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
                0.0,
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                0,
                0,
                0,
                0,
                logger=logger)
        
    except Exception as exc:
        err_msg = f'{fname}: cannot fill mpds_geosClassicInput C structure'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg) from exc

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

    if verbose > 0 and nth > os.cpu_count():
        if logger:
            logger.warning(f'{fname}: number of threads used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: number of threads used will exceed number of cpu(s) of the system...')

    if verbose > 1:
        if logger:
            logger.info('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        else:
            print('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            sys.stdout.flush()
            sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

    # Free memory on C side: mpds_geosClassicInput
    geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
    geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)

    if err:
        # Free memory on C side: mpds_geosClassicOutput
        geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
        geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)
        # Free memory on C side: mpds_progressMonitor
        geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)
        # Raise error
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        err_msg = f'{fname}: {err_message}'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor, logger=logger)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if geosclassic_output is not None and x is not None and aggregate_data_op == 'krige':
        # Set kriging standard deviation at grid cell containing a data
        geosclassic_output['image'].val[1, 0, 0, ind_agg[0]] = v_agg_std
        # geosclassic_output['image'].val[1, 0, 0, *ind_agg] = v_agg_std

    if geosclassic_output is not None and mask is not None and add_data_point_to_mask:
        # Remove the value out of the original mask (using its copy see above)
        geosclassic_output['image'].val[:, mask_original==0.0] = np.nan

    if verbose > 1 and geosclassic_output:
        if logger:
            logger.info(f'{fname}: Geos-Classic run complete')
        else:
            print(f'{fname}: Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose > 0 and geosclassic_output and geosclassic_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({geosclassic_output['nwarning']} times in all):")
            for i, warning_message in enumerate(geosclassic_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def estimate2D(
        cov_model,
        dimension, spacing=(1.0, 1.0), origin=(0.0, 0.0),
        method='simple_kriging',
        mean=None, var=None,
        x=None, v=None,
        aggregate_data_op=None,
        aggregate_data_op_kwargs=None,
        mask=None,
        add_data_point_to_mask=True,
        use_unique_neighborhood=False,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2,
        logger=None):
    """
    Computes kriging estimates and standard deviations in 2D.

    Interpolation takes place in (center of) grid cells, based on simple or
    ordinary kriging.

    Parameters
    ----------
    cov_model : :class:`geone.covModel.CovModel2D`
        covariance model in 2D

    dimension : 2-tuple of ints
        `dimension=(nx, ny)`, number of cells in the 2D simulation grid along
        each axis

    spacing : 2-tuple of floats, default: (1.0, 1.0)
        `spacing=(sx, sy)`, cell size along each axis

    origin : 2-tuple of floats, default: (0.0, 0.0)
        `origin=(ox, oy)`, origin of the 2D simulation grid (lower-left corner)

    method : str {'simple_kriging', 'ordinary_kriging'}, default: 'simple_kriging'
        type of kriging

    mean : function (callable), or array-like of floats, or float, optional
        kriging mean value:

        - if a function: function of two arguments (xi, yi) that returns the mean \
        at location (xi, yi)
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), mean values at grid cells (for \
        non-stationary mean)
        - if a float: same mean value at every grid cell
        - by default (`None`): the mean of data value (`v`) (0.0 if no data) is \
        considered at every grid cell

    var : function (callable), or array-like of floats, or float, optional
        kriging variance value:

        - if a function: function of two arguments (xi, yi) that returns the \
        variance at location (xi, yi)
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), variance values at grid cells (for \
        non-stationary variance)
        - if a float: same variance value at every grid cell
        - by default (`None`): not used (use of covariance model only)

    x : 2D array of floats of shape (n, 2), optional
        data points locations, with n the number of data points, each row of `x`
        is the float coordinates of one data point; note: if n=1, a 1D array of
        shape (2,) is accepted

    v : 1D array of floats of shape (n,), optional
        data values at `x` (`v[i]` is the data value at `x[i]`)

    aggregate_data_op : str {'krige', 'min', 'max', 'mean', 'quantile', \
                        'most_freq', 'random'}, optional
        operation used to aggregate data points falling in the same grid cells

        - if `aggregate_data_op='krige'`: function :func:`geone.covModel.krige` is used \
        with the covariance model `cov_model` given in arguments, as well as \
        the parameters `use_unique_neighborhood`, `nneighborMax` given in \
        arguments unless they are given in `aggregate_data_op_kwargs`
        - if `aggregate_data_op='most_freq'`: most frequent value is selected \
        (smallest one if more than one value with the maximal frequence)
        - if `aggregate_data_op='random'`: value from a random point is selected
        - otherwise: the function `numpy.<aggregate_data_op>` is used with the \
        additional parameters given by `aggregate_data_op_kwargs`, note that, e.g. \
        `aggregate_data_op='quantile'` requires the additional parameter \
        `q=<quantile_to_compute>`

        By default: if covariance model has stationary ranges and weight (sill),
        `aggregate_data_op='krige'` is used, otherwise `aggregate_data_op='mean'`

    aggregate_data_op_kwargs : dict, optional
        keyword arguments to be passed to `geone.covModel.krige` or
        `numpy.<aggregate_data_op>`, according to the parameter
        `aggregate_data_op`

    mask : array-like, optional
        mask value at grid cells (value 1 for simulated cells, value 0 for not
        simulated cells); the size of the array must be equal to the number of
        grid cells (the array is reshaped if needed)

    add_data_point_to_mask : bool, default: True
        - if `True`: any grid cell that contains a data point is added to (the \
        simulated part of) the mask (if present), i.e. mask value at those cells \
        are set to 1; at the end of the computation the "new mask cells" are \
        removed (by setting a missing value (`numpy.nan`) for the variable out of \
        the original mask)
        - if `False`: original mask is kept as given in input, and data point \
        falling out of (the simulated part of) the mask (if present) are ignored

    use_unique_neighborhood : bool, default: False
        indicates if a unique neighborhood is used:

        - if `True`: all data points are taken into account for computing \
        estimates and standard deviations; in this case: parameters \
        `searchRadiusRelative`, `nneighborMax`, `searchNeighborhoodSortMode` are \
        not used
        - if `False`: only data points within a search ellipsoid are taken into \
        account for computing estimates and standard deviations (see parameters \
        `searchRadiusRelative`, `nneighborMax`, `searchNeighborhoodSortMode`)

    searchRadiusRelative : float, default: 1.2
        indicates how the search ellipsoid is limited (should be positive): let
        r_i be the ranges of the covariance model along its main axes, when
        estimating/simulating a cell x, a cell y is taken into account iff it is
        within the ellipsoid centered at x of half axes equal to
        `searchRadiusRelative` * r_i;
        note: if a range r_i is non-stationary over the grid, its maximal value
        over the grid is considered

    nneighborMax : int, default: 12
        maximum number of cells retrieved from the search ellipsoid (when
        estimating/simulating a cell), `nneighborMax=-1` for unlimited

    searchNeighborhoodSortMode : int, optional
        indicates how to sort the search neighboorhood cells (neighbors); they
        are sorted in increasing order according to:

        - `searchNeighborhoodSortMode=0`: distance in the usual axes system
        - `searchNeighborhoodSortMode=1`: distance in the axes sytem supporting \
        the covariance model and accounting for anisotropy given by the ranges
        - `searchNeighborhoodSortMode=2`: minus the evaluation of the covariance \
        model

        Notes:
        - if the covariance model has any non-stationary parameter, then \
        `searchNeighborhoodSortMode=2` is not allowed
        - if the covariance model has any non-stationary range or non-stationary \
        angle and `searchNeighborhoodSortMode=1`: "maximal ranges" (adapted to \
        direction from the central cell) are used to compute distance for sorting \
        the neighbors

        By default (`None`): the greatest possible value is used (i.e. 2 for
        stationary covariance model, or 1 otherwise)

    outputReportFile : str, optional
        name of the report file (if desired in output); by default (`None`): no
        report file

    nthreads : int, default: -1
        number of thread(s) to use for "GeosClassicSim" C program;
        `nthreads = -n <= 0`: maximal number of threads of the system except n
        (but at least 1)

    verbose : int, default: 2
        verbose mode, higher implies more printing (info):

        - 0: no display
        - 1: warnings
        - 2: warnings + basic info
        - 3 (or >2): all information

        note that if an error occurred, it is raised
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    geosclassic_output : dict
        geosclassic output in python, dictionary

        {'image':image, 'nwarning':nwarning, 'warnings':warnings}

        with:

        - image : :class:`geone.img.Img`
            output image, with `image.nv=2` variables (estimates and standard
            deviations);
            note: `image=None` if `mpds_geosClassicOutput->outputImage=NULL`

        - nwarning : int
            total number of warning(s) encountered (same warnings can be counted
            several times)

        - warnings : list of strs
            list of distinct warnings encountered (can be empty)
    """
    fname = 'estimate2D'

    if logger:
        logger.warning(f'{fname}: deprecated function, use function `estimate` instead')
    else:
        print(f'{fname}: WARNING: deprecated function, use function `estimate` instead')

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
        err_msg = f'{fname}: `cov_model` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    for el in cov_model.elem:
        # weight
        w = el[1]['w']
        if np.size(w) != 1 and np.size(w) != nxyz:
            err_msg = f"{fname}: `cov_model`: weight ('w') not compatible with simulation grid"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # ranges
        if 'r' in el[1].keys():
            for r in el[1]['r']:
                if np.size(r) != 1 and np.size(r) != nxyz:
                    err_msg = f"{fname}: `cov_model`: range ('r') not compatible with simulation grid"
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

        # additional parameter (s)
        if 's' in el[1].keys():
            s  = el[1]['s']
            if np.size(s) != 1 and np.size(s) != nxyz:
                err_msg = f"{fname}: `cov_model`: parameter ('s') not compatible with simulation grid"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # alpha
    angle = cov_model.alpha
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        err_msg = f"{fname}: `cov_model`: angle ('alpha') not compatible with simulation grid"
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # aggregate_data_op (default)
    if aggregate_data_op is None:
        if not cov_model.is_weight_stationary() or not cov_model.is_range_stationary():
            aggregate_data_op = 'mean'
        else:
            aggregate_data_op = 'krige'

    if aggregate_data_op_kwargs is None:
        aggregate_data_op_kwargs = {}

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    if method == 'simple_kriging':
        computationMode = 1
    elif method == 'ordinary_kriging':
        computationMode = 0
    else:
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # If unique neighborhood is used, set searchRadiusRelative to -1
    #    (and initialize nneighborMax, searchNeighborhoodSortMode (unused))
    if use_unique_neighborhood:
        searchRadiusRelative = -1.0
        nneighborMax = 1
        searchNeighborhoodSortMode = 0

    else:
        # Check parameters - searchRadiusRelative
        if searchRadiusRelative < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
            err_msg = f'{fname}: `searchRadiusRelative` too small (should be at least {geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN})'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # Check parameters - nneighborMax
        if nneighborMax != -1 and nneighborMax <= 0:
            err_msg = f'{fname}: `nneighborMax` should be greater than 0 or equal to -1 (unlimited)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # Check parameters - searchNeighborhoodSortMode
        if searchNeighborhoodSortMode is None:
            # set greatest possible value
            if cov_model.is_stationary():
                searchNeighborhoodSortMode = 2
            else:
                searchNeighborhoodSortMode = 1
        else:
            if searchNeighborhoodSortMode == 2:
                if not cov_model.is_stationary():
                    err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

        # if searchNeighborhoodSortMode is None:
        #     # set greatest possible value
        #     if cov_model.is_stationary():
        #         searchNeighborhoodSortMode = 2
        #     elif cov_model.is_orientation_stationary() and cov_model.is_range_stationary():
        #         searchNeighborhoodSortMode = 1
        #     else:
        #         searchNeighborhoodSortMode = 0
        # else:
        #     if searchNeighborhoodSortMode == 2:
        #         if not cov_model.is_stationary():
        #             err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
        #             if logger: logger.error(err_msg)
        #             raise GeosclassicinterfaceError(err_msg)
        #     elif searchNeighborhoodSortMode == 1:
        #         if not cov_model.is_orientation_stationary() or not cov_model.is_range_stationary():
        #             err_msg = f'{fname}: `searchNeighborhoodSortMode=1` not allowed with non-stationary range or non-stationary orientation in covariance model'
        #             if logger: logger.error(err_msg)
        #             raise GeosclassicinterfaceError(err_msg)

    # Preparation of data points
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 2) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Check parameters - mean
    mean_x = mean
    if mean is not None:
        # if method == 'ordinary_kriging':
        #     err_msg = f'{fname}: specifying `mean` not allowed with ordinary kriging'
        #     if logger: logger.error(err_msg)
        #     raise GeosclassicinterfaceError(err_msg)

        if callable(mean):
            if x is not None:
                mean_x = mean(x[:, 0], x[:, 1])
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            yyi, xxi = np.meshgrid(yi, xi, indexing='ij')
            mean = mean(xxi, yyi) # replace function 'mean' by its evaluation on the grid
        else:
            mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if mean.size == 1:
                if x is not None:
                    mean_x = mean
            elif mean.size == nxyz:
                mean = mean.reshape(ny, nx)
                if x is not None:
                    mean_x = img.Img_interp_func(img.Img(nx, ny, 1, sx, sy, 1., ox, oy, 0., nv=1, val=mean, logger=logger), iz=0, logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `mean` is not valid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # Check parameters - var
    var_x = var
    if var is not None:
        if method == 'ordinary_kriging':
            err_msg = f'{fname}: specifying `var` not allowed with ordinary kriging'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        if callable(var):
            if x is not None:
                var_x = var(x[:, 0], x[:, 1])
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            yyi, xxi = np.meshgrid(yi, xi, indexing='ij')
            var = var(xxi, yyi) # replace function 'var' by its evaluation on the grid
        else:
            var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if var.size == 1:
                if x is not None:
                    var_x = var
            elif var.size == nxyz:
                var = var.reshape(ny, nx)
                if x is not None:
                    var_x = img.Img_interp_func(img.Img(nx, ny, 1, sx, sy, 1., ox, oy, 0., nv=1, val=var, logger=logger), iz=0, logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `var` is not valid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # data points: x, v
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        if aggregate_data_op == 'krige':
            if not cov_model.is_weight_stationary() or not cov_model.is_range_stationary():
                err_msg = f"{fname}: covariance model with non-stationary weight or range cannot be used with `aggregate_data_op`='{aggregate_data_op}'"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            if cov_model.is_orientation_stationary():
                cov_model_agg = cov_model
            else:
                cov_model_agg = gcm.copyCovModel(cov_model)
                cov_model_agg.set_alpha(0.0)
            # Get grid cell with at least one data point:
            # x_agg: 2D array, each row contains the coordinates of the center of such cell
            im_tmp = img.imageFromPoints(x, values=None, varname=None,
                                         nx=nx, ny=ny, sx=sx, sy=sy, ox=ox, oy=oy,
                                         indicator_var=True, 
                                         count_var=False,
                                         logger=logger)
            ind_agg = np.where(im_tmp.val[0])
            if len(ind_agg[0]) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            x_agg = np.array((im_tmp.xx()[ind_agg].reshape(-1), im_tmp.yy()[ind_agg].reshape(-1))).T
            # x_agg = np.array((im_tmp.xx()[*ind_agg].reshape(-1), im_tmp.yy()[*ind_agg].reshape(-1))).T
            ind_agg = ind_agg[1:] # remove index along z axis
            del(im_tmp)
            # Compute kriging estimate (v_agg) and kriging std (v_agg_std) at x_agg
            if mean is not None and mean.size > 1:
                mean_x_agg = mean[ind_agg]
                # mean_x_agg = mean[*ind_agg]
            else:
                mean_x_agg = mean
            if var is not None and var.size > 1:
                var_x_agg = var[ind_agg]
                # var_x_agg = var[*ind_agg]
            else:
                var_x_agg = var
            if isinstance(cov_model.alpha, np.ndarray) and cov_model.alpha.size == nxyz:
                alpha_x_agg = cov_model.alpha.reshape(ny, nx)[ind_agg]
                # alpha_x_agg = cov_model.alpha.reshape(ny, nx)[*ind_agg]
            else:
                alpha_x_agg = cov_model.alpha
            # Set parameters `use_unique_neighborhood` and `nneighborMax`
            # from the arguments if not given in `aggregate_data_op_kwargs`
            if 'use_unique_neighborhood' not in aggregate_data_op_kwargs.keys():
                aggregate_data_op_kwargs['use_unique_neighborhood'] = use_unique_neighborhood
            if 'nneighborMax' not in aggregate_data_op_kwargs.keys():
                aggregate_data_op_kwargs['nneighborMax'] = nneighborMax
            try:
                v_agg, v_agg_std = gcm.krige(x, v, x_agg, cov_model_agg, method=method,
                                             mean_x=mean_x, mean_xu=mean_x_agg,
                                             var_x=var_x, var_xu=var_x_agg,
                                             alpha_xu=alpha_x_agg,
                                             verbose=0, logger=logger,
                                             **aggregate_data_op_kwargs)
            except Exception as exc:
                err_msg = f'{fname}: kriging error'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

            xx_agg, yy_agg = x_agg.T
            zz_agg = np.ones_like(xx_agg) * oz + 0.5 * sz
        else:
            # Aggregate data on grid cell by using the given operation
            xx, yy = x.T
            zz = np.ones_like(xx) * oz + 0.5 * sz
            try:
                xx_agg, yy_agg, zz_agg, v_agg = img.aggregateDataPointsWrtGrid(
                                                    xx, yy, zz, v,
                                                    nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                    op=aggregate_data_op, 
                                                    logger=logger,
                                                    **aggregate_data_op_kwargs)
            except Exception as exc:
                err_msg = f"{fname}: data aggregation (`aggregate_data_op='{aggregate_data_op}'`) failed"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

            if len(xx_agg) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

        dataPointSet.append(
            PointSet(npt=v_agg.shape[0], nv=4, val=np.array((xx_agg, yy_agg, zz_agg, v_agg)), varname=['X', 'Y', 'Z', varname])
            )

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            err_msg = f'{fname}: `mask` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if mask is not None and add_data_point_to_mask:
        # Make a copy of the original mask, to remove value in added mask cell at the end
        mask_original = np.copy(mask)
        # Add cell to mask if needed
        pts = np.zeros((0,3))
        if x is not None:
            pts = np.vstack((pts, np.array((xx_agg, yy_agg, zz_agg)).T))
        if pts.shape[0]:
            im_tmp = img.imageFromPoints(pts,
                        nx=nx, ny=ny, nz=nz,
                        sx=sx, sy=sy, sz=sz,
                        ox=ox, oy=oy, oz=oz,
                        indicator_var=True,
                        logger=logger)
            mask = 1.0*np.any((im_tmp.val[0], mask), axis=0)
            del(im_tmp)
        del(pts)

    # --- Fill mpds_geosClassicInput structure (C)
    try:
        mpds_geosClassicInput = fill_mpds_geosClassicInput(
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
                0.0,
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                0,
                0,
                0,
                0,
                logger=logger)
        
    except Exception as exc:
        err_msg = f'{fname}: cannot fill mpds_geosClassicInput C structure'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg) from exc

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

    if verbose > 0 and nth > os.cpu_count():
        if logger:
            logger.warning(f'{fname}: number of threads used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: number of threads used will exceed number of cpu(s) of the system...')

    if verbose > 1:
        if logger:
            logger.info('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        else:
            print('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            sys.stdout.flush()
            sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

    # Free memory on C side: mpds_geosClassicInput
    geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
    geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)

    if err:
        # Free memory on C side: mpds_geosClassicOutput
        geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
        geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)
        # Free memory on C side: mpds_progressMonitor
        geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)
        # Raise error
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        err_msg = f'{fname}: {err_message}'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor, logger=logger)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if geosclassic_output is not None and x is not None and aggregate_data_op == 'krige':
        # Set kriging standard deviation at grid cell containing a data
        geosclassic_output['image'].val[1, 0, ind_agg[0], ind_agg[1]] = v_agg_std
        # geosclassic_output['image'].val[1, 0, *ind_agg] = v_agg_std

    if geosclassic_output is not None and mask is not None and add_data_point_to_mask:
        # Remove the value out of the original mask (using its copy see above)
        geosclassic_output['image'].val[:, mask_original==0.0] = np.nan

    if verbose > 1 and geosclassic_output:
        if logger:
            logger.info(f'{fname}: Geos-Classic run complete')
        else:
            print(f'{fname}: Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose > 0 and geosclassic_output and geosclassic_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({geosclassic_output['nwarning']} times in all):")
            for i, warning_message in enumerate(geosclassic_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

    return geosclassic_output
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def estimate3D(
        cov_model,
        dimension, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
        method='simple_kriging',
        mean=None, var=None,
        x=None, v=None,
        aggregate_data_op=None,
        aggregate_data_op_kwargs=None,
        mask=None,
        add_data_point_to_mask=True,
        use_unique_neighborhood=False,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2,
        logger=None):
    """
    Computes kriging estimates and standard deviations in 3D.

    Interpolation takes place in (center of) grid cells, based on simple or
    ordinary kriging.

    Parameters
    ----------
    cov_model : :class:`geone.covModel.CovModel3D`
        covariance model in 3D

    dimension : 3-tuple of ints
        `dimension=(nx, ny, nz)`, number of cells in the 3D simulation grid along
        each axis

    spacing : 3-tuple of floats, default: (1.0,1.0, 1.0)
        `spacing=(sx, sy, sz)`, cell size along each axis

    origin : 3-tuple of floats, default: (0.0, 0.0, 0.0)
        `origin=(ox, oy, oz)`, origin of the 3D simulation grid (bottom-lower-left
        corner)

    method : str {'simple_kriging', 'ordinary_kriging'}, default: 'simple_kriging'
        type of kriging

    mean : function (callable), or array-like of floats, or float, optional
        kriging mean value:

        - if a function: function of three arguments (xi, yi, zi) that returns \
        the mean at location (xi, yi, zi)
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), mean values at grid cells (for \
        non-stationary mean)
        - if a float: same mean value at every grid cell
        - by default (`None`): the mean of data value (`v`) (0.0 if no data) is \
        considered at every grid cell

    var : function (callable), or array-like of floats, or float, optional
        kriging variance value:

        - if a function: function of three arguments (xi, yi, yi) that returns \
        the variance at location (xi, yi, zi)
        - if array-like: its size must be equal to the number of grid cells \
        (the array is reshaped if needed), variance values at grid cells (for \
        non-stationary variance)
        - if a float: same variance value at every grid cell
        - by default (`None`): not used (use of covariance model only)

    x : 2D array of floats of shape (n, 3), optional
        data points locations, with n the number of data points, each row of `x`
        is the float coordinates of one data point; note: if n=1, a 1D array of
        shape (3,) is accepted

    v : 1D array of floats of shape (n,), optional
        data values at `x` (`v[i]` is the data value at `x[i]`)

    aggregate_data_op : str {'krige', 'min', 'max', 'mean', 'quantile', \
                        'most_freq', 'random'}, optional
        operation used to aggregate data points falling in the same grid cells

        - if `aggregate_data_op='krige'`: function :func:`geone.covModel.krige` is used \
        with the covariance model `cov_model` given in arguments, as well as \
        the parameters `use_unique_neighborhood`, `nneighborMax` given in \
        arguments unless they are given in `aggregate_data_op_kwargs`
        - if `aggregate_data_op='most_freq'`: most frequent value is selected \
        (smallest one if more than one value with the maximal frequence)
        - if `aggregate_data_op='random'`: value from a random point is selected
        - otherwise: the function `numpy.<aggregate_data_op>` is used with the \
        additional parameters given by `aggregate_data_op_kwargs`, note that, e.g. \
        `aggregate_data_op='quantile'` requires the additional parameter \
        `q=<quantile_to_compute>`

        By default: if covariance model has stationary ranges and weight (sill),
        `aggregate_data_op='krige'` is used, otherwise `aggregate_data_op='mean'`

    aggregate_data_op_kwargs : dict, optional
        keyword arguments to be passed to `geone.covModel.krige` or
        `numpy.<aggregate_data_op>`, according to the parameter
        `aggregate_data_op`

    mask : array-like, optional
        mask value at grid cells (value 1 for simulated cells, value 0 for not
        simulated cells); the size of the array must be equal to the number of
        grid cells (the array is reshaped if needed)

    add_data_point_to_mask : bool, default: True
        - if `True`: any grid cell that contains a data point is added to (the \
        simulated part of) the mask (if present), i.e. mask value at those cells \
        are set to 1; at the end of the computation the "new mask cells" are \
        removed (by setting a missing value (`numpy.nan`) for the variable out of \
        the original mask)
        - if `False`: original mask is kept as given in input, and data point \
        falling out of (the simulated part of) the mask (if present) are ignored

    use_unique_neighborhood : bool, default: False
        indicates if a unique neighborhood is used:

        - if `True`: all data points are taken into account for computing \
        estimates and standard deviations; in this case: parameters \
        `searchRadiusRelative`, `nneighborMax`, `searchNeighborhoodSortMode` are \
        not used
        - if `False`: only data points within a search ellipsoid are taken into \
        account for computing estimates and standard deviations (see parameters \
        `searchRadiusRelative`, `nneighborMax`, `searchNeighborhoodSortMode`)

    searchRadiusRelative : float, default: 1.2
        indicates how the search ellipsoid is limited (should be positive): let
        r_i be the ranges of the covariance model along its main axes, when
        estimating/simulating a cell x, a cell y is taken into account iff it is
        within the ellipsoid centered at x of half axes equal to
        `searchRadiusRelative` * r_i;
        note: if a range r_i is non-stationary over the grid, its maximal value
        over the grid is considered

    nneighborMax : int, default: 12
        maximum number of cells retrieved from the search ellipsoid (when
        estimating/simulating a cell), `nneighborMax=-1` for unlimited

    searchNeighborhoodSortMode : int, optional
        indicates how to sort the search neighboorhood cells (neighbors); they
        are sorted in increasing order according to:

        - `searchNeighborhoodSortMode=0`: distance in the usual axes system
        - `searchNeighborhoodSortMode=1`: distance in the axes sytem supporting \
        the covariance model and accounting for anisotropy given by the ranges
        - `searchNeighborhoodSortMode=2`: minus the evaluation of the covariance \
        model

        Notes:
        - if the covariance model has any non-stationary parameter, then \
        `searchNeighborhoodSortMode=2` is not allowed
        - if the covariance model has any non-stationary range or non-stationary \
        angle and `searchNeighborhoodSortMode=1`: "maximal ranges" (adapted to \
        direction from the central cell) are used to compute distance for sorting \
        the neighbors

        By default (`None`): the greatest possible value is used (i.e. 2 for
        stationary covariance model, or 1 otherwise)

    outputReportFile : str, optional
        name of the report file (if desired in output); by default (`None`): no
        report file

    nthreads : int, default: -1
        number of thread(s) to use for "GeosClassicSim" C program;
        `nthreads = -n <= 0`: maximal number of threads of the system except n
        (but at least 1)

    verbose : int, default: 2
        verbose mode, higher implies more printing (info):

        - 0: no display
        - 1: warnings
        - 2: warnings + basic info
        - 3 (or >2): all information

        note that if an error occurred, it is raised
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    geosclassic_output : dict
        geosclassic output in python, dictionary

        {'image':image, 'nwarning':nwarning, 'warnings':warnings}

        with:

        - image : :class:`geone.img.Img`
            output image, with `image.nv=2` variables (estimates and standard
            deviations);
            note: `image=None` if `mpds_geosClassicOutput->outputImage=NULL`

        - nwarning : int
            total number of warning(s) encountered (same warnings can be counted
            several times)

        - warnings : list of strs
            list of distinct warnings encountered (can be empty)
    """
    fname = 'estimate3D'

    if logger:
        logger.warning(f'{fname}: deprecated function, use function `estimate` instead')
    else:
        print(f'{fname}: WARNING: deprecated function, use function `estimate` instead')

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
        err_msg = f'{fname}: `cov_model` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    for el in cov_model.elem:
        # weight
        w = el[1]['w']
        if np.size(w) != 1 and np.size(w) != nxyz:
            err_msg = f"{fname}: `cov_model`: weight ('w') not compatible with simulation grid"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # ranges
        if 'r' in el[1].keys():
            for r in el[1]['r']:
                if np.size(r) != 1 and np.size(r) != nxyz:
                    err_msg = f"{fname}: `cov_model`: range ('r') not compatible with simulation grid"
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

        # additional parameter (s)
        if 's' in el[1].keys():
            s  = el[1]['s']
            if np.size(s) != 1 and np.size(s) != nxyz:
                err_msg = f"{fname}: `cov_model`: parameter ('s') not compatible with simulation grid"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # alpha
    angle = cov_model.alpha
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        err_msg = f"{fname}: `cov_model`: angle ('alpha') not compatible with simulation grid"
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # beta
    angle = cov_model.beta
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        err_msg = f"{fname}: `cov_model`: angle ('beta') not compatible with simulation grid"
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # gamma
    angle = cov_model.gamma
    if np.size(angle) != 1 and np.size(angle) != nxyz:
        err_msg = f"{fname}: `cov_model`: angle ('gamma') not compatible with simulation grid"
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # aggregate_data_op (default)
    if aggregate_data_op is None:
        if not cov_model.is_weight_stationary() or not cov_model.is_range_stationary():
            aggregate_data_op = 'mean'
        else:
            aggregate_data_op = 'krige'

    if aggregate_data_op_kwargs is None:
        aggregate_data_op_kwargs = {}

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    if method == 'simple_kriging':
        computationMode = 1
    elif method == 'ordinary_kriging':
        computationMode = 0
    else:
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # If unique neighborhood is used, set searchRadiusRelative to -1
    #    (and initialize nneighborMax, searchNeighborhoodSortMode (unused))
    if use_unique_neighborhood:
        searchRadiusRelative = -1.0
        nneighborMax = 1
        searchNeighborhoodSortMode = 0

    else:
        # Check parameters - searchRadiusRelative
        if searchRadiusRelative < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
            err_msg = f'{fname}: `searchRadiusRelative` too small (should be at least {geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN})'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # Check parameters - nneighborMax
        if nneighborMax != -1 and nneighborMax <= 0:
            err_msg = f'{fname}: `nneighborMax` should be greater than 0 or equal to -1 (unlimited)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # Check parameters - searchNeighborhoodSortMode
        if searchNeighborhoodSortMode is None:
            # set greatest possible value
            if cov_model.is_stationary():
                searchNeighborhoodSortMode = 2
            else:
                searchNeighborhoodSortMode = 1
        else:
            if searchNeighborhoodSortMode == 2:
                if not cov_model.is_stationary():
                    err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

        # if searchNeighborhoodSortMode is None:
        #     # set greatest possible value
        #     if cov_model.is_stationary():
        #         searchNeighborhoodSortMode = 2
        #     elif cov_model.is_orientation_stationary() and cov_model.is_range_stationary():
        #         searchNeighborhoodSortMode = 1
        #     else:
        #         searchNeighborhoodSortMode = 0
        # else:
        #     if searchNeighborhoodSortMode == 2:
        #         if not cov_model.is_stationary():
        #             err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
        #             if logger: logger.error(err_msg)
        #             raise GeosclassicinterfaceError(err_msg)
        #     elif searchNeighborhoodSortMode == 1:
        #         if not cov_model.is_orientation_stationary() or not cov_model.is_range_stationary():
        #             err_msg = f'{fname}: `searchNeighborhoodSortMode=1` not allowed with non-stationary range or non-stationary orientation in covariance model'
        #             if logger: logger.error(err_msg)
        #             raise GeosclassicinterfaceError(err_msg)

    # Preparation of data points
    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 3) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Check parameters - mean
    mean_x = mean
    if mean is not None:
        # if method == 'ordinary_kriging':
        #     err_msg = f'{fname}: specifying `mean` not allowed with ordinary kriging'
        #     if logger: logger.error(err_msg)
        #     raise GeosclassicinterfaceError(err_msg)

        if callable(mean):
            if x is not None:
                mean_x = mean(x[:, 0], x[:, 1], x[:, 2])
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            zi = oz + sz*(0.5+np.arange(nz)) # z-coordinate of cell center
            zzi, yyi, xxi = np.meshgrid(zi, yi, xi, indexing='ij')
            mean = mean(xxi, yyi, zzi) # replace function 'mean' by its evaluation on the grid
        else:
            mean = np.asarray(mean, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if mean.size == 1:
                if x is not None:
                    mean_x = mean
            elif mean.size == nxyz:
                mean = mean.reshape(nz, ny, nx)
                if x is not None:
                    mean_x = img.Img_interp_func(img.Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv=1, val=mean, logger=logger), logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `mean` is not valid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # Check parameters - var
    var_x = var
    if var is not None:
        if method == 'ordinary_kriging':
            err_msg = f'{fname}: specifying `var` not allowed with ordinary kriging'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        if callable(var):
            if x is not None:
                var_x = var(x[:, 0], x[:, 1], x[:, 2])
            xi = ox + sx*(0.5+np.arange(nx)) # x-coordinate of cell center
            yi = oy + sy*(0.5+np.arange(ny)) # y-coordinate of cell center
            zi = oz + sz*(0.5+np.arange(nz)) # z-coordinate of cell center
            zzi, yyi, xxi = np.meshgrid(zi, yi, xi, indexing='ij')
            var = var(xxi, yyi, zzi) # replace function 'var' by its evaluation on the grid
        else:
            var = np.asarray(var, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
            if var.size == 1:
                if x is not None:
                    var_x = var
            elif var.size == nxyz:
                var = var.reshape(nz, ny, nx)
                if x is not None:
                    var_x = img.Img_interp_func(img.Img(nx, ny, nz, sx, sy, sz, ox, oy, oz, nv=1, val=var, logger=logger), logger=logger)(x)
            else:
                err_msg = f'{fname}: size of `var` is not valid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

    # data points: x, v
    dataPointSet = []

    # data point set from x, v
    if x is not None:
        if aggregate_data_op == 'krige':
            if not cov_model.is_weight_stationary() or not cov_model.is_range_stationary():
                err_msg = f"{fname}: covariance model with non-stationary weight or range cannot be used with `aggregate_data_op`='{aggregate_data_op}'"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            if cov_model.is_orientation_stationary():
                cov_model_agg = cov_model
            else:
                cov_model_agg = gcm.copyCovModel(cov_model)
                cov_model_agg.set_alpha(0.0)
                cov_model_agg.set_beta(0.0)
                cov_model_agg.set_gamma(0.0)
            # Get grid cell with at least one data point:
            # x_agg: 2D array, each row contains the coordinates of the center of such cell
            im_tmp = img.imageFromPoints(x, values=None, varname=None,
                                         nx=nx, ny=ny, nz=nz, sx=sx, sy=sy, sz=sz, ox=ox, oy=oy, oz=oz,
                                         indicator_var=True, 
                                         count_var=False,
                                         logger=logger)
            ind_agg = np.where(im_tmp.val[0])
            if len(ind_agg[0]) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            x_agg = np.array((im_tmp.xx()[ind_agg].reshape(-1), im_tmp.yy()[ind_agg].reshape(-1), im_tmp.zz()[ind_agg].reshape(-1))).T
            # x_agg = np.array((im_tmp.xx()[*ind_agg].reshape(-1), im_tmp.yy()[*ind_agg].reshape(-1), im_tmp.zz()[*ind_agg].reshape(-1))).T
            del(im_tmp)
            # Compute kriging estimate (v_agg) and kriging std (v_agg_std) at x_agg
            if mean is not None and mean.size > 1:
                mean_x_agg = mean[ind_agg]
                # mean_x_agg = mean[*ind_agg]
            else:
                mean_x_agg = mean
            if var is not None and var.size > 1:
                var_x_agg = var[ind_agg]
                # var_x_agg = var[*ind_agg]
            else:
                var_x_agg = var
            if isinstance(cov_model.alpha, np.ndarray) and cov_model.alpha.size == nxyz:
                alpha_x_agg = cov_model.alpha[ind_agg]
                # alpha_x_agg = cov_model.alpha[*ind_agg]
            else:
                alpha_x_agg = cov_model.alpha
            if isinstance(cov_model.beta, np.ndarray) and cov_model.beta.size == nxyz:
                beta_x_agg = cov_model.beta[ind_agg]
                # beta_x_agg = cov_model.beta[*ind_agg]
            else:
                beta_x_agg = cov_model.beta
            if isinstance(cov_model.gamma, np.ndarray) and cov_model.gamma.size == nxyz:
                gamma_x_agg = cov_model.gamma[ind_agg]
                # gamma_x_agg = cov_model.gamma[*ind_agg]
            else:
                gamma_x_agg = cov_model.gamma
            # Set parameters `use_unique_neighborhood` and `nneighborMax`
            # from the arguments if not given in `aggregate_data_op_kwargs`
            if 'use_unique_neighborhood' not in aggregate_data_op_kwargs.keys():
                aggregate_data_op_kwargs['use_unique_neighborhood'] = use_unique_neighborhood
            if 'nneighborMax' not in aggregate_data_op_kwargs.keys():
                aggregate_data_op_kwargs['nneighborMax'] = nneighborMax
            try:
                v_agg, v_agg_std = gcm.krige(x, v, x_agg, cov_model_agg, method=method,
                                             mean_x=mean_x, mean_xu=mean_x_agg,
                                             var_x=var_x, var_xu=var_x_agg,
                                             alpha_xu=alpha_x_agg, beta_xu=beta_x_agg, gamma_xu=gamma_x_agg,
                                             verbose=0, logger=logger,
                                             **aggregate_data_op_kwargs)
            except Exception as exc:
                err_msg = f'{fname}: kriging error'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

            xx_agg, yy_agg, zz_agg = x_agg.T
        else:
            # Aggregate data on grid cell by using the given operation
            xx, yy, zz = x.T
            try:
                xx_agg, yy_agg, zz_agg, v_agg = img.aggregateDataPointsWrtGrid(
                                                    xx, yy, zz, v,
                                                    nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                    op=aggregate_data_op, 
                                                    logger=logger,
                                                    **aggregate_data_op_kwargs)
            except Exception as exc:
                err_msg = f"{fname}: data aggregation (`aggregate_data_op='{aggregate_data_op}'`) failed"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg) from exc

            if len(xx_agg) == 0:
                err_msg = f'{fname}: no data point in grid'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

        dataPointSet.append(
            PointSet(npt=v_agg.shape[0], nv=4, val=np.array((xx_agg, yy_agg, zz_agg, v_agg)), varname=['X', 'Y', 'Z', varname])
            )

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            err_msg = f'{fname}: `mask` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if mask is not None and add_data_point_to_mask:
        # Make a copy of the original mask, to remove value in added mask cell at the end
        mask_original = np.copy(mask)
        # Add cell to mask if needed
        pts = np.zeros((0,3))
        if x is not None:
            pts = np.vstack((pts, np.array((xx_agg, yy_agg, zz_agg)).T))
        if pts.shape[0]:
            im_tmp = img.imageFromPoints(pts,
                        nx=nx, ny=ny, nz=nz,
                        sx=sx, sy=sy, sz=sz,
                        ox=ox, oy=oy, oz=oz,
                        indicator_var=True,
                        logger=logger)
            mask = 1.0*np.any((im_tmp.val[0], mask), axis=0)
            del(im_tmp)
        del(pts)

    # --- Fill mpds_geosClassicInput structure (C)
    try:
        mpds_geosClassicInput = fill_mpds_geosClassicInput(
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
                0.0,
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                0,
                0,
                0,
                0,
                logger=logger)
        
    except Exception as exc:
        err_msg = f'{fname}: cannot fill mpds_geosClassicInput C structure'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg) from exc

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

    if verbose > 0 and nth > os.cpu_count():
        if logger:
            logger.warning(f'{fname}: number of threads used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: number of threads used will exceed number of cpu(s) of the system...')

    if verbose > 1:
        if logger:
            logger.info('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        else:
            print('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            sys.stdout.flush()
            sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicSim(mpds_geosClassicInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

    # Free memory on C side: mpds_geosClassicInput
    geosclassic.MPDSGeosClassicFreeGeosClassicInput(mpds_geosClassicInput)
    geosclassic.free_MPDS_GEOSCLASSICINPUT(mpds_geosClassicInput)

    if err:
        # Free memory on C side: mpds_geosClassicOutput
        geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
        geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)
        # Free memory on C side: mpds_progressMonitor
        geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)
        # Raise error
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        err_msg = f'{fname}: {err_message}'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor, logger=logger)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if geosclassic_output is not None and x is not None and aggregate_data_op == 'krige':
        # Set kriging standard deviation at grid cell containing a data
        geosclassic_output['image'].val[1, ind_agg[0], ind_agg[1], ind_agg[2]] = v_agg_std
        # geosclassic_output['image'].val[1, *ind_agg] = v_agg_std

    if geosclassic_output is not None and mask is not None and add_data_point_to_mask:
        # Remove the value out of the original mask (using its copy see above)
        geosclassic_output['image'].val[:, mask_original==0.0] = np.nan

    if verbose > 1 and geosclassic_output:
        if logger:
            logger.info(f'{fname}: Geos-Classic run complete')
        else:
            print(f'{fname}: Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose > 0 and geosclassic_output and geosclassic_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({geosclassic_output['nwarning']} times in all):")
            for i, warning_message in enumerate(geosclassic_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

    return geosclassic_output
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
        add_data_point_to_mask=True,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        seed=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2,
        logger=None):
    """
    Generates 1D simulations (Sequential Indicator Simulation, SIS).

    A simulation takes place in (center of) grid cells, based on simple or
    ordinary kriging of the indicator variables of the categories.

    Parameters
    ----------
    category_values : 1D array-like
        sequence of category values; let `ncategory` be the number of categories,
        then:

        - if `ncategory=1`: the unique category value given must not be equal to \
        zero; it is used for a binary case with values "unique category value" \
        and 0, where 0 indicates the absence of the considered medium; the \
        conditioning data values should be equal to"unique category value" or 0
        - if `ncategory>=2`: it is used for a multi-category case with given \
        category values (distinct); the conditioning data values should be in the \
        `category_values`

    cov_model_for_category : [sequence of] :class:`geone.covModel.CovModel1D`
        sequence of same length as `category_values` of covariance model in 1D,
        or a unique covariance model in 1D (recycled):
        covariance model for each category

    dimension : int
        `dimension=nx`, number of cells in the 1D simulation grid

    spacing : float, default: 1.0
        `spacing=sx`, cell size

    origin : float, default: 0.0
        `origin=ox`, origin of the 1D simulation grid (left border)

    method : str {'simple_kriging', 'ordinary_kriging'}, default: 'simple_kriging'
        type of kriging

    nreal : int, default: 1
        number of realizations

    probability : array-like of floats, optional
        probability for each category:

        - sequence of same length as `category_values`: \
        probability[i]: probability (proportion, kriging mean value for the \
        indicator variable) for category `category_values[i]`, used for \
        every grid cell
        - array-like of size ncategory * ngrid_cells, where ncategory is the \
        length of `category_values` and ngrid_cells is the number of grid \
        cells (the array is reshaped if needed): first ngrid_cells values are \
        the probabilities (proportions, kriging mean values for the indicator \
        variable) for the first category at grid cells, etc. \
        (for non-stationary probailities / proportions)

        By default (`None`): proportion of each category computed from the
        data values (`v`) are used for every grid cell (uniform probabilities
        if no data point)

        Note: for ordinary kriging (`method='ordinary_kriging'`), it is used for
        case with no neighbor

    x : 1D array-like of floats, optional
        data points locations (float coordinates); note: if one point, a float
        is accepted

    v : 1D array-like of floats, optional
        data values at `x` (`v[i]` is the data value at `x[i]`), array of same
        length as `x` (or float if one point)

    mask : array-like, optional
        mask value at grid cells (value 1 for simulated cells, value 0 for not
        simulated cells); the size of the array must be equal to the number of
        grid cells (the array is reshaped if needed)

    add_data_point_to_mask : bool, default: True
        - if `True`: any grid cell that contains a data point is added to (the \
        simulated part of) the mask (if present), i.e. mask value at those cells \
        are set to 1; at the end of the computation the "new mask cells" are \
        removed (by setting a missing value (`numpy.nan`) for the variable out of \
        the original mask)
        - if `False`: original mask is kept as given in input, and data point \
        falling out of (the simulated part of) the mask (if present) are ignored

    searchRadiusRelative : [sequence of] float(s), default: 1.0
        sequence of floats of same length as `category_values`, or
        a unique float (recycled); one parameter per category:
        indicates how the search ellipsoid is limited (should be positive): let
        r_i be the ranges of the covariance model along its main axes, when
        estimating/simulating a cell x, a cell y is taken into account iff it is
        within the ellipsoid centered at x of half axes equal to
        `searchRadiusRelative` * r_i;
        note: if a range r_i is non-stationary over the grid, its maximal value
        over the grid is considered

    nneighborMax : int, default: 12
        sequence of ints of same length as `category_values`, or
        a unique int (recycled); one parameter per category:
        maximum number of cells retrieved from the search ellipsoid (when
        estimating/simulating a cell), `nneighborMax=-1` for unlimited

    searchNeighborhoodSortMode : int, optional
        sequence of ints of same length as `category_values`, or
        a unique int (recycled); one parameter per category:
        indicates how to sort the search neighboorhood cells (neighbors); they
        are sorted in increasing order according to:

        - `searchNeighborhoodSortMode=0`: distance in the usual axes system
        - `searchNeighborhoodSortMode=1`: distance in the axes sytem supporting \
        the covariance model and accounting for anisotropy given by the ranges
        - `searchNeighborhoodSortMode=2`: minus the evaluation of the covariance \
        model

        Notes:

        - if the covariance model has any non-stationary parameter, then \
        `searchNeighborhoodSortMode=2` is not allowed
        - if the covariance model has any non-stationary range or non-stationary \
        angle and `searchNeighborhoodSortMode=1`: "maximal ranges" (adapted to \
        direction from the central cell) are used to compute distance for sorting \
        the neighbors

        By default (`None`): the greatest possible value is used (i.e. 2 for
        stationary covariance model, or 1 otherwise)

    seed : int, optional
        seed for initializing random number generator

    outputReportFile : str, optional
        name of the report file (if desired in output); by default (`None`): no
        report file

    nthreads : int, default: -1
        number of thread(s) to use for "GeosClassicIndicatorSim" C program;
        `nthreads = -n <= 0`: maximal number of threads of the system except n
        (but at least 1)

    verbose : int, default: 2
        verbose mode, higher implies more printing (info):

        - 0: no display
        - 1: warnings
        - 2: warnings + basic info
        - 3 (or >2): all information

        note that if an error occurred, it is raised
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    geosclassic_output : dict
        geosclassic output in python, dictionary

        {'image':image, 'nwarning':nwarning, 'warnings':warnings}

        with:

        - image : :class:`geone.img.Img`
            output image, with `image.nv=nreal` variables (simulations);
            note: `image=None` if `mpds_geosClassicOutput->outputImage=NULL`

        - nwarning : int
            total number of warning(s) encountered (same warnings can be counted
            several times)

        - warnings : list of strs
            list of distinct warnings encountered (can be empty)
    """
    fname = 'simulateIndicator1D'

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
        err_msg = f'{fname}: `category_values` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    ncategory = len(category_values)
    if ncategory <= 0:
        err_msg = f'{fname}: `category_values` is empty'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # cov_model_for_category
    cm_for_cat = cov_model_for_category # no need to work on a copy in 1D

    cm_for_cat = np.asarray(cm_for_cat).reshape(-1)
    if len(cm_for_cat) == 1:
        cm_for_cat = np.repeat(cm_for_cat, ncategory)
    elif len(cm_for_cat) != ncategory:
        err_msg = f'{fname}: `cov_model_for_category` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    if not np.all([isinstance(c, gcm.CovModel1D) for c in cm_for_cat]):
        err_msg = f'{fname}: `cov_model_for_category` should contains CovModel1D objects'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    for cov_model in cm_for_cat:
        for el in cov_model.elem:
            # weight
            w = el[1]['w']
            if np.size(w) != 1 and np.size(w) != nxyz:
                err_msg = f"{fname}: covariance model: weight ('w') not compatible with simulation grid"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            # ranges
            if 'r' in el[1].keys():
                r  = el[1]['r']
                if np.size(r) != 1 and np.size(r) != nxyz:
                    err_msg = f"{fname}: covariance model: range ('r') not compatible with simulation grid"
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

            # additional parameter (s)
            if 's' in el[1].keys():
                s  = el[1]['s']
                if np.size(s) != 1 and np.size(s) != nxyz:
                    err_msg = f"{fname}: covariance model: parameter ('s') not compatible with simulation grid"
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    if method == 'simple_kriging':
        computationMode = 3
    elif method == 'ordinary_kriging':
        computationMode = 2
    else:
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - searchRadiusRelative
    searchRadiusRelative = np.asarray(searchRadiusRelative, dtype='float').reshape(-1)
    if len(searchRadiusRelative) == 1:
        searchRadiusRelative = np.repeat(searchRadiusRelative, ncategory)
    elif len(searchRadiusRelative) != ncategory:
        err_msg = f'{fname}: `searchRadiusRelative` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    for srr in searchRadiusRelative:
        if srr < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
            err_msg = f'{fname}: a `searchRadiusRelative` is too small (should be at least {geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN})'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Check parameters - nneighborMax
    nneighborMax = np.asarray(nneighborMax, dtype='intc').reshape(-1)
    if len(nneighborMax) == 1:
        nneighborMax = np.repeat(nneighborMax, ncategory)
    elif len(nneighborMax) != ncategory:
        err_msg = f'{fname}: `nneighborMax` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    for nn in nneighborMax:
        if nn != -1 and nn <= 0:
            err_msg = f'{fname}: any `nneighborMax` should be greater than 0 or equal to -1 (unlimited)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Check parameters - searchNeighborhoodSortMode
    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode).reshape(-1)
    if len(searchNeighborhoodSortMode) == 1:
        searchNeighborhoodSortMode = np.repeat(searchNeighborhoodSortMode, ncategory)
    elif len(searchNeighborhoodSortMode) != ncategory:
        err_msg = f'{fname}: `searchNeighborhoodSortMode` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    for i in range(ncategory):
        if searchNeighborhoodSortMode[i] is None:
            # set greatest possible value
            if cm_for_cat[i].is_stationary():
                searchNeighborhoodSortMode[i] = 2
            else:
                searchNeighborhoodSortMode[i] = 1
        else:
            if searchNeighborhoodSortMode[i] == 2:
                if not cm_for_cat[i].is_stationary():
                    err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

    # for i in range(ncategory):
    #     if searchNeighborhoodSortMode[i] is None:
    #         # set greatest possible value
    #         if cm_for_cat[i].is_stationary():
    #             searchNeighborhoodSortMode[i] = 2
    #         elif cm_for_cat[i].is_orientation_stationary() and cm_for_cat[i].is_range_stationary():
    #             searchNeighborhoodSortMode[i] = 1
    #         else:
    #             searchNeighborhoodSortMode[i] = 0
    #     else:
    #         if searchNeighborhoodSortMode[i] == 2:
    #             if not cm_for_cat[i].is_stationary():
    #                 err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
    #                 if logger: logger.error(err_msg)
    #                 raise GeosclassicinterfaceError(err_msg)
    #         elif searchNeighborhoodSortMode[i] == 1:
    #             if not cm_for_cat[i].is_orientation_stationary() or not cm_for_cat[i].is_range_stationary():
    #                 err_msg = f'{fname}: `searchNeighborhoodSortMode=1` not allowed with non-stationary range or non-stationary orientation in covariance model'
    #                 if logger: logger.error(err_msg)
    #                 raise GeosclassicinterfaceError(err_msg)

    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode, dtype='intc')

    # data points: x, v
    dataPointSet = []

    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 1) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # Aggregate data on grid by taking the most frequent value in grid cell
        xx = x[:, 0]
        yy = np.ones_like(xx) * oy + 0.5 * sy
        zz = np.ones_like(xx) * oz + 0.5 * sz
        try:
            xx_agg, yy_agg, zz_agg, v_agg = img.aggregateDataPointsWrtGrid(
                                                xx, yy, zz, v,
                                                nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                op='most_freq',
                                                logger=logger)
        except Exception as exc:
            err_msg = f"{fname}: data aggregation ('most_freq') failed"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

        if len(xx_agg) == 0:
            err_msg = f'{fname}: no data point in grid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        dataPointSet.append(
            PointSet(npt=v_agg.shape[0], nv=4, val=np.array((xx_agg, yy_agg, zz_agg, v_agg)), varname=['X', 'Y', 'Z', varname])
            )

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            err_msg = f'{fname}: `mask` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if mask is not None and add_data_point_to_mask:
        # Make a copy of the original mask, to remove value in added mask cell at the end
        mask_original = np.copy(mask)
        # Add cell to mask if needed
        pts = np.zeros((0,3))
        if x is not None:
            pts = np.vstack((pts, np.array((xx_agg, yy_agg, zz_agg)).T))
        if pts.shape[0]:
            im_tmp = img.imageFromPoints(pts,
                        nx=nx, ny=ny, nz=nz,
                        sx=sx, sy=sy, sz=sz,
                        ox=ox, oy=oy, oz=oz,
                        indicator_var=True,
                        logger=logger)
            mask = 1.0*np.any((im_tmp.val[0], mask), axis=0)
            del(im_tmp)
        del(pts)

    # Check parameters - probability
    if probability is not None:
        # if method == 'ordinary_kriging':
        #     err_msg = f"{fname}: specifying 'probability' not allowed with ordinary kriging"
        #     if logger: logger.error(err_msg)
        #     raise GeosclassicinterfaceError(err_msg)
        probability = np.asarray(probability, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if probability.size not in (ncategory, ncategory*nxyz):
            err_msg = f'{fname}: size of `probability` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Check parameters - nreal
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose > 0:
            if logger:
                logger.warning(f'{fname}: `nreal` <= 0: `None` is returned')
            else:
                print(f'{fname}: WARNING: `nreal` <= 0: `None` is returned')
        return None

    # --- Fill mpds_geosClassicInput structure (C)
    try:
        mpds_geosClassicIndicatorInput = fill_mpds_geosClassicIndicatorInput(
                False,
                space_dim,
                ncategory,
                category_values,
                cm_for_cat,
                nx, ny, nz,
                sx, sy, sz,
                ox, oy, oz,
                varname,
                outputReportFile,
                computationMode,
                None,
                dataPointSet,
                mask,
                probability,
                np.zeros(ncategory),
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                seed,
                nreal,
                logger=logger)

    except Exception as exc:
        err_msg = f'{fname}: cannot fill mpds_geosClassicIndicatorInput C structure'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg) from exc

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

    if verbose > 0 and nth > os.cpu_count():
        if logger:
            logger.warning(f'{fname}: number of threads used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: number of threads used will exceed number of cpu(s) of the system...')

    if verbose > 1:
        if logger:
            logger.info('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        else:
            print('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            sys.stdout.flush()
            sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

    # Free memory on C side: mpds_geosClassicIndicatorInput
    geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
    geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)

    if err:
        # Free memory on C side: mpds_geosClassicOutput
        geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
        geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)
        # Free memory on C side: mpds_progressMonitor
        geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)
        # Raise error
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        err_msg = f'{fname}: {err_message}'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor, logger=logger)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if geosclassic_output is not None and mask is not None and add_data_point_to_mask:
        # Remove the value out of the original mask (using its copy see above)
        geosclassic_output['image'].val[:, mask_original==0.0] = np.nan

    if verbose > 1 and geosclassic_output:
        if logger:
            logger.info(f'{fname}: Geos-Classic run complete')
        else:
            print(f'{fname}: Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose > 0 and geosclassic_output and geosclassic_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({geosclassic_output['nwarning']} times in all):")
            for i, warning_message in enumerate(geosclassic_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

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
        add_data_point_to_mask=True,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        seed=None,
        outputReportFile=None,
        treat_image_one_by_one=False,
        nproc=None, nthreads_per_proc=None,
        verbose=2, 
        logger=None):
    """
    Computes the same as the function :func:`simulateIndicator1D`, using multiprocessing.

    All the parameters are the same as those of the function :func:`simulateIndicator1D`,
    except `nthreads` that is replaced by the parameters `nproc` and
    `nthreads_per_proc`, and an extra parameter `treat_image_one_by_one`.

    This function launches multiple processes (based on `multiprocessing`
    package):

    - `nproc` parallel processes using each one `nthreads_per_proc` threads \
    are launched [parallel calls of the function :func:`simulateIndicator1D`]
    - the set of realizations (specified by `nreal`) is distributed in a \
    balanced way over the processes
    - in terms of resources, this implies the use of `nproc*nthreads_per_proc` \
    cpu(s)

    See function :func:`simulateIndicator1D`.

    **Parameters (new)**
    --------------------
    nproc : int, optional
        number of process(es); by default (`None`):
        `nproc` is set to `min(nmax-1, nreal)` (but at least 1), where nmax is
        the total number of cpu(s) of the system (retrieved by
        `multiprocessing.cpu_count()`)

    nthreads_per_proc : int, optional
        number of thread(s) per process (should be > 0); by default (`None`):
        `nthreads_per_proc` is automatically computed as the maximal integer
        (but at least 1) such that `nproc*nthreads_per_proc <= nmax-1`, where
        nmax is the total number of cpu(s) of the system (retrieved by
        `multiprocessing.cpu_count()`)

    treat_image_one_by_one : bool, default: False
        keyword argument passed to the function :func:`geone.img.gatherImages`:

        - if `True`: images (result of each process) are gathered one by one, \
        i.e. the variables of each image are inserted in an output image one by \
        one and removed from the source (slower, may save memory)
        - if `False`: images (result of each process) are gathered at once, \
        i.e. the variables of all images are inserted in an output image at once, \
        and then removed (faster)
    """
    fname = 'simulateIndicator1D_mp'

    # Set number of process(es): nproc
    if nproc is None:
        nproc = max(min(multiprocessing.cpu_count()-1, nreal), 1)
    else:
        nproc_tmp = nproc
        nproc = max(min(int(nproc), nreal), 1)
        if verbose > 1 and nproc != nproc_tmp:
            if logger:
                logger.info(f'{fname}: number of process(es) has been changed (now: nproc={nproc})')
            else:
                print(f'{fname}: number of process(es) has been changed (now: nproc={nproc})')

    # Set number of threads per process: nth
    if nthreads_per_proc is None:
        nth = max(int(np.floor((multiprocessing.cpu_count()-1) / nproc)), 1)
    else:
        nth = max(int(nthreads_per_proc), 1)
        if verbose > 1 and nth != nthreads_per_proc:
            if logger:
                logger.info(f'{fname}: number of threads per process has been changed (now: nthreads_per_proc={nth})')
            else:
                print(f'{fname}: number of threads per process has been changed (now: nthreads_per_proc={nth})')

    if verbose > 0 and nproc * nth > multiprocessing.cpu_count():
        if logger:
            logger.warning(f'{fname}: total number of cpu(s) used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: total number of cpu(s) used will exceed number of cpu(s) of the system...')

    # Set the distribution of the realizations over the processes
    # Condider the Euclidean division of nreal by nproc:
    #     nreal = q * nproc + r, with 0 <= r < nproc
    # Then, (q+1) realizations will be done on process 0, 1, ..., r-1, and q realization on process r, ..., nproc-1
    # Define the list real_index_proc of length (nproc+1) such that
    #   real_index_proc[i], ..., real_index_proc[i+1] - 1 : are the realization indices run on process i
    q, r = np.divmod(nreal, nproc)
    real_index_proc = [i*q + min(i, r) for i in range(nproc+1)]

    if verbose > 1:
        if logger:
            logger.info('{}: Geos-Classic running on {} process(es)... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, nproc, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        else:
            print('{}: Geos-Classic running on {} process(es)... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, nproc, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            sys.stdout.flush()
            sys.stdout.flush() # twice!, so that the previous print is flushed before launching geos-classic...

    # Prepare seed
    if seed is None:
        seed = np.random.randint(1, 1000000)
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
        verbose_p = 0
        # if i==0:
        #     verbose_p = min(verbose, 1) # allow to print warnings for process i
        # else:
        #     verbose_p = 0
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
                add_data_point_to_mask,
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                seed_p,                      # seed (adjusted)
                outputReportFile_p,          # outputReportFile (adjusted)
                nth,                         # nthreads
                verbose_p),                  # verbose (adjusted)
                kwds={'logger':logger})
            )

    # Properly end working process
    pool.close() # Prevents any more tasks from being submitted to the pool,
    pool.join()  # then, wait for the worker processes to exit.

    # Get result from each process
    geosclassic_output_proc = [p.get() for p in out_pool]

    if np.any([out is None for out in geosclassic_output_proc]):
        return None

    # Gather results from every process
    # image
    image = []
    for out in geosclassic_output_proc:
        if out['image'] is not None:
            image.append(out['image'])
            del(out['image'])
    if len(image) == 0:
        image = None
    # Gather images and adjust variable names
    all_image = img.gatherImages(image, keep_varname=True, rem_var_from_source=True, treat_image_one_by_one=treat_image_one_by_one, logger=logger)
    ndigit = geosclassic.MPDS_GEOS_CLASSIC_NB_DIGIT_FOR_REALIZATION_NUMBER
    for j in range(all_image.nv):
        all_image.varname[j] = all_image.varname[j][:-ndigit] + f'{j:0{ndigit}d}'

    # nwarning
    nwarning = np.sum([out['nwarning'] for out in geosclassic_output_proc])
    # warnings
    warnings = list(np.unique(np.hstack([out['warnings'] for out in geosclassic_output_proc])))

    geosclassic_output = {'image':all_image, 'nwarning':nwarning, 'warnings':warnings}

    if verbose > 1 and geosclassic_output:
        if logger:
            logger.info(f'{fname}: Geos-Classic run complete (all process(es))')
        else:
           print(f'{fname}: Geos-Classic run complete (all process(es))')

    # Show (print) encountered warnings
    if verbose > 0 and geosclassic_output and geosclassic_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({geosclassic_output['nwarning']} times in all):")
            for i, warning_message in enumerate(geosclassic_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

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
        add_data_point_to_mask=True,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        seed=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2,
        logger=None):
    """
    Generates 2D simulations (Sequential Indicator Simulation, SIS).

    A simulation takes place in (center of) grid cells, based on simple or
    ordinary kriging of the indicator variables of the categories.

    Parameters
    ----------
    category_values : 1D array-like
        sequence of category values; let `ncategory` be the number of categories,
        then:

        - if `ncategory=1`: the unique category value given must not be equal to \
        zero; it is used for a binary case with values "unique category value" \
        and 0, where 0 indicates the absence of the considered medium; the \
        conditioning data values should be equal to"unique category value" or 0
        - if `ncategory>=2`: it is used for a multi-category case with given \
        category values (distinct); the conditioning data values should be in the \
        `category_values`

    cov_model_for_category : [sequence of] :class:`geone.covModel.CovModel2D`
        sequence of same length as `category_values` of covariance model in 2D,
        or a unique covariance model in 2D (recycled):
        covariance model for each category

    dimension : 2-tuple of ints
        `dimension=(nx, ny)`, number of cells in the 2D simulation grid along
        each axis

    spacing : 2-tuple of floats, default: (1.0, 1.0)
        `spacing=(sx, sy)`, cell size along each axis

    origin : 2-tuple of floats, default: (0.0, 0.0)
        `origin=(ox, oy)`, origin of the 2D simulation grid (lower-left corner)

    method : str {'simple_kriging', 'ordinary_kriging'}, default: 'simple_kriging'
        type of kriging

    nreal : int, default: 1
        number of realizations

    probability : array-like of floats, optional
        probability for each category:

        - sequence of same length as `category_values`: \
        probability[i]: probability (proportion, kriging mean value for the \
        indicator variable) for category `category_values[i]`, used for \
        every grid cell
        - array-like of size ncategory * ngrid_cells, where ncategory is the \
        length of `category_values` and ngrid_cells is the number of grid \
        cells (the array is reshaped if needed): first ngrid_cells values are \
        the probabilities (proportions, kriging mean values for the indicator \
        variable) for the first category at grid cells, etc. \
        (for non-stationary probailities / proportions)

        By default (`None`): proportion of each category computed from the
        data values (`v`) are used for every grid cell (uniform probabilities
        if no data point)

        Note: for ordinary kriging (`method='ordinary_kriging'`), it is used for
        case with no neighbor

    x : 2D array of floats of shape (n, 2), optional
        data points locations, with n the number of data points, each row of `x`
        is the float coordinates of one data point; note: if n=1, a 1D array of
        shape (2,) is accepted

    v : 1D array of floats of shape (n,), optional
        data values at `x` (`v[i]` is the data value at `x[i]`)

    mask : array-like, optional
        mask value at grid cells (value 1 for simulated cells, value 0 for not
        simulated cells); the size of the array must be equal to the number of
        grid cells (the array is reshaped if needed)

    add_data_point_to_mask : bool, default: True
        - if `True`: any grid cell that contains a data point is added to (the \
        simulated part of) the mask (if present), i.e. mask value at those cells \
        are set to 1; at the end of the computation the "new mask cells" are \
        removed (by setting a missing value (`numpy.nan`) for the variable out of \
        the original mask)
        - if `False`: original mask is kept as given in input, and data point \
        falling out of (the simulated part of) the mask (if present) are ignored

    searchRadiusRelative : [sequence of] float(s), default: 1.0
        sequence of floats of same length as `category_values`, or
        a unique float (recycled); one parameter per category:
        indicates how the search ellipsoid is limited (should be positive): let
        r_i be the ranges of the covariance model along its main axes, when
        estimating/simulating a cell x, a cell y is taken into account iff it is
        within the ellipsoid centered at x of half axes equal to
        `searchRadiusRelative` * r_i;
        note: if a range r_i is non-stationary over the grid, its maximal value
        over the grid is considered

    nneighborMax : int, default: 12
        sequence of ints of same length as `category_values`, or
        a unique int (recycled); one parameter per category:
        maximum number of cells retrieved from the search ellipsoid (when
        estimating/simulating a cell), `nneighborMax=-1` for unlimited

    searchNeighborhoodSortMode : int, optional
        sequence of ints of same length as `category_values`, or
        a unique int (recycled); one parameter per category:
        indicates how to sort the search neighboorhood cells (neighbors); they
        are sorted in increasing order according to:

        - `searchNeighborhoodSortMode=0`: distance in the usual axes system
        - `searchNeighborhoodSortMode=1`: distance in the axes sytem supporting \
        the covariance model and accounting for anisotropy given by the ranges
        - `searchNeighborhoodSortMode=2`: minus the evaluation of the covariance \
        model

        Notes:

        - if the covariance model has any non-stationary parameter, then \
        `searchNeighborhoodSortMode=2` is not allowed
        - if the covariance model has any non-stationary range or non-stationary \
        angle and `searchNeighborhoodSortMode=1`: "maximal ranges" (adapted to \
        direction from the central cell) are used to compute distance for sorting \
        the neighbors

        By default (`None`): the greatest possible value is used (i.e. 2 for
        stationary covariance model, or 1 otherwise)

    seed : int, optional
        seed for initializing random number generator

    outputReportFile : str, optional
        name of the report file (if desired in output); by default (`None`): no
        report file

    nthreads : int, default: -1
        number of thread(s) to use for "GeosClassicIndicatorSim" C program;
        `nthreads = -n <= 0`: maximal number of threads of the system except n
        (but at least 1)

    verbose : int, default: 2
        verbose mode, higher implies more printing (info):

        - 0: no display
        - 1: warnings
        - 2: warnings + basic info
        - 3 (or >2): all information

        note that if an error occurred, it is raised
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    geosclassic_output : dict
        geosclassic output in python, dictionary

        {'image':image, 'nwarning':nwarning, 'warnings':warnings}

        with:

        - image : :class:`geone.img.Img`
            output image, with `image.nv=nreal` variables (simulations);
            note: `image=None` if `mpds_geosClassicOutput->outputImage=NULL`

        - nwarning : int
            total number of warning(s) encountered (same warnings can be counted
            several times)

        - warnings : list of strs
            list of distinct warnings encountered (can be empty)
    """
    fname = 'simulateIndicator2D'

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
        err_msg = f'{fname}: `category_values` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    ncategory = len(category_values)
    if ncategory <= 0:
        err_msg = f'{fname}: `category_values` is empty'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # cov_model_for_category
    cov_model_for_category = np.asarray(cov_model_for_category).reshape(-1)
    if not np.all([isinstance(c, gcm.CovModel2D) for c in cov_model_for_category]):
        # cov model will be converted:
        #    as applying modification in an array is persistent at exit,
        #    work on a copy to ensure no modification of the initial entry
        cm_for_cat = copy.deepcopy(cov_model_for_category)
    else:
        cm_for_cat = cov_model_for_category

    cm_for_cat = np.asarray(cm_for_cat).reshape(-1)
    for i in range(len(cm_for_cat)):
        if isinstance(cm_for_cat[i], gcm.CovModel1D):
            cm_for_cat[i] = gcm.covModel1D_to_covModel2D(cm_for_cat[i]) # convert model 1D in 2D
    if len(cm_for_cat) == 1:
        cm_for_cat = np.repeat(cm_for_cat, ncategory)
    elif len(cm_for_cat) != ncategory:
        err_msg = f'{fname}: `cov_model_for_category` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    if not np.all([isinstance(c, gcm.CovModel2D) for c in cm_for_cat]):
        err_msg = f'{fname}: `cov_model_for_category` should contains CovModel2D objects'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    for cov_model in cm_for_cat:
        for el in cov_model.elem:
            # weight
            w = el[1]['w']
            if np.size(w) != 1 and np.size(w) != nxyz:
                err_msg = f"{fname}: covariance model: weight ('w') not compatible with simulation grid"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            # ranges
            if 'r' in el[1].keys():
                for r in el[1]['r']:
                    if np.size(r) != 1 and np.size(r) != nxyz:
                        err_msg = f"{fname}: covariance model: range ('r') not compatible with simulation grid"
                        if logger: logger.error(err_msg)
                        raise GeosclassicinterfaceError(err_msg)

            # additional parameter (s)
            if 's' in el[1].keys():
                s  = el[1]['s']
                if np.size(s) != 1 and np.size(s) != nxyz:
                    err_msg = f"{fname}: covariance model: parameter ('s') not compatible with simulation grid"
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

        # alpha
        angle = cov_model.alpha
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            err_msg = f"{fname}: covariance model: angle ('alpha') not compatible with simulation grid"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    if method == 'simple_kriging':
        computationMode = 3
    elif method == 'ordinary_kriging':
        computationMode = 2
    else:
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - searchRadiusRelative
    searchRadiusRelative = np.asarray(searchRadiusRelative, dtype='float').reshape(-1)
    if len(searchRadiusRelative) == 1:
        searchRadiusRelative = np.repeat(searchRadiusRelative, ncategory)
    elif len(searchRadiusRelative) != ncategory:
        err_msg = f'{fname}: `searchRadiusRelative` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    for srr in searchRadiusRelative:
        if srr < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
            err_msg = f'{fname}: a `searchRadiusRelative` is too small (should be at least {geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN})'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Check parameters - nneighborMax
    nneighborMax = np.asarray(nneighborMax, dtype='intc').reshape(-1)
    if len(nneighborMax) == 1:
        nneighborMax = np.repeat(nneighborMax, ncategory)
    elif len(nneighborMax) != ncategory:
        err_msg = f'{fname}: `nneighborMax` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    for nn in nneighborMax:
        if nn != -1 and nn <= 0:
            err_msg = f'{fname}: any `nneighborMax` should be greater than 0 or equal to -1 (unlimited)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Check parameters - searchNeighborhoodSortMode
    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode).reshape(-1)
    if len(searchNeighborhoodSortMode) == 1:
        searchNeighborhoodSortMode = np.repeat(searchNeighborhoodSortMode, ncategory)
    elif len(searchNeighborhoodSortMode) != ncategory:
        err_msg = f'{fname}: `searchNeighborhoodSortMode` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    for i in range(ncategory):
        if searchNeighborhoodSortMode[i] is None:
            # set greatest possible value
            if cm_for_cat[i].is_stationary():
                searchNeighborhoodSortMode[i] = 2
            else:
                searchNeighborhoodSortMode[i] = 1
        else:
            if searchNeighborhoodSortMode[i] == 2:
                if not cm_for_cat[i].is_stationary():
                    err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

    # for i in range(ncategory):
    #     if searchNeighborhoodSortMode[i] is None:
    #         # set greatest possible value
    #         if cm_for_cat[i].is_stationary():
    #             searchNeighborhoodSortMode[i] = 2
    #         elif cm_for_cat[i].is_orientation_stationary() and cm_for_cat[i].is_range_stationary():
    #             searchNeighborhoodSortMode[i] = 1
    #         else:
    #             searchNeighborhoodSortMode[i] = 0
    #     else:
    #         if searchNeighborhoodSortMode[i] == 2:
    #             if not cm_for_cat[i].is_stationary():
    #                 err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
    #                 if logger: logger.error(err_msg)
    #                 raise GeosclassicinterfaceError(err_msg)
    #         elif searchNeighborhoodSortMode[i] == 1:
    #             if not cm_for_cat[i].is_orientation_stationary() or not cm_for_cat[i].is_range_stationary():
    #                 err_msg = f'{fname}: `searchNeighborhoodSortMode=1` not allowed with non-stationary range or non-stationary orientation in covariance model'
    #                 if logger: logger.error(err_msg)
    #                 raise GeosclassicinterfaceError(err_msg)

    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode, dtype='intc')

    # data points: x, v
    dataPointSet = []

    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 2) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # Aggregate data on grid by taking the most frequent value in grid cell
        xx, yy = x.T
        zz = np.ones_like(xx) * oz + 0.5 * sz
        try:
            xx_agg, yy_agg, zz_agg, v_agg = img.aggregateDataPointsWrtGrid(
                                                xx, yy, zz, v,
                                                nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                op='most_freq',
                                                logger=logger)
        except Exception as exc:
            err_msg = f"{fname}: data aggregation ('most_freq') failed"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

        if len(xx_agg) == 0:
            err_msg = f'{fname}: no data point in grid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        dataPointSet.append(
            PointSet(npt=v_agg.shape[0], nv=4, val=np.array((xx_agg, yy_agg, zz_agg, v_agg)), varname=['X', 'Y', 'Z', varname])
            )

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            err_msg = f'{fname}: `mask` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if mask is not None and add_data_point_to_mask:
        # Make a copy of the original mask, to remove value in added mask cell at the end
        mask_original = np.copy(mask)
        # Add cell to mask if needed
        pts = np.zeros((0,3))
        if x is not None:
            pts = np.vstack((pts, np.array((xx_agg, yy_agg, zz_agg)).T))
        if pts.shape[0]:
            im_tmp = img.imageFromPoints(pts,
                        nx=nx, ny=ny, nz=nz,
                        sx=sx, sy=sy, sz=sz,
                        ox=ox, oy=oy, oz=oz,
                        indicator_var=True,
                        logger=logger)
            mask = 1.0*np.any((im_tmp.val[0], mask), axis=0)
            del(im_tmp)
        del(pts)

    # Check parameters - probability
    if probability is not None:
        # if method == 'ordinary_kriging':
        #     err_msg = f"{fname}: specifying 'probability' not allowed with ordinary kriging"
        #     if logger: logger.error(err_msg)
        #     raise GeosclassicinterfaceError(err_msg)
        probability = np.asarray(probability, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if probability.size not in (ncategory, ncategory*nxyz):
            err_msg = f'{fname}: size of `probability` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Check parameters - nreal
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose > 0:
            if logger:
                logger.warning(f'{fname}: `nreal` <= 0: `None` is returned')
            else:
                print(f'{fname}: WARNING: `nreal` <= 0: `None` is returned')
        return None

    # --- Fill mpds_geosClassicInput structure (C)
    try:
        mpds_geosClassicIndicatorInput = fill_mpds_geosClassicIndicatorInput(
                False,
                space_dim,
                ncategory,
                category_values,
                cm_for_cat,
                nx, ny, nz,
                sx, sy, sz,
                ox, oy, oz,
                varname,
                outputReportFile,
                computationMode,
                None,
                dataPointSet,
                mask,
                probability,
                np.zeros(ncategory),
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                seed,
                nreal,
                logger=logger)

    except Exception as exc:
        err_msg = f'{fname}: cannot fill mpds_geosClassicIndicatorInput C structure'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg) from exc

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

    if verbose > 0 and nth > os.cpu_count():
        if logger:
            logger.warning(f'{fname}: number of threads used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: number of threads used will exceed number of cpu(s) of the system...')

    if verbose > 1:
        if logger:
            logger.info('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        else:
            print('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            sys.stdout.flush()
            sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

    # Free memory on C side: mpds_geosClassicIndicatorInput
    geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
    geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)

    if err:
        # Free memory on C side: mpds_geosClassicOutput
        geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
        geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)
        # Free memory on C side: mpds_progressMonitor
        geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)
        # Raise error
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        err_msg = f'{fname}: {err_message}'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor, logger=logger)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if geosclassic_output is not None and mask is not None and add_data_point_to_mask:
        # Remove the value out of the original mask (using its copy see above)
        geosclassic_output['image'].val[:, mask_original==0.0] = np.nan

    if verbose > 1 and geosclassic_output:
        if logger:
            logger.info(f'{fname}: Geos-Classic run complete')
        else:
            print(f'{fname}: Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose > 0 and geosclassic_output and geosclassic_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({geosclassic_output['nwarning']} times in all):")
            for i, warning_message in enumerate(geosclassic_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

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
        add_data_point_to_mask=True,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        seed=None,
        outputReportFile=None,
        treat_image_one_by_one=False,
        nproc=None, nthreads_per_proc=None,
        verbose=2,
        logger=None):
    """
    Computes the same as the function :func:`simulateIndicator2D`, using multiprocessing.

    All the parameters are the same as those of the function :func:`simulateIndicator2D`,
    except `nthreads` that is replaced by the parameters `nproc` and
    `nthreads_per_proc`, and an extra parameter `treat_image_one_by_one`.

    This function launches multiple processes (based on `multiprocessing`
    package):

    - `nproc` parallel processes using each one `nthreads_per_proc` threads \
    are launched [parallel calls of the function :func:`simulateIndicator2D`]
    - the set of realizations (specified by `nreal`) is distributed in a \
    balanced way over the processes
    - in terms of resources, this implies the use of `nproc*nthreads_per_proc` \
    cpu(s)

    See function :func:`simulateIndicator2D`.

    **Parameters (new)**
    --------------------
    nproc : int, optional
        number of process(es); by default (`None`):
        `nproc` is set to `min(nmax-1, nreal)` (but at least 1), where nmax is
        the total number of cpu(s) of the system (retrieved by
        `multiprocessing.cpu_count()`)

    nthreads_per_proc : int, optional
        number of thread(s) per process (should be > 0); by default (`None`):
        `nthreads_per_proc` is automatically computed as the maximal integer
        (but at least 1) such that `nproc*nthreads_per_proc <= nmax-1`, where
        nmax is the total number of cpu(s) of the system (retrieved by
        `multiprocessing.cpu_count()`)

    treat_image_one_by_one : bool, default: False
        keyword argument passed to the function :func:`geone.img.gatherImages`:

        - if `True`: images (result of each process) are gathered one by one, \
        i.e. the variables of each image are inserted in an output image one by \
        one and removed from the source (slower, may save memory)
        - if `False`: images (result of each process) are gathered at once, \
        i.e. the variables of all images are inserted in an output image at once, \
        and then removed (faster)
    """
    fname = 'simulateIndicator2D_mp'

    # Set number of process(es): nproc
    if nproc is None:
        nproc = max(min(multiprocessing.cpu_count()-1, nreal), 1)
    else:
        nproc_tmp = nproc
        nproc = max(min(int(nproc), nreal), 1)
        if verbose > 1 and nproc != nproc_tmp:
            if logger:
                logger.info(f'{fname}: number of process(es) has been changed (now: nproc={nproc})')
            else:
                print(f'{fname}: number of process(es) has been changed (now: nproc={nproc})')

    # Set number of threads per process: nth
    if nthreads_per_proc is None:
        nth = max(int(np.floor((multiprocessing.cpu_count()-1) / nproc)), 1)
    else:
        nth = max(int(nthreads_per_proc), 1)
        if verbose > 1 and nth != nthreads_per_proc:
            if logger:
                logger.info(f'{fname}: number of threads per process has been changed (now: nthreads_per_proc={nth})')
            else:
                print(f'{fname}: number of threads per process has been changed (now: nthreads_per_proc={nth})')

    if verbose > 0 and nproc * nth > multiprocessing.cpu_count():
        if logger:
            logger.warning(f'{fname}: total number of cpu(s) used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: total number of cpu(s) used will exceed number of cpu(s) of the system...')

    # Set the distribution of the realizations over the processes
    # Condider the Euclidean division of nreal by nproc:
    #     nreal = q * nproc + r, with 0 <= r < nproc
    # Then, (q+1) realizations will be done on process 0, 1, ..., r-1, and q realization on process r, ..., nproc-1
    # Define the list real_index_proc of length (nproc+1) such that
    #   real_index_proc[i], ..., real_index_proc[i+1] - 1 : are the realization indices run on process i
    q, r = np.divmod(nreal, nproc)
    real_index_proc = [i*q + min(i, r) for i in range(nproc+1)]

    if verbose > 1:
        if logger:
            logger.info('{}: Geos-Classic running on {} process(es)... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, nproc, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        else:
            print('{}: Geos-Classic running on {} process(es)... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, nproc, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            sys.stdout.flush()
            sys.stdout.flush() # twice!, so that the previous print is flushed before launching geos-classic...

    # Prepare seed
    if seed is None:
        seed = np.random.randint(1, 1000000)
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
        verbose_p = 0
        # if i==0:
        #     verbose_p = min(verbose, 1) # allow to print warnings for process i
        # else:
        #     verbose_p = 0
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
                add_data_point_to_mask,
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                seed_p,                      # seed (adjusted)
                outputReportFile_p,          # outputReportFile (adjusted)
                nth,                         # nthreads
                verbose_p),                  # verbose (adjusted)
                kwds={'logger':logger})
            )

    # Properly end working process
    pool.close() # Prevents any more tasks from being submitted to the pool,
    pool.join()  # then, wait for the worker processes to exit.

    # Get result from each process
    geosclassic_output_proc = [p.get() for p in out_pool]

    if np.any([out is None for out in geosclassic_output_proc]):
        return None

    # Gather results from every process
    # image
    image = []
    for out in geosclassic_output_proc:
        if out['image'] is not None:
            image.append(out['image'])
            del(out['image'])
    if len(image) == 0:
        image = None
    # Gather images and adjust variable names
    all_image = img.gatherImages(image, keep_varname=True, rem_var_from_source=True, treat_image_one_by_one=treat_image_one_by_one, logger=logger)
    ndigit = geosclassic.MPDS_GEOS_CLASSIC_NB_DIGIT_FOR_REALIZATION_NUMBER
    for j in range(all_image.nv):
        all_image.varname[j] = all_image.varname[j][:-ndigit] + f'{j:0{ndigit}d}'

    # nwarning
    nwarning = np.sum([out['nwarning'] for out in geosclassic_output_proc])
    # warnings
    warnings = list(np.unique(np.hstack([out['warnings'] for out in geosclassic_output_proc])))

    geosclassic_output = {'image':all_image, 'nwarning':nwarning, 'warnings':warnings}

    if verbose > 1 and geosclassic_output:
        if logger:
            logger.info(f'{fname}: Geos-Classic run complete (all process(es))')
        else:
            print(f'{fname}: Geos-Classic run complete (all process(es))')

    # Show (print) encountered warnings
    if verbose > 0 and geosclassic_output and geosclassic_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({geosclassic_output['nwarning']} times in all):")
            for i, warning_message in enumerate(geosclassic_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

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
        add_data_point_to_mask=True,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        seed=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2,
        logger=None):
    """
    Generates 3D simulations (Sequential Indicator Simulation, SIS).

    A simulation takes place in (center of) grid cells, based on simple or
    ordinary kriging of the indicator variables of the categories.

    Parameters
    ----------
    category_values : 1D array-like
        sequence of category values; let `ncategory` be the number of categories,
        then:

        - if `ncategory=1`: the unique category value given must not be equal to \
        zero; it is used for a binary case with values "unique category value" \
        and 0, where 0 indicates the absence of the considered medium; the \
        conditioning data values should be equal to "unique category value" or 0
        - if `ncategory>=2`: it is used for a multi-category case with given \
        category values (distinct); the conditioning data values should be in the \
        `category_values`

    cov_model_for_category : [sequence of] :class:`geone.covModel.CovModel3D`
        sequence of same length as `category_values` of covariance model in 3D,
        or a unique covariance model in 3D (recycled):
        covariance model for each category

    dimension : 3-tuple of ints
        `dimension=(nx, ny, nz)`, number of cells in the 3D simulation grid along
        each axis

    spacing : 3-tuple of floats, default: (1.0,1.0, 1.0)
        `spacing=(sx, sy, sz)`, cell size along each axis

    origin : 3-tuple of floats, default: (0.0, 0.0, 0.0)
        `origin=(ox, oy, oz)`, origin of the 3D simulation grid (bottom-lower-left
        corner)

    method : str {'simple_kriging', 'ordinary_kriging'}, default: 'simple_kriging'
        type of kriging

    nreal : int, default: 1
        number of realizations

    probability : array-like of floats, optional
        probability for each category:

        - sequence of same length as `category_values`: \
        probability[i]: probability (proportion, kriging mean value for the \
        indicator variable) for category `category_values[i]`, used for \
        every grid cell
        - array-like of size ncategory * ngrid_cells, where ncategory is the \
        length of `category_values` and ngrid_cells is the number of grid \
        cells (the array is reshaped if needed): first ngrid_cells values are \
        the probabilities (proportions, kriging mean values for the indicator \
        variable) for the first category at grid cells, etc. \
        (for non-stationary probailities / proportions)

        By default (`None`): proportion of each category computed from the
        data values (`v`) are used for every grid cell (uniform probabilities
        if no data point)

        Note: for ordinary kriging (`method='ordinary_kriging'`), it is used for
        case with no neighbor

    x : 2D array of floats of shape (n, 3), optional
        data points locations, with n the number of data points, each row of `x`
        is the float coordinates of one data point; note: if n=1, a 1D array of
        shape (3,) is accepted

    v : 1D array of floats of shape (n,), optional
        data values at `x` (`v[i]` is the data value at `x[i]`)

    mask : array-like, optional
        mask value at grid cells (value 1 for simulated cells, value 0 for not
        simulated cells); the size of the array must be equal to the number of
        grid cells (the array is reshaped if needed)

    add_data_point_to_mask : bool, default: True
        - if `True`: any grid cell that contains a data point is added to (the \
        simulated part of) the mask (if present), i.e. mask value at those cells \
        are set to 1; at the end of the computation the "new mask cells" are \
        removed (by setting a missing value (`numpy.nan`) for the variable out of \
        the original mask)
        - if `False`: original mask is kept as given in input, and data point \
        falling out of (the simulated part of) the mask (if present) are ignored

    searchRadiusRelative : [sequence of] float(s), default: 1.0
        sequence of floats of same length as `category_values`, or
        a unique float (recycled); one parameter per category:
        indicates how the search ellipsoid is limited (should be positive): let
        r_i be the ranges of the covariance model along its main axes, when
        estimating/simulating a cell x, a cell y is taken into account iff it is
        within the ellipsoid centered at x of half axes equal to
        `searchRadiusRelative` * r_i;
        note: if a range r_i is non-stationary over the grid, its maximal value
        over the grid is considered

    nneighborMax : int, default: 12
        sequence of ints of same length as `category_values`, or
        a unique int (recycled); one parameter per category:
        maximum number of cells retrieved from the search ellipsoid (when
        estimating/simulating a cell), `nneighborMax=-1` for unlimited

    searchNeighborhoodSortMode : int, optional
        sequence of ints of same length as `category_values`, or
        a unique int (recycled); one parameter per category:
        indicates how to sort the search neighboorhood cells (neighbors); they
        are sorted in increasing order according to:

        - `searchNeighborhoodSortMode=0`: distance in the usual axes system
        - `searchNeighborhoodSortMode=1`: distance in the axes sytem supporting \
        the covariance model and accounting for anisotropy given by the ranges
        - `searchNeighborhoodSortMode=2`: minus the evaluation of the covariance \
        model

        Notes:

        - if the covariance model has any non-stationary parameter, then \
        `searchNeighborhoodSortMode=2` is not allowed
        - if the covariance model has any non-stationary range or non-stationary \
        angle and `searchNeighborhoodSortMode=1`: "maximal ranges" (adapted to \
        direction from the central cell) are used to compute distance for sorting \
        the neighbors

        By default (`None`): the greatest possible value is used (i.e. 2 for
        stationary covariance model, or 1 otherwise)

    seed : int, optional
        seed for initializing random number generator

    outputReportFile : str, optional
        name of the report file (if desired in output); by default (`None`): no
        report file

    nthreads : int, default: -1
        number of thread(s) to use for "GeosClassicIndicatorSim" C program;
        `nthreads = -n <= 0`: maximal number of threads of the system except n
        (but at least 1)

    verbose : int, default: 2
        verbose mode, higher implies more printing (info):

        - 0: no display
        - 1: warnings
        - 2: warnings + basic info
        - 3 (or >2): all information

        note that if an error occurred, it is raised
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    geosclassic_output : dict
        geosclassic output in python, dictionary

        {'image':image, 'nwarning':nwarning, 'warnings':warnings}

        with:

        - image : :class:`geone.img.Img`
            output image, with `image.nv=nreal` variables (simulations);
            note: `image=None` if `mpds_geosClassicOutput->outputImage=NULL`

        - nwarning : int
            total number of warning(s) encountered (same warnings can be counted
            several times)

        - warnings : list of strs
            list of distinct warnings encountered (can be empty)
    """
    fname = 'simulateIndicator3D'

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
        err_msg = f'{fname}: `category_values` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    ncategory = len(category_values)
    if ncategory <= 0:
        err_msg = f'{fname}: `category_values` is empty'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # cov_model_for_category
    cov_model_for_category = np.asarray(cov_model_for_category).reshape(-1)
    if not np.all([isinstance(c, gcm.CovModel3D) for c in cov_model_for_category]):
        # cov model will be converted:
        #    as applying modification in an array is persistent at exit,
        #    work on a copy to ensure no modification of the initial entry
        cm_for_cat = copy.deepcopy(cov_model_for_category)
    else:
        cm_for_cat = cov_model_for_category

    cm_for_cat = np.asarray(cm_for_cat).reshape(-1)
    for i in range(len(cm_for_cat)):
        if isinstance(cm_for_cat[i], gcm.CovModel1D):
            cm_for_cat[i] = gcm.covModel1D_to_covModel3D(cm_for_cat[i]) # convert model 1D in 3D
    if len(cm_for_cat) == 1:
        cm_for_cat = np.repeat(cm_for_cat, ncategory)
    elif len(cm_for_cat) != ncategory:
        err_msg = f'{fname}: `cov_model_for_category` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    if not np.all([isinstance(c, gcm.CovModel3D) for c in cm_for_cat]):
        err_msg = f'{fname}: `cov_model_for_category` should contains CovModel3D objects'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    for cov_model in cm_for_cat:
        for el in cov_model.elem:
            # weight
            w = el[1]['w']
            if np.size(w) != 1 and np.size(w) != nxyz:
                err_msg = f"{fname}: covariance model: weight ('w') not compatible with simulation grid"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            # ranges
            if 'r' in el[1].keys():
                for r in el[1]['r']:
                    if np.size(r) != 1 and np.size(r) != nxyz:
                        err_msg = f"{fname}: covariance model: range ('r') not compatible with simulation grid"
                        if logger: logger.error(err_msg)
                        raise GeosclassicinterfaceError(err_msg)

            # additional parameter (s)
            if 's' in el[1].keys():
                s  = el[1]['s']
                if np.size(s) != 1 and np.size(s) != nxyz:
                    err_msg = f"{fname}: covariance model: parameter ('s') not compatible with simulation grid"
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

        # alpha
        angle = cov_model.alpha
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            err_msg = f"{fname}: covariance model: angle ('alpha') not compatible with simulation grid"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # beta
        angle = cov_model.beta
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            err_msg = f"{fname}: covariance model: angle ('beta') not compatible with simulation grid"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # gamma
        angle = cov_model.gamma
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            err_msg = f"{fname}: covariance model: angle ('gamma') not compatible with simulation grid"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    if method == 'simple_kriging':
        computationMode = 3
    elif method == 'ordinary_kriging':
        computationMode = 2
    else:
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - searchRadiusRelative
    searchRadiusRelative = np.asarray(searchRadiusRelative, dtype='float').reshape(-1)
    if len(searchRadiusRelative) == 1:
        searchRadiusRelative = np.repeat(searchRadiusRelative, ncategory)
    elif len(searchRadiusRelative) != ncategory:
        err_msg = f'{fname}: `searchRadiusRelative` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    for srr in searchRadiusRelative:
        if srr < geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN:
            err_msg = f'{fname}: a `searchRadiusRelative` is too small (should be at least {geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN})'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Check parameters - nneighborMax
    nneighborMax = np.asarray(nneighborMax, dtype='intc').reshape(-1)
    if len(nneighborMax) == 1:
        nneighborMax = np.repeat(nneighborMax, ncategory)
    elif len(nneighborMax) != ncategory:
        err_msg = f'{fname}: `nneighborMax` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    for nn in nneighborMax:
        if nn != -1 and nn <= 0:
            err_msg = f'{fname}: any `nneighborMax` should be greater than 0 or equal to -1 (unlimited)'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Check parameters - searchNeighborhoodSortMode
    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode).reshape(-1)
    if len(searchNeighborhoodSortMode) == 1:
        searchNeighborhoodSortMode = np.repeat(searchNeighborhoodSortMode, ncategory)
    elif len(searchNeighborhoodSortMode) != ncategory:
        err_msg = f'{fname}: `searchNeighborhoodSortMode` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    for i in range(ncategory):
        if searchNeighborhoodSortMode[i] is None:
            # set greatest possible value
            if cm_for_cat[i].is_stationary():
                searchNeighborhoodSortMode[i] = 2
            else:
                searchNeighborhoodSortMode[i] = 1
        else:
            if searchNeighborhoodSortMode[i] == 2:
                if not cm_for_cat[i].is_stationary():
                    err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

    # for i in range(ncategory):
    #     if searchNeighborhoodSortMode[i] is None:
    #         # set greatest possible value
    #         if cm_for_cat[i].is_stationary():
    #             searchNeighborhoodSortMode[i] = 2
    #         elif cm_for_cat[i].is_orientation_stationary() and cm_for_cat[i].is_range_stationary():
    #             searchNeighborhoodSortMode[i] = 1
    #         else:
    #             searchNeighborhoodSortMode[i] = 0
    #     else:
    #         if searchNeighborhoodSortMode[i] == 2:
    #             if not cm_for_cat[i].is_stationary():
    #                 err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
    #                 if logger: logger.error(err_msg)
    #                 raise GeosclassicinterfaceError(err_msg)
    #         elif searchNeighborhoodSortMode[i] == 1:
    #             if not cm_for_cat[i].is_orientation_stationary() or not cm_for_cat[i].is_range_stationary():
    #                 err_msg = f'{fname}: `searchNeighborhoodSortMode=1` not allowed with non-stationary range or non-stationary orientation in covariance model'
    #                 if logger: logger.error(err_msg)
    #                 raise GeosclassicinterfaceError(err_msg)

    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode, dtype='intc')

    # data points: x, v
    dataPointSet = []

    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 3) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # Aggregate data on grid by taking the most frequent value in grid cell
        xx, yy, zz = x.T
        try:
            xx_agg, yy_agg, zz_agg, v_agg = img.aggregateDataPointsWrtGrid(
                                                xx, yy, zz, v,
                                                nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                op='most_freq',
                                                logger=logger)
        except Exception as exc:
            err_msg = f"{fname}: data aggregation ('most_freq') failed"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

        if len(xx_agg) == 0:
            err_msg = f'{fname}: no data point in grid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        dataPointSet.append(
            PointSet(npt=v_agg.shape[0], nv=4, val=np.array((xx_agg, yy_agg, zz_agg, v_agg)), varname=['X', 'Y', 'Z', varname])
            )

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            err_msg = f'{fname}: `mask` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if mask is not None and add_data_point_to_mask:
        # Make a copy of the original mask, to remove value in added mask cell at the end
        mask_original = np.copy(mask)
        # Add cell to mask if needed
        pts = np.zeros((0,3))
        if x is not None:
            pts = np.vstack((pts, np.array((xx_agg, yy_agg, zz_agg)).T))
        if pts.shape[0]:
            im_tmp = img.imageFromPoints(pts,
                        nx=nx, ny=ny, nz=nz,
                        sx=sx, sy=sy, sz=sz,
                        ox=ox, oy=oy, oz=oz,
                        indicator_var=True,
                        logger=logger)
            mask = 1.0*np.any((im_tmp.val[0], mask), axis=0)
            del(im_tmp)
        del(pts)

    # Check parameters - probability
    if probability is not None:
        # if method == 'ordinary_kriging':
        #     err_msg = f"{fname}: specifying 'probability' not allowed with ordinary kriging"
        #     if logger: logger.error(err_msg)
        #     raise GeosclassicinterfaceError(err_msg)
        probability = np.asarray(probability, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if probability.size not in (ncategory, ncategory*nxyz):
            err_msg = f'{fname}: size of `probability` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # Check parameters - nreal
    nreal = int(nreal) # cast to int if needed

    if nreal <= 0:
        if verbose > 0:
            if logger:
                logger.warning(f'{fname}: `nreal` <= 0: `None` is returned')
            else:
                print(f'{fname}: WARNING: `nreal` <= 0: `None` is returned')
        return None

    # --- Fill mpds_geosClassicInput structure (C)
    try:
        mpds_geosClassicIndicatorInput = fill_mpds_geosClassicIndicatorInput(
                False,
                space_dim,
                ncategory,
                category_values,
                cm_for_cat,
                nx, ny, nz,
                sx, sy, sz,
                ox, oy, oz,
                varname,
                outputReportFile,
                computationMode,
                None,
                dataPointSet,
                mask,
                probability,
                np.zeros(ncategory),
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                seed,
                nreal,
                logger=logger)

    except Exception as exc:
        err_msg = f'{fname}: cannot fill mpds_geosClassicIndicatorInput C structure'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg) from exc

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

    if verbose > 0 and nth > os.cpu_count():
        if logger:
            logger.warning(f'{fname}: number of threads used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: number of threads used will exceed number of cpu(s) of the system...')

    if verbose > 1:
        if logger:
            logger.info('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        else:
            print('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            sys.stdout.flush()
            sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

    # Free memory on C side: mpds_geosClassicIndicatorInput
    geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
    geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)

    if err:
        # Free memory on C side: mpds_geosClassicOutput
        geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
        geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)
        # Free memory on C side: mpds_progressMonitor
        geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)
        # Raise error
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        err_msg = f'{fname}: {err_message}'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor, logger=logger)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if geosclassic_output is not None and mask is not None and add_data_point_to_mask:
        # Remove the value out of the original mask (using its copy see above)
        geosclassic_output['image'].val[:, mask_original==0.0] = np.nan

    if verbose > 1 and geosclassic_output:
        if logger:
            logger.info(f'{fname}: Geos-Classic run complete')
        else:
            print(f'{fname}: Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose > 0 and geosclassic_output and geosclassic_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({geosclassic_output['nwarning']} times in all):")
            for i, warning_message in enumerate(geosclassic_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

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
        add_data_point_to_mask=True,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        seed=None,
        outputReportFile=None,
        treat_image_one_by_one=False,
        nproc=None, nthreads_per_proc=None,
        verbose=2, 
        logger=None):
    """
    Computes the same as the function :func:`simulateIndicator3D`, using multiprocessing.

    All the parameters are the same as those of the function :func:`simulateIndicator3D`,
    except `nthreads` that is replaced by the parameters `nproc` and
    `nthreads_per_proc`, and an extra parameter `treat_image_one_by_one`.

    This function launches multiple processes (based on `multiprocessing`
    package):

    - `nproc` parallel processes using each one `nthreads_per_proc` threads \
    are launched [parallel calls of the function :func:`simulateIndicator3D`]
    - the set of realizations (specified by `nreal`) is distributed in a \
    balanced way over the processes
    - in terms of resources, this implies the use of `nproc*nthreads_per_proc` \
    cpu(s)

    See function :func:`simulateIndicator3D`.

    **Parameters (new)**
    --------------------
    nproc : int, optional
        number of process(es); by default (`None`):
        `nproc` is set to `min(nmax-1, nreal)` (but at least 1), where nmax is
        the total number of cpu(s) of the system (retrieved by
        `multiprocessing.cpu_count()`)

    nthreads_per_proc : int, optional
        number of thread(s) per process (should be > 0); by default (`None`):
        `nthreads_per_proc` is automatically computed as the maximal integer
        (but at least 1) such that `nproc*nthreads_per_proc <= nmax-1`, where
        nmax is the total number of cpu(s) of the system (retrieved by
        `multiprocessing.cpu_count()`)

    treat_image_one_by_one : bool, default: False
        keyword argument passed to the function :func:`geone.img.gatherImages`:

        - if `True`: images (result of each process) are gathered one by one, \
        i.e. the variables of each image are inserted in an output image one by \
        one and removed from the source (slower, may save memory)
        - if `False`: images (result of each process) are gathered at once, \
        i.e. the variables of all images are inserted in an output image at once, \
        and then removed (faster)
    """
    fname = 'simulateIndicator3D_mp'

    # Set number of process(es): nproc
    if nproc is None:
        nproc = max(min(multiprocessing.cpu_count()-1, nreal), 1)
    else:
        nproc_tmp = nproc
        nproc = max(min(int(nproc), nreal), 1)
        if verbose > 1 and nproc != nproc_tmp:
            if logger:
                logger.info(f'{fname}: number of process(es) has been changed (now: nproc={nproc})')
            else:
                print(f'{fname}: number of process(es) has been changed (now: nproc={nproc})')

    # Set number of threads per process: nth
    if nthreads_per_proc is None:
        nth = max(int(np.floor((multiprocessing.cpu_count()-1) / nproc)), 1)
    else:
        nth = max(int(nthreads_per_proc), 1)
        if verbose > 1 and nth != nthreads_per_proc:
            if logger:
                logger.info(f'{fname}: number of threads per process has been changed (now: nthreads_per_proc={nth})')
            else:
                print(f'{fname}: number of threads per process has been changed (now: nthreads_per_proc={nth})')

    if verbose > 0 and nproc * nth > multiprocessing.cpu_count():
        if logger:
            logger.warning(f'{fname}: total number of cpu(s) used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: total number of cpu(s) used will exceed number of cpu(s) of the system...')

    # Set the distribution of the realizations over the processes
    # Condider the Euclidean division of nreal by nproc:
    #     nreal = q * nproc + r, with 0 <= r < nproc
    # Then, (q+1) realizations will be done on process 0, 1, ..., r-1, and q realization on process r, ..., nproc-1
    # Define the list real_index_proc of length (nproc+1) such that
    #   real_index_proc[i], ..., real_index_proc[i+1] - 1 : are the realization indices run on process i
    q, r = np.divmod(nreal, nproc)
    real_index_proc = [i*q + min(i, r) for i in range(nproc+1)]

    if verbose > 1:
        if logger:
            logger.info('{}: Geos-Classic running on {} process(es)... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, nproc, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        else:
            print('{}: Geos-Classic running on {} process(es)... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, nproc, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            sys.stdout.flush()
            sys.stdout.flush() # twice!, so that the previous print is flushed before launching geos-classic...

    # Prepare seed
    if seed is None:
        seed = np.random.randint(1, 1000000)
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
        verbose_p = 0
        # if i==0:
        #     verbose_p = min(verbose, 1) # allow to print warnings for process i
        # else:
        #     verbose_p = 0
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
                add_data_point_to_mask,
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                seed_p,                      # seed (adjusted)
                outputReportFile_p,          # outputReportFile (adjusted)
                nth,                         # nthreads
                verbose_p),                  # verbose (adjusted)
                kwds={'logger':logger})
            )

    # Properly end working process
    pool.close() # Prevents any more tasks from being submitted to the pool,
    pool.join()  # then, wait for the worker processes to exit.

    # Get result from each process
    geosclassic_output_proc = [p.get() for p in out_pool]

    if np.any([out is None for out in geosclassic_output_proc]):
        return None

    # Gather results from every process
    # image
    image = []
    for out in geosclassic_output_proc:
        if out['image'] is not None:
            image.append(out['image'])
            del(out['image'])
    if len(image) == 0:
        image = None
    # Gather images and adjust variable names
    all_image = img.gatherImages(image, keep_varname=True, rem_var_from_source=True, treat_image_one_by_one=treat_image_one_by_one, logger=logger)
    ndigit = geosclassic.MPDS_GEOS_CLASSIC_NB_DIGIT_FOR_REALIZATION_NUMBER
    for j in range(all_image.nv):
        all_image.varname[j] = all_image.varname[j][:-ndigit] + f'{j:0{ndigit}d}'

    # nwarning
    nwarning = np.sum([out['nwarning'] for out in geosclassic_output_proc])
    # warnings
    warnings = list(np.unique(np.hstack([out['warnings'] for out in geosclassic_output_proc])))

    geosclassic_output = {'image':all_image, 'nwarning':nwarning, 'warnings':warnings}

    if verbose > 1 and geosclassic_output:
        if logger:
            logger.info(f'{fname}: Geos-Classic run complete (all process(es))')
        else:
            print(f'{fname}: Geos-Classic run complete (all process(es))')

    # Show (print) encountered warnings
    if verbose > 0 and geosclassic_output and geosclassic_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({geosclassic_output['nwarning']} times in all):")
            for i, warning_message in enumerate(geosclassic_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

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
        add_data_point_to_mask=True,
        use_unique_neighborhood=False,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2,
        logger=None):
    """
    Computes estimate probabilities / proportions of categories (indicators) in 1D.
    based on simple or ordinary kriging.

    Interpolation (of the indicator variable of each category) takes place in
    (center of) grid cells, based on simple or ordinary kriging.

    Parameters
    ----------
    category_values : 1D array-like
        sequence of category values; let `ncategory` be the number of categories,
        then:

        - if `ncategory=1`: the unique category value given must not be equal to \
        zero; it is used for a binary case with values "unique category value" \
        and 0, where 0 indicates the absence of the considered medium; the \
        conditioning data values should be equal to"unique category value" or 0
        - if `ncategory>=2`: it is used for a multi-category case with given \
        category values (distinct); the conditioning data values should be in the \
        `category_values`

    cov_model_for_category : [sequence of] :class:`geone.covModel.CovModel1D`
        sequence of same length as `category_values` of covariance model in 1D,
        or a unique covariance model in 1D (recycled):
        covariance model for each category

    dimension : int
        `dimension=nx`, number of cells in the 1D simulation grid

    spacing : float, default: 1.0
        `spacing=sx`, cell size

    origin : float, default: 0.0
        `origin=ox`, origin of the 1D simulation grid (left border)

    method : str {'simple_kriging', 'ordinary_kriging'}, default: 'simple_kriging'
        type of kriging

    probability : array-like of floats, optional
        probability for each category:

        - sequence of same length as `category_values`: \
        probability[i]: probability (proportion, kriging mean value for the \
        indicator variable) for category `category_values[i]`, used for \
        every grid cell
        - array-like of size ncategory * ngrid_cells, where ncategory is the \
        length of `category_values` and ngrid_cells is the number of grid \
        cells (the array is reshaped if needed): first ngrid_cells values are \
        the probabilities (proportions, kriging mean values for the indicator \
        variable) for the first category at grid cells, etc. \
        (for non-stationary probailities / proportions)

        By default (`None`): proportion of each category computed from the
        data values (`v`) are used for every grid cell (uniform probabilities
        if no data point)

        Note: for ordinary kriging (`method='ordinary_kriging'`), it is used for
        case with no neighbor

    x : 1D array-like of floats, optional
        data points locations (float coordinates); note: if one point, a float
        is accepted

    v : 1D array-like of floats, optional
        data values at `x` (`v[i]` is the data value at `x[i]`), array of same
        length as `x` (or float if one point)

    mask : array-like, optional
        mask value at grid cells (value 1 for simulated cells, value 0 for not
        simulated cells); the size of the array must be equal to the number of
        grid cells (the array is reshaped if needed)

    add_data_point_to_mask : bool, default: True
        - if `True`: any grid cell that contains a data point is added to (the \
        simulated part of) the mask (if present), i.e. mask value at those cells \
        are set to 1; at the end of the computation the "new mask cells" are \
        removed (by setting a missing value (`numpy.nan`) for the variable out of \
        the original mask)
        - if `False`: original mask is kept as given in input, and data point \
        falling out of (the simulated part of) the mask (if present) are ignored

    use_unique_neighborhood : bool, default: False
        indicates if a unique neighborhood is used:

        - if `True`: all data points are taken into account for computing \
        estimates and standard deviations; in this case: parameters \
        `searchRadiusRelative`, `nneighborMax`, `searchNeighborhoodSortMode` are \
        not used
        - if `False`: only data points within a search ellipsoid are taken into \
        account for computing estimates and standard deviations (see parameters \
        `searchRadiusRelative`, `nneighborMax`, `searchNeighborhoodSortMode`)

    searchRadiusRelative : [sequence of] float(s), default: 1.0
        sequence of floats of same length as `category_values`, or
        a unique float (recycled); one parameter per category:
        indicates how the search ellipsoid is limited (should be positive): let
        r_i be the ranges of the covariance model along its main axes, when
        estimating/simulating a cell x, a cell y is taken into account iff it is
        within the ellipsoid centered at x of half axes equal to
        `searchRadiusRelative` * r_i;
        note: if a range r_i is non-stationary over the grid, its maximal value
        over the grid is considered

    nneighborMax : int, default: 12
        sequence of ints of same length as `category_values`, or
        a unique int (recycled); one parameter per category:
        maximum number of cells retrieved from the search ellipsoid (when
        estimating/simulating a cell), `nneighborMax=-1` for unlimited

    searchNeighborhoodSortMode : int, optional
        sequence of ints of same length as `category_values`, or
        a unique int (recycled); one parameter per category:
        indicates how to sort the search neighboorhood cells (neighbors); they
        are sorted in increasing order according to:

        - `searchNeighborhoodSortMode=0`: distance in the usual axes system
        - `searchNeighborhoodSortMode=1`: distance in the axes sytem supporting \
        the covariance model and accounting for anisotropy given by the ranges
        - `searchNeighborhoodSortMode=2`: minus the evaluation of the covariance \
        model

        Notes:

        - if the covariance model has any non-stationary parameter, then \
        `searchNeighborhoodSortMode=2` is not allowed
        - if the covariance model has any non-stationary range or non-stationary \
        angle and `searchNeighborhoodSortMode=1`: "maximal ranges" (adapted to \
        direction from the central cell) are used to compute distance for sorting \
        the neighbors

        By default (`None`): the greatest possible value is used (i.e. 2 for
        stationary covariance model, or 1 otherwise)

    outputReportFile : str, optional
        name of the report file (if desired in output); by default (`None`): no
        report file

    nthreads : int, default: -1
        number of thread(s) to use for "GeosClassicIndicatorSim" C program;
        `nthreads = -n <= 0`: maximal number of threads of the system except n
        (but at least 1)

    verbose : int, default: 2
        verbose mode, higher implies more printing (info):

        - 0: no display
        - 1: warnings
        - 2: warnings + basic info
        - 3 (or >2): all information

        note that if an error occurred, it is raised
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    geosclassic_output : dict
        geosclassic output in python, dictionary

        {'image':image, 'nwarning':nwarning, 'warnings':warnings}

        with:

        - image : :class:`geone.img.Img`
            output image, with `image.nv=ncategory` variables (probability /
            proportion estimates, of each category);
            note: `image=None` if `mpds_geosClassicOutput->outputImage=NULL`

        - nwarning : int
            total number of warning(s) encountered (same warnings can be counted
            several times)

        - warnings : list of strs
            list of distinct warnings encountered (can be empty)
    """
    fname = 'estimateIndicator1D'

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
        err_msg = f'{fname}: `category_values` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    ncategory = len(category_values)
    if ncategory <= 0:
        err_msg = f'{fname}: `category_values` is empty'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # cov_model_for_category
    cm_for_cat = cov_model_for_category # no need to work on a copy in 1D

    cm_for_cat = np.asarray(cm_for_cat).reshape(-1)
    if len(cm_for_cat) == 1:
        cm_for_cat = np.repeat(cm_for_cat, ncategory)
    elif len(cm_for_cat) != ncategory:
        err_msg = f'{fname}: `cov_model_for_category` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    if not np.all([isinstance(c, gcm.CovModel1D) for c in cm_for_cat]):
        err_msg = f'{fname}: `cov_model_for_category` should contains CovModel1D objects'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    for cov_model in cm_for_cat:
        for el in cov_model.elem:
            # weight
            w = el[1]['w']
            if np.size(w) != 1 and np.size(w) != nxyz:
                err_msg = f"{fname}: covariance model: weight ('w') not compatible with simulation grid"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            # ranges
            if 'r' in el[1].keys():
                r  = el[1]['r']
                if np.size(r) != 1 and np.size(r) != nxyz:
                    err_msg = f"{fname}: covariance model: range ('r') not compatible with simulation grid"
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

            # additional parameter (s)
            if 's' in el[1].keys():
                s  = el[1]['s']
                if np.size(s) != 1 and np.size(s) != nxyz:
                    err_msg = f"{fname}: covariance model: parameter ('s') not compatible with simulation grid"
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    if method == 'simple_kriging':
        computationMode = 1
    elif method == 'ordinary_kriging':
        computationMode = 0
    else:
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - use_unique_neighborhood (length)
    use_unique_neighborhood = np.asarray(use_unique_neighborhood, dtype='bool').reshape(-1)
    if len(use_unique_neighborhood) == 1:
        use_unique_neighborhood = np.repeat(use_unique_neighborhood, ncategory)
    elif len(use_unique_neighborhood) != ncategory:
        err_msg = f'{fname}: `use_unique_neighborhood` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - searchRadiusRelative (length)
    searchRadiusRelative = np.asarray(searchRadiusRelative, dtype='float').reshape(-1)
    if len(searchRadiusRelative) == 1:
        searchRadiusRelative = np.repeat(searchRadiusRelative, ncategory)
    elif len(searchRadiusRelative) != ncategory:
        err_msg = f'{fname}: `searchRadiusRelative` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - nneighborMax (length)
    nneighborMax = np.asarray(nneighborMax, dtype='intc').reshape(-1)
    if len(nneighborMax) == 1:
        nneighborMax = np.repeat(nneighborMax, ncategory)
    elif len(nneighborMax) != ncategory:
        err_msg = f'{fname}: `nneighborMax` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - searchNeighborhoodSortMode (length)
    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode).reshape(-1)
    if len(searchNeighborhoodSortMode) == 1:
        searchNeighborhoodSortMode = np.repeat(searchNeighborhoodSortMode, ncategory)
    elif len(searchNeighborhoodSortMode) != ncategory:
        err_msg = f'{fname}: `searchNeighborhoodSortMode` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

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
                err_msg = f'{fname}: a `searchRadiusRelative` is too small (should be at least {geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN})'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            if nneighborMax[i] != -1 and nneighborMax[i] <= 0:
                err_msg = f'{fname}: any `nneighborMax` should be greater than 0 or equal to -1 (unlimited)'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            if searchNeighborhoodSortMode[i] is None:
                # set greatest possible value
                if cm_for_cat[i].is_stationary():
                    searchNeighborhoodSortMode[i] = 2
                else:
                    searchNeighborhoodSortMode[i] = 1
            else:
                if searchNeighborhoodSortMode[i] == 2:
                    if not cm_for_cat[i].is_stationary():
                        err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
                        if logger: logger.error(err_msg)
                        raise GeosclassicinterfaceError(err_msg)

            # if searchNeighborhoodSortMode[i] is None:
            #     # set greatest possible value
            #     if cm_for_cat[i].is_stationary():
            #         searchNeighborhoodSortMode[i] = 2
            #     elif cm_for_cat[i].is_orientation_stationary() and cm_for_cat[i].is_range_stationary():
            #         searchNeighborhoodSortMode[i] = 1
            #     else:
            #         searchNeighborhoodSortMode[i] = 0
            # else:
            #     if searchNeighborhoodSortMode[i] == 2:
            #         if not cm_for_cat[i].is_stationary():
            #             err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
            #             if logger: logger.error(err_msg)
            #             raise GeosclassicinterfaceError(err_msg)
            #     elif searchNeighborhoodSortMode[i] == 1:
            #         if not cm_for_cat[i].is_orientation_stationary() or not cm_for_cat[i].is_range_stationary():
            #             err_msg = f'{fname}: `searchNeighborhoodSortMode=1` not allowed with non-stationary range or non-stationary orientation in covariance model'
            #             if logger: logger.error(err_msg)
            #             raise GeosclassicinterfaceError(err_msg)

    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode, dtype='intc')

    # data points: x, v
    dataPointSet = []

    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 1) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # Aggregate data on grid by taking the most frequent value in grid cell
        xx = x[:, 0]
        yy = np.ones_like(xx) * oy + 0.5 * sy
        zz = np.ones_like(xx) * oz + 0.5 * sz
        try:
            xx_agg, yy_agg, zz_agg, v_agg = img.aggregateDataPointsWrtGrid(
                                                xx, yy, zz, v,
                                                nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                op='most_freq',
                                                logger=logger)
        except Exception as exc:
            err_msg = f"{fname}: data aggregation ('most_freq') failed"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

        if len(xx_agg) == 0:
            err_msg = f'{fname}: no data point in grid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        dataPointSet.append(
            PointSet(npt=v_agg.shape[0], nv=4, val=np.array((xx_agg, yy_agg, zz_agg, v_agg)), varname=['X', 'Y', 'Z', varname])
            )

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            err_msg = f'{fname}: `mask` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if mask is not None and add_data_point_to_mask:
        # Make a copy of the original mask, to remove value in added mask cell at the end
        mask_original = np.copy(mask)
        # Add cell to mask if needed
        pts = np.zeros((0,3))
        if x is not None:
            pts = np.vstack((pts, np.array((xx_agg, yy_agg, zz_agg)).T))
        if pts.shape[0]:
            im_tmp = img.imageFromPoints(pts,
                        nx=nx, ny=ny, nz=nz,
                        sx=sx, sy=sy, sz=sz,
                        ox=ox, oy=oy, oz=oz,
                        indicator_var=True,
                        logger=logger)
            mask = 1.0*np.any((im_tmp.val[0], mask), axis=0)
            del(im_tmp)
        del(pts)

    # Check parameters - probability
    if probability is not None:
        # if method == 'ordinary_kriging':
        #     if verbose > 0:
        #         print(f"ERROR ({fname}): specifying 'probability' not allowed with ordinary kriging")
        #     return None
        probability = np.asarray(probability, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if probability.size not in (ncategory, ncategory*nxyz):
            err_msg = f'{fname}: size of `probability` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # --- Fill mpds_geosClassicInput structure (C)
    try:
        mpds_geosClassicIndicatorInput = fill_mpds_geosClassicIndicatorInput(
                False,
                space_dim,
                ncategory,
                category_values,
                cm_for_cat,
                nx, ny, nz,
                sx, sy, sz,
                ox, oy, oz,
                varname,
                outputReportFile,
                computationMode,
                None,
                dataPointSet,
                mask,
                probability,
                np.zeros(ncategory),
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                0,
                0,
                logger=logger)

    except Exception as exc:
        err_msg = f'{fname}: cannot fill mpds_geosClassicIndicatorInput C structure'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg) from exc

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

    if verbose > 0 and nth > os.cpu_count():
        if logger:
            logger.warning(f'{fname}: number of threads used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: number of threads used will exceed number of cpu(s) of the system...')

    if verbose > 1:
        if logger:
            logger.info('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        else:
            print('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            sys.stdout.flush()
            sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

    # Free memory on C side: mpds_geosClassicIndicatorInput
    geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
    geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)

    if err:
        # Free memory on C side: mpds_geosClassicOutput
        geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
        geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)
        # Free memory on C side: mpds_progressMonitor
        geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)
        # Raise error
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        err_msg = f'{fname}: {err_message}'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor, logger=logger)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if geosclassic_output is not None and mask is not None and add_data_point_to_mask:
        # Remove the value out of the original mask (using its copy see above)
        geosclassic_output['image'].val[:, mask_original==0.0] = np.nan

    if verbose > 1 and geosclassic_output:
        if logger:
            logger.info(f'{fname}: Geos-Classic run complete')
        else:
            print(f'{fname}: Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose > 0 and geosclassic_output and geosclassic_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({geosclassic_output['nwarning']} times in all):")
            for i, warning_message in enumerate(geosclassic_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

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
        add_data_point_to_mask=True,
        use_unique_neighborhood=False,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2,
        logger=None):
    """
    Computes estimate probabilities / proportions of categories (indicators) in 2D.
    based on simple or ordinary kriging.

    Interpolation (of the indicator variable of each category) takes place in
    (center of) grid cells, based on simple or ordinary kriging.

    Parameters
    ----------
    category_values : 1D array-like
        sequence of category values; let `ncategory` be the number of categories,
        then:

        - if `ncategory=1`: the unique category value given must not be equal to \
        zero; it is used for a binary case with values "unique category value" \
        and 0, where 0 indicates the absence of the considered medium; the \
        conditioning data values should be equal to"unique category value" or 0
        - if `ncategory>=2`: it is used for a multi-category case with given \
        category values (distinct); the conditioning data values should be in the \
        `category_values`

    cov_model_for_category : [sequence of] :class:`geone.covModel.CovModel2D`
        sequence of same length as `category_values` of covariance model in 2D,
        or a unique covariance model in 2D (recycled):
        covariance model for each category

    dimension : 2-tuple of ints
        `dimension=(nx, ny)`, number of cells in the 2D simulation grid along
        each axis

    spacing : 2-tuple of floats, default: (1.0, 1.0)
        `spacing=(sx, sy)`, cell size along each axis

    origin : 2-tuple of floats, default: (0.0, 0.0)
        `origin=(ox, oy)`, origin of the 2D simulation grid (lower-left corner)

    method : str {'simple_kriging', 'ordinary_kriging'}, default: 'simple_kriging'
        type of kriging

    probability : array-like of floats, optional
        probability for each category:

        - sequence of same length as `category_values`: \
        probability[i]: probability (proportion, kriging mean value for the \
        indicator variable) for category `category_values[i]`, used for \
        every grid cell
        - array-like of size ncategory * ngrid_cells, where ncategory is the \
        length of `category_values` and ngrid_cells is the number of grid \
        cells (the array is reshaped if needed): first ngrid_cells values are \
        the probabilities (proportions, kriging mean values for the indicator \
        variable) for the first category at grid cells, etc. \
        (for non-stationary probailities / proportions)

        By default (`None`): proportion of each category computed from the
        data values (`v`) are used for every grid cell (uniform probabilities
        if no data point)

        Note: for ordinary kriging (`method='ordinary_kriging'`), it is used for
        case with no neighbor

    x : 2D array of floats of shape (n, 2), optional
        data points locations, with n the number of data points, each row of `x`
        is the float coordinates of one data point; note: if n=1, a 1D array of
        shape (2,) is accepted

    v : 1D array of floats of shape (n,), optional
        data values at `x` (`v[i]` is the data value at `x[i]`)

    mask : array-like, optional
        mask value at grid cells (value 1 for simulated cells, value 0 for not
        simulated cells); the size of the array must be equal to the number of
        grid cells (the array is reshaped if needed)

    add_data_point_to_mask : bool, default: True
        - if `True`: any grid cell that contains a data point is added to (the \
        simulated part of) the mask (if present), i.e. mask value at those cells \
        are set to 1; at the end of the computation the "new mask cells" are \
        removed (by setting a missing value (`numpy.nan`) for the variable out of \
        the original mask)
        - if `False`: original mask is kept as given in input, and data point \
        falling out of (the simulated part of) the mask (if present) are ignored

    use_unique_neighborhood : bool, default: False
        indicates if a unique neighborhood is used:

        - if `True`: all data points are taken into account for computing \
        estimates and standard deviations; in this case: parameters \
        `searchRadiusRelative`, `nneighborMax`, `searchNeighborhoodSortMode` are \
        not used
        - if `False`: only data points within a search ellipsoid are taken into \
        account for computing estimates and standard deviations (see parameters \
        `searchRadiusRelative`, `nneighborMax`, `searchNeighborhoodSortMode`)

    searchRadiusRelative : [sequence of] float(s), default: 1.0
        sequence of floats of same length as `category_values`, or
        a unique float (recycled); one parameter per category:
        indicates how the search ellipsoid is limited (should be positive): let
        r_i be the ranges of the covariance model along its main axes, when
        estimating/simulating a cell x, a cell y is taken into account iff it is
        within the ellipsoid centered at x of half axes equal to
        `searchRadiusRelative` * r_i;
        note: if a range r_i is non-stationary over the grid, its maximal value
        over the grid is considered

    nneighborMax : int, default: 12
        sequence of ints of same length as `category_values`, or
        a unique int (recycled); one parameter per category:
        maximum number of cells retrieved from the search ellipsoid (when
        estimating/simulating a cell), `nneighborMax=-1` for unlimited

    searchNeighborhoodSortMode : int, optional
        sequence of ints of same length as `category_values`, or
        a unique int (recycled); one parameter per category:
        indicates how to sort the search neighboorhood cells (neighbors); they
        are sorted in increasing order according to:

        - `searchNeighborhoodSortMode=0`: distance in the usual axes system
        - `searchNeighborhoodSortMode=1`: distance in the axes sytem supporting \
        the covariance model and accounting for anisotropy given by the ranges
        - `searchNeighborhoodSortMode=2`: minus the evaluation of the covariance \
        model

        Notes:

        - if the covariance model has any non-stationary parameter, then \
        `searchNeighborhoodSortMode=2` is not allowed
        - if the covariance model has any non-stationary range or non-stationary \
        angle and `searchNeighborhoodSortMode=1`: "maximal ranges" (adapted to \
        direction from the central cell) are used to compute distance for sorting \
        the neighbors

        By default (`None`): the greatest possible value is used (i.e. 2 for
        stationary covariance model, or 1 otherwise)

    outputReportFile : str, optional
        name of the report file (if desired in output); by default (`None`): no
        report file

    nthreads : int, default: -1
        number of thread(s) to use for "GeosClassicIndicatorSim" C program;
        `nthreads = -n <= 0`: maximal number of threads of the system except n
        (but at least 1)

    verbose : int, default: 2
        verbose mode, higher implies more printing (info):

        - 0: no display
        - 1: warnings
        - 2: warnings + basic info
        - 3 (or >2): all information

        note that if an error occurred, it is raised
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    geosclassic_output : dict
        geosclassic output in python, dictionary

        {'image':image, 'nwarning':nwarning, 'warnings':warnings}

        with:

        - image : :class:`geone.img.Img`
            output image, with `image.nv=ncategory` variables (probability /
            proportion estimates, of each category);
            note: `image=None` if `mpds_geosClassicOutput->outputImage=NULL`

        - nwarning : int
            total number of warning(s) encountered (same warnings can be counted
            several times)

        - warnings : list of strs
            list of distinct warnings encountered (can be empty)
    """
    fname = 'estimateIndicator2D'

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
        err_msg = f'{fname}: `category_values` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    ncategory = len(category_values)
    if ncategory <= 0:
        err_msg = f'{fname}: `category_values` is empty'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # cov_model_for_category
    cov_model_for_category = np.asarray(cov_model_for_category).reshape(-1)
    if not np.all([isinstance(c, gcm.CovModel2D) for c in cov_model_for_category]):
        # cov model will be converted:
        #    as applying modification in an array is persistent at exit,
        #    work on a copy to ensure no modification of the initial entry
        cm_for_cat = copy.deepcopy(cov_model_for_category)
    else:
        cm_for_cat = cov_model_for_category

    cm_for_cat = np.asarray(cm_for_cat).reshape(-1)
    for i in range(len(cm_for_cat)):
        if isinstance(cm_for_cat[i], gcm.CovModel1D):
            cm_for_cat[i] = gcm.covModel1D_to_covModel2D(cm_for_cat[i]) # convert model 1D in 2D
    if len(cm_for_cat) == 1:
        cm_for_cat = np.repeat(cm_for_cat, ncategory)
    elif len(cm_for_cat) != ncategory:
        err_msg = f'{fname}: `cov_model_for_category` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    if not np.all([isinstance(c, gcm.CovModel2D) for c in cm_for_cat]):
        err_msg = f'{fname}: `cov_model_for_category` should contains CovModel2D objects'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    for cov_model in cm_for_cat:
        for el in cov_model.elem:
            # weight
            w = el[1]['w']
            if np.size(w) != 1 and np.size(w) != nxyz:
                err_msg = f"{fname}: covariance model: weight ('w') not compatible with simulation grid"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            # ranges
            if 'r' in el[1].keys():
                for r in el[1]['r']:
                    if np.size(r) != 1 and np.size(r) != nxyz:
                        err_msg = f"{fname}: covariance model: range ('r') not compatible with simulation grid"
                        if logger: logger.error(err_msg)
                        raise GeosclassicinterfaceError(err_msg)

            # additional parameter (s)
            if 's' in el[1].keys():
                s  = el[1]['s']
                if np.size(s) != 1 and np.size(s) != nxyz:
                    err_msg = f"{fname}: covariance model: parameter ('s') not compatible with simulation grid"
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

        # alpha
        angle = cov_model.alpha
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            err_msg = f"{fname}: covariance model: angle ('alpha') not compatible with simulation grid"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    if method == 'simple_kriging':
        computationMode = 1
    elif method == 'ordinary_kriging':
        computationMode = 0
    else:
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - use_unique_neighborhood (length)
    use_unique_neighborhood = np.asarray(use_unique_neighborhood, dtype='bool').reshape(-1)
    if len(use_unique_neighborhood) == 1:
        use_unique_neighborhood = np.repeat(use_unique_neighborhood, ncategory)
    elif len(use_unique_neighborhood) != ncategory:
        err_msg = f'{fname}: `use_unique_neighborhood` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - searchRadiusRelative (length)
    searchRadiusRelative = np.asarray(searchRadiusRelative, dtype='float').reshape(-1)
    if len(searchRadiusRelative) == 1:
        searchRadiusRelative = np.repeat(searchRadiusRelative, ncategory)
    elif len(searchRadiusRelative) != ncategory:
        err_msg = f'{fname}: `searchRadiusRelative` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - nneighborMax (length)
    nneighborMax = np.asarray(nneighborMax, dtype='intc').reshape(-1)
    if len(nneighborMax) == 1:
        nneighborMax = np.repeat(nneighborMax, ncategory)
    elif len(nneighborMax) != ncategory:
        err_msg = f'{fname}: `nneighborMax` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - searchNeighborhoodSortMode (length)
    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode).reshape(-1)
    if len(searchNeighborhoodSortMode) == 1:
        searchNeighborhoodSortMode = np.repeat(searchNeighborhoodSortMode, ncategory)
    elif len(searchNeighborhoodSortMode) != ncategory:
        err_msg = f'{fname}: `searchNeighborhoodSortMode` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

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
                err_msg = f'{fname}: a `searchRadiusRelative` is too small (should be at least {geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN})'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            if nneighborMax[i] != -1 and nneighborMax[i] <= 0:
                err_msg = f'{fname}: any `nneighborMax` should be greater than 0 or equal to -1 (unlimited)'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            if searchNeighborhoodSortMode[i] is None:
                # set greatest possible value
                if cm_for_cat[i].is_stationary():
                    searchNeighborhoodSortMode[i] = 2
                else:
                    searchNeighborhoodSortMode[i] = 1
            else:
                if searchNeighborhoodSortMode[i] == 2:
                    if not cm_for_cat[i].is_stationary():
                        err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
                        if logger: logger.error(err_msg)
                        raise GeosclassicinterfaceError(err_msg)

            # if searchNeighborhoodSortMode[i] is None:
            #     # set greatest possible value
            #     if cm_for_cat[i].is_stationary():
            #         searchNeighborhoodSortMode[i] = 2
            #     elif cm_for_cat[i].is_orientation_stationary() and cm_for_cat[i].is_range_stationary():
            #         searchNeighborhoodSortMode[i] = 1
            #     else:
            #         searchNeighborhoodSortMode[i] = 0
            # else:
            #     if searchNeighborhoodSortMode[i] == 2:
            #         if not cm_for_cat[i].is_stationary():
            #             err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
            #             if logger: logger.error(err_msg)
            #             raise GeosclassicinterfaceError(err_msg)
            #     elif searchNeighborhoodSortMode[i] == 1:
            #         if not cm_for_cat[i].is_orientation_stationary() or not cm_for_cat[i].is_range_stationary():
            #             err_msg = f'{fname}: `searchNeighborhoodSortMode=1` not allowed with non-stationary range or non-stationary orientation in covariance model'
            #             if logger: logger.error(err_msg)
            #             raise GeosclassicinterfaceError(err_msg)

    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode, dtype='intc')

    # data points: x, v
    dataPointSet = []

    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 2) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # Aggregate data on grid by taking the most frequent value in grid cell
        xx, yy = x.T
        zz = np.ones_like(xx) * oz + 0.5 * sz
        try:
            xx_agg, yy_agg, zz_agg, v_agg = img.aggregateDataPointsWrtGrid(
                                                xx, yy, zz, v,
                                                nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                op='most_freq',
                                                logger=logger)
        except Exception as exc:
            err_msg = f"{fname}: data aggregation ('most_freq') failed"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

        if len(xx_agg) == 0:
            err_msg = f'{fname}: no data point in grid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        dataPointSet.append(
            PointSet(npt=v_agg.shape[0], nv=4, val=np.array((xx_agg, yy_agg, zz_agg, v_agg)), varname=['X', 'Y', 'Z', varname])
            )

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            err_msg = f'{fname}: `mask` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if mask is not None and add_data_point_to_mask:
        # Make a copy of the original mask, to remove value in added mask cell at the end
        mask_original = np.copy(mask)
        # Add cell to mask if needed
        pts = np.zeros((0,3))
        if x is not None:
            pts = np.vstack((pts, np.array((xx_agg, yy_agg, zz_agg)).T))
        if pts.shape[0]:
            im_tmp = img.imageFromPoints(pts,
                        nx=nx, ny=ny, nz=nz,
                        sx=sx, sy=sy, sz=sz,
                        ox=ox, oy=oy, oz=oz,
                        indicator_var=True,
                        logger=logger)
            mask = 1.0*np.any((im_tmp.val[0], mask), axis=0)
            del(im_tmp)
        del(pts)

    # Check parameters - probability
    if probability is not None:
        # if method == 'ordinary_kriging':
        #     if verbose > 0:
        #         print(f"ERROR ({fname}): specifying `probability` not allowed with ordinary kriging")
        #     return None
        probability = np.asarray(probability, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if probability.size not in (ncategory, ncategory*nxyz):
            err_msg = f'{fname}: size of `probability` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # --- Fill mpds_geosClassicInput structure (C)
    try:
        mpds_geosClassicIndicatorInput = fill_mpds_geosClassicIndicatorInput(
                False,
                space_dim,
                ncategory,
                category_values,
                cm_for_cat,
                nx, ny, nz,
                sx, sy, sz,
                ox, oy, oz,
                varname,
                outputReportFile,
                computationMode,
                None,
                dataPointSet,
                mask,
                probability,
                np.zeros(ncategory),
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                0,
                0,
                logger=logger)

    except Exception as exc:
        err_msg = f'{fname}: cannot fill mpds_geosClassicIndicatorInput C structure'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg) from exc

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

    if verbose > 0 and nth > os.cpu_count():
        if logger:
            logger.warning(f'{fname}: number of threads used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: number of threads used will exceed number of cpu(s) of the system...')

    if verbose > 1:
        if logger:
            logger.info('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        else:
            print('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            sys.stdout.flush()
            sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

    # Free memory on C side: mpds_geosClassicIndicatorInput
    geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
    geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)

    if err:
        # Free memory on C side: mpds_geosClassicOutput
        geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
        geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)
        # Free memory on C side: mpds_progressMonitor
        geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)
        # Raise error
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        err_msg = f'{fname}: {err_message}'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor, logger=logger)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if geosclassic_output is not None and mask is not None and add_data_point_to_mask:
        # Remove the value out of the original mask (using its copy see above)
        geosclassic_output['image'].val[:, mask_original==0.0] = np.nan

    if verbose > 1 and geosclassic_output:
        if logger:
            logger.info(f'{fname}: Geos-Classic run complete')
        else:
            print(f'{fname}: Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose > 0 and geosclassic_output and geosclassic_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({geosclassic_output['nwarning']} times in all):")
            for i, warning_message in enumerate(geosclassic_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

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
        add_data_point_to_mask=True,
        use_unique_neighborhood=False,
        searchRadiusRelative=1.2,
        nneighborMax=12,
        searchNeighborhoodSortMode=None,
        outputReportFile=None,
        nthreads=-1,
        verbose=2, 
        logger=None):
    """
    Computes estimate probabilities / proportions of categories (indicators) in 2D.
    based on simple or ordinary kriging.

    Interpolation (of the indicator variable of each category) takes place in
    (center of) grid cells, based on simple or ordinary kriging.

    Parameters
    ----------
    category_values : 1D array-like
        sequence of category values; let `ncategory` be the number of categories,
        then:

        - if `ncategory=1`: the unique category value given must not be equal to \
        zero; it is used for a binary case with values "unique category value" \
        and 0, where 0 indicates the absence of the considered medium; the \
        conditioning data values should be equal to "unique category value" or 0
        - if `ncategory>=2`: it is used for a multi-category case with given \
        category values (distinct); the conditioning data values should be in the \
        `category_values`

    cov_model_for_category : [sequence of] :class:`geone.covModel.CovModel3D`
        sequence of same length as `category_values` of covariance model in 3D,
        or a unique covariance model in 3D (recycled):
        covariance model for each category

    dimension : 3-tuple of ints
        `dimension=(nx, ny, nz)`, number of cells in the 3D simulation grid along
        each axis

    spacing : 3-tuple of floats, default: (1.0,1.0, 1.0)
        `spacing=(sx, sy, sz)`, cell size along each axis

    origin : 3-tuple of floats, default: (0.0, 0.0, 0.0)
        `origin=(ox, oy, oz)`, origin of the 3D simulation grid (bottom-lower-left
        corner)

    method : str {'simple_kriging', 'ordinary_kriging'}, default: 'simple_kriging'
        type of kriging

    probability : array-like of floats, optional
        probability for each category:

        - sequence of same length as `category_values`: \
        probability[i]: probability (proportion, kriging mean value for the \
        indicator variable) for category `category_values[i]`, used for \
        every grid cell
        - array-like of size ncategory * ngrid_cells, where ncategory is the \
        length of `category_values` and ngrid_cells is the number of grid \
        cells (the array is reshaped if needed): first ngrid_cells values are \
        the probabilities (proportions, kriging mean values for the indicator \
        variable) for the first category at grid cells, etc. \
        (for non-stationary probailities / proportions)

        By default (`None`): proportion of each category computed from the
        data values (`v`) are used for every grid cell (uniform probabilities
        if no data point)

        Note: for ordinary kriging (`method='ordinary_kriging'`), it is used for
        case with no neighbor

    x : 2D array of floats of shape (n, 3), optional
        data points locations, with n the number of data points, each row of `x`
        is the float coordinates of one data point; note: if n=1, a 1D array of
        shape (3,) is accepted

    v : 1D array of floats of shape (n,), optional
        data values at `x` (`v[i]` is the data value at `x[i]`)

    mask : array-like, optional
        mask value at grid cells (value 1 for simulated cells, value 0 for not
        simulated cells); the size of the array must be equal to the number of
        grid cells (the array is reshaped if needed)

    add_data_point_to_mask : bool, default: True
        - if `True`: any grid cell that contains a data point is added to (the \
        simulated part of) the mask (if present), i.e. mask value at those cells \
        are set to 1; at the end of the computation the "new mask cells" are \
        removed (by setting a missing value (`numpy.nan`) for the variable out of \
        the original mask)
        - if `False`: original mask is kept as given in input, and data point \
        falling out of (the simulated part of) the mask (if present) are ignored

    use_unique_neighborhood : bool, default: False
        indicates if a unique neighborhood is used:

        - if `True`: all data points are taken into account for computing \
        estimates and standard deviations; in this case: parameters \
        `searchRadiusRelative`, `nneighborMax`, `searchNeighborhoodSortMode` are \
        not used
        - if `False`: only data points within a search ellipsoid are taken into \
        account for computing estimates and standard deviations (see parameters \
        `searchRadiusRelative`, `nneighborMax`, `searchNeighborhoodSortMode`)

    searchRadiusRelative : [sequence of] float(s), default: 1.0
        sequence of floats of same length as `category_values`, or
        a unique float (recycled); one parameter per category:
        indicates how the search ellipsoid is limited (should be positive): let
        r_i be the ranges of the covariance model along its main axes, when
        estimating/simulating a cell x, a cell y is taken into account iff it is
        within the ellipsoid centered at x of half axes equal to
        `searchRadiusRelative` * r_i;
        note: if a range r_i is non-stationary over the grid, its maximal value
        over the grid is considered

    nneighborMax : int, default: 12
        sequence of ints of same length as `category_values`, or
        a unique int (recycled); one parameter per category:
        maximum number of cells retrieved from the search ellipsoid (when
        estimating/simulating a cell), `nneighborMax=-1` for unlimited

    searchNeighborhoodSortMode : int, optional
        sequence of ints of same length as `category_values`, or
        a unique int (recycled); one parameter per category:
        indicates how to sort the search neighboorhood cells (neighbors); they
        are sorted in increasing order according to:

        - `searchNeighborhoodSortMode=0`: distance in the usual axes system
        - `searchNeighborhoodSortMode=1`: distance in the axes sytem supporting \
        the covariance model and accounting for anisotropy given by the ranges
        - `searchNeighborhoodSortMode=2`: minus the evaluation of the covariance \
        model

        Notes:

        - if the covariance model has any non-stationary parameter, then \
        `searchNeighborhoodSortMode=2` is not allowed
        - if the covariance model has any non-stationary range or non-stationary \
        angle and `searchNeighborhoodSortMode=1`: "maximal ranges" (adapted to \
        direction from the central cell) are used to compute distance for sorting \
        the neighbors

        By default (`None`): the greatest possible value is used (i.e. 2 for
        stationary covariance model, or 1 otherwise)

    outputReportFile : str, optional
        name of the report file (if desired in output); by default (`None`): no
        report file

    nthreads : int, default: -1
        number of thread(s) to use for "GeosClassicIndicatorSim" C program;
        `nthreads = -n <= 0`: maximal number of threads of the system except n
        (but at least 1)

    verbose : int, default: 2
        verbose mode, higher implies more printing (info):

        - 0: no display
        - 1: warnings
        - 2: warnings + basic info
        - 3 (or >2): all information

        note that if an error occurred, it is raised
    
    logger : :class:`logging.Logger`, optional
        logger (see package `logging`)
        if specified, messages are written via `logger` (no print)

    Returns
    -------
    geosclassic_output : dict
        geosclassic output in python, dictionary

        {'image':image, 'nwarning':nwarning, 'warnings':warnings}

        with:

        - image : :class:`geone.img.Img`
            output image, with `image.nv=ncategory` variables (probability /
            proportion estimates, of each category);
            note: `image=None` if `mpds_geosClassicOutput->outputImage=NULL`

        - nwarning : int
            total number of warning(s) encountered (same warnings can be counted
            several times)

        - warnings : list of strs
            list of distinct warnings encountered (can be empty)
    """
    fname = 'estimateIndicator3D'

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
        err_msg = f'{fname}: `category_values` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    ncategory = len(category_values)
    if ncategory <= 0:
        err_msg = f'{fname}: `category_values` is empty'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # cov_model_for_category
    cov_model_for_category = np.asarray(cov_model_for_category).reshape(-1)
    if not np.all([isinstance(c, gcm.CovModel3D) for c in cov_model_for_category]):
        # cov model will be converted:
        #    as applying modification in an array is persistent at exit,
        #    work on a copy to ensure no modification of the initial entry
        cm_for_cat = copy.deepcopy(cov_model_for_category)
    else:
        cm_for_cat = cov_model_for_category

    cm_for_cat = np.asarray(cm_for_cat).reshape(-1)
    for i in range(len(cm_for_cat)):
        if isinstance(cm_for_cat[i], gcm.CovModel1D):
            cm_for_cat[i] = gcm.covModel1D_to_covModel3D(cm_for_cat[i]) # convert model 1D in 3D
    if len(cm_for_cat) == 1:
        cm_for_cat = np.repeat(cm_for_cat, ncategory)
    elif len(cm_for_cat) != ncategory:
        err_msg = f'{fname}: `cov_model_for_category` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    if not np.all([isinstance(c, gcm.CovModel3D) for c in cm_for_cat]):
        err_msg = f'{fname}: `cov_model_for_category` should contains CovModel3D objects'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    for cov_model in cm_for_cat:
        for el in cov_model.elem:
            # weight
            w = el[1]['w']
            if np.size(w) != 1 and np.size(w) != nxyz:
                err_msg = f"{fname}: covariance model: weight ('w') not compatible with simulation grid"
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            # ranges
            if 'r' in el[1].keys():
                for r in el[1]['r']:
                    if np.size(r) != 1 and np.size(r) != nxyz:
                        err_msg = f"{fname}: covariance model: range ('r') not compatible with simulation grid"
                        if logger: logger.error(err_msg)
                        raise GeosclassicinterfaceError(err_msg)

            # additional parameter (s)
            if 's' in el[1].keys():
                s  = el[1]['s']
                if np.size(s) != 1 and np.size(s) != nxyz:
                    err_msg = f"{fname}: covariance model: parameter ('s') not compatible with simulation grid"
                    if logger: logger.error(err_msg)
                    raise GeosclassicinterfaceError(err_msg)

        # alpha
        angle = cov_model.alpha
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            err_msg = f"{fname}: covariance model: angle ('alpha') not compatible with simulation grid"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # beta
        angle = cov_model.beta
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            err_msg = f"{fname}: covariance model: angle ('beta') not compatible with simulation grid"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # gamma
        angle = cov_model.gamma
        if np.size(angle) != 1 and np.size(angle) != nxyz:
            err_msg = f"{fname}: covariance model: angle ('gamma') not compatible with simulation grid"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # method
    #    computationMode=0: GEOS_CLASSIC_OK
    #    computationMode=1: GEOS_CLASSIC_SK
    #    computationMode=2: GEOS_CLASSIC_SIM_OK
    #    computationMode=3: GEOS_CLASSIC_SIM_SK
    if method == 'simple_kriging':
        computationMode = 1
    elif method == 'ordinary_kriging':
        computationMode = 0
    else:
        err_msg = f'{fname}: `method` invalid'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - use_unique_neighborhood (length)
    use_unique_neighborhood = np.asarray(use_unique_neighborhood, dtype='bool').reshape(-1)
    if len(use_unique_neighborhood) == 1:
        use_unique_neighborhood = np.repeat(use_unique_neighborhood, ncategory)
    elif len(use_unique_neighborhood) != ncategory:
        err_msg = f'{fname}: `use_unique_neighborhood` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - searchRadiusRelative (length)
    searchRadiusRelative = np.asarray(searchRadiusRelative, dtype='float').reshape(-1)
    if len(searchRadiusRelative) == 1:
        searchRadiusRelative = np.repeat(searchRadiusRelative, ncategory)
    elif len(searchRadiusRelative) != ncategory:
        err_msg = f'{fname}: `searchRadiusRelative` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - nneighborMax (length)
    nneighborMax = np.asarray(nneighborMax, dtype='intc').reshape(-1)
    if len(nneighborMax) == 1:
        nneighborMax = np.repeat(nneighborMax, ncategory)
    elif len(nneighborMax) != ncategory:
        err_msg = f'{fname}: `nneighborMax` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    # Check parameters - searchNeighborhoodSortMode (length)
    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode).reshape(-1)
    if len(searchNeighborhoodSortMode) == 1:
        searchNeighborhoodSortMode = np.repeat(searchNeighborhoodSortMode, ncategory)
    elif len(searchNeighborhoodSortMode) != ncategory:
        err_msg = f'{fname}: `searchNeighborhoodSortMode` of invalid length'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

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
                err_msg = f'{fname}: a `searchRadiusRelative` is too small (should be at least {geosclassic.MPDS_GEOSCLASSIC_SEARCHRADIUSRELATIVE_MIN})'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            if nneighborMax[i] != -1 and nneighborMax[i] <= 0:
                err_msg = f'{fname}: any `nneighborMax` should be greater than 0 or equal to -1 (unlimited)'
                if logger: logger.error(err_msg)
                raise GeosclassicinterfaceError(err_msg)

            if searchNeighborhoodSortMode[i] is None:
                # set greatest possible value
                if cm_for_cat[i].is_stationary():
                    searchNeighborhoodSortMode[i] = 2
                else:
                    searchNeighborhoodSortMode[i] = 1
            else:
                if searchNeighborhoodSortMode[i] == 2:
                    if not cm_for_cat[i].is_stationary():
                        err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
                        if logger: logger.error(err_msg)
                        raise GeosclassicinterfaceError(err_msg)

            # if searchNeighborhoodSortMode[i] is None:
            #     # set greatest possible value
            #     if cm_for_cat[i].is_stationary():
            #         searchNeighborhoodSortMode[i] = 2
            #     elif cm_for_cat[i].is_orientation_stationary() and cm_for_cat[i].is_range_stationary():
            #         searchNeighborhoodSortMode[i] = 1
            #     else:
            #         searchNeighborhoodSortMode[i] = 0
            # else:
            #     if searchNeighborhoodSortMode[i] == 2:
            #         if not cm_for_cat[i].is_stationary():
            #             err_msg = f'{fname}: `searchNeighborhoodSortMode=2` not allowed with non-stationary covariance model'
            #             if logger: logger.error(err_msg)
            #             raise GeosclassicinterfaceError(err_msg)
            #     elif searchNeighborhoodSortMode[i] == 1:
            #         if not cm_for_cat[i].is_orientation_stationary() or not cm_for_cat[i].is_range_stationary():
            #             err_msg = f'{fname}: `searchNeighborhoodSortMode=1` not allowed with non-stationary range or non-stationary orientation in covariance model'
            #             if logger: logger.error(err_msg)
            #             raise GeosclassicinterfaceError(err_msg)

    searchNeighborhoodSortMode = np.asarray(searchNeighborhoodSortMode, dtype='intc')

    # data points: x, v
    dataPointSet = []

    if x is not None:
        x = np.asarray(x, dtype='float').reshape(-1, 3) # cast in 2-dimensional array if needed
        v = np.asarray(v, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if len(v) != x.shape[0]:
            err_msg = f'{fname}: length of `v` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        # Aggregate data on grid by taking the most frequent value in grid cell
        xx, yy, zz = x.T
        try:
            xx_agg, yy_agg, zz_agg, v_agg = img.aggregateDataPointsWrtGrid(
                                                xx, yy, zz, v,
                                                nx, ny, nz, sx, sy, sz, ox, oy, oz,
                                                op='most_freq',
                                                logger=logger)
        except Exception as exc:
            err_msg = f"{fname}: data aggregation ('most_freq') failed"
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg) from exc

        if len(xx_agg) == 0:
            err_msg = f'{fname}: no data point in grid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

        dataPointSet.append(
            PointSet(npt=v_agg.shape[0], nv=4, val=np.array((xx_agg, yy_agg, zz_agg, v_agg)), varname=['X', 'Y', 'Z', varname])
            )

    # Check parameters - mask
    if mask is not None:
        try:
            mask = np.asarray(mask).reshape(nz, ny, nx)
        except:
            err_msg = f'{fname}: `mask` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    if mask is not None and add_data_point_to_mask:
        # Make a copy of the original mask, to remove value in added mask cell at the end
        mask_original = np.copy(mask)
        # Add cell to mask if needed
        pts = np.zeros((0,3))
        if x is not None:
            pts = np.vstack((pts, np.array((xx_agg, yy_agg, zz_agg)).T))
        if pts.shape[0]:
            im_tmp = img.imageFromPoints(pts,
                        nx=nx, ny=ny, nz=nz,
                        sx=sx, sy=sy, sz=sz,
                        ox=ox, oy=oy, oz=oz,
                        indicator_var=True,
                        logger=logger)
            mask = 1.0*np.any((im_tmp.val[0], mask), axis=0)
            del(im_tmp)
        del(pts)

    # Check parameters - probability
    if probability is not None:
        # if method == 'ordinary_kriging':
        #     if verbose > 0:
        #         print(f"ERROR ({fname}): specifying `probability` not allowed with ordinary kriging")
        #     return None
        probability = np.asarray(probability, dtype='float').reshape(-1) # cast in 1-dimensional array if needed
        if probability.size not in (ncategory, ncategory*nxyz):
            err_msg = f'{fname}: size of `probability` is not valid'
            if logger: logger.error(err_msg)
            raise GeosclassicinterfaceError(err_msg)

    # --- Fill mpds_geosClassicInput structure (C)
    try:
        mpds_geosClassicIndicatorInput = fill_mpds_geosClassicIndicatorInput(
                False,
                space_dim,
                ncategory,
                category_values,
                cm_for_cat,
                nx, ny, nz,
                sx, sy, sz,
                ox, oy, oz,
                varname,
                outputReportFile,
                computationMode,
                None,
                dataPointSet,
                mask,
                probability,
                np.zeros(ncategory),
                searchRadiusRelative,
                nneighborMax,
                searchNeighborhoodSortMode,
                0,
                0,
                logger=logger)

    except Exception as exc:
        err_msg = f'{fname}: cannot fill mpds_geosClassicIndicatorInput C structure'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg) from exc

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

    if verbose > 0 and nth > os.cpu_count():
        if logger:
            logger.warning(f'{fname}: number of threads used will exceed number of cpu(s) of the system...')
        else:
            print(f'{fname}: WARNING: number of threads used will exceed number of cpu(s) of the system...')

    if verbose > 1:
        if logger:
            logger.info('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
        else:
            print('{}: Geos-Classic running... [VERSION {:s} / BUILD NUMBER {:s} / OpenMP {:d} thread(s)]'.format(fname, geosclassic.MPDS_GEOS_CLASSIC_VERSION_NUMBER, geosclassic.MPDS_GEOS_CLASSIC_BUILD_NUMBER, nth))
            sys.stdout.flush()
            sys.stdout.flush() # twice!, so that the previous print is flushed before launching GeosClassic...

    # --- Launch "GeosClassicSim" (launch C code)
    # err = geosclassic.MPDSGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor )
    err = geosclassic.MPDSOMPGeosClassicIndicatorSim(mpds_geosClassicIndicatorInput, mpds_geosClassicOutput, mpds_progressMonitor, mpds_updateProgressMonitor, nth)

    # Free memory on C side: mpds_geosClassicIndicatorInput
    geosclassic.MPDSGeosClassicFreeGeosClassicIndicatorInput(mpds_geosClassicIndicatorInput)
    geosclassic.free_MPDS_GEOSCLASSICINDICATORINPUT(mpds_geosClassicIndicatorInput)

    if err:
        # Free memory on C side: mpds_geosClassicOutput
        geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
        geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)
        # Free memory on C side: mpds_progressMonitor
        geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)
        # Raise error
        err_message = geosclassic.mpds_get_error_message(-err)
        err_message = err_message.replace('\n', '')
        err_msg = f'{fname}: {err_message}'
        if logger: logger.error(err_msg)
        raise GeosclassicinterfaceError(err_msg)

    geosclassic_output = geosclassic_output_C2py(mpds_geosClassicOutput, mpds_progressMonitor, logger=logger)

    # Free memory on C side: mpds_geosClassicOutput
    geosclassic.MPDSGeosClassicFreeGeosClassicOutput(mpds_geosClassicOutput)
    geosclassic.free_MPDS_GEOSCLASSICOUTPUT(mpds_geosClassicOutput)

    # Free memory on C side: mpds_progressMonitor
    geosclassic.free_MPDS_PROGRESSMONITOR(mpds_progressMonitor)

    if geosclassic_output is not None and mask is not None and add_data_point_to_mask:
        # Remove the value out of the original mask (using its copy see above)
        geosclassic_output['image'].val[:, mask_original==0.0] = np.nan

    if verbose > 1 and geosclassic_output:
        if logger:
            logger.info(f'{fname}: Geos-Classic run complete')
        else:
            print(f'{fname}: Geos-Classic run complete')

    # Show (print) encountered warnings
    if verbose > 0 and geosclassic_output and geosclassic_output['nwarning']:
        # note: not logged even if `logger` is not `None` (list of warning(s) is returned)
        if logger is None:
            print(f"{fname}: warnings encountered ({geosclassic_output['nwarning']} times in all):")
            for i, warning_message in enumerate(geosclassic_output['warnings']):
                print(f'#{i+1:3d}: {warning_message}')

    return geosclassic_output
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Module 'geone.geosclassicinterface'.")
