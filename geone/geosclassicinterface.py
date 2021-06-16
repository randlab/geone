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
def covModel1Delem_py2C(covModelElem_py):
    """
    Converts an elementary covariance model 1D from python to C.

    :param covModelElem_py: (2-tuple) elementary covariance model 1D in python:
                                (t, d) corresponds to an elementary model with:
                                    t: (string) the type, could be
                                        'nugget', 'spherical', 'exponential',
                                        'gaussian', 'cubic', 'power'
                                    d: (dict) dictionary of required parameters
                                        to be passed to the elementary model
                                e.g.
                                    (t, d) = ('power', {w:2.0, r:1.5, s:1.7})
    :return covModeElem_c:  (MPDS_COVMODEL *) covariance model converted (C struct)
    """

    covModelElem_c = geosclassic.malloc_MPDS_COVMODELELEM()
    geosclassic.MPDSGeosClassicInitCovModelElem(covModelElem_c)

    if covModelElem_py[0] == 'nugget':
        # type
        covModelElem_c.covModelType = geosclassic.COV_NUGGET
        # weight
        covModelElem_c.weight = covModelElem_py[1]['w']
    elif covModelElem_py[0] == 'spherical':
        # type
        covModelElem_c.covModelType = geosclassic.COV_SPHERICAL
        # weight
        covModelElem_c.weight = covModelElem_py[1]['w']
        # ranges
        covModelElem_c.rx = covModelElem_py[1]['r']
        covModelElem_c.ry = 0.0
        covModelElem_c.rz = 0.0
    elif covModelElem_py[0] == 'exponential':
        # type
        covModelElem_c.covModelType = geosclassic.COV_EXPONENTIAL
        # weight
        covModelElem_c.weight = covModelElem_py[1]['w']
        # ranges
        covModelElem_c.rx = covModelElem_py[1]['r']
        covModelElem_c.ry = 0.0
        covModelElem_c.rz = 0.0
    elif covModelElem_py[0] == 'gaussian':
        # type
        covModelElem_c.covModelType = geosclassic.COV_GAUSSIAN
        # weight
        covModelElem_c.weight = covModelElem_py[1]['w']
        # ranges
        covModelElem_c.rx = covModelElem_py[1]['r']
        covModelElem_c.ry = 0.0
        covModelElem_c.rz = 0.0
    elif covModelElem_py[0] == 'cubic':
        # type
        covModelElem_c.covModelType = geosclassic.COV_CUBIC
        # weight
        covModelElem_c.weight = covModelElem_py[1]['w']
        # ranges
        covModelElem_c.rx = covModelElem_py[1]['r']
        covModelElem_c.ry = 0.0
        covModelElem_c.rz = 0.0
    elif covModelElem_py[0] == 'power':
        # type
        covModelElem_c.covModelType = geosclassic.COV_POWER
        # weight
        covModelElem_c.weight = covModelElem_py[1]['w']
        # ranges
        covModelElem_c.rx = covModelElem_py[1]['r']
        covModelElem_c.ry = 0.0
        covModelElem_c.rz = 0.0
        # other parameters
        covModelElem_c.s = covModelElem_py[1]['s']

    return covModelElem_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel2Delem_py2C(covModelElem_py):
    """
    Converts an elementary covariance model 2D from python to C.

    :param covModelElem_py: (2-tuple) elementary covariance model 2D in python:
                                (t, d) corresponds to an elementary model with:
                                    t: (string) the type, could be
                                        'nugget', 'spherical', 'exponential',
                                        'gaussian', 'cubic', 'power'
                                    d: (dict) dictionary of required parameters
                                        to be passed to the elementary model
                                e.g.
                                    (t, d) = ('gaussian', {'w':10., 'r':[150, 50]})
    :return covModeElem_c:  (MPDS_COVMODEL *) covariance model converted (C struct)
    """

    covModelElem_c = geosclassic.malloc_MPDS_COVMODELELEM()
    geosclassic.MPDSGeosClassicInitCovModelElem(covModelElem_c)

    if covModelElem_py[0] == 'nugget':
        # type
        covModelElem_c.covModelType = geosclassic.COV_NUGGET
        # weight
        covModelElem_c.weight = covModelElem_py[1]['w']
    elif covModelElem_py[0] == 'spherical':
        # type
        covModelElem_c.covModelType = geosclassic.COV_SPHERICAL
        # weight
        covModelElem_c.weight = covModelElem_py[1]['w']
        # ranges
        covModelElem_c.rx = float(covModelElem_py[1]['r'][0])
        covModelElem_c.ry = float(covModelElem_py[1]['r'][1])
        covModelElem_c.rz = 0.0
    elif covModelElem_py[0] == 'exponential':
        # type
        covModelElem_c.covModelType = geosclassic.COV_EXPONENTIAL
        # weight
        covModelElem_c.weight = covModelElem_py[1]['w']
        # ranges
        covModelElem_c.rx = float(covModelElem_py[1]['r'][0])
        covModelElem_c.ry = float(covModelElem_py[1]['r'][1])
        covModelElem_c.rz = 0.0
    elif covModelElem_py[0] == 'gaussian':
        # type
        covModelElem_c.covModelType = geosclassic.COV_GAUSSIAN
        # weight
        covModelElem_c.weight = covModelElem_py[1]['w']
        # ranges
        covModelElem_c.rx = float(covModelElem_py[1]['r'][0])
        covModelElem_c.ry = float(covModelElem_py[1]['r'][1])
        covModelElem_c.rz = 0.0
    elif covModelElem_py[0] == 'cubic':
        # type
        covModelElem_c.covModelType = geosclassic.COV_CUBIC
        # weight
        covModelElem_c.weight = covModelElem_py[1]['w']
        # ranges
        covModelElem_c.rx = float(covModelElem_py[1]['r'][0])
        covModelElem_c.ry = float(covModelElem_py[1]['r'][1])
        covModelElem_c.rz = 0.0
    elif covModelElem_py[0] == 'power':
        # type
        covModelElem_c.covModelType = geosclassic.COV_POWER
        # weight
        covModelElem_c.weight = covModelElem_py[1]['w']
        # ranges
        covModelElem_c.rx = float(covModelElem_py[1]['r'][0])
        covModelElem_c.ry = float(covModelElem_py[1]['r'][1])
        covModelElem_c.rz = 0.0
        # other parameters
        covModelElem_c.s = covModelElem_py[1]['s']

    return covModelElem_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel3Delem_py2C(covModelElem_py):
    """
    Converts an elementary covariance model 3D from python to C.

    :param covModelElem_py: (2-tuple) elementary covariance model 3D in python:
                                (t, d) corresponds to an elementary model with:
                                    t: (string) the type, could be
                                        'nugget', 'spherical', 'exponential',
                                        'gaussian', 'cubic', 'power'
                                    d: (dict) dictionary of required parameters
                                        to be passed to the elementary model
                                e.g.
                                    (t, d) = ('power', {w:2.0, r:[1.5, 2.5, 3.0], s:1.7})
    :return covModeElem_c:  (MPDS_COVMODEL *) covariance model converted (C struct)
    """

    covModelElem_c = geosclassic.malloc_MPDS_COVMODELELEM()
    geosclassic.MPDSGeosClassicInitCovModelElem(covModelElem_c)

    if covModelElem_py[0] == 'nugget':
        # type
        covModelElem_c.covModelType = geosclassic.COV_NUGGET
        # weight
        covModelElem_c.weight = covModelElem_py[1]['w']
    elif covModelElem_py[0] == 'spherical':
        # type
        covModelElem_c.covModelType = geosclassic.COV_SPHERICAL
        # weight
        covModelElem_c.weight = covModelElem_py[1]['w']
        # ranges
        covModelElem_c.rx = float(covModelElem_py[1]['r'][0])
        covModelElem_c.ry = float(covModelElem_py[1]['r'][1])
        covModelElem_c.rz = float(covModelElem_py[1]['r'][2])
    elif covModelElem_py[0] == 'exponential':
        # type
        covModelElem_c.covModelType = geosclassic.COV_EXPONENTIAL
        # weight
        covModelElem_c.weight = covModelElem_py[1]['w']
        # ranges
        covModelElem_c.rx = float(covModelElem_py[1]['r'][0])
        covModelElem_c.ry = float(covModelElem_py[1]['r'][1])
        covModelElem_c.rz = float(covModelElem_py[1]['r'][2])
    elif covModelElem_py[0] == 'gaussian':
        # type
        covModelElem_c.covModelType = geosclassic.COV_GAUSSIAN
        # weight
        covModelElem_c.weight = covModelElem_py[1]['w']
        # ranges
        covModelElem_c.rx = float(covModelElem_py[1]['r'][0])
        covModelElem_c.ry = float(covModelElem_py[1]['r'][1])
        covModelElem_c.rz = float(covModelElem_py[1]['r'][2])
    elif covModelElem_py[0] == 'cubic':
        # type
        covModelElem_c.covModelType = geosclassic.COV_CUBIC
        # weight
        covModelElem_c.weight = covModelElem_py[1]['w']
        # ranges
        covModelElem_c.rx = float(covModelElem_py[1]['r'][0])
        covModelElem_c.ry = float(covModelElem_py[1]['r'][1])
        covModelElem_c.rz = float(covModelElem_py[1]['r'][2])
    elif covModelElem_py[0] == 'power':
        # type
        covModelElem_c.covModelType = geosclassic.COV_POWER
        # weight
        covModelElem_c.weight = covModelElem_py[1]['w']
        # ranges
        covModelElem_c.rx = float(covModelElem_py[1]['r'][0])
        covModelElem_c.ry = float(covModelElem_py[1]['r'][1])
        covModelElem_c.rz = float(covModelElem_py[1]['r'][2])
        # other parameters
        covModelElem_c.s = covModelElem_py[1]['s']

    return covModelElem_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel1D_py2C(covModel_py):
    """
    Converts a covariance model 1D from python to C.

    :param covModel_py:   (CovModel1D class) covariance model 1D (python class)
    :return covModel_c:   (MPDS_COVMODEL *) covariance model converted (C struct)
    """

    covModel_c = geosclassic.malloc_MPDS_COVMODEL()
    geosclassic.MPDSGeosClassicInitCovModel(covModel_c)

    n = len(covModel_py.elem)
    covModel_c.nelem = n
    covModel_c.covModelElem = geosclassic.new_MPDS_COVMODELELEM_array(n)
    for i, covModelElem in enumerate(covModel_py.elem):
        geosclassic.MPDS_COVMODELELEM_array_setitem(covModel_c.covModelElem, i, covModel1Delem_py2C(covModelElem))

    # covModel_c.angle1, covModel_c.angle2, covModel_c.angle3: keep to 0.0
    covModel_c.angle1 = 0.0
    covModel_c.angle2 = 0.0
    covModel_c.angle3 = 0.0

    return covModel_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel2D_py2C(covModel_py):
    """
    Converts a covariance model 2D from python to C.

    :param covModel_py:   (CovModel2D class) covariance model 2D (python class)
    :return covModel_c:   (MPDS_COVMODEL *) covariance model converted (C struct)
    """

    covModel_c = geosclassic.malloc_MPDS_COVMODEL()
    geosclassic.MPDSGeosClassicInitCovModel(covModel_c)

    n = len(covModel_py.elem)
    covModel_c.nelem = n
    covModel_c.covModelElem = geosclassic.new_MPDS_COVMODELELEM_array(n)
    for i, covModelElem in enumerate(covModel_py.elem):
        geosclassic.MPDS_COVMODELELEM_array_setitem(covModel_c.covModelElem, i, covModel2Delem_py2C(covModelElem))

    # covModel_c.angle2, covModel_c.angle3: keep to 0.0
    covModel_c.angle1 = covModel_py.alpha
    covModel_c.angle2 = 0.0
    covModel_c.angle3 = 0.0

    return covModel_c
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def covModel3D_py2C(covModel_py):
    """
    Converts a covariance model 3D from python to C.

    :param covModel_py:   (CovModel3D class) covariance model 3D (python class)
    :return covModel_c:   (MPDS_COVMODEL *) covariance model converted (C struct)
    """

    covModel_c = geosclassic.malloc_MPDS_COVMODEL()
    geosclassic.MPDSGeosClassicInitCovModel(covModel_c)

    n = len(covModel_py.elem)
    covModel_c.nelem = n
    covModel_c.covModelElem = geosclassic.new_MPDS_COVMODELELEM_array(n)
    for i, covModelElem in enumerate(covModel_py.elem):
        geosclassic.MPDS_COVMODELELEM_array_setitem(covModel_c.covModelElem, i, covModel3Delem_py2C(covModelElem))

    covModel_c.angle1 = covModel_py.alpha
    covModel_c.angle2 = covModel_py.beta
    covModel_c.angle3 = covModel_py.gamma

    return covModel_c
# ----------------------------------------------------------------------------

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

    :return mpds_geosClassicInput: C structure for "GeosClassicSim" program (C)
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

    mpds_geosClassicInput.simGrid.nx = nx
    mpds_geosClassicInput.simGrid.ny = ny
    mpds_geosClassicInput.simGrid.nz = nz

    mpds_geosClassicInput.simGrid.sx = sx
    mpds_geosClassicInput.simGrid.sy = sy
    mpds_geosClassicInput.simGrid.sz = sz

    mpds_geosClassicInput.simGrid.ox = ox
    mpds_geosClassicInput.simGrid.oy = oy
    mpds_geosClassicInput.simGrid.oz = oz

    mpds_geosClassicInput.simGrid.nxy = nxy
    mpds_geosClassicInput.simGrid.nxyz = nxyz

    # mpds_geosClassicInput.varname
    geosclassic.mpds_set_geosClassicInput_varname(mpds_geosClassicInput, varname)

    # mpds_geosClassicInput.outputMode
    mpds_geosClassicInput.outputMode = geosclassic.GEOS_CLASSIC_OUTPUT_NO_FILE

    # mpds_geosClassicInput.outputReportFlag and mpds_geosClassicInput.outputReportFileName
    if outputReportFile is not None:
        mpds_geosClassicInput.outputReportFlag = geosclassic.TRUE
        geosclassic.mpds_set_outputReportFileName(mpds_geosClassicInput, outputReportFile)
    else:
        mpds_geosClassicInput.outputReportFlag = geosclassic.FALSE

    # mpds_geosClassicInput.computationMode
    mpds_geosClassicInput.computationMode = computationMode

    # mpds_geosClassicInput.covModel
    if space_dim==1:
        mpds_geosClassicInput.covModel = covModel1D_py2C(cov_model)
    elif space_dim==2:
        mpds_geosClassicInput.covModel = covModel2D_py2C(cov_model)
    elif space_dim==3:
        mpds_geosClassicInput.covModel = covModel3D_py2C(cov_model)

    # mpds_geosClassicInput.searchRadiusRelative
    mpds_geosClassicInput.searchRadiusRelative = searchRadiusRelative

    # mpds_geosClassicInput.nneighborMax
    mpds_geosClassicInput.nneighborMax = nneighborMax

    # mpds_geosClassicInput.searchNeighborhoodSortMode
    mpds_geosClassicInput.searchNeighborhoodSortMode = searchNeighborhoodSortMode

    # mpds_geosClassicInput.ndataImage
    mpds_geosClassicInput.ndataImage = 0

    # mpds_geosClassicInput.ndataPointSet and mpds_geosClassicInput.dataPointSet
    n = len(dataPointSet)
    mpds_geosClassicInput.ndataPointSet = n
    if n:
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
        mpds_geosClassicInput.meanValue = mean[0]
    elif mean.size == nxyz:
        mpds_geosClassicInput.meanUsage = 2
        im = Img(nx=nx, ny=ny, nz=nz,
                 sx=sx, sy=sy, sz=sz,
                 ox=ox, oy=oy, oz=oz,
                 nv=1, val=mean)
        mpds_geosClassicInput.meanImage = img_py2C(im)

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

    # mpds_geosClassicInput.nGibbsSamplerPath
    mpds_geosClassicInput.nGibbsSamplerPath = nGibbsSamplerPath

    # mpds_geosClassicInput.seed
    if seed is None:
        seed = np.random.randint(1,1000000)
    mpds_geosClassicInput.seed = seed

    # mpds_geosClassicInput.seedIncrement
    mpds_geosClassicInput.seedIncrement = 1

    # mpds_geosClassicInput.nrealization
    mpds_geosClassicInput.nrealization = nreal

    return mpds_geosClassicInput
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
def simulate1D(cov_model,
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
        searchNeighborhoodSortMode=2,
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

    :param mean:        (None or float or ndarray) mean of the simulation
                            (for simple kriging only):
                                - None   : mean of hard data values (stationary),
                                           (0 if no hard data)
                                - float  : for stationary mean (set manually)
                                - ndarray: for non stationary mean, must contain
                                    as many entries as number of grid cells
                                    (reshaped if needed)
                            For ordinary kriging (method = 'ordinary_kriging'),
                            this parameter must be None (mean of hard data values
                            or zero if no data is used when no neighbor)

    :param var:         (None or float or ndarray) variance of the simulation
                            (for simple kriging only):
                                - None   : variance not modified
                                           (only covariance model is used)
                                - float  : for stationary variance (set manually)
                                - ndarray: for non stationary variance, must contain
                                    as many entries as number of grid cells
                                    (reshaped if needed)
                            For ordinary kriging (method = 'ordinary_kriging'),
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
                            let r_i be the ranges of the covariance model along the main axes,
                            if x is node to be simulated, a node y is taken into account iff it is
                            within the ellipsoid centered at x of half-axes searchRadiusRelative * r_i

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

    # Check parameters - mean
    if mean is not None:
        if method == 'ordinary_kriging':
            print("ERROR (SIMUL_1D): specifying 'mean' not allowed with ordinary kriging")
            return None
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
    mpds_geosClassicInput = fill_mpds_geosClassicInput(
        space_dim,
        cov_model,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        varname,
        outputReportFile,
        computationMode,
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
    # err = geosclassic.MPDSGeosClassicSim(mpds_siminput, mpds_simoutput, mpds_progressMonitor, mpds_updateProgressMonitor )
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
def simulate2D(cov_model,
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
        searchNeighborhoodSortMode=2,
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

    :param mean:        (None or float or ndarray) mean of the simulation
                            (for simple kriging only):
                                - None   : mean of hard data values (stationary),
                                           (0 if no hard data)
                                - float  : for stationary mean (set manually)
                                - ndarray: for non stationary mean, must contain
                                    as many entries as number of grid cells
                                    (reshaped if needed)
                            For ordinary kriging (method = 'ordinary_kriging'),
                            this parameter must be None (mean of hard data values
                            or zero if no data is used when no neighbor)

    :param var:         (None or float or ndarray) variance of the simulation
                            (for simple kriging only):
                                - None   : variance not modified
                                           (only covariance model is used)
                                - float  : for stationary variance (set manually)
                                - ndarray: for non stationary variance, must contain
                                    as many entries as number of grid cells
                                    (reshaped if needed)
                            For ordinary kriging (method = 'ordinary_kriging'),
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
                            let r_i be the ranges of the covariance model along the main axes,
                            if x is node to be simulated, a node y is taken into account iff it is
                            within the ellipsoid centered at x of half-axes searchRadiusRelative * r_i

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

    # Check parameters - mean
    if mean is not None:
        if method == 'ordinary_kriging':
            print("ERROR (SIMUL_2D): specifying 'mean' not allowed with ordinary kriging")
            return None
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
    mpds_geosClassicInput = fill_mpds_geosClassicInput(
        space_dim,
        cov_model,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        varname,
        outputReportFile,
        computationMode,
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
    # err = geosclassic.MPDSGeosClassicSim(mpds_siminput, mpds_simoutput, mpds_progressMonitor, mpds_updateProgressMonitor )
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
def simulate3D(cov_model,
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
        searchNeighborhoodSortMode=2,
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

    :param mean:        (None or float or ndarray) mean of the simulation
                            (for simple kriging only):
                                - None   : mean of hard data values (stationary),
                                           (0 if no hard data)
                                - float  : for stationary mean (set manually)
                                - ndarray: for non stationary mean, must contain
                                    as many entries as number of grid cells
                                    (reshaped if needed)
                            For ordinary kriging (method = 'ordinary_kriging'),
                            this parameter must be None (mean of hard data values
                            or zero if no data is used when no neighbor)

    :param var:         (None or float or ndarray) variance of the simulation
                            (for simple kriging only):
                                - None   : variance not modified
                                           (only covariance model is used)
                                - float  : for stationary variance (set manually)
                                - ndarray: for non stationary variance, must contain
                                    as many entries as number of grid cells
                                    (reshaped if needed)
                            For ordinary kriging (method = 'ordinary_kriging'),
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
                            let r_i be the ranges of the covariance model along the main axes,
                            if x is node to be simulated, a node y is taken into account iff it is
                            within the ellipsoid centered at x of half-axes searchRadiusRelative * r_i

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

    # Check parameters - mean
    if mean is not None:
        if method == 'ordinary_kriging':
            print("ERROR (SIMUL_3D): specifying 'mean' not allowed with ordinary kriging")
            return None
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
    mpds_geosClassicInput = fill_mpds_geosClassicInput(
        space_dim,
        cov_model,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        varname,
        outputReportFile,
        computationMode,
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
    # err = geosclassic.MPDSGeosClassicSim(mpds_siminput, mpds_simoutput, mpds_progressMonitor, mpds_updateProgressMonitor )
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
def estimate1D(cov_model,
        dimension, spacing=1.0, origin=0.0,
        method='simple_kriging',
        mean=None, var=None,
        x=None, v=None,
        mask=None,
        use_unique_neighborhood=False,
        searchRadiusRelative=1.0,
        nneighborMax=12,
        searchNeighborhoodSortMode=2,
        outputReportFile=None,
        nthreads=-1, verbose=2):
    """
    Compute estimate and standard deviation for 1D grid of simple or ordinary kriging.

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

    :param mean:        (None or float or ndarray) mean of the simulation
                            (for simple kriging only):
                                - None   : mean of hard data values (stationary),
                                           (0 if no hard data)
                                - float  : for stationary mean (set manually)
                                - ndarray: for non stationary mean, must contain
                                    as many entries as number of grid cells
                                    (reshaped if needed)
                            For ordinary kriging (method = 'ordinary_kriging'),
                            this parameter must be None (mean of hard data values
                            or zero if no data is used when no neighbor)

    :param var:         (None or float or ndarray) variance of the simulation
                            (for simple kriging only):
                                - None   : variance not modified
                                           (only covariance model is used)
                                - float  : for stationary variance (set manually)
                                - ndarray: for non stationary variance, must contain
                                    as many entries as number of grid cells
                                    (reshaped if needed)
                            For ordinary kriging (method = 'ordinary_kriging'),
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
                            let r_i be the ranges of the covariance model along the main axes,
                            if x is node to be simulated, a node y is taken into account iff it is
                            within the ellipsoid centered at x of half-axes searchRadiusRelative * r_i

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

    # Check parameters - mean
    if mean is not None:
        if method == 'ordinary_kriging':
            print("ERROR (ESTIM_1D): specifying 'mean' not allowed with ordinary kriging")
            return None
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

    # Set searchRadiusRelative to -1 if unique neighborhood is used
    if use_unique_neighborhood:
        searchRadiusRelative = -1.0

    # --- Fill mpds_geosClassicInput structure (C)
    mpds_geosClassicInput = fill_mpds_geosClassicInput(
        space_dim,
        cov_model,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        varname,
        outputReportFile,
        computationMode,
        dataPointSet,
        mask,
        mean,
        var,
        searchRadiusRelative,
        nneighborMax,
        searchNeighborhoodSortMode,
        0, 0, 0)

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
    # err = geosclassic.MPDSGeosClassicSim(mpds_siminput, mpds_simoutput, mpds_progressMonitor, mpds_updateProgressMonitor )
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
def estimate2D(cov_model,
        dimension, spacing=(1.0, 1.0), origin=(0.0, 0.0),
        method='simple_kriging',
        mean=None, var=None,
        x=None, v=None,
        mask=None,
        use_unique_neighborhood=False,
        searchRadiusRelative=1.0,
        nneighborMax=12,
        searchNeighborhoodSortMode=2,
        outputReportFile=None,
        nthreads=-1, verbose=2):
    """
    Compute estimate and standard deviation for 2D grid of simple or ordinary kriging.

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

    :param mean:        (None or float or ndarray) mean of the simulation
                            (for simple kriging only):
                                - None   : mean of hard data values (stationary),
                                           (0 if no hard data)
                                - float  : for stationary mean (set manually)
                                - ndarray: for non stationary mean, must contain
                                    as many entries as number of grid cells
                                    (reshaped if needed)
                            For ordinary kriging (method = 'ordinary_kriging'),
                            this parameter must be None (mean of hard data values
                            or zero if no data is used when no neighbor)

    :param var:         (None or float or ndarray) variance of the simulation
                            (for simple kriging only):
                                - None   : variance not modified
                                           (only covariance model is used)
                                - float  : for stationary variance (set manually)
                                - ndarray: for non stationary variance, must contain
                                    as many entries as number of grid cells
                                    (reshaped if needed)
                            For ordinary kriging (method = 'ordinary_kriging'),
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
                            let r_i be the ranges of the covariance model along the main axes,
                            if x is node to be simulated, a node y is taken into account iff it is
                            within the ellipsoid centered at x of half-axes searchRadiusRelative * r_i

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

    # Check parameters - mean
    if mean is not None:
        if method == 'ordinary_kriging':
            print("ERROR (ESTIM_2D): specifying 'mean' not allowed with ordinary kriging")
            return None
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

    # Set searchRadiusRelative to -1 if unique neighborhood is used
    if use_unique_neighborhood:
        searchRadiusRelative = -1.0

    # --- Fill mpds_geosClassicInput structure (C)
    mpds_geosClassicInput = fill_mpds_geosClassicInput(
        space_dim,
        cov_model,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        varname,
        outputReportFile,
        computationMode,
        dataPointSet,
        mask,
        mean,
        var,
        searchRadiusRelative,
        nneighborMax,
        searchNeighborhoodSortMode,
        0, 0, 0)

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
    # err = geosclassic.MPDSGeosClassicSim(mpds_siminput, mpds_simoutput, mpds_progressMonitor, mpds_updateProgressMonitor )
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
def estimate3D(cov_model,
        dimension, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
        method='simple_kriging',
        mean=None, var=None,
        x=None, v=None,
        mask=None,
        use_unique_neighborhood=False,
        searchRadiusRelative=1.0,
        nneighborMax=12,
        searchNeighborhoodSortMode=2,
        outputReportFile=None,
        nthreads=-1, verbose=2):
    """
    Compute estimate and standard deviation for 3D grid of simple or ordinary kriging.

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

    :param mean:        (None or float or ndarray) mean of the simulation
                            (for simple kriging only):
                                - None   : mean of hard data values (stationary),
                                           (0 if no hard data)
                                - float  : for stationary mean (set manually)
                                - ndarray: for non stationary mean, must contain
                                    as many entries as number of grid cells
                                    (reshaped if needed)
                            For ordinary kriging (method = 'ordinary_kriging'),
                            this parameter must be None (mean of hard data values
                            or zero if no data is used when no neighbor)

    :param var:         (None or float or ndarray) variance of the simulation
                            (for simple kriging only):
                                - None   : variance not modified
                                           (only covariance model is used)
                                - float  : for stationary variance (set manually)
                                - ndarray: for non stationary variance, must contain
                                    as many entries as number of grid cells
                                    (reshaped if needed)
                            For ordinary kriging (method = 'ordinary_kriging'),
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
                            let r_i be the ranges of the covariance model along the main axes,
                            if x is node to be simulated, a node y is taken into account iff it is
                            within the ellipsoid centered at x of half-axes searchRadiusRelative * r_i

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

    # Check parameters - mean
    if mean is not None:
        if method == 'ordinary_kriging':
            print("ERROR (ESTIM_3D): specifying 'mean' not allowed with ordinary kriging")
            return None
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

    # Set searchRadiusRelative to -1 if unique neighborhood is used
    if use_unique_neighborhood:
        searchRadiusRelative = -1.0

    # --- Fill mpds_geosClassicInput structure (C)
    mpds_geosClassicInput = fill_mpds_geosClassicInput(
        space_dim,
        cov_model,
        nx, ny, nz,
        sx, sy, sz,
        ox, oy, oz,
        varname,
        outputReportFile,
        computationMode,
        dataPointSet,
        mask,
        mean,
        var,
        searchRadiusRelative,
        nneighborMax,
        searchNeighborhoodSortMode,
        0, 0, 0)

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
    # err = geosclassic.MPDSGeosClassicSim(mpds_siminput, mpds_simoutput, mpds_progressMonitor, mpds_updateProgressMonitor )
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

if __name__ == "__main__":
    print("Module 'geone.geosclassicinterface'.")
